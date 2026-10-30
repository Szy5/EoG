#!/usr/bin/env python3
"""Build a safe mmap-friendly bidirectional CSR index for ogbl-wikikg2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TEXT_FILES = ("node_ids.npy", "node_labels.npy", "node_descriptions.npy", "relation_ids.npy", "relation_labels.npy")
SOURCE_FILES = TEXT_FILES + ("edge_ids.npy", "relation_types.npy", "RELEASE_v1.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array(path: Path, array: np.ndarray) -> dict:
    np.save(path, array, allow_pickle=False)
    written = path.with_suffix(".npy") if path.suffix != ".npy" else path
    loaded = np.load(written, mmap_mode="r", allow_pickle=False)
    return {
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
        "bytes": written.stat().st_size,
        "sha256": sha256(written),
    }


def offsets(keys: np.ndarray, size: int) -> np.ndarray:
    counts = np.bincount(keys, minlength=size).astype(np.int64, copy=False)
    result = np.empty(size + 1, dtype=np.int64)
    result[0] = 0
    np.cumsum(counts, out=result[1:])
    return result


def save_text_column(temp: Path, source_name: str, raw: np.ndarray) -> tuple[dict[str, dict], list[str]]:
    """Encode variable-width text as mmap-safe UTF-8 payload plus offsets."""
    values = [str(value) for value in raw]
    encoded = [value.encode("utf-8") for value in values]
    lengths = np.fromiter((len(value) for value in encoded), dtype=np.int64, count=len(encoded))
    text_offsets = np.empty(len(values) + 1, dtype=np.int64)
    text_offsets[0] = 0
    np.cumsum(lengths, out=text_offsets[1:])
    payload = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    stem = source_name.removesuffix(".npy")
    bytes_name = f"{stem}.bytes.npy"
    offsets_name = f"{stem}.offsets.npy"
    specs = {
        bytes_name: save_array(temp / bytes_name, payload),
        offsets_name: save_array(temp / offsets_name, text_offsets),
    }
    return specs, values


def build(source: Path, output: Path, overwrite: bool = False) -> dict:
    """Build and atomically publish a validated GTSQA index snapshot."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing GTSQA source files: {missing}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-", dir=output.parent))
    backup: Path | None = None
    try:
        arrays: dict[str, dict] = {}
        text_values = {}
        for name in TEXT_FILES:
            raw = np.load(source / name, allow_pickle=True)
            specs, values = save_text_column(temp, name, raw)
            arrays.update(specs)
            text_values[name] = values

        edge_ids = np.load(source / "edge_ids.npy", mmap_mode="r", allow_pickle=False)
        relation_types = np.load(source / "relation_types.npy", mmap_mode="r", allow_pickle=False)
        if edge_ids.ndim != 2 or edge_ids.shape[0] != 2 or edge_ids.shape[1] != len(relation_types):
            raise ValueError("edge_ids must have shape (2, E) matching relation_types")
        node_count = len(text_values["node_ids.npy"])
        relation_count = len(text_values["relation_ids.npy"])
        heads = np.asarray(edge_ids[0], dtype=np.int32)
        tails = np.asarray(edge_ids[1], dtype=np.int32)
        relation_values = np.asarray(relation_types, dtype=np.uint16)
        if heads.min() < 0 or tails.min() < 0 or heads.max() >= node_count or tails.max() >= node_count:
            raise ValueError("edge endpoint is outside node table")
        if relation_values.min() < 0 or relation_values.max() >= relation_count:
            raise ValueError("relation type is outside relation table")
        arrays["edge_heads.npy"] = save_array(temp / "edge_heads.npy", heads)
        arrays["edge_tails.npy"] = save_array(temp / "edge_tails.npy", tails)
        arrays["edge_relation_types.npy"] = save_array(temp / "edge_relation_types.npy", relation_values)

        # Keep source edge rows canonical and store two adjacency permutations.
        # An incoming traversal therefore never requires reversing the triple.
        out_order = np.argsort(heads, kind="stable").astype(np.int64, copy=False)
        in_order = np.argsort(tails, kind="stable").astype(np.int64, copy=False)
        out_offsets = offsets(heads, node_count)
        in_offsets = offsets(tails, node_count)
        arrays["out_order.npy"] = save_array(temp / "out_order.npy", out_order)
        arrays["in_order.npy"] = save_array(temp / "in_order.npy", in_order)
        arrays["out_offsets.npy"] = save_array(temp / "out_offsets.npy", out_offsets)
        arrays["in_offsets.npy"] = save_array(temp / "in_offsets.npy", in_offsets)

        source_specs = [
            {"path": str(source / name), "bytes": (source / name).stat().st_size, "sha256": sha256(source / name)}
            for name in SOURCE_FILES
        ]
        snapshot_digest = hashlib.sha256(
            "".join(item["sha256"] for item in source_specs).encode()
        ).hexdigest()
        manifest = {
            "schema": "kg-platform.gtsqa-index",
            "schema_version": 1,
            "graph": {"graph_id": "gtsqa", "snapshot_id": f"gtsqa-ogbl-wikikg2@release-v1+sha256:{snapshot_digest}"},
            "builder": {"module": "scripts/build_gtsqa_index.py", "created_at": datetime.now(timezone.utc).isoformat()},
            "semantics": {
                "edge_direction": "canonical directed ogbl-wikikg2 edge",
                "reverse_edges_materialized": False,
                "entity_key": "QID",
                "relation_key": "PID",
                "text_columns": "packed UTF-8 uint8 payload with int64 offsets; no pickle/object arrays",
                "out_offsets": "node i uses out_order[out_offsets[i]:out_offsets[i+1]]",
                "in_offsets": "node i uses in_order[in_offsets[i]:in_offsets[i+1]]",
            },
            "counts": {"nodes": node_count, "edges": len(heads), "relations": relation_count},
            "source_files": source_specs,
            "arrays": arrays,
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        if output.exists():
            # Publish only after all arrays and the manifest are durable. The
            # backup makes an interrupted overwrite recoverable in except.
            backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
            output.rename(backup)
        temp.rename(output)
        if backup:
            shutil.rmtree(backup)
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        if backup and backup.exists() and not output.exists():
            backup.rename(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(args.source, args.output, args.overwrite)
    print(json.dumps({"graph": manifest["graph"], "counts": manifest["counts"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
