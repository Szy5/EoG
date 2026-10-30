#!/usr/bin/env python3
"""Build a validated mmap-friendly global index from Car KG JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

NODE_FIELDS = frozenset({"id", "name", "type"})
RELATIONSHIP_FIELDS = frozenset({"id", "source", "relation", "target"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    np.save(path, array, allow_pickle=False)
    written = path.with_suffix(".npy") if path.suffix != ".npy" else path
    loaded = np.load(written, mmap_mode="r", allow_pickle=False)
    return {
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
        "bytes": written.stat().st_size,
        "sha256": sha256(written),
    }


def save_text_column(output: Path, stem: str, values: list[str]) -> dict[str, dict[str, Any]]:
    encoded = [value.encode("utf-8") for value in values]
    lengths = np.fromiter((len(value) for value in encoded), dtype=np.int64, count=len(encoded))
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    payload = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    return {
        f"{stem}.bytes.npy": save_array(output / f"{stem}.bytes.npy", payload),
        f"{stem}.offsets.npy": save_array(output / f"{stem}.offsets.npy", offsets),
    }


def read_jsonl(path: Path, expected_fields: frozenset[str], kind: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(raw, dict) or frozenset(raw) != expected_fields:
                actual = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
                raise ValueError(f"{path}:{line_number}: {kind} fields must be {sorted(expected_fields)}; got {actual}")
            record: dict[str, str] = {}
            for field in expected_fields:
                value = raw[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{path}:{line_number}: {kind}.{field} must be a non-empty string")
                record[field] = value.strip()
            records.append(record)
    if not records:
        raise ValueError(f"{path}: {kind} file is empty")
    return records


def require_unique(records: list[dict[str, str]], field: str, kind: str) -> None:
    seen: set[str] = set()
    for record in records:
        value = record[field]
        if value in seen:
            raise ValueError(f"duplicate {kind} {field}: {value}")
        seen.add(value)


def csr_offsets(keys: np.ndarray, size: int) -> np.ndarray:
    counts = np.bincount(keys, minlength=size).astype(np.int64, copy=False)
    result = np.empty(size + 1, dtype=np.int64)
    result[0] = 0
    np.cumsum(counts, out=result[1:])
    return result


def validate_arrays(output: Path, node_count: int, edge_count: int, relation_count: int) -> None:
    arrays = {
        name: np.load(output / name, mmap_mode="r", allow_pickle=False)
        for name in (
            "edge_heads.npy",
            "edge_tails.npy",
            "edge_relation_types.npy",
            "out_order.npy",
            "out_offsets.npy",
            "in_order.npy",
            "in_offsets.npy",
        )
    }
    heads = arrays["edge_heads.npy"]
    tails = arrays["edge_tails.npy"]
    relations = arrays["edge_relation_types.npy"]
    if heads.shape != (edge_count,) or tails.shape != (edge_count,) or relations.shape != (edge_count,):
        raise ValueError("edge arrays have invalid shapes")
    if int(heads.min()) < 0 or int(tails.min()) < 0 or int(heads.max()) >= node_count or int(tails.max()) >= node_count:
        raise ValueError("edge endpoint is outside node table")
    if int(relations.min()) < 0 or int(relations.max()) >= relation_count:
        raise ValueError("edge relation is outside relation table")
    for prefix, keys in (("out", heads), ("in", tails)):
        order = arrays[f"{prefix}_order.npy"]
        offsets = arrays[f"{prefix}_offsets.npy"]
        if order.shape != (edge_count,) or offsets.shape != (node_count + 1,):
            raise ValueError(f"{prefix} adjacency has invalid shapes")
        if int(offsets[0]) != 0 or int(offsets[-1]) != edge_count or np.any(np.diff(offsets) < 0):
            raise ValueError(f"{prefix} offsets are invalid")
        if not np.array_equal(np.sort(order), np.arange(edge_count, dtype=order.dtype)):
            raise ValueError(f"{prefix} order is not an edge permutation")
        if not np.array_equal(np.diff(offsets), np.bincount(keys, minlength=node_count)):
            raise ValueError(f"{prefix} offsets do not match endpoint counts")
        if np.any(np.diff(keys[order]) < 0):
            raise ValueError(f"{prefix} order is not sorted by endpoint")


def build(
    nodes_path: Path,
    relationships_path: Path,
    output: Path,
    overwrite: bool = False,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish a validated Car global-array snapshot."""

    nodes_path = nodes_path.expanduser().resolve()
    relationships_path = relationships_path.expanduser().resolve()
    output = output.expanduser().resolve()
    for source in (nodes_path, relationships_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")

    nodes = read_jsonl(nodes_path, NODE_FIELDS, "node")
    relationships = read_jsonl(relationships_path, RELATIONSHIP_FIELDS, "relationship")
    require_unique(nodes, "id", "node")
    require_unique(relationships, "id", "relationship")
    nodes.sort(key=lambda record: record["id"])
    node_indexes = {record["id"]: index for index, record in enumerate(nodes)}

    facts: set[tuple[str, str, str]] = set()
    for record in relationships:
        if record["source"] not in node_indexes:
            raise ValueError(f"relationship {record['id']} has unknown source: {record['source']}")
        if record["target"] not in node_indexes:
            raise ValueError(f"relationship {record['id']} has unknown target: {record['target']}")
        fact = (record["source"], record["relation"], record["target"])
        if fact in facts:
            raise ValueError(f"duplicate relationship fact: {fact}")
        facts.add(fact)

    relation_ids = sorted({record["relation"] for record in relationships})
    relation_indexes = {relation_id: index for index, relation_id in enumerate(relation_ids)}
    heads = np.fromiter((node_indexes[item["source"]] for item in relationships), dtype=np.int32, count=len(relationships))
    tails = np.fromiter((node_indexes[item["target"]] for item in relationships), dtype=np.int32, count=len(relationships))
    relation_types = np.fromiter(
        (relation_indexes[item["relation"]] for item in relationships),
        dtype=np.uint16,
        count=len(relationships),
    )

    source_specs = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (nodes_path, relationships_path)
    ]
    source_digest = hashlib.sha256(
        "".join(f"{Path(item['path']).name}:{item['sha256']}" for item in source_specs).encode()
    ).hexdigest()
    effective_snapshot_id = snapshot_id or f"car-kg-jsonl@sha256:{source_digest}"
    if not effective_snapshot_id.strip():
        raise ValueError("snapshot_id must be non-empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-", dir=output.parent))
    backup: Path | None = None
    try:
        arrays: dict[str, dict[str, Any]] = {}
        text = {
            "node_ids": [item["id"] for item in nodes],
            "node_labels": [item["name"] for item in nodes],
            "node_descriptions": [f"Entity type: {item['type']}" for item in nodes],
            "node_types": [item["type"] for item in nodes],
            "relation_ids": relation_ids,
            "relation_labels": relation_ids,
            "edge_ids": [item["id"] for item in relationships],
        }
        for stem, values in text.items():
            arrays.update(save_text_column(temp, stem, values))

        numeric = {
            "edge_heads.npy": heads,
            "edge_tails.npy": tails,
            "edge_relation_types.npy": relation_types,
            "out_order.npy": np.argsort(heads, kind="stable").astype(np.int64, copy=False),
            "out_offsets.npy": csr_offsets(heads, len(nodes)),
            "in_order.npy": np.argsort(tails, kind="stable").astype(np.int64, copy=False),
            "in_offsets.npy": csr_offsets(tails, len(nodes)),
        }
        for name, values in numeric.items():
            arrays[name] = save_array(temp / name, values)

        validate_arrays(temp, len(nodes), len(relationships), len(relation_ids))
        manifest = {
            "schema": "kg-platform.global-array-index",
            "schema_version": 1,
            "graph": {"graph_id": "car_kg", "snapshot_id": effective_snapshot_id},
            "builder": {
                "module": "scripts/build_car_kg_index.py",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "semantics": {
                "edge_direction": "canonical directed Car KG relationship",
                "reverse_edges_materialized": False,
                "entity_key": "Car KG node id",
                "relation_key": "Car KG relationship type",
                "edge_provenance": "source relationship id from relationships.jsonl",
                "node_description": "Entity type: <node type>",
                "text_columns": "packed UTF-8 uint8 payload with int64 offsets; no pickle/object arrays",
                "out_offsets": "node i uses out_order[out_offsets[i]:out_offsets[i+1]]",
                "in_offsets": "node i uses in_order[in_offsets[i]:in_offsets[i+1]]",
            },
            "counts": {"nodes": len(nodes), "edges": len(relationships), "relations": len(relation_ids)},
            "source_files": source_specs,
            "arrays": arrays,
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if output.exists():
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
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--relationships", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(args.nodes, args.relationships, args.output, args.overwrite, args.snapshot_id)
    print(json.dumps({"graph": manifest["graph"], "counts": manifest["counts"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
