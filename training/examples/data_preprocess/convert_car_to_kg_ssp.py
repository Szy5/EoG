#!/usr/bin/env python3
"""Convert CarKGQA path JSONL into quarl/veRL Proposer training records.

Source rows look like ``car_train.jsonl`` / ``carbench.jsonl``:
  qid, question, question_entities, answer, cypher_query, template_id, path, …

Missing SVTS-task fields are derived:
  triples / num_edge from ``path``; sparql_query from cypher_query;
  domains := \"car\"; qid normalized to a collision-safe int.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util

_SVTS_CONV = Path(__file__).resolve().parent / "convert_svts_to_kg_ssp.py"
_SPEC = importlib.util.spec_from_file_location("convert_svts_to_kg_ssp", _SVTS_CONV)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load {_SVTS_CONV}")
_CORE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CORE)
build_record = _CORE.build_record
read_text = _CORE.read_text
write_parquet = _CORE.write_parquet

TOKEN_RE = re.compile(
    r"(?:(?P<fwd>--(?P<frel>.+?)-->)|(?P<bwd><--(?P<brel>.+?)--))"
)
QID_BASE = 9_100_000_000_000


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise TypeError(f"Expected a JSON object at line {line_no}")
            yield parsed


def path_line_to_triples(line: str) -> List[List[str]]:
    """Parse one path branch into KG triples ``[head, rel, tail]``.

    Display conventions match train_v9:
      ``A --rel--> B``  => [A, rel, B]
      ``A <--rel-- B``  => [B, rel, A]
    """
    parts: List[Tuple[str, str]] = []
    pos = 0
    for match in TOKEN_RE.finditer(line):
        node = line[pos : match.start()].strip()
        if node:
            parts.append(("n", node))
        if match.group("fwd"):
            parts.append(("fwd", match.group("frel").strip()))
        else:
            parts.append(("bwd", match.group("brel").strip()))
        pos = match.end()
    tail = line[pos:].strip()
    if tail:
        parts.append(("n", tail))
    if not parts or parts[0][0] != "n":
        raise ValueError(f"Unrecognized path line (no leading entity): {line!r}")

    triples: List[List[str]] = []
    index = 0
    while index + 2 < len(parts):
        left_kind, left = parts[index]
        edge_kind, rel = parts[index + 1]
        right_kind, right = parts[index + 2]
        if left_kind != "n" or right_kind != "n" or edge_kind not in {"fwd", "bwd"}:
            raise ValueError(f"Unrecognized path sequence in {line!r}: {parts}")
        if not rel:
            raise ValueError(f"Empty relation in path line: {line!r}")
        if edge_kind == "fwd":
            triples.append([left, rel, right])
        else:
            triples.append([right, rel, left])
        index += 2
    if index != len(parts) - 1:
        raise ValueError(f"Trailing tokens in path line {line!r}: {parts[index:]}")
    if not triples:
        raise ValueError(f"Path line produced no triples: {line!r}")
    return triples


def path_to_triples(path_values: Sequence[str]) -> List[List[str]]:
    triples: List[List[str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for line in path_values:
        text = str(line).strip()
        if not text:
            continue
        for triple in path_line_to_triples(text):
            key = (triple[0], triple[1], triple[2])
            if key in seen:
                continue
            seen.add(key)
            triples.append(triple)
    if not triples:
        raise ValueError("path produced no triples")
    return triples


def normalize_qid(raw: Any, index: int) -> int:
    text = str(raw).strip() if raw is not None else ""
    match = re.search(r"(\d+)$", text)
    if match:
        return QID_BASE + int(match.group(1))
    return QID_BASE + index + 1


def adapt_car_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    path = [str(item).strip() for item in (row.get("path") or []) if str(item).strip()]
    if not path:
        raise ValueError(f"row {index}: path must be a non-empty string list")

    question_entities = row.get("question_entities") or row.get("question_entity") or []
    answer_entities = row.get("answer") or row.get("answer_entity") or []
    if not isinstance(question_entities, list) or not question_entities:
        raise ValueError(f"row {index}: question_entities missing")
    if not isinstance(answer_entities, list) or not answer_entities:
        raise ValueError(f"row {index}: answer missing")

    triples = path_to_triples(path)
    cypher = str(row.get("cypher_query") or row.get("sparql_query") or "").strip()
    template_id = str(row.get("template_id") or "").strip()
    if not template_id:
        raise ValueError(f"row {index}: template_id missing")

    return {
        "qid": normalize_qid(row.get("qid"), index),
        "source_qid": str(row.get("qid") or ""),
        "question": str(row.get("question") or ""),
        "question_entity": question_entities,
        "answer_entity": answer_entities,
        "path": path,
        "triples": triples,
        "num_edge": len(triples),
        "domains": "car",
        "sparql_query": cypher or None,
        "template_id": template_id,
        "schema_path_id": row.get("schema_path_id"),
    }


def convert_car(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    sys_template_path: Path,
    user_template_path: Path,
    data_source: str,
    split: str,
    ability: str,
    question_placeholder: str,
    dataset_type: str,
    synthesis_mode: str,
    max_samples: Optional[int],
    output_parquet: Optional[Path],
) -> Dict[str, Any]:
    sys_prompt = read_text(sys_template_path)
    user_template = read_text(user_template_path)

    records: List[Dict[str, Any]] = []
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for index, raw in enumerate(iter_jsonl(input_jsonl)):
            if max_samples is not None and index >= max_samples:
                break
            adapted = adapt_car_row(raw, index)
            record = build_record(
                index=index,
                sample_obj=adapted,
                sys_prompt=sys_prompt,
                user_template=user_template,
                data_source=data_source,
                split=split,
                ability=ability,
                question_placeholder=question_placeholder,
                dataset_type=dataset_type,
                synthesis_mode=synthesis_mode,
            )
            # train_v9-compatible topology / identity fields
            record["extra_info"]["template_id"] = adapted["template_id"]
            record["metadata"] = {
                "qid": adapted["qid"],
                "source_format": "car_path_jsonl",
                "source_run": None,
                "task_id": adapted.get("source_qid") or None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    if not records:
        raise ValueError(f"No records converted from {input_jsonl}")

    if output_parquet is not None:
        write_parquet(records, output_parquet)

    from collections import Counter

    return {
        "input": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_parquet": str(output_parquet) if output_parquet else None,
        "rows": len(records),
        "template_id": dict(Counter(r["extra_info"]["template_id"] for r in records)),
        "data_source": data_source,
        "dataset_type": dataset_type,
        "synthesis_mode": synthesis_mode,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_template_dir = REPO_ROOT / "training" / "quarl" / "template"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=REPO_ROOT / "data/raw_datasets/Car/raw_qa/car_train.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=REPO_ROOT / "data/train/car_train.jsonl",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=REPO_ROOT / "data/train/car_train.parquet",
    )
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument(
        "--sys-template",
        type=Path,
        default=default_template_dir / "kgqa_ssp_proposer_sys.jinja",
    )
    parser.add_argument(
        "--user-template",
        type=Path,
        default=default_template_dir / "kgqa_ssp_proposer_user.jinja",
    )
    parser.add_argument("--data-source", type=str, default="kgqa_svts_car")
    parser.add_argument("--dataset-type", type=str, default="car_kgqa")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ability", type=str, default="kgqa-self-play")
    parser.add_argument(
        "--question-placeholder",
        type=str,
        default="placeholder_question_to_be_generated",
    )
    parser.add_argument(
        "--synthesis-mode",
        choices=["kg_path", "kg_explore_rag"],
        default="kg_path",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = convert_car(
        input_jsonl=args.input_jsonl.expanduser().resolve(),
        output_jsonl=args.output_jsonl.expanduser().resolve(),
        sys_template_path=args.sys_template.expanduser().resolve(),
        user_template_path=args.user_template.expanduser().resolve(),
        data_source=args.data_source,
        split=args.split,
        ability=args.ability,
        question_placeholder=args.question_placeholder,
        dataset_type=args.dataset_type,
        synthesis_mode=args.synthesis_mode,
        max_samples=args.max_samples,
        output_parquet=None
        if args.no_parquet
        else args.output_parquet.expanduser().resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
