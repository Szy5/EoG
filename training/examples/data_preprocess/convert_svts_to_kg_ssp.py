#!/usr/bin/env python3
"""
Convert SVTS task JSONL into quarl/veRL self-play training records.

Each input row is one symbolically verified task: a grounded query graph
(``path`` / ``triples`` / ``sparql_query``), its anchor entities
(``question_entity``) and the gold answer set obtained by executing the query
(``answer_entity``). Rows carry no natural-language question; the Proposer
writes one during self-play.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ENTITY_PLACEHOLDER_RE = re.compile(r"entity_(\d+)")
ANS_PLACEHOLDER_RE = re.compile(r"\bAns\b(?!@)")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    根据给定的路径读取JSONL文件
    """
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


def normalize_entity_list(entity_list: Any) -> List[Dict[str, str]]:
    """
    将实体列表转换为标准格式: (去掉空格和换行符)
    """
    normalized: List[Dict[str, str]] = []
    if not isinstance(entity_list, list):
        return normalized

    for item in entity_list:
        if isinstance(item, dict):
            if "name" in item and "id" in item:
                name = str(item["name"]).strip()
                entity_id = str(item["id"]).strip()
                if name:
                    normalized.append({"name": name, "id": entity_id})
                continue

            for name, entity_id in item.items():
                normalized_name = str(name).strip()
                normalized_id = str(entity_id).strip()
                if normalized_name:
                    normalized.append({"name": normalized_name, "id": normalized_id})
        elif isinstance(item, str):
            stripped = item.strip()
            if stripped:
                normalized.append({"name": stripped, "id": ""})

    return normalized


def normalize_path_placeholders(text: str) -> str:
    text = ENTITY_PLACEHOLDER_RE.sub(lambda match: f"Mid{match.group(1)}@", text)
    return ANS_PLACEHOLDER_RE.sub("Ans@", text)


def normalize_path_values(path_values: Any) -> List[str]:
    if not isinstance(path_values, list):
        return []
    return [normalize_path_placeholders(str(item).strip()) for item in path_values if str(item).strip()]


def render_user_template(user_template: str, sample_obj: Dict[str, Any]) -> str:
    """
    渲染用户提示词
    """
    # Seed prompts use graph tasks only; discard legacy prewritten questions.
    sample_obj = {key: value for key, value in sample_obj.items() if key not in {"question", "source_question"}}
    if re.search(r"\{\{\s*source_question\s*\}\}", user_template):
        raise ValueError("Prewritten question templates are not supported for self-play seeds")
    question_entities = normalize_entity_list(sample_obj.get("question_entity", []))
    question_entity_text = ", ".join(entity["name"] for entity in question_entities if entity.get("name"))
    answer_entities = normalize_entity_list(sample_obj.get("answer_entity", []))
    answer_entity_text = ", ".join(entity["name"] for entity in answer_entities if entity.get("name"))

    path_values = normalize_path_values(sample_obj.get("path", []))
    path_text = "\n".join(path_values)
    
    compact_json = json.dumps(sample_obj, ensure_ascii=False)
    pretty_json = json.dumps(sample_obj, ensure_ascii=False, indent=2)
    rendered = re.sub(r"\{\{\s*kgqa_sample_json\s*\}\}", lambda _: compact_json, user_template)
    rendered = re.sub(r"\{\{\s*kgqa_sample_pretty_json\s*\}\}", lambda _: pretty_json, rendered)
    rendered = re.sub(r"\{\{\s*path\s*\}\}", lambda _: path_text, rendered)
    rendered = re.sub(r"\{\{\s*question_entity\s*\}\}", lambda _: question_entity_text, rendered)
    rendered = re.sub(r"\{\{\s*answer_entity\s*\}\}", lambda _: answer_entity_text, rendered)
    return rendered


def build_record(
    index: int,
    sample_obj: Dict[str, Any],
    sys_prompt: str,
    user_template: str,
    data_source: str,
    split: str,
    ability: str,
    question_placeholder: str,
    dataset_type: str,
    synthesis_mode: str = "kg_path",
) -> Dict[str, Any]:
    question_entities = normalize_entity_list(sample_obj.get("question_entity", []))
    answer_entities = normalize_entity_list(sample_obj.get("answer_entity", []))
    answer_targets = [entity["name"] for entity in answer_entities]

    if not answer_targets:
        raise ValueError("Input object must contain at least one answer entity")

    qid = sample_obj.get("qid", index)
    path = normalize_path_values(sample_obj.get("path", []))
    triples = sample_obj.get("triples", [])
    # 用户提示词渲染
    user_prompt = render_user_template(user_template, sample_obj)

    kgqa_tool_payload = {
        "data_source": data_source,
        "dataset_type": dataset_type,
        "ground_truth": {"target": answer_targets},
        "question": question_placeholder,
        "qid": qid,
        "question_entities": question_entities,
        "answer_entities": answer_entities,
        "path": path,
        "triples": triples,
        "sparql_query": sample_obj.get("sparql_query"),
    }
    tools_kwargs = {
        tool_name: {
            "create_kwargs": dict(kgqa_tool_payload),
        }
        for tool_name in ("get_relations", "get_triples")
    }

    source_format = "svts_kg_explore_rag_jsonl" if synthesis_mode == "kg_explore_rag" else "svts_path_jsonl"

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "ability": ability,
        "reward_model": {
            "ground_truth": {"target": answer_targets},
            "style": "rule",
        },
        "extra_info": {
            "index": index,
            "qid": qid,
            "split": split,
            "question": question_placeholder,
            "question_entities": question_entities,
            "answer_entities": answer_entities,
            "path": path,
            "triples": triples,
            "num_edge": sample_obj.get("num_edge"),
            "domain": sample_obj.get("domains"),
            "template_id": sample_obj.get("template_id"),
            "sparql_query": sample_obj.get("sparql_query"),
            "dataset_type": dataset_type,
            "synthesis_mode": synthesis_mode,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
        "metadata": {
            "source_format": source_format,
            "qid": qid,
            "task_id": sample_obj.get("task_id"),
        },
    }


def write_parquet(records: List[Dict[str, Any]], output_parquet: Path) -> None:
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(records)
        pq.write_table(table, output_parquet)
    except Exception as exc:
        try:
            import pandas as pd
        except ModuleNotFoundError as pandas_exc:
            raise ModuleNotFoundError(
                "Parquet export requires pyarrow or pandas+pyarrow. Please install one of them."
            ) from pandas_exc

        try:
            dataframe = pd.DataFrame(records)
            dataframe.to_parquet(output_parquet, index=False)
        except Exception as pandas_write_exc:
            raise RuntimeError(
                "Failed to write parquet through both pyarrow and pandas fallback."
            ) from pandas_write_exc


def convert(
    input_jsonl: Path,
    output_jsonl: Path,
    sys_template_path: Path,
    user_template_path: Path,
    data_source: str,
    split: str,
    ability: str,
    question_placeholder: str,
    dataset_type: str,
    synthesis_mode: str = "kg_path",
    max_samples: Optional[int] = None,
    collect_records_for_parquet: bool = False,
) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
    """
    数据格式转换入口函数
    """
    sys_prompt = read_text(sys_template_path)
    user_template = read_text(user_template_path)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    records: Optional[List[Dict[str, Any]]] = [] if collect_records_for_parquet else None

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for index, sample_obj in enumerate(iter_jsonl(input_jsonl)):
            if max_samples is not None and index >= max_samples:
                break
            record = build_record(
                index=index,
                sample_obj=sample_obj,
                sys_prompt=sys_prompt,
                user_template=user_template,
                data_source=data_source,
                split=split,
                ability=ability,
                question_placeholder=question_placeholder,
                dataset_type=dataset_type,
                synthesis_mode=synthesis_mode,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if records is not None:
                records.append(record)
            written += 1

    return written, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SVTS task JSONL into quarl/veRL proposer seed records."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Input SVTS task JSONL file.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output training JSONL file.")
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=None,
        help="Optional parquet output path. Defaults to replacing the JSONL suffix with .parquet.",
    )
    parser.add_argument("--no-parquet", action="store_true", help="Disable parquet export.")
    default_template_dir = Path(__file__).resolve().parent.parent.parent / "quarl" / "template"
    parser.add_argument(
        "--synthesis-mode",
        choices=["kg_path", "kg_explore_rag"],
        default="kg_path",
        help="Proposer synthesis mode for generated seed records.",
    )
    parser.add_argument(
        "--sys-template",
        type=Path,
        default=None,
        help="System template used for proposer seed prompts.",
    )
    parser.add_argument(
        "--user-template",
        type=Path,
        default=None,
        help="User template used for proposer seed prompts.",
    )
    parser.add_argument("--data-source", type=str, default="kgqa_svts_freebase", help="Output data_source (kgqa_svts_<kg>).")
    parser.add_argument("--split", type=str, default="train", help="Split label stored in extra_info.")
    parser.add_argument("--ability", type=str, default="kgqa-self-play", help="Ability label for the sample.")
    parser.add_argument(
        "--question-placeholder",
        type=str,
        default="placeholder_question_to_be_generated",
        help="Placeholder question stored before proposer generation runs.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="kgqa",
        help="Dataset type stored in extra_info and future tool kwargs.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional debug row cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    should_write_parquet = not args.no_parquet
    output_parquet = args.output_parquet or args.output_jsonl.with_suffix(".parquet")
    default_template_dir = Path(__file__).resolve().parent.parent.parent / "quarl" / "template"
    if args.sys_template is None:
        args.sys_template = (
            default_template_dir / "kgqa_explore_rag_proposer_sys.jinja"
            if args.synthesis_mode == "kg_explore_rag"
            else default_template_dir / "kgqa_ssp_proposer_sys.jinja"
        )
    if args.user_template is None:
        args.user_template = (
            default_template_dir / "kgqa_explore_rag_proposer_user.jinja"
            if args.synthesis_mode == "kg_explore_rag"
            else default_template_dir / "kgqa_ssp_proposer_user.jinja"
        )

    total, records = convert(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        sys_template_path=args.sys_template,
        user_template_path=args.user_template,
        data_source=args.data_source,
        split=args.split,
        ability=args.ability,
        question_placeholder=args.question_placeholder,
        dataset_type=args.dataset_type,
        synthesis_mode=args.synthesis_mode,
        max_samples=args.max_samples,
        collect_records_for_parquet=should_write_parquet,
    )
    print(f"Done. Wrote {total} rows to {args.output_jsonl}")
    if should_write_parquet:
        assert records is not None
        write_parquet(records, output_parquet)
        print(f"Done. Wrote {total} rows to {output_parquet}")

if __name__ == "__main__":
    main()
