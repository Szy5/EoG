#!/usr/bin/env python3
"""
Convert WebQSP + CWQ into separate Solver-format validation parquet files.

python training/examples/data_preprocess/webqsp_cwq_val/convert.py \
  --webqsp-path data/raw_datasets/WebQSP.test.json \
  --cwq-path data/raw_datasets/ComplexWebQuestions_test.question_entities.jsonl \
  --webqsp-output-parquet data/val/val_webqsp.parquet \
  --cwq-output-parquet data/val/val_cwq.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests
from jinja2 import Template

PACKAGE_DIR = Path(__file__).resolve().parent
MID_PATTERN = re.compile(r"^[mg]\.[0-9a-zA-Z_]+$")


@dataclass(frozen=True)
class PromptTemplates:
    system: str
    user: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def resolve_template_path(template_dir: Path, stem: str) -> Path:
    for suffix in (".jinja", ".txt"):
        candidate = template_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing prompt template '{stem}' under {template_dir} (expected .jinja or .txt)"
    )


def load_prompt_templates(template_dir: Path) -> PromptTemplates:
    return PromptTemplates(
        system=read_text(resolve_template_path(template_dir, "solver_system")),
        user=read_text(resolve_template_path(template_dir, "solver_user")),
    )


def format_initial_entities(question_entities: List[Dict[str, str]]) -> str:
    if not question_entities:
        return "(none provided)"

    entity_names: List[str] = []
    for entity in question_entities:
        name = str(entity.get("name", "")).strip()
        if name:
            entity_names.append(name)
    return ", ".join(entity_names) if entity_names else "(none provided)"


def render_user_prompt(user_template: str, question: str, question_entities: List[Dict[str, str]]) -> str:
    return Template(user_template).render(
        question=question,
        initial_entities=format_initial_entities(question_entities),
    )


def build_solver_record(
    *,
    question: str,
    question_entities: List[Dict[str, str]],
    answer_targets: List[str],
    answer_entities: List[Dict[str, str]],
    data_source: str,
    dataset_type: str,
    qid: Any,
    split: str,
    index: int,
    sparql_query: Optional[str],
    prompt_templates: PromptTemplates,
    ability: str = "kgqa-self-play",
) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("question must be non-empty")
    if not answer_targets:
        raise ValueError("answer_targets must be non-empty")

    ground_truth = {"target": list(answer_targets)}
    kgqa_create_kwargs = {
        "data_source": data_source,
        "dataset_type": dataset_type,
        "ground_truth": ground_truth,
        "question": question,
        "qid": qid,
        "question_entities": question_entities,
        "answer_entities": answer_entities,
        "sparql_query": sparql_query,
    }
    extra_info = {
        "index": index,
        "qid": qid,
        "split": split,
        "question": question,
        "source_gold_answers": list(answer_targets),
        "question_entities": question_entities,
        "answer_entities": answer_entities,
        "sparql_query": sparql_query,
        "dataset_type": dataset_type,
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "get_relations": {"create_kwargs": dict(kgqa_create_kwargs)},
            "get_triples": {"create_kwargs": dict(kgqa_create_kwargs)},
        },
    }

    prompt = [
        {"role": "system", "content": prompt_templates.system},
        {
            "role": "user",
            "content": render_user_prompt(prompt_templates.user, question, question_entities),
        },
    ]

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": {"ground_truth": ground_truth, "style": "rule"},
        "extra_info": extra_info,
        "metadata": {"source_format": f"{dataset_type}_solver_val", "qid": qid},
    }


def normalize_mid(raw_mid: str) -> str:
    mid = raw_mid.strip()
    if mid.startswith("http://rdf.freebase.com/ns/"):
        mid = mid.rsplit("/", 1)[-1]
    if mid.startswith(":"):
        mid = mid[1:]
    return mid


def setup_no_proxy(endpoint: str) -> None:
    hostname = urlparse(endpoint).hostname
    if not hostname:
        return
    for key in ("no_proxy", "NO_PROXY"):
        items = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        if hostname not in items:
            items.append(hostname)
        os.environ[key] = ",".join(items)


def resolve_mid_names(
    mids: Iterable[str],
    endpoint: str,
    timeout: float = 15.0,
    cache: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Resolve Freebase MIDs to English entity names via SPARQL."""
    cache = cache if cache is not None else {}
    pending = []
    for mid in mids:
        normalized = normalize_mid(mid)
        if not MID_PATTERN.match(normalized):
            continue
        if normalized not in cache:
            pending.append(normalized)

    for mid in pending:
        query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?name WHERE {{
  ns:{mid} ns:type.object.name ?name .
}}
LIMIT 50
""".strip()
        response = requests.get(
            endpoint,
            params={"query": query, "format": "application/sparql-results+json"},
            timeout=timeout,
        )
        response.raise_for_status()
        bindings = response.json().get("results", {}).get("bindings", [])
        best_name = None
        for binding in bindings:
            value = binding.get("name", {}).get("value", "").strip()
            lang = binding.get("name", {}).get("xml:lang", "")
            if not value:
                continue
            if lang == "en":
                best_name = value
                break
            if best_name is None:
                best_name = value
        if not best_name:
            raise ValueError(f"SPARQL returned no name for MID ns:{mid}")
        cache[mid] = best_name

    return cache


def pick_webqsp_parse(question_obj: Dict[str, Any]) -> Dict[str, Any]:
    parses = question_obj.get("Parses") or []
    if not parses:
        raise ValueError(f"WebQSP question {question_obj.get('QuestionId')} has no parses")

    for parse_obj in parses:
        comment = parse_obj.get("AnnotatorComment") or {}
        if comment.get("ParseQuality") == "Complete":
            return parse_obj
    return parses[0]


def convert_webqsp_question(
    question_obj: Dict[str, Any],
    index: int,
    prompt_templates: PromptTemplates,
    endpoint: str,
    mid_name_cache: Dict[str, str],
) -> Dict[str, Any]:
    parse_obj = pick_webqsp_parse(question_obj)
    question = (question_obj.get("RawQuestion") or question_obj.get("ProcessedQuestion") or "").strip()
    if not question:
        raise ValueError(f"WebQSP question {question_obj.get('QuestionId')} has empty question text")

    topic_name = (parse_obj.get("TopicEntityName") or "").strip()
    topic_mid = normalize_mid(parse_obj.get("TopicEntityMid") or "")
    if not topic_name or not topic_mid:
        raise ValueError(f"WebQSP question {question_obj.get('QuestionId')} missing topic entity")

    answer_entities: List[Dict[str, str]] = []
    answer_targets: List[str] = []
    literal_mids: List[str] = []
    for answer in parse_obj.get("Answers") or []:
        answer_type = (answer.get("AnswerType") or "Entity").strip()
        argument = str(answer.get("AnswerArgument") or "").strip()
        name = (answer.get("EntityName") or "").strip()

        if answer_type == "Value":
            if argument:
                answer_targets.append(argument)
                answer_entities.append({"name": argument, "id": ""})
            continue

        mid = normalize_mid(argument)
        if name:
            answer_targets.append(name)
            answer_entities.append({"name": name, "id": mid})
        elif MID_PATTERN.match(mid):
            literal_mids.append(mid)
        elif argument:
            answer_targets.append(argument)
            answer_entities.append({"name": argument, "id": ""})

    if literal_mids:
        resolve_mid_names(literal_mids, endpoint, cache=mid_name_cache)
        for mid in literal_mids:
            name = mid_name_cache[mid]
            answer_targets.append(name)
            answer_entities.append({"name": name, "id": mid})

    if not answer_targets:
        raise ValueError(f"WebQSP question {question_obj.get('QuestionId')} has no answers")

    return build_solver_record(
        question=question,
        question_entities=[{"name": topic_name, "id": topic_mid}],
        answer_targets=answer_targets,
        answer_entities=answer_entities,
        data_source="kgqa_val_webqsp",
        dataset_type="webqsp",
        qid=question_obj.get("QuestionId", index),
        split="val",
        index=index,
        sparql_query=parse_obj.get("Sparql"),
        prompt_templates=prompt_templates,
    )


def convert_cwq_question(
    row: Dict[str, Any],
    index: int,
    prompt_templates: PromptTemplates,
    endpoint: str,
    mid_name_cache: Dict[str, str],
) -> Dict[str, Any]:
    question = (row.get("question") or "").strip()
    if not question:
        raise ValueError(f"CWQ row {row.get('ID')} has empty question")

    question_entities: List[Dict[str, str]] = []
    for entity in row.get("question_entities") or []:
        name = str(entity.get("name", "")).strip()
        entity_id = normalize_mid(entity.get("kb_id") or entity.get("id") or "")
        if name:
            question_entities.append({"name": name, "id": entity_id})

    answer_mids = [normalize_mid(mid) for mid in (row.get("answer") or []) if str(mid).strip()]
    if not answer_mids:
        raise ValueError(f"CWQ row {row.get('ID')} has no answer mids")

    literal_answers: List[str] = []
    mids_to_resolve: List[str] = []
    for mid in answer_mids:
        if MID_PATTERN.match(mid):
            mids_to_resolve.append(mid)
        else:
            literal_answers.append(mid)

    if mids_to_resolve:
        resolve_mid_names(mids_to_resolve, endpoint, cache=mid_name_cache)

    answer_entities = []
    answer_targets = []
    for mid in mids_to_resolve:
        name = mid_name_cache[mid]
        answer_targets.append(name)
        answer_entities.append({"name": name, "id": mid})
    for literal in literal_answers:
        answer_targets.append(literal)
        answer_entities.append({"name": literal, "id": ""})

    return build_solver_record(
        question=question,
        question_entities=question_entities,
        answer_targets=answer_targets,
        answer_entities=answer_entities,
        data_source="kgqa_val_cwq",
        dataset_type="cwq",
        qid=row.get("ID", index),
        split="val",
        index=index,
        sparql_query=row.get("sparql"),
        prompt_templates=prompt_templates,
    )


def write_parquet(records: List[Dict[str, Any]], output_parquet: Path) -> None:
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(records), output_parquet)


def write_jsonl(records: List[Dict[str, Any]], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_webqsp_records(
    questions: List[Dict[str, Any]],
    sample_size: int,
    seed: int,
    prompt_templates: PromptTemplates,
    endpoint: str,
    mid_name_cache: Dict[str, str],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    candidates = list(questions)
    rng.shuffle(candidates)

    records: List[Dict[str, Any]] = []
    skipped: List[Tuple[Any, str]] = []
    for question_obj in candidates:
        if len(records) >= sample_size:
            break
        try:
            records.append(
                convert_webqsp_question(
                    question_obj,
                    index=len(records),
                    prompt_templates=prompt_templates,
                    endpoint=endpoint,
                    mid_name_cache=mid_name_cache,
                )
            )
        except ValueError as exc:
            skipped.append((question_obj.get("QuestionId"), str(exc)))

    if len(records) < sample_size and len(records) + len(skipped) < len(candidates):
        raise ValueError(
            f"Only converted {len(records)}/{sample_size} WebQSP samples. "
            f"Last skips: {skipped[:5]}"
        )
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} WebQSP samples (converted {len(records)})")
    return records


def collect_cwq_records(
    rows: List[Dict[str, Any]],
    sample_size: int,
    seed: int,
    prompt_templates: PromptTemplates,
    endpoint: str,
    mid_name_cache: Dict[str, str],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    candidates = list(rows)
    rng.shuffle(candidates)

    records: List[Dict[str, Any]] = []
    skipped: List[Tuple[Any, str]] = []
    for row in candidates:
        if len(records) >= sample_size:
            break
        try:
            records.append(
                convert_cwq_question(
                    row,
                    index=len(records),
                    prompt_templates=prompt_templates,
                    endpoint=endpoint,
                    mid_name_cache=mid_name_cache,
                )
            )
        except (ValueError, requests.RequestException) as exc:
            skipped.append((row.get("ID"), str(exc)))

    if len(records) < sample_size and len(records) + len(skipped) < len(candidates):
        raise ValueError(
            f"Only converted {len(records)}/{sample_size} CWQ samples. "
            f"Last skips: {skipped[:5]}"
        )
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} CWQ samples (converted {len(records)})")
    return records


@dataclass(frozen=True)
class ConvertResult:
    webqsp_records: List[Dict[str, Any]]
    cwq_records: List[Dict[str, Any]]

    @property
    def webqsp_count(self) -> int:
        return len(self.webqsp_records)

    @property
    def cwq_count(self) -> int:
        return len(self.cwq_records)

    @property
    def total_count(self) -> int:
        return self.webqsp_count + self.cwq_count


def reindex_records(records: List[Dict[str, Any]]) -> None:
    for idx, record in enumerate(records):
        record["extra_info"]["index"] = idx


def convert(
    webqsp_path: Path,
    cwq_path: Path,
    webqsp_output_parquet: Path,
    cwq_output_parquet: Path,
    webqsp_output_jsonl: Optional[Path],
    cwq_output_jsonl: Optional[Path],
    webqsp_sample_size: int,
    cwq_sample_size: int,
    seed: int,
    sparql_endpoint: str,
    template_dir: Optional[Path] = None,
) -> ConvertResult:
    setup_no_proxy(sparql_endpoint)
    prompt_templates = load_prompt_templates(template_dir or PACKAGE_DIR)
    mid_name_cache: Dict[str, str] = {}

    with webqsp_path.open("r", encoding="utf-8") as handle:
        webqsp_questions = json.load(handle).get("Questions") or []
    with cwq_path.open("r", encoding="utf-8") as handle:
        cwq_rows = [json.loads(line) for line in handle if line.strip()]

    webqsp_records = collect_webqsp_records(
        webqsp_questions,
        webqsp_sample_size,
        seed,
        prompt_templates,
        sparql_endpoint,
        mid_name_cache,
    )
    cwq_records = collect_cwq_records(
        cwq_rows,
        cwq_sample_size,
        seed + 1,
        prompt_templates,
        sparql_endpoint,
        mid_name_cache,
    )

    reindex_records(webqsp_records)
    reindex_records(cwq_records)

    write_parquet(webqsp_records, webqsp_output_parquet)
    write_parquet(cwq_records, cwq_output_parquet)
    if webqsp_output_jsonl is not None:
        write_jsonl(webqsp_records, webqsp_output_jsonl)
    if cwq_output_jsonl is not None:
        write_jsonl(cwq_records, cwq_output_jsonl)

    return ConvertResult(webqsp_records=webqsp_records, cwq_records=cwq_records)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Build Solver-format val parquet from WebQSP + CWQ.")
    parser.add_argument(
        "--webqsp-path",
        type=Path,
        default=repo_root / "data/raw_datasets/WebQSP.test.json",
    )
    parser.add_argument(
        "--cwq-path",
        type=Path,
        default=repo_root / "data/raw_datasets/ComplexWebQuestions_test.question_entities.jsonl",
    )
    parser.add_argument(
        "--webqsp-output-parquet",
        type=Path,
        default=repo_root / "data/val/val_webqsp.parquet",
    )
    parser.add_argument(
        "--webqsp-output-jsonl",
        type=Path,
        default=repo_root / "data/val/val_webqsp.jsonl",
    )
    parser.add_argument(
        "--cwq-output-parquet",
        type=Path,
        default=repo_root / "data/val/val_cwq.parquet",
    )
    parser.add_argument(
        "--cwq-output-jsonl",
        type=Path,
        default=repo_root / "data/val/val_cwq.jsonl",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=PACKAGE_DIR,
        help="Directory containing solver_system.{jinja,txt} and solver_user.{jinja,txt}",
    )
    parser.add_argument("--webqsp-sample-size", type=int, default=1639)
    parser.add_argument("--cwq-sample-size", type=int, default=3531)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sparql-endpoint",
        default=os.environ.get("KGQA_SPARQL_ENDPOINT", "http://127.0.0.1:3001/sparql"),
    )
    args = parser.parse_args()

    result = convert(
        webqsp_path=args.webqsp_path,
        cwq_path=args.cwq_path,
        webqsp_output_parquet=args.webqsp_output_parquet,
        cwq_output_parquet=args.cwq_output_parquet,
        webqsp_output_jsonl=args.webqsp_output_jsonl,
        cwq_output_jsonl=args.cwq_output_jsonl,
        webqsp_sample_size=args.webqsp_sample_size,
        cwq_sample_size=args.cwq_sample_size,
        seed=args.seed,
        sparql_endpoint=args.sparql_endpoint,
        template_dir=args.template_dir,
    )

    print(f"Wrote {result.webqsp_count} WebQSP records")
    print(f"  Parquet: {args.webqsp_output_parquet}")
    if args.webqsp_output_jsonl:
        print(f"  JSONL  : {args.webqsp_output_jsonl}")
    print(f"Wrote {result.cwq_count} CWQ records")
    print(f"  Parquet: {args.cwq_output_parquet}")
    if args.cwq_output_jsonl:
        print(f"  JSONL  : {args.cwq_output_jsonl}")
    print(f"  Templates: {args.template_dir}")


if __name__ == "__main__":
    main()
