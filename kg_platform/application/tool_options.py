from __future__ import annotations

"""Shared helpers for Console / Direct-SPARQL training-parity tool options."""

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class ToolHyperparams:
    """Per-call overrides for Direct-SPARQL Freebase tooling.

    Defaults mirror KGQATool + DirectSparqlClient training behaviour.
    """

    sparql_limit: int = 100
    max_relations: int = 20
    bm25_top_k: int | None = None  # default: max_relations * 3
    llm_trigger_threshold: int = 10
    enable_llm_relation_filter: bool | None = None  # None → backend default
    include_pending_flatten: bool = True
    max_requested_relations: int = 4
    limit_per_relation: int = 5
    cvt_flatten_top_k: int = 8

    def resolved_bm25_top_k(self) -> int:
        if self.bm25_top_k is not None and self.bm25_top_k > 0:
            return int(self.bm25_top_k)
        return max(1, int(self.max_relations) * 3)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bm25_top_k"] = self.resolved_bm25_top_k()
        return data


def merge_hyperparams(base: ToolHyperparams, overrides: dict[str, Any] | None) -> ToolHyperparams:
    if not overrides:
        return ToolHyperparams(**asdict(base))
    allowed = {f.name for f in fields(ToolHyperparams)}
    payload = asdict(base)
    for key, value in overrides.items():
        if key not in allowed or value is None:
            continue
        payload[key] = value
    return ToolHyperparams(**payload)


def augmented_question(question: str, entity_names: list[str] | tuple[str, ...]) -> str:
    """Match KGQATool._augment_question: BM25 query = question + seed entity names."""
    q = (question or "").strip()
    names: list[str] = []
    seen: set[str] = set()
    for name in entity_names:
        text = str(name or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(text)
    if not names:
        return q
    return (q + " " + " ".join(names)).strip()


def parse_question_entities(raw: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Normalize console/create_kwargs-style question_entities."""
    if not raw:
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        native_id = str(item.get("native_id") or item.get("id") or "").strip()
        label = str(item.get("label") or item.get("name") or "").strip()
        if not native_id or not label:
            continue
        out.append({"native_id": native_id, "label": label})
    return out
