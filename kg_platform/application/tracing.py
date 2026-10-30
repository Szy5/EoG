from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


StageStatus = Literal["ok", "error", "skipped"]

STAGE_TITLES = {
    "S0_resolve": "实体解析 Resolve",
    "S1_backend_raw": "后端召回 SPARQL/Raw",
    "S1_relation_check": "关系合法性检查",
    "S2_platform_policy": "平台策略截断/授权",
    "S3_optional_rank_filter": "LLM / Rank Filter",
    "S4_final": "最终返回给 Agent",
}


@dataclass
class TraceStage:
    id: str
    status: StageStatus = "ok"
    latency_ms: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    items_preview: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    title: str = ""
    sparql_queries: list[dict[str, Any]] = field(default_factory=list)
    display_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title or STAGE_TITLES.get(self.id, self.id),
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "counts": dict(self.counts),
            "items_preview": list(self.items_preview),
            "display_lines": list(self.display_lines),
            "sparql_queries": list(self.sparql_queries),
            "truncated": self.truncated,
            "notes": list(self.notes),
        }


@dataclass
class TraceBuilder:
    """Accumulate timed stages for one tool call without changing agent text."""

    stages: list[TraceStage] = field(default_factory=list)
    _clock_start: float = field(default_factory=time.perf_counter)

    def start(self) -> float:
        return time.perf_counter()

    def add(
        self,
        stage_id: str,
        *,
        started_at: float,
        status: StageStatus = "ok",
        counts: dict[str, int] | None = None,
        items_preview: list[dict[str, Any]] | None = None,
        truncated: bool = False,
        notes: list[str] | None = None,
        sparql_queries: list[dict[str, Any]] | None = None,
        display_lines: list[str] | None = None,
        title: str | None = None,
    ) -> TraceStage:
        stage = TraceStage(
            id=stage_id,
            status=status,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            counts=counts or {},
            items_preview=items_preview or [],
            truncated=truncated,
            notes=notes or [],
            title=title or STAGE_TITLES.get(stage_id, stage_id),
            sparql_queries=sparql_queries or [],
            display_lines=display_lines or [],
        )
        self.stages.append(stage)
        return stage

    def skip(self, stage_id: str, reason: str) -> TraceStage:
        stage = TraceStage(
            id=stage_id,
            status="skipped",
            notes=[reason],
            title=STAGE_TITLES.get(stage_id, stage_id),
            display_lines=[reason],
        )
        self.stages.append(stage)
        return stage

    def summary(self) -> dict[str, Any]:
        by_id = {stage.id: stage for stage in self.stages}
        drop_counts: dict[str, int] = {}
        s1 = by_id.get("S1_backend_raw")
        s2 = by_id.get("S2_platform_policy")
        s4 = by_id.get("S4_final")
        if s1 and s2:
            available = int(s1.counts.get("available", s1.counts.get("returned", 0)))
            selected = int(s2.counts.get("returned", s2.counts.get("selected", 0)))
            drop = max(available - selected, 0)
            if drop:
                drop_counts["S1_to_S2"] = drop
        if s2 and s4:
            selected = int(s2.counts.get("returned", s2.counts.get("selected", 0)))
            final = int(s4.counts.get("returned", 0))
            drop = max(selected - final, 0)
            if drop:
                drop_counts["S2_to_S4"] = drop
        elif s1 and s4 and "S2_platform_policy" not in by_id:
            available = int(s1.counts.get("available", s1.counts.get("returned", 0)))
            final = int(s4.counts.get("returned", 0))
            drop = max(available - final, 0)
            if drop:
                drop_counts["S1_to_S4"] = drop

        bottleneck = None
        if drop_counts:
            if "S1_to_S2" in drop_counts and drop_counts.get("S1_to_S2", 0) >= drop_counts.get("S2_to_S4", 0):
                bottleneck = "S2_platform_policy"
            elif "S2_to_S4" in drop_counts:
                bottleneck = "S4_final"
            elif "S1_to_S4" in drop_counts:
                bottleneck = "S4_final"
        else:
            slow = max(self.stages, key=lambda stage: stage.latency_ms, default=None)
            if slow and slow.status != "skipped":
                bottleneck = slow.id

        return {"bottleneck_hint": bottleneck, "drop_counts": drop_counts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "summary": self.summary(),
            "total_latency_ms": round((time.perf_counter() - self._clock_start) * 1000.0, 3),
        }


PREVIEW_LIMIT = 80


def preview_relations(choices: list[Any], *, limit: int = PREVIEW_LIMIT) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for choice in choices[:limit]:
        relation = choice.relation
        items.append(
            {
                "relation": relation.label,
                "relation_id": relation.key.native_id,
                "direction": choice.direction.value,
                "edge_count": choice.edge_count,
            }
        )
    return items


def preview_aliases(aliases: list[str], *, limit: int = PREVIEW_LIMIT) -> list[dict[str, Any]]:
    return [{"alias": alias} for alias in aliases[:limit]]


def preview_triples(rendered: list[tuple[Any, str]], *, limit: int = PREVIEW_LIMIT) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for triple, alias in rendered[:limit]:
        tail = triple.tail
        if hasattr(tail, "key"):
            tail_label = getattr(tail, "label", str(tail))
        else:
            tail_label = getattr(tail, "lexical", str(tail))
        items.append(
            {
                "head": triple.head.label,
                "relation": alias,
                "tail": tail_label,
                "direction": (triple.matched_direction.value if triple.matched_direction else None),
            }
        )
    return items


def relation_display_lines(choices: list[Any], *, limit: int = PREVIEW_LIMIT) -> list[str]:
    lines: list[str] = []
    for choice in choices[:limit]:
        count = choice.edge_count
        suffix = f"  (x{count}, {choice.direction.value})" if count is not None else f"  ({choice.direction.value})"
        lines.append(f"{choice.relation.label}{suffix}")
    if len(choices) > limit:
        lines.append(f"… and {len(choices) - limit} more")
    return lines


def alias_display_lines(aliases: list[str], *, limit: int = PREVIEW_LIMIT) -> list[str]:
    lines = list(aliases[:limit])
    if len(aliases) > limit:
        lines.append(f"… and {len(aliases) - limit} more")
    return lines


def triple_display_lines(rendered: list[tuple[Any, str]], *, limit: int = PREVIEW_LIMIT) -> list[str]:
    lines: list[str] = []
    for triple, alias in rendered[:limit]:
        tail = triple.tail
        tail_label = tail.label if hasattr(tail, "key") else getattr(tail, "lexical", str(tail))
        lines.append(f"[{triple.head.label}, {alias}, {tail_label}]")
    if len(rendered) > limit:
        lines.append(f"… and {len(rendered) - limit} more")
    return lines


def drain_backend_sparql(managed) -> list[dict[str, Any]]:
    backend = getattr(getattr(managed, "backend_session", None), "backend", None)
    drain = getattr(backend, "drain_query_log", None)
    if callable(drain):
        return drain()
    return []
