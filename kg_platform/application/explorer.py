from __future__ import annotations

from kg_platform.application.formatting import format_error, format_relations, format_triples
from kg_platform.application.sessions import KGSessionManager
from kg_platform.application.tracing import (
    TraceBuilder,
    alias_display_lines,
    drain_backend_sparql,
    preview_aliases,
    preview_relations,
    preview_triples,
    relation_display_lines,
    triple_display_lines,
)
from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import (
    Direction,
    EntityQuery,
    QueryContext,
    QueryPolicy,
    RelationChoice,
    RelationQueryOptions,
    ToolResult,
    TripleQueryOptions,
)


class KGApplicationService:
    """Backend-neutral orchestration used by both HTTP and veRL tool surfaces.

    Backends expose native graph records; this layer applies rollout policy,
    deterministic ordering, entity registration, and relation grant security.
    """
    def __init__(self, sessions: KGSessionManager, policy: QueryPolicy | None = None):
        self.sessions = sessions
        self.policy = policy or QueryPolicy()

    async def acquire(self, context: QueryContext) -> None:
        await self.sessions.acquire(context)

    async def release(self, session_id: str) -> None:
        await self.sessions.release(session_id)

    async def _resolve(self, session_id: str, raw_entity: str):
        """
        Resolve session aliases first, then fall back to the active graph.
        """
        managed = self.sessions.get(session_id)
        entity = managed.resolve_registered_entity(raw_entity)
        if entity:
            return managed, entity, []
        drain_backend_sparql(managed)  # drop unrelated prior probes
        # 具体的resolve_entity实现由各个KG后端自己实现
        result = await managed.backend_session.resolve_entity(EntityQuery(raw_entity))
        sparql = drain_backend_sparql(managed)
        if result.entity:
            managed.register_entity(result.entity)
            return managed, result.entity, sparql
        if result.candidates:
            for candidate in result.candidates:
                managed.register_entity(candidate)
            raise KGPlatformError(
                ErrorCode.ENTITY_AMBIGUOUS,
                "Entity label is ambiguous in this graph snapshot.",
                details={"candidate_count": len(result.candidates), "sparql_queries": sparql},
            )
        raise KGPlatformError(
            ErrorCode.ENTITY_NOT_FOUND,
            "Entity was not found.",
            details={"sparql_queries": sparql},
        )

    async def list_relations(
        self,
        session_id: str,
        raw_entity: str,
        *,
        overrides: dict | None = None,
    ) -> ToolResult:
        trace = TraceBuilder()
        try:
            t0 = trace.start()
            managed, entity, resolve_sparql = await self._resolve(session_id, raw_entity)
            trace.add(
                "S0_resolve",
                started_at=t0,
                counts={"candidates": 1},
                items_preview=[
                    {
                        "label": entity.label,
                        "native_id": entity.key.native_id,
                        "scope_id": entity.key.scope_id,
                    }
                ],
                display_lines=[f"{entity.label}  →  {entity.key.native_id}"],
                sparql_queries=resolve_sparql,
                notes=["Resolved via session alias or backend label/mid lookup"],
            )

            if hasattr(managed.backend_session, "set_runtime_overrides"):
                managed.backend_session.set_runtime_overrides(overrides)

            t1 = trace.start()
            drain_backend_sparql(managed)
            page = await managed.backend_session.list_relations(
                entity.key,
                RelationQueryOptions(limit=self.policy.relation_candidate_limit),
            )
            sparql_raw = drain_backend_sparql(managed)
            hyperparams = {}
            rank_filter = {}
            if hasattr(managed.backend_session, "last_hyperparams"):
                hyperparams = managed.backend_session.last_hyperparams()  # type: ignore[attr-defined]
            if hasattr(managed.backend_session, "last_rank_filter"):
                rank_filter = managed.backend_session.last_rank_filter()  # type: ignore[attr-defined]

            canonical: dict[tuple[str, Direction], RelationChoice] = {}
            ordered: list[RelationChoice] = []
            for choice in page.choices:
                key = (choice.relation.key.native_id, choice.direction)
                if key in canonical:
                    continue
                canonical[key] = choice
                ordered.append(choice)
            if not self.policy.preserve_backend_order:
                ordered = sorted(
                    ordered,
                    key=lambda item: (item.relation.label.casefold(), item.relation.key.native_id, item.direction.value),
                )
            available = page.available_count if page.available_count is not None else len(ordered)
            whitelist_lines = list((rank_filter.get("stages") or {}).get("after_whitelist") or [])[:80]
            s1_lines = whitelist_lines or relation_display_lines(ordered)
            trace.add(
                "S1_backend_raw",
                started_at=t1,
                counts={"available": available, "returned": len(ordered), "after_whitelist": len(whitelist_lines) or available},
                items_preview=preview_relations(ordered),
                display_lines=s1_lines,
                sparql_queries=sparql_raw,
                truncated=bool(page.truncated),
                notes=[
                    "SPARQL + whitelist/meaningless filter (Direct-SPARQL) or raw backend candidates",
                    f"hyperparams={ {k: hyperparams.get(k) for k in ('sparql_limit','dataset_type','include_pending_flatten') if hyperparams} }",
                ],
            )

            effective_max = int(hyperparams.get("max_relations") or self.policy.max_relations)
            t2 = trace.start()
            selected = ordered[:effective_max]
            aliases = managed.grant_relations(entity.key, selected)
            truncated = page.truncated or len(ordered) > len(selected)
            trace.add(
                "S2_platform_policy",
                started_at=t2,
                counts={"selected": len(selected), "returned": len(aliases), "max_relations": effective_max},
                items_preview=preview_aliases(aliases),
                display_lines=alias_display_lines(aliases),
                truncated=truncated,
                notes=(
                    ["grant_relations; backend order preserved (Direct-SPARQL/BM25/LLM)"]
                    if self.policy.preserve_backend_order
                    else ["grant_relations + max_relations truncate (no LLM on this path)"]
                ),
            )

            if rank_filter.get("operation") == "get_relations":
                t3 = trace.start()
                bm25 = rank_filter.get("bm25") or {}
                llm = rank_filter.get("llm") or {}
                display = [
                    f"BM25 query (augmented): {rank_filter.get('question_aug')!r}",
                    f"LLM question (raw): {rank_filter.get('question_raw')!r}",
                    f"seed entities: {rank_filter.get('seed_entity_names')}",
                    f"BM25: {bm25.get('input_count')} → {bm25.get('output_count')} (top_k)",
                    f"LLM triggered={llm.get('triggered')} skipped={llm.get('skipped_reason')}",
                    "--- after BM25 ---",
                    *[str(x) for x in (bm25.get("output_relations") or [])[:40]],
                    "--- after LLM ---",
                    *[str(x) for x in (llm.get("output_relations") or [])[:40]],
                ]
                trace.add(
                    "S3_optional_rank_filter",
                    started_at=t3,
                    counts={
                        "bm25_out": int(bm25.get("output_count") or 0),
                        "llm_out": len(llm.get("output_relations") or []),
                        "llm_triggered": 1 if llm.get("triggered") else 0,
                    },
                    items_preview=[
                        {"bm25_query": rank_filter.get("question_aug"), "llm_question": rank_filter.get("question_raw")}
                    ],
                    display_lines=display,
                    notes=[
                        "BM25 uses augmented_question; LLM Filter uses raw session.question + Topic Entity",
                        f"llm_prompt_kind={llm.get('prompt_kind')} fallback={llm.get('fallback')}",
                    ],
                )
            elif self.policy.preserve_backend_order:
                trace.skip(
                    "S3_optional_rank_filter",
                    "Direct-SPARQL path but no rank_filter payload attached.",
                )
            else:
                trace.skip(
                    "S3_optional_rank_filter",
                    "当前 KG Platform / KGExplorerTool 路径不做 LLM Filter；Direct-SPARQL KGQATool 才会在候选>10 时调用 LLM。",
                )

            text = format_relations(aliases)
            t4 = trace.start()
            trace.add(
                "S4_final",
                started_at=t4,
                counts={"returned": len(aliases), "available": available},
                items_preview=preview_aliases(aliases),
                display_lines=text.splitlines() if text else [],
                truncated=truncated,
                notes=["Text the agent receives inside <information>"],
            )
            return ToolResult(
                operation="get_relations",
                text=text,
                entity=entity,
                returned_count=len(aliases),
                available_count=available,
                truncated=truncated,
                debug={
                    "choices": selected,
                    "trace": trace.to_dict(),
                    "hyperparams": hyperparams,
                    "rank_filter": rank_filter,
                },
            )
        except KGPlatformError as exc:
            if not any(stage.id == "S0_resolve" for stage in trace.stages):
                candidate_count = exc.details.get("candidate_count") if isinstance(exc.details, dict) else None
                counts = {"candidates": int(candidate_count)} if isinstance(candidate_count, int) else {}
                sparql = exc.details.get("sparql_queries") if isinstance(exc.details, dict) else None
                trace.add(
                    "S0_resolve",
                    started_at=trace._clock_start,
                    status="error",
                    notes=[exc.message],
                    counts=counts,
                    display_lines=[exc.message],
                    sparql_queries=list(sparql) if isinstance(sparql, list) else [],
                )
            for stage_id in ("S1_backend_raw", "S2_platform_policy", "S3_optional_rank_filter", "S4_final"):
                if not any(stage.id == stage_id for stage in trace.stages):
                    trace.skip(stage_id, f"skipped after {exc.code.value}")
            details = dict(exc.details)
            details["trace"] = trace.to_dict()
            return ToolResult(
                "get_relations",
                format_error(exc.code.value, details if isinstance(details, dict) else None),
                error_code=exc.code.value,
                debug=details,
            )

    async def fetch_triples(
        self,
        session_id: str,
        raw_entity: str,
        relation_aliases: list[str],
        *,
        overrides: dict | None = None,
    ) -> ToolResult:
        trace = TraceBuilder()
        try:
            t0 = trace.start()
            managed, entity, resolve_sparql = await self._resolve(session_id, raw_entity)
            trace.add(
                "S0_resolve",
                started_at=t0,
                counts={"candidates": 1},
                items_preview=[
                    {
                        "label": entity.label,
                        "native_id": entity.key.native_id,
                        "scope_id": entity.key.scope_id,
                    }
                ],
                display_lines=[f"{entity.label}  →  {entity.key.native_id}"],
                sparql_queries=resolve_sparql,
            )

            if hasattr(managed.backend_session, "set_runtime_overrides"):
                managed.backend_session.set_runtime_overrides(overrides)

            max_requested = self.policy.max_requested_relations
            if overrides and overrides.get("max_requested_relations") is not None:
                max_requested = int(overrides["max_requested_relations"])

            t1 = trace.start()
            choices = managed.consume_relations(entity.key, relation_aliases, max_requested)
            trace.add(
                "S1_relation_check",
                started_at=t1,
                counts={"requested": len(relation_aliases), "accepted": len(choices), "max_requested": max_requested},
                items_preview=[{"alias": alias} for alias in relation_aliases[:50]],
                display_lines=list(relation_aliases),
                notes=["Aliases must come from the latest get_relations grant"],
            )

            limit_per = self.policy.max_triples_per_relation
            if overrides and overrides.get("limit_per_relation") is not None:
                limit_per = int(overrides["limit_per_relation"])

            t2 = trace.start()
            drain_backend_sparql(managed)
            page = await managed.backend_session.list_triples(
                entity.key,
                choices,
                TripleQueryOptions(
                    limit_per_choice=limit_per,
                    total_limit=self.policy.max_triples_total,
                ),
            )
            sparql_raw = drain_backend_sparql(managed)
            hyperparams = {}
            rank_filter = {}
            if hasattr(managed.backend_session, "last_hyperparams"):
                hyperparams = managed.backend_session.last_hyperparams()  # type: ignore[attr-defined]
            if hasattr(managed.backend_session, "last_rank_filter"):
                rank_filter = managed.backend_session.last_rank_filter()  # type: ignore[attr-defined]

            alias_by_choice: dict[tuple[str, Direction], str] = {}
            for alias in relation_aliases:
                grant = managed.relation_grants.get((entity.key, alias.strip().casefold()))
                if grant:
                    for choice in grant.choices:
                        alias_by_choice[(choice.relation.key.native_id, choice.direction)] = alias
            rendered: list[tuple[object, str]] = []
            for triple in page.triples:
                direction = triple.matched_direction or Direction.OUT
                alias = alias_by_choice.get((triple.relation.key.native_id, direction), triple.relation.label)
                rendered.append((triple, alias))
                managed.register_entity(triple.head)
                if hasattr(triple.tail, "key"):
                    managed.register_entity(triple.tail)  # type: ignore[arg-type]
            available = page.available_count if page.available_count is not None else len(page.triples)
            trace.add(
                "S2_backend_raw",
                started_at=t2,
                counts={"available": available, "returned": len(page.triples)},
                items_preview=preview_triples(rendered),
                display_lines=triple_display_lines(rendered),
                sparql_queries=sparql_raw,
                truncated=bool(page.truncated),
                notes=[f"limit_per_relation={hyperparams.get('limit_per_relation_effective', limit_per)}"],
            )

            cvt_debug = []
            pending_flatten: list[str] = list(rank_filter.get("pending_flatten") or [])
            backend_session = managed.backend_session
            if hasattr(backend_session, "last_cvt_debug"):
                cvt_debug = list(backend_session.last_cvt_debug())  # type: ignore[attr-defined]
            if not pending_flatten and hasattr(backend_session, "_get_client"):
                try:
                    client = backend_session._get_client()  # type: ignore[attr-defined]
                    pending_flatten = list(client._pending_flatten_relations.get(entity.key.native_id, []))
                except Exception:
                    pending_flatten = []

            flatten_cands = list(rank_filter.get("cvt_candidates") or [])
            flatten_llm = rank_filter.get("flatten_llm") or {}
            if cvt_debug or pending_flatten or flatten_cands:
                t3 = trace.start()
                display = [
                    f"BM25/aug question: {rank_filter.get('question_aug')!r}",
                    f"LLM raw question: {rank_filter.get('question_raw')!r}",
                    f"flatten LLM triggered={flatten_llm.get('triggered')} kind={flatten_llm.get('prompt_kind')}",
                    "--- CVT flatten candidates ---",
                    *[
                        f"{c.get('flatten_relation')}  ({c.get('original_relation')} / {c.get('cvt_relation')})"
                        for c in flatten_cands[:40]
                    ],
                    "--- pending flatten (next get_relations) ---",
                    *pending_flatten[:40],
                ]
                trace.add(
                    "S3_cvt_flatten",
                    started_at=t3,
                    counts={
                        "cvt_nodes": len(cvt_debug),
                        "flatten_candidates": len(flatten_cands),
                        "pending_flatten": len(pending_flatten),
                    },
                    items_preview=[{"flatten_relation": item} for item in pending_flatten[:50]],
                    display_lines=display,
                    notes=[
                        "Direct-SPARQL: CVT flatten during get_triples; pending appears on next get_relations.",
                        f"cvt_flatten_top_k={hyperparams.get('cvt_flatten_top_k')}",
                    ],
                )
            else:
                trace.skip("S3_cvt_flatten", "No CVT flatten candidates for this triples call.")

            text = format_triples(rendered)  # type: ignore[arg-type]
            t4 = trace.start()
            trace.add(
                "S4_final",
                started_at=t4,
                counts={"returned": len(page.triples), "available": available},
                items_preview=preview_triples(rendered),
                display_lines=text.splitlines() if text else [],
                truncated=bool(page.truncated),
                notes=["Text the agent receives inside <information>"],
            )
            return ToolResult(
                operation="get_triples",
                text=text,
                entity=entity,
                returned_count=len(page.triples),
                available_count=page.available_count,
                truncated=page.truncated,
                debug={
                    "triples": page.triples,
                    "trace": trace.to_dict(),
                    "cvt_debug": cvt_debug,
                    "pending_flatten": pending_flatten,
                    "hyperparams": hyperparams,
                    "rank_filter": rank_filter,
                },
            )
        except KGPlatformError as exc:
            if not any(stage.id == "S0_resolve" for stage in trace.stages):
                sparql = exc.details.get("sparql_queries") if isinstance(exc.details, dict) else None
                trace.add(
                    "S0_resolve",
                    started_at=trace._clock_start,
                    status="error",
                    notes=[exc.message],
                    display_lines=[exc.message],
                    sparql_queries=list(sparql) if isinstance(sparql, list) else [],
                )
            elif not any(stage.id == "S1_relation_check" for stage in trace.stages):
                trace.add(
                    "S1_relation_check",
                    started_at=trace._clock_start,
                    status="error",
                    notes=[exc.message],
                    display_lines=[exc.message],
                )
            for stage_id in ("S1_relation_check", "S2_backend_raw", "S3_cvt_flatten", "S4_final"):
                if not any(stage.id == stage_id for stage in trace.stages):
                    trace.skip(stage_id, f"skipped after {exc.code.value}")
            details = dict(exc.details)
            details["trace"] = trace.to_dict()
            return ToolResult(
                "get_triples",
                format_error(exc.code.value, details if isinstance(details, dict) else None),
                error_code=exc.code.value,
                debug=details,
            )
