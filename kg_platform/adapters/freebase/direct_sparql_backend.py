from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections.abc import Sequence
from typing import Any

from kg_platform.adapters.freebase.direct_sparql.relation_normalizer import ensure_endpoint_in_no_proxy
from kg_platform.domain.capabilities import BackendCapabilities, BackendMetadata
from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import (
    Direction,
    EdgeProvenance,
    EntityKey,
    EntityQuery,
    EntityRef,
    GraphRef,
    LiteralRef,
    QueryContext,
    RelationChoice,
    RelationKey,
    RelationPage,
    RelationQueryOptions,
    RelationRef,
    ResolutionResult,
    TriplePage,
    TripleQueryOptions,
    TripleRef,
)

ID_RE = re.compile(r"^(?:m|g|en)\.[A-Za-z0-9_]+$")


class FreebaseDirectSparqlBackend:
    """Freebase adapter with training-parity Direct-SPARQL semantics.

    Differences vs ``FreebaseSparqlBackend`` (intentional, for training parity):

    - predicate whitelist + meaningless-relation skip
    - BM25 ranking by session question
    - optional LLM relation filter (same env as training)
    - CVT flatten during ``get_triples``, pending flatten on next ``get_relations``
    - flatten relation naming (``rel1_suffix.rel2_N``), not ``virtual:a|b``
    """

    def __init__(
        self,
        endpoint: str,
        *,
        snapshot_id: str = "freebase@direct-sparql",
        timeout: float = 15.0,
        dataset_type: str = "cwq",
        full_list: bool = False,
        enable_llm_relation_filter: bool = True,
        max_relations: int = 20,
        triples_limit_per_relation: int = 5,
    ):
        self.endpoint = endpoint
        self.graph = GraphRef("freebase", snapshot_id)
        self.timeout = timeout
        self.dataset_type = dataset_type
        self.full_list = full_list
        self.enable_llm_relation_filter = enable_llm_relation_filter
        self.max_relations = max_relations
        self.triples_limit_per_relation = triples_limit_per_relation
        self.query_log: list[dict] = []
        ensure_endpoint_in_no_proxy(endpoint)

    def drain_query_log(self) -> list[dict]:
        rows = list(self.query_log)
        self.query_log.clear()
        return rows

    async def metadata(self) -> BackendMetadata:
        # Readiness uses httpx so a missing SPARQLWrapper install is not
        # misreported as "endpoint unavailable".
        missing = self._missing_runtime_deps()
        if missing:
            raise KGPlatformError(
                ErrorCode.BACKEND_UNAVAILABLE,
                (
                    "Freebase Direct-SPARQL runtime dependencies are missing: "
                    + ", ".join(missing)
                    + ". Activate the `verl` conda env and install SPARQLWrapper, "
                    "rank_bm25, and jinja2. "
                    f"endpoint={self.endpoint}"
                ),
            )

        async def _probe() -> None:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                    response = await client.get(
                        self.endpoint,
                        params={"query": "SELECT (1 AS ?ok) WHERE {}", "format": "application/sparql-results+json"},
                        headers={"Accept": "application/sparql-results+json"},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    rows = payload.get("results", {}).get("bindings", [])
                    if not rows or rows[0].get("ok", {}).get("value") != "1":
                        raise KGPlatformError(
                            ErrorCode.BACKEND_UNAVAILABLE,
                            f"Freebase SPARQL readiness query failed. endpoint={self.endpoint}",
                        )
            except KGPlatformError:
                raise
            except Exception as exc:
                raise KGPlatformError(
                    ErrorCode.BACKEND_UNAVAILABLE,
                    f"Freebase SPARQL endpoint is unavailable. endpoint={self.endpoint}; cause={type(exc).__name__}: {exc}",
                ) from exc

        await _probe()
        return BackendMetadata(
            self.graph,
            BackendCapabilities(
                scope="global",
                directions=frozenset({Direction.OUT, Direction.IN}),
                supports_literals=True,
                supports_virtual_relations=True,
            ),
        )

    @staticmethod
    def _missing_runtime_deps() -> list[str]:
        missing: list[str] = []
        try:
            import SPARQLWrapper  # noqa: F401
        except Exception:
            missing.append("SPARQLWrapper")
        try:
            import rank_bm25  # noqa: F401
        except Exception:
            missing.append("rank_bm25")
        try:
            import jinja2  # noqa: F401
        except Exception:
            missing.append("jinja2")
        return missing

    async def open_session(self, context: QueryContext):
        if context.graph != self.graph:
            raise KGPlatformError(ErrorCode.SNAPSHOT_MISMATCH, "Freebase snapshot mismatch.")
        return FreebaseDirectSparqlSession(self, context)

    async def aclose(self) -> None:
        return None


class FreebaseDirectSparqlSession:
    """
    Per-rollout Direct-SPARQL client with training-compatible CVT state.
    每次问答/LLM Rollout 都会创建一个Session会话，这个会话会维护一些缓存：
    问题文本、实体名缓存、CVT Flatten挂起列表、BM25、LLM Filter RANK结果等等
    """

    def __init__(self, backend: FreebaseDirectSparqlBackend, context: QueryContext):
        self.backend = backend
        self.context = context
        self._client = None
        self._filter_client = None
        self._filter_ready = False
        self._last_cvt_debug: list[dict[str, Any]] = []
        self._last_rank_filter: dict[str, Any] = {}
        self._last_hyperparams: dict[str, Any] = {}
        self._runtime_overrides: dict[str, Any] = {}
        self.names: dict[str, str] = {}

    def set_runtime_overrides(self, overrides: dict[str, Any] | None) -> None:
        self._runtime_overrides = dict(overrides or {})

    def last_rank_filter(self) -> dict[str, Any]:
        return dict(self._last_rank_filter)

    def last_hyperparams(self) -> dict[str, Any]:
        return dict(self._last_hyperparams)

    def _seed_entity_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for entity in self.context.anchors + self.context.entity_hints:
            label = (entity.label or "").strip()
            if not label:
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(label)
        return names

    def _question_raw(self) -> str:
        return (self.context.question or "").strip()

    def _question_aug(self) -> str:
        from kg_platform.application.tool_options import augmented_question

        return augmented_question(self._question_raw(), self._seed_entity_names())

    def _hyperparams(self):
        from kg_platform.application.tool_options import ToolHyperparams, merge_hyperparams

        base = ToolHyperparams(
            max_relations=self.backend.max_relations,
            limit_per_relation=self.backend.triples_limit_per_relation,
            enable_llm_relation_filter=self.backend.enable_llm_relation_filter,
        )
        return merge_hyperparams(base, self._runtime_overrides)

    def _llm_enabled(self, hp) -> bool:
        if hp.enable_llm_relation_filter is None:
            return bool(self.backend.enable_llm_relation_filter)
        return bool(hp.enable_llm_relation_filter)

    def _get_client(self):
        if self._client is None:
            from kg_platform.adapters.freebase.direct_sparql.client import DirectSparqlClient

            self._client = DirectSparqlClient(
                sparql_endpoint=self.backend.endpoint,
                timeout=int(self.backend.timeout),
                dataset_type=self.backend.dataset_type,
                full_list=self.backend.full_list,
            )
        return self._client

    def _get_filter_client(self, *, enabled: bool):
        if not enabled:
            return None
        if self._filter_ready:
            return self._filter_client
        self._filter_ready = True
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except Exception:
            pass
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL")):
            self._filter_client = None
            return None
        try:
            from kg_platform.adapters.freebase.direct_sparql.llm_client import BaseModelClient, ModelConfig

            self._filter_client = BaseModelClient(
                ModelConfig(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    base_url_env="OPENAI_BASE_URL",
                    api_key_env="OPENAI_API_KEY",
                    temperature=0.0,
                    max_tokens=1024,
                    stop=["\n\n"],
                    timeout=10.0,
                )
            )
        except Exception:
            self._filter_client = None
        return self._filter_client

    async def _filter_relations_llm(
        self,
        relations: list[dict[str, str]],
        *,
        question: str,
        entity_name: str,
        use_flatten_prompt: bool = False,
        max_keep: int,
        enabled: bool,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        meta: dict[str, Any] = {
            "triggered": False,
            "prompt_kind": "flatten" if use_flatten_prompt else "relation",
            "question": question,
            "topic_entity": entity_name,
            "input_relations": [r.get("relation", "") for r in relations],
            "output_relations": [],
            "client_available": False,
            "fallback": False,
        }
        client = self._get_filter_client(enabled=enabled)
        meta["client_available"] = client is not None
        if not relations or client is None:
            kept = relations[:max_keep]
            meta["output_relations"] = [r.get("relation", "") for r in kept]
            meta["fallback"] = True
            return kept, meta
        meta["triggered"] = True
        try:
            from kg_platform.adapters.freebase.direct_sparql.prompts import render_filter_prompt

            rel_strs = [relation["relation"] for relation in relations]
            prompt = render_filter_prompt(
                question=question,
                entity_name=entity_name,
                relations=rel_strs,
                use_flatten_prompt=use_flatten_prompt,
            )
            meta["prompt_preview"] = prompt[:1200]
            response = await asyncio.to_thread(client.generate, prompt)
            match = re.search(r"\[.*?\]", response or "", re.DOTALL)
            if not match:
                kept = relations[:max_keep]
                meta["output_relations"] = [r.get("relation", "") for r in kept]
                meta["fallback"] = True
                return kept, meta
            selected = json.loads(match.group(0))
            if not isinstance(selected, list):
                kept = relations[:max_keep]
                meta["output_relations"] = [r.get("relation", "") for r in kept]
                meta["fallback"] = True
                return kept, meta
            rel_map = {relation["relation"]: relation for relation in relations}
            ordered = [rel_map[item] for item in selected if isinstance(item, str) and item in rel_map]
            kept = ordered or relations[:max_keep]
            meta["output_relations"] = [r.get("relation", "") for r in kept]
            meta["fallback"] = not bool(ordered)
            return kept, meta
        except Exception as exc:
            kept = relations[:max_keep]
            meta["output_relations"] = [r.get("relation", "") for r in kept]
            meta["fallback"] = True
            meta["error"] = f"{type(exc).__name__}: {exc}"
            return kept, meta

    def key(self, native_id: str) -> EntityKey:
        if not ID_RE.fullmatch(native_id):
            raise KGPlatformError(ErrorCode.INVALID_ARGUMENT, "Invalid Freebase entity ID.")
        return EntityKey("freebase", native_id)

    async def resolve_entity(self, query: EntityQuery) -> ResolutionResult:
        client = self._get_client()
        raw = query.text.strip()

        def _resolve() -> ResolutionResult:
            mid = client._resolve_entity(raw)
            if not mid:
                return ResolutionResult()
            label = raw if not ID_RE.fullmatch(raw) else mid
            if ID_RE.fullmatch(raw):
                try:
                    reverse = {v: k for k, v in client.resolver.cache.items() if isinstance(v, str) and v == mid}
                    if reverse:
                        label = next(iter(reverse))
                except Exception:
                    pass
            self.names[mid] = label
            return ResolutionResult(entity=EntityRef(self.key(mid), label))

        return await asyncio.to_thread(_resolve)

    async def list_relations(self, entity: EntityKey, options: RelationQueryOptions) -> RelationPage:
        client = self._get_client()
        native_id = self.key(entity.native_id).native_id
        label = self.names.get(native_id, native_id)
        hp = self._hyperparams()
        q_raw = self._question_raw()
        q_aug = self._question_aug()
        bm25_top_k = hp.resolved_bm25_top_k()
        llm_enabled = self._llm_enabled(hp)
        sparql_limit = max(1, int(hp.sparql_limit))

        def _query_whitelist_pool() -> list[dict[str, str]]:
            # Empty question skips BM25 inside the client; we rank explicitly below.
            return client.get_relations(
                native_id,
                limit=sparql_limit,
                question="",
                top_k=max(bm25_top_k * 5, 500),
                include_flatten=bool(hp.include_pending_flatten),
            )

        after_whitelist = await asyncio.to_thread(_query_whitelist_pool)
        after_whitelist_names = [r.get("relation", "") for r in after_whitelist if r.get("relation")]

        def _bm25() -> list[dict[str, str]]:
            ranked = list(after_whitelist)
            if q_aug and ranked:
                ranked = client.rank_by_similarity(ranked, q_aug, "relation")
            return ranked[:bm25_top_k]

        after_bm25 = await asyncio.to_thread(_bm25)
        after_bm25_names = [r.get("relation", "") for r in after_bm25]

        llm_meta: dict[str, Any] = {
            "triggered": False,
            "prompt_kind": "relation",
            "question": q_raw,
            "topic_entity": label,
            "input_relations": after_bm25_names,
            "output_relations": after_bm25_names,
            "skipped_reason": None,
        }
        after_llm = after_bm25
        if len(after_bm25) > hp.llm_trigger_threshold and llm_enabled:
            shuffled = list(after_bm25)
            random.shuffle(shuffled)
            after_llm, llm_meta = await self._filter_relations_llm(
                shuffled,
                question=q_raw,
                entity_name=label,
                use_flatten_prompt=False,
                max_keep=hp.max_relations,
                enabled=True,
            )
        elif len(after_bm25) > hp.llm_trigger_threshold and not llm_enabled:
            llm_meta["skipped_reason"] = "enable_llm_relation_filter=false"
        else:
            llm_meta["skipped_reason"] = f"candidate_count<={hp.llm_trigger_threshold}"

        final = after_llm[: hp.max_relations]
        final_names = [r.get("relation", "") for r in final if r.get("relation")]

        choices: list[RelationChoice] = []
        for predicate in final_names:
            choices.append(
                RelationChoice(
                    RelationRef(RelationKey("freebase", predicate), predicate),
                    Direction.OUT,
                )
            )
        entity_ref = EntityRef(self.key(native_id), label)
        hyperparams = {
            **hp.to_dict(),
            "enable_llm_relation_filter": llm_enabled,
            "dataset_type": self.backend.dataset_type,
            "full_list": self.backend.full_list,
            "endpoint": self.backend.endpoint,
            "bm25_top_k": bm25_top_k,
        }
        rank_filter = {
            "operation": "get_relations",
            "question_raw": q_raw,
            "question_aug": q_aug,
            "seed_entity_names": self._seed_entity_names(),
            "bm25": {
                "query": q_aug,
                "document_field": "relation",
                "input_count": len(after_whitelist_names),
                "output_count": len(after_bm25_names),
                "output_relations": after_bm25_names,
            },
            "llm": llm_meta,
            "stages": {
                "after_whitelist": after_whitelist_names[:200],
                "after_bm25": after_bm25_names,
                "after_llm": [r.get("relation", "") for r in after_llm],
                "final": final_names,
            },
            "pending_flatten": list(client._pending_flatten_relations.get(native_id, []))[:20],
        }
        self._last_hyperparams = hyperparams
        self._last_rank_filter = rank_filter
        self.backend.query_log.append(
            {
                "purpose": "direct_sparql_list_relations",
                "entity": native_id,
                "available": len(after_whitelist_names),
                "returned": len(choices),
                "bm25_query": q_aug,
                "llm_question": q_raw,
            }
        )
        truncated = len(after_whitelist_names) > len(choices)
        return RelationPage(entity_ref, tuple(choices), truncated, len(after_whitelist_names))

    async def list_triples(
        self, entity: EntityKey, choices: Sequence[RelationChoice], options: TripleQueryOptions
    ) -> TriplePage:
        client = self._get_client()
        native_id = self.key(entity.native_id).native_id
        label = self.names.get(native_id, native_id)
        hp = self._hyperparams()
        q_raw = self._question_raw()
        q_aug = self._question_aug()
        llm_enabled = self._llm_enabled(hp)
        relations = [choice.relation.key.native_id for choice in choices]
        limit = min(options.limit_per_choice, hp.limit_per_relation)

        def _query() -> dict[str, Any]:
            return client.get_triples(
                native_id,
                relations,
                limit_per_relation=limit,
                question=q_aug,
                return_with_cvt_info=True,
            )

        result = await asyncio.to_thread(_query)
        raw_triples = result.get("triples", []) if isinstance(result, dict) else result
        cvt_info = result.get("cvt_info", []) if isinstance(result, dict) else []

        pending_info = client.get_pending_flatten_candidates(native_id)
        flatten_candidates = list(pending_info.get("candidates") or []) if pending_info else []
        flatten_llm_meta: dict[str, Any] = {
            "triggered": False,
            "prompt_kind": "flatten",
            "question": str((pending_info or {}).get("question") or q_aug),
            "topic_entity": str((pending_info or {}).get("entity_name") or label),
            "input_relations": [c.get("flatten_relation", "") for c in flatten_candidates],
            "output_relations": [],
        }
        if flatten_candidates and llm_enabled:
            relations_for_llm = [{"relation": c["flatten_relation"]} for c in flatten_candidates]
            filtered, flatten_llm_meta = await self._filter_relations_llm(
                relations_for_llm,
                question=str(pending_info.get("question") or q_aug),
                entity_name=str(pending_info.get("entity_name") or label),
                use_flatten_prompt=True,
                max_keep=hp.cvt_flatten_top_k,
                enabled=True,
            )
            candidate_map = {c["flatten_relation"]: c for c in flatten_candidates}
            filtered_rels = []
            for item in filtered[: hp.cvt_flatten_top_k]:
                name = item.get("relation", "")
                if name in candidate_map:
                    filtered_rels.append(name)
            if filtered_rels:
                client._pending_flatten_relations[native_id] = filtered_rels
            client.clear_pending_flatten_candidates()
        elif flatten_candidates:
            client._pending_flatten_relations[native_id] = [
                c["flatten_relation"] for c in flatten_candidates[: hp.cvt_flatten_top_k]
            ]
            flatten_llm_meta["output_relations"] = list(client._pending_flatten_relations[native_id])
            flatten_llm_meta["skipped_reason"] = "enable_llm_relation_filter=false"
            client.clear_pending_flatten_candidates()

        pending = list(client._pending_flatten_relations.get(native_id, []))
        menu_meta: dict[str, Any] = {"applied": False}
        if pending:
            other_relations = [{"relation": rel} for rel in relations[1:]]
            combined = other_relations + [{"relation": rel} for rel in pending]
            if q_aug and combined:

                def _rank_menu() -> list[dict[str, str]]:
                    return client.rank_by_similarity(combined, q_aug, "relation")

                ranked_menu = await asyncio.to_thread(_rank_menu)
                menu_meta = {
                    "applied": True,
                    "bm25_query": q_aug,
                    "input_relations": [r.get("relation", "") for r in combined],
                    "after_bm25": [r.get("relation", "") for r in ranked_menu],
                }
                if len(ranked_menu) > hp.llm_trigger_threshold and llm_enabled:
                    shuffled = list(ranked_menu)
                    random.shuffle(shuffled)
                    filtered_menu, menu_llm = await self._filter_relations_llm(
                        shuffled,
                        question=q_raw,
                        entity_name=label,
                        use_flatten_prompt=False,
                        max_keep=hp.max_relations,
                        enabled=True,
                    )
                    menu_meta["llm"] = menu_llm
                    kept_flatten = [
                        r.get("relation", "")
                        for r in filtered_menu
                        if r.get("relation") in set(pending)
                    ]
                    if kept_flatten:
                        client._pending_flatten_relations[native_id] = kept_flatten
                        pending = kept_flatten

        self._last_cvt_debug = [
            {
                "original_relation": item.get("original_relation"),
                "cvt_node_id": item.get("cvt_node_id"),
                "cvt_relation": item.get("cvt_relation"),
                "flatten_relation": item.get("flatten_relation"),
                "direction": item.get("direction"),
            }
            for item in (cvt_info or [])[:50]
            if isinstance(item, dict)
        ]
        pending = list(client._pending_flatten_relations.get(native_id, []))
        hyperparams = {
            **hp.to_dict(),
            "enable_llm_relation_filter": llm_enabled,
            "dataset_type": self.backend.dataset_type,
            "limit_per_relation_effective": limit,
            "endpoint": self.backend.endpoint,
        }
        rank_filter = {
            "operation": "get_triples",
            "question_raw": q_raw,
            "question_aug": q_aug,
            "seed_entity_names": self._seed_entity_names(),
            "requested_relations": relations,
            "physical_triple_count": len(raw_triples),
            "cvt_candidates": [
                {
                    "flatten_relation": c.get("flatten_relation"),
                    "original_relation": c.get("original_relation"),
                    "cvt_node_id": c.get("cvt_node_id"),
                    "cvt_relation": c.get("cvt_relation"),
                    "direction": c.get("direction"),
                }
                for c in flatten_candidates[:80]
            ],
            "flatten_llm": flatten_llm_meta,
            "pending_flatten": pending[:40],
            "post_cvt_menu": menu_meta,
        }
        self._last_hyperparams = hyperparams
        self._last_rank_filter = rank_filter
        self.backend.query_log.append(
            {
                "purpose": "direct_sparql_list_triples",
                "entity": native_id,
                "relations": relations,
                "triple_count": len(raw_triples),
                "cvt_count": len(self._last_cvt_debug),
                "pending_flatten": pending[:20],
            }
        )

        entity_ref = EntityRef(self.key(native_id), label)
        self.names[native_id] = label
        triples: list[TripleRef] = []
        for item in raw_triples:
            if not isinstance(item, dict):
                continue
            head_id = str(item.get("head_id", "")).strip()
            tail_id = str(item.get("tail_id", "")).strip()
            relation = str(item.get("relation", "")).strip()
            head_name = str(item.get("head", head_id)).strip() or head_id
            tail_name = str(item.get("tail", tail_id)).strip() or tail_id
            if not relation:
                continue
            if head_id and ID_RE.fullmatch(head_id):
                self.names[head_id] = head_name
                head: EntityRef | LiteralRef = EntityRef(self.key(head_id), head_name)
            else:
                continue
            if tail_id and ID_RE.fullmatch(tail_id):
                self.names[tail_id] = tail_name
                tail: EntityRef | LiteralRef = EntityRef(self.key(tail_id), tail_name)
            else:
                tail = LiteralRef(tail_name)
            matched = Direction.OUT if head_id == native_id else Direction.IN
            path: tuple[str, ...] = ()
            virtual = False
            if relation in client._cvt_mapping:
                mapping = client._cvt_mapping[relation]
                if len(mapping) >= 3:
                    virtual = True
                    path = (str(mapping[0]), str(mapping[2]))
            triples.append(
                TripleRef(
                    head,
                    RelationRef(RelationKey("freebase", relation), relation),
                    tail,
                    EdgeProvenance(self.backend.graph, virtual=virtual, source_path=path),
                    matched,
                )
            )
            if len(triples) >= options.total_limit:
                break

        return TriplePage(entity_ref, tuple(triples), False, len(triples))

    def last_cvt_debug(self) -> list[dict[str, Any]]:
        return list(self._last_cvt_debug)

    async def aclose(self) -> None:
        self._client = None
