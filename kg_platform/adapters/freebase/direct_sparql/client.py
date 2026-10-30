from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

from kg_platform.adapters.freebase.direct_sparql.entity_resolver import EntityResolver
from kg_platform.adapters.freebase.direct_sparql.relation_normalizer import (
    ensure_endpoint_in_no_proxy,
    normalize_relation,
    uri_to_dotted,
)
from kg_platform.adapters.freebase.direct_sparql.whitelist import load_relation_whitelist

logger = logging.getLogger(__name__)

MEANINGLESS_RELATION_PATTERNS = (
    "type.object.type",
    "type.type.instance",
    "type.type.properties",
    "type.property.schema",
    "common.topic.article",
    "common.topic.image",
    "common.image_source",
    "common.document",
    "common.webpage",
    "common.license",
    "common.topic.description",
    "freebase.type_hints.included_types",
    "freebase.property_hints.best_value_properties",
    "freebase.property_hints.mediator",
    "freebase.property_hints.unit",
    "freebase.property_hints.expected_type",
    "freebase.type_hints.enumeration",
    "type.object.key",
    "type.object.permission",
    "type.object.profile",
    "type.object.service",
    "common.topic.topic_equivalent_webpage",
    "common.topic.topic_equivalent",
)


class DirectSparqlClient:
    """SPARQL client: resolve entities, query Freebase, whitelist + BM25 + CVT flatten."""

    def __init__(
        self,
        sparql_endpoint: str = "http://localhost:18890/sparql",
        timeout: int = 10,
        llm_filter_callback: Callable[..., Any] | None = None,
        dataset_type: str = "cwq",
        full_list: bool = False,
        entity_cache: dict[str, str | None] | None = None,
    ):
        from SPARQLWrapper import JSON, SPARQLWrapper

        self.sparql_endpoint = sparql_endpoint
        self.timeout = timeout
        ensure_endpoint_in_no_proxy(sparql_endpoint)
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        self.resolver = EntityResolver(sparql_endpoint, timeout=timeout, cache=entity_cache)
        self._relation_whitelist = load_relation_whitelist(dataset_type, full_list=full_list)
        self._llm_filter_callback = llm_filter_callback
        self._rel_cache: dict[str, list[str]] = {}
        # flatten_relation -> (original_relation, cvt_node_id, cvt_relation, direction)
        self._cvt_mapping: dict[str, tuple] = {}
        self._pending_flatten_relations: dict[str, list[str]] = {}
        self._pending_flatten_candidates: dict[str, dict[str, Any]] = {}

    def _apply_relation_whitelist(self, relations: list[dict[str, str]]) -> list[dict[str, str]]:
        if not relations or not self._relation_whitelist:
            return relations
        filtered = [r for r in relations if r.get("relation") in self._relation_whitelist]
        return filtered or relations

    @staticmethod
    def _is_cvt_node(tail_id: str, tail_name: str) -> bool:
        if not (tail_id.startswith("m.") or tail_id.startswith("g.")):
            return False
        if not tail_name or tail_name == tail_id:
            return True
        if tail_name.replace("/", ".").replace(":", ".") == tail_id:
            return True
        return False

    @staticmethod
    def _is_meaningless_pattern_relation(relation: str) -> bool:
        for pattern in MEANINGLESS_RELATION_PATTERNS:
            if relation == pattern or relation.startswith(pattern + "."):
                return True
        return False

    @staticmethod
    def _get_common_prefix(rel1: str, rel2: str) -> str:
        parts1 = rel1.split(".")
        parts2 = rel2.split(".")
        common: list[str] = []
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common.append(p1)
            else:
                break
        return ".".join(common) if common else ""

    def _flatten_relation(self, rel1: str, rel2: str) -> str:
        common_prefix = self._get_common_prefix(rel1, rel2)
        if common_prefix:
            rel1_suffix = rel1[len(common_prefix) :].lstrip(".")
            return f"{rel1_suffix}.{rel2}"
        return f"{rel1}.{rel2}"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        for char in ".,!?;:()[]{}\"'/\\":
            text = text.replace(char, " ")
        return [token.lower() for token in text.split() if token.strip()]

    def rank_by_similarity(
        self,
        items: list[dict[str, str]],
        question: str,
        text_field: str = "name",
    ) -> list[dict[str, str]]:
        if not items or not question:
            return items
        from rank_bm25 import BM25Okapi

        query_tokens = self._tokenize(question)
        if not query_tokens:
            return items
        docs_tokens = [
            self._tokenize(item.get(text_field, "") or item.get("entity", "") or item.get("relation", ""))
            for item in items
        ]
        if not any(docs_tokens):
            return items
        bm25 = BM25Okapi(docs_tokens)
        scores = bm25.get_scores(query_tokens)
        scored_items = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items]

    def _resolve_entity(self, entity: str) -> str | None:
        if not entity:
            return None
        if entity.startswith(("m.", "g.", "en.")):
            return entity
        if len(entity) > 100:
            logger.debug("Skipping entity resolution for long string: '%s...'", entity[:50])
            return None
        entity_id = self.resolver.resolve(entity)
        if entity_id:
            return entity_id
        logger.warning("Failed to resolve entity: '%s'", entity)
        return None

    def get_relations(
        self,
        entity: str,
        limit: int = 100,
        question: str = "",
        top_k: int = 10,
        include_flatten: bool = True,
    ) -> list[dict[str, str]]:
        entity_id = self._resolve_entity(entity)
        if not entity_id:
            logger.warning("Cannot resolve entity '%s', returning empty results", entity)
            return []

        try:
            if entity_id in self._rel_cache:
                cached = self._rel_cache[entity_id]
                relations = [{"relation": rid} for rid in cached]
            else:
                rel_ids: set[str] = set()
                query_out = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?relation WHERE {{
                    ns:{entity_id} ?relation ?tail .
                    FILTER(isIRI(?relation))
                    FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
                }} LIMIT {limit}
                """
                started = time.time()
                self.sparql.setQuery(query_out)
                results_out = self.sparql.query().convert()
                if time.time() - started > 0.5:
                    logger.info("SPARQL outgoing relations for %s: %.2fs", entity_id, time.time() - started)
                for binding in results_out.get("results", {}).get("bindings", []):
                    val = binding.get("relation", {}).get("value")
                    if val:
                        rel_ids.add(uri_to_dotted(val))

                query_in = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?relation WHERE {{
                    ?head ?relation ns:{entity_id} .
                    FILTER(isIRI(?relation))
                    FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
                }} LIMIT {limit}
                """
                started = time.time()
                self.sparql.setQuery(query_in)
                results_in = self.sparql.query().convert()
                if time.time() - started > 0.5:
                    logger.info("SPARQL incoming relations for %s: %.2fs", entity_id, time.time() - started)
                for binding in results_in.get("results", {}).get("bindings", []):
                    val = binding.get("relation", {}).get("value")
                    if val:
                        rel_ids.add(uri_to_dotted(val))

                rel_ids_list = list(rel_ids)
                relations = [{"relation": rid} for rid in rel_ids_list]
                self._rel_cache[entity_id] = rel_ids_list

            if include_flatten and entity_id in self._pending_flatten_relations:
                existing = {r.get("relation") for r in relations}
                for flatten_rel in self._pending_flatten_relations[entity_id]:
                    if flatten_rel not in existing:
                        relations.append({"relation": flatten_rel})

            relations = self._apply_relation_whitelist(relations)
            relations = [
                r for r in relations if not self._is_meaningless_pattern_relation(r.get("relation", ""))
            ]
            if question and relations:
                relations = self.rank_by_similarity(relations, question, "relation")
            return relations[:top_k]
        except Exception as exc:
            logger.error("Error in get_relations: %s", exc)
            return []

    def get_triples(
        self,
        entity: str,
        relations: list[str],
        limit_per_relation: int = 20,
        question: str = "",
        return_with_cvt_info: bool = False,
    ) -> Any:
        entity_id = self._resolve_entity(entity)
        if not entity_id:
            if return_with_cvt_info:
                return {"triples": [], "cvt_info": []}
            return []

        entity_name = entity
        query_name = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?name WHERE {{
            ns:{entity_id} ns:type.object.name ?name .
        }} LIMIT 50
        """
        try:
            self.sparql.setQuery(query_name)
            res = self.sparql.query().convert()
            if res:
                bindings = res.get("results", {}).get("bindings", [])
                if bindings:
                    best_name = None
                    for binding in bindings:
                        val = binding.get("name", {}).get("value", "")
                        lang = binding.get("name", {}).get("xml:lang", "")
                        if lang == "en":
                            best_name = val
                            break
                        if not best_name:
                            best_name = val
                    if best_name:
                        entity_name = best_name
        except Exception:
            pass

        seen_relations: list[str] = []
        seen_set: set[str] = set()
        for rel in relations:
            if rel not in seen_set:
                seen_relations.append(rel)
                seen_set.add(rel)
        relations = seen_relations

        all_triples: list[dict[str, str]] = []
        flatten_relations: list[str] = []
        cvt_info: list[dict[str, Any]] = []
        flatten_rel_counter: dict[str, int] = {}
        all_flatten_candidates_out: list[dict[str, str]] = []
        all_flatten_candidates_in: list[dict[str, str]] = []

        for relation in relations:
            if relation in self._cvt_mapping:
                mapping = self._cvt_mapping[relation]
                if len(mapping) == 4:
                    original_rel, cvt_node_id, cvt_rel, direction = mapping
                else:
                    original_rel, cvt_node_id, cvt_rel = mapping
                    direction = "out"
                triples_for_rel = self._query_flatten_relation(
                    entity_id,
                    entity_name,
                    original_rel,
                    cvt_node_id,
                    cvt_rel,
                    relation,
                    limit_per_relation,
                    direction,
                )
                all_triples.extend(triples_for_rel)
                continue

            relation_id = normalize_relation(relation)
            triples_for_rel: list[dict[str, str]] = []
            cvt_nodes_found_out: list[tuple[str, str]] = []
            cvt_nodes_found_in: list[tuple[str, str]] = []

            try:
                query_tail = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?tail (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {{
                    ns:{entity_id} ns:{relation_id} ?tail .
                    OPTIONAL {{ ?tail ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }}
                    OPTIONAL {{ ?tail ns:type.object.name ?name_any . }}
                }} GROUP BY ?tail LIMIT {limit_per_relation * 2}
                """
                self.sparql.setQuery(query_tail)
                results = self.sparql.query().convert()
                for binding in results.get("results", {}).get("bindings", []):
                    tail_uri = binding.get("tail", {}).get("value", "")
                    if not tail_uri:
                        continue
                    tail_id = uri_to_dotted(tail_uri)
                    name = (
                        binding.get("preferred_name", {}).get("value", "")
                        or binding.get("fallback_name", {}).get("value", "")
                    )
                    if self._is_cvt_node(tail_id, name):
                        cvt_nodes_found_out.append((tail_id, relation))
                    else:
                        triples_for_rel.append(
                            {
                                "head": entity_name,
                                "head_id": entity_id,
                                "relation": relation,
                                "tail": name if name else tail_id,
                                "tail_id": tail_id,
                            }
                        )
            except Exception as exc:
                logger.error("Error querying tail triples for %s: %s", relation, exc)

            try:
                query_head = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?head (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {{
                    ?head ns:{relation_id} ns:{entity_id} .
                    OPTIONAL {{ ?head ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }}
                    OPTIONAL {{ ?head ns:type.object.name ?name_any . }}
                }} GROUP BY ?head LIMIT {limit_per_relation * 4}
                """
                self.sparql.setQuery(query_head)
                results = self.sparql.query().convert()
                for binding in results.get("results", {}).get("bindings", []):
                    head_uri = binding.get("head", {}).get("value", "")
                    if not head_uri:
                        continue
                    head_id = uri_to_dotted(head_uri)
                    name = (
                        binding.get("preferred_name", {}).get("value", "")
                        or binding.get("fallback_name", {}).get("value", "")
                    )
                    if self._is_cvt_node(head_id, name):
                        cvt_nodes_found_in.append((head_id, relation))
                    else:
                        triples_for_rel.append(
                            {
                                "head": name if name else head_id,
                                "head_id": head_id,
                                "relation": relation,
                                "tail": entity_name,
                                "tail_id": entity_id,
                            }
                        )
            except Exception as exc:
                logger.error("Error querying head triples for %s: %s", relation, exc)

            for cvt_node_id, original_rel in cvt_nodes_found_out:
                try:
                    all_flatten_candidates_out.extend(
                        self._collect_cvt_flatten_candidates(
                            cvt_node_id,
                            original_rel,
                            direction="out",
                            flatten_rel_counter=flatten_rel_counter,
                        )
                    )
                except Exception as exc:
                    logger.error("Error processing outgoing CVT node %s: %s", cvt_node_id, exc)

            for cvt_node_id, original_rel in cvt_nodes_found_in:
                try:
                    all_flatten_candidates_in.extend(
                        self._collect_cvt_flatten_candidates(
                            cvt_node_id,
                            original_rel,
                            direction="in",
                            flatten_rel_counter=flatten_rel_counter,
                        )
                    )
                except Exception as exc:
                    logger.error("Error processing incoming CVT node %s: %s", cvt_node_id, exc)

            has_cvt_nodes = bool(cvt_nodes_found_out or cvt_nodes_found_in)
            effective_limit = 15 if has_cvt_nodes else 5
            if len(triples_for_rel) > effective_limit:
                triples_for_rel = random.sample(triples_for_rel, effective_limit)
            all_triples.extend(triples_for_rel)

        all_flatten_candidates = all_flatten_candidates_out + all_flatten_candidates_in
        if all_flatten_candidates:
            if question:
                query_text = f"{question} {entity_name}".strip()
                ranked_candidates = self.rank_by_similarity(
                    [{"relation": c["flatten_relation"]} for c in all_flatten_candidates],
                    query_text,
                    "relation",
                )
                top_80_flatten_rels = {r["relation"] for r in ranked_candidates[:80]}
                top_80_candidates = [
                    c for c in all_flatten_candidates if c["flatten_relation"] in top_80_flatten_rels
                ]
            else:
                top_80_candidates = all_flatten_candidates[:80]

            meaningful_candidates: list[dict[str, str]] = []
            for candidate in top_80_candidates:
                flattened_triples = self._query_flatten_relation(
                    entity_id,
                    entity_name,
                    candidate["original_relation"],
                    candidate["cvt_node_id"],
                    candidate["cvt_relation"],
                    candidate["flatten_relation"],
                    limit_per_relation=limit_per_relation,
                    direction=candidate["direction"],
                )
                meaningful_triples = [
                    triple
                    for triple in flattened_triples
                    if triple.get("head_id")
                    and triple.get("tail_id")
                    and triple.get("head_id") != triple.get("tail_id")
                ]
                if meaningful_triples:
                    meaningful_candidates.append(candidate)

            # Always stash candidates so the session layer can run async LLM filter.
            if meaningful_candidates:
                self._pending_flatten_candidates[entity_id] = {
                    "candidates": meaningful_candidates,
                    "question": question or "",
                    "entity_name": entity_name,
                }
            final_candidates = meaningful_candidates[:8]

            flatten_triples_all: list[dict[str, str]] = []
            for candidate in final_candidates:
                flatten_rel = candidate["flatten_relation"]
                flattened_triples = self._query_flatten_relation(
                    entity_id,
                    entity_name,
                    candidate["original_relation"],
                    candidate["cvt_node_id"],
                    candidate["cvt_relation"],
                    flatten_rel,
                    limit_per_relation=limit_per_relation,
                    direction=candidate["direction"],
                )
                meaningful_triples = [
                    triple
                    for triple in flattened_triples
                    if triple.get("head_id")
                    and triple.get("tail_id")
                    and triple.get("head_id") != triple.get("tail_id")
                ]
                if not meaningful_triples:
                    continue
                self._cvt_mapping[flatten_rel] = (
                    candidate["original_relation"],
                    candidate["cvt_node_id"],
                    candidate["cvt_relation"],
                    candidate["direction"],
                )
                flatten_relations.append(flatten_rel)
                flatten_triples_all.extend(meaningful_triples)
                cvt_info.append(
                    {
                        "original_relation": candidate["original_relation"],
                        "cvt_node_id": candidate["cvt_node_id"],
                        "cvt_relation": candidate["cvt_relation"],
                        "flatten_relation": flatten_rel,
                        "direction": candidate["direction"],
                        "flattened_triples": meaningful_triples,
                    }
                )

            logger.info(
                "Selected %s meaningful flatten relations from %s candidates for entity %s",
                len(flatten_relations),
                len(meaningful_candidates),
                entity_id,
            )
            all_triples.extend(flatten_triples_all)

        if flatten_relations:
            if entity_id not in self._pending_flatten_relations:
                self._pending_flatten_relations[entity_id] = []
            existing = set(self._pending_flatten_relations[entity_id])
            for flatten_rel in flatten_relations:
                if flatten_rel not in existing:
                    self._pending_flatten_relations[entity_id].append(flatten_rel)

        if return_with_cvt_info:
            return {"triples": all_triples, "cvt_info": cvt_info}
        return all_triples

    def _collect_cvt_flatten_candidates(
        self,
        cvt_node_id: str,
        original_rel: str,
        *,
        direction: str,
        flatten_rel_counter: dict[str, int],
    ) -> list[dict[str, str]]:
        if direction == "out":
            query_cvt_rels = f"""
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?relation WHERE {{
                ns:{cvt_node_id} ?relation ?tail .
                FILTER(isIRI(?relation))
                FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
            }} LIMIT 50
            """
        else:
            query_cvt_rels = f"""
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?relation WHERE {{
                ?head ?relation ns:{cvt_node_id} .
                FILTER(isIRI(?relation))
                FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
            }} LIMIT 50
            """
        self.sparql.setQuery(query_cvt_rels)
        cvt_results = self.sparql.query().convert()
        cvt_relations_raw: list[str] = []
        for binding in cvt_results.get("results", {}).get("bindings", []):
            cvt_rel_uri = binding.get("relation", {}).get("value", "")
            if not cvt_rel_uri:
                continue
            cvt_rel_id = uri_to_dotted(cvt_rel_uri)
            if self._is_meaningless_pattern_relation(cvt_rel_id):
                continue
            cvt_relations_raw.append(cvt_rel_id)

        if self._relation_whitelist and cvt_relations_raw:
            cvt_relations = [r for r in cvt_relations_raw if r in self._relation_whitelist] or cvt_relations_raw
        else:
            cvt_relations = cvt_relations_raw

        candidates: list[dict[str, str]] = []
        for cvt_rel_id in cvt_relations:
            if direction == "out":
                base_flatten_rel = self._flatten_relation(original_rel, cvt_rel_id)
            else:
                base_flatten_rel = self._flatten_relation(cvt_rel_id, original_rel)
            flatten_rel_counter[base_flatten_rel] = flatten_rel_counter.get(base_flatten_rel, 0) + 1
            flatten_rel = f"{base_flatten_rel}_{flatten_rel_counter[base_flatten_rel]}"
            candidates.append(
                {
                    "flatten_relation": flatten_rel,
                    "original_relation": original_rel,
                    "cvt_node_id": cvt_node_id,
                    "cvt_relation": cvt_rel_id,
                    "direction": direction,
                }
            )
        return candidates

    def _query_flatten_relation(
        self,
        entity_id: str,
        entity_name: str,
        original_rel: str,
        cvt_node_id: str,
        cvt_rel: str,
        flatten_rel: str,
        limit_per_relation: int,
        direction: str = "out",
    ) -> list[dict[str, str]]:
        triples: list[dict[str, str]] = []
        try:
            if direction == "out":
                query = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?final_tail (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {{
                    ns:{entity_id} ns:{normalize_relation(original_rel)} ?cvt .
                    FILTER(?cvt = ns:{cvt_node_id})
                    ?cvt ns:{normalize_relation(cvt_rel)} ?final_tail .
                    OPTIONAL {{ ?final_tail ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }}
                    OPTIONAL {{ ?final_tail ns:type.object.name ?name_any . }}
                }} GROUP BY ?final_tail LIMIT {limit_per_relation * 2}
                """
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                for binding in results.get("results", {}).get("bindings", []):
                    tail_uri = binding.get("final_tail", {}).get("value", "")
                    if not tail_uri:
                        continue
                    tail_id = uri_to_dotted(tail_uri)
                    name = (
                        binding.get("preferred_name", {}).get("value", "")
                        or binding.get("fallback_name", {}).get("value", "")
                    )
                    triples.append(
                        {
                            "head": entity_name,
                            "head_id": entity_id,
                            "relation": flatten_rel,
                            "tail": name if name else tail_id,
                            "tail_id": tail_id,
                        }
                    )
            else:
                query = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?final_head (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {{
                    ?final_head ns:{normalize_relation(cvt_rel)} ?cvt .
                    FILTER(?cvt = ns:{cvt_node_id})
                    ?cvt ns:{normalize_relation(original_rel)} ns:{entity_id} .
                    OPTIONAL {{ ?final_head ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }}
                    OPTIONAL {{ ?final_head ns:type.object.name ?name_any . }}
                }} GROUP BY ?final_head LIMIT {limit_per_relation * 2}
                """
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                for binding in results.get("results", {}).get("bindings", []):
                    head_uri = binding.get("final_head", {}).get("value", "")
                    if not head_uri:
                        continue
                    head_id = uri_to_dotted(head_uri)
                    name = (
                        binding.get("preferred_name", {}).get("value", "")
                        or binding.get("fallback_name", {}).get("value", "")
                    )
                    triples.append(
                        {
                            "head": name if name else head_id,
                            "head_id": head_id,
                            "relation": flatten_rel,
                            "tail": entity_name,
                            "tail_id": entity_id,
                        }
                    )
        except Exception as exc:
            logger.error(
                "Error querying flatten relation %s (direction=%s): %s",
                flatten_rel,
                direction,
                exc,
            )

        if len(triples) > limit_per_relation:
            triples = random.sample(triples, limit_per_relation)
        return triples

    def get_pending_flatten_relations(self, entity_id: str | None = None) -> list[str]:
        if entity_id is not None:
            return self._pending_flatten_relations.get(entity_id, []).copy()
        all_relations: list[str] = []
        for rels in self._pending_flatten_relations.values():
            all_relations.extend(rels)
        return all_relations

    def clear_pending_flatten_relations(self) -> None:
        self._pending_flatten_relations = {}

    def get_pending_flatten_candidates(self, entity_id: str) -> dict[str, Any] | None:
        return self._pending_flatten_candidates.get(entity_id)

    def apply_filtered_flatten_candidates(
        self,
        entity_id: str,
        filtered_candidates: list[dict[str, Any]],
    ) -> None:
        if filtered_candidates:
            filtered_rels = [c["flatten_relation"] for c in filtered_candidates[:8]]
            self._pending_flatten_relations[entity_id] = filtered_rels

    def clear_pending_flatten_candidates(self) -> None:
        self._pending_flatten_candidates = {}

    def format_relations_for_prompt(self, relations: list[dict[str, str]]) -> str:
        if not relations:
            return "No relations found."
        return "\n".join(rel.get("relation", "") for rel in relations)
