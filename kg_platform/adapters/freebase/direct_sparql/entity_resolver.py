from __future__ import annotations

import logging
import os
import time
from typing import Any

from kg_platform.adapters.freebase.direct_sparql.relation_normalizer import ensure_endpoint_in_no_proxy, uri_to_dotted

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolve entity names to Freebase IDs via SPARQL label search."""

    def __init__(
        self,
        sparql_endpoint: str,
        timeout: int = 10,
        cache: dict[str, str | None] | None = None,
    ):
        from SPARQLWrapper import JSON, SPARQLWrapper

        self.endpoint = sparql_endpoint
        ensure_endpoint_in_no_proxy(sparql_endpoint)
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        self.cache: dict[str, str | None] = cache if cache is not None else {}

    def resolve(self, entity_name: str, limit: int = 10) -> str | None:
        cache_key = entity_name.lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>

        SELECT DISTINCT ?entity ?type WHERE {{
          ?entity ns:type.object.name "{entity_name}"@en .
          ?entity ns:type.object.type ?type .
        }}
        LIMIT {limit * 10}
        """
        try:
            started = time.time()
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            elapsed = time.time() - started
            if elapsed > 1.0:
                logger.info("Entity resolution SPARQL for '%s': %.2fs", entity_name, elapsed)

            bindings = results.get("results", {}).get("bindings", []) if results else []
            best = self._pick_best(bindings)
            if best:
                self.cache[cache_key] = best
                return best

            query_alt = f"""
            PREFIX ns: <http://rdf.freebase.com/ns/>

            SELECT DISTINCT ?entity ?name ?type WHERE {{
              ?entity ns:type.object.name ?name .
              ?entity ns:type.object.type ?type .
              FILTER (LCASE(STR(?name)) = "{entity_name.lower()}")
            }}
            LIMIT {limit * 5}
            """
            started = time.time()
            self.sparql.setQuery(query_alt)
            results = self.sparql.query().convert()
            elapsed = time.time() - started
            if elapsed > 1.0:
                logger.info("Entity resolution fallback SPARQL for '%s': %.2fs", entity_name, elapsed)
            bindings = results.get("results", {}).get("bindings", []) if results else []
            best = self._pick_best(bindings)
            self.cache[cache_key] = best
            return best
        except Exception as exc:
            logger.error("Entity resolution failed for '%s': %s", entity_name, exc)
            self.cache[cache_key] = None
            return None

    @staticmethod
    def _pick_best(bindings: list[dict[str, Any]]) -> str | None:
        if not bindings:
            return None
        entity_types: dict[str, list[str]] = {}
        for binding in bindings:
            eid = uri_to_dotted(binding.get("entity", {}).get("value", ""))
            etype = uri_to_dotted(binding.get("type", {}).get("value", ""))
            if not eid:
                continue
            entity_types.setdefault(eid, []).append(etype)
        if not entity_types:
            return None
        return max(entity_types.items(), key=lambda kv: len(kv[1]))[0]
