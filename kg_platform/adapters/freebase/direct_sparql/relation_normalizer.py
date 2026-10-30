from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

NS = "http://rdf.freebase.com/ns/"


def ensure_endpoint_in_no_proxy(endpoint: str) -> None:
    """Add the endpoint host (and common localhost aliases) to no_proxy."""
    host = endpoint.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].strip()
    extras = ["localhost", "127.0.0.1", "0.0.0.0"]
    targets = [h for h in [host, *extras] if h]
    for key in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(key, "")
        parts = [item.strip() for item in current.split(",") if item.strip()]
        changed = False
        for item in targets:
            if item not in parts:
                parts.append(item)
                changed = True
        if changed:
            os.environ[key] = ",".join(parts)


def uri_to_dotted(uri: str) -> str:
    if uri.startswith(NS):
        return uri[len(NS) :].replace("/", ".")
    if "/ns/" in uri:
        return uri.split("/ns/", 1)[-1].replace("/", ".")
    return uri.replace("/", ".")


class RelationNormalizer:
    """Strip Freebase URI prefixes and normalize separators for SPARQL ids."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def normalize(self, relation: str) -> str:
        if not relation:
            return relation
        key = relation.lower().strip()
        if key in self._cache:
            return self._cache[key]

        cleaned = relation.strip()
        if cleaned.startswith(NS):
            cleaned = cleaned[len(NS) :]
        elif cleaned.startswith("ns:"):
            cleaned = cleaned[len("ns:") :]
        if "/" in cleaned:
            cleaned = cleaned.replace("/", ".")
        cleaned = re.sub(r"\s+", "_", cleaned)
        self._cache[key] = cleaned
        logger.debug("Normalized relation '%s' -> '%s'", relation, cleaned)
        return cleaned

    def clear_cache(self) -> None:
        self._cache.clear()


_normalizer: RelationNormalizer | None = None


def get_normalizer() -> RelationNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = RelationNormalizer()
    return _normalizer


def normalize_relation(relation: str) -> str:
    return get_normalizer().normalize(relation)
