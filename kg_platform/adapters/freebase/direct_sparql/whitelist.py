from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

NS = "http://rdf.freebase.com/ns/"

_WHITELIST_DIR = Path(__file__).resolve().parents[1] / "data" / "predicates_white_list"

_WHITELIST_MAP = {
    "cwq": "filtered_cwq_white_list.json",
    "webqsp": "filtered_webqsp_white_list.json",
    "grailqa": "filtered_grailqa_white_list.json",
    "simpleqa": "filtered_cwq_white_list.json",
}


def load_relation_whitelist(dataset_type: str = "cwq", *, full_list: bool = False) -> set[str] | None:
    """Load dotted Freebase predicate ids from package whitelist JSON."""
    if full_list:
        filename = "filtered_full_white_list.json"
    else:
        filename = _WHITELIST_MAP.get(dataset_type.lower())
        if not filename:
            logger.warning("Unknown dataset type '%s', falling back to CWQ whitelist", dataset_type)
            filename = _WHITELIST_MAP["cwq"]

    path = _WHITELIST_DIR / filename
    if not path.is_file():
        logger.warning("Whitelist file not found: %s, predicate filtering disabled", path)
        return None

    with path.open("r", encoding="utf-8") as handle:
        uris = json.load(handle)

    dotted = {u[len(NS) :].replace("/", ".") for u in uris if isinstance(u, str) and u.startswith(NS)}
    logger.info("Loaded %s relations from whitelist: %s", len(dotted), path.name)
    return dotted
