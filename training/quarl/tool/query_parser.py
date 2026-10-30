"""Parse the two KG exploration calls emitted during rollout."""

import re
from typing import Any, Dict, List, Optional, Tuple


def parse_kg_query_call(raw_query: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    query = raw_query.strip()
    get_relations_match = re.fullmatch(r'get_relations\s*\(\s*"([^"]+)"\s*\)', query)
    if get_relations_match:
        return "get_relations", {"entity": get_relations_match.group(1).strip()}

    get_triples_match = re.fullmatch(r'get_triples\s*\(\s*"([^"]+)"\s*,\s*\[(.*)\]\s*\)', query, re.DOTALL)
    if get_triples_match:
        entity = get_triples_match.group(1).strip()
        raw_relations = get_triples_match.group(2).strip()
        relations: List[str] = []
        if raw_relations:
            for relation_part in raw_relations.split(","):
                relation_part = relation_part.strip()
                if relation_part.startswith('"') and relation_part.endswith('"'):
                    normalized = relation_part[1:-1].strip()
                    if normalized:
                        relations.append(normalized)
        return "get_triples", {"entity": entity, "relations": relations}

    return None
