from __future__ import annotations

from kg_platform.domain.models import LiteralRef, TripleRef


def format_relations(aliases: list[str]) -> str:
    return "\n".join(aliases) if aliases else "No relations found."


def _node_label(node: object) -> str:
    if isinstance(node, LiteralRef):
        suffix = f" [literal:{node.datatype}]" if node.datatype else " [literal]"
        return node.lexical + suffix
    return str(getattr(node, "label", ""))


def format_triples(triples: list[tuple[TripleRef, str]]) -> str:
    if not triples:
        return "No triples found."
    return "\n".join(
        f"[{triple.head.label}, {relation_alias}, {_node_label(triple.tail)}]"
        for triple, relation_alias in triples
    )


def format_error(code: str, details: dict | None = None) -> str:
    details = details or {}
    if code == "STALE_RELATION_CHOICE":
        invalid = str(details.get("invalid_alias", "")).strip() or "unknown"
        last_text = str(details.get("last_relations_text", "")).strip()
        if not last_text and isinstance(details.get("last_relations"), list):
            last_text = "\n".join(str(item) for item in details["last_relations"])
        # Match KGQATool._relation_choice_error so Console matches training trajectories.
        return (
            f"The relation '{invalid}' is not in the latest predicate list. "
            "Choose predicates from the list below:\n\n"
            f"{last_text}"
        )
    messages = {
        "ENTITY_NOT_FOUND": "Entity not found in the active graph.",
        "ENTITY_AMBIGUOUS": "Entity is ambiguous. Use one of the qualified aliases returned for this session.",
        "STALE_RELATION_CHOICE": "Invalid relation choice. Call get_relations for this entity and copy a returned relation exactly.",
        "RELATION_NOT_APPLICABLE": "Invalid relation choice for this entity.",
        "BACKEND_UNAVAILABLE": "The knowledge graph is temporarily unavailable.",
        "DEADLINE_EXCEEDED": "The knowledge graph request timed out.",
    }
    return messages.get(code, "The knowledge graph request failed.")
