import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional


PROMPT_PAYLOAD_MARKER = "## The input I provided:\n"
_JSON_ESCAPE_CHARS = {'"', "\\", "/", "b", "f", "n", "r", "t"}

# Heuristic node taxonomy derived from Process_Reward_KG_Node_Audit.md.
# The goal for process_reward_v4 is to keep concrete named entities and filter
# out relation-like bridge nodes plus abstract concept nodes.
_LANGUAGE_TITLES = {"english", "german", "french", "latin script"}
_RELATION_TITLE_EXACT = {
    "male",
    "female",
    "researcher",
    "actor",
    "politician",
    "scientist",
    "artist",
    "journalist",
    "interviewer",
    "botanist",
    "biologist",
    "mycologist",
    "taxonomist",
    "trauma surgeon",
    "film director",
    "scholarly article",
    "scientific article",
    "review article",
    "article",
    "journal",
    "filing",
    "reverse strand",
    "forward strand",
    "public domain",
    "copyrighted",
    "family name",
    "male given name",
    "nazi concentration camp commandant",
    "national register of historic places listed place",
    "jinshi",
    "juren",
    "shengyuan",
    "clinical trial",
    "cell line",
    "wikipedia:vital articles/level/4",
    "wikimedia category",
    "organization",
    "writer",
    "painter",
    "listed building",
    "olympic sport",
    "model organism",
    "taxon",
    "holocaust victim",
    "catholic priest",
    "architectural heritage monument in bavaria",
    "protein-coding gene",
}
_RELATION_TITLE_SUBSTRINGS = (
    " listed place",
    " concentration camp commandant",
    "division i",
    "given name",
    "family name",
    " strand",
    " category",
    "interpro family",
    "interpro domain",
    " olympic ",
    " monument",
    " heritage",
    " victim",
    " priest",
    " taxon",
    " model organism",
)
_RELATION_TEXT_PREFIXES = (
    "person who ",
    "person who engages ",
    "person who acts ",
    "person who practices ",
    "person who controls ",
    "scientist studying ",
    "chief commanding position ",
    "person who use scientific methods",
    "person working in ",
    "biologist who studies ",
    "article in an academic publication",
    "scientific article",
    "journal",
    "delivery of a document to a court",
    'to be used in "sex or gender"',
    "highest degree in china",
    "in the chinese imperial triennial",
    "building, structure, site, object or district",
)
_ABSTRACT_TITLE_EXACT = {
    "english",
    "german",
    "human",
    "association football",
    "basketball",
    "chromosome",
    "protein",
    "gene",
    "telomere",
    "mycology",
    "taxonomy",
    "psychotherapy",
    "group psychotherapy",
    "oncology",
    "accident",
    "injury",
    "misadventure",
    "viviparity",
    "oviparity",
    "light novel",
    "visual novel",
    "interactive fiction",
    "conversation",
    "dialogue",
    "literary genre",
    "biological process",
    "kangaroo",
    "testicle",
    "ovary",
    "medicine",
    "research",
    "gothic architecture",
    "romanesque architecture",
    "irish mythology",
    "celtic mythology",
    "chinese characters",
    "dna spiral",
    "atp binding",
    "alzheimer's disease",
    "parkinson's disease",
    "homo sapiens",
}
_ABSTRACT_TEXT_PREFIXES = (
    "branch of ",
    "study of ",
    "science that ",
    "science of ",
    "quality of ",
    "reproduction via ",
    "process specifically pertinent",
    "the process whose ",
    "team sport ",
    "sport that ",
    "west germanic language",
    "writing system ",
    "biomolecule ",
    "dna molecule ",
    "basic physical and functional unit of heredity",
    "physiological wound ",
    "surgical specialty ",
    "clinically applied psychology",
    "branch of medicine",
    "legal state of a work",
    "internal organ used in the male reproductive system",
    "ovum-producing reproductive organ",
    "common name for some marsupials",
    "any single member of homo sapiens",
    "terminal region of a linear chromosome",
)
_SPECIFIC_TITLE_EXCEPTIONS = {
    "English",
    "German",
    "Latin script",
    "male",
    "female",
    "human",
    "association football",
    "basketball",
    "researcher",
    "actor",
    "politician",
    "scientist",
    "artist",
    "chromosome",
    "protein",
    "gene",
    "public domain",
    "copyrighted",
    "review article",
    "scholarly article",
    "scientific article",
    "article",
    "journal",
    "reverse strand",
    "forward strand",
    "National Register of Historic Places listed place",
    "NCAA Division I men's basketball",
    "NCAA Division I men's basketball tournament",
    "Olympic sport",
    "listed building",
    "model organism",
    "taxon",
    "Catholic priest",
    "Holocaust victim",
    "Chinese characters",
}
_VALID_NODE_FILTER_MODES = {"none", "named_entity_only"}


def _extract_balanced_json_block(text: str, marker: str = PROMPT_PAYLOAD_MARKER) -> Optional[str]:
    if not isinstance(text, str):
        return None

    marker_pos = text.find(marker)
    if marker_pos < 0:
        return None

    start = text.find("{", marker_pos + len(marker))
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _sanitize_json_string_content(text: str) -> str:
    if not isinstance(text, str):
        return text

    sanitized_chars: List[str] = []
    in_string = False
    idx = 0

    while idx < len(text):
        char = text[idx]

        if not in_string:
            sanitized_chars.append(char)
            if char == '"':
                in_string = True
            idx += 1
            continue

        if char == '"':
            sanitized_chars.append(char)
            in_string = False
            idx += 1
            continue

        if char == "\\":
            if idx + 1 >= len(text):
                sanitized_chars.append("\\\\")
                idx += 1
                continue

            next_char = text[idx + 1]
            if next_char in _JSON_ESCAPE_CHARS:
                sanitized_chars.append(char)
                sanitized_chars.append(next_char)
                idx += 2
                continue

            if next_char == "u" and idx + 5 < len(text):
                hex_digits = text[idx + 2 : idx + 6]
                if all(digit in "0123456789abcdefABCDEF" for digit in hex_digits):
                    sanitized_chars.append(text[idx : idx + 6])
                    idx += 6
                    continue

            sanitized_chars.append("\\\\")
            idx += 1
            continue

        if char == "\n":
            sanitized_chars.append("\\n")
        elif char == "\r":
            sanitized_chars.append("\\r")
        elif char == "\t":
            sanitized_chars.append("\\t")
        elif ord(char) < 0x20:
            sanitized_chars.append(f"\\u{ord(char):04x}")
        else:
            sanitized_chars.append(char)
        idx += 1

    return "".join(sanitized_chars)


def _loads_relaxed_json_dict(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        sanitized_text = _sanitize_json_string_content(text)
        try:
            payload = json.loads(sanitized_text)
        except json.JSONDecodeError:
            return {}

    return payload if isinstance(payload, dict) else {}


def extract_subgraph_payload_from_prompt(prompt_text: str) -> Dict[str, Any]:
    payload_text = _extract_balanced_json_block(prompt_text)
    if payload_text is None:
        return {}

    return _loads_relaxed_json_dict(payload_text)


def split_path_nodes(path_text: Any) -> List[str]:
    if not isinstance(path_text, str):
        return []

    return [node.strip() for node in path_text.split("->") if isinstance(node, str) and node.strip()]


def _normalize_node_filter_mode(node_filter_mode: Any) -> str:
    if not isinstance(node_filter_mode, str):
        return "none"

    normalized = node_filter_mode.strip().lower()
    return normalized if normalized in _VALID_NODE_FILTER_MODES else "none"


def _first_line_text(node_text: Any) -> str:
    if not isinstance(node_text, str):
        return ""
    return node_text.split("\n", 1)[0].strip().lower()


def _build_title_to_node_map(path_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes = path_obj.get("nodes", []) if isinstance(path_obj, dict) else []
    title_to_node: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        title = node.get("title")
        if not isinstance(title, str):
            continue
        normalized_title = title.strip()
        if normalized_title and normalized_title not in title_to_node:
            title_to_node[normalized_title] = node
    return title_to_node


def _looks_like_specific_entity_title(title: str) -> bool:
    normalized_title = title.strip()
    if not normalized_title or normalized_title in _SPECIFIC_TITLE_EXCEPTIONS:
        return False

    normalized_lower = normalized_title.lower()
    if normalized_title.startswith('"') or ":" in normalized_title:
        return True
    if any(char.isdigit() for char in normalized_title):
        return True

    words = re.split(r"\s+", normalized_title)
    capitalized_words = 0
    for word in words:
        core = word.strip('"\'()[],.-')
        if not core:
            continue
        if core[:1].isupper() or core.isupper():
            capitalized_words += 1

    if capitalized_words >= 2:
        return True
    if capitalized_words >= 1 and len(words) == 1 and len(normalized_title) > 2 and normalized_lower not in _LANGUAGE_TITLES:
        return True
    return False


def classify_process_reward_node(title: str, node_payload: Optional[Dict[str, Any]] = None) -> str:
    normalized_title = title.strip()
    title_lower = normalized_title.lower()
    node_payload = node_payload if isinstance(node_payload, dict) else {}
    answer_type = node_payload.get("answer_type")
    text_first_line = _first_line_text(node_payload.get("text"))
    specific_title = _looks_like_specific_entity_title(normalized_title)

    if title_lower in _RELATION_TITLE_EXACT or any(fragment in title_lower for fragment in _RELATION_TITLE_SUBSTRINGS):
        return "relation_like"
    if (not specific_title) and any(text_first_line.startswith(prefix) for prefix in _RELATION_TEXT_PREFIXES):
        return "relation_like"

    if title_lower in _ABSTRACT_TITLE_EXACT:
        return "abstract_concept"
    if (not specific_title) and any(text_first_line.startswith(prefix) for prefix in _ABSTRACT_TEXT_PREFIXES):
        return "abstract_concept"

    if answer_type in {"location", "organization"} and title_lower not in _RELATION_TITLE_EXACT:
        return "named_entity"
    if specific_title:
        return "named_entity"
    if answer_type == "person" and title_lower not in _RELATION_TITLE_EXACT:
        return "named_entity"
    if answer_type == "date" and title_lower not in _RELATION_TITLE_EXACT:
        return "named_entity"

    if normalized_title[:1].islower() or title_lower in _LANGUAGE_TITLES:
        return "abstract_concept"
    return "unclear"


def _categorize_precursor_nodes(
    precursor_nodes: List[str],
    title_to_node: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    categorized_nodes: List[Dict[str, str]] = []
    for title in precursor_nodes:
        category = classify_process_reward_node(title, title_to_node.get(title, {}))
        categorized_nodes.append({"title": title, "category": category})
    return categorized_nodes


def build_process_reward_metadata(prompt_text: str, node_filter_mode: str = "none") -> Dict[str, Any]:
    payload = extract_subgraph_payload_from_prompt(prompt_text)
    path_obj = payload.get("path", {}) if isinstance(payload, dict) else {}
    path_text = path_obj.get("path") if isinstance(path_obj, dict) else None

    path_nodes = split_path_nodes(path_text)
    answer_node = path_nodes[-1] if path_nodes else None
    raw_precursor_nodes = path_nodes[:-1] if len(path_nodes) > 1 else []

    normalized_filter_mode = _normalize_node_filter_mode(node_filter_mode)
    title_to_node = _build_title_to_node_map(path_obj)
    precursor_node_categories = _categorize_precursor_nodes(raw_precursor_nodes, title_to_node)
    named_entity_precursor_nodes = [
        item["title"] for item in precursor_node_categories if item["category"] == "named_entity"
    ]
    filtered_out_precursor_nodes = [
        item["title"] for item in precursor_node_categories if item["category"] != "named_entity"
    ]
    precursor_nodes = (
        named_entity_precursor_nodes if normalized_filter_mode == "named_entity_only" else raw_precursor_nodes
    )

    return {
        "path_text": path_text if isinstance(path_text, str) else "",
        "path_nodes": path_nodes,
        "answer_node": answer_node,
        "raw_precursor_nodes": raw_precursor_nodes,
        "named_entity_precursor_nodes": named_entity_precursor_nodes,
        "filtered_out_precursor_nodes": filtered_out_precursor_nodes,
        "precursor_node_categories": precursor_node_categories,
        "process_reward_node_filter_mode": normalized_filter_mode,
        "precursor_nodes": precursor_nodes,
    }


def enrich_extra_info_with_process_reward(
    extra_info: Optional[Dict[str, Any]],
    prompt_text: str,
    node_filter_mode: str = "none",
) -> Dict[str, Any]:
    enriched = deepcopy(extra_info) if isinstance(extra_info, dict) else {}
    process_reward_info = build_process_reward_metadata(prompt_text, node_filter_mode=node_filter_mode)

    for key, value in process_reward_info.items():
        if key not in enriched or not enriched[key]:
            enriched[key] = value

    return enriched
