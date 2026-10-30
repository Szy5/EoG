from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=False,
        lstrip_blocks=False,
    )


def render_relation_filter(*, question: str, entity_name: str, relations: list[str]) -> str:
    return _env().get_template("relation_filter.j2").render(
        question=question,
        entity_name=entity_name,
        relations=relations,
    )


def render_flatten_relation_filter(*, question: str, entity_name: str, relations: list[str]) -> str:
    return _env().get_template("flatten_relation_filter.j2").render(
        question=question,
        entity_name=entity_name,
        relations=relations,
    )


def render_filter_prompt(
    *,
    question: str,
    entity_name: str,
    relations: list[str],
    use_flatten_prompt: bool = False,
) -> str:
    if use_flatten_prompt:
        return render_flatten_relation_filter(
            question=question,
            entity_name=entity_name,
            relations=relations,
        )
    return render_relation_filter(
        question=question,
        entity_name=entity_name,
        relations=relations,
    )
