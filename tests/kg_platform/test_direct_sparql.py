from __future__ import annotations

from kg_platform.adapters.freebase.direct_sparql.client import DirectSparqlClient, MEANINGLESS_RELATION_PATTERNS
from kg_platform.adapters.freebase.direct_sparql.prompts import (
    render_flatten_relation_filter,
    render_relation_filter,
)
from kg_platform.adapters.freebase.direct_sparql.relation_normalizer import normalize_relation
from kg_platform.adapters.freebase.direct_sparql.whitelist import load_relation_whitelist


def test_normalize_relation_strips_uri_and_slashes() -> None:
    assert (
        normalize_relation("http://rdf.freebase.com/ns/people.person.children")
        == "people.person.children"
    )
    assert normalize_relation("ns:people/person.children") == "people.person.children"


def test_flatten_relation_naming() -> None:
    client = DirectSparqlClient.__new__(DirectSparqlClient)
    assert (
        client._flatten_relation(
            "location.administrative_division.capital",
            "location.administrative_division_capital_relationship.capital",
        )
        == "administrative_division.capital.location.administrative_division_capital_relationship.capital"
    )


def test_meaningless_patterns_are_distinct() -> None:
    assert "freebase.type_hints.enumeration" in MEANINGLESS_RELATION_PATTERNS
    assert "type.object.key" in MEANINGLESS_RELATION_PATTERNS
    assert DirectSparqlClient._is_meaningless_pattern_relation("type.object.type")
    assert not DirectSparqlClient._is_meaningless_pattern_relation(
        "people.person.children"
    )


def test_whitelist_loads_from_package_data() -> None:
    dotted = load_relation_whitelist("cwq", full_list=False)
    assert dotted is not None
    assert len(dotted) > 100
    assert all("." in item for item in list(dotted)[:5])


def test_jinja_relation_filter_render() -> None:
    text = render_relation_filter(
        question="Who founded Microsoft?",
        entity_name="Microsoft",
        relations=["organization.organization.founders", "common.topic.alias"],
    )
    assert "Who founded Microsoft?" in text
    assert 'Topic Entity: ["Microsoft"]' in text
    assert "organization.organization.founders" in text
    assert "Your Selections:" in text


def test_jinja_flatten_filter_render() -> None:
    text = render_flatten_relation_filter(
        question="Where is the capital?",
        entity_name="Down District Council",
        relations=["administrative_division.capital.location.x_1"],
    )
    assert "Flatten Relations" in text or "flatten relations" in text.lower()
    assert "administrative_division.capital.location.x_1" in text
    assert "Your Selections:" in text
