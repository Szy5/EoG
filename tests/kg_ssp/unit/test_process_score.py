import pytest

from quarl.reward.score.process_score import has_broken_format


@pytest.mark.unit
def test_has_broken_format_flags_repeated_malformed_kg_query_tags() -> None:
    response = " ".join("</kg-query|" for _ in range(5))

    assert has_broken_format(response) is True


@pytest.mark.unit
def test_has_broken_format_ignores_short_or_well_formed_kg_query_sequences() -> None:
    short_broken = " ".join("</kg-query|" for _ in range(4))
    well_formed = " ".join("<kg-query>get_relations(\"Barack Obama\")</kg-query>" for _ in range(5))

    assert has_broken_format(short_broken) is False
    assert has_broken_format(well_formed) is False
