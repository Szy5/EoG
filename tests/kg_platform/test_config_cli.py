from pathlib import Path

import pytest
import yaml

from kg_platform.config.loader import load_profile, materialize_profile


PROFILES = Path("configs/kg/profiles")


@pytest.mark.parametrize(
    "name", ["gtsqa", "gtsqa_local", "wikiwebqsp", "freebase", "freebase_training"]
)
def test_profiles_strict_parse_and_materialize_deterministically(tmp_path: Path, name: str) -> None:
    profile_path = PROFILES / f"{name}.yaml"
    profile = load_profile(profile_path)
    first_path, first = materialize_profile(profile_path, tmp_path / "first")
    second_path, second = materialize_profile(profile_path, tmp_path / "second")
    assert first.resolved_digest == second.resolved_digest
    assert first_path.read_text() == second_path.read_text()
    expected_graph = "freebase" if name.startswith("freebase") else name.removesuffix("_local")
    assert first.active_graph == expected_graph
    index_path = Path(first.graphs[expected_graph].service.index_path or "")
    if expected_graph == "gtsqa" and index_path.joinpath(
        "manifest.json"
    ).exists():
        assert (tmp_path / "first" / "kg_manifest.json").exists()


def test_unknown_profile_key_is_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load((PROFILES / "wikiwebqsp.yaml").read_text())
    data["graphs"]["wikiwebqsp"]["typo"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception, match="typo"):
        load_profile(path)
