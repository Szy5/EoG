from pathlib import Path

from test_build_car_kg_index import build, make_source

from kg_platform.adapters.global_array import GlobalArrayBackend
from kg_platform.config.loader import load_profile
from kg_platform.config.runtime import build_backend

SNAPSHOT = "car-kg-benchmark-v2+sha256:05b30c36dd5962864a35236d25e6bc51d8c197bb14b31aeec7bab2cf50e9363b"


def test_car_profile_builds_global_array_service_backend(tmp_path: Path, monkeypatch) -> None:
    nodes, relationships = make_source(tmp_path)
    output = tmp_path / "index"
    build(nodes, relationships, output, snapshot_id=SNAPSHOT)
    monkeypatch.setenv("CAR_KG_INDEX_PATH", str(output))

    profile = load_profile("configs/kg/profiles/car_kg.yaml")
    backend = build_backend(profile, service_backend=True)

    assert isinstance(backend, GlobalArrayBackend)
    assert backend.graph.graph_id == "car_kg"
    assert backend.graph.snapshot_id == SNAPSHOT
