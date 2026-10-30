from pathlib import Path
import importlib.util

import numpy as np
import pytest

from kg_platform.adapters.gtsqa import GTSQAArrayBackend

SCRIPT = Path(__file__).parents[2] / "scripts" / "build_gtsqa_index.py"
SPEC = importlib.util.spec_from_file_location("build_gtsqa_index", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build = MODULE.build


def make_source(path: Path) -> None:
    path.mkdir()
    np.save(path / "node_ids.npy", np.array(["Q1", "Q2", "Q3"], dtype=object))
    np.save(path / "node_labels.npy", np.array(["one", "same", "same"], dtype=object))
    np.save(path / "node_descriptions.npy", np.array(["first", "second", "third"], dtype=object))
    np.save(path / "relation_ids.npy", np.array(["P1", "P2"], dtype=object))
    np.save(path / "relation_labels.npy", np.array(["out", "back"], dtype=object))
    np.save(path / "edge_ids.npy", np.array([[0, 2], [1, 0]], dtype=np.int64))
    np.save(path / "relation_types.npy", np.array([0, 1], dtype=np.int64))
    (path / "RELEASE_v1.txt").write_text("fixture")


def test_gtsqa_builder_packs_text_builds_both_csr_and_protects_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "index"
    make_source(source)
    manifest = build(source, output)
    assert manifest["counts"] == {"nodes": 3, "edges": 2, "relations": 2}
    assert "node_labels.bytes.npy" in manifest["arrays"]
    assert not (output / "node_labels.npy").exists()
    backend = GTSQAArrayBackend(output, verify_checksums=True)
    assert backend.node_index(backend.entity(2).key) == 2
    assert backend.out_offsets.tolist() == [0, 1, 1, 2]
    assert backend.in_offsets.tolist() == [0, 1, 2, 2]
    with pytest.raises(FileExistsError):
        build(source, output)
