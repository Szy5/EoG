import importlib.util

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "required"),
    [
        ("torch", True),
        ("ray", True),
        ("verl", True),
        ("quarl", True),
        ("sglang", False),
    ],
)
def test_core_import_specs_are_available(module_name: str, required: bool) -> None:
    spec = importlib.util.find_spec(module_name)
    if required:
        assert spec is not None, f"Expected import spec for required module '{module_name}'"


@pytest.mark.unit
def test_repo_local_packages_resolve_from_workspace() -> None:
    assert importlib.util.find_spec("quarl") is not None
    assert importlib.util.find_spec("verl") is not None
