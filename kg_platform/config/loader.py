from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import yaml

from kg_platform.config.models import KGProfile


ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")
SECRET_PATTERN = re.compile(r"(?:TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)", re.IGNORECASE)


def _expand(value):
    """Expand only explicit ${UPPER_CASE_VAR:-default} placeholders."""
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"required environment variable is missing: {name}")

    return ENV_PATTERN.sub(replace, value)


def load_profile(path: str | Path) -> KGProfile:
    """Load and strictly validate a profile after controlled env expansion."""
    profile_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(profile_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("KG profile must be a YAML object")
    return KGProfile.model_validate(_expand(data))


def _canonical(profile: KGProfile) -> tuple[dict, str]:
    """Return stable JSON-compatible data and its runtime identity digest."""
    data = profile.model_dump(mode="json", exclude={"resolved_digest"})
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return data, hashlib.sha256(encoded).hexdigest()


def materialize_profile(profile_path: str | Path, run_dir: str | Path) -> tuple[Path, KGProfile]:
    """Freeze an experiment's resolved KG configuration into its run folder.

    Training consumes this artifact rather than the mutable source profile, so
    retries and later investigation can identify the exact backend snapshot and
    policy. Profiles intentionally have no secret-valued fields to persist.
    """
    profile = load_profile(profile_path)
    data, digest = _canonical(profile)
    data["resolved_digest"] = digest
    resolved = KGProfile.model_validate(data)
    destination = Path(run_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "kg_runtime.resolved.yaml"
    output.write_text(yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    (destination / "kg_profile.sha256").write_text(digest + "\n")
    active = resolved.graphs[resolved.active_graph]
    if active.service.index_path:
        manifest = Path(active.service.index_path).expanduser().resolve() / "manifest.json"
        if manifest.is_file():
            (destination / "kg_manifest.json").write_bytes(manifest.read_bytes())
    return output, resolved
