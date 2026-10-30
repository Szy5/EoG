from __future__ import annotations

from pathlib import Path

from kg_platform.adapters.global_array import (
    GlobalArrayBackend,
    GlobalArraySession,
    PackedTextColumn,
)


class GTSQAArrayBackend(GlobalArrayBackend):
    """Compatibility wrapper for the original GTSQA array manifest."""

    def __init__(self, index_path: str | Path, *, verify_checksums: bool = False):
        super().__init__(
            index_path,
            verify_checksums=verify_checksums,
            expected_schemas=frozenset({"kg-platform.gtsqa-index"}),
            expected_graph_id="gtsqa",
            backend_name="GTSQA",
        )


GTSQAArraySession = GlobalArraySession

__all__ = ["GTSQAArrayBackend", "GTSQAArraySession", "PackedTextColumn"]
