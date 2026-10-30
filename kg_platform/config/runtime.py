from __future__ import annotations

from kg_platform.adapters.freebase import FreebaseDirectSparqlBackend, FreebaseSparqlBackend
from kg_platform.adapters.gtsqa import GTSQAArrayBackend
from kg_platform.adapters.global_array import GlobalArrayBackend
from kg_platform.adapters.http_client import RemoteKGBackend
from kg_platform.adapters.sample_graph import EmbeddedSampleGraphBackend
from kg_platform.config.models import KGProfile
from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import GraphRef, QueryPolicy


def graph_ref(profile: KGProfile) -> GraphRef:
    graph = profile.graphs[profile.active_graph]
    return GraphRef(profile.active_graph, graph.snapshot_id)


def policy(profile: KGProfile) -> QueryPolicy:
    return QueryPolicy(**profile.policy.model_dump())


def build_backend(profile: KGProfile, *, service_backend: bool = False):
    """Build the correct side of the configured transport boundary.

    A training worker uses ``RemoteKGBackend`` for HTTP profiles.  The service
    process passes ``service_backend=True`` to open the actual mmap/SPARQL
    adapter and perform stronger local snapshot verification.
    """
    graph_id = profile.active_graph
    config = profile.graphs[graph_id]
    expected = graph_ref(profile)
    if not service_backend and config.transport.kind == "http":
        return RemoteKGBackend(
            config.transport.endpoint or "",
            expected,
            timeout=profile.reliability.request_timeout_seconds,
            retries=profile.reliability.transient_retries,
        )
    if config.adapter == "global_array":
        backend = GlobalArrayBackend(config.service.index_path or "", verify_checksums=service_backend)
    elif config.adapter == "gtsqa_array":
        backend = GTSQAArrayBackend(config.service.index_path or "", verify_checksums=service_backend)
    elif config.adapter == "embedded_sample_graph":
        backend = EmbeddedSampleGraphBackend(expected)
    elif config.adapter == "freebase_sparql":
        backend = FreebaseSparqlBackend(
            config.service.sparql_endpoint or config.transport.endpoint or "",
            snapshot_id=config.snapshot_id,
            timeout=profile.reliability.request_timeout_seconds,
        )
    elif config.adapter == "freebase_direct_sparql":
        backend = FreebaseDirectSparqlBackend(
            config.service.sparql_endpoint or config.transport.endpoint or "",
            snapshot_id=config.snapshot_id,
            timeout=profile.reliability.request_timeout_seconds,
            dataset_type=config.service.dataset_type,
            full_list=config.service.full_list,
            enable_llm_relation_filter=config.service.enable_llm_relation_filter,
            max_relations=profile.policy.max_relations,
            triples_limit_per_relation=min(5, profile.policy.max_triples_per_relation),
        )
    else:
        raise KGPlatformError(ErrorCode.INVALID_ARGUMENT, f"Unsupported adapter: {config.adapter}")
    if backend.graph != expected:
        # Never "helpfully" accept the index discovered on disk: the resolved
        # profile is the experiment contract and must name that exact snapshot.
        raise KGPlatformError(ErrorCode.SNAPSHOT_MISMATCH, "Profile snapshot does not match backend manifest.")
    return backend
