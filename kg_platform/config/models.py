from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TransportConfig(StrictModel):
    kind: Literal["http", "embedded", "direct"]
    endpoint: str | None = None

    @model_validator(mode="after")
    def endpoint_required_for_http(self):
        if self.kind == "http" and not self.endpoint:
            raise ValueError("http transport requires endpoint")
        return self


class ServiceConfig(StrictModel):
    index_path: str | None = None
    sparql_endpoint: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=18101, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    # Direct-SPARQL / KGQATool Freebase compat (ignored by other adapters)
    dataset_type: str = "cwq"
    full_list: bool = False
    enable_llm_relation_filter: bool = True


class ExpectedCapabilities(StrictModel):
    scope: Literal["global", "sample"]
    reverse_edges_materialized: bool | None = None
    supports_literals: bool | None = None


class GraphConfig(StrictModel):
    snapshot_id: str
    adapter: Literal[
        "global_array",
        "gtsqa_array",
        "embedded_sample_graph",
        "freebase_sparql",
        "freebase_direct_sparql",
    ]
    transport: TransportConfig
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    expected_capabilities: ExpectedCapabilities


class PolicyConfig(StrictModel):
    relation_candidate_limit: int = Field(default=200, ge=1, le=1000)
    max_relations: int = Field(default=20, ge=1, le=200)
    max_requested_relations: int = Field(default=4, ge=1, le=16)
    max_triples_per_relation: int = Field(default=20, ge=1, le=200)
    max_triples_total: int = Field(default=60, ge=1, le=1000)
    # Keep BM25/LLM ranking from Direct-SPARQL instead of alphabetical re-sort.
    preserve_backend_order: bool = False


class ReliabilityConfig(StrictModel):
    request_timeout_seconds: float = Field(default=5.0, gt=0, le=300)
    transient_retries: int = Field(default=2, ge=0, le=10)
    max_in_flight_per_process: int = Field(default=64, ge=1, le=4096)


class KGProfile(StrictModel):
    schema_version: Literal[1]
    profile_name: str
    active_graph: str
    allowed_graphs: list[str]
    graphs: dict[str, GraphConfig]
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    tool_contract_version: int = 1
    resolved_digest: str | None = None

    @model_validator(mode="after")
    def graph_allowlist_is_consistent(self):
        if self.active_graph not in self.graphs:
            raise ValueError("active_graph is not defined in graphs")
        if self.active_graph not in self.allowed_graphs:
            raise ValueError("active_graph is not allowed")
        unknown = set(self.allowed_graphs) - set(self.graphs)
        if unknown:
            raise ValueError(f"allowed_graphs contains undefined graphs: {sorted(unknown)}")
        return self
