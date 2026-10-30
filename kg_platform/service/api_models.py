from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphDTO(StrictModel):
    graph_id: str
    snapshot_id: str
    schema_version: int = 1


class EntityKeyDTO(StrictModel):
    graph_id: str
    native_id: str
    scope_id: str | None = None


class EntityDTO(StrictModel):
    key: EntityKeyDTO
    label: str
    types: list[str] = Field(default_factory=list)
    description: str | None = None
    external_ids: list[list[str]] = Field(default_factory=list)


class RelationDTO(StrictModel):
    graph_id: str
    native_id: str
    label: str
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)


class ChoiceDTO(StrictModel):
    relation: RelationDTO | None = None
    relation_id: str | None = None
    direction: Literal["out", "in"]
    edge_count: int | None = None

    @model_validator(mode="after")
    def require_relation_identity(self):
        if self.relation is None and self.relation_id is None:
            raise ValueError("choice requires relation or relation_id")
        return self


class ResolveRequest(StrictModel):
    request_id: str
    graph: GraphDTO
    query: str


class RelationsRequest(StrictModel):
    request_id: str
    graph: GraphDTO
    entity_id: str
    directions: list[Literal["out", "in"]] = Field(default_factory=lambda: ["out", "in"])
    limit: int = Field(default=200, ge=1, le=1000)


class TriplesRequest(StrictModel):
    request_id: str
    graph: GraphDTO
    entity_id: str
    choices: list[ChoiceDTO] = Field(min_length=1, max_length=16)
    limit_per_choice: int = Field(default=20, ge=1, le=200)
    total_limit: int = Field(default=60, ge=1, le=1000)
