from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op

from kg_platform.application import KGApplicationService, KGSessionManager
from kg_platform.config.loader import load_profile
from kg_platform.config.runtime import build_backend, graph_ref, policy
from kg_platform.domain.models import (
    EmbeddedEntity,
    EmbeddedGraph,
    EmbeddedRelation,
    EmbeddedTriple,
    EntityKey,
    EntityRef,
    LiteralRef,
    QueryContext,
)


@dataclass
class Runtime:
    """Process-local backend and application objects shared by KG tools."""
    digest: str
    profile: object
    manager: KGSessionManager
    explorer: KGApplicationService


class RuntimeRegistry:
    """Deduplicate runtimes created from the same materialized profile.

    veRL instantiates get_relations and get_triples separately. Sharing by the
    resolved digest is what lets the second operation consume relation grants
    issued by the first without opening duplicate mmap files or HTTP clients.
    """
    runtimes: ClassVar[dict[str, Runtime]] = {}

    @classmethod
    def get(cls, config_path: str) -> Runtime:
        profile = load_profile(config_path)
        digest = profile.resolved_digest or str(Path(config_path).expanduser().resolve())
        runtime = cls.runtimes.get(digest)
        if runtime is None:
            manager = KGSessionManager(build_backend(profile))
            runtime = Runtime(digest, profile, manager, KGApplicationService(manager, policy(profile)))
            cls.runtimes[digest] = runtime
        return runtime

    @classmethod
    async def aclose(cls) -> None:
        runtimes = list(cls.runtimes.values())
        cls.runtimes.clear()
        for runtime in runtimes:
            await runtime.manager.aclose()


def _entity(graph_id: str, scope_id: str | None, raw: dict) -> EntityRef:
    native_id = str(raw.get("native_id", raw.get("id", ""))).strip()
    label = str(raw.get("label", raw.get("name", ""))).strip()
    return EntityRef(
        EntityKey(graph_id, native_id, scope_id),
        label,
        tuple(str(value) for value in raw.get("types", [])),
        raw.get("description"),
    )


def _embedded(raw: dict | None) -> EmbeddedGraph | None:
    """Decode a validated sample-local graph without answer supervision."""
    if not raw:
        return None
    entities = tuple(
        EmbeddedEntity(
            str(item["native_id"]),
            str(item["label"]),
            tuple(str(value) for value in item.get("types", [])),
            item.get("description"),
        ) for item in raw.get("entities", [])
    )
    relations = tuple(EmbeddedRelation(str(item["native_id"]), str(item["label"])) for item in raw.get("relations", []))
    triples = []
    for item in raw.get("triples", []):
        literal_data = item.get("literal")
        literal = None if literal_data is None else LiteralRef(
            str(literal_data["lexical"]), literal_data.get("datatype"), literal_data.get("language")
        )
        triples.append(
            EmbeddedTriple(
                str(item["head_id"]),
                str(item["relation_id"]),
                str(item["tail_id"]) if item.get("tail_id") is not None else None,
                literal,
            )
        )
    return EmbeddedGraph(entities, relations, tuple(triples))


def build_query_context(instance_id: str, profile, kwargs: dict[str, Any]) -> QueryContext:
    """Adapt dataset kwargs to the immutable application session context."""
    expected = graph_ref(profile)
    raw = kwargs.get("kg_context")
    if not isinstance(raw, dict):
        raw = {}
    graph_data = raw.get("graph") or {}
    if graph_data:
        # Dataset rows and runtime profiles are independently versioned. Refuse
        # accidental cross-snapshot execution instead of returning plausible
        # but irreproducible neighbors.
        actual_graph_id = str(graph_data.get("graph_id", ""))
        actual_snapshot = str(graph_data.get("snapshot_id", ""))
        if actual_graph_id != expected.graph_id or actual_snapshot != expected.snapshot_id:
            raise ValueError("dataset kg_context graph/snapshot does not match resolved runtime")
    sample_id = str(raw.get("sample_id", kwargs.get("sample_id", kwargs.get("qid", instance_id))))
    expected_scope = profile.graphs[profile.active_graph].expected_capabilities.scope
    # Native IDs in embedded subgraphs are only unique inside one sample.
    scope_id = sample_id if expected_scope == "sample" else None
    anchor_data = raw.get("anchors")
    if anchor_data is None:
        anchor_data = kwargs.get("question_entities", [])
    hint_data = raw.get("entity_hints", [])
    anchors = tuple(_entity(expected.graph_id, scope_id, item) for item in anchor_data if isinstance(item, dict))
    hints = tuple(_entity(expected.graph_id, scope_id, item) for item in hint_data if isinstance(item, dict))
    if any(not entity.key.native_id or not entity.label for entity in anchors + hints):
        raise ValueError("anchors and entity_hints require stable native_id and label")
    return QueryContext(
        session_id=instance_id,
        request_id=instance_id,
        graph=expected,
        sample_id=sample_id,
        question=str(raw.get("question", kwargs.get("question", ""))),
        anchors=anchors,
        entity_hints=hints,
        embedded_graph=_embedded(raw.get("embedded_graph")),
        deterministic_seed=int(raw.get("deterministic_seed", 0)),
    )


class KGExplorerTool(BaseTool):
    """veRL lifecycle adapter for the backend-neutral KG application service."""
    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.operation = str(config.get("operation", self.name))
        if self.operation not in {"get_relations", "get_triples"} or self.operation != self.name:
            raise ValueError("KGExplorerTool operation must match tool schema name")
        config_path = config.get("runtime_config_path")
        if not config_path:
            raise ValueError("runtime_config_path is required")
        self.runtime = RuntimeRegistry.get(str(config_path))
        self.instances: set[str] = set()

    async def create(self, instance_id: str | None = None, **kwargs) -> str:
        instance_id = await super().create(instance_id=instance_id, **kwargs)
        context = build_query_context(instance_id, self.runtime.profile, kwargs)
        await self.runtime.explorer.acquire(context)
        self.instances.add(instance_id)
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        # Keep the model-facing payload compatible with the existing text tool;
        # structured counts/errors travel separately as rollout metrics.
        if instance_id not in self.instances:
            return json.dumps({"result": "KG tool instance not initialized."}), 0.0, {"error_code": "SESSION_NOT_FOUND"}
        if self.operation == "get_relations":
            entity = str(parameters.get("entity", "")).strip()
            if not entity:
                return json.dumps({"result": "Error: missing required parameter 'entity'."}), 0.0, {"error_code": "INVALID_ARGUMENT"}
            result = await self.runtime.explorer.list_relations(instance_id, entity)
        else:
            entity = str(parameters.get("entity", "")).strip()
            relations = parameters.get("relations")
            if not entity or not isinstance(relations, list) or not relations:
                return json.dumps({"result": "Error: missing required KG tool parameters."}), 0.0, {"error_code": "INVALID_ARGUMENT"}
            result = await self.runtime.explorer.fetch_triples(instance_id, entity, [str(value).strip() for value in relations])
        metrics = {
            "operation": result.operation,
            "returned_count": result.returned_count,
            "available_count": result.available_count,
            "truncated": result.truncated,
            "error_code": result.error_code,
            "graph_id": result.entity.key.graph_id if result.entity else graph_ref(self.runtime.profile).graph_id,
        }
        return json.dumps({"result": result.text}, ensure_ascii=False), 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id not in self.instances:
            return
        self.instances.remove(instance_id)
        await self.runtime.explorer.release(instance_id)
