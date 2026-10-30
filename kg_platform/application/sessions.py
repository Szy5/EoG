from __future__ import annotations

import asyncio
import hashlib
import json
import unicodedata
from dataclasses import dataclass, field

from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import EntityKey, EntityRef, QueryContext, RelationChoice
from kg_platform.domain.protocols import GraphBackend, GraphSession


def normalize_alias(value: str) -> str:
    """Normalize model-visible aliases without changing their displayed spelling."""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def context_digest(context: QueryContext) -> str:
    """Bind a session ID to one immutable graph/sample context.

    A rollout may construct the two KG tools independently.  Reusing the same
    session ID is safe only when both tools received byte-equivalent context;
    otherwise sample-local graph data could leak between rollouts.
    """
    payload = {
        "session_id": context.session_id,
        "graph": [context.graph.graph_id, context.graph.snapshot_id, context.graph.schema_version],
        "sample_id": context.sample_id,
        "anchors": [(e.key.native_id, e.label, e.key.scope_id) for e in context.anchors],
        "hints": [(e.key.native_id, e.label, e.key.scope_id) for e in context.entity_hints],
        "embedded": repr(context.embedded_graph),
        "seed": context.deterministic_seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class RelationGrant:
    """A short-lived authorization to turn a displayed alias back into edges."""
    focus: EntityKey
    generation: int
    choices: tuple[RelationChoice, ...]


@dataclass
class ManagedSession:
    """Application state shared by all KG tool operations in one rollout."""
    context: QueryContext
    digest: str
    backend_session: GraphSession
    ref_count: int = 1
    entities: dict[str, list[EntityRef]] = field(default_factory=dict)
    entity_aliases_by_key: dict[EntityKey, str] = field(default_factory=dict)
    relation_grants: dict[tuple[EntityKey, str], RelationGrant] = field(default_factory=dict)
    generations: dict[EntityKey, int] = field(default_factory=dict)

    def register_entity(self, entity: EntityRef) -> str:
        """Register an entity and return an unambiguous model-visible alias."""
        base = entity.label.strip() or entity.key.native_id
        normalized = normalize_alias(base)
        existing = self.entities.setdefault(normalized, [])
        if entity not in existing:
            existing.append(entity)
            existing.sort(key=lambda item: (item.key.native_id, item.key.scope_id or ""))
        if len(existing) == 1:
            alias = base
        else:
            # Labels are not identifiers.  Once a collision appears, qualify
            # every candidate so later tool calls cannot silently pick one.
            for candidate in existing:
                qualifier = candidate.description or (", ".join(candidate.types) if candidate.types else "")
                qualifier = qualifier or f"{candidate.key.graph_id}:{candidate.key.native_id}"
                qualified = f"{candidate.label} [{qualifier}]"
                self.entity_aliases_by_key[candidate.key] = qualified
                self.entities[normalize_alias(qualified)] = [candidate]
            alias = self.entity_aliases_by_key[entity.key]
        self.entities.setdefault(normalize_alias(alias), [entity])
        self.entity_aliases_by_key[entity.key] = alias
        return alias

    def resolve_registered_entity(self, raw: str) -> EntityRef | None:
        candidates = self.entities.get(normalize_alias(raw), [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise KGPlatformError(
                ErrorCode.ENTITY_AMBIGUOUS,
                "Entity alias is ambiguous in this session.",
                details={"candidate_count": len(candidates)},
            )
        return None

    def grant_relations(self, focus: EntityKey, choices: list[RelationChoice]) -> list[str]:
        """Publish relation aliases scoped to this entity and query generation."""
        generation = self.generations.get(focus, 0) + 1
        self.generations[focus] = generation
        # A new get_relations call revokes the prior menu for this entity.  This
        # prevents a model from replaying an alias whose meaning has changed.
        self.relation_grants = {key: grant for key, grant in self.relation_grants.items() if grant.focus != focus}
        grouped: dict[str, list[RelationChoice]] = {}
        order: list[str] = []
        for choice in choices:
            base = choice.relation.label.strip() or choice.relation.key.native_id
            if base not in grouped:
                order.append(base)
            grouped.setdefault(base, []).append(choice)
        aliases: list[str] = []
        used: set[str] = set()
        # Preserve backend/BM25 order: do not alphabetical-sort menu labels.
        for base in order:
            group = grouped[base]
            relation_ids = {choice.relation.key.native_id for choice in group}
            if len(relation_ids) == 1:
                alias = base
                self.relation_grants[(focus, normalize_alias(alias))] = RelationGrant(focus, generation, tuple(group))
                aliases.append(alias)
                continue
            for choice in sorted(group, key=lambda item: (item.relation.key.native_id, item.direction.value)):
                qualifier = " → ".join(choice.relation.source_types + choice.relation.target_types)
                qualifier = qualifier or choice.relation.key.native_id
                alias = f"{base} [{qualifier}]"
                while normalize_alias(alias) in used:
                    alias = f"{alias} [{choice.direction.value}]"
                used.add(normalize_alias(alias))
                self.relation_grants[(focus, normalize_alias(alias))] = RelationGrant(focus, generation, (choice,))
                aliases.append(alias)
        return aliases

    def consume_relations(self, focus: EntityKey, aliases: list[str], max_requested: int) -> list[RelationChoice]:
        """Resolve only aliases granted by the latest relation query for focus."""
        if not aliases or len(aliases) > max_requested:
            raise KGPlatformError(ErrorCode.INVALID_ARGUMENT, "Invalid number of requested relations.")
        choices: list[RelationChoice] = []
        seen: set[tuple[str, str]] = set()
        current_generation = self.generations.get(focus)
        last_relation_lines: list[str] = []
        seen_alias: set[str] = set()
        for key, grant in self.relation_grants.items():
            if grant.focus != focus or grant.generation != current_generation:
                continue
            # key is (EntityKey, normalized_alias); recover a display label from choices.
            for choice in grant.choices:
                label = choice.relation.label.strip() or choice.relation.key.native_id
                norm = normalize_alias(label)
                if norm not in seen_alias:
                    seen_alias.add(norm)
                    last_relation_lines.append(label)
                break
        for alias in aliases:
            grant = self.relation_grants.get((focus, normalize_alias(alias)))
            if grant is None or grant.focus != focus or grant.generation != current_generation:
                raise KGPlatformError(
                    ErrorCode.STALE_RELATION_CHOICE,
                    "Relation alias is not current for this entity.",
                    details={
                        "invalid_alias": alias,
                        "last_relations": last_relation_lines,
                        "last_relations_text": "\n".join(last_relation_lines),
                    },
                )
            for choice in grant.choices:
                identity = (choice.relation.key.native_id, choice.direction.value)
                if identity not in seen:
                    seen.add(identity)
                    choices.append(choice)
        return choices


class KGSessionManager:
    """Reference-counted owner of backend sessions.

    veRL creates one tool object per operation, so two create/release pairs can
    refer to the same logical rollout.  Reference counting closes the backend
    only after the last operation releases it.
    """
    def __init__(self, backend: GraphBackend):
        self.backend = backend
        self._sessions: dict[str, ManagedSession] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def acquire(self, context: QueryContext) -> ManagedSession:
        digest = context_digest(context)
        async with self._lock:
            existing = self._sessions.get(context.session_id)
            if existing:
                # Never let an instance ID become an implicit cache key across
                # different samples or snapshots.
                if existing.digest != digest:
                    raise KGPlatformError(ErrorCode.SESSION_CONTEXT_MISMATCH, "Session context changed.")
                existing.ref_count += 1
                return existing
            backend_session = await self.backend.open_session(context)
            managed = ManagedSession(context=context, digest=digest, backend_session=backend_session)
            for entity in context.anchors + context.entity_hints:
                managed.register_entity(entity)
            self._sessions[context.session_id] = managed
            return managed

    def get(self, session_id: str) -> ManagedSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KGPlatformError(ErrorCode.SESSION_NOT_FOUND, "Session does not exist.") from exc

    async def release(self, session_id: str) -> None:
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return
            managed.ref_count -= 1
            if managed.ref_count > 0:
                return
            self._sessions.pop(session_id, None)
        await managed.backend_session.aclose()

    async def aclose(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for managed in sessions:
            await managed.backend_session.aclose()
        await self.backend.aclose()
