"""Process-local skill-evolution store for tests and single-process use."""

from __future__ import annotations

import asyncio

from deerflow.skill_evolution.models import (
    ClusterStatus,
    EvolutionCluster,
    EvolutionEvent,
    EvolutionEventKind,
    ProposalStatus,
    SkillEvaluation,
    SkillProposal,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
    EvolutionStoreNotFound,
    PutResult,
    SkillEvolutionStore,
    validate_proposal_transition,
)


class InMemorySkillEvolutionStore(SkillEvolutionStore):
    """Lock-protected in-memory implementation of the store contract."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._events: dict[tuple[str, str], EvolutionEvent] = {}
        self._event_idempotency: dict[tuple[str, str, str], str] = {}
        self._clusters: dict[tuple[str, str], EvolutionCluster] = {}
        self._proposals: dict[tuple[str, str], SkillProposal] = {}
        self._evaluations: dict[tuple[str, str], SkillEvaluation] = {}

    async def upsert_event(
        self,
        event: EvolutionEvent,
    ) -> PutResult[EvolutionEvent]:
        event_key = (event.user_id, event.event_id)
        idempotency_key = (
            event.user_id,
            event.run_id,
            event.extractor_version,
        )
        async with self._lock:
            existing_event_id = self._event_idempotency.get(idempotency_key)
            if existing_event_id is not None:
                existing = self._events[(event.user_id, existing_event_id)]
                if existing == event:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("event idempotency key already contains a different payload")

            existing = self._events.get(event_key)
            if existing is not None:
                if existing == event:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("event ID already contains a different payload")

            self._events[event_key] = event
            self._event_idempotency[idempotency_key] = event.event_id
            return PutResult(event, created=True)

    async def get_event(
        self,
        user_id: str,
        event_id: str,
    ) -> EvolutionEvent | None:
        async with self._lock:
            return self._events.get((user_id, event_id))

    async def list_events(
        self,
        user_id: str,
        *,
        event_kind: EvolutionEventKind | None = None,
    ) -> list[EvolutionEvent]:
        async with self._lock:
            events = [event for (owner, _), event in self._events.items() if owner == user_id and (event_kind is None or event.event_kind is event_kind)]
        return sorted(events, key=lambda item: (item.created_at, item.event_id))

    async def put_cluster(
        self,
        cluster: EvolutionCluster,
    ) -> PutResult[EvolutionCluster]:
        key = (cluster.user_id, cluster.cluster_id)
        async with self._lock:
            existing = self._clusters.get(key)
            if existing is not None:
                if existing == cluster:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("cluster ID already contains a different payload")
            self._clusters[key] = cluster
            return PutResult(cluster, created=True)

    async def get_cluster(
        self,
        user_id: str,
        cluster_id: str,
    ) -> EvolutionCluster | None:
        async with self._lock:
            return self._clusters.get((user_id, cluster_id))

    async def list_ready_clusters(
        self,
        user_id: str,
        *,
        min_distinct_runs: int,
    ) -> list[EvolutionCluster]:
        if min_distinct_runs < 1:
            raise ValueError("min_distinct_runs must be at least 1")
        async with self._lock:
            clusters = [cluster for (owner, _), cluster in self._clusters.items() if owner == user_id and cluster.status is ClusterStatus.ready and cluster.independent_run_count >= min_distinct_runs]
        return sorted(
            clusters,
            key=lambda item: (item.updated_at, item.cluster_id),
        )

    async def put_proposal(
        self,
        proposal: SkillProposal,
    ) -> PutResult[SkillProposal]:
        key = (proposal.user_id, proposal.proposal_id)
        async with self._lock:
            existing = self._proposals.get(key)
            if existing is not None:
                if existing == proposal:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("proposal ID already contains a different payload")
            self._proposals[key] = proposal
            return PutResult(proposal, created=True)

    async def get_proposal(
        self,
        user_id: str,
        proposal_id: str,
    ) -> SkillProposal | None:
        async with self._lock:
            return self._proposals.get((user_id, proposal_id))

    async def transition_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
        expected_status: ProposalStatus,
        new_status: ProposalStatus,
    ) -> SkillProposal:
        key = (user_id, proposal_id)
        async with self._lock:
            existing = self._proposals.get(key)
            if existing is None:
                raise EvolutionStoreNotFound(f"proposal {proposal_id!r} was not found")
            if existing.status is not expected_status:
                raise EvolutionStoreConflict(f"proposal expected status {expected_status.value!r}, found {existing.status.value!r}")
            validate_proposal_transition(expected_status, new_status)
            updated = SkillProposal.model_validate(
                {
                    **existing.model_dump(mode="python"),
                    "status": new_status,
                }
            )
            self._proposals[key] = updated
            return updated

    async def put_evaluation(
        self,
        evaluation: SkillEvaluation,
    ) -> PutResult[SkillEvaluation]:
        key = (evaluation.user_id, evaluation.evaluation_id)
        async with self._lock:
            existing = self._evaluations.get(key)
            if existing is not None:
                if existing == evaluation:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("evaluation ID already contains a different payload")
            self._evaluations[key] = evaluation
            return PutResult(evaluation, created=True)

    async def get_evaluation(
        self,
        user_id: str,
        evaluation_id: str,
    ) -> SkillEvaluation | None:
        async with self._lock:
            return self._evaluations.get((user_id, evaluation_id))
