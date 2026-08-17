"""Abstract persistence contract for skill-evolution records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from deerflow.skill_evolution.models import (
    EvolutionCluster,
    EvolutionEvent,
    EvolutionEventKind,
    ProposalStatus,
    SkillEvaluation,
    SkillProposal,
)


class EvolutionStoreError(RuntimeError):
    """Base class for skill-evolution persistence failures."""


class EvolutionStoreConflict(EvolutionStoreError):
    """Raised when idempotency or optimistic-concurrency checks fail."""


class EvolutionStoreNotFound(EvolutionStoreError):
    """Raised when a required evolution record does not exist."""


PROPOSAL_TRANSITIONS: dict[ProposalStatus, frozenset[ProposalStatus]] = {
    ProposalStatus.staged: frozenset(
        {
            ProposalStatus.validating,
            ProposalStatus.rejected,
            ProposalStatus.expired,
        }
    ),
    ProposalStatus.validating: frozenset(
        {
            ProposalStatus.approved,
            ProposalStatus.rejected,
            ProposalStatus.expired,
        }
    ),
    ProposalStatus.approved: frozenset(
        {
            ProposalStatus.published,
            ProposalStatus.rejected,
            ProposalStatus.expired,
        }
    ),
    ProposalStatus.rejected: frozenset(),
    ProposalStatus.published: frozenset(),
    ProposalStatus.expired: frozenset(),
}


def validate_proposal_transition(
    expected_status: ProposalStatus,
    new_status: ProposalStatus,
) -> None:
    """Reject status transitions outside the shared proposal state machine."""
    if new_status not in PROPOSAL_TRANSITIONS[expected_status]:
        raise EvolutionStoreConflict(f"invalid proposal status transition {expected_status.value!r} -> {new_status.value!r}")


@dataclass(frozen=True, slots=True)
class PutResult[RecordT]:
    """Result of an idempotent create operation."""

    value: RecordT
    created: bool


class SkillEvolutionStore(ABC):
    """Async store used by online and background evolution paths."""

    @abstractmethod
    async def upsert_event(
        self,
        event: EvolutionEvent,
    ) -> PutResult[EvolutionEvent]:
        """Insert one event, idempotent by user/run/extractor version."""

    @abstractmethod
    async def get_event(
        self,
        user_id: str,
        event_id: str,
    ) -> EvolutionEvent | None:
        """Return one event in the user's scope."""

    @abstractmethod
    async def list_events(
        self,
        user_id: str,
        *,
        event_kind: EvolutionEventKind | None = None,
    ) -> list[EvolutionEvent]:
        """List events in deterministic creation order."""

    @abstractmethod
    async def put_cluster(
        self,
        cluster: EvolutionCluster,
    ) -> PutResult[EvolutionCluster]:
        """Create one immutable cluster snapshot idempotently."""

    @abstractmethod
    async def get_cluster(
        self,
        user_id: str,
        cluster_id: str,
    ) -> EvolutionCluster | None:
        """Return one cluster in the user's scope."""

    @abstractmethod
    async def list_ready_clusters(
        self,
        user_id: str,
        *,
        min_distinct_runs: int,
    ) -> list[EvolutionCluster]:
        """List ready clusters that satisfy the evidence threshold."""

    @abstractmethod
    async def put_proposal(
        self,
        proposal: SkillProposal,
    ) -> PutResult[SkillProposal]:
        """Create one proposal idempotently."""

    @abstractmethod
    async def get_proposal(
        self,
        user_id: str,
        proposal_id: str,
    ) -> SkillProposal | None:
        """Return one proposal in the user's scope."""

    @abstractmethod
    async def transition_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
        expected_status: ProposalStatus,
        new_status: ProposalStatus,
    ) -> SkillProposal:
        """Compare-and-swap one proposal status."""

    @abstractmethod
    async def put_evaluation(
        self,
        evaluation: SkillEvaluation,
    ) -> PutResult[SkillEvaluation]:
        """Create one evaluation idempotently."""

    @abstractmethod
    async def get_evaluation(
        self,
        user_id: str,
        evaluation_id: str,
    ) -> SkillEvaluation | None:
        """Return one evaluation in the user's scope."""
