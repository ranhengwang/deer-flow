"""Abstract persistence contract for skill-evolution records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

from deerflow.skill_evolution.models import (
    ClusterStatus,
    CreditKind,
    EvaluationDecision,
    EvolutionCluster,
    EvolutionEvent,
    EvolutionEventKind,
    EvolutionJob,
    EvolutionJobStatus,
    ProposalStatus,
    ProposalStatusSource,
    ProposalStatusTransition,
    PublicationStatus,
    SkillCredit,
    SkillEvaluation,
    SkillProposal,
    SkillPublication,
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
            ProposalStatus.publishing,
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
            ProposalStatus.publishing,
            ProposalStatus.rejected,
            ProposalStatus.expired,
        }
    ),
    ProposalStatus.publishing: frozenset(
        {
            ProposalStatus.staged,
            ProposalStatus.approved,
            ProposalStatus.published,
        }
    ),
    ProposalStatus.rejected: frozenset(),
    ProposalStatus.published: frozenset(
        {
            ProposalStatus.rolled_back,
        }
    ),
    ProposalStatus.rolled_back: frozenset(),
    ProposalStatus.expired: frozenset(),
}

PUBLICATION_TRANSITIONS: dict[PublicationStatus, frozenset[PublicationStatus]] = {
    PublicationStatus.preparing: frozenset(
        {
            PublicationStatus.published,
        }
    ),
    PublicationStatus.published: frozenset(
        {
            PublicationStatus.rolled_back,
        }
    ),
    PublicationStatus.rolled_back: frozenset(),
}


def validate_proposal_transition(
    expected_status: ProposalStatus,
    new_status: ProposalStatus,
    transition: ProposalStatusTransition | None = None,
) -> ProposalStatusTransition:
    """Reject status transitions outside the shared proposal state machine."""
    if new_status not in PROPOSAL_TRANSITIONS[expected_status]:
        raise EvolutionStoreConflict(f"invalid proposal status transition {expected_status.value!r} -> {new_status.value!r}")
    if transition is None:
        return ProposalStatusTransition(
            from_status=expected_status,
            to_status=new_status,
            source=ProposalStatusSource.system,
            reason_code="status_transition",
            reason=f"Proposal status changed from {expected_status.value} to {new_status.value}.",
            occurred_at=datetime.now(UTC),
        )
    if transition.from_status is not expected_status or transition.to_status is not new_status:
        raise EvolutionStoreConflict("proposal status transition metadata does not match the requested statuses")
    return transition


def validate_publication_transition(
    expected_status: PublicationStatus,
    publication: SkillPublication,
) -> None:
    if publication.status not in PUBLICATION_TRANSITIONS[expected_status]:
        raise EvolutionStoreConflict(f"invalid publication status transition {expected_status.value!r} -> {publication.status.value!r}")


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
    async def list_event_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        event_kind: EvolutionEventKind | None = None,
    ) -> list[EvolutionEvent]:
        """List one newest-first owner-scoped event page."""

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
    async def list_cluster_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: ClusterStatus | None = None,
    ) -> list[EvolutionCluster]:
        """List one newest-first owner-scoped Cluster page."""

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
    async def get_proposal_by_cluster(
        self,
        user_id: str,
        cluster_id: str,
    ) -> SkillProposal | None:
        """Return the deterministic first Proposal already stored for a Cluster."""

    @abstractmethod
    async def list_proposal_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: ProposalStatus | None = None,
    ) -> list[SkillProposal]:
        """List one newest-first owner-scoped Proposal page."""

    @abstractmethod
    async def transition_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
        expected_status: ProposalStatus,
        new_status: ProposalStatus,
        transition: ProposalStatusTransition | None = None,
    ) -> SkillProposal:
        """Compare-and-swap one proposal status and append its audit reason."""

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

    @abstractmethod
    async def list_evaluation_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        decision: EvaluationDecision | None = None,
    ) -> list[SkillEvaluation]:
        """List one newest-first owner-scoped evaluation page."""

    @abstractmethod
    async def put_publication(
        self,
        publication: SkillPublication,
    ) -> PutResult[SkillPublication]:
        """Persist a preparing publication before any Skill mutation."""

    @abstractmethod
    async def get_publication(
        self,
        user_id: str,
        publication_id: str,
    ) -> SkillPublication | None:
        """Return one publication/version record in the user's scope."""

    @abstractmethod
    async def list_publication_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: PublicationStatus | None = None,
    ) -> list[SkillPublication]:
        """List one newest-first owner-scoped publication/version page."""

    @abstractmethod
    async def transition_publication(
        self,
        publication: SkillPublication,
        *,
        expected_status: PublicationStatus,
    ) -> SkillPublication:
        """Compare-and-swap a complete publication record."""

    @abstractmethod
    async def put_credit(
        self,
        credit: SkillCredit,
    ) -> PutResult[SkillCredit]:
        """Create one immutable or revisioned Credit record idempotently."""

    @abstractmethod
    async def get_credit(
        self,
        user_id: str,
        credit_id: str,
    ) -> SkillCredit | None:
        """Return one Credit record in the user's scope."""

    @abstractmethod
    async def list_credits(
        self,
        user_id: str,
        *,
        limit: int,
        kind: CreditKind | None = None,
        skill_name: str | None = None,
    ) -> list[SkillCredit]:
        """List newest-first Credits after owner and optional filters."""

    @abstractmethod
    async def replace_credit(
        self,
        credit: SkillCredit,
        *,
        expected_revision: int,
    ) -> SkillCredit:
        """Compare-and-swap a mutable Credit revision."""

    @abstractmethod
    async def enqueue_job(
        self,
        job: EvolutionJob,
    ) -> PutResult[EvolutionJob]:
        """Persist one run-level job idempotently before publishing a wake-up."""

    @abstractmethod
    async def get_job(
        self,
        user_id: str,
        job_id: str,
    ) -> EvolutionJob | None:
        """Return one durable job in the user's scope."""

    @abstractmethod
    async def count_jobs(
        self,
        *,
        statuses: tuple[EvolutionJobStatus, ...],
    ) -> int:
        """Count jobs in fixed lifecycle states for low-cardinality metrics."""

    @abstractmethod
    async def claim_jobs(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_seconds: float,
        limit: int,
    ) -> list[EvolutionJob]:
        """Claim due pending/retry jobs and expired running leases."""

    @abstractmethod
    async def renew_job_lease(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_token: str,
        now: datetime,
        lease_seconds: float,
    ) -> EvolutionJob | None:
        """Renew the current claim, or return None after ownership changes."""

    @abstractmethod
    async def complete_job(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_token: str,
        now: datetime,
    ) -> EvolutionJob | None:
        """Commit successful completion only for the current claim token."""

    @abstractmethod
    async def retry_job(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_token: str,
        now: datetime,
        next_attempt_at: datetime | None,
        error_code: str,
        terminal: bool,
    ) -> EvolutionJob | None:
        """Release the current claim to retry or terminal dead state."""
