"""Process-local skill-evolution store for tests and single-process use."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

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
    ProposalStatusTransition,
    PublicationStatus,
    SkillCredit,
    SkillEvaluation,
    SkillProposal,
    SkillPublication,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
    EvolutionStoreNotFound,
    PutResult,
    SkillEvolutionStore,
    validate_proposal_transition,
    validate_publication_transition,
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
        self._publications: dict[tuple[str, str], SkillPublication] = {}
        self._credits: dict[tuple[str, str], SkillCredit] = {}
        self._jobs: dict[tuple[str, str], EvolutionJob] = {}
        self._job_idempotency: dict[tuple[str, str], str] = {}

    @staticmethod
    def _updated_job(
        job: EvolutionJob,
        **updates,
    ) -> EvolutionJob:
        return EvolutionJob.model_validate(
            {
                **job.model_dump(mode="python"),
                **updates,
                "revision": job.revision + 1,
            }
        )

    @staticmethod
    def _validate_page(
        *,
        limit: int,
        offset: int,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")

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

    async def list_event_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        event_kind: EvolutionEventKind | None = None,
    ) -> list[EvolutionEvent]:
        self._validate_page(limit=limit, offset=offset)
        async with self._lock:
            events = [event for (owner, _), event in self._events.items() if owner == user_id and (event_kind is None or event.event_kind is event_kind)]
        ordered = sorted(
            events,
            key=lambda item: (
                item.created_at,
                item.event_id,
            ),
            reverse=True,
        )
        return ordered[offset : offset + limit]

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

    async def list_cluster_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: ClusterStatus | None = None,
    ) -> list[EvolutionCluster]:
        self._validate_page(limit=limit, offset=offset)
        async with self._lock:
            clusters = [cluster for (owner, _), cluster in self._clusters.items() if owner == user_id and (status is None or cluster.status is status)]
        ordered = sorted(
            clusters,
            key=lambda item: (
                item.updated_at,
                item.cluster_id,
            ),
            reverse=True,
        )
        return ordered[offset : offset + limit]

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

    async def get_proposal_by_cluster(
        self,
        user_id: str,
        cluster_id: str,
    ) -> SkillProposal | None:
        async with self._lock:
            proposals = [proposal for (owner, _), proposal in self._proposals.items() if owner == user_id and proposal.cluster_id == cluster_id]
        return min(
            proposals,
            key=lambda proposal: (
                proposal.created_at,
                proposal.proposal_id,
            ),
            default=None,
        )

    async def list_proposal_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: ProposalStatus | None = None,
    ) -> list[SkillProposal]:
        self._validate_page(limit=limit, offset=offset)
        async with self._lock:
            proposals = [proposal for (owner, _), proposal in self._proposals.items() if owner == user_id and (status is None or proposal.status is status)]
        ordered = sorted(
            proposals,
            key=lambda item: (
                item.created_at,
                item.proposal_id,
            ),
            reverse=True,
        )
        return ordered[offset : offset + limit]

    async def transition_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
        expected_status: ProposalStatus,
        new_status: ProposalStatus,
        transition: ProposalStatusTransition | None = None,
    ) -> SkillProposal:
        key = (user_id, proposal_id)
        async with self._lock:
            existing = self._proposals.get(key)
            if existing is None:
                raise EvolutionStoreNotFound(f"proposal {proposal_id!r} was not found")
            if existing.status is not expected_status:
                raise EvolutionStoreConflict(f"proposal expected status {expected_status.value!r}, found {existing.status.value!r}")
            transition = validate_proposal_transition(
                expected_status,
                new_status,
                transition,
            )
            updated = SkillProposal.model_validate(
                {
                    **existing.model_dump(mode="python"),
                    "status": new_status,
                    "status_history": [
                        *existing.status_history,
                        transition,
                    ],
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

    async def list_evaluation_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        decision: EvaluationDecision | None = None,
    ) -> list[SkillEvaluation]:
        self._validate_page(limit=limit, offset=offset)
        async with self._lock:
            evaluations = [evaluation for (owner, _), evaluation in self._evaluations.items() if owner == user_id and (decision is None or evaluation.decision is decision)]
        ordered = sorted(
            evaluations,
            key=lambda item: (
                item.created_at,
                item.evaluation_id,
            ),
            reverse=True,
        )
        return ordered[offset : offset + limit]

    async def put_publication(
        self,
        publication: SkillPublication,
    ) -> PutResult[SkillPublication]:
        key = (
            publication.user_id,
            publication.publication_id,
        )
        async with self._lock:
            existing = self._publications.get(key)
            if existing is not None:
                if existing == publication:
                    return PutResult(
                        existing,
                        created=False,
                    )
                raise EvolutionStoreConflict("publication ID already contains a different payload")
            self._publications[key] = publication
            return PutResult(
                publication,
                created=True,
            )

    async def get_publication(
        self,
        user_id: str,
        publication_id: str,
    ) -> SkillPublication | None:
        async with self._lock:
            return self._publications.get(
                (
                    user_id,
                    publication_id,
                )
            )

    async def list_publication_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: PublicationStatus | None = None,
    ) -> list[SkillPublication]:
        self._validate_page(limit=limit, offset=offset)
        async with self._lock:
            publications = [publication for (owner, _), publication in self._publications.items() if owner == user_id and (status is None or publication.status is status)]
        ordered = sorted(
            publications,
            key=lambda item: (
                item.created_at,
                item.publication_id,
            ),
            reverse=True,
        )
        return ordered[offset : offset + limit]

    async def transition_publication(
        self,
        publication: SkillPublication,
        *,
        expected_status: PublicationStatus,
    ) -> SkillPublication:
        key = (
            publication.user_id,
            publication.publication_id,
        )
        async with self._lock:
            existing = self._publications.get(key)
            if existing is None:
                raise EvolutionStoreNotFound(f"publication {publication.publication_id!r} was not found")
            if existing.status is not expected_status:
                raise EvolutionStoreConflict(f"publication expected status {expected_status.value!r}, found {existing.status.value!r}")
            if existing.user_id != publication.user_id or existing.publication_id != publication.publication_id or existing.proposal_id != publication.proposal_id or existing.evaluation_id != publication.evaluation_id:
                raise EvolutionStoreConflict("publication identity cannot change")
            validate_publication_transition(
                expected_status,
                publication,
            )
            self._publications[key] = publication
            return publication

    async def put_credit(
        self,
        credit: SkillCredit,
    ) -> PutResult[SkillCredit]:
        key = (credit.user_id, credit.credit_id)
        async with self._lock:
            existing = self._credits.get(key)
            if existing is not None:
                if existing == credit:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("credit ID already contains a different payload")
            self._credits[key] = credit
            return PutResult(credit, created=True)

    async def get_credit(
        self,
        user_id: str,
        credit_id: str,
    ) -> SkillCredit | None:
        async with self._lock:
            return self._credits.get((user_id, credit_id))

    async def list_credits(
        self,
        user_id: str,
        *,
        limit: int,
        kind: CreditKind | None = None,
        skill_name: str | None = None,
    ) -> list[SkillCredit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._lock:
            credits = [
                credit
                for (owner, _), credit in self._credits.items()
                if owner == user_id
                and (kind is None or credit.kind is kind)
                and (
                    skill_name is None
                    or getattr(
                        credit,
                        "skill_name",
                        getattr(
                            credit,
                            "selected_skill_name",
                            None,
                        ),
                    )
                    == skill_name
                )
            ]
        return sorted(
            credits,
            key=lambda item: (
                item.created_at,
                item.credit_id,
            ),
            reverse=True,
        )[:limit]

    async def replace_credit(
        self,
        credit: SkillCredit,
        *,
        expected_revision: int,
    ) -> SkillCredit:
        key = (credit.user_id, credit.credit_id)
        async with self._lock:
            existing = self._credits.get(key)
            if existing is None:
                raise EvolutionStoreNotFound(f"credit {credit.credit_id!r} was not found")
            if existing.revision != expected_revision:
                raise EvolutionStoreConflict("credit revision conflict")
            if existing.kind is not credit.kind or credit.revision != expected_revision + 1:
                raise EvolutionStoreConflict("credit identity or revision cannot change")
            self._credits[key] = credit
            return credit

    async def enqueue_job(
        self,
        job: EvolutionJob,
    ) -> PutResult[EvolutionJob]:
        key = (job.user_id, job.job_id)
        idempotency_key = (
            job.user_id,
            job.idempotency_key,
        )
        async with self._lock:
            existing_id = self._job_idempotency.get(idempotency_key)
            if existing_id is not None:
                existing = self._jobs[(job.user_id, existing_id)]
                if existing.run_id == job.run_id and existing.thread_id == job.thread_id and existing.snapshot_hash == job.snapshot_hash and existing.pipeline_version == job.pipeline_version:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("job idempotency key already contains a different source")
            existing = self._jobs.get(key)
            if existing is not None:
                raise EvolutionStoreConflict("job ID already contains a different payload")
            self._jobs[key] = job
            self._job_idempotency[idempotency_key] = job.job_id
            return PutResult(job, created=True)

    async def get_job(
        self,
        user_id: str,
        job_id: str,
    ) -> EvolutionJob | None:
        async with self._lock:
            return self._jobs.get((user_id, job_id))

    async def count_jobs(
        self,
        *,
        statuses: tuple[EvolutionJobStatus, ...],
    ) -> int:
        selected = set(statuses)
        if not selected:
            return 0
        async with self._lock:
            return sum(job.status in selected for job in self._jobs.values())

    async def claim_jobs(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_seconds: float,
        limit: int,
    ) -> list[EvolutionJob]:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._lock:
            due = [
                job
                for job in self._jobs.values()
                if (
                    job.status
                    in {
                        EvolutionJobStatus.pending,
                        EvolutionJobStatus.retry,
                    }
                    and job.next_attempt_at is not None
                    and job.next_attempt_at <= now
                )
                or (job.status is EvolutionJobStatus.running and job.lease_expires_at is not None and job.lease_expires_at <= now)
            ]
            due.sort(
                key=lambda job: (
                    job.next_attempt_at or job.lease_expires_at or job.created_at,
                    job.created_at,
                    job.job_id,
                )
            )
            claimed: list[EvolutionJob] = []
            for job in due[:limit]:
                if job.attempt_count >= job.max_attempts:
                    dead = self._updated_job(
                        job,
                        status=EvolutionJobStatus.dead,
                        next_attempt_at=None,
                        lease_owner=None,
                        lease_token=None,
                        lease_expires_at=None,
                        last_error_code=(job.last_error_code or "attempts_exhausted"),
                        updated_at=now,
                        completed_at=now,
                    )
                    self._jobs[(job.user_id, job.job_id)] = dead
                    continue
                claimed_job = self._updated_job(
                    job,
                    status=EvolutionJobStatus.running,
                    attempt_count=job.attempt_count + 1,
                    next_attempt_at=None,
                    lease_owner=lease_owner,
                    lease_token=uuid.uuid4().hex,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                    completed_at=None,
                )
                self._jobs[(job.user_id, job.job_id)] = claimed_job
                claimed.append(claimed_job)
            return claimed

    async def renew_job_lease(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_token: str,
        now: datetime,
        lease_seconds: float,
    ) -> EvolutionJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        key = (user_id, job_id)
        async with self._lock:
            job = self._jobs.get(key)
            if job is None or job.status is not EvolutionJobStatus.running or job.lease_token != lease_token or job.lease_expires_at is None or job.lease_expires_at <= now:
                return None
            renewed = self._updated_job(
                job,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                updated_at=now,
            )
            self._jobs[key] = renewed
            return renewed

    async def complete_job(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_token: str,
        now: datetime,
    ) -> EvolutionJob | None:
        key = (user_id, job_id)
        async with self._lock:
            job = self._jobs.get(key)
            if job is None or job.status is not EvolutionJobStatus.running or job.lease_token != lease_token or job.lease_expires_at is None or job.lease_expires_at <= now:
                return None
            completed = self._updated_job(
                job,
                status=EvolutionJobStatus.completed,
                next_attempt_at=None,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                last_error_code=None,
                updated_at=now,
                completed_at=now,
            )
            self._jobs[key] = completed
            return completed

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
        key = (user_id, job_id)
        async with self._lock:
            job = self._jobs.get(key)
            if job is None or job.status is not EvolutionJobStatus.running or job.lease_token != lease_token:
                return None
            if terminal:
                status = EvolutionJobStatus.dead
                due_at = None
                completed_at = now
            else:
                if next_attempt_at is None:
                    raise ValueError("retry requires next_attempt_at")
                status = EvolutionJobStatus.retry
                due_at = next_attempt_at
                completed_at = None
            updated = self._updated_job(
                job,
                status=status,
                next_attempt_at=due_at,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                last_error_code=error_code,
                updated_at=now,
                completed_at=completed_at,
            )
            self._jobs[key] = updated
            return updated
