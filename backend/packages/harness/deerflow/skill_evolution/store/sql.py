"""SQLAlchemy-backed skill-evolution store."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import TypeAdapter
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.skill_evolution.model import (
    SkillEvolutionClusterRow,
    SkillEvolutionCreditRow,
    SkillEvolutionEvaluationRow,
    SkillEvolutionEventRow,
    SkillEvolutionJobRow,
    SkillEvolutionProposalRow,
    SkillEvolutionPublicationRow,
)
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


def _payload(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json")


_CREDIT_ADAPTER = TypeAdapter(SkillCredit)


class SqlSkillEvolutionStore(SkillEvolutionStore):
    """Short-session SQL implementation shared by SQLite and PostgreSQL."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._sf = session_factory

    @staticmethod
    def _event(row: SkillEvolutionEventRow) -> EvolutionEvent:
        return EvolutionEvent.model_validate(row.payload)

    @staticmethod
    def _cluster(row: SkillEvolutionClusterRow) -> EvolutionCluster:
        return EvolutionCluster.model_validate(row.payload)

    @staticmethod
    def _proposal(row: SkillEvolutionProposalRow) -> SkillProposal:
        return SkillProposal.model_validate(row.payload)

    @staticmethod
    def _evaluation(
        row: SkillEvolutionEvaluationRow,
    ) -> SkillEvaluation:
        return SkillEvaluation.model_validate(row.payload)

    @staticmethod
    def _publication(
        row: SkillEvolutionPublicationRow,
    ) -> SkillPublication:
        return SkillPublication.model_validate(row.payload)

    @staticmethod
    def _job(row: SkillEvolutionJobRow) -> EvolutionJob:
        return EvolutionJob.model_validate(row.payload)

    @staticmethod
    def _credit(
        row: SkillEvolutionCreditRow,
    ) -> SkillCredit:
        return _CREDIT_ADAPTER.validate_python(row.payload)

    @staticmethod
    def _updated_job(
        job: EvolutionJob,
        **updates: Any,
    ) -> EvolutionJob:
        return EvolutionJob.model_validate(
            {
                **job.model_dump(mode="python"),
                **updates,
                "revision": job.revision + 1,
            }
        )

    @staticmethod
    def _job_values(job: EvolutionJob) -> dict[str, Any]:
        return {
            "status": job.status.value,
            "attempt_count": job.attempt_count,
            "max_attempts": job.max_attempts,
            "next_attempt_at": job.next_attempt_at,
            "lease_owner": job.lease_owner,
            "lease_token": job.lease_token,
            "lease_expires_at": job.lease_expires_at,
            "last_error_code": job.last_error_code,
            "revision": job.revision,
            "payload": _payload(job),
            "updated_at": job.updated_at,
            "completed_at": job.completed_at,
        }

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

    async def _event_by_idempotency(
        self,
        session: AsyncSession,
        event: EvolutionEvent,
    ) -> SkillEvolutionEventRow | None:
        result = await session.execute(
            select(SkillEvolutionEventRow).where(
                SkillEvolutionEventRow.user_id == event.user_id,
                SkillEvolutionEventRow.run_id == event.run_id,
                SkillEvolutionEventRow.extractor_version == event.extractor_version,
            )
        )
        return result.scalar_one_or_none()

    async def upsert_event(
        self,
        event: EvolutionEvent,
    ) -> PutResult[EvolutionEvent]:
        async with self._sf() as session:
            idempotent = await self._event_by_idempotency(session, event)
            if idempotent is not None:
                existing = self._event(idempotent)
                if existing == event:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("event idempotency key already contains a different payload")

            by_id = await session.get(
                SkillEvolutionEventRow,
                (event.user_id, event.event_id),
            )
            if by_id is not None:
                existing = self._event(by_id)
                if existing == event:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("event ID already contains a different payload")

            session.add(
                SkillEvolutionEventRow(
                    user_id=event.user_id,
                    id=event.event_id,
                    run_id=event.run_id,
                    thread_id=event.thread_id,
                    extractor_version=event.extractor_version,
                    event_kind=event.event_kind.value,
                    task_signature=event.task_signature,
                    target_skill_name=(event.target_skill.name if event.target_skill is not None else None),
                    source_snapshot_hash=event.source_snapshot_hash,
                    task_input_hash=event.task_input_hash,
                    payload=_payload(event),
                    created_at=event.created_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                idempotent = await self._event_by_idempotency(session, event)
                if idempotent is not None:
                    existing = self._event(idempotent)
                    if existing == event:
                        return PutResult(existing, created=False)
                    raise EvolutionStoreConflict("event idempotency key already contains a different payload") from None
                by_id = await session.get(
                    SkillEvolutionEventRow,
                    (event.user_id, event.event_id),
                )
                if by_id is not None:
                    existing = self._event(by_id)
                    if existing == event:
                        return PutResult(existing, created=False)
                    raise EvolutionStoreConflict("event ID already contains a different payload") from None
                raise
            return PutResult(event, created=True)

    async def get_event(
        self,
        user_id: str,
        event_id: str,
    ) -> EvolutionEvent | None:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionEventRow,
                (user_id, event_id),
            )
            return self._event(row) if row is not None else None

    async def list_events(
        self,
        user_id: str,
        *,
        event_kind: EvolutionEventKind | None = None,
    ) -> list[EvolutionEvent]:
        stmt = select(SkillEvolutionEventRow).where(SkillEvolutionEventRow.user_id == user_id)
        if event_kind is not None:
            stmt = stmt.where(SkillEvolutionEventRow.event_kind == event_kind.value)
        stmt = stmt.order_by(
            SkillEvolutionEventRow.created_at.asc(),
            SkillEvolutionEventRow.id.asc(),
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._event(row) for row in result.scalars()]

    async def list_event_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        event_kind: EvolutionEventKind | None = None,
    ) -> list[EvolutionEvent]:
        self._validate_page(limit=limit, offset=offset)
        stmt = select(SkillEvolutionEventRow).where(SkillEvolutionEventRow.user_id == user_id)
        if event_kind is not None:
            stmt = stmt.where(SkillEvolutionEventRow.event_kind == event_kind.value)
        stmt = (
            stmt.order_by(
                SkillEvolutionEventRow.created_at.desc(),
                SkillEvolutionEventRow.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._event(row) for row in result.scalars()]

    async def put_cluster(
        self,
        cluster: EvolutionCluster,
    ) -> PutResult[EvolutionCluster]:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionClusterRow,
                (cluster.user_id, cluster.cluster_id),
            )
            if row is not None:
                existing = self._cluster(row)
                if existing == cluster:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("cluster ID already contains a different payload")
            session.add(
                SkillEvolutionClusterRow(
                    user_id=cluster.user_id,
                    id=cluster.cluster_id,
                    event_kind=cluster.event_kind.value,
                    target_skill_name=(cluster.target_skill.name if cluster.target_skill is not None else None),
                    canonical_signature=cluster.canonical_signature,
                    independent_run_count=cluster.independent_run_count,
                    status=cluster.status.value,
                    payload=_payload(cluster),
                    created_at=cluster.created_at,
                    updated_at=cluster.updated_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                row = await session.get(
                    SkillEvolutionClusterRow,
                    (cluster.user_id, cluster.cluster_id),
                )
                if row is not None and self._cluster(row) == cluster:
                    return PutResult(cluster, created=False)
                raise EvolutionStoreConflict("cluster ID already contains a different payload") from None
            return PutResult(cluster, created=True)

    async def get_cluster(
        self,
        user_id: str,
        cluster_id: str,
    ) -> EvolutionCluster | None:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionClusterRow,
                (user_id, cluster_id),
            )
            return self._cluster(row) if row is not None else None

    async def list_ready_clusters(
        self,
        user_id: str,
        *,
        min_distinct_runs: int,
    ) -> list[EvolutionCluster]:
        if min_distinct_runs < 1:
            raise ValueError("min_distinct_runs must be at least 1")
        stmt = (
            select(SkillEvolutionClusterRow)
            .where(
                SkillEvolutionClusterRow.user_id == user_id,
                SkillEvolutionClusterRow.status == ClusterStatus.ready.value,
                SkillEvolutionClusterRow.independent_run_count >= min_distinct_runs,
            )
            .order_by(
                SkillEvolutionClusterRow.updated_at.asc(),
                SkillEvolutionClusterRow.id.asc(),
            )
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._cluster(row) for row in result.scalars()]

    async def list_cluster_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: ClusterStatus | None = None,
    ) -> list[EvolutionCluster]:
        self._validate_page(limit=limit, offset=offset)
        stmt = select(SkillEvolutionClusterRow).where(SkillEvolutionClusterRow.user_id == user_id)
        if status is not None:
            stmt = stmt.where(SkillEvolutionClusterRow.status == status.value)
        stmt = (
            stmt.order_by(
                SkillEvolutionClusterRow.updated_at.desc(),
                SkillEvolutionClusterRow.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._cluster(row) for row in result.scalars()]

    async def put_proposal(
        self,
        proposal: SkillProposal,
    ) -> PutResult[SkillProposal]:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionProposalRow,
                (proposal.user_id, proposal.proposal_id),
            )
            if row is not None:
                existing = self._proposal(row)
                if existing == proposal:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("proposal ID already contains a different payload")
            session.add(
                SkillEvolutionProposalRow(
                    user_id=proposal.user_id,
                    id=proposal.proposal_id,
                    cluster_id=proposal.cluster_id,
                    operation=proposal.operation.value,
                    skill_name=proposal.skill_name,
                    base_skill_hash=proposal.base_skill_hash,
                    status=proposal.status.value,
                    payload=_payload(proposal),
                    created_at=proposal.created_at,
                    updated_at=proposal.created_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                row = await session.get(
                    SkillEvolutionProposalRow,
                    (proposal.user_id, proposal.proposal_id),
                )
                if row is not None and self._proposal(row) == proposal:
                    return PutResult(proposal, created=False)
                raise EvolutionStoreConflict("proposal ID already contains a different payload") from None
            return PutResult(proposal, created=True)

    async def get_proposal(
        self,
        user_id: str,
        proposal_id: str,
    ) -> SkillProposal | None:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionProposalRow,
                (user_id, proposal_id),
            )
            return self._proposal(row) if row is not None else None

    async def get_proposal_by_cluster(
        self,
        user_id: str,
        cluster_id: str,
    ) -> SkillProposal | None:
        async with self._sf() as session:
            result = await session.execute(
                select(SkillEvolutionProposalRow)
                .where(
                    SkillEvolutionProposalRow.user_id == user_id,
                    SkillEvolutionProposalRow.cluster_id == cluster_id,
                )
                .order_by(
                    SkillEvolutionProposalRow.created_at.asc(),
                    SkillEvolutionProposalRow.id.asc(),
                )
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return self._proposal(row) if row is not None else None

    async def list_proposal_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: ProposalStatus | None = None,
    ) -> list[SkillProposal]:
        self._validate_page(limit=limit, offset=offset)
        stmt = select(SkillEvolutionProposalRow).where(SkillEvolutionProposalRow.user_id == user_id)
        if status is not None:
            stmt = stmt.where(SkillEvolutionProposalRow.status == status.value)
        stmt = (
            stmt.order_by(
                SkillEvolutionProposalRow.created_at.desc(),
                SkillEvolutionProposalRow.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._proposal(row) for row in result.scalars()]

    async def transition_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
        expected_status: ProposalStatus,
        new_status: ProposalStatus,
        transition: ProposalStatusTransition | None = None,
    ) -> SkillProposal:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionProposalRow,
                (user_id, proposal_id),
            )
            if row is None:
                raise EvolutionStoreNotFound(f"proposal {proposal_id!r} was not found")
            existing = self._proposal(row)
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
            result = await session.execute(
                update(SkillEvolutionProposalRow)
                .where(
                    SkillEvolutionProposalRow.user_id == user_id,
                    SkillEvolutionProposalRow.id == proposal_id,
                    SkillEvolutionProposalRow.status == expected_status.value,
                )
                .values(
                    status=new_status.value,
                    payload=_payload(updated),
                )
            )
            if result.rowcount != 1:
                await session.rollback()
                current = await session.get(
                    SkillEvolutionProposalRow,
                    (user_id, proposal_id),
                )
                if current is None:
                    raise EvolutionStoreNotFound(f"proposal {proposal_id!r} was not found")
                raise EvolutionStoreConflict(f"proposal expected status {expected_status.value!r}, found {current.status!r}")
            await session.commit()
            return updated

    async def put_evaluation(
        self,
        evaluation: SkillEvaluation,
    ) -> PutResult[SkillEvaluation]:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionEvaluationRow,
                (evaluation.user_id, evaluation.evaluation_id),
            )
            if row is not None:
                existing = self._evaluation(row)
                if existing == evaluation:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("evaluation ID already contains a different payload")
            session.add(
                SkillEvolutionEvaluationRow(
                    user_id=evaluation.user_id,
                    id=evaluation.evaluation_id,
                    proposal_id=evaluation.proposal_id,
                    decision=evaluation.decision.value,
                    quality_score=evaluation.quality_score,
                    payload=_payload(evaluation),
                    created_at=evaluation.created_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                row = await session.get(
                    SkillEvolutionEvaluationRow,
                    (evaluation.user_id, evaluation.evaluation_id),
                )
                if row is not None and self._evaluation(row) == evaluation:
                    return PutResult(evaluation, created=False)
                raise EvolutionStoreConflict("evaluation ID already contains a different payload") from None
            return PutResult(evaluation, created=True)

    async def get_evaluation(
        self,
        user_id: str,
        evaluation_id: str,
    ) -> SkillEvaluation | None:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionEvaluationRow,
                (user_id, evaluation_id),
            )
            return self._evaluation(row) if row is not None else None

    async def list_evaluation_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        decision: EvaluationDecision | None = None,
    ) -> list[SkillEvaluation]:
        self._validate_page(limit=limit, offset=offset)
        stmt = select(SkillEvolutionEvaluationRow).where(SkillEvolutionEvaluationRow.user_id == user_id)
        if decision is not None:
            stmt = stmt.where(SkillEvolutionEvaluationRow.decision == decision.value)
        stmt = (
            stmt.order_by(
                SkillEvolutionEvaluationRow.created_at.desc(),
                SkillEvolutionEvaluationRow.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._evaluation(row) for row in result.scalars()]

    async def put_publication(
        self,
        publication: SkillPublication,
    ) -> PutResult[SkillPublication]:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionPublicationRow,
                (
                    publication.user_id,
                    publication.publication_id,
                ),
            )
            if row is not None:
                existing = self._publication(row)
                if existing == publication:
                    return PutResult(
                        existing,
                        created=False,
                    )
                raise EvolutionStoreConflict("publication ID already contains a different payload")
            session.add(
                SkillEvolutionPublicationRow(
                    user_id=publication.user_id,
                    id=publication.publication_id,
                    proposal_id=publication.proposal_id,
                    evaluation_id=publication.evaluation_id,
                    skill_name=publication.skill_name,
                    status=publication.status.value,
                    base_package_hash=publication.base_snapshot.snapshot_hash,
                    published_package_hash=None,
                    payload=_payload(publication),
                    created_at=publication.created_at,
                    updated_at=publication.created_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                row = await session.get(
                    SkillEvolutionPublicationRow,
                    (
                        publication.user_id,
                        publication.publication_id,
                    ),
                )
                if row is not None and self._publication(row) == publication:
                    return PutResult(
                        publication,
                        created=False,
                    )
                raise EvolutionStoreConflict("publication ID or Proposal already contains a different payload") from None
            return PutResult(
                publication,
                created=True,
            )

    async def get_publication(
        self,
        user_id: str,
        publication_id: str,
    ) -> SkillPublication | None:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionPublicationRow,
                (
                    user_id,
                    publication_id,
                ),
            )
            return self._publication(row) if row is not None else None

    async def list_publication_page(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        status: PublicationStatus | None = None,
    ) -> list[SkillPublication]:
        self._validate_page(limit=limit, offset=offset)
        stmt = select(SkillEvolutionPublicationRow).where(SkillEvolutionPublicationRow.user_id == user_id)
        if status is not None:
            stmt = stmt.where(SkillEvolutionPublicationRow.status == status.value)
        stmt = (
            stmt.order_by(
                SkillEvolutionPublicationRow.created_at.desc(),
                SkillEvolutionPublicationRow.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._publication(row) for row in result.scalars()]

    async def transition_publication(
        self,
        publication: SkillPublication,
        *,
        expected_status: PublicationStatus,
    ) -> SkillPublication:
        validate_publication_transition(
            expected_status,
            publication,
        )
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionPublicationRow,
                (
                    publication.user_id,
                    publication.publication_id,
                ),
            )
            if row is None:
                raise EvolutionStoreNotFound(f"publication {publication.publication_id!r} was not found")
            existing = self._publication(row)
            if existing.status is not expected_status:
                raise EvolutionStoreConflict(f"publication expected status {expected_status.value!r}, found {existing.status.value!r}")
            if existing.proposal_id != publication.proposal_id or existing.evaluation_id != publication.evaluation_id or existing.skill_name != publication.skill_name:
                raise EvolutionStoreConflict("publication identity cannot change")
            result = await session.execute(
                update(SkillEvolutionPublicationRow)
                .where(
                    SkillEvolutionPublicationRow.user_id == publication.user_id,
                    SkillEvolutionPublicationRow.id == publication.publication_id,
                    SkillEvolutionPublicationRow.status == expected_status.value,
                )
                .values(
                    status=publication.status.value,
                    published_package_hash=(publication.published_snapshot.snapshot_hash if publication.published_snapshot is not None else None),
                    payload=_payload(publication),
                    updated_at=datetime.now(UTC),
                )
            )
            if result.rowcount != 1:
                await session.rollback()
                current = await session.get(
                    SkillEvolutionPublicationRow,
                    (
                        publication.user_id,
                        publication.publication_id,
                    ),
                )
                if current is None:
                    raise EvolutionStoreNotFound(f"publication {publication.publication_id!r} was not found")
                raise EvolutionStoreConflict(f"publication expected status {expected_status.value!r}, found {current.status!r}")
            await session.commit()
            return publication

    async def put_credit(
        self,
        credit: SkillCredit,
    ) -> PutResult[SkillCredit]:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionCreditRow,
                (credit.user_id, credit.credit_id),
            )
            if row is not None:
                existing = self._credit(row)
                if existing == credit:
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("credit ID already contains a different payload")
            updated_at = getattr(
                credit,
                "updated_at",
                credit.created_at,
            )
            session.add(
                SkillEvolutionCreditRow(
                    user_id=credit.user_id,
                    id=credit.credit_id,
                    kind=credit.kind.value,
                    run_id=getattr(credit, "run_id", None),
                    skill_name=getattr(
                        credit,
                        "skill_name",
                        getattr(
                            credit,
                            "selected_skill_name",
                            None,
                        ),
                    ),
                    publication_id=getattr(
                        credit,
                        "publication_id",
                        None,
                    ),
                    revision=credit.revision,
                    payload=_payload(credit),
                    created_at=credit.created_at,
                    updated_at=updated_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                row = await session.get(
                    SkillEvolutionCreditRow,
                    (credit.user_id, credit.credit_id),
                )
                if row is not None and self._credit(row) == credit:
                    return PutResult(
                        credit,
                        created=False,
                    )
                raise EvolutionStoreConflict("credit ID already contains a different payload") from None
            return PutResult(credit, created=True)

    async def get_credit(
        self,
        user_id: str,
        credit_id: str,
    ) -> SkillCredit | None:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionCreditRow,
                (user_id, credit_id),
            )
            return self._credit(row) if row is not None else None

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
        stmt = select(SkillEvolutionCreditRow).where(SkillEvolutionCreditRow.user_id == user_id)
        if kind is not None:
            stmt = stmt.where(SkillEvolutionCreditRow.kind == kind.value)
        if skill_name is not None:
            stmt = stmt.where(SkillEvolutionCreditRow.skill_name == skill_name)
        stmt = stmt.order_by(
            SkillEvolutionCreditRow.created_at.desc(),
            SkillEvolutionCreditRow.id.desc(),
        ).limit(limit)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._credit(row) for row in result.scalars()]

    async def replace_credit(
        self,
        credit: SkillCredit,
        *,
        expected_revision: int,
    ) -> SkillCredit:
        if credit.revision != expected_revision + 1:
            raise EvolutionStoreConflict("credit revision must advance by one")
        updated_at = getattr(
            credit,
            "updated_at",
            credit.created_at,
        )
        async with self._sf() as session:
            result = await session.execute(
                update(SkillEvolutionCreditRow)
                .where(
                    SkillEvolutionCreditRow.user_id == credit.user_id,
                    SkillEvolutionCreditRow.id == credit.credit_id,
                    SkillEvolutionCreditRow.kind == credit.kind.value,
                    SkillEvolutionCreditRow.revision == expected_revision,
                )
                .values(
                    revision=credit.revision,
                    payload=_payload(credit),
                    updated_at=updated_at,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                await session.rollback()
                row = await session.get(
                    SkillEvolutionCreditRow,
                    (credit.user_id, credit.credit_id),
                )
                if row is None:
                    raise EvolutionStoreNotFound(f"credit {credit.credit_id!r} was not found")
                raise EvolutionStoreConflict("credit revision conflict")
            await session.commit()
            return credit

    async def _job_by_idempotency(
        self,
        session: AsyncSession,
        job: EvolutionJob,
    ) -> SkillEvolutionJobRow | None:
        result = await session.execute(
            select(SkillEvolutionJobRow).where(
                SkillEvolutionJobRow.user_id == job.user_id,
                SkillEvolutionJobRow.idempotency_key == job.idempotency_key,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _same_job_source(
        existing: EvolutionJob,
        requested: EvolutionJob,
    ) -> bool:
        return existing.run_id == requested.run_id and existing.thread_id == requested.thread_id and existing.snapshot_hash == requested.snapshot_hash and existing.pipeline_version == requested.pipeline_version

    async def enqueue_job(
        self,
        job: EvolutionJob,
    ) -> PutResult[EvolutionJob]:
        async with self._sf() as session:
            existing_row = await self._job_by_idempotency(
                session,
                job,
            )
            if existing_row is not None:
                existing = self._job(existing_row)
                if self._same_job_source(existing, job):
                    return PutResult(existing, created=False)
                raise EvolutionStoreConflict("job idempotency key already contains a different source")
            by_id = await session.get(
                SkillEvolutionJobRow,
                (job.user_id, job.job_id),
            )
            if by_id is not None:
                raise EvolutionStoreConflict("job ID already contains a different payload")
            session.add(
                SkillEvolutionJobRow(
                    user_id=job.user_id,
                    id=job.job_id,
                    idempotency_key=job.idempotency_key,
                    run_id=job.run_id,
                    thread_id=job.thread_id,
                    snapshot_hash=job.snapshot_hash,
                    pipeline_version=job.pipeline_version,
                    status=job.status.value,
                    attempt_count=job.attempt_count,
                    max_attempts=job.max_attempts,
                    next_attempt_at=job.next_attempt_at,
                    lease_owner=job.lease_owner,
                    lease_token=job.lease_token,
                    lease_expires_at=job.lease_expires_at,
                    last_error_code=job.last_error_code,
                    revision=job.revision,
                    payload=_payload(job),
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    completed_at=job.completed_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing_row = await self._job_by_idempotency(
                    session,
                    job,
                )
                if existing_row is not None:
                    existing = self._job(existing_row)
                    if self._same_job_source(existing, job):
                        return PutResult(existing, created=False)
                raise EvolutionStoreConflict("job ID or idempotency key already contains a different source") from None
            return PutResult(job, created=True)

    async def get_job(
        self,
        user_id: str,
        job_id: str,
    ) -> EvolutionJob | None:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionJobRow,
                (user_id, job_id),
            )
            return self._job(row) if row is not None else None

    async def count_jobs(
        self,
        *,
        statuses: tuple[EvolutionJobStatus, ...],
    ) -> int:
        if not statuses:
            return 0
        async with self._sf() as session:
            value = await session.scalar(select(func.count()).select_from(SkillEvolutionJobRow).where(SkillEvolutionJobRow.status.in_(tuple(status.value for status in statuses))))
            return int(value or 0)

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
        due = or_(
            and_(
                SkillEvolutionJobRow.status.in_(
                    (
                        EvolutionJobStatus.pending.value,
                        EvolutionJobStatus.retry.value,
                    )
                ),
                SkillEvolutionJobRow.next_attempt_at.is_not(None),
                SkillEvolutionJobRow.next_attempt_at <= now,
            ),
            and_(
                SkillEvolutionJobRow.status == EvolutionJobStatus.running.value,
                SkillEvolutionJobRow.lease_expires_at.is_not(None),
                SkillEvolutionJobRow.lease_expires_at <= now,
            ),
        )
        async with self._sf() as session:
            result = await session.execute(
                select(SkillEvolutionJobRow)
                .where(due)
                .order_by(
                    SkillEvolutionJobRow.updated_at.asc(),
                    SkillEvolutionJobRow.id.asc(),
                )
                .limit(max(limit * 4, limit))
            )
            claimed: list[EvolutionJob] = []
            for row in result.scalars():
                if len(claimed) >= limit:
                    break
                job = self._job(row)
                if job.attempt_count >= job.max_attempts:
                    updated_job = self._updated_job(
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
                else:
                    updated_job = self._updated_job(
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
                updated = await session.execute(
                    update(SkillEvolutionJobRow)
                    .where(
                        SkillEvolutionJobRow.user_id == job.user_id,
                        SkillEvolutionJobRow.id == job.job_id,
                        SkillEvolutionJobRow.revision == job.revision,
                        due,
                    )
                    .values(**self._job_values(updated_job))
                    .execution_options(synchronize_session=False)
                )
                if updated.rowcount != 1:
                    continue
                if updated_job.status is EvolutionJobStatus.running:
                    claimed.append(updated_job)
            await session.commit()
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
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionJobRow,
                (user_id, job_id),
            )
            if row is None:
                return None
            job = self._job(row)
            if job.status is not EvolutionJobStatus.running or job.lease_token != lease_token or job.lease_expires_at is None or job.lease_expires_at <= now:
                return None
            renewed = self._updated_job(
                job,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                updated_at=now,
            )
            result = await session.execute(
                update(SkillEvolutionJobRow)
                .where(
                    SkillEvolutionJobRow.user_id == user_id,
                    SkillEvolutionJobRow.id == job_id,
                    SkillEvolutionJobRow.revision == job.revision,
                    SkillEvolutionJobRow.status == EvolutionJobStatus.running.value,
                    SkillEvolutionJobRow.lease_token == lease_token,
                    SkillEvolutionJobRow.lease_expires_at > now,
                )
                .values(**self._job_values(renewed))
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                await session.rollback()
                return None
            await session.commit()
            return renewed

    async def complete_job(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_token: str,
        now: datetime,
    ) -> EvolutionJob | None:
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionJobRow,
                (user_id, job_id),
            )
            if row is None:
                return None
            job = self._job(row)
            if job.status is not EvolutionJobStatus.running or job.lease_token != lease_token or job.lease_expires_at is None or job.lease_expires_at <= now:
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
            result = await session.execute(
                update(SkillEvolutionJobRow)
                .where(
                    SkillEvolutionJobRow.user_id == user_id,
                    SkillEvolutionJobRow.id == job_id,
                    SkillEvolutionJobRow.revision == job.revision,
                    SkillEvolutionJobRow.status == EvolutionJobStatus.running.value,
                    SkillEvolutionJobRow.lease_token == lease_token,
                    SkillEvolutionJobRow.lease_expires_at > now,
                )
                .values(**self._job_values(completed))
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                await session.rollback()
                return None
            await session.commit()
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
        if not terminal and next_attempt_at is None:
            raise ValueError("retry requires next_attempt_at")
        async with self._sf() as session:
            row = await session.get(
                SkillEvolutionJobRow,
                (user_id, job_id),
            )
            if row is None:
                return None
            job = self._job(row)
            if job.status is not EvolutionJobStatus.running or job.lease_token != lease_token:
                return None
            updated_job = self._updated_job(
                job,
                status=(EvolutionJobStatus.dead if terminal else EvolutionJobStatus.retry),
                next_attempt_at=(None if terminal else next_attempt_at),
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                last_error_code=error_code,
                updated_at=now,
                completed_at=now if terminal else None,
            )
            result = await session.execute(
                update(SkillEvolutionJobRow)
                .where(
                    SkillEvolutionJobRow.user_id == user_id,
                    SkillEvolutionJobRow.id == job_id,
                    SkillEvolutionJobRow.revision == job.revision,
                    SkillEvolutionJobRow.status == EvolutionJobStatus.running.value,
                    SkillEvolutionJobRow.lease_token == lease_token,
                )
                .values(**self._job_values(updated_job))
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                await session.rollback()
                return None
            await session.commit()
            return updated_job
