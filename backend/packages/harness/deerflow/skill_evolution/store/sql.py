"""SQLAlchemy-backed skill-evolution store."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.skill_evolution.model import (
    SkillEvolutionClusterRow,
    SkillEvolutionEvaluationRow,
    SkillEvolutionEventRow,
    SkillEvolutionProposalRow,
)
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


def _payload(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json")


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

    async def transition_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
        expected_status: ProposalStatus,
        new_status: ProposalStatus,
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
            validate_proposal_transition(expected_status, new_status)
            updated = SkillProposal.model_validate(
                {
                    **existing.model_dump(mode="python"),
                    "status": new_status,
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
