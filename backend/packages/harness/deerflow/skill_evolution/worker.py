"""Lease-aware execution for one durable skill-evolution job."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from deerflow.config.app_config import AppConfig
from deerflow.config.skill_evolution_config import (
    SkillEvolutionCoordinatorConfig,
)
from deerflow.runtime.events.catalog import EVOLUTION_TRACE_EVENT
from deerflow.skill_evolution.cluster_confirmation import (
    StructuredClusterConfirmer,
    confirm_cluster_readiness,
)
from deerflow.skill_evolution.credit import CreditRecorder
from deerflow.skill_evolution.distiller import NewSkillDistiller
from deerflow.skill_evolution.eligibility import (
    evaluate_evolution_eligibility,
)
from deerflow.skill_evolution.extractor import pre_extract_evolution
from deerflow.skill_evolution.grouping import group_evolution_events
from deerflow.skill_evolution.models import (
    ClusterStatus,
    EvolutionEvent,
    EvolutionEventKind,
    EvolutionJob,
    EvolutionTraceSnapshot,
)
from deerflow.skill_evolution.observability import (
    EvolutionLifecycleKind,
    EvolutionObservability,
    get_evolution_observability,
    safe_emit_evolution_event,
    safe_increment_evolution_metric,
    safe_observe_evolution_metric,
)
from deerflow.skill_evolution.patch_distiller import (
    PatchSkillDistiller,
    SkillStorageVersionSource,
)
from deerflow.skill_evolution.semantic_retrieval import (
    retrieve_event_candidates,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
    PutResult,
    SkillEvolutionStore,
)
from deerflow.skill_evolution.structured_extractor import (
    StructuredEvolutionExtractor,
)
from deerflow.skill_evolution.verifier import (
    VerificationContext,
    verify_outcome,
)

logger = logging.getLogger(__name__)

EvolutionJobProcessor = Callable[[EvolutionJob], Awaitable[None]]
DirectProposalPublisher = Callable[..., Awaitable[Any]]


class EvolutionSnapshotUnavailable(RuntimeError):
    """The durable job exists but its source trace cannot be loaded yet."""


class EvolutionPipelineProcessor:
    """Advance one redacted run through extraction and staged distillation."""

    def __init__(
        self,
        *,
        event_store: Any,
        evolution_store: SkillEvolutionStore,
        app_config_provider: Callable[[], AppConfig],
        extractor_factory: Callable[
            [AppConfig],
            Any,
        ]
        | None = None,
        confirmer_factory: Callable[
            [AppConfig],
            Any,
        ]
        | None = None,
        new_skill_distiller_factory: Callable[
            [AppConfig],
            Any,
        ]
        | None = None,
        patch_distiller_factory: Callable[
            [AppConfig],
            Any,
        ]
        | None = None,
        direct_publisher: DirectProposalPublisher | None = None,
        credit_recorder: CreditRecorder | None = None,
        observability: EvolutionObservability | None = None,
    ) -> None:
        self._event_store = event_store
        self._store = evolution_store
        self._app_config_provider = app_config_provider
        self._extractor_factory = extractor_factory or StructuredEvolutionExtractor.from_app_config
        self._confirmer_factory = confirmer_factory or StructuredClusterConfirmer.from_app_config
        self._new_skill_distiller_factory = new_skill_distiller_factory or NewSkillDistiller.from_app_config
        self._patch_distiller_factory = patch_distiller_factory or PatchSkillDistiller.from_app_config
        self._direct_publisher = direct_publisher
        self._cluster_locks = tuple(asyncio.Lock() for _ in range(64))
        self._distilled_cluster_ids: set[str] = set()
        self._credit_recorder = credit_recorder or CreditRecorder(evolution_store)
        self._observability = observability or get_evolution_observability()

    async def _load_snapshot(
        self,
        job: EvolutionJob,
    ) -> EvolutionTraceSnapshot:
        records = await self._event_store.list_events(
            job.thread_id,
            job.run_id,
            event_types=[EVOLUTION_TRACE_EVENT.event_type],
            limit=2,
            user_id=job.user_id,
        )
        if len(records) != 1:
            raise EvolutionSnapshotUnavailable("evolution trace snapshot is unavailable")
        snapshot = EvolutionTraceSnapshot.model_validate(records[0].get("content"))
        if snapshot.user_id != job.user_id or snapshot.thread_id != job.thread_id or snapshot.run_id != job.run_id or snapshot.snapshot_hash != job.snapshot_hash:
            raise EvolutionSnapshotUnavailable("evolution trace snapshot identity does not match its job")
        return snapshot

    @staticmethod
    def _target_name(event: EvolutionEvent) -> str | None:
        return event.target_skill.name if event.target_skill is not None else None

    async def _existing_event(
        self,
        snapshot: EvolutionTraceSnapshot,
    ) -> EvolutionEvent | None:
        events = await self._store.list_events(snapshot.user_id)
        for event in events:
            if event.run_id == snapshot.run_id and event.source_snapshot_hash == snapshot.snapshot_hash:
                return event
        return None

    async def _distill_ready_cluster(
        self,
        *,
        job: EvolutionJob,
        app_config: AppConfig,
        cluster,
        events: list[EvolutionEvent],
    ) -> PutResult[Any] | None:
        existing = await self._store.get_proposal_by_cluster(
            cluster.user_id,
            cluster.cluster_id,
        )
        if existing is not None:
            result = PutResult(existing, created=False)
        elif cluster.event_kind is EvolutionEventKind.new_skill_evidence:
            distiller = self._new_skill_distiller_factory(app_config)
            result = await distiller.distill_and_persist(
                cluster,
                events,
                self._store,
            )
        else:
            from deerflow.skills.storage import (
                get_or_new_user_skill_storage,
            )

            storage = await asyncio.to_thread(
                get_or_new_user_skill_storage,
                job.user_id,
                app_config=app_config,
            )
            distiller = self._patch_distiller_factory(app_config)
            result = await distiller.distill_and_persist(
                cluster,
                events,
                SkillStorageVersionSource(storage),
                self._store,
            )
        if result is None:
            safe_emit_evolution_event(
                self._observability,
                kind=EvolutionLifecycleKind.rejected,
                stage="proposal_distillation",
                user_id=cluster.user_id,
                job_id=job.job_id,
                cluster_id=cluster.cluster_id,
                reason_codes=["distillation_failed"],
            )
            return None
        if result.created:
            safe_emit_evolution_event(
                self._observability,
                kind=EvolutionLifecycleKind.distilled,
                stage="proposal_distillation",
                user_id=cluster.user_id,
                skill_name=result.value.skill_name,
                job_id=job.job_id,
                cluster_id=cluster.cluster_id,
                proposal_id=result.value.proposal_id,
                event_count=len(result.value.supporting_event_ids),
                occurred_at=result.value.created_at,
            )
        if app_config.skill_evolution.publication.mode == "direct":
            if self._direct_publisher is None:
                raise RuntimeError("direct Skill publication is not configured")
            await self._direct_publisher(
                user_id=result.value.user_id,
                proposal_id=result.value.proposal_id,
            )
        return result

    async def _advance_event(
        self,
        *,
        job: EvolutionJob,
        app_config: AppConfig,
        event: EvolutionEvent,
    ) -> None:
        all_events = await self._store.list_events(
            event.user_id,
            event_kind=event.event_kind,
        )
        target_name = self._target_name(event)
        partition = [candidate for candidate in all_events if self._target_name(candidate) == target_name][-2_048:]
        grouping = group_evolution_events(
            partition,
            config=app_config.skill_evolution.grouping,
        )
        cluster = next(
            (candidate for candidate in grouping.clusters if event.event_id in candidate.member_event_ids),
            None,
        )
        if cluster is None:
            return
        safe_emit_evolution_event(
            self._observability,
            kind=EvolutionLifecycleKind.clustered,
            stage="deterministic_grouping",
            user_id=event.user_id,
            skill_name=target_name,
            job_id=job.job_id,
            evolution_event_id=event.event_id,
            cluster_id=cluster.cluster_id,
            event_count=len(cluster.member_event_ids),
            distinct_run_count=cluster.independent_run_count,
        )

        lock_index = int(
            hashlib.sha256(cluster.cluster_id.encode("utf-8")).hexdigest()[:8],
            16,
        ) % len(self._cluster_locks)
        async with self._cluster_locks[lock_index]:
            existing_cluster = await self._store.get_cluster(
                event.user_id,
                cluster.cluster_id,
            )
            if existing_cluster is not None and existing_cluster.status is ClusterStatus.ready:
                if existing_cluster.cluster_id in self._distilled_cluster_ids:
                    return
                distilled = await self._distill_ready_cluster(
                    job=job,
                    app_config=app_config,
                    cluster=existing_cluster,
                    events=partition,
                )
                if distilled:
                    self._distilled_cluster_ids.add(existing_cluster.cluster_id)
                return

            prototype_id = cluster.member_event_ids[0]
            prototype = next(candidate for candidate in partition if candidate.event_id == prototype_id)
            retrieval = await retrieve_event_candidates(
                prototype,
                [candidate for candidate in partition if candidate.event_id != prototype.event_id],
                config=app_config.skill_evolution.grouping,
            )
            confirmer = self._confirmer_factory(app_config)
            readiness = await confirm_cluster_readiness(
                cluster,
                partition,
                confirmer=confirmer,
                retrieval=retrieval,
                evidence_config=app_config.skill_evolution.evidence,
                grouping_config=app_config.skill_evolution.grouping,
            )
            accepted_count = len(readiness.cluster.member_event_ids)
            candidate_count = accepted_count + len(readiness.rejected_event_ids) + len(readiness.unconfirmed_event_ids)
            if candidate_count:
                safe_observe_evolution_metric(
                    self._observability,
                    "cluster_purity",
                    accepted_count / candidate_count,
                    deduplication_key=(f"{cluster.cluster_id}:{candidate_count}:{accepted_count}"),
                )
            if readiness.cluster.status is not ClusterStatus.ready:
                safe_emit_evolution_event(
                    self._observability,
                    kind=EvolutionLifecycleKind.rejected,
                    stage="cluster_confirmation",
                    user_id=event.user_id,
                    skill_name=target_name,
                    job_id=job.job_id,
                    evolution_event_id=event.event_id,
                    cluster_id=cluster.cluster_id,
                    reason_codes=readiness.readiness_blockers,
                    candidate_count=candidate_count,
                    accepted_count=accepted_count,
                )
                return
            try:
                stored = await self._store.put_cluster(readiness.cluster)
                ready_cluster = stored.value
            except EvolutionStoreConflict:
                # A peer owns the post-cluster distillation continuation. If it
                # exits before Proposal persistence, its durable run job will
                # retry and enter the existing-ready recovery branch above.
                return
            if stored.created:
                safe_emit_evolution_event(
                    self._observability,
                    kind=EvolutionLifecycleKind.ready,
                    stage="cluster_confirmation",
                    user_id=event.user_id,
                    skill_name=target_name,
                    job_id=job.job_id,
                    evolution_event_id=event.event_id,
                    cluster_id=ready_cluster.cluster_id,
                    event_count=len(ready_cluster.member_event_ids),
                    distinct_run_count=(ready_cluster.independent_run_count),
                    candidate_count=candidate_count,
                    accepted_count=accepted_count,
                    occurred_at=ready_cluster.updated_at,
                )
            distilled = await self._distill_ready_cluster(
                job=job,
                app_config=app_config,
                cluster=ready_cluster,
                events=partition,
            )
            if distilled:
                self._distilled_cluster_ids.add(ready_cluster.cluster_id)

    async def __call__(self, job: EvolutionJob) -> None:
        snapshot = await self._load_snapshot(job)
        app_config = self._app_config_provider()
        if not app_config.skill_evolution.enabled:
            return
        outcome = verify_outcome(VerificationContext(snapshot=snapshot))
        try:
            await self._credit_recorder.record_verified_run(
                snapshot,
                outcome,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to persist Skill Credit records for run %s (non-fatal)",
                snapshot.run_id,
            )
        event = await self._existing_event(snapshot)
        if event is None:
            eligibility = evaluate_evolution_eligibility(
                snapshot,
                outcome,
                config=app_config.skill_evolution.evidence,
            )
            if not eligibility.eligible:
                safe_emit_evolution_event(
                    self._observability,
                    kind=EvolutionLifecycleKind.rejected,
                    stage="eligibility",
                    user_id=snapshot.user_id,
                    run_id=snapshot.run_id,
                    job_id=job.job_id,
                    snapshot_hash=snapshot.snapshot_hash,
                    decision=eligibility.branch.value,
                    reason_codes=eligibility.rejection_reasons,
                    occurred_at=snapshot.created_at,
                )
                return
            safe_emit_evolution_event(
                self._observability,
                kind=EvolutionLifecycleKind.admitted,
                stage="eligibility",
                user_id=snapshot.user_id,
                run_id=snapshot.run_id,
                job_id=job.job_id,
                snapshot_hash=snapshot.snapshot_hash,
                decision=eligibility.branch.value,
                occurred_at=snapshot.created_at,
            )
            extraction = pre_extract_evolution(
                snapshot,
                outcome,
                eligibility,
            )
            extractor = self._extractor_factory(app_config)
            persisted = await extractor.extract_and_persist(
                extraction,
                self._store,
            )
            if persisted is None:
                safe_increment_evolution_metric(
                    self._observability,
                    "extraction_failures",
                )
                safe_emit_evolution_event(
                    self._observability,
                    kind=EvolutionLifecycleKind.rejected,
                    stage="structured_extraction",
                    user_id=snapshot.user_id,
                    run_id=snapshot.run_id,
                    job_id=job.job_id,
                    snapshot_hash=snapshot.snapshot_hash,
                    reason_codes=["structured_extraction_failed"],
                )
                return
            event = persisted.value
            if persisted.created:
                safe_emit_evolution_event(
                    self._observability,
                    kind=EvolutionLifecycleKind.extracted,
                    stage="structured_extraction",
                    user_id=event.user_id,
                    skill_name=self._target_name(event),
                    run_id=event.run_id,
                    job_id=job.job_id,
                    evolution_event_id=event.event_id,
                    snapshot_hash=event.source_snapshot_hash,
                    occurred_at=event.created_at,
                )
        await self._advance_event(
            job=job,
            app_config=app_config,
            event=event,
        )


class EvolutionJobWorker:
    """Run one claimed job while renewing its durable lease."""

    def __init__(
        self,
        *,
        store: SkillEvolutionStore,
        processor: EvolutionJobProcessor,
        config: SkillEvolutionCoordinatorConfig,
    ) -> None:
        self._store = store
        self._processor = processor
        self._config = config

    def _retry_delay(self, attempt_count: int) -> float:
        exponent = max(0, attempt_count - 1)
        return min(
            self._config.retry_max_delay_seconds,
            self._config.retry_base_delay_seconds * (2**exponent),
        )

    async def _release_after_cancellation(
        self,
        job: EvolutionJob,
    ) -> None:
        if job.lease_token is None:
            return
        now = datetime.now(UTC)
        try:
            await asyncio.shield(
                self._store.retry_job(
                    user_id=job.user_id,
                    job_id=job.job_id,
                    lease_token=job.lease_token,
                    now=now,
                    next_attempt_at=now,
                    error_code="worker_cancelled",
                    terminal=False,
                )
            )
        except BaseException:
            # The running lease itself remains the restart-recovery boundary.
            logger.warning(
                "Failed to release cancelled evolution job %s; lease expiry will recover it",
                job.job_id,
                exc_info=True,
            )

    async def execute(self, job: EvolutionJob) -> None:
        """Process, then commit completion/retry under the current claim token."""
        if job.lease_token is None:
            raise ValueError("claimed evolution job requires a lease token")
        processor_task = asyncio.create_task(
            self._processor(job),
            name=f"skill-evolution-processor:{job.job_id}",
        )
        renewal_interval = max(
            0.01,
            self._config.lease_seconds / 3.0,
        )
        current = job
        try:
            while True:
                done, _ = await asyncio.wait(
                    {processor_task},
                    timeout=renewal_interval,
                )
                if processor_task in done:
                    await processor_task
                    break
                now = datetime.now(UTC)
                renewed = await self._store.renew_job_lease(
                    user_id=current.user_id,
                    job_id=current.job_id,
                    lease_token=current.lease_token,
                    now=now,
                    lease_seconds=self._config.lease_seconds,
                )
                if renewed is None:
                    processor_task.cancel()
                    await asyncio.gather(
                        processor_task,
                        return_exceptions=True,
                    )
                    logger.info(
                        "Discarded evolution job result after lease loss (job_id=%s)",
                        current.job_id,
                    )
                    return
                current = renewed

            completed = await self._store.complete_job(
                user_id=current.user_id,
                job_id=current.job_id,
                lease_token=current.lease_token,
                now=datetime.now(UTC),
            )
            if completed is None:
                logger.info(
                    "Discarded evolution job completion after lease loss (job_id=%s)",
                    current.job_id,
                )
        except asyncio.CancelledError:
            processor_task.cancel()
            await asyncio.gather(
                processor_task,
                return_exceptions=True,
            )
            await self._release_after_cancellation(current)
            raise
        except Exception as exc:
            processor_task.cancel()
            await asyncio.gather(
                processor_task,
                return_exceptions=True,
            )
            now = datetime.now(UTC)
            terminal = current.attempt_count >= current.max_attempts
            delay = self._retry_delay(current.attempt_count)
            next_attempt_at = None if terminal else now + timedelta(seconds=delay)
            error_code = type(exc).__name__[:128] or "processing_error"
            updated = await self._store.retry_job(
                user_id=current.user_id,
                job_id=current.job_id,
                lease_token=current.lease_token,
                now=now,
                next_attempt_at=next_attempt_at,
                error_code=error_code,
                terminal=terminal,
            )
            if updated is None:
                logger.info(
                    "Discarded evolution job failure after lease loss (job_id=%s)",
                    current.job_id,
                )
                return
            if terminal:
                logger.error(
                    "Evolution job exhausted attempts (job_id=%s, error=%s)",
                    current.job_id,
                    error_code,
                )
            else:
                logger.warning(
                    "Evolution job failed and will retry (job_id=%s, attempt=%d, error=%s)",
                    current.job_id,
                    current.attempt_count,
                    error_code,
                )
