"""Bounded, restart-safe coordination for background Skill evolution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
import uuid
from datetime import UTC, datetime

from deerflow.config.skill_evolution_config import (
    SkillEvolutionCoordinatorConfig,
)
from deerflow.skill_evolution.models import (
    EvolutionJob,
    EvolutionJobStatus,
    EvolutionTraceSnapshot,
)
from deerflow.skill_evolution.observability import (
    EvolutionObservability,
    get_evolution_observability,
    safe_set_evolution_gauge,
)
from deerflow.skill_evolution.store.base import (
    PutResult,
    SkillEvolutionStore,
)
from deerflow.skill_evolution.worker import (
    EvolutionJobProcessor,
    EvolutionJobWorker,
)

logger = logging.getLogger(__name__)

EVOLUTION_PIPELINE_VERSION = "skill-evolution-pipeline-v1"


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_evolution_job(
    *,
    user_id: str,
    thread_id: str,
    run_id: str,
    snapshot_hash: str,
    max_attempts: int,
    now: datetime | None = None,
    pipeline_version: str = EVOLUTION_PIPELINE_VERSION,
) -> EvolutionJob:
    """Build the deterministic durable identity for one snapshot pipeline."""
    source = {
        "user_id": user_id,
        "run_id": run_id,
        "snapshot_hash": snapshot_hash,
        "pipeline_version": pipeline_version,
    }
    idempotency_key = hashlib.sha256(_stable_json(source).encode("utf-8")).hexdigest()
    created_at = now or datetime.now(UTC)
    return EvolutionJob(
        job_id=f"job-{idempotency_key[:32]}",
        idempotency_key=idempotency_key,
        user_id=user_id,
        thread_id=thread_id,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        pipeline_version=pipeline_version,
        max_attempts=max_attempts,
        next_attempt_at=created_at,
        created_at=created_at,
        updated_at=created_at,
    )


class EvolutionCoordinator:
    """Persist jobs first, then use a bounded queue only as a wake-up hint."""

    def __init__(
        self,
        *,
        store: SkillEvolutionStore,
        processor: EvolutionJobProcessor,
        config: SkillEvolutionCoordinatorConfig,
        observability: EvolutionObservability | None = None,
    ) -> None:
        self._store = store
        self._config = config
        self._lease_owner = (f"{socket.gethostname()}:{uuid.uuid4().hex}")[:128]
        self._worker = EvolutionJobWorker(
            store=store,
            processor=processor,
            config=config,
        )
        self._wakeup: asyncio.Queue[None] = asyncio.Queue(maxsize=config.queue_capacity)
        self._stop = asyncio.Event()
        self._poller: asyncio.Task[None] | None = None
        self._active: set[asyncio.Task[None]] = set()
        self._accepting_wakeups = True
        self._observability = observability or get_evolution_observability()
        safe_set_evolution_gauge(
            self._observability,
            "queue_depth",
            0,
        )
        safe_set_evolution_gauge(
            self._observability,
            "active_jobs",
            0,
        )

    @property
    def wakeup_queue_size(self) -> int:
        return self._wakeup.qsize()

    @property
    def active_job_count(self) -> int:
        return len(self._active)

    def _signal(self) -> None:
        if not self._accepting_wakeups:
            return
        try:
            self._wakeup.put_nowait(None)
        except asyncio.QueueFull:
            # SQL remains authoritative; the periodic poll will find the row.
            pass

    async def enqueue_snapshot(
        self,
        snapshot: EvolutionTraceSnapshot,
    ) -> PutResult[EvolutionJob]:
        """Durably enqueue a snapshot before publishing any local wake-up."""
        job = build_evolution_job(
            user_id=snapshot.user_id,
            thread_id=snapshot.thread_id,
            run_id=snapshot.run_id,
            snapshot_hash=snapshot.snapshot_hash,
            max_attempts=self._config.max_attempts,
        )
        result = await self._store.enqueue_job(job)
        self._signal()
        return result

    async def start(self) -> None:
        if self._poller is not None:
            return
        self._accepting_wakeups = True
        self._stop.clear()
        self._poller = asyncio.create_task(
            self._run_loop(),
            name="deerflow-skill-evolution-coordinator",
        )

    async def stop(self) -> bool:
        """Stop claiming, then drain active work within the configured bound."""
        self._accepting_wakeups = False
        self._stop.set()
        poller = self._poller
        self._poller = None
        if poller is not None:
            poller.cancel()
            await asyncio.gather(
                poller,
                return_exceptions=True,
            )

        active = set(self._active)
        if not active:
            return True
        done, pending = await asyncio.wait(
            active,
            timeout=self._config.shutdown_timeout_seconds,
        )
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(
                *pending,
                return_exceptions=True,
            )
        return not pending

    async def _dispatch_due(self) -> None:
        capacity = self._config.max_concurrent_jobs - len(self._active)
        if capacity <= 0 or self._stop.is_set():
            return
        claimed = await self._store.claim_jobs(
            now=datetime.now(UTC),
            lease_owner=self._lease_owner,
            lease_seconds=self._config.lease_seconds,
            limit=capacity,
        )
        for job in claimed:
            task = asyncio.create_task(
                self._worker.execute(job),
                name=f"skill-evolution-job:{job.job_id}",
            )
            self._active.add(task)
            task.add_done_callback(self._job_done)
        safe_set_evolution_gauge(
            self._observability,
            "active_jobs",
            len(self._active),
        )
        await self._refresh_queue_depth()

    async def _refresh_queue_depth(self) -> None:
        depth = await self._store.count_jobs(
            statuses=(
                EvolutionJobStatus.pending,
                EvolutionJobStatus.retry,
            )
        )
        safe_set_evolution_gauge(
            self._observability,
            "queue_depth",
            depth,
        )

    def _job_done(self, task: asyncio.Task[None]) -> None:
        self._active.discard(task)
        safe_set_evolution_gauge(
            self._observability,
            "active_jobs",
            len(self._active),
        )
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "Evolution job worker failed outside its processing boundary",
                    exc_info=(
                        type(error),
                        error,
                        error.__traceback__,
                    ),
                )
        self._signal()

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._refresh_queue_depth()
                await self._dispatch_due()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Evolution coordinator poll failed; retrying")
            try:
                await asyncio.wait_for(
                    self._wakeup.get(),
                    timeout=self._config.poll_interval_seconds,
                )
            except TimeoutError:
                continue
