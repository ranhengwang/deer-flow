from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from deerflow.config.skill_evolution_config import (
    SkillEvolutionCoordinatorConfig,
)
from deerflow.skill_evolution.coordinator import EvolutionCoordinator
from deerflow.skill_evolution.models import (
    EvolutionJobStatus,
    EvolutionTraceSnapshot,
)
from deerflow.skill_evolution.observability import (
    EvolutionObservability,
)
from deerflow.skill_evolution.store.memory import (
    InMemorySkillEvolutionStore,
)


def _snapshot(
    *,
    run_id: str,
    snapshot_hash: str,
) -> EvolutionTraceSnapshot:
    return EvolutionTraceSnapshot.model_validate(
        {
            "snapshot_hash": snapshot_hash,
            "run_id": run_id,
            "thread_id": "thread-1",
            "user_id": "user-1",
            "run_status": "success",
            "task_input": "Run focused tests.",
            "final_answer": "Done.",
            "environment": {
                "os": "macos",
                "shell": "zsh",
                "runtime": "python3.12",
            },
            "source_event_count": 0,
            "included_event_count": 0,
            "truncated": False,
            "created_at": datetime(2026, 8, 17, tzinfo=UTC),
        }
    )


def _config(**overrides) -> SkillEvolutionCoordinatorConfig:
    return SkillEvolutionCoordinatorConfig(
        queue_capacity=overrides.pop("queue_capacity", 2),
        max_concurrent_jobs=overrides.pop("max_concurrent_jobs", 1),
        poll_interval_seconds=overrides.pop("poll_interval_seconds", 0.01),
        lease_seconds=overrides.pop("lease_seconds", 1.0),
        max_attempts=overrides.pop("max_attempts", 3),
        retry_base_delay_seconds=overrides.pop(
            "retry_base_delay_seconds",
            0.0,
        ),
        retry_max_delay_seconds=overrides.pop(
            "retry_max_delay_seconds",
            0.0,
        ),
        shutdown_timeout_seconds=overrides.pop(
            "shutdown_timeout_seconds",
            1.0,
        ),
        **overrides,
    )


async def _wait_for_status(
    store: InMemorySkillEvolutionStore,
    *,
    job_id: str,
    status: EvolutionJobStatus,
    timeout: float = 1.0,
):
    async with asyncio.timeout(timeout):
        while True:
            job = await store.get_job("user-1", job_id)
            if job is not None and job.status is status:
                return job
            await asyncio.sleep(0.005)


@pytest.mark.anyio
async def test_queue_full_keeps_every_job_durable() -> None:
    store = InMemorySkillEvolutionStore()

    async def processor(_job) -> None:
        raise AssertionError("coordinator was not started")

    coordinator = EvolutionCoordinator(
        store=store,
        processor=processor,
        config=_config(queue_capacity=1),
    )

    first = await coordinator.enqueue_snapshot(_snapshot(run_id="run-1", snapshot_hash="a" * 64))
    second = await coordinator.enqueue_snapshot(_snapshot(run_id="run-2", snapshot_hash="b" * 64))

    assert first.created is True
    assert second.created is True
    assert coordinator.wakeup_queue_size == 1
    assert (await store.get_job("user-1", first.value.job_id)) == first.value
    assert (await store.get_job("user-1", second.value.job_id)) == second.value


@pytest.mark.anyio
async def test_start_recovers_pending_job_and_duplicate_enqueue_runs_once() -> None:
    store = InMemorySkillEvolutionStore()
    processed: list[str] = []
    completed = asyncio.Event()

    async def processor(job) -> None:
        processed.append(job.job_id)
        completed.set()

    first = EvolutionCoordinator(
        store=store,
        processor=processor,
        config=_config(),
    )
    snapshot = _snapshot(run_id="run-1", snapshot_hash="a" * 64)
    created = await first.enqueue_snapshot(snapshot)
    duplicate = await first.enqueue_snapshot(snapshot)
    assert created.created is True
    assert duplicate.created is False
    assert duplicate.value.job_id == created.value.job_id

    recovered = EvolutionCoordinator(
        store=store,
        processor=processor,
        config=_config(),
    )
    await recovered.start()
    await asyncio.wait_for(completed.wait(), timeout=1.0)
    await _wait_for_status(
        store,
        job_id=created.value.job_id,
        status=EvolutionJobStatus.completed,
    )
    assert await recovered.stop() is True
    assert processed == [created.value.job_id]


@pytest.mark.anyio
async def test_processor_failure_retries_then_completes() -> None:
    store = InMemorySkillEvolutionStore()
    attempts = 0

    async def processor(_job) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("provider unavailable")

    coordinator = EvolutionCoordinator(
        store=store,
        processor=processor,
        config=_config(),
    )
    result = await coordinator.enqueue_snapshot(_snapshot(run_id="run-1", snapshot_hash="a" * 64))

    await coordinator.start()
    completed = await _wait_for_status(
        store,
        job_id=result.value.job_id,
        status=EvolutionJobStatus.completed,
    )
    assert await coordinator.stop() is True
    assert attempts == 2
    assert completed.attempt_count == 2
    assert completed.last_error_code is None


@pytest.mark.anyio
async def test_shutdown_drains_active_job_within_timeout() -> None:
    store = InMemorySkillEvolutionStore()
    started = asyncio.Event()
    release = asyncio.Event()

    async def processor(_job) -> None:
        started.set()
        await release.wait()

    coordinator = EvolutionCoordinator(
        store=store,
        processor=processor,
        config=_config(shutdown_timeout_seconds=1.0),
    )
    result = await coordinator.enqueue_snapshot(_snapshot(run_id="run-1", snapshot_hash="a" * 64))
    await coordinator.start()
    await asyncio.wait_for(started.wait(), timeout=1.0)

    stop_task = asyncio.create_task(coordinator.stop())
    await asyncio.sleep(0)
    release.set()

    assert await stop_task is True
    job = await store.get_job("user-1", result.value.job_id)
    assert job is not None
    assert job.status is EvolutionJobStatus.completed


@pytest.mark.anyio
async def test_shutdown_timeout_leaves_job_restart_recoverable() -> None:
    store = InMemorySkillEvolutionStore()
    started = asyncio.Event()
    recovered = asyncio.Event()

    async def blocked_processor(_job) -> None:
        started.set()
        await asyncio.Event().wait()

    first = EvolutionCoordinator(
        store=store,
        processor=blocked_processor,
        config=_config(
            lease_seconds=0.05,
            shutdown_timeout_seconds=0.01,
        ),
    )
    result = await first.enqueue_snapshot(_snapshot(run_id="run-1", snapshot_hash="a" * 64))
    await first.start()
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert await first.stop() is False
    abandoned = await store.get_job("user-1", result.value.job_id)
    assert abandoned is not None
    assert abandoned.status in {
        EvolutionJobStatus.retry,
        EvolutionJobStatus.running,
    }

    async def recovery_processor(_job) -> None:
        recovered.set()

    second = EvolutionCoordinator(
        store=store,
        processor=recovery_processor,
        config=_config(
            lease_seconds=0.05,
            shutdown_timeout_seconds=1.0,
        ),
    )
    await asyncio.sleep(0.06)
    await second.start()
    await asyncio.wait_for(recovered.wait(), timeout=1.0)
    await _wait_for_status(
        store,
        job_id=result.value.job_id,
        status=EvolutionJobStatus.completed,
    )
    assert await second.stop() is True


@pytest.mark.anyio
async def test_exhausted_attempts_move_job_to_dead_state() -> None:
    store = InMemorySkillEvolutionStore()

    async def processor(_job) -> None:
        raise RuntimeError("permanent failure")

    coordinator = EvolutionCoordinator(
        store=store,
        processor=processor,
        config=_config(max_attempts=2),
    )
    result = await coordinator.enqueue_snapshot(_snapshot(run_id="run-1", snapshot_hash="a" * 64))
    await coordinator.start()
    dead = await _wait_for_status(
        store,
        job_id=result.value.job_id,
        status=EvolutionJobStatus.dead,
    )
    assert await coordinator.stop() is True
    assert dead.attempt_count == 2
    assert dead.last_error_code == "RuntimeError"
    assert dead.next_attempt_at is None


@pytest.mark.anyio
async def test_coordinator_reports_durable_backlog_and_active_jobs() -> None:
    store = InMemorySkillEvolutionStore()
    observer = EvolutionObservability()
    started = asyncio.Event()
    release = asyncio.Event()

    async def processor(_job) -> None:
        started.set()
        await release.wait()

    coordinator = EvolutionCoordinator(
        store=store,
        processor=processor,
        config=_config(max_concurrent_jobs=1),
        observability=observer,
    )
    await coordinator.enqueue_snapshot(_snapshot(run_id="run-1", snapshot_hash="a" * 64))
    await coordinator.enqueue_snapshot(_snapshot(run_id="run-2", snapshot_hash="b" * 64))

    await coordinator.start()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    async with asyncio.timeout(1.0):
        while True:
            metrics = observer.snapshot()
            if metrics.queue_depth == 1 and metrics.active_jobs == 1:
                break
            await asyncio.sleep(0.005)

    release.set()
    async with asyncio.timeout(1.0):
        while True:
            metrics = observer.snapshot()
            if metrics.queue_depth == 0 and metrics.active_jobs == 0:
                break
            await asyncio.sleep(0.005)
    assert await coordinator.stop() is True
