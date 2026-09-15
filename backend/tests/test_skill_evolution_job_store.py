from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from deerflow.persistence.base import Base
from deerflow.persistence.skill_evolution.model import (
    SkillEvolutionJobRow,
)
from deerflow.skill_evolution.coordinator import build_evolution_job
from deerflow.skill_evolution.models import EvolutionJobStatus
from deerflow.skill_evolution.store.memory import (
    InMemorySkillEvolutionStore,
)
from deerflow.skill_evolution.store.sql import SqlSkillEvolutionStore


async def _make_sql_store(database_url: str):
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[SkillEvolutionJobRow.__table__],
        )
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    return engine, SqlSkillEvolutionStore(session_factory)


@pytest_asyncio.fixture(params=["memory", "sql"])
async def store(request: pytest.FixtureRequest, tmp_path):
    if request.param == "memory":
        yield InMemorySkillEvolutionStore()
        return
    engine, sql_store = await _make_sql_store(f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}")
    try:
        yield sql_store
    finally:
        await engine.dispose()


def _job(*, run_id: str = "run-1", snapshot_hash: str = "a" * 64):
    return build_evolution_job(
        user_id="user-1",
        thread_id="thread-1",
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        max_attempts=3,
        now=datetime(2026, 8, 17, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_enqueue_job_is_idempotent_by_source_and_pipeline(store) -> None:
    job = _job()

    first = await store.enqueue_job(job)
    second = await store.enqueue_job(job)

    assert first.created is True
    assert second.created is False
    assert second.value == first.value
    assert await store.get_job(job.user_id, job.job_id) == job


@pytest.mark.asyncio
async def test_job_count_is_filtered_by_status(store) -> None:
    first = _job()
    second = _job(
        run_id="run-2",
        snapshot_hash="b" * 64,
    )
    await store.enqueue_job(first)
    await store.enqueue_job(second)

    assert (
        await store.count_jobs(
            statuses=(EvolutionJobStatus.pending,),
        )
        == 2
    )
    claimed = (
        await store.claim_jobs(
            now=datetime(2026, 8, 17, tzinfo=UTC),
            lease_owner="worker-1",
            lease_seconds=30.0,
            limit=1,
        )
    )[0]
    assert (
        await store.count_jobs(
            statuses=(EvolutionJobStatus.pending,),
        )
        == 1
    )
    assert (
        await store.count_jobs(
            statuses=(EvolutionJobStatus.running,),
        )
        == 1
    )
    assert (
        await store.count_jobs(
            statuses=(
                EvolutionJobStatus.pending,
                EvolutionJobStatus.running,
            ),
        )
        == 2
    )
    await store.complete_job(
        user_id=claimed.user_id,
        job_id=claimed.job_id,
        lease_token=claimed.lease_token,
        now=datetime(2026, 8, 17, 0, 0, 1, tzinfo=UTC),
    )
    assert (
        await store.count_jobs(
            statuses=(EvolutionJobStatus.completed,),
        )
        == 1
    )


@pytest.mark.asyncio
async def test_claim_is_exclusive_until_lease_expiry(store) -> None:
    now = datetime(2026, 8, 17, tzinfo=UTC)
    job = _job()
    await store.enqueue_job(job)

    first = await store.claim_jobs(
        now=now,
        lease_owner="worker-1",
        lease_seconds=30.0,
        limit=1,
    )
    blocked = await store.claim_jobs(
        now=now + timedelta(seconds=10),
        lease_owner="worker-2",
        lease_seconds=30.0,
        limit=1,
    )
    recovered = await store.claim_jobs(
        now=now + timedelta(seconds=31),
        lease_owner="worker-2",
        lease_seconds=30.0,
        limit=1,
    )

    assert len(first) == 1
    assert first[0].status is EvolutionJobStatus.running
    assert first[0].attempt_count == 1
    assert first[0].lease_token is not None
    assert blocked == []
    assert len(recovered) == 1
    assert recovered[0].attempt_count == 2
    assert recovered[0].lease_owner == "worker-2"
    assert recovered[0].lease_token != first[0].lease_token

    stale = await store.complete_job(
        user_id=job.user_id,
        job_id=job.job_id,
        lease_token=first[0].lease_token,
        now=now + timedelta(seconds=32),
    )
    completed = await store.complete_job(
        user_id=job.user_id,
        job_id=job.job_id,
        lease_token=recovered[0].lease_token,
        now=now + timedelta(seconds=32),
    )
    assert stale is None
    assert completed is not None
    assert completed.status is EvolutionJobStatus.completed
    assert completed.completed_at == now + timedelta(seconds=32)


@pytest.mark.asyncio
async def test_lease_renewal_requires_current_claim_token(store) -> None:
    now = datetime(2026, 8, 17, tzinfo=UTC)
    job = _job()
    await store.enqueue_job(job)
    claimed = (
        await store.claim_jobs(
            now=now,
            lease_owner="worker-1",
            lease_seconds=30.0,
            limit=1,
        )
    )[0]

    assert (
        await store.renew_job_lease(
            user_id=job.user_id,
            job_id=job.job_id,
            lease_token="wrong-token",
            now=now + timedelta(seconds=5),
            lease_seconds=30.0,
        )
        is None
    )
    renewed = await store.renew_job_lease(
        user_id=job.user_id,
        job_id=job.job_id,
        lease_token=claimed.lease_token,
        now=now + timedelta(seconds=5),
        lease_seconds=30.0,
    )
    assert renewed is not None
    assert renewed.lease_expires_at == now + timedelta(seconds=35)


@pytest.mark.asyncio
async def test_retry_is_not_claimable_before_next_attempt(store) -> None:
    now = datetime(2026, 8, 17, tzinfo=UTC)
    job = _job()
    await store.enqueue_job(job)
    claimed = (
        await store.claim_jobs(
            now=now,
            lease_owner="worker-1",
            lease_seconds=30.0,
            limit=1,
        )
    )[0]
    due_at = now + timedelta(seconds=20)

    retrying = await store.retry_job(
        user_id=job.user_id,
        job_id=job.job_id,
        lease_token=claimed.lease_token,
        now=now + timedelta(seconds=1),
        next_attempt_at=due_at,
        error_code="ConnectionError",
        terminal=False,
    )

    assert retrying is not None
    assert retrying.status is EvolutionJobStatus.retry
    assert retrying.last_error_code == "ConnectionError"
    assert (
        await store.claim_jobs(
            now=due_at - timedelta(microseconds=1),
            lease_owner="worker-2",
            lease_seconds=30.0,
            limit=1,
        )
        == []
    )
    assert (
        len(
            await store.claim_jobs(
                now=due_at,
                lease_owner="worker-2",
                lease_seconds=30.0,
                limit=1,
            )
        )
        == 1
    )


@pytest.mark.asyncio
async def test_terminal_failure_is_never_claimed_again(store) -> None:
    now = datetime(2026, 8, 17, tzinfo=UTC)
    job = _job()
    await store.enqueue_job(job)
    claimed = (
        await store.claim_jobs(
            now=now,
            lease_owner="worker-1",
            lease_seconds=30.0,
            limit=1,
        )
    )[0]

    dead = await store.retry_job(
        user_id=job.user_id,
        job_id=job.job_id,
        lease_token=claimed.lease_token,
        now=now + timedelta(seconds=1),
        next_attempt_at=None,
        error_code="ValueError",
        terminal=True,
    )

    assert dead is not None
    assert dead.status is EvolutionJobStatus.dead
    assert dead.completed_at == now + timedelta(seconds=1)
    assert (
        await store.claim_jobs(
            now=now + timedelta(days=1),
            lease_owner="worker-2",
            lease_seconds=30.0,
            limit=1,
        )
        == []
    )


@pytest.mark.asyncio
async def test_sql_job_survives_store_recreation(tmp_path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'durable-jobs.db'}"
    first_engine, first_store = await _make_sql_store(database_url)
    job = _job()
    try:
        await first_store.enqueue_job(job)
    finally:
        await first_engine.dispose()

    second_engine, second_store = await _make_sql_store(database_url)
    try:
        assert await second_store.get_job(job.user_id, job.job_id) == job
    finally:
        await second_engine.dispose()
