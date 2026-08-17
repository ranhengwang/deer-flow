from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from deerflow.persistence.base import Base
from deerflow.persistence.skill_evolution.model import (
    SkillEvolutionClusterRow,
    SkillEvolutionEvaluationRow,
    SkillEvolutionEventRow,
    SkillEvolutionProposalRow,
)
from deerflow.skill_evolution.models import (
    ClusterStatus,
    ComplexitySignals,
    EnvironmentSignature,
    EvaluationDecision,
    EvaluationMetrics,
    EvolutionCluster,
    EvolutionEvent,
    EvolutionEventKind,
    OutcomeEvidence,
    OutcomeStatus,
    ProposalOperation,
    ProposalStatus,
    ProposedSkillFile,
    SkillEvaluation,
    SkillProposal,
    SkillUsage,
    TaskEvaluationResult,
    ToolSignature,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
    SkillEvolutionStore,
)
from deerflow.skill_evolution.store.memory import InMemorySkillEvolutionStore
from deerflow.skill_evolution.store.sql import SqlSkillEvolutionStore

_EVOLUTION_TABLES = [
    SkillEvolutionEventRow.__table__,
    SkillEvolutionClusterRow.__table__,
    SkillEvolutionProposalRow.__table__,
    SkillEvolutionEvaluationRow.__table__,
]


async def _make_sql_store(
    database_url: str,
) -> tuple[AsyncEngine, SqlSkillEvolutionStore]:
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=_EVOLUTION_TABLES,
        )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, SqlSkillEvolutionStore(session_factory)


@pytest_asyncio.fixture(params=["memory", "sql"])
async def store(
    request: pytest.FixtureRequest,
    tmp_path,
):
    if request.param == "memory":
        yield InMemorySkillEvolutionStore()
        return

    engine, sql_store = await _make_sql_store(f"sqlite+aiosqlite:///{tmp_path / 'skill-evolution.db'}")
    try:
        yield sql_store
    finally:
        await engine.dispose()


def _event(
    *,
    event_id: str = "event-1",
    run_id: str = "run-1",
    user_id: str = "user-1",
    extractor_version: str = "extractor-v1",
    task_goal: str = "Install a dependency and verify its import.",
) -> EvolutionEvent:
    return EvolutionEvent(
        event_id=event_id,
        run_id=run_id,
        thread_id="thread-1",
        user_id=user_id,
        extractor_version=extractor_version,
        source_snapshot_hash="a" * 64,
        task_input_hash="b" * 64,
        event_kind=EvolutionEventKind.new_skill_evidence,
        task_signature="python-package-install",
        task_goal=task_goal,
        environment=EnvironmentSignature(
            os="macos",
            shell="zsh",
            runtime="python3.12",
        ),
        outcome=OutcomeEvidence(
            status=OutcomeStatus.success,
            confidence=0.95,
            sources=["focused-tests"],
        ),
        complexity=ComplexitySignals(
            tool_calls=7,
            had_recoverable_errors=True,
        ),
        tool_signature=ToolSignature(
            tool_names=["bash", "read_file"],
            error_types=["permission"],
        ),
        skill_usage=SkillUsage(used=False),
        successful_path=["Use the project environment.", "Verify the import."],
        reusable_lessons=["Inspect the environment before installation."],
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


def _cluster(
    *,
    cluster_id: str = "cluster-1",
    user_id: str = "user-1",
    independent_run_count: int = 3,
    status: ClusterStatus = ClusterStatus.ready,
) -> EvolutionCluster:
    return EvolutionCluster(
        cluster_id=cluster_id,
        user_id=user_id,
        event_kind=EvolutionEventKind.new_skill_evidence,
        canonical_signature="python-package-install",
        member_event_ids=["event-1", "event-2", "event-3"],
        independent_run_count=independent_run_count,
        status=status,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
        updated_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


def _proposal(
    *,
    proposal_id: str = "proposal-1",
    user_id: str = "user-1",
    status: ProposalStatus = ProposalStatus.staged,
) -> SkillProposal:
    return SkillProposal(
        proposal_id=proposal_id,
        cluster_id="cluster-1",
        user_id=user_id,
        operation=ProposalOperation.create,
        skill_name="python-package-manager",
        proposed_files=[
            ProposedSkillFile(
                path="SKILL.md",
                content="---\nname: python-package-manager\n---\n",
                executable=False,
            )
        ],
        supporting_event_ids=["event-1", "event-2", "event-3"],
        rationale="Capture a repeated dependency installation workflow.",
        expected_improvements=["Avoid global-environment installation failures."],
        status=status,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


def _evaluation(
    *,
    evaluation_id: str = "evaluation-1",
    user_id: str = "user-1",
) -> SkillEvaluation:
    candidate = TaskEvaluationResult(
        task_id="held-out-1",
        split="held_out",
        condition="candidate_skill",
        success=True,
        metrics=EvaluationMetrics(
            tool_calls=3,
            input_tokens=100,
            output_tokens=20,
            latency_seconds=1.0,
        ),
    )
    return SkillEvaluation(
        evaluation_id=evaluation_id,
        proposal_id="proposal-1",
        user_id=user_id,
        source_replay_results=[],
        held_out_results=[candidate],
        baseline_results=[],
        candidate_results=[candidate],
        regression_results=[],
        safety_results={"static_scan": "allow"},
        quality_score=0.8,
        decision=EvaluationDecision.approve,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_event_upsert_is_idempotent_by_user_run_and_extractor(
    store: SkillEvolutionStore,
) -> None:
    event = _event()

    first = await store.upsert_event(event)
    second = await store.upsert_event(event)

    assert first.created is True
    assert second.created is False
    assert second.value == event
    assert await store.list_events("user-1") == [event]


@pytest.mark.asyncio
async def test_event_upsert_rejects_payload_drift_for_idempotency_key(
    store: SkillEvolutionStore,
) -> None:
    await store.upsert_event(_event())

    with pytest.raises(EvolutionStoreConflict, match="idempotency"):
        await store.upsert_event(
            _event(
                event_id="event-2",
                task_goal="A changed extraction for the same run.",
            )
        )


@pytest.mark.asyncio
async def test_same_run_can_be_reprocessed_by_new_extractor_version(
    store: SkillEvolutionStore,
) -> None:
    first = _event()
    reprocessed = _event(
        event_id="event-2",
        extractor_version="extractor-v2",
    )

    assert (await store.upsert_event(first)).created is True
    assert (await store.upsert_event(reprocessed)).created is True
    assert await store.list_events("user-1") == [first, reprocessed]


@pytest.mark.asyncio
async def test_event_id_collision_is_rejected(
    store: SkillEvolutionStore,
) -> None:
    await store.upsert_event(_event())

    with pytest.raises(EvolutionStoreConflict, match="event ID"):
        await store.upsert_event(
            _event(
                run_id="run-2",
                task_goal="Different run reusing the event ID.",
            )
        )


@pytest.mark.asyncio
async def test_same_run_and_extractor_are_isolated_by_user(
    store: SkillEvolutionStore,
) -> None:
    alice = _event(user_id="alice")
    bob = _event(user_id="bob")

    assert (await store.upsert_event(alice)).created is True
    assert (await store.upsert_event(bob)).created is True
    assert await store.list_events("alice") == [alice]
    assert await store.list_events("bob") == [bob]


@pytest.mark.asyncio
async def test_ready_cluster_query_requires_status_and_distinct_run_count(
    store: SkillEvolutionStore,
) -> None:
    ready = _cluster()
    collecting = _cluster(
        cluster_id="cluster-2",
        status=ClusterStatus.collecting,
    )
    insufficient = _cluster(
        cluster_id="cluster-3",
        independent_run_count=2,
    )

    await store.put_cluster(ready)
    await store.put_cluster(collecting)
    await store.put_cluster(insufficient)

    assert await store.list_ready_clusters(
        "user-1",
        min_distinct_runs=3,
    ) == [ready]


@pytest.mark.asyncio
async def test_cluster_put_is_idempotent_but_rejects_drift(
    store: SkillEvolutionStore,
) -> None:
    cluster = _cluster()

    assert (await store.put_cluster(cluster)).created is True
    assert (await store.put_cluster(cluster)).created is False

    with pytest.raises(EvolutionStoreConflict, match="cluster ID"):
        await store.put_cluster(_cluster(status=ClusterStatus.distilled))


@pytest.mark.asyncio
async def test_proposal_transition_uses_compare_and_swap(
    store: SkillEvolutionStore,
) -> None:
    proposal = _proposal()
    await store.put_proposal(proposal)

    updated = await store.transition_proposal(
        user_id="user-1",
        proposal_id="proposal-1",
        expected_status=ProposalStatus.staged,
        new_status=ProposalStatus.validating,
    )

    assert updated.status is ProposalStatus.validating
    assert (await store.get_proposal("user-1", "proposal-1")) == updated

    with pytest.raises(EvolutionStoreConflict, match="expected status"):
        await store.transition_proposal(
            user_id="user-1",
            proposal_id="proposal-1",
            expected_status=ProposalStatus.staged,
            new_status=ProposalStatus.approved,
        )


@pytest.mark.asyncio
async def test_evaluation_put_and_get_are_user_scoped(
    store: SkillEvolutionStore,
) -> None:
    evaluation = _evaluation()

    assert (await store.put_evaluation(evaluation)).created is True
    assert (await store.put_evaluation(evaluation)).created is False
    assert (await store.get_evaluation("user-1", "evaluation-1")) == evaluation
    assert await store.get_evaluation("other-user", "evaluation-1") is None


@pytest.mark.asyncio
async def test_sql_events_survive_store_recreation(tmp_path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'durable-skill-evolution.db'}"
    first_engine, first_store = await _make_sql_store(database_url)
    event = _event()
    try:
        await first_store.upsert_event(event)
    finally:
        await first_engine.dispose()

    second_engine, second_store = await _make_sql_store(database_url)
    try:
        assert await second_store.get_event("user-1", "event-1") == event
    finally:
        await second_engine.dispose()
