from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta

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
    SkillEvolutionJobRow,
    SkillEvolutionProposalRow,
    SkillEvolutionPublicationRow,
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
    ProposalStatusSource,
    ProposalStatusTransition,
    ProposedSkillFile,
    PublicationStatus,
    SkillEvaluation,
    SkillPackageSnapshot,
    SkillPackageSnapshotFile,
    SkillProposal,
    SkillPublication,
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
    SkillEvolutionPublicationRow.__table__,
    SkillEvolutionJobRow.__table__,
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


def _package_snapshot(
    *,
    exists: bool,
    content: bytes = b"",
) -> SkillPackageSnapshot:
    files = []
    skill_hash = None
    if exists:
        skill_hash = hashlib.sha256(content).hexdigest()
        files = [
            SkillPackageSnapshotFile(
                path="SKILL.md",
                content_base64=base64.b64encode(content).decode("ascii"),
                content_hash=skill_hash,
                size_bytes=len(content),
                executable=False,
            )
        ]
    package_hash = hashlib.sha256()
    if files:
        record = b"\0".join(
            [
                b"SKILL.md",
                str(len(content)).encode("ascii"),
                skill_hash.encode("ascii"),
                b"0",
            ]
        )
        package_hash.update(len(record).to_bytes(8, "big"))
        package_hash.update(record)
    return SkillPackageSnapshot(
        snapshot_hash=package_hash.hexdigest(),
        user_id="user-1",
        skill_name="python-package-manager",
        exists=exists,
        skill_md_hash=skill_hash,
        files=files,
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
    )


def _publication(
    *,
    publication_id: str = "publication-proposal-1",
    proposal_id: str = "proposal-1",
    evaluation_id: str | None = "evaluation-1",
) -> SkillPublication:
    base = _package_snapshot(exists=False)
    candidate = _package_snapshot(
        exists=True,
        content=b"---\nname: python-package-manager\n---\n",
    )
    return SkillPublication(
        publication_id=publication_id,
        user_id="user-1",
        proposal_id=proposal_id,
        evaluation_id=evaluation_id,
        skill_name="python-package-manager",
        operation=ProposalOperation.create,
        status=PublicationStatus.preparing,
        base_snapshot=base,
        candidate_snapshot=candidate,
        base_skill_hash=None,
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_direct_publication_round_trips_without_evaluation(
    store: SkillEvolutionStore,
) -> None:
    publication = _publication(
        publication_id="publication-direct-1",
        proposal_id="proposal-direct-1",
        evaluation_id=None,
    )

    result = await store.put_publication(publication)

    assert result.created is True
    assert result.value.evaluation_id is None
    assert (
        await store.get_publication(
            publication.user_id,
            publication.publication_id,
        )
        == publication
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
    transition = ProposalStatusTransition(
        from_status=ProposalStatus.staged,
        to_status=ProposalStatus.validating,
        source=ProposalStatusSource.evaluator,
        reason_code="evaluation_started",
        reason="Evaluation evaluation-1 started.",
        occurred_at=datetime(2026, 8, 17, tzinfo=UTC),
        evaluation_id="evaluation-1",
    )

    updated = await store.transition_proposal(
        user_id="user-1",
        proposal_id="proposal-1",
        expected_status=ProposalStatus.staged,
        new_status=ProposalStatus.validating,
        transition=transition,
    )

    assert updated.status is ProposalStatus.validating
    assert updated.status_history == [transition]
    assert (await store.get_proposal("user-1", "proposal-1")) == updated
    approval = ProposalStatusTransition(
        from_status=ProposalStatus.validating,
        to_status=ProposalStatus.approved,
        source=ProposalStatusSource.approval_policy,
        reason_code="auto_approval_eligible",
        reason="All automatic approval gates passed.",
        occurred_at=datetime(2026, 8, 17, 1, tzinfo=UTC),
        evaluation_id="evaluation-1",
        policy_version="skill-approval-v1",
    )
    approved = await store.transition_proposal(
        user_id="user-1",
        proposal_id="proposal-1",
        expected_status=ProposalStatus.validating,
        new_status=ProposalStatus.approved,
        transition=approval,
    )

    assert approved.status is ProposalStatus.approved
    assert approved.status_history == [
        transition,
        approval,
    ]
    assert (await store.get_proposal("user-1", "proposal-1")) == approved

    with pytest.raises(EvolutionStoreConflict, match="expected status"):
        await store.transition_proposal(
            user_id="user-1",
            proposal_id="proposal-1",
            expected_status=ProposalStatus.staged,
            new_status=ProposalStatus.approved,
        )


@pytest.mark.asyncio
async def test_proposal_lookup_by_cluster_is_user_scoped(
    store: SkillEvolutionStore,
) -> None:
    proposal = _proposal()
    await store.put_proposal(proposal)

    assert (
        await store.get_proposal_by_cluster(
            proposal.user_id,
            proposal.cluster_id,
        )
        == proposal
    )
    assert (
        await store.get_proposal_by_cluster(
            "other-user",
            proposal.cluster_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_legacy_terminal_proposal_without_status_history_still_loads(
    store: SkillEvolutionStore,
) -> None:
    legacy = _proposal(
        status=ProposalStatus.rejected,
    )

    await store.put_proposal(legacy)

    restored = await store.get_proposal(
        legacy.user_id,
        legacy.proposal_id,
    )
    assert restored == legacy
    assert restored is not None
    assert restored.status_history == []


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
async def test_publication_put_and_transition_are_idempotent_and_user_scoped(
    store: SkillEvolutionStore,
) -> None:
    preparing = _publication()

    assert (await store.put_publication(preparing)).created is True
    assert (await store.put_publication(preparing)).created is False
    assert await store.get_publication("other-user", preparing.publication_id) is None

    published = SkillPublication.model_validate(
        {
            **preparing.model_dump(mode="python"),
            "status": PublicationStatus.published,
            "published_snapshot": preparing.candidate_snapshot,
            "published_skill_hash": preparing.candidate_snapshot.skill_md_hash,
            "published_at": datetime(2026, 8, 17, 1, tzinfo=UTC),
        }
    )
    stored = await store.transition_publication(
        published,
        expected_status=PublicationStatus.preparing,
    )

    assert stored == published
    assert await store.get_publication("user-1", preparing.publication_id) == published
    with pytest.raises(EvolutionStoreConflict, match="expected status"):
        await store.transition_publication(
            published,
            expected_status=PublicationStatus.preparing,
        )


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


@pytest.mark.asyncio
async def test_sql_publication_snapshots_survive_store_recreation(
    tmp_path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'durable-skill-publication.db'}"
    first_engine, first_store = await _make_sql_store(database_url)
    publication = _publication()
    try:
        await first_store.put_publication(publication)
    finally:
        await first_engine.dispose()

    second_engine, second_store = await _make_sql_store(database_url)
    try:
        assert (
            await second_store.get_publication(
                publication.user_id,
                publication.publication_id,
            )
            == publication
        )
    finally:
        await second_engine.dispose()


@pytest.mark.asyncio
async def test_read_pages_are_owner_scoped_filtered_and_stably_paginated(
    store: SkillEvolutionStore,
) -> None:
    created = datetime(2026, 8, 16, tzinfo=UTC)
    first_event = _event(
        event_id="event-1",
        run_id="run-1",
    )
    second_event = EvolutionEvent.model_validate(
        {
            **first_event.model_dump(mode="python"),
            "event_id": "event-2",
            "run_id": "run-2",
            "task_input_hash": "c" * 64,
            "created_at": created + timedelta(minutes=1),
        }
    )
    third_event = EvolutionEvent.model_validate(
        {
            **first_event.model_dump(mode="python"),
            "event_id": "event-3",
            "run_id": "run-3",
            "task_input_hash": "d" * 64,
            "created_at": created + timedelta(minutes=2),
        }
    )
    other_event = _event(
        event_id="event-other",
        run_id="run-other",
        user_id="other-user",
    )
    for event in (
        first_event,
        second_event,
        third_event,
        other_event,
    ):
        await store.upsert_event(event)

    assert [
        event.event_id
        for event in await store.list_event_page(
            "user-1",
            limit=2,
            offset=0,
        )
    ] == ["event-3", "event-2"]
    assert [
        event.event_id
        for event in await store.list_event_page(
            "user-1",
            limit=2,
            offset=2,
        )
    ] == ["event-1"]
    assert (
        await store.list_event_page(
            "user-1",
            limit=5,
            offset=0,
            event_kind=EvolutionEventKind.skill_patch_evidence,
        )
        == []
    )

    collecting = _cluster(
        cluster_id="cluster-collecting",
        status=ClusterStatus.collecting,
    )
    ready = EvolutionCluster.model_validate(
        {
            **_cluster(cluster_id="cluster-ready").model_dump(mode="python"),
            "updated_at": created + timedelta(minutes=1),
        }
    )
    await store.put_cluster(collecting)
    await store.put_cluster(ready)
    assert [
        cluster.cluster_id
        for cluster in await store.list_cluster_page(
            "user-1",
            limit=5,
            offset=0,
            status=ClusterStatus.ready,
        )
    ] == ["cluster-ready"]

    first_proposal = _proposal(proposal_id="proposal-1")
    second_proposal = SkillProposal.model_validate(
        {
            **first_proposal.model_dump(mode="python"),
            "proposal_id": "proposal-2",
            "cluster_id": "cluster-ready",
            "created_at": created + timedelta(minutes=1),
        }
    )
    await store.put_proposal(first_proposal)
    await store.put_proposal(second_proposal)
    assert [
        proposal.proposal_id
        for proposal in await store.list_proposal_page(
            "user-1",
            limit=5,
            offset=0,
            status=ProposalStatus.staged,
        )
    ] == ["proposal-2", "proposal-1"]

    first_evaluation = _evaluation(evaluation_id="evaluation-1")
    second_evaluation = SkillEvaluation.model_validate(
        {
            **first_evaluation.model_dump(mode="python"),
            "evaluation_id": "evaluation-2",
            "proposal_id": "proposal-2",
            "decision": EvaluationDecision.manual_review,
            "created_at": created + timedelta(minutes=2),
        }
    )
    await store.put_evaluation(first_evaluation)
    await store.put_evaluation(second_evaluation)
    assert [
        evaluation.evaluation_id
        for evaluation in await store.list_evaluation_page(
            "user-1",
            limit=5,
            offset=0,
            decision=EvaluationDecision.manual_review,
        )
    ] == ["evaluation-2"]

    first_publication = _publication()
    second_publication = SkillPublication.model_validate(
        {
            **first_publication.model_dump(mode="python"),
            "publication_id": "publication-proposal-2",
            "proposal_id": "proposal-2",
            "evaluation_id": "evaluation-2",
            "created_at": (first_publication.created_at + timedelta(minutes=3)),
        }
    )
    await store.put_publication(first_publication)
    await store.put_publication(second_publication)
    assert [
        publication.publication_id
        for publication in await store.list_publication_page(
            "user-1",
            limit=5,
            offset=0,
            status=PublicationStatus.preparing,
        )
    ] == [
        "publication-proposal-2",
        "publication-proposal-1",
    ]
