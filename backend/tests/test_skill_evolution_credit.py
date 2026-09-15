from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from deerflow.persistence.base import Base
from deerflow.skill_evolution.credit import (
    DISTILLATION_CREDIT_FORMULA_VERSION,
    SELECTION_CREDIT_FORMULA_VERSION,
    CreditRecorder,
)
from deerflow.skill_evolution.models import (
    CreditKind,
    CreditOutcomeSample,
    DistillationCredit,
    DistillationCreditStatus,
    EnvironmentSignature,
    EvolutionEvent,
    EvolutionEventKind,
    EvolutionTraceSnapshot,
    OutcomeEvidence,
    OutcomeStatus,
    ProposalOperation,
    ProposalStatus,
    ProposedSkillFile,
    PublicationStatus,
    SelectionDecisionSource,
    SkillProposal,
    SkillPublication,
    SkillTarget,
    SkillUsage,
    TraceRunStatus,
    TraceSkillEvent,
    TraceToolEvent,
)
from deerflow.skill_evolution.store.base import (
    SkillEvolutionStore,
)
from deerflow.skill_evolution.store.memory import (
    InMemorySkillEvolutionStore,
)
from deerflow.skill_evolution.store.sql import (
    SqlSkillEvolutionStore,
)
from deerflow.skills.package import (
    SkillPackageFile,
    compute_skill_package_hash,
)

_NOW = datetime(2026, 8, 18, tzinfo=UTC)
_BASE_CONTENT = b"---\nname: verified-workflow\n---\nUse the old check.\n"
_PUBLISHED_CONTENT = b"---\nname: verified-workflow\n---\nUse the deterministic check.\n"
_SKILL_HASH = hashlib.sha256(_BASE_CONTENT).hexdigest()
_PUBLISHED_HASH = hashlib.sha256(_PUBLISHED_CONTENT).hexdigest()


@pytest_asyncio.fixture(params=["memory", "sql"])
async def store(
    request: pytest.FixtureRequest,
    tmp_path,
):
    if request.param == "memory":
        yield InMemorySkillEvolutionStore()
        return

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'credits.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sql_store = SqlSkillEvolutionStore(async_sessionmaker(engine, expire_on_commit=False))
    try:
        yield sql_store
    finally:
        await engine.dispose()


def _snapshot(
    *,
    run_id: str,
    reward_skill: bool,
    created_at: datetime,
) -> EvolutionTraceSnapshot:
    skill_events = (
        [
            TraceSkillEvent(
                skill_name="verified-workflow",
                skill_path=("/mnt/skills/custom/verified-workflow/SKILL.md"),
                content_hash=_PUBLISHED_HASH,
                activation_source="slash",
                category="custom",
            )
        ]
        if reward_skill
        else []
    )
    return EvolutionTraceSnapshot(
        snapshot_hash=(run_id[-1] * 64),
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        user_id="user-1",
        model_name="policy-model",
        run_status=TraceRunStatus.success,
        task_input="redacted task",
        final_answer="redacted answer",
        tool_events=[
            TraceToolEvent(
                sequence=1,
                tool_call_id=f"tool-{run_id}",
                tool_name="bash",
                arguments="{}",
                result="ok",
                status="success",
            )
        ],
        skill_events=skill_events,
        environment=EnvironmentSignature(
            os="macos",
            shell="zsh",
            runtime="python3.12",
        ),
        source_event_count=1,
        included_event_count=1,
        truncated=False,
        created_at=created_at,
    )


def _outcome(
    reward: float,
) -> OutcomeEvidence:
    return OutcomeEvidence(
        status=(OutcomeStatus.success if reward > 0 else OutcomeStatus.failure),
        confidence=0.95,
        sources=["deterministic_verifier"],
        final_reward=reward,
    )


def _event(
    *,
    event_id: str,
    run_id: str,
    reward: float,
) -> EvolutionEvent:
    return EvolutionEvent(
        event_id=event_id,
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        user_id="user-1",
        extractor_version="extractor-v1",
        source_snapshot_hash=(event_id[-1] * 64),
        task_input_hash=(run_id[-1] * 64),
        event_kind=EvolutionEventKind.skill_patch_evidence,
        task_signature="verified-workflow",
        task_goal="Use the verified workflow.",
        environment=EnvironmentSignature(os="macos"),
        outcome=_outcome(reward),
        complexity={
            "tool_calls": 6,
            "had_recoverable_errors": True,
        },
        tool_signature={
            "tool_names": ["bash"],
            "error_types": [],
        },
        skill_usage=SkillUsage(
            used=True,
            skill_name="verified-workflow",
            skill_path=("/mnt/skills/custom/verified-workflow/SKILL.md"),
            content_hash=_SKILL_HASH,
            activation_source="slash",
        ),
        successful_path=["Run the verified command."],
        skill_gaps=[
            {
                "category": "weak_verification",
                "evidence": "The old verification was incomplete.",
                "recommended_change": "Use the deterministic check.",
            }
        ],
        target_skill=SkillTarget(
            name="verified-workflow",
            content_hash=_SKILL_HASH,
        ),
        created_at=_NOW - timedelta(days=3),
    )


def _proposal() -> SkillProposal:
    return SkillProposal(
        proposal_id="proposal-1",
        cluster_id="cluster-1",
        user_id="user-1",
        operation=ProposalOperation.patch,
        skill_name="verified-workflow",
        base_skill_hash=_SKILL_HASH,
        proposed_files=[
            ProposedSkillFile(
                path="SKILL.md",
                content=("---\nname: verified-workflow\n---\nUse the deterministic check.\n"),
                executable=False,
            )
        ],
        supporting_event_ids=["event-1", "event-2", "event-3"],
        rationale="Improve repeated verification.",
        expected_improvements=["Increase task success."],
        patch_operations=[
            {
                "find": "old check",
                "replace": "deterministic check",
                "reason": "Repeated evidence supports the change.",
                "supporting_event_ids": [
                    "event-1",
                    "event-2",
                    "event-3",
                ],
            }
        ],
        source_skill_hashes=[_SKILL_HASH],
        status=ProposalStatus.published,
        created_at=_NOW - timedelta(days=2),
    )


def _package_snapshot(
    *,
    content: bytes,
) -> dict:
    package_file = SkillPackageFile(
        path="SKILL.md",
        content=content,
        executable=False,
    )
    return {
        "snapshot_hash": compute_skill_package_hash((package_file,)),
        "user_id": "user-1",
        "skill_name": "verified-workflow",
        "exists": True,
        "skill_md_hash": package_file.content_hash,
        "files": [
            {
                "path": "SKILL.md",
                "content_base64": base64.b64encode(content).decode("ascii"),
                "content_hash": package_file.content_hash,
                "size_bytes": len(content),
                "executable": False,
            }
        ],
        "created_at": _NOW,
    }


def _publication() -> SkillPublication:
    base = _package_snapshot(
        content=_BASE_CONTENT,
    )
    published = _package_snapshot(
        content=_PUBLISHED_CONTENT,
    )
    return SkillPublication(
        publication_id="publication-proposal-1",
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        skill_name="verified-workflow",
        operation=ProposalOperation.patch,
        status=PublicationStatus.published,
        base_snapshot=base,
        candidate_snapshot=published,
        published_snapshot=published,
        base_skill_hash=_SKILL_HASH,
        published_skill_hash=_PUBLISHED_HASH,
        created_at=_NOW,
        published_at=_NOW,
    )


def test_credit_models_reject_forged_mature_state() -> None:
    with pytest.raises(ValidationError):
        DistillationCredit(
            credit_id="credit-distillation-1",
            user_id="user-1",
            proposal_id="proposal-1",
            publication_id="publication-1",
            skill_name="verified-workflow",
            published_skill_hash=_PUBLISHED_HASH,
            source_event_ids=["event-1"],
            baseline_outcomes=[
                CreditOutcomeSample(
                    run_id="source-1",
                    reward=1.0,
                    outcome_status=OutcomeStatus.success,
                    observed_at=_NOW,
                )
            ],
            future_outcomes=[],
            minimum_future_samples=3,
            status=DistillationCreditStatus.mature,
            formula_version=(DISTILLATION_CREDIT_FORMULA_VERSION),
            credit_value=0.2,
            created_at=_NOW,
            updated_at=_NOW,
        )


@pytest.mark.asyncio
async def test_credit_store_is_owner_scoped_idempotent_and_cas(
    store: SkillEvolutionStore,
) -> None:
    recorder = CreditRecorder(store)
    result = await recorder.record_verified_run(
        _snapshot(
            run_id="run-1",
            reward_skill=False,
            created_at=_NOW,
        ),
        _outcome(1.0),
    )
    repeated = await recorder.record_verified_run(
        _snapshot(
            run_id="run-1",
            reward_skill=False,
            created_at=_NOW,
        ),
        _outcome(1.0),
    )

    assert result.selection.kind is CreditKind.selection
    assert result.selection.no_skill_selected is True
    assert result.selection.decision_source is SelectionDecisionSource.no_skill
    assert result.selection.formula_version == (SELECTION_CREDIT_FORMULA_VERSION)
    assert result.selection.credit_value == 1.0
    assert repeated.selection == result.selection
    assert repeated.selection_created is False
    assert (
        await store.list_credits(
            "other-user",
            limit=10,
        )
        == []
    )


@pytest.mark.asyncio
async def test_verified_skill_run_records_utilization_and_selection_trend(
    store: SkillEvolutionStore,
) -> None:
    recorder = CreditRecorder(store)
    first = await recorder.record_verified_run(
        _snapshot(
            run_id="run-1",
            reward_skill=True,
            created_at=_NOW,
        ),
        _outcome(1.0),
    )
    second = await recorder.record_verified_run(
        _snapshot(
            run_id="run-2",
            reward_skill=True,
            created_at=_NOW + timedelta(minutes=1),
        ),
        _outcome(0.0),
    )

    assert first.selection.search_query == ("select:verified-workflow")
    assert first.selection.policy_model_name is None
    assert first.selection.log_probability is None
    assert len(first.utilizations) == 1
    utilization = first.utilizations[0]
    assert utilization.skill_content_hash == _PUBLISHED_HASH
    assert utilization.activation_source == "slash"
    assert utilization.credit_value == 1.0
    assert utilization.cost.tool_call_count == 1
    assert "token_usage" in utilization.cost.unavailable_fields
    assert utilization.instruction_adherence is None
    assert second.selection.utility_sample_count == 2
    assert 0.0 < second.selection.credit_value < 1.0


@pytest.mark.asyncio
async def test_distillation_credit_matures_after_three_distinct_future_runs(
    store: SkillEvolutionStore,
) -> None:
    events = [
        _event(
            event_id=f"event-{index}",
            run_id=f"source-{index}",
            reward=0.5,
        )
        for index in range(1, 4)
    ]
    for event in events:
        await store.upsert_event(event)
    recorder = CreditRecorder(store)
    created = await recorder.record_publication(
        _proposal(),
        _publication(),
    )

    assert created.status is DistillationCreditStatus.collecting
    assert created.credit_value is None
    assert created.formula_version == (DISTILLATION_CREDIT_FORMULA_VERSION)

    for index, reward in enumerate(
        [1.0, 0.75, 1.0],
        start=1,
    ):
        await recorder.record_verified_run(
            _snapshot(
                run_id=f"future-{index}",
                reward_skill=True,
                created_at=_NOW + timedelta(days=index),
            ),
            _outcome(reward),
        )

    current = await store.get_credit(
        "user-1",
        created.credit_id,
    )
    assert isinstance(current, DistillationCredit)
    assert current.status is DistillationCreditStatus.mature
    assert len(current.future_outcomes) == 3
    assert current.credit_value == pytest.approx(((1.0 + 0.75 + 1.0) / 3) - 0.5)

    await recorder.record_verified_run(
        _snapshot(
            run_id="future-3",
            reward_skill=True,
            created_at=_NOW + timedelta(days=3),
        ),
        _outcome(1.0),
    )
    repeated = await store.get_credit(
        "user-1",
        created.credit_id,
    )
    assert isinstance(repeated, DistillationCredit)
    assert len(repeated.future_outcomes) == 3


@pytest.mark.asyncio
async def test_rollback_closes_distillation_credit(
    store: SkillEvolutionStore,
) -> None:
    for index in range(1, 4):
        await store.upsert_event(
            _event(
                event_id=f"event-{index}",
                run_id=f"source-{index}",
                reward=1.0,
            )
        )
    recorder = CreditRecorder(store)
    created = await recorder.record_publication(
        _proposal(),
        _publication(),
    )
    closed = await recorder.record_rollback(
        _publication(),
        rolled_back_at=_NOW + timedelta(days=1),
    )

    assert closed is not None
    assert closed.credit_id == created.credit_id
    assert closed.status is DistillationCreditStatus.rolled_back
