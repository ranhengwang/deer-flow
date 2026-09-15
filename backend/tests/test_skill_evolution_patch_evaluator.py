from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from deerflow.config.skill_evolution_config import (
    SkillEvolutionQualityConfig,
)
from deerflow.skill_evolution.evaluator import (
    PatchSkillProposalEvaluator,
    ReplayAgentExecution,
    ReplayAgentRequest,
    ReplayArtifactVerifier,
    ReplayCommandExecution,
    ReplayCommandRequest,
    ReplayCommandVerifier,
    ReplaySkillPackage,
    ReplayTaskSpec,
    build_replay_fixture_snapshot,
    build_replay_skill_package,
    build_replay_task_spec,
)
from deerflow.skill_evolution.models import (
    ComplexitySignals,
    EnvironmentSignature,
    EvaluationDecision,
    EvolutionEvent,
    EvolutionEventKind,
    EvolutionTraceSnapshot,
    OutcomeEvidence,
    OutcomeStatus,
    ProposalOperation,
    ProposalStatus,
    ProposedSkillFile,
    QualityDesignation,
    SkillGap,
    SkillGapCategory,
    SkillPatchOperation,
    SkillProposal,
    SkillTarget,
    SkillUsage,
    ToolSignature,
    TraceRunStatus,
)
from deerflow.skill_evolution.observability import (
    EvolutionLifecycleKind,
    EvolutionObservability,
)
from deerflow.skill_evolution.store.memory import (
    InMemorySkillEvolutionStore,
)

_CREATED = datetime(2026, 8, 17, tzinfo=UTC)
_BASE_SKILL = "---\nname: fixture-transform\ndescription: Transform fixtures.\n---\n\n# Fixture Transform\n\nRun the legacy transform command.\n"
_CANDIDATE_SKILL = _BASE_SKILL.replace(
    "legacy transform command",
    "portable transform command",
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _event(
    event_id: str,
    *,
    task_family: str = "fixture-transform-patch",
) -> EvolutionEvent:
    task_input = f"Transform the fixture for {event_id}."
    base_hash = _sha256(_BASE_SKILL)
    return EvolutionEvent(
        event_id=event_id,
        run_id=f"run-{event_id}",
        thread_id=f"thread-{event_id}",
        user_id="user-1",
        extractor_version="structured-v1:test-model",
        source_snapshot_hash=_sha256(f"snapshot-{event_id}"),
        task_input_hash=_sha256(task_input),
        event_kind=EvolutionEventKind.skill_patch_evidence,
        task_signature=task_family,
        task_goal="Transform a fixture and verify the output.",
        environment=EnvironmentSignature(
            os="macOS",
            shell="zsh",
            runtime="Python 3.12",
        ),
        outcome=OutcomeEvidence(
            status=OutcomeStatus.success,
            confidence=0.95,
            sources=["command"],
        ),
        complexity=ComplexitySignals(
            tool_calls=6,
            had_recoverable_errors=True,
        ),
        tool_signature=ToolSignature(
            tool_names=["read_file", "bash", "write_file"],
        ),
        skill_usage=SkillUsage(
            used=True,
            skill_name="fixture-transform",
            skill_path="skills/custom/fixture-transform/SKILL.md",
            content_hash=base_hash,
        ),
        target_skill=SkillTarget(
            name="fixture-transform",
            content_hash=base_hash,
        ),
        successful_path=[
            "Read the fixture.",
            "Use the portable command.",
            "Verify the output.",
        ],
        reusable_lessons=["Use the portable command."],
        skill_gaps=[
            SkillGap(
                category=SkillGapCategory.wrong_tool_guidance,
                evidence="The legacy command failed.",
                recommended_change="Use the portable command.",
            )
        ],
        created_at=_CREATED,
    )


def _snapshot(event: EvolutionEvent) -> EvolutionTraceSnapshot:
    return EvolutionTraceSnapshot(
        snapshot_hash=event.source_snapshot_hash,
        run_id=event.run_id,
        thread_id=event.thread_id,
        user_id=event.user_id,
        model_name="qwen3-local",
        run_status=TraceRunStatus.success,
        task_input=f"Transform the fixture for {event.event_id}.",
        final_answer="The fixture was transformed.",
        environment=event.environment,
        source_event_count=1,
        included_event_count=1,
        truncated=False,
        created_at=_CREATED,
    )


def _task(
    event_id: str,
    *,
    task_family: str = "fixture-transform-patch",
    with_verifiers: bool = True,
) -> ReplayTaskSpec:
    event = _event(
        event_id,
        task_family=task_family,
    )
    verifiers = []
    if with_verifiers:
        verifiers = [
            ReplayCommandVerifier(
                command="verify-output",
                expected_exit_code=0,
            ),
            ReplayArtifactVerifier(
                path="outputs/result.txt",
                contains_text="verified",
            ),
        ]
    return build_replay_task_spec(
        _snapshot(event),
        event,
        fixture=build_replay_fixture_snapshot({"project/input.txt": event_id.encode()}),
        verifiers=verifiers,
    )


def _proposal(
    *,
    status: ProposalStatus = ProposalStatus.staged,
    executable_support: bool = False,
) -> SkillProposal:
    proposed_files = [
        ProposedSkillFile(
            path="SKILL.md",
            content=_CANDIDATE_SKILL,
            executable=False,
        )
    ]
    if executable_support:
        proposed_files.append(
            ProposedSkillFile(
                path="scripts/migrate.sh",
                content="#!/bin/sh\nprintf migrated\n",
                executable=True,
            )
        )
    return SkillProposal(
        proposal_id="proposal-patch-1",
        cluster_id="cluster-patch-1",
        user_id="user-1",
        operation=ProposalOperation.patch,
        skill_name="fixture-transform",
        base_skill_hash=_sha256(_BASE_SKILL),
        proposed_files=proposed_files,
        supporting_event_ids=[
            "source-1",
            "source-2",
            "source-3",
        ],
        rationale="Replace the failing legacy command.",
        expected_improvements=["Work across supported environments."],
        patch_operations=[
            SkillPatchOperation(
                find="legacy transform command",
                replace="portable transform command",
                expected_count=1,
                reason="The legacy command failed in source runs.",
                supporting_event_ids=[
                    "source-1",
                    "source-2",
                    "source-3",
                ],
            )
        ],
        source_skill_hashes=[_sha256(_BASE_SKILL)],
        status=status,
        created_at=_CREATED,
    )


def _base_package(
    *,
    complete: bool = True,
    skill_md: str = _BASE_SKILL,
) -> ReplaySkillPackage:
    return build_replay_skill_package(
        user_id="user-1",
        skill_name="fixture-transform",
        files={
            "SKILL.md": skill_md,
            "references/guide.txt": "shared-reference",
        },
        complete=complete,
        omitted_paths=([] if complete else ["scripts/legacy-helper.sh"]),
    )


def _source_tasks() -> list[ReplayTaskSpec]:
    return [
        _task("source-1"),
        _task("source-2"),
        _task("source-3"),
    ]


def _regression_tasks() -> list[ReplayTaskSpec]:
    return [
        _task(
            "regression-1",
            task_family="historical-import",
        ),
        _task(
            "regression-2",
            task_family="historical-export",
        ),
    ]


def test_replay_skill_package_preserves_exact_skill_content() -> None:
    content = "\n" + _BASE_SKILL + "\n"
    package = build_replay_skill_package(
        user_id="user-1",
        skill_name="fixture-transform",
        files={"SKILL.md": content},
    )

    assert package.files[0].content == content
    assert package.skill_md_hash == _sha256(content)


class FakePatchReplayRuntime:
    def __init__(
        self,
        *,
        failures: set[tuple[str, str]] | None = None,
    ) -> None:
        self.failures = failures or set()
        self.agent_calls: list[tuple[str, str, str, bool]] = []
        self.command_calls: list[tuple[str, str]] = []

    async def run_agent(
        self,
        request: ReplayAgentRequest,
    ) -> ReplayAgentExecution:
        active_path = request.active_skill_path
        assert active_path is not None
        skill_content = (active_path / "SKILL.md").read_text(encoding="utf-8")
        support_present = (active_path / "references" / "guide.txt").read_text(encoding="utf-8") == "shared-reference"
        fixture = (request.paths.workspace / "project" / "input.txt").read_text(encoding="utf-8")
        assert fixture == request.spec.source_event_id
        self.agent_calls.append(
            (
                request.spec.source_event_id,
                request.condition,
                skill_content,
                support_present,
            )
        )
        output = "verified"
        if (
            request.condition,
            request.spec.source_event_id,
        ) in self.failures:
            output = "incorrect"
        (request.paths.outputs / "result.txt").write_text(output, encoding="utf-8")
        return ReplayAgentExecution(
            tool_calls=(3 if request.condition == "candidate_skill" else 5),
            input_tokens=120,
            output_tokens=24,
        )

    async def run_command(
        self,
        request: ReplayCommandRequest,
    ) -> ReplayCommandExecution:
        self.command_calls.append(
            (
                request.spec.source_event_id,
                request.condition,
            )
        )
        return ReplayCommandExecution(exit_code=0)


@pytest.mark.asyncio
async def test_patch_evaluation_runs_base_candidate_source_and_regression_pairs(
    tmp_path: Path,
) -> None:
    runtime = FakePatchReplayRuntime(
        failures={
            ("base_skill", "source-1"),
            ("base_skill", "source-2"),
            ("base_skill", "source-3"),
        }
    )
    evaluator = PatchSkillProposalEvaluator(runtime=runtime)

    evaluation = await evaluator.evaluate(
        _proposal(),
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.approve
    assert evaluation.quality_score > 0.0
    assert evaluation.quality is not None
    assert evaluation.quality.designation is QualityDesignation.insufficient_evidence
    assert "insufficient_environment_diversity" in evaluation.quality.blockers
    assert len(runtime.agent_calls) == 10
    assert len(runtime.command_calls) == 10
    assert len(evaluation.source_replay_results) == 3
    assert evaluation.held_out_results == []
    assert len(evaluation.regression_results) == 2
    assert len(evaluation.baseline_results) == 5
    assert len(evaluation.candidate_results) == 5
    assert all(result.condition == "base_skill" for result in evaluation.baseline_results)
    assert all(result.condition == "candidate_skill" for result in evaluation.candidate_results)
    assert evaluation.safety_results["regression_rate"] == "0.000000"
    assert evaluation.safety_results["regression_sample"] == "2"

    base_calls = [call for call in runtime.agent_calls if call[1] == "base_skill"]
    candidate_calls = [call for call in runtime.agent_calls if call[1] == "candidate_skill"]
    assert all("legacy transform command" in call[2] for call in base_calls)
    assert all("portable transform command" in call[2] for call in candidate_calls)
    assert all(call[3] for call in runtime.agent_calls)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_regression_above_threshold_rejects_patch(
    tmp_path: Path,
) -> None:
    evaluator = PatchSkillProposalEvaluator(
        runtime=FakePatchReplayRuntime(
            failures={
                ("candidate_skill", "regression-1"),
            }
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        parent_dir=tmp_path,
    )

    assert all(result.success for result in evaluation.source_replay_results)
    assert evaluation.regression_results[0].success is False
    assert evaluation.safety_results["regression_rate"] == "0.500000"
    assert evaluation.decision is EvaluationDecision.reject


@pytest.mark.asyncio
async def test_configured_regression_threshold_is_applied(
    tmp_path: Path,
) -> None:
    evaluator = PatchSkillProposalEvaluator(
        runtime=FakePatchReplayRuntime(
            failures={
                ("candidate_skill", "regression-1"),
            }
        ),
        quality_config=SkillEvolutionQualityConfig(
            max_regression_rate=0.5,
        ),
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        parent_dir=tmp_path,
    )

    assert evaluation.safety_results["regression_rate"] == "0.500000"
    assert evaluation.decision is EvaluationDecision.approve


@pytest.mark.asyncio
async def test_base_failure_is_not_counted_as_candidate_regression(
    tmp_path: Path,
) -> None:
    evaluator = PatchSkillProposalEvaluator(
        runtime=FakePatchReplayRuntime(
            failures={
                ("base_skill", "regression-1"),
                ("candidate_skill", "regression-1"),
            }
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        parent_dir=tmp_path,
    )

    assert evaluation.safety_results["regression_rate"] == "0.000000"
    assert evaluation.safety_results["regression_sample"] == "1"
    assert evaluation.decision is EvaluationDecision.approve


@pytest.mark.asyncio
async def test_missing_regression_tasks_requires_manual_review_without_running(
    tmp_path: Path,
) -> None:
    runtime = FakePatchReplayRuntime()
    evaluator = PatchSkillProposalEvaluator(runtime=runtime)

    evaluation = await evaluator.evaluate(
        _proposal(),
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=[],
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.manual_review
    assert evaluation.safety_results["regression"] == "missing"
    assert runtime.agent_calls == []


@pytest.mark.asyncio
async def test_no_successful_base_regression_sample_requires_manual_review(
    tmp_path: Path,
) -> None:
    evaluator = PatchSkillProposalEvaluator(
        runtime=FakePatchReplayRuntime(
            failures={
                ("base_skill", "regression-1"),
                ("base_skill", "regression-2"),
            }
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        parent_dir=tmp_path,
    )

    assert evaluation.safety_results["regression_sample"] == "0"
    assert evaluation.decision is EvaluationDecision.manual_review


@pytest.mark.asyncio
async def test_incomplete_base_package_requires_manual_review_without_running(
    tmp_path: Path,
) -> None:
    runtime = FakePatchReplayRuntime()
    evaluator = PatchSkillProposalEvaluator(runtime=runtime)

    evaluation = await evaluator.evaluate(
        _proposal(),
        base_skill=_base_package(complete=False),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.manual_review
    assert evaluation.safety_results["base_skill_package"] == "incomplete"
    assert runtime.agent_calls == []


@pytest.mark.asyncio
async def test_base_skill_hash_mismatch_is_rejected_before_runtime(
    tmp_path: Path,
) -> None:
    runtime = FakePatchReplayRuntime()
    evaluator = PatchSkillProposalEvaluator(runtime=runtime)

    with pytest.raises(ValueError, match="base Skill hash"):
        await evaluator.evaluate(
            _proposal(),
            base_skill=_base_package(
                skill_md=_BASE_SKILL.replace(
                    "legacy",
                    "different",
                )
            ),
            source_tasks=_source_tasks(),
            regression_tasks=_regression_tasks(),
            parent_dir=tmp_path,
        )

    assert runtime.agent_calls == []


@pytest.mark.asyncio
async def test_base_skill_user_mismatch_is_rejected_before_runtime(
    tmp_path: Path,
) -> None:
    runtime = FakePatchReplayRuntime()
    evaluator = PatchSkillProposalEvaluator(runtime=runtime)
    base_skill = build_replay_skill_package(
        user_id="other-user",
        skill_name="fixture-transform",
        files={"SKILL.md": _BASE_SKILL},
    )

    with pytest.raises(ValueError, match="user"):
        await evaluator.evaluate(
            _proposal(),
            base_skill=base_skill,
            source_tasks=_source_tasks(),
            regression_tasks=_regression_tasks(),
            parent_dir=tmp_path,
        )

    assert runtime.agent_calls == []


@pytest.mark.asyncio
async def test_manual_replay_task_blocks_patch_evaluation(
    tmp_path: Path,
) -> None:
    runtime = FakePatchReplayRuntime()
    evaluator = PatchSkillProposalEvaluator(runtime=runtime)
    regression_tasks = _regression_tasks()
    regression_tasks[0] = _task(
        "regression-1",
        task_family="historical-import",
        with_verifiers=False,
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=regression_tasks,
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.manual_review
    assert evaluation.safety_results["replayability"] == "manual_review_required"
    assert runtime.agent_calls == []


@pytest.mark.asyncio
async def test_executable_support_change_forces_manual_review(
    tmp_path: Path,
) -> None:
    runtime = FakePatchReplayRuntime()
    evaluator = PatchSkillProposalEvaluator(runtime=runtime)

    evaluation = await evaluator.evaluate(
        _proposal(executable_support=True),
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        parent_dir=tmp_path,
    )

    assert all(result.success for result in evaluation.candidate_results)
    assert evaluation.decision is EvaluationDecision.manual_review
    assert evaluation.safety_results["executable_support"] == "manual_review_required"


@pytest.mark.asyncio
async def test_source_tasks_must_cover_patch_supporting_events(
    tmp_path: Path,
) -> None:
    evaluator = PatchSkillProposalEvaluator(runtime=FakePatchReplayRuntime())

    with pytest.raises(ValueError, match="supporting events"):
        await evaluator.evaluate(
            _proposal(),
            base_skill=_base_package(),
            source_tasks=_source_tasks()[:2],
            regression_tasks=_regression_tasks(),
            parent_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_new_skill_proposal_is_not_evaluated_by_patch_path(
    tmp_path: Path,
) -> None:
    proposal = SkillProposal(
        proposal_id="proposal-create-1",
        cluster_id="cluster-create-1",
        user_id="user-1",
        operation=ProposalOperation.create,
        skill_name="fixture-transform",
        proposed_files=[
            ProposedSkillFile(
                path="SKILL.md",
                content=_CANDIDATE_SKILL,
                executable=False,
            )
        ],
        supporting_event_ids=[
            "source-1",
            "source-2",
            "source-3",
        ],
        rationale="Create a new Skill.",
        expected_improvements=["Reuse the workflow."],
        status=ProposalStatus.staged,
        created_at=_CREATED,
    )
    evaluator = PatchSkillProposalEvaluator(runtime=FakePatchReplayRuntime())

    with pytest.raises(ValueError, match="patch proposal"):
        await evaluator.evaluate(
            proposal,
            base_skill=_base_package(),
            source_tasks=_source_tasks(),
            regression_tasks=_regression_tasks(),
            parent_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_patch_evaluate_and_persist_is_idempotent_and_keeps_passed_validating(
    tmp_path: Path,
) -> None:
    runtime = FakePatchReplayRuntime()
    evaluator = PatchSkillProposalEvaluator(runtime=runtime)
    store = InMemorySkillEvolutionStore()
    proposal = _proposal()
    await store.put_proposal(proposal)

    first = await evaluator.evaluate_and_persist(
        proposal,
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        store=store,
        parent_dir=tmp_path,
    )
    call_count = len(runtime.agent_calls)
    second = await evaluator.evaluate_and_persist(
        proposal,
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        store=store,
        parent_dir=tmp_path,
    )

    assert first.created is True
    assert second.created is False
    assert second.value == first.value
    assert len(runtime.agent_calls) == call_count
    stored = await store.get_proposal(
        proposal.user_id,
        proposal.proposal_id,
    )
    assert stored is not None
    assert stored.status is ProposalStatus.validating


@pytest.mark.asyncio
async def test_failed_patch_evaluation_rejects_proposal(
    tmp_path: Path,
) -> None:
    observer = EvolutionObservability()
    evaluator = PatchSkillProposalEvaluator(
        runtime=FakePatchReplayRuntime(
            failures={
                ("candidate_skill", "regression-1"),
            }
        ),
        observability=observer,
    )
    store = InMemorySkillEvolutionStore()
    proposal = _proposal()
    await store.put_proposal(proposal)

    result = await evaluator.evaluate_and_persist(
        proposal,
        base_skill=_base_package(),
        source_tasks=_source_tasks(),
        regression_tasks=_regression_tasks(),
        store=store,
        parent_dir=tmp_path,
    )

    assert result.value.decision is EvaluationDecision.reject
    stored = await store.get_proposal(
        proposal.user_id,
        proposal.proposal_id,
    )
    assert stored is not None
    assert stored.status is ProposalStatus.rejected
    assert [item.reason_code for item in stored.status_history] == [
        "evaluation_started",
        "evaluation_rejected",
    ]
    assert stored.status_history[-1].evaluation_id == result.value.evaluation_id
    assert [event.kind for event in observer.recent_events()] == [
        EvolutionLifecycleKind.evaluated,
        EvolutionLifecycleKind.rejected,
    ]
    assert observer.snapshot().regression_rate.last == 0.5
