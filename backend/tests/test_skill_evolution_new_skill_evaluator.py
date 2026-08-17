from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from deerflow.config.skill_evolution_config import (
    SkillEvolutionQualityConfig,
)
from deerflow.skill_evolution.evaluator import (
    NewSkillProposalEvaluator,
    ReplayAgentExecution,
    ReplayAgentRequest,
    ReplayArtifactVerifier,
    ReplayCommandExecution,
    ReplayCommandRequest,
    ReplayCommandVerifier,
    ReplayTaskSpec,
    build_replay_fixture_snapshot,
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
    SkillPatchOperation,
    SkillProposal,
    SkillUsage,
    ToolSignature,
    TraceRunStatus,
)
from deerflow.skill_evolution.store.memory import InMemorySkillEvolutionStore

_CREATED = datetime(2026, 8, 17, tzinfo=UTC)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _event(
    event_id: str,
    *,
    user_id: str = "user-1",
    task_family: str = "fixture-transform",
) -> EvolutionEvent:
    task_input = f"Transform the fixture for {event_id}."
    return EvolutionEvent(
        event_id=event_id,
        run_id=f"run-{event_id}",
        thread_id=f"thread-{event_id}",
        user_id=user_id,
        extractor_version="structured-v1:test-model",
        source_snapshot_hash=_sha256(f"snapshot-{event_id}"),
        task_input_hash=_sha256(task_input),
        event_kind=EvolutionEventKind.new_skill_evidence,
        task_signature=task_family,
        task_goal="Transform an input fixture and verify the output.",
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
        complexity=ComplexitySignals(tool_calls=6),
        tool_signature=ToolSignature(
            tool_names=["read_file", "bash", "write_file"],
        ),
        skill_usage=SkillUsage(used=False),
        successful_path=[
            "Read the fixture.",
            "Transform the data.",
            "Verify the output.",
        ],
        reusable_lessons=["Verify the generated artifact deterministically."],
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
        final_answer="The fixture was transformed and verified.",
        environment=event.environment,
        source_event_count=1,
        included_event_count=1,
        truncated=False,
        created_at=_CREATED,
    )


def _task(
    event_id: str,
    *,
    with_verifiers: bool = True,
    timeout_seconds: int = 30,
    task_family: str = "fixture-transform",
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
                timeout_seconds=10,
            ),
            ReplayArtifactVerifier(
                path="outputs/result.txt",
                must_exist=True,
                contains_text="verified",
            ),
        ]
    return build_replay_task_spec(
        _snapshot(event),
        event,
        fixture=build_replay_fixture_snapshot({"project/input.txt": event_id.encode()}),
        verifiers=verifiers,
        timeout_seconds=timeout_seconds,
    )


def _proposal(
    *,
    operation: ProposalOperation = ProposalOperation.create,
    status: ProposalStatus = ProposalStatus.staged,
    requires_manual_review: bool = False,
) -> SkillProposal:
    kwargs = {}
    if operation is ProposalOperation.patch:
        kwargs["base_skill_hash"] = "a" * 64
        kwargs["patch_operations"] = [
            SkillPatchOperation(
                find="# Fixture Transform\n",
                replace="# Improved Fixture Transform\n",
                expected_count=1,
                reason="Improve the repeated workflow.",
                supporting_event_ids=[
                    "source-1",
                    "source-2",
                ],
            )
        ]
        kwargs["source_skill_hashes"] = ["b" * 64]
    elif operation is not ProposalOperation.create:
        kwargs["base_skill_hash"] = "a" * 64
    return SkillProposal(
        proposal_id="proposal-1",
        cluster_id="cluster-1",
        user_id="user-1",
        operation=operation,
        skill_name="fixture-transform",
        proposed_files=[
            ProposedSkillFile(
                path="SKILL.md",
                content=("---\nname: fixture-transform\ndescription: Transform fixtures safely.\n---\n\n# Fixture Transform\n"),
                executable=False,
            )
        ],
        supporting_event_ids=["source-1", "source-2", "source-3"],
        rationale="Evaluate the repeated transformation workflow.",
        expected_improvements=["Reduce transformation failures."],
        requires_manual_review=requires_manual_review,
        review_reasons=["proposal_conflict"] if requires_manual_review else [],
        status=status,
        created_at=_CREATED,
        **kwargs,
    )


class FakeReplayRuntime:
    def __init__(
        self,
        *,
        fail_candidate_events: set[str] | None = None,
        error_events: set[str] | None = None,
        timeout_events: set[str] | None = None,
        mutate_candidate: bool = False,
        output_symlink_target: Path | None = None,
    ) -> None:
        self.fail_candidate_events = fail_candidate_events or set()
        self.error_events = error_events or set()
        self.timeout_events = timeout_events or set()
        self.mutate_candidate = mutate_candidate
        self.output_symlink_target = output_symlink_target
        self.agent_calls: list[tuple[str, str, bool, str | None]] = []
        self.command_calls: list[tuple[str, str]] = []

    async def run_agent(
        self,
        request: ReplayAgentRequest,
    ) -> ReplayAgentExecution:
        candidate_path = request.candidate_skill_path
        candidate_content = None
        if candidate_path is not None:
            candidate_content = (candidate_path / "SKILL.md").read_text(encoding="utf-8")
        self.agent_calls.append(
            (
                request.spec.source_event_id,
                request.condition,
                candidate_path is not None,
                candidate_content,
            )
        )
        fixture = (request.paths.workspace / "project" / "input.txt").read_text(encoding="utf-8")
        assert fixture == request.spec.source_event_id
        if request.spec.source_event_id in self.timeout_events:
            await asyncio.sleep(2)
        if request.spec.source_event_id in self.error_events:
            raise RuntimeError("sensitive runtime detail")

        content = "verified"
        if request.condition == "candidate_skill" and request.spec.source_event_id in self.fail_candidate_events:
            content = "incorrect"
        output = request.paths.outputs / "result.txt"
        if self.output_symlink_target is not None:
            output.symlink_to(self.output_symlink_target)
        else:
            output.write_text(content, encoding="utf-8")
        if self.mutate_candidate and candidate_path is not None:
            skill_file = candidate_path / "SKILL.md"
            skill_file.chmod(0o644)
            skill_file.write_text(
                "mutated",
                encoding="utf-8",
            )
        return ReplayAgentExecution(
            tool_calls=2 if request.condition == "candidate_skill" else 5,
            input_tokens=100,
            output_tokens=20,
        )

    async def run_command(
        self,
        request: ReplayCommandRequest,
    ) -> ReplayCommandExecution:
        self.command_calls.append(
            (
                request.spec.source_event_id,
                request.command,
            )
        )
        return ReplayCommandExecution(exit_code=0)


def _source_tasks() -> list[ReplayTaskSpec]:
    return [
        _task("source-1"),
        _task("source-2"),
        _task("source-3"),
    ]


def test_runtime_errors_must_be_stable_codes() -> None:
    with pytest.raises(ValueError, match="stable codes"):
        ReplayAgentExecution(
            tool_calls=0,
            input_tokens=0,
            output_tokens=0,
            errors=["credential value leaked"],
        )


def test_absent_artifact_verifier_rejects_content_assertions() -> None:
    verifier = ReplayArtifactVerifier(
        path="outputs/forbidden.txt",
        must_exist=False,
    )

    assert verifier.must_exist is False
    with pytest.raises(
        ValueError,
        match="cannot assert content",
    ):
        ReplayArtifactVerifier(
            path="outputs/forbidden.txt",
            must_exist=False,
            contains_text="secret",
        )


@pytest.mark.asyncio
async def test_new_skill_evaluation_runs_paired_source_and_held_out_tasks(
    tmp_path: Path,
) -> None:
    runtime = FakeReplayRuntime()
    evaluator = NewSkillProposalEvaluator(runtime=runtime)

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.approve
    assert evaluation.quality_score == 0.0
    assert len(evaluation.source_replay_results) == 3
    assert len(evaluation.held_out_results) == 1
    assert len(evaluation.baseline_results) == 4
    assert len(evaluation.candidate_results) == 4
    assert evaluation.regression_results == []
    assert evaluation.safety_results["quality_score"] == "deferred_phase_6_4"
    assert evaluation.safety_results["source_replay"] == "3/3"
    assert evaluation.safety_results["held_out"] == "1/1"

    assert len(runtime.agent_calls) == 8
    assert len(runtime.command_calls) == 8
    baseline_calls = [call for call in runtime.agent_calls if call[1] == "no_skill"]
    candidate_calls = [call for call in runtime.agent_calls if call[1] == "candidate_skill"]
    assert all(call[2] is False for call in baseline_calls)
    assert all(call[2] is True for call in candidate_calls)
    assert all(call[3] is not None and "name: fixture-transform" in call[3] for call in candidate_calls)

    candidate = evaluation.candidate_results[0]
    assert candidate.success is True
    assert candidate.metrics.tool_calls == 2
    assert candidate.metrics.input_tokens == 100
    assert candidate.metrics.output_tokens == 20
    assert candidate.metrics.latency_seconds >= 0
    assert candidate.errors == []
    assert [artifact.path for artifact in candidate.artifacts] == ["outputs/result.txt"]
    assert candidate.artifacts[0].content_hash == _sha256("verified")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_held_out_failure_rejects_candidate_after_source_passes(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(
        runtime=FakeReplayRuntime(
            fail_candidate_events={"held-out-1"},
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    assert all(result.success for result in evaluation.source_replay_results)
    assert evaluation.held_out_results[0].success is False
    assert evaluation.decision is EvaluationDecision.reject
    assert "artifact_contains_text_mismatch" in evaluation.held_out_results[0].errors


@pytest.mark.asyncio
async def test_configured_held_out_threshold_is_applied(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(
        runtime=FakeReplayRuntime(
            fail_candidate_events={"held-out-1"},
        ),
        quality_config=SkillEvolutionQualityConfig(
            min_held_out_success_rate=0.0,
        ),
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    assert evaluation.held_out_results[0].success is False
    assert evaluation.decision is EvaluationDecision.approve


@pytest.mark.asyncio
async def test_source_failure_rejects_candidate(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(
        runtime=FakeReplayRuntime(
            fail_candidate_events={"source-2"},
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.reject
    assert evaluation.safety_results["source_replay"] == "2/3"


@pytest.mark.asyncio
async def test_manual_replay_task_blocks_automatic_evaluation(
    tmp_path: Path,
) -> None:
    runtime = FakeReplayRuntime()
    evaluator = NewSkillProposalEvaluator(runtime=runtime)
    source_tasks = _source_tasks()
    source_tasks[1] = _task(
        "source-2",
        with_verifiers=False,
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=source_tasks,
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.manual_review
    assert runtime.agent_calls == []
    assert evaluation.safety_results["replayability"] == "manual_review_required"


@pytest.mark.asyncio
async def test_missing_held_out_tasks_requires_manual_review(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(runtime=FakeReplayRuntime())

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=_source_tasks(),
        held_out_tasks=[],
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.manual_review
    assert evaluation.safety_results["held_out"] == "missing"


@pytest.mark.asyncio
async def test_risky_proposal_runs_evaluation_but_remains_manual_review(
    tmp_path: Path,
) -> None:
    runtime = FakeReplayRuntime()
    evaluator = NewSkillProposalEvaluator(runtime=runtime)

    evaluation = await evaluator.evaluate(
        _proposal(requires_manual_review=True),
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    assert len(runtime.agent_calls) == 8
    assert all(result.success for result in evaluation.candidate_results)
    assert evaluation.decision is EvaluationDecision.manual_review
    assert evaluation.safety_results["proposal_review"] == "manual_review_required"


@pytest.mark.asyncio
async def test_runtime_error_is_bounded_and_does_not_leak_message(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(
        runtime=FakeReplayRuntime(
            error_events={"held-out-1"},
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    result = evaluation.held_out_results[0]
    assert result.success is False
    assert "agent_runtime_error:RuntimeError" in result.errors
    assert "sensitive runtime detail" not in result.model_dump_json()
    assert evaluation.decision is EvaluationDecision.reject


@pytest.mark.asyncio
async def test_agent_timeout_is_recorded_and_workspace_is_cleaned(
    tmp_path: Path,
) -> None:
    source_tasks = _source_tasks()
    source_tasks[0] = _task(
        "source-1",
        timeout_seconds=1,
    )
    evaluator = NewSkillProposalEvaluator(
        runtime=FakeReplayRuntime(
            timeout_events={"source-1"},
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=source_tasks,
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    assert evaluation.source_replay_results[0].success is False
    assert "agent_timeout" in evaluation.source_replay_results[0].errors
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_candidate_skill_mutation_is_a_policy_failure(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(
        runtime=FakeReplayRuntime(
            mutate_candidate=True,
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path,
    )

    assert evaluation.decision is EvaluationDecision.reject
    assert all("side_effect_policy_violation" in result.errors for result in evaluation.candidate_results)
    assert evaluation.safety_results["side_effects"] == "violation"


@pytest.mark.asyncio
async def test_output_symlink_fails_closed_without_reading_target(
    tmp_path: Path,
) -> None:
    external_target = tmp_path / "external.txt"
    external_target.write_text(
        "external-sensitive-content",
        encoding="utf-8",
    )
    external_target.chmod(0o600)
    evaluator = NewSkillProposalEvaluator(
        runtime=FakeReplayRuntime(
            output_symlink_target=external_target,
        )
    )

    evaluation = await evaluator.evaluate(
        _proposal(),
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        parent_dir=tmp_path / "replays",
    )

    assert evaluation.decision is EvaluationDecision.reject
    assert all(result.artifacts[0].kind == "symlink" for result in evaluation.candidate_results)
    assert all(result.artifacts[0].content_hash is None for result in evaluation.candidate_results)
    assert external_target.read_text(encoding="utf-8") == "external-sensitive-content"
    assert external_target.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_source_tasks_must_cover_every_supporting_event(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(runtime=FakeReplayRuntime())

    with pytest.raises(
        ValueError,
        match="supporting events",
    ):
        await evaluator.evaluate(
            _proposal(),
            source_tasks=_source_tasks()[:2],
            held_out_tasks=[_task("held-out-1")],
            parent_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_held_out_tasks_must_match_source_family(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(runtime=FakeReplayRuntime())

    with pytest.raises(ValueError, match="task family"):
        await evaluator.evaluate(
            _proposal(),
            source_tasks=_source_tasks(),
            held_out_tasks=[
                _task(
                    "held-out-1",
                    task_family="different-workflow",
                )
            ],
            parent_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_patch_proposal_is_not_evaluated_by_new_skill_path(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(runtime=FakeReplayRuntime())

    with pytest.raises(ValueError, match="create proposal"):
        await evaluator.evaluate(
            _proposal(operation=ProposalOperation.patch),
            source_tasks=_source_tasks(),
            held_out_tasks=[_task("held-out-1")],
            parent_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_evaluate_and_persist_is_idempotent_and_leaves_passed_proposal_validating(
    tmp_path: Path,
) -> None:
    runtime = FakeReplayRuntime()
    evaluator = NewSkillProposalEvaluator(runtime=runtime)
    store = InMemorySkillEvolutionStore()
    proposal = _proposal()
    await store.put_proposal(proposal)

    first = await evaluator.evaluate_and_persist(
        proposal,
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        store=store,
        parent_dir=tmp_path,
    )
    call_count = len(runtime.agent_calls)
    second = await evaluator.evaluate_and_persist(
        proposal,
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        store=store,
        parent_dir=tmp_path,
    )

    assert first.created is True
    assert second.created is False
    assert second.value == first.value
    assert len(runtime.agent_calls) == call_count
    stored_proposal = await store.get_proposal(
        "user-1",
        "proposal-1",
    )
    assert stored_proposal is not None
    assert stored_proposal.status is ProposalStatus.validating


@pytest.mark.asyncio
async def test_persisted_failed_evaluation_rejects_proposal(
    tmp_path: Path,
) -> None:
    evaluator = NewSkillProposalEvaluator(
        runtime=FakeReplayRuntime(
            fail_candidate_events={"held-out-1"},
        )
    )
    store = InMemorySkillEvolutionStore()
    proposal = _proposal()
    await store.put_proposal(proposal)

    result = await evaluator.evaluate_and_persist(
        proposal,
        source_tasks=_source_tasks(),
        held_out_tasks=[_task("held-out-1")],
        store=store,
        parent_dir=tmp_path,
    )

    assert result.value.decision is EvaluationDecision.reject
    stored_proposal = await store.get_proposal(
        "user-1",
        "proposal-1",
    )
    assert stored_proposal is not None
    assert stored_proposal.status is ProposalStatus.rejected
