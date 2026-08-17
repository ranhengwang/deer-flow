from __future__ import annotations

from datetime import UTC, datetime

from deerflow.skill_evolution.models import (
    EnvironmentSignature,
    EvolutionTraceSnapshot,
    OutcomeStatus,
    TraceRunStatus,
    TraceToolEvent,
)
from deerflow.skill_evolution.trajectory import (
    build_evolution_trace_snapshot,
)
from deerflow.skill_evolution.verifier import (
    ArtifactVerification,
    CommandExpectation,
    EnvironmentRewardEvidence,
    EvidencePriority,
    LlmJudgmentEvidence,
    OutcomeVerifier,
    UserAcceptanceEvidence,
    VerificationContext,
    VerificationSignal,
    verify_outcome,
)


def _tool(
    sequence: int,
    tool_call_id: str,
    *,
    command: str,
    result: str,
    status: str = "success",
    truncated: bool = False,
) -> TraceToolEvent:
    return TraceToolEvent(
        sequence=sequence,
        tool_call_id=tool_call_id,
        tool_name="bash",
        arguments=f'{{"command":"{command}"}}',
        result=result,
        status=status,
        result_truncated=truncated,
    )


def _snapshot(
    *,
    status: TraceRunStatus = TraceRunStatus.success,
    final_answer: str = "I completed the task successfully.",
    tools: list[TraceToolEvent] | None = None,
    artifacts: list[str] | None = None,
    truncated: bool = False,
) -> EvolutionTraceSnapshot:
    included = 8
    source = 12 if truncated else included
    return EvolutionTraceSnapshot(
        snapshot_hash="a" * 64,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        model_name="test-model",
        run_status=status,
        stop_reason=None,
        task_input="Repair the package and verify it.",
        final_answer=final_answer,
        tool_events=tools or [],
        skill_events=[],
        user_corrections=[],
        artifacts=artifacts or [],
        environment=EnvironmentSignature(
            os="macos",
            shell="zsh",
            runtime="python3.12",
        ),
        source_event_count=source,
        included_event_count=included,
        truncated=truncated,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


def test_final_answer_and_success_run_status_are_not_success_evidence() -> None:
    outcome = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(
                final_answer="Everything passed. The task is complete.",
            )
        )
    )

    assert outcome.status is OutcomeStatus.unknown
    assert outcome.confidence == 0.0
    assert outcome.checks == []


def test_non_success_run_status_is_authoritative_failure() -> None:
    outcome = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(status=TraceRunStatus.error),
        )
    )

    assert outcome.status is OutcomeStatus.failure
    assert outcome.confidence == 1.0
    assert outcome.sources == ["run_status"]


def test_latest_recognized_test_command_is_used() -> None:
    snapshot = _snapshot(
        tools=[
            _tool(
                1,
                "test-1",
                command="uv run pytest tests/test_widget.py -q",
                result="1 failed\nExit Code: 1",
            ),
            _tool(
                2,
                "edit-1",
                command="printf fixed",
                result="fixed",
            ),
            _tool(
                3,
                "test-2",
                command="uv run pytest tests/test_widget.py -q",
                result="12 passed",
            ),
        ]
    )

    outcome = verify_outcome(VerificationContext(snapshot=snapshot))

    assert outcome.status is OutcomeStatus.success
    assert outcome.confidence == 0.95
    assert outcome.sources == ["test_command:test-2"]
    assert outcome.checks[0].confidence == 0.95
    assert outcome.checks[0].authoritative is True
    assert outcome.checks[0].priority == 80


def test_run_events_flow_through_snapshot_builder_into_verifier() -> None:
    created_at = datetime(2026, 8, 16, tzinfo=UTC).isoformat()
    events = [
        {
            "seq": 1,
            "event_type": "llm.human.input",
            "content": {
                "type": "human",
                "content": "Repair the package.",
                "additional_kwargs": {},
            },
            "metadata": {"caller": "lead_agent"},
            "created_at": created_at,
        },
        {
            "seq": 2,
            "event_type": "llm.ai.response",
            "content": {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "id": "test-real",
                        "name": "bash",
                        "args": {"command": ("uv run pytest tests/test_package.py -q")},
                    }
                ],
                "additional_kwargs": {},
            },
            "metadata": {"caller": "lead_agent"},
            "created_at": created_at,
        },
        {
            "seq": 3,
            "event_type": "llm.tool.result",
            "content": {
                "type": "tool",
                "name": "bash",
                "tool_call_id": "test-real",
                "status": "success",
                "content": "7 passed",
                "additional_kwargs": {},
            },
            "metadata": {},
            "created_at": created_at,
        },
        {
            "seq": 4,
            "event_type": "llm.ai.response",
            "content": {
                "type": "ai",
                "content": "The package is repaired.",
                "tool_calls": [],
                "additional_kwargs": {},
            },
            "metadata": {"caller": "lead_agent"},
            "created_at": created_at,
        },
    ]
    snapshot = build_evolution_trace_snapshot(
        events,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        model_name="test-model",
        run_status=TraceRunStatus.success,
        stop_reason=None,
        environment={"os": "macos"},
    )

    outcome = verify_outcome(VerificationContext(snapshot=snapshot))

    assert outcome.status is OutcomeStatus.success
    assert outcome.sources == ["test_command:test-real"]


def test_latest_failed_test_command_is_failure() -> None:
    snapshot = _snapshot(
        tools=[
            _tool(
                1,
                "test-1",
                command="pnpm test",
                result="3 failed\nExit Code: 1",
            )
        ]
    )

    outcome = verify_outcome(VerificationContext(snapshot=snapshot))

    assert outcome.status is OutcomeStatus.failure
    assert outcome.confidence == 0.98


def test_explicit_command_expectation_handles_success_and_missing_evidence() -> None:
    snapshot = _snapshot(
        tools=[
            _tool(
                1,
                "verify-import",
                command="python -c 'import package'",
                result="(no output)",
            )
        ]
    )
    success = verify_outcome(
        VerificationContext(
            snapshot=snapshot,
            command_expectations=(
                CommandExpectation(
                    tool_call_id="verify-import",
                    label="package import",
                ),
            ),
        )
    )
    missing = verify_outcome(
        VerificationContext(
            snapshot=snapshot,
            command_expectations=(
                CommandExpectation(
                    tool_call_id="missing-command",
                    label="schema check",
                ),
            ),
        )
    )

    assert success.status is OutcomeStatus.success
    assert success.sources == ["command:verify-import"]
    assert missing.status is OutcomeStatus.failure
    assert missing.confidence == 0.98


def test_missing_expected_command_is_unknown_when_snapshot_is_truncated() -> None:
    outcome = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(truncated=True),
            command_expectations=(
                CommandExpectation(
                    tool_call_id="outside-window",
                    label="focused tests",
                ),
            ),
        )
    )

    assert outcome.status is OutcomeStatus.unknown
    assert outcome.sources == ["command:outside-window"]


def test_artifact_existence_and_schema_are_task_specific_evidence() -> None:
    success = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(
                artifacts=["/mnt/user-data/outputs/report.json"],
            ),
            artifact_verifications=(
                ArtifactVerification(
                    path="/mnt/user-data/outputs/report.json",
                    exists=True,
                    schema_valid=True,
                    values_valid=True,
                    detail="Report matches the required schema and values.",
                ),
            ),
        )
    )
    invalid = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(
                artifacts=["/mnt/user-data/outputs/report.json"],
            ),
            artifact_verifications=(
                ArtifactVerification(
                    path="/mnt/user-data/outputs/report.json",
                    exists=True,
                    schema_valid=False,
                    detail="Required field 'rows' is missing.",
                ),
            ),
        )
    )

    assert success.status is OutcomeStatus.success
    assert success.confidence == 0.98
    assert invalid.status is OutcomeStatus.failure
    assert invalid.confidence == 0.99


def test_environment_reward_is_preserved_and_thresholded() -> None:
    success = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(),
            environment_reward=EnvironmentRewardEvidence(
                value=0.9,
                success_threshold=0.8,
                failure_threshold=0.2,
                detail="Environment score reached the target.",
            ),
        )
    )
    undecided = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(),
            environment_reward=EnvironmentRewardEvidence(
                value=0.5,
                success_threshold=0.8,
                failure_threshold=0.2,
                detail="Environment score is inconclusive.",
            ),
        )
    )

    assert success.status is OutcomeStatus.success
    assert success.final_reward == 0.9
    assert undecided.status is OutcomeStatus.unknown
    assert undecided.final_reward == 0.5


def test_explicit_user_acceptance_is_lower_confidence_success() -> None:
    outcome = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(),
            user_acceptance=UserAcceptanceEvidence(
                accepted=True,
                detail="User explicitly confirmed the generated report.",
            ),
        )
    )

    assert outcome.status is OutcomeStatus.success
    assert outcome.confidence == 0.75


def test_explicit_zero_confidence_is_not_replaced_by_default() -> None:
    outcome = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(),
            user_acceptance=UserAcceptanceEvidence(
                accepted=True,
                confidence=0.0,
                detail="Acceptance source is not trusted.",
            ),
        )
    )

    assert outcome.status is OutcomeStatus.success
    assert outcome.confidence == 0.0


def test_llm_judgment_alone_cannot_mark_success() -> None:
    outcome = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(),
            llm_judgment=LlmJudgmentEvidence(
                status=OutcomeStatus.success,
                confidence=0.99,
                detail="The response appears correct.",
            ),
        )
    )

    assert outcome.status is OutcomeStatus.unknown
    assert outcome.sources == ["llm_judge"]


def test_broken_plugin_isolated_as_unknown_evidence() -> None:
    class BrokenVerifier:
        name = "broken"

        def verify(
            self,
            context: VerificationContext,
        ) -> tuple[VerificationSignal, ...]:
            raise RuntimeError("private runtime detail")

    outcome = verify_outcome(
        VerificationContext(snapshot=_snapshot()),
        verifiers=(BrokenVerifier(),),
    )

    assert outcome.status is OutcomeStatus.unknown
    assert outcome.sources == ["verifier_error:broken"]
    assert "private runtime detail" not in outcome.model_dump_json()


def test_equal_priority_conflict_yields_unknown() -> None:
    class ConflictingVerifier:
        name = "conflict"

        def verify(
            self,
            context: VerificationContext,
        ) -> tuple[VerificationSignal, ...]:
            return (
                VerificationSignal(
                    source="deterministic-pass",
                    status=OutcomeStatus.success,
                    confidence=0.95,
                    detail="Schema is valid.",
                    priority=EvidencePriority.task_specific,
                ),
                VerificationSignal(
                    source="deterministic-fail",
                    status=OutcomeStatus.failure,
                    confidence=0.9,
                    detail="Expected value is wrong.",
                    priority=EvidencePriority.task_specific,
                ),
            )

    verifier: OutcomeVerifier = ConflictingVerifier()
    outcome = verify_outcome(
        VerificationContext(snapshot=_snapshot()),
        verifiers=(verifier,),
    )

    assert outcome.status is OutcomeStatus.unknown
    assert outcome.confidence < 0.5


def test_lower_priority_rejection_reduces_deterministic_success_confidence() -> None:
    outcome = verify_outcome(
        VerificationContext(
            snapshot=_snapshot(
                tools=[
                    _tool(
                        1,
                        "test-1",
                        command="go test ./...",
                        result="ok example/project",
                    )
                ]
            ),
            user_acceptance=UserAcceptanceEvidence(
                accepted=False,
                detail="User rejected the delivered behavior.",
            ),
        )
    )

    assert outcome.status is OutcomeStatus.success
    assert 0.8 < outcome.confidence < 0.95
