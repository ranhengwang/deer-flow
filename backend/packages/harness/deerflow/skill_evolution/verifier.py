"""Deterministic, pluggable outcome verification for evolution traces."""

from __future__ import annotations

import hashlib
import json
import re
from enum import IntEnum
from typing import Annotated, Protocol, runtime_checkable

from pydantic import Field, StringConstraints, model_validator

from deerflow.skill_evolution.models import (
    DetailText,
    EvolutionModel,
    EvolutionTraceSnapshot,
    Identifier,
    OutcomeCheck,
    OutcomeEvidence,
    OutcomeStatus,
    ShortText,
    TraceRunStatus,
    TraceToolEvent,
)

ArtifactPath = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1_024,
    ),
]

_EXIT_CODE_RE = re.compile(
    r"(?:^|\n)(?:Exit Code:|Command exited with code)\s*(-?\d+)\s*$",
    re.IGNORECASE,
)
_TIMEOUT_RE = re.compile(
    r"(?:command\s+)?timed out",
    re.IGNORECASE,
)
_TEST_COMMAND_PATTERNS = (
    re.compile(
        r"(?:^|[;&|]\s*|\s)(?:(?:uv|poetry|pipenv)\s+run\s+)?"
        r"(?:python(?:\d+(?:\.\d+)?)?\s+-m\s+)?pytest(?:\s|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[;&|]\s*|\s)(?:npm|pnpm|yarn|bun)\s+"
        r"(?:run\s+)?test(?:\s|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[;&|]\s*|\s)(?:go\s+test|cargo\s+test|mvn\s+test|"
        r"gradle\s+test|dotnet\s+test)(?:\s|$)",
        re.IGNORECASE,
    ),
)
_MAX_SIGNALS = 256


class EvidencePriority(IntEnum):
    """Relative authority used when deterministic evidence conflicts."""

    llm_supporting = 10
    user_acceptance = 60
    automated_check = 80
    task_specific = 90
    run_terminal = 100


class VerificationSignal(EvolutionModel):
    """One normalized verifier observation."""

    source: Identifier
    status: OutcomeStatus
    confidence: float = Field(ge=0.0, le=1.0)
    detail: DetailText
    priority: EvidencePriority
    authoritative: bool = True
    final_reward: float | None = None


class CommandExpectation(EvolutionModel):
    """Expected result for one explicitly selected command tool call."""

    tool_call_id: Identifier
    label: ShortText
    expected_exit_codes: tuple[int, ...] = Field(
        default=(0,),
        min_length=1,
        max_length=16,
    )
    confidence: float = Field(default=0.98, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_exit_codes(self) -> CommandExpectation:
        if len(set(self.expected_exit_codes)) != len(self.expected_exit_codes):
            raise ValueError("expected exit codes must be unique")
        return self


class ArtifactVerification(EvolutionModel):
    """Trusted result of an expected artifact check."""

    path: ArtifactPath
    exists: bool
    schema_valid: bool | None = None
    values_valid: bool | None = None
    detail: DetailText
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EnvironmentRewardEvidence(EvolutionModel):
    """Task-environment reward and deterministic decision thresholds."""

    value: float
    success_threshold: float
    failure_threshold: float
    detail: DetailText
    confidence: float = Field(default=0.99, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> EnvironmentRewardEvidence:
        if self.failure_threshold >= self.success_threshold:
            raise ValueError("failure_threshold must be lower than success_threshold")
        return self


class UserAcceptanceEvidence(EvolutionModel):
    """Explicit user acceptance or rejection supplied out of band."""

    accepted: bool | None
    detail: DetailText
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class LlmJudgmentEvidence(EvolutionModel):
    """Optional precomputed LLM judgment; never authoritative by default."""

    status: OutcomeStatus
    confidence: float = Field(ge=0.0, le=1.0)
    detail: DetailText


class VerificationContext(EvolutionModel):
    """Bounded inputs available to outcome-verifier plugins."""

    snapshot: EvolutionTraceSnapshot
    command_expectations: tuple[CommandExpectation, ...] = Field(
        default=(),
        max_length=32,
    )
    artifact_verifications: tuple[ArtifactVerification, ...] = Field(
        default=(),
        max_length=32,
    )
    environment_reward: EnvironmentRewardEvidence | None = None
    user_acceptance: UserAcceptanceEvidence | None = None
    llm_judgment: LlmJudgmentEvidence | None = None


@runtime_checkable
class OutcomeVerifier(Protocol):
    """Protocol for bounded outcome-verifier plugins."""

    name: str

    def verify(
        self,
        context: VerificationContext,
    ) -> tuple[VerificationSignal, ...]: ...


def _signal_source(prefix: str, value: str) -> str:
    source = f"{prefix}:{value}"
    if len(source) <= 128:
        return source
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{source[:115]}-{digest}"


def _tool_command(tool: TraceToolEvent) -> str | None:
    if tool.tool_name != "bash":
        return None
    try:
        arguments = json.loads(tool.arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict):
        return None
    command = arguments.get("command")
    return command if isinstance(command, str) and command else None


def _is_test_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in _TEST_COMMAND_PATTERNS)


def _exit_code(tool: TraceToolEvent) -> int | None:
    match = _EXIT_CODE_RE.search(tool.result)
    if match is not None:
        return int(match.group(1))
    if tool.status == "success":
        return 0
    return None


def _command_detail(
    label: str,
    command: str | None,
    suffix: str,
) -> str:
    command_text = (command or "<unavailable>")[:512]
    return f"{label}: {suffix}; command={command_text}"[:2_000]


def _command_signal(
    tool: TraceToolEvent,
    *,
    source: str,
    label: str,
    expected_exit_codes: tuple[int, ...],
    confidence: float,
    priority: EvidencePriority,
) -> VerificationSignal:
    command = _tool_command(tool)
    explicit_timeout = bool(_TIMEOUT_RE.search(tool.result))
    exit_code = _exit_code(tool)
    if explicit_timeout or tool.status == "error":
        status = OutcomeStatus.failure
        resolved_confidence = max(0.98, confidence)
        suffix = "command failed or timed out"
    elif exit_code is None:
        status = OutcomeStatus.unknown
        resolved_confidence = 0.0
        suffix = "exit code is unavailable"
    elif exit_code in expected_exit_codes:
        status = OutcomeStatus.success
        resolved_confidence = confidence
        suffix = f"exit code {exit_code} matched"
    else:
        status = OutcomeStatus.failure
        resolved_confidence = max(0.98, confidence)
        suffix = f"exit code {exit_code} did not match {list(expected_exit_codes)}"
    if tool.result_truncated and status is not OutcomeStatus.unknown:
        resolved_confidence = min(resolved_confidence, 0.85)
    return VerificationSignal(
        source=source,
        status=status,
        confidence=resolved_confidence,
        detail=_command_detail(label, command, suffix),
        priority=priority,
    )


class RunStatusVerifier:
    name = "run_status"

    def verify(
        self,
        context: VerificationContext,
    ) -> tuple[VerificationSignal, ...]:
        status = context.snapshot.run_status
        if status is TraceRunStatus.success:
            return ()
        return (
            VerificationSignal(
                source=self.name,
                status=OutcomeStatus.failure,
                confidence=1.0,
                detail=f"Run terminated with status {status.value}.",
                priority=EvidencePriority.run_terminal,
            ),
        )


class TestCommandVerifier:
    name = "test_command"

    def verify(
        self,
        context: VerificationContext,
    ) -> tuple[VerificationSignal, ...]:
        candidates = [tool for tool in context.snapshot.tool_events if (command := _tool_command(tool)) is not None and _is_test_command(command)]
        if not candidates:
            return ()
        latest = max(
            candidates,
            key=lambda tool: (tool.sequence, tool.tool_call_id),
        )
        return (
            _command_signal(
                latest,
                source=_signal_source(self.name, latest.tool_call_id),
                label="latest recognized test command",
                expected_exit_codes=(0,),
                confidence=0.95,
                priority=EvidencePriority.automated_check,
            ),
        )


class ExpectedCommandVerifier:
    name = "command"

    def verify(
        self,
        context: VerificationContext,
    ) -> tuple[VerificationSignal, ...]:
        tools_by_id = {tool.tool_call_id: tool for tool in context.snapshot.tool_events}
        signals: list[VerificationSignal] = []
        for expectation in context.command_expectations:
            source = _signal_source(
                self.name,
                expectation.tool_call_id,
            )
            tool = tools_by_id.get(expectation.tool_call_id)
            if tool is None:
                if context.snapshot.truncated:
                    signals.append(
                        VerificationSignal(
                            source=source,
                            status=OutcomeStatus.unknown,
                            confidence=0.0,
                            detail=(f"{expectation.label}: command is outside the bounded snapshot or was not executed."),
                            priority=EvidencePriority.task_specific,
                        )
                    )
                else:
                    signals.append(
                        VerificationSignal(
                            source=source,
                            status=OutcomeStatus.failure,
                            confidence=0.98,
                            detail=(f"{expectation.label}: expected command was not executed."),
                            priority=EvidencePriority.task_specific,
                        )
                    )
                continue
            signals.append(
                _command_signal(
                    tool,
                    source=source,
                    label=expectation.label,
                    expected_exit_codes=expectation.expected_exit_codes,
                    confidence=expectation.confidence,
                    priority=EvidencePriority.task_specific,
                )
            )
        return tuple(signals)


def _artifact_was_reported(
    artifacts: list[str],
    expected_path: str,
) -> bool:
    for artifact in artifacts:
        root = artifact.rstrip("/")
        if expected_path == root or expected_path.startswith(f"{root}/"):
            return True
    return False


class ArtifactVerifier:
    name = "artifact"

    def verify(
        self,
        context: VerificationContext,
    ) -> tuple[VerificationSignal, ...]:
        signals: list[VerificationSignal] = []
        for verification in context.artifact_verifications:
            source = _signal_source(self.name, verification.path)
            reported = _artifact_was_reported(
                context.snapshot.artifacts,
                verification.path,
            )
            if not reported:
                status = OutcomeStatus.unknown if context.snapshot.truncated else OutcomeStatus.failure
                confidence = 0.0 if status is OutcomeStatus.unknown else 0.98
                detail = f"{verification.detail} Artifact is not present in the run snapshot."
            elif not verification.exists:
                status = OutcomeStatus.failure
                confidence = verification.confidence if verification.confidence is not None else 0.99
                detail = verification.detail
            elif verification.schema_valid is False:
                status = OutcomeStatus.failure
                confidence = verification.confidence if verification.confidence is not None else 0.99
                detail = verification.detail
            elif verification.values_valid is False:
                status = OutcomeStatus.failure
                confidence = verification.confidence if verification.confidence is not None else 0.99
                detail = verification.detail
            else:
                status = OutcomeStatus.success
                has_structured_validation = verification.schema_valid is not None or verification.values_valid is not None
                confidence = verification.confidence if verification.confidence is not None else (0.98 if has_structured_validation else 0.85)
                detail = verification.detail
            signals.append(
                VerificationSignal(
                    source=source,
                    status=status,
                    confidence=confidence,
                    detail=detail[:2_000],
                    priority=EvidencePriority.task_specific,
                )
            )
        return tuple(signals)


class EnvironmentRewardVerifier:
    name = "environment_reward"

    def verify(
        self,
        context: VerificationContext,
    ) -> tuple[VerificationSignal, ...]:
        evidence = context.environment_reward
        if evidence is None:
            return ()
        if evidence.value >= evidence.success_threshold:
            status = OutcomeStatus.success
            confidence = evidence.confidence
        elif evidence.value <= evidence.failure_threshold:
            status = OutcomeStatus.failure
            confidence = evidence.confidence
        else:
            status = OutcomeStatus.unknown
            confidence = 0.0
        return (
            VerificationSignal(
                source=self.name,
                status=status,
                confidence=confidence,
                detail=evidence.detail,
                priority=EvidencePriority.task_specific,
                final_reward=evidence.value,
            ),
        )


class UserAcceptanceVerifier:
    name = "user_acceptance"

    def verify(
        self,
        context: VerificationContext,
    ) -> tuple[VerificationSignal, ...]:
        evidence = context.user_acceptance
        if evidence is None:
            return ()
        if evidence.accepted is True:
            status = OutcomeStatus.success
            confidence = evidence.confidence if evidence.confidence is not None else 0.75
        elif evidence.accepted is False:
            status = OutcomeStatus.failure
            confidence = evidence.confidence if evidence.confidence is not None else 0.9
        else:
            status = OutcomeStatus.unknown
            confidence = 0.0
        return (
            VerificationSignal(
                source=self.name,
                status=status,
                confidence=confidence,
                detail=evidence.detail,
                priority=EvidencePriority.user_acceptance,
            ),
        )


class LlmJudgmentVerifier:
    name = "llm_judge"

    def verify(
        self,
        context: VerificationContext,
    ) -> tuple[VerificationSignal, ...]:
        evidence = context.llm_judgment
        if evidence is None:
            return ()
        return (
            VerificationSignal(
                source=self.name,
                status=evidence.status,
                confidence=evidence.confidence,
                detail=evidence.detail,
                priority=EvidencePriority.llm_supporting,
                authoritative=False,
            ),
        )


DEFAULT_OUTCOME_VERIFIERS: tuple[OutcomeVerifier, ...] = (
    RunStatusVerifier(),
    TestCommandVerifier(),
    ExpectedCommandVerifier(),
    ArtifactVerifier(),
    EnvironmentRewardVerifier(),
    UserAcceptanceVerifier(),
    LlmJudgmentVerifier(),
)


def _dedupe_sources(
    signals: list[VerificationSignal],
) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        if signal.source in seen:
            continue
        seen.add(signal.source)
        sources.append(signal.source)
        if len(sources) >= 32:
            break
    return sources


def _selected_reward(
    signals: list[VerificationSignal],
) -> float | None:
    reward_signals = [signal for signal in signals if signal.final_reward is not None]
    if not reward_signals:
        return None
    selected = max(
        reward_signals,
        key=lambda signal: (signal.priority, signal.source),
    )
    return selected.final_reward


def _outcome_checks(
    signals: list[VerificationSignal],
) -> list[OutcomeCheck]:
    return [
        OutcomeCheck(
            source=signal.source,
            passed=signal.status is OutcomeStatus.success,
            detail=signal.detail,
            confidence=signal.confidence,
            authoritative=signal.authoritative,
            priority=int(signal.priority),
        )
        for signal in signals
        if signal.status is not OutcomeStatus.unknown
    ][:64]


def _aggregate_status(
    signals: list[VerificationSignal],
) -> tuple[OutcomeStatus, float]:
    authoritative = [signal for signal in signals if signal.authoritative and signal.status is not OutcomeStatus.unknown]
    if not authoritative:
        return OutcomeStatus.unknown, 0.0

    success = [signal for signal in authoritative if signal.status is OutcomeStatus.success]
    failure = [signal for signal in authoritative if signal.status is OutcomeStatus.failure]
    success_priority = max(
        (signal.priority for signal in success),
        default=None,
    )
    failure_priority = max(
        (signal.priority for signal in failure),
        default=None,
    )

    if success_priority is not None and failure_priority is not None and success_priority == failure_priority:
        success_confidence = max(signal.confidence for signal in success if signal.priority == success_priority)
        failure_confidence = max(signal.confidence for signal in failure if signal.priority == failure_priority)
        conflict_confidence = min(
            0.49,
            abs(success_confidence - failure_confidence) * 0.5,
        )
        return OutcomeStatus.unknown, conflict_confidence

    if failure_priority is None or (success_priority is not None and success_priority > failure_priority):
        status = OutcomeStatus.success
        winner_priority = success_priority
        winner_signals = success
        opposing_signals = failure
    else:
        status = OutcomeStatus.failure
        winner_priority = failure_priority
        winner_signals = failure
        opposing_signals = success

    assert winner_priority is not None
    confidence = max(signal.confidence for signal in winner_signals if signal.priority == winner_priority)
    if opposing_signals:
        strongest_opposition = max(
            opposing_signals,
            key=lambda signal: (signal.priority, signal.confidence),
        )
        priority_ratio = min(
            1.0,
            float(strongest_opposition.priority) / float(winner_priority),
        )
        confidence *= 1.0 - 0.1 * strongest_opposition.confidence * priority_ratio

    supporting_same = [signal for signal in signals if not signal.authoritative and signal.status is status]
    supporting_opposed = [signal for signal in signals if not signal.authoritative and signal.status not in {status, OutcomeStatus.unknown}]
    if supporting_same:
        support = max(signal.confidence for signal in supporting_same)
        confidence += (1.0 - confidence) * 0.03 * support
    if supporting_opposed:
        opposition = max(signal.confidence for signal in supporting_opposed)
        confidence *= 1.0 - 0.05 * opposition
    return status, round(
        max(0.0, min(1.0, confidence)),
        6,
    )


def aggregate_outcome_signals(
    signals: list[VerificationSignal],
) -> OutcomeEvidence:
    """Aggregate normalized signals into persisted outcome evidence."""
    bounded = signals[:_MAX_SIGNALS]
    status, confidence = _aggregate_status(bounded)
    return OutcomeEvidence(
        status=status,
        confidence=confidence,
        sources=_dedupe_sources(bounded),
        checks=_outcome_checks(bounded),
        final_reward=_selected_reward(bounded),
    )


def verify_outcome(
    context: VerificationContext,
    *,
    verifiers: tuple[OutcomeVerifier, ...] | None = None,
) -> OutcomeEvidence:
    """Run verifier plugins and conservatively aggregate their evidence."""
    selected = DEFAULT_OUTCOME_VERIFIERS if verifiers is None else verifiers
    signals: list[VerificationSignal] = []
    for verifier in selected:
        try:
            results = verifier.verify(context)
            for result in results:
                if not isinstance(result, VerificationSignal):
                    raise TypeError("verifier results must be VerificationSignal")
                signals.append(result)
                if len(signals) >= _MAX_SIGNALS:
                    break
        except Exception:
            verifier_name = str(getattr(verifier, "name", type(verifier).__name__))
            signals.append(
                VerificationSignal(
                    source=_signal_source(
                        "verifier_error",
                        verifier_name,
                    ),
                    status=OutcomeStatus.unknown,
                    confidence=0.0,
                    detail=(f"Verifier {verifier_name[:128]} failed; its evidence was ignored."),
                    priority=EvidencePriority.llm_supporting,
                    authoritative=False,
                )
            )
        if len(signals) >= _MAX_SIGNALS:
            break
    return aggregate_outcome_signals(signals)
