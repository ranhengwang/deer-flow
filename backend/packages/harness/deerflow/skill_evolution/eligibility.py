"""Eligibility and complexity detection for verified evolution traces."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from deerflow.config.skill_evolution_config import (
    SkillEvolutionEvidenceConfig,
)
from deerflow.skill_evolution.models import (
    ComplexitySignals,
    EvolutionModel,
    EvolutionTraceSnapshot,
    Identifier,
    OutcomeEvidence,
    OutcomeStatus,
)

EXCLUDED_STOP_REASONS = frozenset(
    {
        "loop_capped",
        "safety_capped",
        "subagent_limit_capped",
        "token_capped",
    }
)

_EXPLICIT_REQUEST_PATTERNS = (
    re.compile(
        r"\bremember\s+(?:this|the)\s+"
        r"(?:workflow|process|procedure|steps?|method)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:create|make|build|generate)\s+(?:a\s+)?"
        r"(?:new\s+)?skill\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:save|turn|convert)\s+(?:this|the).{0,40}"
        r"\b(?:as|into)\s+(?:a\s+)?skill\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"记住.{0,24}(?:流程|方法|步骤|做法)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:创建|新建|生成|制作).{0,12}(?:skill|技能)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:把|将).{0,40}(?:流程|方法|步骤|做法).{0,30}"
        r"(?:沉淀|保存|整理|转成|做成).{0,20}(?:skill|技能)",
        re.IGNORECASE,
    ),
)
_NEGATED_REQUEST_PREFIX_RE = re.compile(
    r"(?:do\s+not|don't|dont|never|not|不要|无需|不需要|别)"
    r"[\s，,、]*(?:\w+[\s，,、]*){0,2}$",
    re.IGNORECASE,
)


class EligibilityBranch(StrEnum):
    """Evolution branch selected from observed Skill usage."""

    no_skill = "no_skill"
    skill_used = "skill_used"


class EligibilityHints(EvolutionModel):
    """Trusted extraction flags not derivable from the bounded snapshot."""

    had_user_correction: bool = False
    non_trivial_workflow: bool = False
    explicit_remember_request: bool = False


class EligibilityDecision(EvolutionModel):
    """Auditable result of the evolution admission gate."""

    eligible: bool
    branch: EligibilityBranch
    outcome_status: OutcomeStatus
    outcome_confidence: float = Field(ge=0.0, le=1.0)
    min_success_confidence: float = Field(ge=0.0, le=1.0)
    complexity: ComplexitySignals
    qualifying_signals: list[Identifier] = Field(
        default_factory=list,
        max_length=8,
    )
    rejection_reasons: list[Identifier] = Field(
        default_factory=list,
        max_length=8,
    )
    excluded_stop_reason: Identifier | None = None

    @model_validator(mode="after")
    def _validate_decision(self) -> Self:
        if self.eligible:
            if self.rejection_reasons:
                raise ValueError("eligible decision must not contain rejection reasons")
            if not self.qualifying_signals:
                raise ValueError("eligible decision requires a qualifying signal")
        elif not self.rejection_reasons:
            raise ValueError("ineligible decision requires a rejection reason")
        return self


def _has_recovered_error(
    snapshot: EvolutionTraceSnapshot,
) -> bool:
    ordered = sorted(
        snapshot.tool_events,
        key=lambda tool: (tool.sequence, tool.tool_call_id),
    )
    for index, tool in enumerate(ordered):
        if tool.status != "error":
            continue
        if any(later.status == "success" for later in ordered[index + 1 :]):
            return True
    return False


def _has_explicit_remember_request(task_input: str) -> bool:
    for pattern in _EXPLICIT_REQUEST_PATTERNS:
        for match in pattern.finditer(task_input):
            prefix = task_input[max(0, match.start() - 32) : match.start()]
            if _NEGATED_REQUEST_PREFIX_RE.search(prefix):
                continue
            return True
    return False


def _complexity_signals(
    snapshot: EvolutionTraceSnapshot,
    hints: EligibilityHints,
) -> ComplexitySignals:
    return ComplexitySignals(
        tool_calls=len(snapshot.tool_events),
        had_recoverable_errors=_has_recovered_error(snapshot),
        had_user_correction=(bool(snapshot.user_corrections) or hints.had_user_correction),
        non_trivial_workflow=hints.non_trivial_workflow,
        explicit_remember_request=(hints.explicit_remember_request or _has_explicit_remember_request(snapshot.task_input)),
    )


def _qualifying_signals(
    complexity: ComplexitySignals,
    config: SkillEvolutionEvidenceConfig,
) -> list[str]:
    signals: list[str] = []
    if complexity.tool_calls >= config.tool_call_complexity_threshold:
        signals.append("tool_call_threshold")
    if complexity.had_recoverable_errors and config.accept_recovered_errors:
        signals.append("recovered_error")
    if complexity.had_user_correction and config.accept_user_corrections:
        signals.append("user_correction")
    if complexity.non_trivial_workflow and config.accept_non_trivial_workflow:
        signals.append("non_trivial_workflow")
    if complexity.explicit_remember_request and config.accept_explicit_remember_requests:
        signals.append("explicit_remember_request")
    return signals


def evaluate_evolution_eligibility(
    snapshot: EvolutionTraceSnapshot,
    outcome: OutcomeEvidence,
    *,
    config: SkillEvolutionEvidenceConfig | None = None,
    hints: EligibilityHints | None = None,
) -> EligibilityDecision:
    """Decide whether a verified trace is eligible for event extraction."""
    resolved_config = config or SkillEvolutionEvidenceConfig()
    resolved_hints = hints or EligibilityHints()
    complexity = _complexity_signals(snapshot, resolved_hints)
    qualifying_signals = _qualifying_signals(
        complexity,
        resolved_config,
    )
    rejection_reasons: list[str] = []

    if outcome.status is not OutcomeStatus.success:
        rejection_reasons.append("outcome_not_success")
    elif outcome.confidence < resolved_config.min_success_confidence:
        rejection_reasons.append("confidence_below_threshold")

    excluded_stop_reason = snapshot.stop_reason if snapshot.stop_reason in EXCLUDED_STOP_REASONS else None
    if excluded_stop_reason is not None:
        rejection_reasons.append("excluded_stop_reason")
    if not qualifying_signals:
        rejection_reasons.append("insufficient_complexity")

    branch = EligibilityBranch.skill_used if snapshot.skill_events else EligibilityBranch.no_skill
    return EligibilityDecision(
        eligible=not rejection_reasons,
        branch=branch,
        outcome_status=outcome.status,
        outcome_confidence=outcome.confidence,
        min_success_confidence=(resolved_config.min_success_confidence),
        complexity=complexity,
        qualifying_signals=qualifying_signals,
        rejection_reasons=rejection_reasons,
        excluded_stop_reason=excluded_stop_reason,
    )
