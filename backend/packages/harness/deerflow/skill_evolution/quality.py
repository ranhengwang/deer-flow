"""Deterministic, versioned quality scoring for Skill evaluations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from deerflow.config.skill_evolution_config import (
    SkillEvolutionQualityConfig,
)
from deerflow.skill_evolution.models import (
    EvaluationDecision,
    QualityDesignation,
    SkillEvaluation,
    SkillQualityDimension,
    SkillQualityReport,
    TaskEvaluationResult,
)

SKILL_QUALITY_FORMULA_VERSION = "skill-quality-v1"

_DIMENSION_WEIGHTS = {
    "success_lift": 0.25,
    "held_out_generalization": 0.20,
    "tool_call_reduction": 0.10,
    "token_reduction": 0.10,
    "latency_reduction": 0.10,
    "environment_robustness": 0.10,
    "regression_resistance": 0.10,
    "safety": 0.05,
}


def _clamp(
    value: float,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


def _symmetric_score(value: float) -> float:
    return (_clamp(value) + 1.0) / 2.0


def _available_dimension(
    name: str,
    *,
    raw_value: float,
    score: float,
    sample_count: int,
) -> SkillQualityDimension:
    return SkillQualityDimension(
        raw_value=_clamp(raw_value),
        score=_clamp(score, 0.0, 1.0),
        weight=_DIMENSION_WEIGHTS[name],
        sample_count=sample_count,
        available=True,
    )


def _unavailable_dimension(
    name: str,
) -> SkillQualityDimension:
    return SkillQualityDimension(
        weight=_DIMENSION_WEIGHTS[name],
        sample_count=0,
        available=False,
    )


def _pair_results(
    evaluation: SkillEvaluation,
) -> list[
    tuple[
        TaskEvaluationResult,
        TaskEvaluationResult,
    ]
]:
    baseline: dict[
        tuple[str, str],
        TaskEvaluationResult,
    ] = {}
    candidate: dict[
        tuple[str, str],
        TaskEvaluationResult,
    ] = {}
    for item in evaluation.baseline_results:
        key = (item.task_id, item.split)
        if key in baseline:
            raise ValueError("baseline results must be uniquely paired")
        baseline[key] = item
    for item in evaluation.candidate_results:
        key = (item.task_id, item.split)
        if key in candidate:
            raise ValueError("candidate results must be uniquely paired")
        candidate[key] = item
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate results must be paired")
    return [
        (
            baseline[key],
            candidate[key],
        )
        for key in sorted(baseline)
    ]


def _success_lift_dimension(
    pairs: list[
        tuple[
            TaskEvaluationResult,
            TaskEvaluationResult,
        ]
    ],
) -> SkillQualityDimension:
    if not pairs:
        return _unavailable_dimension("success_lift")
    baseline_rate = sum(baseline.success for baseline, _ in pairs) / len(pairs)
    candidate_rate = sum(candidate.success for _, candidate in pairs) / len(pairs)
    lift = candidate_rate - baseline_rate
    return _available_dimension(
        "success_lift",
        raw_value=lift,
        score=_symmetric_score(lift),
        sample_count=len(pairs),
    )


def _held_out_dimension(
    evaluation: SkillEvaluation,
) -> SkillQualityDimension:
    if not evaluation.held_out_results:
        return _unavailable_dimension("held_out_generalization")
    success_rate = sum(result.success for result in evaluation.held_out_results) / len(evaluation.held_out_results)
    return _available_dimension(
        "held_out_generalization",
        raw_value=success_rate,
        score=success_rate,
        sample_count=len(evaluation.held_out_results),
    )


def _relative_reduction_dimension(
    name: str,
    pairs: list[
        tuple[
            TaskEvaluationResult,
            TaskEvaluationResult,
        ]
    ],
    metric: Callable[[TaskEvaluationResult], float],
) -> SkillQualityDimension:
    successful_pairs = [(baseline, candidate) for baseline, candidate in pairs if baseline.success and candidate.success]
    if not successful_pairs:
        return _unavailable_dimension(name)
    baseline_total = sum(metric(baseline) for baseline, _ in successful_pairs)
    if baseline_total <= 0:
        return _unavailable_dimension(name)
    candidate_total = sum(metric(candidate) for _, candidate in successful_pairs)
    reduction = _clamp((baseline_total - candidate_total) / baseline_total)
    return _available_dimension(
        name,
        raw_value=reduction,
        score=_symmetric_score(reduction),
        sample_count=len(successful_pairs),
    )


def _environment_dimension(
    evaluation: SkillEvaluation,
) -> tuple[SkillQualityDimension, int, bool]:
    groups: dict[str, list[bool]] = defaultdict(list)
    missing = False
    for result in evaluation.candidate_results:
        fingerprint = result.environment_fingerprint
        if fingerprint is None:
            missing = True
            continue
        groups[fingerprint].append(result.success)
    if not groups:
        return (
            _unavailable_dimension("environment_robustness"),
            0,
            missing,
        )
    rates = [sum(values) / len(values) for values in groups.values()]
    robustness = min(rates)
    return (
        _available_dimension(
            "environment_robustness",
            raw_value=robustness,
            score=robustness,
            sample_count=len(groups),
        ),
        len(groups),
        missing,
    )


def _regression_dimension(
    pairs: list[
        tuple[
            TaskEvaluationResult,
            TaskEvaluationResult,
        ]
    ],
) -> tuple[SkillQualityDimension, int]:
    regression_pairs = [(baseline, candidate) for baseline, candidate in pairs if baseline.split == "regression" and baseline.success]
    if not regression_pairs:
        return (
            _unavailable_dimension("regression_resistance"),
            0,
        )
    regression_rate = sum(not candidate.success for _, candidate in regression_pairs) / len(regression_pairs)
    return (
        _available_dimension(
            "regression_resistance",
            raw_value=regression_rate,
            score=1.0 - regression_rate,
            sample_count=len(regression_pairs),
        ),
        len(regression_pairs),
    )


def _safety_dimension(
    evaluation: SkillEvaluation,
) -> tuple[SkillQualityDimension, str | None]:
    values = set(evaluation.safety_results.values())
    if "violation" in values:
        risk = 1.0
        blocker = "safety_violation"
    elif any(
        value
        in {
            "manual_review_required",
            "incomplete",
            "missing",
        }
        for value in values
    ):
        risk = 0.5
        blocker = "safety_review_required"
    else:
        risk = 0.0
        blocker = None
    return (
        _available_dimension(
            "safety",
            raw_value=risk,
            score=1.0 - risk,
            sample_count=1,
        ),
        blocker,
    )


def _aggregate_score(
    dimensions: dict[
        str,
        SkillQualityDimension,
    ],
) -> float:
    available = [dimension for dimension in dimensions.values() if dimension.available]
    total_weight = sum(dimension.weight for dimension in available)
    if total_weight <= 0:
        return 0.0
    weighted = sum(dimension.weight * (dimension.score or 0.0) for dimension in available)
    return round(weighted / total_weight, 6)


def compute_skill_quality(
    evaluation: SkillEvaluation,
    *,
    config: SkillEvolutionQualityConfig | None = None,
) -> SkillQualityReport:
    """Compute the deterministic v1 report from immutable raw results."""
    resolved = config or SkillEvolutionQualityConfig()
    pairs = _pair_results(evaluation)
    regression, regression_base_successes = _regression_dimension(pairs)
    environment, distinct_environments, missing_environment = _environment_dimension(evaluation)
    safety, safety_blocker = _safety_dimension(evaluation)
    dimensions = {
        "success_lift": _success_lift_dimension(pairs),
        "held_out_generalization": (_held_out_dimension(evaluation)),
        "tool_call_reduction": (
            _relative_reduction_dimension(
                "tool_call_reduction",
                pairs,
                lambda result: float(result.metrics.tool_calls),
            )
        ),
        "token_reduction": (
            _relative_reduction_dimension(
                "token_reduction",
                pairs,
                lambda result: float(result.metrics.input_tokens + result.metrics.output_tokens),
            )
        ),
        "latency_reduction": (
            _relative_reduction_dimension(
                "latency_reduction",
                pairs,
                lambda result: result.metrics.latency_seconds,
            )
        ),
        "environment_robustness": environment,
        "regression_resistance": regression,
        "safety": safety,
    }
    candidate_count = len(evaluation.candidate_results)
    held_out_count = len(evaluation.held_out_results)
    regression_count = len(evaluation.regression_results)
    sample_counts = {
        "candidate_tasks": candidate_count,
        "source_tasks": len(evaluation.source_replay_results),
        "held_out_tasks": held_out_count,
        "regression_tasks": regression_count,
        "regression_base_successes": (regression_base_successes),
        "distinct_environments": distinct_environments,
    }
    sample_blockers: list[str] = []
    if candidate_count < resolved.min_total_candidate_tasks:
        sample_blockers.append("insufficient_candidate_tasks")
    baseline_conditions = {result.condition for result in evaluation.baseline_results}
    is_new_skill = "no_skill" in baseline_conditions
    is_patch = "base_skill" in baseline_conditions
    if is_new_skill and (held_out_count < resolved.min_held_out_tasks):
        sample_blockers.append("insufficient_held_out_tasks")
    elif is_patch and (regression_base_successes < resolved.min_regression_tasks):
        sample_blockers.append("insufficient_regression_tasks")
    elif not is_new_skill and not is_patch:
        sample_blockers.append("missing_quality_branch")
    if distinct_environments < resolved.min_distinct_environments or missing_environment:
        sample_blockers.append("insufficient_environment_diversity")
    sample_blockers = list(dict.fromkeys(sample_blockers))
    sample_sufficient = not sample_blockers
    aggregate_score = _aggregate_score(dimensions)
    blockers = list(sample_blockers)
    if evaluation.decision is not EvaluationDecision.approve:
        blockers.append("evaluation_not_approved")
    if safety_blocker is not None:
        blockers.append(safety_blocker)
    if aggregate_score < resolved.high_quality_threshold:
        blockers.append("quality_below_threshold")
    blockers = list(dict.fromkeys(blockers))
    if not sample_sufficient:
        designation = QualityDesignation.insufficient_evidence
    elif blockers:
        designation = QualityDesignation.evaluated
    else:
        designation = QualityDesignation.high_quality
    return SkillQualityReport(
        formula_version=SKILL_QUALITY_FORMULA_VERSION,
        dimensions=dimensions,
        aggregate_score=aggregate_score,
        high_quality_threshold=(resolved.high_quality_threshold),
        designation=designation,
        sample_sufficient=sample_sufficient,
        sample_counts=sample_counts,
        blockers=blockers,
    )


def apply_skill_quality(
    evaluation: SkillEvaluation,
    *,
    config: SkillEvolutionQualityConfig | None = None,
) -> SkillEvaluation:
    """Return a scored copy while preserving every raw result."""
    report = compute_skill_quality(
        evaluation,
        config=config,
    )
    return SkillEvaluation.model_validate(
        {
            **evaluation.model_dump(mode="python"),
            "quality_score": report.aggregate_score,
            "quality": report,
        }
    )
