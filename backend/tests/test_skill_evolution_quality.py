from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from deerflow.config.skill_evolution_config import (
    SkillEvolutionQualityConfig,
)
from deerflow.skill_evolution.models import (
    EvaluationDecision,
    EvaluationMetrics,
    QualityDesignation,
    SkillEvaluation,
    SkillQualityDimension,
    SkillQualityReport,
    TaskEvaluationResult,
)
from deerflow.skill_evolution.quality import (
    SKILL_QUALITY_FORMULA_VERSION,
    apply_skill_quality,
    compute_skill_quality,
)

_CREATED = datetime(2026, 8, 17, tzinfo=UTC)
_ENV_A = "a" * 64
_ENV_B = "b" * 64


def _result(
    task_id: str,
    *,
    split: str,
    condition: str,
    success: bool,
    environment: str | None,
    tool_calls: int = 10,
    input_tokens: int = 100,
    output_tokens: int = 50,
    latency_seconds: float = 10.0,
) -> TaskEvaluationResult:
    return TaskEvaluationResult(
        task_id=task_id,
        split=split,
        condition=condition,
        success=success,
        metrics=EvaluationMetrics(
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
        ),
        failure_reason=None if success else "verification_failed",
        errors=[] if success else ["verification_failed"],
        environment_fingerprint=environment,
    )


def _new_skill_evaluation(
    *,
    held_out_count: int = 2,
    environments: tuple[str, ...] = (_ENV_A, _ENV_B),
    decision: EvaluationDecision = EvaluationDecision.approve,
    side_effects: str = "allow",
) -> SkillEvaluation:
    task_specs = [
        ("source-1", "source"),
        ("source-2", "source"),
        ("source-3", "source"),
        *[(f"held-{index}", "held_out") for index in range(1, held_out_count + 1)],
    ]
    baseline: list[TaskEvaluationResult] = []
    candidate: list[TaskEvaluationResult] = []
    for index, (task_id, split) in enumerate(task_specs):
        environment = environments[index % len(environments)]
        baseline.append(
            _result(
                task_id,
                split=split,
                condition="no_skill",
                success=split == "held_out",
                environment=environment,
            )
        )
        candidate.append(
            _result(
                task_id,
                split=split,
                condition="candidate_skill",
                success=True,
                environment=environment,
                tool_calls=5,
                input_tokens=50,
                output_tokens=25,
                latency_seconds=5.0,
            )
        )
    source = [result for result in candidate if result.split == "source"]
    held_out = [result for result in candidate if result.split == "held_out"]
    return SkillEvaluation(
        evaluation_id="evaluation-new",
        proposal_id="proposal-new",
        user_id="user-1",
        source_replay_results=source,
        held_out_results=held_out,
        baseline_results=baseline,
        candidate_results=candidate,
        regression_results=[],
        safety_results={"side_effects": side_effects},
        quality_score=0.0,
        decision=decision,
        created_at=_CREATED,
    )


def _patch_evaluation(
    *,
    regression_failures: int = 0,
    decision: EvaluationDecision = EvaluationDecision.approve,
) -> SkillEvaluation:
    task_specs = [
        ("source-1", "source"),
        ("source-2", "source"),
        ("source-3", "source"),
        ("regression-1", "regression"),
        ("regression-2", "regression"),
    ]
    baseline: list[TaskEvaluationResult] = []
    candidate: list[TaskEvaluationResult] = []
    for index, (task_id, split) in enumerate(task_specs):
        environment = (_ENV_A, _ENV_B)[index % 2]
        baseline.append(
            _result(
                task_id,
                split=split,
                condition="base_skill",
                success=split == "regression",
                environment=environment,
            )
        )
        candidate_success = not (split == "regression" and int(task_id.rsplit("-", 1)[1]) <= regression_failures)
        candidate.append(
            _result(
                task_id,
                split=split,
                condition="candidate_skill",
                success=candidate_success,
                environment=environment,
                tool_calls=5,
                input_tokens=50,
                output_tokens=25,
                latency_seconds=5.0,
            )
        )
    return SkillEvaluation(
        evaluation_id="evaluation-patch",
        proposal_id="proposal-patch",
        user_id="user-1",
        source_replay_results=[result for result in candidate if result.split == "source"],
        held_out_results=[],
        baseline_results=baseline,
        candidate_results=candidate,
        regression_results=[result for result in candidate if result.split == "regression"],
        safety_results={"side_effects": "allow"},
        quality_score=0.0,
        decision=decision,
        created_at=_CREATED,
    )


def test_formula_v1_exposes_all_versioned_dimensions() -> None:
    report = compute_skill_quality(_new_skill_evaluation())

    assert report.formula_version == SKILL_QUALITY_FORMULA_VERSION
    assert set(report.dimensions) == {
        "success_lift",
        "held_out_generalization",
        "tool_call_reduction",
        "token_reduction",
        "latency_reduction",
        "environment_robustness",
        "regression_resistance",
        "safety",
    }
    assert report.dimensions["regression_resistance"].available is False
    assert report.dimensions["success_lift"].raw_value == pytest.approx(0.6)
    assert report.dimensions["success_lift"].score == pytest.approx(0.8)
    assert report.dimensions["tool_call_reduction"].raw_value == pytest.approx(0.5)


def test_new_skill_with_held_out_and_environment_evidence_can_be_high_quality() -> None:
    report = compute_skill_quality(_new_skill_evaluation())

    assert report.sample_sufficient is True
    assert report.aggregate_score >= 0.75
    assert report.designation is QualityDesignation.high_quality
    assert report.high_quality is True
    assert report.sample_counts == {
        "candidate_tasks": 5,
        "source_tasks": 3,
        "held_out_tasks": 2,
        "regression_tasks": 0,
        "regression_base_successes": 0,
        "distinct_environments": 2,
    }
    assert report.blockers == []


def test_three_source_runs_alone_can_never_be_high_quality() -> None:
    report = compute_skill_quality(
        _new_skill_evaluation(
            held_out_count=0,
            decision=EvaluationDecision.manual_review,
        )
    )

    assert report.sample_sufficient is False
    assert report.high_quality is False
    assert report.designation is QualityDesignation.insufficient_evidence
    assert "insufficient_candidate_tasks" in report.blockers
    assert "insufficient_held_out_tasks" in report.blockers


def test_one_held_out_sample_is_insufficient() -> None:
    report = compute_skill_quality(_new_skill_evaluation(held_out_count=1))

    assert report.sample_sufficient is False
    assert "insufficient_held_out_tasks" in report.blockers
    assert report.high_quality is False


def test_one_environment_is_insufficient_for_high_quality() -> None:
    report = compute_skill_quality(
        _new_skill_evaluation(
            environments=(_ENV_A,),
        )
    )

    assert report.sample_sufficient is False
    assert "insufficient_environment_diversity" in report.blockers
    assert report.high_quality is False


def test_patch_quality_uses_regression_resistance_instead_of_held_out() -> None:
    report = compute_skill_quality(_patch_evaluation())

    assert report.dimensions["held_out_generalization"].available is False
    regression = report.dimensions["regression_resistance"]
    assert regression.available is True
    assert regression.raw_value == 0.0
    assert regression.score == 1.0
    assert report.sample_sufficient is True
    assert report.designation is QualityDesignation.high_quality


def test_regression_and_rejected_decision_block_high_quality() -> None:
    report = compute_skill_quality(
        _patch_evaluation(
            regression_failures=1,
            decision=EvaluationDecision.reject,
        )
    )

    assert report.dimensions["regression_resistance"].raw_value == 0.5
    assert report.dimensions["regression_resistance"].score == 0.5
    assert report.high_quality is False
    assert "evaluation_not_approved" in report.blockers


def test_safety_review_blocks_high_quality() -> None:
    evaluation = _new_skill_evaluation(
        decision=EvaluationDecision.manual_review,
    ).model_copy(
        update={
            "safety_results": {
                "side_effects": "allow",
                "proposal_review": "manual_review_required",
            }
        }
    )

    report = compute_skill_quality(evaluation)

    assert report.dimensions["safety"].raw_value == 0.5
    assert report.dimensions["safety"].score == 0.5
    assert report.high_quality is False
    assert "safety_review_required" in report.blockers


def test_efficiency_dimensions_ignore_unsuccessful_pairs() -> None:
    evaluation = _new_skill_evaluation()
    baseline = [
        item.model_copy(
            update={
                "success": False,
                "failure_reason": "failed",
                "errors": ["failed"],
            }
        )
        for item in evaluation.baseline_results
    ]
    candidate = [
        item.model_copy(
            update={
                "success": False,
                "failure_reason": "failed",
                "errors": ["failed"],
                "metrics": EvaluationMetrics(
                    tool_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    latency_seconds=0,
                ),
            }
        )
        for item in evaluation.candidate_results
    ]
    evaluation = evaluation.model_copy(
        update={
            "baseline_results": baseline,
            "candidate_results": candidate,
            "source_replay_results": candidate[:3],
            "held_out_results": candidate[3:],
            "decision": EvaluationDecision.reject,
        }
    )

    report = compute_skill_quality(evaluation)

    assert report.dimensions["tool_call_reduction"].available is False
    assert report.dimensions["token_reduction"].available is False
    assert report.dimensions["latency_reduction"].available is False


def test_apply_quality_preserves_raw_results_and_sets_aggregate() -> None:
    evaluation = _new_skill_evaluation()
    raw_before = copy.deepcopy(evaluation.candidate_results)

    scored = apply_skill_quality(evaluation)

    assert scored.candidate_results == raw_before
    assert scored.quality is not None
    assert scored.quality_score == scored.quality.aggregate_score
    assert scored.quality.formula_version == SKILL_QUALITY_FORMULA_VERSION
    assert evaluation.quality is None
    assert evaluation.quality_score == 0.0


def test_unpaired_raw_results_fail_closed() -> None:
    evaluation = _new_skill_evaluation()
    evaluation = evaluation.model_copy(
        update={
            "baseline_results": evaluation.baseline_results[:-1],
        }
    )

    with pytest.raises(
        ValueError,
        match="paired",
    ):
        compute_skill_quality(evaluation)


def test_quality_model_rejects_forged_high_quality_designation() -> None:
    dimension = SkillQualityDimension(
        raw_value=1.0,
        score=1.0,
        weight=1.0,
        sample_count=1,
        available=True,
    )

    with pytest.raises(ValidationError, match="high quality"):
        SkillQualityReport(
            formula_version=SKILL_QUALITY_FORMULA_VERSION,
            dimensions={"safety": dimension},
            aggregate_score=1.0,
            high_quality_threshold=0.75,
            designation=QualityDesignation.high_quality,
            sample_sufficient=False,
            sample_counts={"candidate_tasks": 1},
            blockers=["insufficient_candidate_tasks"],
        )


def test_configured_high_quality_threshold_is_applied() -> None:
    report = compute_skill_quality(
        _new_skill_evaluation(),
        config=SkillEvolutionQualityConfig(
            high_quality_threshold=0.9,
        ),
    )

    assert report.aggregate_score < 0.9
    assert report.designation is QualityDesignation.evaluated
    assert "quality_below_threshold" in report.blockers
