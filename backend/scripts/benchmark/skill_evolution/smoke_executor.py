"""Deterministic synthetic executor for benchmark plumbing tests only."""

from __future__ import annotations

import hashlib

from deerflow.skill_evolution.experiment import (
    BenchmarkTask,
    ExperimentCondition,
    ExperimentExecutionMode,
    ExperimentResult,
    ExperimentResultStatus,
)


class DeterministicSmokeExecutor:
    """Generate synthetic rows that can never pass as production replay."""

    async def execute(
        self,
        task: BenchmarkTask,
        condition: ExperimentCondition,
        seed: int,
    ) -> ExperimentResult:
        identity = "\0".join(
            (
                task.task_id,
                condition.condition_id,
                str(seed),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16)
        rejected = bucket % 29 == 0
        failed = not rejected and bucket % 31 == 0
        status = ExperimentResultStatus.rejected if rejected else (ExperimentResultStatus.failed if failed else ExperimentResultStatus.completed)
        condition_bonus = {
            "no_evolution": 0,
            "immediate_single": 1,
            "staged_single": 2,
            "proposed": 3,
            "custom": condition.evidence_threshold,
        }[condition.baseline_kind]
        success = status is ExperimentResultStatus.completed and (bucket + seed + task.variant_index + condition_bonus) % 5 >= 2
        proposal_created = condition.baseline_kind != "no_evolution" and status is ExperimentResultStatus.completed
        result_id = f"result-{digest[:32]}"
        return ExperimentResult(
            result_id=result_id,
            experiment_id="skill-evolution-phase11-smoke",
            condition_id=condition.condition_id,
            task_id=task.task_id,
            domain=task.domain,
            family_id=task.family_id,
            split=task.split,
            variant_index=task.variant_index,
            seed=seed,
            execution_mode=(ExperimentExecutionMode.deterministic_smoke),
            executor_name="deterministic-smoke",
            executor_version="deterministic-smoke-v1",
            model_name=None,
            model_version=None,
            prompt_version="synthetic-no-model-v1",
            environment_fingerprint=hashlib.sha256(b"deterministic-smoke-environment-v1").hexdigest(),
            status=status,
            task_success=success,
            regression_eligible=(condition.baseline_kind != "no_evolution"),
            regression=(proposal_created and not success and bucket % 7 == 0),
            incorrect_evolution=(proposal_created and not success and bucket % 11 == 0),
            cluster_true_positive=(condition.evidence_threshold if proposal_created else 0),
            cluster_false_positive=(1 if proposal_created and bucket % 13 == 0 else 0),
            tool_calls=2 + bucket % 8,
            input_tokens=100 + bucket % 500,
            output_tokens=20 + bucket % 120,
            latency_seconds=0.5 + (bucket % 200) / 100,
            proposal_created=proposal_created,
            proposal_accepted=proposal_created and success,
            security_rejected=rejected,
            evidence_count=(condition.evidence_threshold if proposal_created else 0),
            evolution_delay_seconds=(float(condition.evidence_threshold * 10) if proposal_created else None),
            error_code=("synthetic_security_rejection" if rejected else ("synthetic_executor_failure" if failed else None)),
        )


def create_executor() -> DeterministicSmokeExecutor:
    return DeterministicSmokeExecutor()
