from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from deerflow.skill_evolution.experiment import (
    EXPERIMENT_RESULT_SCHEMA_VERSION,
    ExperimentCondition,
    ExperimentExecutionMode,
    ExperimentResult,
    ExperimentResultStatus,
    build_default_conditions,
    build_default_manifest,
    build_experiment_report,
    compare_conditions,
    read_results_jsonl,
    run_experiment,
    summarize_condition,
    write_results_jsonl,
)


def _result(
    *,
    condition_id: str,
    task_id: str,
    family_id: str = "repo-preflight",
    domain: str = "repository_repair",
    variant_index: int = 1,
    seed: int = 1,
    success: bool,
    split: str = "held_out",
    status: ExperimentResultStatus = (ExperimentResultStatus.completed),
    regression: bool = False,
    incorrect_evolution: bool = False,
    cluster_tp: int = 1,
    cluster_fp: int = 0,
) -> ExperimentResult:
    return ExperimentResult(
        result_id=(f"result-{condition_id}-{task_id}-{seed}"),
        experiment_id="phase11-test",
        condition_id=condition_id,
        task_id=task_id,
        domain=domain,
        family_id=family_id,
        split=split,
        variant_index=variant_index,
        seed=seed,
        execution_mode=(ExperimentExecutionMode.deterministic_smoke),
        executor_name="test-executor",
        executor_version="test-executor-v1",
        model_name=None,
        model_version=None,
        prompt_version="test-prompt-v1",
        environment_fingerprint="a" * 64,
        status=status,
        task_success=success,
        regression_eligible=True,
        regression=regression,
        incorrect_evolution=incorrect_evolution,
        cluster_true_positive=cluster_tp,
        cluster_false_positive=cluster_fp,
        tool_calls=4,
        input_tokens=100,
        output_tokens=20,
        latency_seconds=1.5,
        proposal_created=True,
        proposal_accepted=success,
        security_rejected=(status is ExperimentResultStatus.rejected),
        evidence_count=3,
        evolution_delay_seconds=10.0,
        error_code=("security_rejected" if status is ExperimentResultStatus.rejected else None),
    )


def test_default_manifest_freezes_sixty_deterministic_tasks() -> None:
    manifest = build_default_manifest()

    assert manifest.schema_version == ("deerflow.skill-evolution.experiment-manifest.v1")
    assert len(manifest.tasks) == 60
    assert len({task.task_id for task in manifest.tasks}) == 60
    assert {task.domain for task in manifest.tasks} == {
        "repository_repair",
        "structured_data",
        "shell_workflow",
    }
    by_family: dict[str, list] = {}
    for task in manifest.tasks:
        by_family.setdefault(task.family_id, []).append(task)
        assert task.verifier.commands or task.verifier.artifacts
    assert len(by_family) == 12
    assert all(len(tasks) == 5 for tasks in by_family.values())
    assert all(sum(task.split == "evidence" for task in tasks) == 3 and sum(task.split == "held_out" for task in tasks) == 2 for tasks in by_family.values())


def test_default_conditions_cover_every_frozen_ablation() -> None:
    conditions = build_default_conditions()
    identifiers = {condition.condition_id for condition in conditions}

    assert {
        "baseline-no-evolution",
        "baseline-immediate-single",
        "baseline-staged-single",
        "proposed-k3-hybrid",
        "ablation-k1",
        "ablation-k5",
        "ablation-grouping-deterministic",
        "ablation-grouping-llm",
        "ablation-publication-immediate",
        "ablation-success-only",
        "ablation-create-branch",
        "ablation-patch-branch",
    } <= identifiers
    assert all(condition.evidence_threshold in {1, 3, 5} for condition in conditions)


def test_result_rejects_incoherent_failure_and_metric_state() -> None:
    with pytest.raises(ValidationError):
        _result(
            condition_id="candidate",
            task_id="task-1",
            success=True,
            status=ExperimentResultStatus.failed,
        )


def test_summary_reports_primary_secondary_and_rejected_counts() -> None:
    rows = [
        _result(
            condition_id="candidate",
            task_id="task-1",
            success=True,
            cluster_tp=3,
        ),
        _result(
            condition_id="candidate",
            task_id="task-2",
            success=False,
            regression=True,
            incorrect_evolution=True,
            status=ExperimentResultStatus.rejected,
            cluster_tp=1,
            cluster_fp=1,
        ),
    ]

    summary = summarize_condition(rows)

    assert summary.task_success_rate == 0.5
    assert summary.held_out_success_rate == 0.5
    assert summary.regression_rate == 0.5
    assert summary.incorrect_evolution_rate == 0.5
    assert summary.cluster_purity == 0.8
    assert summary.mean_tool_calls == 4.0
    assert summary.proposal_acceptance_rate == 0.5
    assert summary.failed_count == 0
    assert summary.rejected_count == 1
    assert summary.security_rejection_rate == 0.5


def test_paired_comparison_bootstrap_and_mcnemar_are_deterministic() -> None:
    baseline = []
    candidate = []
    for index in range(20):
        task_id = f"task-{index}"
        baseline.append(
            _result(
                condition_id="baseline",
                task_id=task_id,
                success=index < 8,
            )
        )
        candidate.append(
            _result(
                condition_id="candidate",
                task_id=task_id,
                success=index < 14,
            )
        )

    first = compare_conditions(
        baseline,
        candidate,
        bootstrap_samples=2_000,
        random_seed=7,
    )
    second = compare_conditions(
        baseline,
        candidate,
        bootstrap_samples=2_000,
        random_seed=7,
    )

    assert first == second
    assert first.paired_sample_count == 20
    assert first.success_rate_lift == pytest.approx(0.3)
    assert first.confidence_interval.low <= 0.3
    assert first.confidence_interval.high >= 0.3
    assert first.mcnemar.discordant_baseline_only == 0
    assert first.mcnemar.discordant_candidate_only == 6
    assert first.mcnemar.p_value is not None


def test_comparison_rejects_incomplete_pairing() -> None:
    baseline = [
        _result(
            condition_id="baseline",
            task_id="task-1",
            success=True,
        )
    ]
    candidate = [
        _result(
            condition_id="candidate",
            task_id="task-2",
            success=True,
        )
    ]

    with pytest.raises(ValueError, match="identical"):
        compare_conditions(baseline, candidate)


def test_report_separates_aggregate_and_task_families() -> None:
    rows = [
        _result(
            condition_id=condition,
            task_id=f"{family}-task",
            family_id=family,
            seed=seed,
            success=(condition == "candidate" or seed == 1),
        )
        for family in ("repo-preflight", "repo-async-retry")
        for condition in ("baseline", "candidate")
        for seed in (1, 2)
    ]

    report = build_experiment_report(
        rows,
        reference_condition_id="baseline",
        bootstrap_samples=500,
        random_seed=11,
    )

    assert set(report.aggregate) == {"baseline", "candidate"}
    assert set(report.by_family) == {
        "repo-preflight",
        "repo-async-retry",
    }
    assert all(set(family_report) == {"baseline", "candidate"} for family_report in report.by_family.values())
    assert len(report.comparisons) == 1
    assert report.comparisons[0].candidate_condition_id == "candidate"


class _DeterministicExecutor:
    async def execute(self, task, condition, seed):
        success = (seed + task.variant_index + condition.evidence_threshold) % 2 == 0
        return _result(
            condition_id=condition.condition_id,
            task_id=task.task_id,
            family_id=task.family_id,
            domain=task.domain,
            variant_index=task.variant_index,
            seed=seed,
            success=success,
            split=task.split,
        )


class _CohortExecutor:
    def __init__(self) -> None:
        self.orders: list[list[str]] = []

    async def execute_cohort(self, tasks, condition, seed):
        self.orders.append([task.task_id for task in tasks])
        return [
            _result(
                condition_id=condition.condition_id,
                task_id=task.task_id,
                family_id=task.family_id,
                domain=task.domain,
                variant_index=task.variant_index,
                seed=seed,
                success=True,
                split=task.split,
            )
            for task in tasks
        ]


@pytest.mark.asyncio
async def test_runner_executes_stateful_families_as_ordered_cohorts() -> None:
    manifest = build_default_manifest()
    executor = _CohortExecutor()
    condition = ExperimentCondition(
        condition_id="candidate",
    )

    rows = await run_experiment(
        experiment_id="phase11-test",
        manifest=manifest,
        conditions=[condition],
        seeds=[1],
        executor=executor,
        max_concurrency=4,
    )

    assert len(rows) == 60
    assert len(executor.orders) == 12
    assert all(len(order) == 5 for order in executor.orders)
    assert all(
        order[:3]
        == sorted(
            order[:3],
            key=lambda task_id: int(task_id.rsplit("-", 1)[1]),
        )
        for order in executor.orders
    )
    assert all("evidence" in task_id for order in executor.orders for task_id in order[:3])
    assert all("held_out" in task_id for order in executor.orders for task_id in order[3:])


@pytest.mark.asyncio
async def test_runner_resumes_only_complete_cohorts_and_checkpoints_new_rows() -> None:
    manifest = build_default_manifest()
    condition = ExperimentCondition(
        condition_id="candidate",
    )
    first_family = sorted({task.family_id for task in manifest.tasks})[0]
    first_tasks = [task for task in manifest.tasks if task.family_id == first_family]
    existing = [
        _result(
            condition_id=condition.condition_id,
            task_id=task.task_id,
            family_id=task.family_id,
            domain=task.domain,
            variant_index=task.variant_index,
            seed=1,
            success=True,
            split=task.split,
        )
        for task in first_tasks
    ]
    executor = _CohortExecutor()
    checkpoints: list[list[str]] = []

    async def checkpoint(rows) -> None:
        checkpoints.append(sorted(row.task_id for row in rows))

    rows = await run_experiment(
        experiment_id="phase11-test",
        manifest=manifest,
        conditions=[condition],
        seeds=[1],
        executor=executor,
        max_concurrency=4,
        existing_results=existing,
        on_results_completed=checkpoint,
    )

    assert len(rows) == 60
    assert len(executor.orders) == 11
    assert all(not any(task_id.startswith(first_family) for task_id in order) for order in executor.orders)
    assert len(checkpoints) == 11
    assert all(len(batch) == 5 for batch in checkpoints)


@pytest.mark.asyncio
async def test_runner_rejects_partial_resume_cohort() -> None:
    manifest = build_default_manifest()
    condition = ExperimentCondition(
        condition_id="candidate",
    )
    task = manifest.tasks[0]
    existing = [
        _result(
            condition_id=condition.condition_id,
            task_id=task.task_id,
            family_id=task.family_id,
            domain=task.domain,
            variant_index=task.variant_index,
            seed=1,
            success=True,
            split=task.split,
        )
    ]

    with pytest.raises(
        ValueError,
        match="partial cohort",
    ):
        await run_experiment(
            experiment_id="phase11-test",
            manifest=manifest,
            conditions=[condition],
            seeds=[1],
            executor=_CohortExecutor(),
            existing_results=existing,
        )


@pytest.mark.asyncio
async def test_runner_pairs_every_task_condition_and_seed(
    tmp_path,
) -> None:
    manifest = build_default_manifest()
    conditions = [
        ExperimentCondition(
            condition_id="baseline",
            baseline_kind="no_evolution",
        ),
        ExperimentCondition(
            condition_id="candidate",
            evidence_threshold=3,
            grouping_strategy="hybrid",
            publication_strategy="staged",
            evidence_mode="success_plus_recovered",
            branch_mode="both",
        ),
    ]

    rows = await run_experiment(
        experiment_id="phase11-test",
        manifest=manifest,
        conditions=conditions,
        seeds=[1, 2, 3],
        executor=_DeterministicExecutor(),
        max_concurrency=8,
    )

    assert len(rows) == 60 * 2 * 3
    assert len({row.result_id for row in rows}) == len(rows)
    assert all(row.schema_version == EXPERIMENT_RESULT_SCHEMA_VERSION for row in rows)
    output = tmp_path / "results.jsonl"
    write_results_jsonl(output, rows)
    loaded = read_results_jsonl(output)
    assert loaded == rows
    assert all(json.loads(line)["schema_version"] == EXPERIMENT_RESULT_SCHEMA_VERSION for line in output.read_text().splitlines())
