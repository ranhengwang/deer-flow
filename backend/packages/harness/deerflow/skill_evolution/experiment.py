"""Reproducible Skill-evolution experiments and paired statistics."""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from deerflow.skill_evolution.models import (
    EvolutionModel,
    Identifier,
    Sha256,
)

EXPERIMENT_MANIFEST_SCHEMA_VERSION = "deerflow.skill-evolution.experiment-manifest.v1"
EXPERIMENT_RESULT_SCHEMA_VERSION = "deerflow.skill-evolution.experiment-result.v1"
EXPERIMENT_REPORT_SCHEMA_VERSION = "deerflow.skill-evolution.experiment-report.v1"

ExperimentDomain = Literal[
    "repository_repair",
    "structured_data",
    "shell_workflow",
]
ExperimentSplit = Literal["evidence", "held_out"]


class ExperimentResultStatus(StrEnum):
    completed = "completed"
    failed = "failed"
    rejected = "rejected"
    skipped = "skipped"


class ExperimentExecutionMode(StrEnum):
    production_replay = "production_replay"
    deterministic_smoke = "deterministic_smoke"


class BenchmarkVerifier(EvolutionModel):
    kind: Literal[
        "repository",
        "structured_data",
        "shell",
    ]
    commands: list[str] = Field(
        default_factory=list,
        max_length=16,
    )
    artifacts: list[str] = Field(
        default_factory=list,
        max_length=16,
    )
    invariants: list[str] = Field(
        default_factory=list,
        max_length=32,
    )

    @model_validator(mode="after")
    def _validate_verifier(self) -> Self:
        if not self.commands and not self.artifacts:
            raise ValueError("benchmark verifier needs a command or artifact")
        return self


class BenchmarkTask(EvolutionModel):
    task_id: Identifier
    domain: ExperimentDomain
    family_id: Identifier
    split: ExperimentSplit
    variant_index: int = Field(ge=1, le=5)
    branch: Literal["create", "patch"]
    fixture_id: Identifier
    objective: str = Field(min_length=1, max_length=1_000)
    verifier: BenchmarkVerifier


class ExperimentManifest(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.experiment-manifest.v1"] = EXPERIMENT_MANIFEST_SCHEMA_VERSION
    manifest_id: Identifier
    tasks: list[BenchmarkTask] = Field(
        min_length=1,
        max_length=1_000,
    )

    @model_validator(mode="after")
    def _validate_tasks(self) -> Self:
        identifiers = [task.task_id for task in self.tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("benchmark task IDs must be unique")
        return self


class ExperimentCondition(EvolutionModel):
    condition_id: Identifier
    baseline_kind: Literal[
        "custom",
        "no_evolution",
        "immediate_single",
        "staged_single",
        "proposed",
    ] = "custom"
    evidence_threshold: Literal[1, 3, 5] = 3
    grouping_strategy: Literal[
        "deterministic",
        "llm",
        "hybrid",
    ] = "hybrid"
    publication_strategy: Literal[
        "immediate",
        "staged",
    ] = "staged"
    evidence_mode: Literal[
        "success_only",
        "success_plus_recovered",
    ] = "success_plus_recovered"
    branch_mode: Literal[
        "create",
        "patch",
        "both",
    ] = "both"


class ExperimentResult(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.experiment-result.v1"] = EXPERIMENT_RESULT_SCHEMA_VERSION
    result_id: Identifier
    experiment_id: Identifier
    condition_id: Identifier
    task_id: Identifier
    domain: ExperimentDomain
    family_id: Identifier
    split: ExperimentSplit
    variant_index: int = Field(ge=1, le=5)
    seed: int = Field(ge=0)
    execution_mode: ExperimentExecutionMode
    executor_name: Identifier
    executor_version: Identifier
    model_name: Identifier | None = None
    model_version: Identifier | None = None
    prompt_version: Identifier
    environment_fingerprint: Sha256
    status: ExperimentResultStatus
    task_success: bool
    regression_eligible: bool
    regression: bool
    incorrect_evolution: bool
    cluster_true_positive: int = Field(ge=0)
    cluster_false_positive: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0.0)
    proposal_created: bool
    proposal_accepted: bool
    security_rejected: bool
    evidence_count: int = Field(ge=0)
    evolution_delay_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )
    error_code: Identifier | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if self.execution_mode is ExperimentExecutionMode.production_replay and (self.model_name is None or self.model_version is None):
            raise ValueError("production replay requires immutable model identity")
        if self.status is not ExperimentResultStatus.completed and self.task_success:
            raise ValueError("non-completed result cannot be successful")
        if self.proposal_accepted and not self.proposal_created:
            raise ValueError("accepted proposal must have been created")
        if self.regression and not self.regression_eligible:
            raise ValueError("regression requires an eligible baseline")
        if (
            self.status
            in {
                ExperimentResultStatus.failed,
                ExperimentResultStatus.rejected,
                ExperimentResultStatus.skipped,
            }
            and self.error_code is None
        ):
            raise ValueError("non-completed result requires error_code")
        return self

    @property
    def pair_key(self) -> tuple[str, int]:
        return self.task_id, self.seed


class ConditionSummary(EvolutionModel):
    condition_id: Identifier
    sample_count: int = Field(ge=0)
    task_success_rate: float | None
    held_out_success_rate: float | None
    regression_rate: float | None
    incorrect_evolution_rate: float | None
    cluster_purity: float | None
    mean_tool_calls: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    mean_latency_seconds: float | None
    proposal_acceptance_rate: float | None
    mean_evidence_count: float | None
    mean_evolution_delay_seconds: float | None
    security_rejection_rate: float | None
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)


class ConfidenceInterval(EvolutionModel):
    confidence: float = Field(gt=0.0, lt=1.0)
    low: float
    high: float


class MetricDelta(EvolutionModel):
    metric: Identifier
    estimate: float | None
    confidence_interval: ConfidenceInterval | None
    paired_sample_count: int = Field(ge=0)


class McNemarResult(EvolutionModel):
    discordant_baseline_only: int = Field(ge=0)
    discordant_candidate_only: int = Field(ge=0)
    p_value: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class ConditionComparison(EvolutionModel):
    baseline_condition_id: Identifier
    candidate_condition_id: Identifier
    paired_sample_count: int = Field(ge=0)
    success_rate_lift: float | None
    confidence_interval: ConfidenceInterval | None
    primary_deltas: dict[Identifier, MetricDelta]
    mcnemar: McNemarResult


class ExperimentReport(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.experiment-report.v1"] = EXPERIMENT_REPORT_SCHEMA_VERSION
    experiment_id: Identifier
    execution_mode: ExperimentExecutionMode
    aggregate: dict[Identifier, ConditionSummary]
    by_family: dict[
        Identifier,
        dict[Identifier, ConditionSummary],
    ]
    comparisons: list[ConditionComparison]


class ExperimentExecutor(Protocol):
    async def execute(
        self,
        task: BenchmarkTask,
        condition: ExperimentCondition,
        seed: int,
    ) -> ExperimentResult: ...


class CohortExperimentExecutor(Protocol):
    async def execute_cohort(
        self,
        tasks: Sequence[BenchmarkTask],
        condition: ExperimentCondition,
        seed: int,
    ) -> Sequence[ExperimentResult]: ...


_FAMILIES: tuple[
    tuple[
        ExperimentDomain,
        str,
        Literal["create", "patch"],
        str,
        BenchmarkVerifier,
    ],
    ...,
] = (
    (
        "repository_repair",
        "repo-preflight",
        "patch",
        "Repair a repository preflight check and preserve focused tests.",
        BenchmarkVerifier(
            kind="repository",
            commands=["pytest -q tests/test_preflight.py"],
            artifacts=["workspace/preflight.patch"],
            invariants=["focused_tests_pass", "target_file_changed"],
        ),
    ),
    (
        "repository_repair",
        "repo-path-normalization",
        "create",
        "Implement a reusable path-normalization workflow.",
        BenchmarkVerifier(
            kind="repository",
            commands=["pytest -q tests/test_paths.py"],
            artifacts=["workspace/path-normalization.patch"],
            invariants=["traversal_rejected", "valid_paths_preserved"],
        ),
    ),
    (
        "repository_repair",
        "repo-async-retry",
        "patch",
        "Repair bounded async retry behavior without duplicate effects.",
        BenchmarkVerifier(
            kind="repository",
            commands=["pytest -q tests/test_retry.py"],
            artifacts=["workspace/retry.patch"],
            invariants=["retry_bounded", "effects_idempotent"],
        ),
    ),
    (
        "repository_repair",
        "repo-config-migration",
        "create",
        "Create a deterministic configuration migration workflow.",
        BenchmarkVerifier(
            kind="repository",
            commands=["pytest -q tests/test_config_upgrade.py"],
            artifacts=["workspace/config-migration.patch"],
            invariants=["old_config_upgrades", "upgrade_idempotent"],
        ),
    ),
    (
        "structured_data",
        "data-csv-normalization",
        "create",
        "Normalize a CSV dataset while preserving row invariants.",
        BenchmarkVerifier(
            kind="structured_data",
            commands=["python verify.py --family csv-normalization"],
            artifacts=["outputs/normalized.csv"],
            invariants=["schema_valid", "row_count_preserved"],
        ),
    ),
    (
        "structured_data",
        "data-json-aggregation",
        "patch",
        "Repair grouped JSON aggregation and deterministic ordering.",
        BenchmarkVerifier(
            kind="structured_data",
            commands=["python verify.py --family json-aggregation"],
            artifacts=["outputs/aggregate.json"],
            invariants=["totals_match", "ordering_deterministic"],
        ),
    ),
    (
        "structured_data",
        "data-schema-validation",
        "create",
        "Build a reusable schema-validation and rejection workflow.",
        BenchmarkVerifier(
            kind="structured_data",
            commands=["python verify.py --family schema-validation"],
            artifacts=["outputs/validated.jsonl"],
            invariants=["invalid_rows_rejected", "valid_rows_retained"],
        ),
    ),
    (
        "structured_data",
        "data-deduplication",
        "patch",
        "Repair stable record deduplication with explicit precedence.",
        BenchmarkVerifier(
            kind="structured_data",
            commands=["python verify.py --family deduplication"],
            artifacts=["outputs/deduplicated.json"],
            invariants=["keys_unique", "precedence_correct"],
        ),
    ),
    (
        "shell_workflow",
        "shell-portable-bootstrap",
        "create",
        "Create a portable environment bootstrap workflow.",
        BenchmarkVerifier(
            kind="shell",
            commands=["sh verify.sh portable-bootstrap"],
            artifacts=["outputs/bootstrap.ok"],
            invariants=["exit_zero", "postcondition_passes"],
        ),
    ),
    (
        "shell_workflow",
        "shell-archive-verification",
        "patch",
        "Repair archive extraction and integrity verification.",
        BenchmarkVerifier(
            kind="shell",
            commands=["sh verify.sh archive-verification"],
            artifacts=["outputs/archive-manifest.json"],
            invariants=["hashes_match", "traversal_blocked"],
        ),
    ),
    (
        "shell_workflow",
        "shell-artifact-pipeline",
        "create",
        "Create a deterministic multi-step artifact pipeline.",
        BenchmarkVerifier(
            kind="shell",
            commands=["sh verify.sh artifact-pipeline"],
            artifacts=["outputs/final-artifact.txt"],
            invariants=["all_stages_pass", "artifact_exact"],
        ),
    ),
    (
        "shell_workflow",
        "shell-command-fallback",
        "patch",
        "Repair OS-specific command fallback and verification.",
        BenchmarkVerifier(
            kind="shell",
            commands=["sh verify.sh command-fallback"],
            artifacts=["outputs/fallback.ok"],
            invariants=["fallback_used_when_needed", "exit_zero"],
        ),
    ),
)


def build_default_manifest() -> ExperimentManifest:
    tasks: list[BenchmarkTask] = []
    for domain, family, branch, objective, verifier in _FAMILIES:
        for variant in range(1, 6):
            split: ExperimentSplit = "evidence" if variant <= 3 else "held_out"
            tasks.append(
                BenchmarkTask(
                    task_id=f"{family}-{split}-{variant}",
                    domain=domain,
                    family_id=family,
                    split=split,
                    variant_index=variant,
                    branch=branch,
                    fixture_id=f"{family}-fixture-{variant}",
                    objective=f"{objective} Variant {variant}.",
                    verifier=verifier,
                )
            )
    return ExperimentManifest(
        manifest_id="skill-evolution-phase11-v1",
        tasks=tasks,
    )


def build_default_conditions() -> list[ExperimentCondition]:
    return [
        ExperimentCondition(
            condition_id="baseline-no-evolution",
            baseline_kind="no_evolution",
        ),
        ExperimentCondition(
            condition_id="baseline-immediate-single",
            baseline_kind="immediate_single",
            evidence_threshold=1,
            publication_strategy="immediate",
        ),
        ExperimentCondition(
            condition_id="baseline-staged-single",
            baseline_kind="staged_single",
            evidence_threshold=1,
        ),
        ExperimentCondition(
            condition_id="proposed-k3-hybrid",
            baseline_kind="proposed",
        ),
        ExperimentCondition(
            condition_id="ablation-k1",
            evidence_threshold=1,
        ),
        ExperimentCondition(
            condition_id="ablation-k5",
            evidence_threshold=5,
        ),
        ExperimentCondition(
            condition_id="ablation-grouping-deterministic",
            grouping_strategy="deterministic",
        ),
        ExperimentCondition(
            condition_id="ablation-grouping-llm",
            grouping_strategy="llm",
        ),
        ExperimentCondition(
            condition_id="ablation-publication-immediate",
            publication_strategy="immediate",
        ),
        ExperimentCondition(
            condition_id="ablation-success-only",
            evidence_mode="success_only",
        ),
        ExperimentCondition(
            condition_id="ablation-create-branch",
            branch_mode="create",
        ),
        ExperimentCondition(
            condition_id="ablation-patch-branch",
            branch_mode="patch",
        ),
    ]


def _rate(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_condition(
    rows: Sequence[ExperimentResult],
) -> ConditionSummary:
    if not rows:
        raise ValueError("condition summary requires results")
    condition_ids = {row.condition_id for row in rows}
    if len(condition_ids) != 1:
        raise ValueError("condition summary cannot mix condition IDs")
    held_out = [row for row in rows if row.split == "held_out"]
    regressions = [row for row in rows if row.regression_eligible]
    cluster_total = sum(row.cluster_true_positive + row.cluster_false_positive for row in rows)
    proposal_rows = [row for row in rows if row.proposal_created]
    delay_values = [row.evolution_delay_seconds for row in rows if row.evolution_delay_seconds is not None]
    status_counts = {status: sum(row.status is status for row in rows) for status in ExperimentResultStatus}
    return ConditionSummary(
        condition_id=next(iter(condition_ids)),
        sample_count=len(rows),
        task_success_rate=_rate([row.task_success for row in rows]),
        held_out_success_rate=_rate([row.task_success for row in held_out]),
        regression_rate=_rate([row.regression for row in regressions]),
        incorrect_evolution_rate=_rate([row.incorrect_evolution for row in rows]),
        cluster_purity=(sum(row.cluster_true_positive for row in rows) / cluster_total if cluster_total else None),
        mean_tool_calls=_mean([row.tool_calls for row in rows]),
        mean_input_tokens=_mean([row.input_tokens for row in rows]),
        mean_output_tokens=_mean([row.output_tokens for row in rows]),
        mean_latency_seconds=_mean([row.latency_seconds for row in rows]),
        proposal_acceptance_rate=_rate([row.proposal_accepted for row in proposal_rows]),
        mean_evidence_count=_mean([row.evidence_count for row in rows]),
        mean_evolution_delay_seconds=_mean(delay_values),
        security_rejection_rate=_rate([row.security_rejected for row in rows]),
        completed_count=status_counts[ExperimentResultStatus.completed],
        failed_count=status_counts[ExperimentResultStatus.failed],
        rejected_count=status_counts[ExperimentResultStatus.rejected],
        skipped_count=status_counts[ExperimentResultStatus.skipped],
    )


def _index_pairs(
    rows: Sequence[ExperimentResult],
) -> dict[tuple[str, int], ExperimentResult]:
    result: dict[tuple[str, int], ExperimentResult] = {}
    for row in rows:
        if row.pair_key in result:
            raise ValueError("condition results contain duplicate task/seed pair")
        result[row.pair_key] = row
    return result


def _percentile(
    values: list[float],
    probability: float,
) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_delta(
    pairs: list[tuple[ExperimentResult, ExperimentResult]],
    metric,
    *,
    samples: int,
    random_seed: int,
) -> tuple[float | None, ConfidenceInterval | None, int]:
    usable = [(baseline, candidate) for baseline, candidate in pairs if metric(baseline) is not None and metric(candidate) is not None]
    if not usable:
        return None, None, 0
    differences = [float(metric(candidate)) - float(metric(baseline)) for baseline, candidate in usable]
    estimate = sum(differences) / len(differences)
    if samples < 1:
        return estimate, None, len(usable)
    generator = random.Random(random_seed)
    bootstrapped = []
    for _ in range(samples):
        drawn = [differences[generator.randrange(len(differences))] for _ in range(len(differences))]
        bootstrapped.append(sum(drawn) / len(drawn))
    return (
        estimate,
        ConfidenceInterval(
            confidence=0.95,
            low=_percentile(bootstrapped, 0.025),
            high=_percentile(bootstrapped, 0.975),
        ),
        len(usable),
    )


def _cluster_row_purity(
    row: ExperimentResult,
) -> float | None:
    total = row.cluster_true_positive + row.cluster_false_positive
    return row.cluster_true_positive / total if total else None


def _mcnemar(
    pairs: Sequence[tuple[ExperimentResult, ExperimentResult]],
) -> McNemarResult:
    baseline_only = sum(baseline.task_success and not candidate.task_success for baseline, candidate in pairs)
    candidate_only = sum(candidate.task_success and not baseline.task_success for baseline, candidate in pairs)
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = None
    else:
        smaller = min(baseline_only, candidate_only)
        tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return McNemarResult(
        discordant_baseline_only=baseline_only,
        discordant_candidate_only=candidate_only,
        p_value=p_value,
    )


def compare_conditions(
    baseline: Sequence[ExperimentResult],
    candidate: Sequence[ExperimentResult],
    *,
    bootstrap_samples: int = 10_000,
    random_seed: int = 0,
) -> ConditionComparison:
    if not baseline or not candidate:
        raise ValueError("comparison requires both conditions")
    baseline_ids = {row.condition_id for row in baseline}
    candidate_ids = {row.condition_id for row in candidate}
    if len(baseline_ids) != 1 or len(candidate_ids) != 1:
        raise ValueError("comparison inputs must each contain one condition")
    baseline_index = _index_pairs(baseline)
    candidate_index = _index_pairs(candidate)
    if set(baseline_index) != set(candidate_index):
        raise ValueError("paired conditions must contain identical task/seed keys")
    keys = sorted(baseline_index)
    pairs = [(baseline_index[key], candidate_index[key]) for key in keys]
    metrics = {
        "task_success_rate": lambda row: row.task_success,
        "held_out_success_rate": lambda row: row.task_success if row.split == "held_out" else None,
        "regression_rate": lambda row: row.regression if row.regression_eligible else None,
        "incorrect_evolution_rate": (lambda row: row.incorrect_evolution),
        "cluster_purity": _cluster_row_purity,
    }
    deltas: dict[str, MetricDelta] = {}
    for offset, (name, metric) in enumerate(metrics.items()):
        estimate, interval, count = _bootstrap_delta(
            pairs,
            metric,
            samples=bootstrap_samples,
            random_seed=random_seed + offset,
        )
        deltas[name] = MetricDelta(
            metric=name,
            estimate=estimate,
            confidence_interval=interval,
            paired_sample_count=count,
        )
    success = deltas["task_success_rate"]
    return ConditionComparison(
        baseline_condition_id=next(iter(baseline_ids)),
        candidate_condition_id=next(iter(candidate_ids)),
        paired_sample_count=len(pairs),
        success_rate_lift=success.estimate,
        confidence_interval=success.confidence_interval,
        primary_deltas=deltas,
        mcnemar=_mcnemar(pairs),
    )


def build_experiment_report(
    rows: Sequence[ExperimentResult],
    *,
    reference_condition_id: str,
    bootstrap_samples: int = 10_000,
    random_seed: int = 0,
) -> ExperimentReport:
    if not rows:
        raise ValueError("experiment report requires results")
    experiment_ids = {row.experiment_id for row in rows}
    if len(experiment_ids) != 1:
        raise ValueError("report cannot mix experiment IDs")
    execution_modes = {row.execution_mode for row in rows}
    if len(execution_modes) != 1:
        raise ValueError("report cannot mix execution modes")
    by_condition: dict[str, list[ExperimentResult]] = defaultdict(list)
    by_family_rows: dict[
        str,
        dict[str, list[ExperimentResult]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_condition[row.condition_id].append(row)
        by_family_rows[row.family_id][row.condition_id].append(row)
    if reference_condition_id not in by_condition:
        raise ValueError("reference condition is absent")
    aggregate = {condition_id: summarize_condition(condition_rows) for condition_id, condition_rows in sorted(by_condition.items())}
    by_family = {family_id: {condition_id: summarize_condition(condition_rows) for condition_id, condition_rows in sorted(conditions.items())} for family_id, conditions in sorted(by_family_rows.items())}
    reference = by_condition[reference_condition_id]
    comparisons = [
        compare_conditions(
            reference,
            condition_rows,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + index,
        )
        for index, (condition_id, condition_rows) in enumerate(sorted(by_condition.items()))
        if condition_id != reference_condition_id
    ]
    return ExperimentReport(
        experiment_id=next(iter(experiment_ids)),
        execution_mode=next(iter(execution_modes)),
        aggregate=aggregate,
        by_family=by_family,
        comparisons=comparisons,
    )


async def run_experiment(
    *,
    experiment_id: str,
    manifest: ExperimentManifest,
    conditions: Sequence[ExperimentCondition],
    seeds: Sequence[int],
    executor: ExperimentExecutor | CohortExperimentExecutor,
    max_concurrency: int = 4,
    existing_results: Sequence[ExperimentResult] = (),
    on_results_completed: (
        Callable[
            [Sequence[ExperimentResult]],
            Awaitable[None],
        ]
        | None
    ) = None,
) -> list[ExperimentResult]:
    if not conditions:
        raise ValueError("experiment requires conditions")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("experiment seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("experiment seeds must be non-negative")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    condition_ids = [condition.condition_id for condition in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition IDs must be unique")
    tasks_by_id = {task.task_id: task for task in manifest.tasks}
    existing_by_key: dict[
        tuple[str, str, int],
        ExperimentResult,
    ] = {}
    for result in existing_results:
        task = tasks_by_id.get(result.task_id)
        if result.experiment_id != experiment_id or result.condition_id not in condition_ids or result.seed not in seeds or task is None:
            raise ValueError("existing experiment result is outside the requested matrix")
        expected = (
            task.domain,
            task.family_id,
            task.split,
            task.variant_index,
        )
        actual = (
            result.domain,
            result.family_id,
            result.split,
            result.variant_index,
        )
        if actual != expected:
            raise ValueError("existing experiment result identity does not match manifest")
        key = (
            result.condition_id,
            result.task_id,
            result.seed,
        )
        if key in existing_by_key:
            raise ValueError("existing experiment results contain duplicate requests")
        existing_by_key[key] = result
    bind_experiment = getattr(executor, "bind_experiment", None)
    if callable(bind_experiment):
        bind_experiment(experiment_id)

    semaphore = asyncio.Semaphore(max_concurrency)
    completion_lock = asyncio.Lock()

    async def notify_completed(
        rows: Sequence[ExperimentResult],
    ) -> None:
        if on_results_completed is None:
            return
        async with completion_lock:
            await on_results_completed(rows)

    async def execute_one(
        task: BenchmarkTask,
        condition: ExperimentCondition,
        seed: int,
    ) -> ExperimentResult:
        existing = existing_by_key.get(
            (
                condition.condition_id,
                task.task_id,
                seed,
            )
        )
        if existing is not None:
            return existing
        async with semaphore:
            result = await executor.execute(
                task,
                condition,
                seed,
            )
        expected = (
            experiment_id,
            condition.condition_id,
            task.task_id,
            task.domain,
            task.family_id,
            task.split,
            task.variant_index,
            seed,
        )
        actual = (
            result.experiment_id,
            result.condition_id,
            result.task_id,
            result.domain,
            result.family_id,
            result.split,
            result.variant_index,
            result.seed,
        )
        if actual != expected:
            raise ValueError("executor result identity does not match request")
        await notify_completed([result])
        return result

    execute_cohort = getattr(executor, "execute_cohort", None)
    if callable(execute_cohort):
        families: dict[str, list[BenchmarkTask]] = defaultdict(list)
        for task in manifest.tasks:
            families[task.family_id].append(task)

        cohort_requests: list[
            tuple[
                list[BenchmarkTask],
                ExperimentCondition,
                int,
            ]
        ] = []
        for condition in conditions:
            for _, tasks in sorted(families.items()):
                ordered = sorted(
                    tasks,
                    key=lambda task: (
                        0 if task.split == "evidence" else 1,
                        task.variant_index,
                        task.task_id,
                    ),
                )
                for seed in seeds:
                    resumed_count = sum(
                        (
                            condition.condition_id,
                            task.task_id,
                            seed,
                        )
                        in existing_by_key
                        for task in ordered
                    )
                    if resumed_count not in {
                        0,
                        len(ordered),
                    }:
                        raise ValueError("existing results contain a partial cohort")
                    cohort_requests.append(
                        (
                            ordered,
                            condition,
                            seed,
                        )
                    )

        async def execute_one_cohort(
            ordered: list[BenchmarkTask],
            condition: ExperimentCondition,
            seed: int,
        ) -> list[ExperimentResult]:
            resumed = [
                existing_by_key[
                    (
                        condition.condition_id,
                        task.task_id,
                        seed,
                    )
                ]
                for task in ordered
                if (
                    condition.condition_id,
                    task.task_id,
                    seed,
                )
                in existing_by_key
            ]
            if resumed:
                return resumed
            async with semaphore:
                result_rows = list(
                    await execute_cohort(
                        ordered,
                        condition,
                        seed,
                    )
                )
            if len(result_rows) != len(ordered):
                raise ValueError("cohort executor returned the wrong result count")
            by_task = {row.task_id: row for row in result_rows}
            if len(by_task) != len(result_rows):
                raise ValueError("cohort executor returned duplicate tasks")
            for task in ordered:
                result = by_task.get(task.task_id)
                if result is None:
                    raise ValueError("cohort executor omitted a requested task")
                expected = (
                    experiment_id,
                    condition.condition_id,
                    task.task_id,
                    task.domain,
                    task.family_id,
                    task.split,
                    task.variant_index,
                    seed,
                )
                actual = (
                    result.experiment_id,
                    result.condition_id,
                    result.task_id,
                    result.domain,
                    result.family_id,
                    result.split,
                    result.variant_index,
                    result.seed,
                )
                if actual != expected:
                    raise ValueError("cohort result identity does not match request")
            await notify_completed(result_rows)
            return result_rows

        cohort_rows = await asyncio.gather(
            *[
                execute_one_cohort(
                    ordered,
                    condition,
                    seed,
                )
                for ordered, condition, seed in cohort_requests
            ]
        )
        rows = [row for cohort in cohort_rows for row in cohort]
    else:
        coroutines = [execute_one(task, condition, seed) for condition in conditions for task in manifest.tasks for seed in seeds]
        rows = await asyncio.gather(*coroutines)
    identifiers = [row.result_id for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("executor produced duplicate result IDs")
    return list(rows)


def write_results_jsonl(
    path: Path,
    rows: Sequence[ExperimentResult],
) -> None:
    serialized = "".join(
        json.dumps(
            row.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(
        f".{path.name}.tmp",
    )
    staged.write_text(serialized, encoding="utf-8")
    staged.replace(path)


def read_results_jsonl(path: Path) -> list[ExperimentResult]:
    rows: list[ExperimentResult] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
            rows.append(ExperimentResult.model_validate(payload))
        except Exception as exc:
            raise ValueError(f"invalid experiment result at line {line_number}") from exc
    return rows
