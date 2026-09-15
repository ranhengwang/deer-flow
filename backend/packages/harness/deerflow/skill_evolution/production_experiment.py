"""Stateful production Replay executor for the Phase 11 benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.skill_evolution.docker_replay_runtime import (
    DEFAULT_REPLAY_IMAGE,
    DockerReplayRuntime,
)
from deerflow.skill_evolution.evaluator import (
    ReplaySkillPackage,
    TaskEvaluationResult,
    run_replay_task,
)
from deerflow.skill_evolution.experiment import (
    BenchmarkTask,
    ExperimentCondition,
    ExperimentExecutionMode,
    ExperimentResult,
    ExperimentResultStatus,
)
from deerflow.skill_evolution.production_evolution import (
    GeneratedCandidateBuilder,
    ProductionCandidateBuilder,
    ReplayEvidence,
)
from deerflow.skill_evolution.replay_suite import (
    MaterializedReplayCase,
    materialize_replay_case,
)

PRODUCTION_EXPERIMENT_EXECUTOR_VERSION = "production-replay-generated-v2"
PRODUCTION_EXPERIMENT_PROMPT_VERSION = "phase11-isolated-agent-v1"


def _result_id(
    experiment_id: str,
    condition_id: str,
    task_id: str,
    seed: int,
) -> str:
    digest = hashlib.sha256(
        "\0".join(
            (
                experiment_id,
                condition_id,
                task_id,
                str(seed),
            )
        ).encode("utf-8")
    ).hexdigest()
    return f"result-{digest[:32]}"


def _ollama_digest(
    app_config: AppConfig,
    model_name: str,
) -> str:
    model_config = app_config.get_model_config(model_name)
    if model_config is None:
        raise ValueError(f"unknown experiment model {model_name!r}")
    base_url = str(getattr(model_config, "base_url", None) or "http://127.0.0.1:11434").rstrip("/")
    provider_model = str(getattr(model_config, "model", None) or model_name)
    with urllib.request.urlopen(  # noqa: S310 - operator-configured local Ollama
        f"{base_url}/api/tags",
        timeout=10,
    ) as response:
        payload = json.load(response)
    for item in payload.get("models", []):
        names = {
            str(item.get("name") or ""),
            str(item.get("model") or ""),
        }
        if provider_model in names or f"{provider_model}:latest" in names:
            digest = item.get("digest")
            if isinstance(digest, str) and len(digest) == 64:
                return digest
    raise ValueError(f"Ollama model digest not found for {provider_model!r}")


def _threshold(condition: ExperimentCondition) -> int:
    if condition.baseline_kind == "no_evolution":
        return 10_000
    if condition.baseline_kind in {
        "immediate_single",
        "staged_single",
    }:
        return 1
    if condition.baseline_kind == "proposed":
        return 3
    return condition.evidence_threshold


def _publication_strategy(
    condition: ExperimentCondition,
) -> str:
    if condition.baseline_kind == "immediate_single":
        return "immediate"
    if condition.baseline_kind in {
        "staged_single",
        "proposed",
    }:
        return "staged"
    return condition.publication_strategy


def _branch_allowed(
    task: BenchmarkTask,
    condition: ExperimentCondition,
) -> bool:
    return condition.branch_mode == "both" or condition.branch_mode == task.branch


def _base_package(
    case: MaterializedReplayCase,
) -> ReplaySkillPackage | None:
    return case.base_skill


class ProductionExperimentExecutor:
    """Execute one stateful family cohort with real LLM and Docker replay."""

    def __init__(
        self,
        *,
        app_config: AppConfig,
        model_name: str,
        model_version: str,
        image: str = DEFAULT_REPLAY_IMAGE,
        docker_binary: str | None = None,
        runtime_factory: (Callable[[int], DockerReplayRuntime] | None) = None,
        parent_dir: str | Path | None = None,
        candidate_builder: ProductionCandidateBuilder | None = None,
    ) -> None:
        self._app_config = app_config
        self.model_name = model_name
        self.model_version = model_version
        self.image = image
        self._docker_binary = docker_binary
        self._runtime_factory = runtime_factory
        self._parent_dir = Path(parent_dir) if parent_dir is not None else None
        self._candidate_builder = (
            candidate_builder
            if candidate_builder is not None
            else GeneratedCandidateBuilder(
                app_config=app_config,
                model_name=model_name,
            )
        )
        self._experiment_id: str | None = None

    def bind_experiment(self, experiment_id: str) -> None:
        self._experiment_id = experiment_id

    def _runtime(self, seed: int) -> DockerReplayRuntime:
        if self._runtime_factory is not None:
            return self._runtime_factory(seed)
        return DockerReplayRuntime(
            app_config=self._app_config,
            model_name=self.model_name,
            image=self.image,
            docker_binary=self._docker_binary,
            thinking_enabled=False,
            recursion_limit=30,
            model_overrides={
                "seed": seed,
                "temperature": 0.2,
            },
            allowed_agent_tools=(
                "read_file",
                "write_json",
            ),
        )

    async def _run_case(
        self,
        runtime: DockerReplayRuntime,
        case: MaterializedReplayCase,
        *,
        replay_condition: str,
        skill_package: ReplaySkillPackage | None,
    ) -> TaskEvaluationResult:
        try:
            return await run_replay_task(
                runtime,
                case.spec,
                split=("source" if case.benchmark_task.split == "evidence" else "held_out"),
                condition=replay_condition,
                skill_package=skill_package,
                parent_dir=self._parent_dir,
            )
        finally:
            await runtime.close_all()

    @staticmethod
    def _failed_runtime(
        result: TaskEvaluationResult,
    ) -> bool:
        return any(
            error.startswith(
                (
                    "agent_runtime_error:",
                    "command_runtime_error:",
                )
            )
            or error
            in {
                "agent_timeout",
                "command_verifier_timeout",
            }
            for error in result.errors
        )

    def _experiment_result(
        self,
        *,
        runtime: DockerReplayRuntime,
        case: MaterializedReplayCase,
        condition: ExperimentCondition,
        seed: int,
        result: TaskEvaluationResult,
        proposal_created: bool,
        proposal_accepted: bool,
        candidate_active: bool,
        evidence_count: int,
        cluster_true_positive: int,
        cluster_false_positive: int,
        evolution_delay_seconds: float | None,
        regression_eligible: bool,
        regression: bool,
    ) -> ExperimentResult:
        if self._experiment_id is None:
            raise RuntimeError("production executor is not bound to an experiment")
        security_rejected = "side_effect_policy_violation" in result.errors
        runtime_failed = self._failed_runtime(result)
        status = ExperimentResultStatus.rejected if security_rejected else (ExperimentResultStatus.failed if runtime_failed else ExperimentResultStatus.completed)
        error_code = "security_rejected" if security_rejected else (result.errors[0] if runtime_failed and result.errors else None)
        return ExperimentResult(
            result_id=_result_id(
                self._experiment_id,
                condition.condition_id,
                case.benchmark_task.task_id,
                seed,
            ),
            experiment_id=self._experiment_id,
            condition_id=condition.condition_id,
            task_id=case.benchmark_task.task_id,
            domain=case.benchmark_task.domain,
            family_id=case.benchmark_task.family_id,
            split=case.benchmark_task.split,
            variant_index=case.benchmark_task.variant_index,
            seed=seed,
            execution_mode=ExperimentExecutionMode.production_replay,
            executor_name=self._candidate_builder.executor_name,
            executor_version=self._candidate_builder.executor_version,
            model_name=self.model_name,
            model_version=self.model_version,
            prompt_version=self._candidate_builder.prompt_version,
            environment_fingerprint=runtime.environment_fingerprint(),
            status=status,
            task_success=result.success and status is ExperimentResultStatus.completed,
            regression_eligible=regression_eligible,
            regression=regression,
            incorrect_evolution=(candidate_active and case.benchmark_task.split == "held_out" and not result.success),
            cluster_true_positive=cluster_true_positive,
            cluster_false_positive=cluster_false_positive,
            tool_calls=result.metrics.tool_calls,
            input_tokens=result.metrics.input_tokens,
            output_tokens=result.metrics.output_tokens,
            latency_seconds=result.metrics.latency_seconds,
            proposal_created=proposal_created,
            proposal_accepted=proposal_accepted,
            security_rejected=security_rejected,
            evidence_count=evidence_count,
            evolution_delay_seconds=evolution_delay_seconds,
            error_code=error_code,
        )

    async def execute_cohort(
        self,
        tasks: Sequence[BenchmarkTask],
        condition: ExperimentCondition,
        seed: int,
    ) -> Sequence[ExperimentResult]:
        if self._experiment_id is None:
            raise RuntimeError("production executor is not bound to an experiment")
        if len({task.family_id for task in tasks}) != 1:
            raise ValueError("production cohort must contain one task family")
        cases = [materialize_replay_case(task) for task in tasks]
        runtime = self._runtime(seed)
        await asyncio.to_thread(runtime.preflight)
        started = time.monotonic()
        qualifying: list[ReplayEvidence] = []
        candidate_package: ReplaySkillPackage | None = None
        candidate_ready = False
        proposal_created = False
        threshold = _threshold(condition)
        strategy = _publication_strategy(condition)
        results: list[ExperimentResult] = []
        try:
            for case in cases:
                branch_allowed = _branch_allowed(
                    case.benchmark_task,
                    condition,
                )
                use_candidate = candidate_ready and branch_allowed and candidate_package is not None
                selected_package = candidate_package if use_candidate else _base_package(case)
                replay_condition = "candidate_skill" if use_candidate else ("base_skill" if selected_package is not None else "no_skill")
                baseline_result: TaskEvaluationResult | None = None
                if use_candidate and case.base_skill is not None:
                    baseline_result = await self._run_case(
                        runtime,
                        case,
                        replay_condition="base_skill",
                        skill_package=case.base_skill,
                    )
                task_result = await self._run_case(
                    runtime,
                    case,
                    replay_condition=replay_condition,
                    skill_package=selected_package,
                )
                created_now = False
                cluster_true_positive = 0
                cluster_false_positive = 0
                if case.benchmark_task.split == "evidence":
                    recovered_allowed = condition.evidence_mode == "success_plus_recovered" or not case.recovered_error_evidence
                    if task_result.success and recovered_allowed:
                        qualifying.append(
                            (
                                case,
                                task_result,
                            )
                        )
                    if branch_allowed and not proposal_created and len(qualifying) >= threshold:
                        proposal_created = True
                        created_now = True
                        built = await self._candidate_builder.build(
                            qualifying,
                            condition,
                            seed,
                        )
                        candidate_package = built.package
                        cluster_true_positive = built.cluster_true_positive
                        cluster_false_positive = built.cluster_false_positive
                        if strategy == "immediate" and candidate_package is not None:
                            candidate_ready = True
                        elif candidate_package is not None:
                            validations = [
                                await self._run_case(
                                    runtime,
                                    evidence_case,
                                    replay_condition="candidate_skill",
                                    skill_package=candidate_package,
                                )
                                for evidence_case, _ in qualifying
                            ]
                            candidate_ready = all(item.success for item in validations)
                regression_eligible = baseline_result is not None and baseline_result.success
                results.append(
                    self._experiment_result(
                        runtime=runtime,
                        case=case,
                        condition=condition,
                        seed=seed,
                        result=task_result,
                        proposal_created=created_now,
                        proposal_accepted=(created_now and candidate_ready),
                        candidate_active=use_candidate,
                        evidence_count=len(qualifying),
                        cluster_true_positive=(cluster_true_positive),
                        cluster_false_positive=(cluster_false_positive),
                        evolution_delay_seconds=(time.monotonic() - started if created_now else None),
                        regression_eligible=regression_eligible,
                        regression=(regression_eligible and not task_result.success),
                    )
                )
            return results
        finally:
            await runtime.close_all()


def create_executor() -> ProductionExperimentExecutor:
    app_config = get_app_config()
    model_name = os.environ.get(
        "DEERFLOW_EXPERIMENT_MODEL",
        app_config.models[0].name if app_config.models else "",
    )
    if not model_name:
        raise ValueError("no experiment model is configured")
    return ProductionExperimentExecutor(
        app_config=app_config,
        model_name=model_name,
        model_version=_ollama_digest(app_config, model_name),
        image=os.environ.get(
            "DEERFLOW_REPLAY_IMAGE",
            DEFAULT_REPLAY_IMAGE,
        ),
        docker_binary=os.environ.get("DEERFLOW_DOCKER_BIN"),
        parent_dir=os.environ.get("DEERFLOW_REPLAY_TEMP_DIR"),
        candidate_builder=GeneratedCandidateBuilder(
            app_config=app_config,
            model_name=model_name,
        ),
    )
