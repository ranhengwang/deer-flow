from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from deerflow.config.app_config import AppConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.skill_evolution.docker_replay_runtime import (
    DockerReplayRuntime,
    _safe_container_path,
)
from deerflow.skill_evolution.evaluator import (
    EvaluationMetrics,
    Replayability,
    ReplayWorkspacePaths,
    TaskEvaluationResult,
)
from deerflow.skill_evolution.experiment import (
    ExperimentCondition,
    ExperimentManifest,
    build_default_manifest,
    run_experiment,
)
from deerflow.skill_evolution.grouping import (
    group_evolution_events,
)
from deerflow.skill_evolution.production_evolution import (
    CandidateBuildResult,
    GeneratedCandidateBuilder,
    SingleEvidenceCandidateDistiller,
    _deterministic_ready_cluster,
    _event_from_replay,
)
from deerflow.skill_evolution.production_experiment import (
    ProductionExperimentExecutor,
)
from deerflow.skill_evolution.replay_suite import (
    materialize_replay_suite,
)


def _config() -> AppConfig:
    return AppConfig(
        sandbox=SandboxConfig(use="test"),
    )


def test_production_suite_materializes_sixty_automatic_hidden_verifiers() -> None:
    manifest = build_default_manifest()
    cases = materialize_replay_suite(manifest)

    assert len(cases) == 60
    assert len({case.spec.task_id for case in cases}) == 60
    assert len({case.spec.task_input for case in cases}) == 60
    assert all(case.spec.replayability is Replayability.automatic for case in cases)
    assert all(case.spec.fixture.complete for case in cases)
    assert all(case.spec.environment.network_required is False and not case.spec.environment.required_secret_names for case in cases)
    assert all("expected=json.loads" not in case.spec.task_input for case in cases)
    assert all(case.candidate_skill.complete and case.candidate_skill.skill_name == case.benchmark_task.family_id for case in cases)
    assert all((case.base_skill is not None) == (case.benchmark_task.branch == "patch") for case in cases)
    assert all(case.reusable_lesson for case in cases)
    assert all(bool(case.skill_gap) == (case.benchmark_task.branch == "patch") for case in cases)
    assert sum(case.recovered_error_evidence for case in cases) == 12


def test_replay_path_guard_rejects_host_and_skill_writes() -> None:
    assert (
        _safe_container_path(
            "/workspace/input.json",
            writable=False,
        )
        == "/workspace/input.json"
    )
    with pytest.raises(ValueError):
        _safe_container_path("/etc/passwd", writable=False)
    with pytest.raises(ValueError):
        _safe_container_path(
            "/skills/example/SKILL.md",
            writable=True,
        )
    with pytest.raises(ValueError):
        _safe_container_path(
            "/workspace/../etc/passwd",
            writable=True,
        )


def test_repo_preflight_candidate_requires_name_strings() -> None:
    manifest = build_default_manifest()
    case = next(case for case in materialize_replay_suite(manifest) if case.benchmark_task.family_id == "repo-preflight")
    skill_md = next(file.content for file in case.candidate_skill.files if file.path == "SKILL.md")

    assert "name strings" in case.spec.task_input
    assert "name strings" in skill_md


def test_evidence_fixtures_are_regular_and_held_out_keeps_edge_cases() -> None:
    cases = materialize_replay_suite(
        build_default_manifest(),
    )
    by_family_split = {
        (
            case.benchmark_task.family_id,
            case.benchmark_task.split,
        ): case
        for case in cases
        if case.benchmark_task.variant_index in {1, 4}
    }

    def input_json(
        family: str,
        split: str,
    ) -> dict:
        case = by_family_split[(family, split)]
        fixture_file = next(item for item in case.spec.fixture.files if item.path == "input.json")
        return json.loads(
            fixture_file.decode_content(),
        )

    evidence_paths = input_json(
        "repo-path-normalization",
        "evidence",
    )["paths"]
    held_out_paths = input_json(
        "repo-path-normalization",
        "held_out",
    )["paths"]
    assert evidence_paths == sorted(evidence_paths)
    assert "../secret.txt" not in evidence_paths
    assert "../secret.txt" in held_out_paths

    evidence_rows = input_json(
        "data-schema-validation",
        "evidence",
    )["rows"]
    held_out_rows = input_json(
        "data-schema-validation",
        "held_out",
    )["rows"]
    assert all(row["id"] and row["value"] >= 0 for row in evidence_rows)
    assert any(not row["id"] or row["value"] < 0 for row in held_out_rows)

    evidence_stages = input_json(
        "shell-artifact-pipeline",
        "evidence",
    )["stages"]
    held_out_stages = input_json(
        "shell-artifact-pipeline",
        "held_out",
    )["stages"]
    assert evidence_stages == [stage.strip().lower() for stage in evidence_stages]
    assert held_out_stages != [stage.strip().lower() for stage in held_out_stages]

    evidence_fallback = input_json(
        "shell-command-fallback",
        "evidence",
    )
    held_out_fallback = input_json(
        "shell-command-fallback",
        "held_out",
    )
    assert evidence_fallback["available"][0] == "python"
    assert held_out_fallback["available"][0] != "python"


def test_docker_runtime_write_json_serializes_structured_value(
    monkeypatch,
) -> None:
    runtime = DockerReplayRuntime(
        app_config=_config(),
        model_name="test-model",
        docker_binary="/bin/echo",
    )
    captured: dict[str, str | None] = {}

    def fake_exec(
        container,
        command,
        *,
        timeout,
        input_text=None,
    ):
        captured["command"] = command
        captured["input_text"] = input_text
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(runtime, "_exec", fake_exec)
    tools = {
        replay_tool.name: replay_tool
        for replay_tool in runtime._tools(
            SimpleNamespace(name="replay-test"),
        )
    }

    result = tools["write_json"].invoke(
        {
            "path": "/outputs/result.json",
            "value": {
                "enabled_checks": [
                    "unit-tests",
                    "lint-4",
                ]
            },
        }
    )

    assert result == "ok"
    assert json.loads(captured["input_text"]) == {
        "enabled_checks": [
            "unit-tests",
            "lint-4",
        ]
    }


def test_docker_runtime_can_restrict_model_visible_tools() -> None:
    runtime = DockerReplayRuntime(
        app_config=_config(),
        model_name="test-model",
        docker_binary="/bin/echo",
        allowed_agent_tools=(
            "read_file",
            "write_json",
        ),
    )

    assert [
        replay_tool.name
        for replay_tool in runtime._tools(
            SimpleNamespace(name="replay-test"),
        )
    ] == [
        "read_file",
        "write_json",
    ]

    with pytest.raises(ValueError, match="unknown replay agent tool"):
        DockerReplayRuntime(
            app_config=_config(),
            model_name="test-model",
            docker_binary="/bin/echo",
            allowed_agent_tools=("network_fetch",),
        )


def test_generated_candidate_events_preserve_branch_and_recovery() -> None:
    cases = materialize_replay_suite(
        build_default_manifest(),
    )
    create_case = next(case for case in cases if case.benchmark_task.family_id == "data-schema-validation" and case.benchmark_task.variant_index == 2)
    patch_cases = [case for case in cases if case.benchmark_task.family_id == "repo-preflight" and case.benchmark_task.split == "evidence"]

    def successful_result(
        case,
    ) -> TaskEvaluationResult:
        return TaskEvaluationResult(
            task_id=case.spec.task_id,
            split="source",
            condition=("base_skill" if case.base_skill is not None else "no_skill"),
            success=True,
            metrics=EvaluationMetrics(
                tool_calls=2,
                input_tokens=100,
                output_tokens=20,
                latency_seconds=1.0,
            ),
        )

    create_event = _event_from_replay(
        create_case,
        successful_result(create_case),
    )
    patch_events = [
        _event_from_replay(
            case,
            successful_result(case),
        )
        for case in patch_cases
    ]

    assert create_event.event_kind.value == "new_skill_evidence"
    assert create_event.skill_usage.used is False
    assert create_event.complexity.had_recoverable_errors is True
    assert all(event.event_kind.value == "skill_patch_evidence" for event in patch_events)
    assert all(event.skill_usage.content_hash == patch_cases[0].base_skill.skill_md_hash for event in patch_events)
    assert all(event.skill_gaps for event in patch_events)

    grouped = group_evolution_events(patch_events)
    ready = _deterministic_ready_cluster(
        grouped.clusters[0],
        patch_events,
        required_events=3,
    )

    assert ready.status.value == "ready"
    assert [item.run_id for item in ready.member_evidence] == [event.run_id for event in patch_events]


@pytest.mark.asyncio
async def test_single_evidence_candidate_is_model_generated() -> None:
    case = next(
        case
        for case in materialize_replay_suite(
            build_default_manifest(),
        )
        if case.benchmark_task.family_id == "repo-preflight" and case.benchmark_task.variant_index == 1
    )
    result = TaskEvaluationResult(
        task_id=case.spec.task_id,
        split="source",
        condition="base_skill",
        success=True,
        metrics=EvaluationMetrics(
            tool_calls=2,
            input_tokens=100,
            output_tokens=20,
            latency_seconds=1.0,
        ),
    )

    class Model:
        def __init__(self) -> None:
            self.messages = None

        async def ainvoke(
            self,
            messages,
            config=None,
        ):
            self.messages = messages
            return AIMessage(
                content=json.dumps(
                    {
                        "skill_name": "repo-preflight",
                        "description": ("Process repository preflight checks."),
                        "overview": ("Filter and order preflight checks."),
                        "steps": [
                            "Read input.json.",
                            "Write the required result JSON.",
                        ],
                        "verification_steps": [
                            "Confirm result.json exists.",
                        ],
                        "rationale": ("The replay succeeded with this workflow."),
                    }
                )
            )

    model = Model()
    package = await SingleEvidenceCandidateDistiller(
        model=model,
        model_name="test-model",
    ).distill(
        case,
        result,
    )

    assert package is not None
    assert package.skill_name == "repo-preflight"
    prompt = model.messages[1].content
    assert "Sort every check alphabetically" in prompt
    skill_md = next(item.content for item in package.files if item.path == "SKILL.md")
    assert "Filter and order preflight checks." in skill_md
    assert skill_md != next(item.content for item in case.candidate_skill.files if item.path == "SKILL.md")


@pytest.mark.asyncio
async def test_generated_builder_reuses_same_seed_evidence_candidate(
    monkeypatch,
) -> None:
    case = next(
        case
        for case in materialize_replay_suite(
            build_default_manifest(),
        )
        if case.benchmark_task.family_id == "repo-preflight" and case.benchmark_task.variant_index == 1
    )
    result = TaskEvaluationResult(
        task_id=case.spec.task_id,
        split="source",
        condition="base_skill",
        success=True,
        metrics=EvaluationMetrics(
            tool_calls=2,
            input_tokens=100,
            output_tokens=20,
            latency_seconds=1.0,
        ),
    )

    class Model:
        calls = 0

        async def ainvoke(
            self,
            messages,
            config=None,
        ):
            del messages, config
            self.calls += 1
            return AIMessage(
                content=json.dumps(
                    {
                        "skill_name": "repo-preflight",
                        "description": ("Process repository preflight checks."),
                        "overview": ("Filter and order preflight checks."),
                        "steps": [
                            "Read input.json.",
                            "Write the required result JSON.",
                        ],
                        "verification_steps": [
                            "Confirm result.json exists.",
                        ],
                        "rationale": ("The replay succeeded with this workflow."),
                    }
                )
            )

    model = Model()
    monkeypatch.setattr(
        "deerflow.skill_evolution.production_evolution.create_chat_model",
        lambda **kwargs: model,
    )
    builder = GeneratedCandidateBuilder(
        app_config=_config(),
        model_name="test-model",
    )
    evidence = [
        (
            case,
            result,
        )
    ]

    first = await builder.build(
        evidence,
        ExperimentCondition(
            condition_id="immediate",
            baseline_kind="immediate_single",
        ),
        1,
    )
    second = await builder.build(
        evidence,
        ExperimentCondition(
            condition_id="staged",
            baseline_kind="staged_single",
        ),
        1,
    )

    assert first == second
    assert first.package is not None
    assert model.calls == 1


def test_docker_runtime_starts_with_hardened_mounts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = ReplayWorkspacePaths(
        root=tmp_path,
        workspace=tmp_path / "workspace",
        outputs=tmp_path / "outputs",
        skills=tmp_path / "skills",
    )
    paths.workspace.mkdir()
    paths.outputs.mkdir()
    paths.skills.mkdir()
    runtime = DockerReplayRuntime(
        app_config=_config(),
        model_name="test-model",
        docker_binary="/bin/echo",
    )
    calls = []

    def fake_run(args, *, timeout, input_text=None):
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="container-id\n",
            stderr="",
        )

    monkeypatch.setattr(runtime, "_run_host", fake_run)
    container = runtime._start_container(paths)

    command = calls[0]
    assert container.name.startswith("deerflow-replay-")
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert ["--cap-drop", "ALL"] == command[command.index("--cap-drop") : command.index("--cap-drop") + 2]
    assert "no-new-privileges" in command
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert any("dst=/workspace" in value for value in mounts)
    assert any("dst=/outputs" in value for value in mounts)
    assert any("dst=/skills,readonly" in value for value in mounts)


class _FakeRuntime:
    def preflight(self) -> None:
        return None

    async def close_all(self) -> None:
        return None

    def environment_fingerprint(self) -> str:
        return "f" * 64


class _RecordingCandidateBuilder:
    executor_name = "recording-builder"
    executor_version = "recording-builder-v1"
    prompt_version = "recording-prompt-v1"

    def __init__(self) -> None:
        self.calls = []

    async def build(
        self,
        evidence,
        condition,
        seed,
    ) -> CandidateBuildResult:
        self.calls.append(
            (
                [case.benchmark_task.task_id for case, _ in evidence],
                condition.grouping_strategy,
                seed,
            )
        )
        return CandidateBuildResult(
            package=evidence[-1][0].candidate_skill,
            cluster_true_positive=len(evidence),
            cluster_false_positive=0,
            generated=True,
        )


@pytest.mark.asyncio
async def test_production_cohort_orders_evidence_validation_and_held_out(
    monkeypatch,
) -> None:
    manifest = build_default_manifest()
    tasks = [task for task in manifest.tasks if task.family_id == "repo-path-normalization"]
    subset = ExperimentManifest(
        manifest_id="production-test",
        tasks=tasks,
    )
    calls: list[tuple[str, str]] = []

    async def fake_replay(
        runtime,
        spec,
        *,
        split,
        condition,
        skill_package,
        parent_dir,
    ):
        calls.append((spec.task_id, condition))
        return TaskEvaluationResult(
            task_id=spec.task_id,
            split=split,
            condition=condition,
            success=True,
            metrics=EvaluationMetrics(
                tool_calls=2,
                input_tokens=100,
                output_tokens=20,
                latency_seconds=1.0,
            ),
            environment_fingerprint="e" * 64,
        )

    monkeypatch.setattr(
        "deerflow.skill_evolution.production_experiment.run_replay_task",
        fake_replay,
    )
    candidate_builder = _RecordingCandidateBuilder()
    executor = ProductionExperimentExecutor(
        app_config=_config(),
        model_name="qwen3-local",
        model_version="a" * 64,
        runtime_factory=lambda seed: _FakeRuntime(),
        candidate_builder=candidate_builder,
    )
    rows = await run_experiment(
        experiment_id="production-test",
        manifest=subset,
        conditions=[
            ExperimentCondition(
                condition_id="proposed-k3-hybrid",
                baseline_kind="proposed",
            )
        ],
        seeds=[1],
        executor=executor,
        max_concurrency=1,
    )

    assert len(rows) == 5
    assert [condition for _, condition in calls[:3]] == [
        "no_skill",
        "no_skill",
        "no_skill",
    ]
    assert [condition for _, condition in calls[3:6]] == [
        "candidate_skill",
        "candidate_skill",
        "candidate_skill",
    ]
    assert [condition for _, condition in calls[6:]] == [
        "candidate_skill",
        "candidate_skill",
    ]
    assert rows[2].proposal_created is True
    assert rows[2].proposal_accepted is True
    assert all(row.task_success for row in rows)
    assert candidate_builder.calls == [
        (
            [
                "repo-path-normalization-evidence-1",
                "repo-path-normalization-evidence-2",
                "repo-path-normalization-evidence-3",
            ],
            "hybrid",
            1,
        )
    ]
    assert all(row.executor_name == "recording-builder" for row in rows)
