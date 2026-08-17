from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from deerflow.skill_evolution.evaluator import (
    Replayability,
    ReplayArtifactVerifier,
    ReplayCommandVerifier,
    ReplayEnvironmentRequirements,
    ReplayFixtureSnapshot,
    ReplaySideEffectPolicy,
    ReplayTaskSpec,
    build_replay_fixture_snapshot,
    build_replay_task_spec,
    capture_replay_fixture_snapshot,
    isolated_replay_workspace,
)
from deerflow.skill_evolution.models import (
    ComplexitySignals,
    EnvironmentSignature,
    EvolutionEvent,
    EvolutionEventKind,
    EvolutionTraceSnapshot,
    OutcomeEvidence,
    OutcomeStatus,
    ProposalOperation,
    ProposalStatus,
    ProposedSkillFile,
    SkillProposal,
    SkillUsage,
    ToolSignature,
    TraceRunStatus,
)

_CREATED = datetime(2026, 8, 16, tzinfo=UTC)
_TASK_INPUT = "Install the dependency from the provided project fixture."


def _event() -> EvolutionEvent:
    return EvolutionEvent(
        event_id="event-1",
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        extractor_version="structured-v1:test-model",
        source_snapshot_hash="a" * 64,
        task_input_hash=hashlib.sha256(_TASK_INPUT.encode()).hexdigest(),
        event_kind=EvolutionEventKind.new_skill_evidence,
        task_signature="python-package-install",
        task_goal="Install a Python dependency and verify the import.",
        environment=EnvironmentSignature(
            os="macOS",
            shell="zsh",
            runtime="Python 3.12",
        ),
        outcome=OutcomeEvidence(
            status=OutcomeStatus.success,
            confidence=0.95,
            sources=["pytest"],
        ),
        complexity=ComplexitySignals(tool_calls=6),
        tool_signature=ToolSignature(
            tool_names=["read_file", "bash", "bash"],
        ),
        skill_usage=SkillUsage(used=False),
        successful_path=[
            "Inspect the project.",
            "Install the dependency.",
            "Run the import check.",
        ],
        reusable_lessons=["Verify the package in the active environment."],
        created_at=_CREATED,
    )


def _snapshot(
    *,
    truncated: bool = False,
) -> EvolutionTraceSnapshot:
    return EvolutionTraceSnapshot(
        snapshot_hash="a" * 64,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        model_name="qwen3-local",
        run_status=TraceRunStatus.success,
        task_input=_TASK_INPUT,
        final_answer="The dependency was installed and verified.",
        environment=EnvironmentSignature(
            os="macOS",
            shell="zsh",
            runtime="Python 3.12",
        ),
        source_event_count=2 if truncated else 1,
        included_event_count=1,
        truncated=truncated,
        created_at=_CREATED,
    )


def _proposal() -> SkillProposal:
    return SkillProposal(
        proposal_id="proposal-1",
        cluster_id="cluster-1",
        user_id="user-1",
        operation=ProposalOperation.create,
        skill_name="python-package-install",
        proposed_files=[
            ProposedSkillFile(
                path="SKILL.md",
                content=("---\nname: python-package-install\ndescription: Install Python dependencies safely.\n---\n\n# Python Package Install\n"),
                executable=False,
            )
        ],
        supporting_event_ids=["event-1"],
        rationale="Evaluate a staged candidate Skill.",
        expected_improvements=["Avoid global environment failures."],
        status=ProposalStatus.staged,
        created_at=_CREATED,
    )


def test_fixture_snapshot_is_deterministic_and_binary_safe() -> None:
    first = build_replay_fixture_snapshot(
        {
            "project/input.txt": b"hello\n",
            "project/data.bin": b"\x00\xff\x10",
        }
    )
    second = build_replay_fixture_snapshot(
        {
            "project/data.bin": b"\x00\xff\x10",
            "project/input.txt": b"hello\n",
        }
    )

    assert first == second
    assert first.total_bytes == 9
    assert first.files[1].decode_content() == b"hello\n"
    assert ReplayFixtureSnapshot.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize(
    "path",
    [
        "../escape.txt",
        "/absolute.txt",
        "folder/../../escape.txt",
        "",
    ],
)
def test_fixture_snapshot_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        build_replay_fixture_snapshot({path: b"data"})


def test_replay_spec_is_automatic_with_deterministic_verifier() -> None:
    fixture = build_replay_fixture_snapshot({"project/pyproject.toml": b"[project]\nname='demo'\n"})
    spec = build_replay_task_spec(
        _snapshot(),
        _event(),
        fixture=fixture,
        verifiers=[
            ReplayCommandVerifier(
                command='python -c "import example"',
                expected_exit_code=0,
                timeout_seconds=30,
            ),
            ReplayArtifactVerifier(
                path="outputs/result.json",
                must_exist=True,
            ),
        ],
    )

    assert spec.replayability is Replayability.automatic
    assert spec.manual_review_reasons == []
    assert spec.task_input == _TASK_INPUT
    assert spec.task_family == "python-package-install"
    assert spec.source_event_id == "event-1"
    assert spec.timeout_seconds == 300
    assert spec.side_effect_policy.skill_library_writes_allowed is False
    assert spec.fixture.snapshot_hash == fixture.snapshot_hash
    assert spec.task_id.startswith("replay-")


def test_missing_verifier_marks_task_manual_review_only() -> None:
    spec = build_replay_task_spec(
        _snapshot(),
        _event(),
        fixture=build_replay_fixture_snapshot({}),
        verifiers=[],
    )

    assert spec.replayability is Replayability.manual_review
    assert "no_deterministic_verifier" in spec.manual_review_reasons


def test_direct_model_cannot_forge_automatic_replayability() -> None:
    manual = build_replay_task_spec(
        _snapshot(),
        _event(),
        fixture=build_replay_fixture_snapshot({}),
        verifiers=[],
    )
    payload = manual.model_dump(mode="python")
    payload["replayability"] = Replayability.automatic
    payload["manual_review_reasons"] = []

    with pytest.raises(ValueError, match="deterministic verifier"):
        ReplayTaskSpec.model_validate(payload)


def test_truncated_trace_and_external_requirements_mark_manual_review() -> None:
    spec = build_replay_task_spec(
        _snapshot(truncated=True),
        _event(),
        fixture=build_replay_fixture_snapshot({}),
        verifiers=[
            ReplayCommandVerifier(
                command="pytest -q",
                expected_exit_code=0,
            )
        ],
        environment=ReplayEnvironmentRequirements(
            os="macOS",
            shell="zsh",
            runtime="Python 3.12",
            network_required=True,
            required_secret_names=["PACKAGE_TOKEN"],
        ),
        side_effect_policy=ReplaySideEffectPolicy(
            network_allowed=False,
        ),
    )

    assert spec.replayability is Replayability.manual_review
    assert set(spec.manual_review_reasons) == {
        "source_trace_truncated",
        "network_required_but_disallowed",
        "external_credentials_required",
    }


def test_snapshot_event_identity_mismatch_is_rejected() -> None:
    event = _event().model_copy(update={"run_id": "other-run"})

    with pytest.raises(ValueError, match="run"):
        build_replay_task_spec(
            _snapshot(),
            event,
            fixture=build_replay_fixture_snapshot({}),
            verifiers=[
                ReplayCommandVerifier(
                    command="pytest -q",
                    expected_exit_code=0,
                )
            ],
        )


@pytest.mark.asyncio
async def test_capture_and_workspace_do_not_mutate_source_or_production(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("source-data", encoding="utf-8")
    production_skill = tmp_path / "production-skills" / "real" / "SKILL.md"
    production_skill.parent.mkdir(parents=True)
    production_skill.write_text("production", encoding="utf-8")
    fixture = await capture_replay_fixture_snapshot(
        source,
        ["input.txt"],
    )
    spec = build_replay_task_spec(
        _snapshot(),
        _event(),
        fixture=fixture,
        verifiers=[
            ReplayCommandVerifier(
                command="pytest -q",
                expected_exit_code=0,
            )
        ],
    )

    async with isolated_replay_workspace(
        spec,
        proposal=_proposal(),
        parent_dir=tmp_path / "replays",
    ) as workspace:
        replay_root = workspace.paths.root
        assert (workspace.paths.workspace / "input.txt").read_text(encoding="utf-8") == "source-data"
        assert (workspace.paths.skills / "python-package-install" / "SKILL.md").exists()
        (workspace.paths.outputs / "result.txt").write_text(
            "result",
            encoding="utf-8",
        )
        assert replay_root != source
        assert replay_root != production_skill.parent

    assert not replay_root.exists()
    assert (source / "input.txt").read_text(encoding="utf-8") == "source-data"
    assert production_skill.read_text(encoding="utf-8") == "production"


@pytest.mark.asyncio
async def test_workspace_cleans_up_after_exception(tmp_path) -> None:
    spec = build_replay_task_spec(
        _snapshot(),
        _event(),
        fixture=build_replay_fixture_snapshot({}),
        verifiers=[
            ReplayCommandVerifier(
                command="pytest -q",
                expected_exit_code=0,
            )
        ],
    )
    replay_root = None

    with pytest.raises(RuntimeError, match="evaluation failed"):
        async with isolated_replay_workspace(
            spec,
            parent_dir=tmp_path,
        ) as workspace:
            replay_root = workspace.paths.root
            raise RuntimeError("evaluation failed")

    assert replay_root is not None
    assert not replay_root.exists()


@pytest.mark.asyncio
async def test_side_effect_audit_allows_outputs_but_rejects_skill_changes(
    tmp_path,
) -> None:
    spec = build_replay_task_spec(
        _snapshot(),
        _event(),
        fixture=build_replay_fixture_snapshot({}),
        verifiers=[
            ReplayCommandVerifier(
                command="pytest -q",
                expected_exit_code=0,
            )
        ],
    )

    async with isolated_replay_workspace(
        spec,
        proposal=_proposal(),
        parent_dir=tmp_path,
    ) as workspace:
        (workspace.paths.outputs / "result.txt").write_text(
            "allowed",
            encoding="utf-8",
        )
        skill_file = workspace.paths.skills / "python-package-install" / "SKILL.md"
        skill_file.chmod(0o644)
        skill_file.write_text("mutated", encoding="utf-8")

        report = await workspace.audit_side_effects()

        assert report.within_policy is False
        assert "outputs/result.txt" in report.changed_paths
        assert "skills/python-package-install/SKILL.md" in report.changed_paths
        assert "skills/python-package-install/SKILL.md" in report.violations


@pytest.mark.asyncio
async def test_capture_rejects_symlinks(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target.txt"
    target.write_text("data", encoding="utf-8")
    link = source / "link.txt"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        await capture_replay_fixture_snapshot(
            source,
            ["link.txt"],
        )
