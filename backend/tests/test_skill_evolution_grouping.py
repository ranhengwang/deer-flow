from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from deerflow.config.skill_evolution_config import (
    SkillEvolutionGroupingConfig,
)
from deerflow.skill_evolution.grouping import (
    DeduplicationReason,
    build_event_fingerprint,
    compare_event_fingerprints,
    group_evolution_events,
)
from deerflow.skill_evolution.models import (
    ClusterStatus,
    ComplexitySignals,
    EnvironmentSignature,
    EvolutionEvent,
    EvolutionEventKind,
    OutcomeEvidence,
    OutcomeStatus,
    SkillGap,
    SkillGapCategory,
    SkillTarget,
    SkillUsage,
    ToolSignature,
)

_CREATED = datetime(2026, 8, 16, tzinfo=UTC)


def _event(
    event_id: str,
    *,
    run_id: str | None = None,
    task_input_hash: str | None = None,
    task_signature: str = "python-package-install",
    task_goal: str = ("Install a Python dependency in the project environment."),
    tool_names: list[str] | None = None,
    error_types: list[str] | None = None,
    os_name: str = "macOS",
    runtime: str = "Python 3.12",
    user_id: str = "user-1",
    target_skill: str | None = None,
    target_hash: str = "b" * 64,
    created_offset: int = 0,
) -> EvolutionEvent:
    patch = target_skill is not None
    skill_usage = (
        SkillUsage(
            used=True,
            skill_name=target_skill,
            skill_path=(f"/mnt/skills/custom/{target_skill}/SKILL.md"),
            content_hash=target_hash,
            activation_source="read",
        )
        if patch
        else SkillUsage(used=False)
    )
    return EvolutionEvent(
        event_id=event_id,
        run_id=run_id or f"run-{event_id}",
        thread_id=f"thread-{event_id}",
        user_id=user_id,
        extractor_version="structured-v1:test-model",
        source_snapshot_hash="a" * 64,
        task_input_hash=task_input_hash or f"{int(event_id[-1], 36) % 16:x}" * 64,
        event_kind=(EvolutionEventKind.skill_patch_evidence if patch else EvolutionEventKind.new_skill_evidence),
        task_signature=task_signature,
        task_goal=task_goal,
        environment=EnvironmentSignature(
            os=os_name,
            shell="zsh" if os_name.lower() != "windows" else "pwsh",
            runtime=runtime,
        ),
        outcome=OutcomeEvidence(
            status=OutcomeStatus.success,
            confidence=0.95,
            sources=["tests"],
        ),
        complexity=ComplexitySignals(tool_calls=6),
        tool_signature=ToolSignature(
            tool_names=tool_names or ["bash", "read_file", "bash"],
            error_types=error_types or [],
        ),
        skill_usage=skill_usage,
        successful_path=[
            "Inspect the environment.",
            "Run the workflow.",
            "Verify the result.",
        ],
        reusable_lessons=["Inspect the environment before execution."],
        skill_gaps=(
            [
                SkillGap(
                    category=(SkillGapCategory.missing_prerequisite),
                    evidence="Environment inspection was missing.",
                    recommended_change=("Inspect the environment first."),
                )
            ]
            if patch
            else []
        ),
        target_skill=(
            SkillTarget(
                name=target_skill,
                content_hash=target_hash,
            )
            if patch
            else None
        ),
        created_at=_CREATED + timedelta(seconds=created_offset),
    )


def test_fingerprint_is_normalized_and_deterministic() -> None:
    first = _event(
        "event-1",
        task_goal="Install a PYTHON dependency, in the project!",
        os_name="Darwin",
        runtime="python3.12.4",
    )
    second = _event(
        "event-2",
        task_goal="install a python dependency in the project",
        os_name="macOS",
        runtime="Python 3.12",
    )

    first_fingerprint = build_event_fingerprint(first)
    second_fingerprint = build_event_fingerprint(second)

    assert first_fingerprint.task_goal_tokens == second_fingerprint.task_goal_tokens
    assert first_fingerprint.environment_family == ("macos|python:3|posix")
    assert first_fingerprint.fingerprint_hash == second_fingerprint.fingerprint_hash


def test_obvious_same_family_events_group_together() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event(
            "event-2",
            task_goal=("Set up a Python package inside the project environment."),
            created_offset=2,
        ),
        _event(
            "event-3",
            task_goal=("Add a Python dependency with the project package manager and verify it."),
            tool_names=["read_file", "bash", "bash"],
            created_offset=3,
        ),
    ]

    result = group_evolution_events(events)

    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.member_event_ids == [
        "event-1",
        "event-2",
        "event-3",
    ]
    assert cluster.independent_run_count == 3
    assert cluster.status is ClusterStatus.collecting
    assert len(cluster.grouping_evidence) == 3
    assert result.duplicates == []


def test_shared_words_with_different_workflow_do_not_group() -> None:
    install = _event("event-1")
    report = _event(
        "event-2",
        task_signature="python-package-report",
        task_goal=("Create a Python package report for the project."),
        tool_names=[
            "read_file",
            "write_file",
            "present_files",
        ],
    )

    comparison = compare_event_fingerprints(
        build_event_fingerprint(install),
        build_event_fingerprint(report),
    )
    result = group_evolution_events([install, report])

    assert comparison.compatible is False
    assert len(result.clusters) == 2


def test_new_skill_and_patch_events_never_group() -> None:
    new_skill = _event("event-1")
    patch = _event(
        "event-2",
        target_skill="package-repair",
    )

    comparison = compare_event_fingerprints(
        build_event_fingerprint(new_skill),
        build_event_fingerprint(patch),
    )
    result = group_evolution_events([new_skill, patch])

    assert comparison.compatible is False
    assert comparison.hard_mismatch == "event_kind"
    assert len(result.clusters) == 2


def test_patch_events_for_different_targets_never_group() -> None:
    first = _event(
        "event-1",
        target_skill="package-repair",
    )
    second = _event(
        "event-2",
        target_skill="environment-setup",
    )

    comparison = compare_event_fingerprints(
        build_event_fingerprint(first),
        build_event_fingerprint(second),
    )

    assert comparison.compatible is False
    assert comparison.hard_mismatch == "target_skill"
    assert len(group_evolution_events([first, second]).clusters) == 2


def test_same_run_is_deduplicated() -> None:
    first = _event("event-1", run_id="run-shared")
    retry = _event(
        "event-2",
        run_id="run-shared",
        task_goal="Install package after retry.",
    )

    result = group_evolution_events([first, retry])

    assert len(result.clusters) == 1
    assert result.clusters[0].member_event_ids == ["event-1"]
    assert result.duplicates[0].reason is (DeduplicationReason.same_run)
    assert result.duplicates[0].canonical_event_id == "event-1"


def test_conflicting_payload_for_same_event_id_is_rejected() -> None:
    first = _event("event-1")
    conflicting = _event(
        "event-1",
        task_goal="Generate an unrelated report.",
        task_signature="report-generation",
    )

    with pytest.raises(ValueError, match="event ID"):
        group_evolution_events([first, conflicting])


def test_same_task_input_hash_is_deduplicated() -> None:
    shared_hash = "c" * 64
    first = _event(
        "event-1",
        task_input_hash=shared_hash,
    )
    duplicate = _event(
        "event-2",
        task_input_hash=shared_hash,
        task_goal=("Install the requested Python dependency and verify it."),
    )

    result = group_evolution_events([first, duplicate])

    assert len(result.clusters[0].member_event_ids) == 1
    assert result.duplicates[0].reason is (DeduplicationReason.same_task_input)


def test_identical_fingerprint_is_near_duplicate() -> None:
    first = _event(
        "event-1",
        task_input_hash="1" * 64,
    )
    duplicate = _event(
        "event-2",
        task_input_hash="2" * 64,
    )

    result = group_evolution_events([first, duplicate])

    assert result.clusters[0].member_event_ids == ["event-1"]
    assert result.duplicates[0].reason is (DeduplicationReason.identical_fingerprint)


def test_user_scope_is_a_hard_grouping_boundary() -> None:
    first = _event("event-1", user_id="alice")
    second = _event("event-2", user_id="bob")

    comparison = compare_event_fingerprints(
        build_event_fingerprint(first),
        build_event_fingerprint(second),
    )
    result = group_evolution_events([first, second])

    assert comparison.compatible is False
    assert comparison.hard_mismatch == "user_id"
    assert len(result.clusters) == 2


def test_grouping_is_independent_of_input_order() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event(
            "event-2",
            task_goal=("Set up a Python package in the project environment."),
            created_offset=2,
        ),
        _event(
            "event-3",
            task_signature="csv-report-generation",
            task_goal="Create a CSV report from the input records.",
            tool_names=["read_file", "write_file"],
            created_offset=3,
        ),
    ]

    first = group_evolution_events(events)
    second = group_evolution_events(list(reversed(events)))

    assert first == second


def test_deterministic_threshold_is_configurable() -> None:
    config = SkillEvolutionGroupingConfig(
        deterministic_threshold=0.9,
    )

    assert config.deterministic_threshold == 0.9
