from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from deerflow.skill_evolution.models import (
    ClusterEnvironmentRelationship,
    ClusterMemberEvidence,
    ClusterStatus,
    ComplexitySignals,
    EnvironmentSignature,
    EvolutionCluster,
    EvolutionEvent,
    EvolutionEventKind,
    GroupingEvidence,
    OutcomeEvidence,
    OutcomeStatus,
    ProposalOperation,
    ProposalStatus,
    SkillGap,
    SkillGapCategory,
    SkillTarget,
    SkillUsage,
    ToolSignature,
)
from deerflow.skill_evolution.patch_distiller import (
    PATCH_SKILL_DISTILLATION_PROMPT_VERSION,
    IneligiblePatchClusterError,
    MissingSkillVersionError,
    PatchSkillDistiller,
    SkillStorageVersionSource,
    apply_structured_patch,
)
from deerflow.skill_evolution.store.memory import InMemorySkillEvolutionStore
from deerflow.skills.storage.local_skill_storage import LocalSkillStorage

_CREATED = datetime(2026, 8, 16, tzinfo=UTC)
_BASE_V1 = """---
name: package-repair
description: Repair Python package installation problems.
---

# Package Repair

## Workflow

1. Install the package globally.
2. Verify the import.

## Unaffected Guidance

Keep this section unchanged.
"""
_BASE_V2 = _BASE_V1.replace(
    "1. Install the package globally.",
    "1. Install the package in the active environment.",
)
_BASE_V3 = _BASE_V2.replace(
    "2. Verify the import.",
    "2. Verify the import from the project directory.",
)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _event(
    index: int,
    *,
    target_hash: str,
    os_name: str = "macOS",
) -> EvolutionEvent:
    return EvolutionEvent(
        event_id=f"event-{index}",
        run_id=f"run-{index}",
        thread_id=f"thread-{index}",
        user_id="user-1",
        extractor_version="structured-v1:test-model",
        source_snapshot_hash=f"{index:x}" * 64,
        task_input_hash=f"{index + 3:x}" * 64,
        event_kind=EvolutionEventKind.skill_patch_evidence,
        task_signature="python-package-repair",
        task_goal=f"Repair a Python package workflow variant {index}.",
        environment=EnvironmentSignature(
            os=os_name,
            shell="pwsh" if os_name == "Windows" else "zsh",
            runtime="Python 3.12",
        ),
        outcome=OutcomeEvidence(
            status=OutcomeStatus.success,
            confidence=0.95,
            sources=["tests"],
        ),
        complexity=ComplexitySignals(tool_calls=7),
        tool_signature=ToolSignature(
            tool_names=["read_file", "bash", "bash"],
            error_types=["permission"],
        ),
        skill_usage=SkillUsage(
            used=True,
            skill_name="package-repair",
            skill_path="/mnt/skills/custom/package-repair/SKILL.md",
            content_hash=target_hash,
            activation_source="read",
        ),
        successful_path=[
            "Inspect the active environment.",
            "Install into the project environment.",
            "Verify the package import.",
        ],
        reusable_lessons=["Avoid modifying the global interpreter."],
        skill_gaps=[
            SkillGap(
                category=SkillGapCategory.missing_prerequisite,
                evidence="The original workflow did not inspect the environment.",
                recommended_change="Inspect the active environment first.",
            )
        ],
        target_skill=SkillTarget(
            name="package-repair",
            content_hash=target_hash,
        ),
        created_at=_CREATED + timedelta(seconds=index),
    )


def _ready_cluster(events: list[EvolutionEvent]) -> EvolutionCluster:
    target = events[0].target_skill
    assert target is not None
    return EvolutionCluster(
        cluster_id="cluster-package-repair",
        user_id="user-1",
        event_kind=EvolutionEventKind.skill_patch_evidence,
        target_skill=target,
        canonical_signature="python-package-repair",
        member_event_ids=[event.event_id for event in events],
        independent_run_count=len({event.run_id for event in events}),
        status=ClusterStatus.ready,
        grouping_evidence=[
            GroupingEvidence(
                method="llm" if index else "deterministic",
                score=1.0,
                reason=("Prototype event." if index == 0 else "Confirmed as the same workflow."),
            )
            for index, _ in enumerate(events)
        ],
        member_evidence=[
            ClusterMemberEvidence(
                event_id=event.event_id,
                run_id=event.run_id,
                relationship=(ClusterEnvironmentRelationship.prototype if index == 0 else ClusterEnvironmentRelationship.same_workflow),
                grouping_evidence=[
                    GroupingEvidence(
                        method=("deterministic" if index == 0 else "llm"),
                        score=1.0,
                        reason=("Prototype event." if index == 0 else "Confirmed as the same workflow."),
                    )
                ],
            )
            for index, event in enumerate(events)
        ],
        confirmation_model_name="cluster-model",
        confirmation_prompt_version="cluster-confirmation-v1",
        created_at=events[0].created_at,
        updated_at=events[-1].created_at,
    )


def _operation(
    *,
    find: str,
    replace: str,
    evidence_event_ids: list[str] | None = None,
    environment_condition: str | None = None,
) -> dict[str, Any]:
    return {
        "path": "SKILL.md",
        "find": find,
        "replace": replace,
        "expected_count": 1,
        "reason": "Add the repeatedly observed missing prerequisite.",
        "environment_condition": environment_condition,
        "evidence_event_ids": evidence_event_ids
        or [
            "event-1",
            "event-2",
            "event-3",
        ],
    }


def _output(
    *,
    operations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "skill_name": "package-repair",
        "patch_operations": operations,
        "conflicts": conflicts or [],
        "rationale": ("Repeated successful repairs show that environment inspection is required."),
        "expected_improvements": ["Avoid global interpreter permission failures."],
        "risks": ["Environment activation commands vary by operating system."],
    }


class _FakeModel:
    def __init__(
        self,
        responses: list[Any],
        *,
        on_call: Any = None,
    ) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[Any], dict[str, Any] | None]] = []
        self.on_call = on_call

    async def ainvoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append((messages, config))
        if self.on_call is not None:
            self.on_call(len(self.calls))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return AIMessage(content=response)
        return response


class _MemoryVersionSource:
    def __init__(
        self,
        *,
        current: str,
        history: list[str],
    ) -> None:
        self.current = current
        self.history = history

    async def read_current(self, skill_name: str) -> str:
        assert skill_name == "package-repair"
        return self.current

    async def resolve_exact(
        self,
        skill_name: str,
        content_hash: str,
    ) -> str | None:
        assert skill_name == "package-repair"
        for content in [self.current, *self.history]:
            if _hash(content) == content_hash:
                return content
        return None


@pytest.mark.asyncio
async def test_patch_distillation_preserves_unaffected_content() -> None:
    target_hash = _hash(_BASE_V1)
    events = [_event(index, target_hash=target_hash) for index in range(1, 4)]
    source = _MemoryVersionSource(current=_BASE_V1, history=[])
    output = _output(
        operations=[
            _operation(
                find="1. Install the package globally.",
                replace=("1. Inspect the active environment, then install the package there."),
            )
        ]
    )
    distiller = PatchSkillDistiller(
        model=_FakeModel([output]),
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
        source,
    )

    assert proposal is not None
    assert proposal.operation is ProposalOperation.patch
    assert proposal.status is ProposalStatus.staged
    assert proposal.skill_name == "package-repair"
    assert proposal.base_skill_hash == target_hash
    assert proposal.source_skill_hashes == [target_hash]
    assert proposal.distiller_prompt_version == (PATCH_SKILL_DISTILLATION_PROMPT_VERSION)
    assert len(proposal.patch_operations) == 1
    patch = proposal.patch_operations[0]
    assert patch.find == "1. Install the package globally."
    assert patch.expected_count == 1
    rendered = proposal.proposed_files[0].content
    assert "Inspect the active environment" in rendered
    assert "Keep this section unchanged." in rendered
    assert _BASE_V1 != rendered
    assert proposal.evidence_mapping[0].section == "patch:1"
    assert proposal.evidence_mapping[0].supporting_event_ids == [
        "event-1",
        "event-2",
        "event-3",
    ]


@pytest.mark.asyncio
async def test_exact_historical_base_is_loaded_before_rebasing_current() -> None:
    target_hash = _hash(_BASE_V1)
    events = [_event(index, target_hash=target_hash) for index in range(1, 4)]
    source = _MemoryVersionSource(
        current=_BASE_V2,
        history=[_BASE_V1],
    )
    output = _output(
        operations=[
            _operation(
                find="2. Verify the import.",
                replace=("2. Verify the import from the active project environment."),
            )
        ]
    )
    model = _FakeModel([output])
    distiller = PatchSkillDistiller(
        model=model,
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
        source,
    )

    assert proposal is not None
    assert proposal.source_skill_hashes == [target_hash]
    assert proposal.base_skill_hash == _hash(_BASE_V2)
    prompt = str(model.calls[0][0][1].content)
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["observed_source_content"] == _BASE_V1
    assert payload["patch_base_content"] == _BASE_V2
    assert "Keep this section unchanged." in proposal.proposed_files[0].content


@pytest.mark.asyncio
async def test_stale_current_version_restarts_distillation() -> None:
    target_hash = _hash(_BASE_V1)
    events = [_event(index, target_hash=target_hash) for index in range(1, 4)]
    source = _MemoryVersionSource(
        current=_BASE_V2,
        history=[_BASE_V1],
    )

    def mutate_after_first_call(call_count: int) -> None:
        if call_count == 1:
            source.current = _BASE_V3

    first = _output(
        operations=[
            _operation(
                find="2. Verify the import.",
                replace="2. Verify from the active environment.",
            )
        ]
    )
    second = _output(
        operations=[
            _operation(
                find="2. Verify the import from the project directory.",
                replace=("2. Verify the import from the active project directory."),
            )
        ]
    )
    model = _FakeModel(
        [first, second],
        on_call=mutate_after_first_call,
    )
    distiller = PatchSkillDistiller(
        model=model,
        model_name="distill-model",
        retry_delay_seconds=0,
        max_rebase_attempts=2,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
        source,
    )

    assert proposal is not None
    assert len(model.calls) == 2
    assert proposal.base_skill_hash == _hash(_BASE_V3)
    assert "active project directory" in proposal.proposed_files[0].content
    second_prompt = str(model.calls[1][0][1].content)
    second_payload = json.loads(second_prompt.split("\n", 1)[1])
    assert second_payload["patch_base_content"] == _BASE_V3


@pytest.mark.asyncio
async def test_environment_specific_patch_keeps_explicit_condition() -> None:
    target_hash = _hash(_BASE_V1)
    events = [
        _event(1, target_hash=target_hash),
        _event(2, target_hash=target_hash),
        _event(3, target_hash=target_hash, os_name="Windows"),
    ]
    output = _output(
        operations=[
            _operation(
                find="2. Verify the import.",
                replace=("2. Verify the import.\n   - On Windows, activate the environment with PowerShell."),
                evidence_event_ids=["event-3"],
                environment_condition="Windows with PowerShell",
            )
        ]
    )
    distiller = PatchSkillDistiller(
        model=_FakeModel([output]),
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
        _MemoryVersionSource(current=_BASE_V1, history=[]),
    )

    assert proposal is not None
    patch = proposal.patch_operations[0]
    assert patch.environment_condition == "Windows with PowerShell"
    assert patch.supporting_event_ids == ["event-3"]
    assert "On Windows" in proposal.proposed_files[0].content


@pytest.mark.asyncio
async def test_full_file_replacement_is_rejected() -> None:
    target_hash = _hash(_BASE_V1)
    events = [_event(index, target_hash=target_hash) for index in range(1, 4)]
    output = _output(
        operations=[
            _operation(
                find=_BASE_V1,
                replace=_BASE_V2,
            )
        ]
    )
    distiller = PatchSkillDistiller(
        model=_FakeModel([output]),
        model_name="distill-model",
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert (
        await distiller.distill(
            _ready_cluster(events),
            events,
            _MemoryVersionSource(current=_BASE_V1, history=[]),
        )
        is None
    )


@pytest.mark.asyncio
async def test_missing_or_mixed_source_versions_fail_before_model() -> None:
    target_hash = _hash(_BASE_V1)
    events = [_event(index, target_hash=target_hash) for index in range(1, 4)]
    model = _FakeModel([])
    distiller = PatchSkillDistiller(
        model=model,
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    with pytest.raises(MissingSkillVersionError, match="exact source"):
        await distiller.distill(
            _ready_cluster(events),
            events,
            _MemoryVersionSource(current=_BASE_V2, history=[]),
        )

    mixed = list(events)
    mixed[2] = _event(3, target_hash=_hash(_BASE_V2))
    with pytest.raises(IneligiblePatchClusterError, match="one source Skill version"):
        await distiller.distill(
            _ready_cluster(events),
            mixed,
            _MemoryVersionSource(
                current=_BASE_V2,
                history=[_BASE_V1],
            ),
        )

    assert model.calls == []


@pytest.mark.asyncio
async def test_conflicts_stage_manual_review_without_applying_conflict() -> None:
    target_hash = _hash(_BASE_V1)
    events = [_event(index, target_hash=target_hash) for index in range(1, 4)]
    conflict = {
        "description": ("The events disagree on whether global installation is ever allowed."),
        "evidence_event_ids": ["event-1", "event-2"],
    }
    output = _output(
        operations=[
            _operation(
                find="2. Verify the import.",
                replace="2. Verify the import from the active environment.",
            )
        ],
        conflicts=[conflict],
    )
    distiller = PatchSkillDistiller(
        model=_FakeModel([output]),
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
        _MemoryVersionSource(current=_BASE_V1, history=[]),
    )

    assert proposal is not None
    assert proposal.requires_manual_review is True
    assert "unresolved_conflicts" in proposal.review_reasons
    assert conflict["description"] in proposal.risks
    assert conflict["description"] not in proposal.proposed_files[0].content


@pytest.mark.asyncio
async def test_patch_proposal_persists_idempotently_without_writing_skill() -> None:
    target_hash = _hash(_BASE_V1)
    events = [_event(index, target_hash=target_hash) for index in range(1, 4)]
    source = _MemoryVersionSource(current=_BASE_V1, history=[])
    output = _output(
        operations=[
            _operation(
                find="2. Verify the import.",
                replace="2. Verify the import from the active environment.",
            )
        ]
    )
    distiller = PatchSkillDistiller(
        model=_FakeModel([output, output]),
        model_name="distill-model",
        retry_delay_seconds=0,
    )
    store = InMemorySkillEvolutionStore()
    cluster = _ready_cluster(events)

    first = await distiller.distill_and_persist(
        cluster,
        events,
        source,
        store,
    )
    second = await distiller.distill_and_persist(
        cluster,
        events,
        source,
        store,
    )

    assert first is not None and first.created is True
    assert second is not None and second.created is False
    assert first.value == second.value
    assert source.current == _BASE_V1
    assert first.value.status is ProposalStatus.staged


def test_apply_structured_patch_requires_unique_exact_target() -> None:
    operation = _operation(
        find="repeat",
        replace="replacement",
        evidence_event_ids=["event-1"],
    )

    with pytest.raises(ValueError, match="expected 1"):
        apply_structured_patch(
            "repeat\nrepeat\n",
            [operation],
            skill_name="package-repair",
        )


@pytest.mark.asyncio
async def test_storage_version_source_resolves_current_and_history(tmp_path) -> None:
    storage = LocalSkillStorage(host_path=str(tmp_path / "skills"))
    storage.write_custom_skill(
        "package-repair",
        "SKILL.md",
        _BASE_V2,
    )
    storage.append_history(
        "package-repair",
        {
            "action": "edit",
            "prev_content": _BASE_V1,
            "new_content": _BASE_V2,
        },
    )
    source = SkillStorageVersionSource(storage)

    assert await source.read_current("package-repair") == _BASE_V2
    assert (
        await source.resolve_exact(
            "package-repair",
            _hash(_BASE_V1),
        )
        == _BASE_V1
    )
    assert (
        await source.resolve_exact(
            "package-repair",
            _hash(_BASE_V2),
        )
        == _BASE_V2
    )
