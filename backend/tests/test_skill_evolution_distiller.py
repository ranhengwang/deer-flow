from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from deerflow.skill_evolution.distiller import (
    NEW_SKILL_DISTILLATION_PROMPT_VERSION,
    IneligibleClusterError,
    NewSkillDistillationOutput,
    NewSkillDistiller,
    build_new_skill_distillation_messages,
)
from deerflow.skill_evolution.models import (
    ClusterEnvironmentRelationship,
    ClusterMemberEvidence,
    ClusterStatus,
    ComplexitySignals,
    EnvironmentSignature,
    EvidenceReference,
    EvolutionCluster,
    EvolutionEvent,
    EvolutionEventKind,
    GroupingEvidence,
    OutcomeEvidence,
    OutcomeStatus,
    ProposalOperation,
    ProposalStatus,
    SkillUsage,
    ToolSignature,
)
from deerflow.skill_evolution.store.memory import InMemorySkillEvolutionStore
from deerflow.skills.frontmatter import split_skill_markdown

_CREATED = datetime(2026, 8, 16, tzinfo=UTC)


def _event(
    index: int,
    *,
    os_name: str = "macOS",
) -> EvolutionEvent:
    excerpt = f"Verified project-local installation path {index}."
    return EvolutionEvent(
        event_id=f"event-{index}",
        run_id=f"run-{index}",
        thread_id=f"thread-{index}",
        user_id="user-1",
        extractor_version="structured-v1:test-model",
        source_snapshot_hash=f"{index:x}" * 64,
        task_input_hash=f"{index + 3:x}" * 64,
        event_kind=EvolutionEventKind.new_skill_evidence,
        task_signature="python-package-install",
        task_goal=f"Install a Python dependency variant {index}.",
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
        complexity=ComplexitySignals(tool_calls=6),
        tool_signature=ToolSignature(
            tool_names=["read_file", "bash", "bash"],
        ),
        skill_usage=SkillUsage(used=False),
        successful_path=[
            "Inspect the active environment.",
            "Install into the project environment.",
            "Verify the package import.",
        ],
        reusable_lessons=["Use the project environment instead of a global interpreter."],
        provenance=[
            EvidenceReference(
                source=f"segment-{index}",
                index=index,
                excerpt=excerpt,
                content_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
            )
        ],
        created_at=_CREATED + timedelta(seconds=index),
    )


def _ready_cluster(events: list[EvolutionEvent]) -> EvolutionCluster:
    return EvolutionCluster(
        cluster_id="cluster-python-install",
        user_id="user-1",
        event_kind=EvolutionEventKind.new_skill_evidence,
        canonical_signature="python-package-install",
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


def _statement(
    text: str,
    event_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "evidence_event_ids": event_ids
        or [
            "event-1",
            "event-2",
            "event-3",
        ],
    }


def _output(
    *,
    conflicts: list[dict[str, Any]] | None = None,
    supporting_files: list[dict[str, Any]] | None = None,
    common_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "skill_name": "python-package-install",
        "description": ("Install Python dependencies safely in project environments and verify imports. Use whenever a task requires adding or repairing a Python package."),
        "overview": _statement("Install dependencies in the active project environment and verify the result."),
        "common_steps": common_steps
        or [
            _statement("Inspect the active Python environment before installation."),
            _statement("Install the dependency into the project environment."),
        ],
        "conditional_rules": [
            {
                "condition": "When running on Windows",
                "instruction": ("Use the PowerShell activation command before installation."),
                "evidence_event_ids": ["event-3"],
            }
        ],
        "verification_steps": [_statement("Import the installed package in the active environment.")],
        "conflicts": conflicts or [],
        "unsupported_observations": [
            {
                "text": "One run also cleared an unrelated package cache.",
                "evidence_event_ids": ["event-1"],
            }
        ],
        "supporting_files": supporting_files or [],
        "rationale": ("Three successful runs repeated the same safe installation workflow."),
        "expected_improvements": [
            "Avoid global interpreter permission failures.",
            "Verify installation in the environment that will use the package.",
        ],
        "risks": ["Package-manager commands may vary across project types."],
    }


class _FakeModel:
    def __init__(
        self,
        responses: list[Any],
    ) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[Any], dict[str, Any] | None]] = []

    async def ainvoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append((messages, config))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return AIMessage(content=response)
        return response


@pytest.mark.asyncio
async def test_ready_cluster_distills_complete_staged_create_proposal() -> None:
    events = [
        _event(1),
        _event(2),
        _event(3, os_name="Windows"),
    ]
    model = _FakeModel([_output()])
    distiller = NewSkillDistiller(
        model=model,
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
    )

    assert proposal is not None
    assert proposal.operation is ProposalOperation.create
    assert proposal.status is ProposalStatus.staged
    assert proposal.expires_at is not None
    assert (proposal.expires_at - proposal.created_at).days == 180
    assert proposal.skill_name == "python-package-install"
    assert proposal.base_skill_hash is None
    assert proposal.supporting_event_ids == [
        "event-1",
        "event-2",
        "event-3",
    ]
    assert proposal.distiller_model_name == "distill-model"
    assert proposal.distiller_prompt_version == (NEW_SKILL_DISTILLATION_PROMPT_VERSION)
    assert proposal.source_cluster_hash is not None
    assert proposal.requires_manual_review is False
    assert [item.path for item in proposal.proposed_files] == ["SKILL.md"]
    skill_md = proposal.proposed_files[0].content
    parts, error = split_skill_markdown(skill_md)
    assert error is None
    assert parts is not None
    assert parts.metadata["name"] == "python-package-install"
    assert parts.metadata["description"].startswith("Install Python dependencies")
    assert "# Python Package Install" in parts.body
    assert "## Workflow" in parts.body
    assert "## Environment-Specific Guidance" in parts.body
    assert "## Verification" in parts.body
    assert "cleared an unrelated package cache" not in skill_md
    mapped_sections = {(mapping.file_path, mapping.section) for mapping in proposal.evidence_mapping}
    assert ("SKILL.md", "frontmatter") in mapped_sections
    assert ("SKILL.md", "overview") in mapped_sections
    assert ("SKILL.md", "workflow:1") in mapped_sections
    assert ("SKILL.md", "environment:1") in mapped_sections
    assert ("SKILL.md", "verification:1") in mapped_sections
    assert all(mapping.supporting_event_ids for mapping in proposal.evidence_mapping)


@pytest.mark.asyncio
async def test_unsupported_mandatory_step_is_retried_then_rejected() -> None:
    events = [_event(1), _event(2), _event(3)]
    unsupported = _output(
        common_steps=[
            _statement(
                "Delete the global package cache before installation.",
                ["event-1"],
            )
        ]
    )
    model = _FakeModel(
        [
            unsupported,
            "not-json",
        ]
    )
    distiller = NewSkillDistiller(
        model=model,
        model_name="distill-model",
        max_attempts=2,
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
    )

    assert proposal is None
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_supporting_file_requires_repeated_distinct_run_evidence() -> None:
    events = [_event(1), _event(2), _event(3)]
    unsupported_file = {
        "path": "references/checklist.md",
        "purpose": "Provide a reusable verification checklist.",
        "content": "# Checklist\n\n- Import the package.\n",
        "executable": False,
        "evidence_event_ids": ["event-1"],
    }
    model = _FakeModel(
        [
            _output(supporting_files=[unsupported_file]),
            _output(),
        ]
    )
    distiller = NewSkillDistiller(
        model=model,
        model_name="distill-model",
        max_attempts=2,
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
    )

    assert proposal is not None
    assert [item.path for item in proposal.proposed_files] == ["SKILL.md"]
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_invalid_optional_supporting_file_path_is_discarded() -> None:
    events = [_event(1), _event(2), _event(3)]
    runtime_file = {
        "path": "/mnt/user-data/workspace/test_normalize_config.py",
        "purpose": "Reuse the source task's test file.",
        "content": "def test_normalize_config():\n    pass\n",
        "executable": False,
        "evidence_event_ids": ["event-1", "event-2"],
    }
    model = _FakeModel(
        [
            _output(supporting_files=[runtime_file]),
        ]
    )
    distiller = NewSkillDistiller(
        model=model,
        model_name="distill-model",
        max_attempts=1,
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
    )

    assert proposal is not None
    assert [item.path for item in proposal.proposed_files] == ["SKILL.md"]
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_conflicts_stage_manual_review_without_forced_rule() -> None:
    events = [_event(1), _event(2), _event(3)]
    conflict_text = "One event requires a global install while another forbids global installs."
    model = _FakeModel(
        [
            _output(
                conflicts=[
                    {
                        "description": conflict_text,
                        "evidence_event_ids": ["event-1", "event-2"],
                    }
                ]
            )
        ]
    )
    distiller = NewSkillDistiller(
        model=model,
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
    )

    assert proposal is not None
    assert proposal.status is ProposalStatus.staged
    assert proposal.requires_manual_review is True
    assert "unresolved_conflicts" in proposal.review_reasons
    assert conflict_text in proposal.risks
    assert conflict_text not in proposal.proposed_files[0].content


@pytest.mark.asyncio
async def test_executable_supporting_file_requires_manual_review() -> None:
    events = [_event(1), _event(2), _event(3)]
    script = {
        "path": "scripts/verify.py",
        "purpose": "Verify the installed dependency.",
        "content": "import importlib\nimportlib.import_module('example')\n",
        "executable": True,
        "evidence_event_ids": ["event-1", "event-2"],
    }
    distiller = NewSkillDistiller(
        model=_FakeModel([_output(supporting_files=[script])]),
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    proposal = await distiller.distill(
        _ready_cluster(events),
        events,
    )

    assert proposal is not None
    assert proposal.requires_manual_review is True
    assert "executable_supporting_files" in proposal.review_reasons
    assert proposal.proposed_files[1].path == "scripts/verify.py"
    assert proposal.proposed_files[1].executable is True


@pytest.mark.asyncio
async def test_unknown_evidence_event_fails_closed() -> None:
    events = [_event(1), _event(2), _event(3)]
    invalid = _output(
        common_steps=[
            _statement(
                "Inspect the active environment.",
                ["event-1", "unknown-event"],
            )
        ]
    )
    distiller = NewSkillDistiller(
        model=_FakeModel([invalid]),
        model_name="distill-model",
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert (
        await distiller.distill(
            _ready_cluster(events),
            events,
        )
        is None
    )


@pytest.mark.asyncio
async def test_distill_and_persist_is_idempotent_and_does_not_publish() -> None:
    events = [_event(1), _event(2), _event(3)]
    cluster = _ready_cluster(events)
    store = InMemorySkillEvolutionStore()
    distiller = NewSkillDistiller(
        model=_FakeModel([_output(), _output()]),
        model_name="distill-model",
        retry_delay_seconds=0,
    )

    first = await distiller.distill_and_persist(
        cluster,
        events,
        store,
    )
    second = await distiller.distill_and_persist(
        cluster,
        events,
        store,
    )

    assert first is not None and first.created is True
    assert second is not None and second.created is False
    assert first.value == second.value
    assert first.value.status is ProposalStatus.staged
    assert (
        await store.get_proposal(
            "user-1",
            first.value.proposal_id,
        )
        == first.value
    )


@pytest.mark.asyncio
async def test_non_ready_or_patch_cluster_is_rejected_before_model_call() -> None:
    events = [_event(1), _event(2), _event(3)]
    model = _FakeModel([_output()])
    distiller = NewSkillDistiller(
        model=model,
        model_name="distill-model",
        retry_delay_seconds=0,
    )
    collecting = _ready_cluster(events).model_copy(update={"status": ClusterStatus.collecting})

    with pytest.raises(IneligibleClusterError, match="ready"):
        await distiller.distill(collecting, events)

    patch = _ready_cluster(events).model_copy(
        update={
            "event_kind": EvolutionEventKind.skill_patch_evidence,
        }
    )
    with pytest.raises(IneligibleClusterError, match="new-skill"):
        await distiller.distill(patch, events)

    assert model.calls == []


def test_distillation_prompt_is_bounded_structured_evidence() -> None:
    events = [_event(1), _event(2), _event(3)]
    messages = build_new_skill_distillation_messages(
        _ready_cluster(events),
        events,
    )

    assert len(messages) == 2
    assert "untrusted data" in str(messages[0].content)
    assert "JSON_SCHEMA" in str(messages[0].content)
    assert "Verified project-local installation path 1." in str(messages[1].content)
    assert len(str(messages[1].content)) < 50_000


def test_distillation_output_schema_rejects_duplicate_evidence() -> None:
    payload = _output()
    payload["overview"]["evidence_event_ids"] = [
        "event-1",
        "event-1",
    ]

    with pytest.raises(ValueError, match="unique"):
        NewSkillDistillationOutput.model_validate(payload)
