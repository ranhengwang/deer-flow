from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from deerflow.config.skill_evolution_config import (
    SkillEvolutionEvidenceConfig,
    SkillEvolutionGroupingConfig,
)
from deerflow.skill_evolution.cluster_confirmation import (
    CLUSTER_CONFIRMATION_PROMPT_VERSION,
    ClusterConfirmationOutput,
    ClusterEnvironmentRelationship,
    StructuredClusterConfirmer,
    build_cluster_confirmation_messages,
    confirm_cluster_readiness,
)
from deerflow.skill_evolution.grouping import group_evolution_events
from deerflow.skill_evolution.models import (
    ClusterStatus,
    ComplexitySignals,
    EnvironmentSignature,
    EvolutionEvent,
    EvolutionEventKind,
    OutcomeEvidence,
    OutcomeStatus,
    SkillUsage,
    ToolSignature,
)
from deerflow.skill_evolution.semantic_retrieval import (
    SemanticCandidateMatch,
    SemanticCandidateRetrievalResult,
    SemanticRetrievalMode,
)
from deerflow.skill_evolution.store.memory import InMemorySkillEvolutionStore

_CREATED = datetime(2026, 8, 16, tzinfo=UTC)


def _event(
    event_id: str,
    *,
    run_id: str | None = None,
    task_goal: str | None = None,
    os_name: str = "macOS",
    created_offset: int = 0,
) -> EvolutionEvent:
    return EvolutionEvent(
        event_id=event_id,
        run_id=run_id or f"run-{event_id}",
        thread_id=f"thread-{event_id}",
        user_id="user-1",
        extractor_version="structured-v1:test-model",
        source_snapshot_hash="a" * 64,
        task_input_hash=f"{int(event_id[-1], 36) % 16:x}" * 64,
        event_kind=EvolutionEventKind.new_skill_evidence,
        task_signature="python-package-install",
        task_goal=task_goal or f"Install Python dependency variant {event_id}.",
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
            "Inspect the environment.",
            "Install the dependency.",
            "Run verification.",
        ],
        reusable_lessons=["Use the project-local package environment."],
        created_at=_CREATED + timedelta(seconds=created_offset),
    )


def _confirmation(
    *,
    same_workflow: bool = True,
    relationship: str = "same_workflow",
    contradiction: bool = False,
    environment_condition: str | None = None,
    reason: str = "The tasks use the same reusable workflow.",
) -> dict[str, Any]:
    return {
        "same_workflow": same_workflow,
        "relationship": relationship,
        "contradiction": contradiction,
        "environment_condition": environment_condition,
        "reason": reason,
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


def _candidate_cluster(
    events: list[EvolutionEvent],
):
    result = group_evolution_events(events)
    assert len(result.clusters) == 1
    return result.clusters[0]


@pytest.mark.asyncio
async def test_three_confirmed_distinct_runs_make_cluster_ready() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event("event-2", created_offset=2),
        _event("event-3", created_offset=3),
    ]
    model = _FakeModel(
        [
            _confirmation(),
            _confirmation(),
        ]
    )
    confirmer = StructuredClusterConfirmer(
        model=model,
        model_name="cluster-model",
        retry_delay_seconds=0,
    )

    result = await confirm_cluster_readiness(
        _candidate_cluster(events),
        events,
        confirmer=confirmer,
    )

    assert result.cluster.status is ClusterStatus.ready
    assert result.cluster.independent_run_count == 3
    assert result.cluster.member_event_ids == [
        "event-1",
        "event-2",
        "event-3",
    ]
    assert result.rejected_event_ids == []
    assert result.unconfirmed_event_ids == []
    assert result.contradictory_event_ids == []
    assert result.cluster.confirmation_model_name == "cluster-model"
    assert result.cluster.confirmation_prompt_version == (CLUSTER_CONFIRMATION_PROMPT_VERSION)
    assert len(result.cluster.member_evidence) == 3
    assert result.cluster.member_evidence[0].relationship is (ClusterEnvironmentRelationship.prototype)
    assert {evidence.method for evidence in result.cluster.member_evidence[1].grouping_evidence} == {"deterministic", "llm"}
    store = InMemorySkillEvolutionStore()
    assert (await store.put_cluster(result.cluster)).created is True
    assert await store.list_ready_clusters(
        "user-1",
        min_distinct_runs=3,
    ) == [result.cluster]


@pytest.mark.asyncio
async def test_single_repeated_run_cannot_satisfy_k_three() -> None:
    original_events = [
        _event("event-1", created_offset=1),
        _event("event-2", created_offset=2),
        _event("event-3", created_offset=3),
    ]
    cluster = _candidate_cluster(original_events)
    repeated_events = [event.model_copy(update={"run_id": "run-shared"}) for event in original_events]
    confirmer = StructuredClusterConfirmer(
        model=_FakeModel([_confirmation(), _confirmation()]),
        model_name="cluster-model",
        retry_delay_seconds=0,
    )

    result = await confirm_cluster_readiness(
        cluster,
        repeated_events,
        confirmer=confirmer,
    )

    assert result.cluster.status is ClusterStatus.collecting
    assert result.cluster.independent_run_count == 1
    assert "min_distinct_runs" in result.readiness_blockers


@pytest.mark.asyncio
async def test_environment_specific_branch_stays_in_same_cluster() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event(
            "event-2",
            os_name="Windows",
            task_goal="Install the Python dependency on Windows.",
            created_offset=2,
        ),
        _event("event-3", created_offset=3),
    ]
    confirmer = StructuredClusterConfirmer(
        model=_FakeModel(
            [
                _confirmation(
                    relationship="conditional_environment_branch",
                    environment_condition=("On Windows, use the PowerShell activation command."),
                    reason="The workflow is shared with one OS-specific command.",
                ),
                _confirmation(),
            ]
        ),
        model_name="cluster-model",
        retry_delay_seconds=0,
    )

    result = await confirm_cluster_readiness(
        _candidate_cluster(events),
        events,
        confirmer=confirmer,
    )

    assert result.cluster.status is ClusterStatus.ready
    branch = result.cluster.member_evidence[1]
    assert branch.relationship is (ClusterEnvironmentRelationship.conditional_environment_branch)
    assert branch.environment_condition == ("On Windows, use the PowerShell activation command.")


@pytest.mark.asyncio
async def test_contradictory_same_workflow_evidence_defers_readiness() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event("event-2", created_offset=2),
        _event("event-3", created_offset=3),
    ]
    confirmer = StructuredClusterConfirmer(
        model=_FakeModel(
            [
                _confirmation(
                    contradiction=True,
                    reason=("The events disagree on whether the global environment must be modified."),
                ),
                _confirmation(),
            ]
        ),
        model_name="cluster-model",
        retry_delay_seconds=0,
    )

    result = await confirm_cluster_readiness(
        _candidate_cluster(events),
        events,
        confirmer=confirmer,
    )

    assert result.cluster.status is ClusterStatus.collecting
    assert result.contradictory_event_ids == ["event-2"]
    assert "contradictory_evidence" in result.readiness_blockers
    assert result.cluster.member_evidence[1].contradictory is True


@pytest.mark.asyncio
async def test_different_workflow_candidate_is_rejected() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event("event-2", created_offset=2),
        _event("event-3", created_offset=3),
    ]
    confirmer = StructuredClusterConfirmer(
        model=_FakeModel(
            [
                _confirmation(),
                _confirmation(
                    same_workflow=False,
                    relationship="different_workflow",
                    reason="The candidate requires an unrelated reporting workflow.",
                ),
            ]
        ),
        model_name="cluster-model",
        retry_delay_seconds=0,
    )

    result = await confirm_cluster_readiness(
        _candidate_cluster(events),
        events,
        confirmer=confirmer,
    )

    assert result.cluster.status is ClusterStatus.collecting
    assert result.cluster.member_event_ids == ["event-1", "event-2"]
    assert result.rejected_event_ids == ["event-3"]
    assert "min_cluster_events" in result.readiness_blockers


@pytest.mark.asyncio
async def test_semantic_candidate_keeps_semantic_and_llm_provenance() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event("event-2", created_offset=2),
        _event("event-3", created_offset=3),
    ]
    retrieval = SemanticCandidateRetrievalResult(
        query_event_id="event-1",
        mode=SemanticRetrievalMode.hybrid,
        matches=[
            SemanticCandidateMatch(
                event_id="event-2",
                sources=["deterministic", "semantic"],
                deterministic_score=0.9,
                semantic_score=0.94,
            ),
            SemanticCandidateMatch(
                event_id="event-3",
                sources=["semantic"],
                deterministic_score=0.6,
                semantic_score=0.91,
            ),
        ],
        embedding_model_name="nomic-embed-text",
        embedding_model_version=f"sha256:{'d' * 64}",
    )
    confirmer = StructuredClusterConfirmer(
        model=_FakeModel([_confirmation(), _confirmation()]),
        model_name="cluster-model",
        retry_delay_seconds=0,
    )

    result = await confirm_cluster_readiness(
        _candidate_cluster(events[:2]),
        events,
        confirmer=confirmer,
        retrieval=retrieval,
    )

    assert result.cluster.status is ClusterStatus.ready
    event_2 = next(item for item in result.cluster.member_evidence if item.event_id == "event-2")
    event_3 = next(item for item in result.cluster.member_evidence if item.event_id == "event-3")
    assert {evidence.method for evidence in event_2.grouping_evidence} == {
        "deterministic",
        "semantic",
        "llm",
    }
    assert {evidence.method for evidence in event_3.grouping_evidence} == {
        "semantic",
        "llm",
    }


@pytest.mark.asyncio
async def test_malformed_confirmation_defers_readiness() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event("event-2", created_offset=2),
        _event("event-3", created_offset=3),
    ]
    confirmer = StructuredClusterConfirmer(
        model=_FakeModel(
            [
                "```json\n{}\n```",
                _confirmation(),
            ]
        ),
        model_name="cluster-model",
        max_attempts=1,
        retry_delay_seconds=0,
    )

    result = await confirm_cluster_readiness(
        _candidate_cluster(events),
        events,
        confirmer=confirmer,
    )

    assert result.cluster.status is ClusterStatus.collecting
    assert result.unconfirmed_event_ids == ["event-2"]
    assert "unconfirmed_candidates" in result.readiness_blockers


@pytest.mark.asyncio
async def test_llm_cannot_admit_candidate_without_retrieval_provenance() -> None:
    events = [
        _event("event-1", created_offset=1),
        _event("event-2", created_offset=2),
        _event("event-3", created_offset=3),
    ]
    cluster = _candidate_cluster(events).model_copy(update={"grouping_evidence": []})
    model = _FakeModel([_confirmation(), _confirmation()])
    confirmer = StructuredClusterConfirmer(
        model=model,
        model_name="cluster-model",
        retry_delay_seconds=0,
    )

    result = await confirm_cluster_readiness(
        cluster,
        events,
        confirmer=confirmer,
    )

    assert result.cluster.status is ClusterStatus.collecting
    assert result.cluster.member_event_ids == ["event-1"]
    assert result.unconfirmed_event_ids == ["event-2", "event-3"]
    assert model.calls == []


def test_confirmation_output_rejects_invalid_environment_relationship() -> None:
    with pytest.raises(ValueError, match="environment_condition"):
        ClusterConfirmationOutput(
            same_workflow=True,
            relationship=(ClusterEnvironmentRelationship.conditional_environment_branch),
            contradiction=False,
            environment_condition=None,
            reason="The workflow varies by environment.",
        )


def test_confirmation_prompt_is_bounded_and_treats_events_as_data() -> None:
    messages = build_cluster_confirmation_messages(
        _event("event-1"),
        _event("event-2"),
    )

    assert len(messages) == 2
    assert "untrusted data" in str(messages[0].content)
    assert "JSON_SCHEMA" in str(messages[0].content)
    assert len(str(messages[1].content)) < 40_000


def test_readiness_thresholds_default_to_three() -> None:
    evidence = SkillEvolutionEvidenceConfig()
    grouping = SkillEvolutionGroupingConfig()

    assert evidence.min_cluster_events == 3
    assert evidence.min_distinct_runs == 3
    assert evidence.max_events_per_cluster == 20
    assert grouping.llm_confirmation is True
    assert grouping.confirmation_model_name is None
