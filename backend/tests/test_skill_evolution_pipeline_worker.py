from __future__ import annotations

from datetime import UTC, datetime

import pytest

from deerflow.config.app_config import AppConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.skill_evolution_config import (
    SkillEvolutionConfig,
    SkillEvolutionPublicationConfig,
)
from deerflow.runtime.events.catalog import EVOLUTION_TRACE_EVENT
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.skill_evolution.cluster_confirmation import (
    ClusterConfirmationOutput,
    ClusterEnvironmentRelationship,
)
from deerflow.skill_evolution.coordinator import build_evolution_job
from deerflow.skill_evolution.models import (
    CreditKind,
    EvolutionEvent,
    ProposalOperation,
    ProposalStatus,
    ProposedSkillFile,
    SkillProposal,
    SkillUsage,
)
from deerflow.skill_evolution.observability import (
    EvolutionLifecycleKind,
    EvolutionObservability,
)
from deerflow.skill_evolution.store.memory import (
    InMemorySkillEvolutionStore,
)
from deerflow.skill_evolution.worker import (
    EvolutionPipelineProcessor,
    EvolutionSnapshotUnavailable,
)


def _config(
    *,
    enabled: bool = True,
    publication_mode: str = "manual",
) -> AppConfig:
    return AppConfig(
        sandbox=SandboxConfig(use="test"),
        skill_evolution=SkillEvolutionConfig(
            enabled=enabled,
            publication=SkillEvolutionPublicationConfig(
                mode=publication_mode,
            ),
        ),
    )


def _snapshot_payload(
    *,
    run_id: str = "run-1",
    thread_id: str = "thread-1",
    snapshot_hash: str = "a" * 64,
    created_at: datetime | None = None,
    runtime: str = "python3.12",
    os_name: str = "macos",
) -> dict:
    return {
        "snapshot_hash": snapshot_hash,
        "run_id": run_id,
        "thread_id": thread_id,
        "user_id": "user-1",
        "run_status": "success",
        "task_input": (f"Create a skill for this verified workflow {run_id}."),
        "final_answer": "Done.",
        "tool_events": [
            {
                "sequence": 0,
                "tool_call_id": "call-1",
                "tool_name": "bash",
                "arguments": '{"command":"pytest -q"}',
                "result": "Process exited with code 0",
                "status": "success",
            }
        ],
        "environment": {
            "os": os_name,
            "shell": "zsh",
            "runtime": runtime,
        },
        "source_event_count": 1,
        "included_event_count": 1,
        "truncated": False,
        "created_at": created_at or datetime(2026, 8, 17, tzinfo=UTC),
    }


def _job(
    *,
    run_id: str = "run-1",
    thread_id: str = "thread-1",
    snapshot_hash: str = "a" * 64,
):
    return build_evolution_job(
        user_id="user-1",
        thread_id=thread_id,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        max_attempts=3,
        now=datetime(2026, 8, 17, tzinfo=UTC),
    )


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    async def extract_and_persist(self, extraction, store):
        self.calls += 1
        event = EvolutionEvent(
            event_id=f"event-{extraction.run_id}",
            run_id=extraction.run_id,
            thread_id=extraction.thread_id,
            user_id=extraction.user_id,
            extractor_version="fake-v1",
            source_snapshot_hash=extraction.source_snapshot_hash,
            task_input_hash=extraction.task_input_hash,
            event_kind=extraction.event_kind,
            task_signature="verified-workflow",
            task_goal="Run focused tests and retain the workflow.",
            environment=extraction.environment,
            outcome=extraction.outcome,
            complexity=extraction.complexity,
            tool_signature=extraction.tool_signature,
            skill_usage=SkillUsage(used=False),
            successful_path=["Run the focused test command."],
            reusable_lessons=["Require deterministic verification."],
            created_at=extraction.created_at,
        )
        return await store.upsert_event(event)


class _FakeConfirmer:
    model_name = "fake-confirmer"
    prompt_version = "fake-v1"

    async def confirm(self, prototype, candidate):
        raise AssertionError("a single event has no confirmation candidate")


class _AcceptingConfirmer:
    model_name = "fake-confirmer"
    prompt_version = "fake-v1"

    async def confirm(self, prototype, candidate):
        return ClusterConfirmationOutput(
            same_workflow=True,
            relationship=(ClusterEnvironmentRelationship.same_workflow),
            contradiction=False,
            reason="Same verified workflow.",
        )


class _FakeDistiller:
    async def distill_and_persist(self, cluster, events, store):
        proposal = SkillProposal(
            proposal_id=f"proposal-{cluster.cluster_id}",
            cluster_id=cluster.cluster_id,
            user_id=cluster.user_id,
            operation=ProposalOperation.create,
            skill_name="verified-workflow",
            proposed_files=[
                ProposedSkillFile(
                    path="SKILL.md",
                    content=("---\nname: verified-workflow\ndescription: Run a verified workflow.\n---\n"),
                    executable=False,
                )
            ],
            supporting_event_ids=[event.event_id for event in events],
            rationale="Three independent verified runs.",
            expected_improvements=["Reuse the verified workflow."],
            status=ProposalStatus.staged,
            created_at=cluster.updated_at,
        )
        return await store.put_proposal(proposal)


class _DirectPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def __call__(
        self,
        *,
        user_id: str,
        proposal_id: str,
    ) -> None:
        self.calls.append((user_id, proposal_id))


@pytest.mark.anyio
async def test_pipeline_extracts_once_and_reuses_persisted_event() -> None:
    run_events = MemoryRunEventStore()
    evolution_store = InMemorySkillEvolutionStore()
    extractor = _FakeExtractor()
    observer = EvolutionObservability()
    await run_events.put(
        thread_id="thread-1",
        run_id="run-1",
        event_type=EVOLUTION_TRACE_EVENT.event_type,
        category=EVOLUTION_TRACE_EVENT.category,
        content=_snapshot_payload(),
        metadata={},
    )
    processor = EvolutionPipelineProcessor(
        event_store=run_events,
        evolution_store=evolution_store,
        app_config_provider=_config,
        extractor_factory=lambda _config: extractor,
        confirmer_factory=lambda _config: _FakeConfirmer(),
        observability=observer,
    )

    await processor(_job())
    await processor(_job())

    events = await evolution_store.list_events("user-1")
    assert len(events) == 1
    assert events[0].run_id == "run-1"
    assert extractor.calls == 1
    credits = await evolution_store.list_credits(
        "user-1",
        kind=CreditKind.selection,
        limit=10,
    )
    assert len(credits) == 1
    assert credits[0].no_skill_selected is True
    assert credits[0].credit_value == 1.0
    assert [item.kind for item in observer.recent_events()] == [
        EvolutionLifecycleKind.admitted,
        EvolutionLifecycleKind.extracted,
        EvolutionLifecycleKind.clustered,
        EvolutionLifecycleKind.rejected,
    ]
    metrics = observer.snapshot()
    assert metrics.cluster_purity.count == 1
    assert metrics.cluster_purity.last == 1.0


@pytest.mark.anyio
async def test_pipeline_rejects_snapshot_identity_drift() -> None:
    run_events = MemoryRunEventStore()
    await run_events.put(
        thread_id="thread-1",
        run_id="run-1",
        event_type=EVOLUTION_TRACE_EVENT.event_type,
        category=EVOLUTION_TRACE_EVENT.category,
        content={
            **_snapshot_payload(),
            "snapshot_hash": "b" * 64,
        },
        metadata={},
    )
    processor = EvolutionPipelineProcessor(
        event_store=run_events,
        evolution_store=InMemorySkillEvolutionStore(),
        app_config_provider=_config,
    )

    with pytest.raises(
        EvolutionSnapshotUnavailable,
        match="identity",
    ):
        await processor(_job())


@pytest.mark.anyio
async def test_disabled_pipeline_does_not_extract() -> None:
    run_events = MemoryRunEventStore()
    extractor = _FakeExtractor()
    await run_events.put(
        thread_id="thread-1",
        run_id="run-1",
        event_type=EVOLUTION_TRACE_EVENT.event_type,
        category=EVOLUTION_TRACE_EVENT.category,
        content=_snapshot_payload(),
        metadata={},
    )
    processor = EvolutionPipelineProcessor(
        event_store=run_events,
        evolution_store=InMemorySkillEvolutionStore(),
        app_config_provider=lambda: _config(enabled=False),
        extractor_factory=lambda _config: extractor,
    )

    await processor(_job())

    assert extractor.calls == 0


@pytest.mark.anyio
async def test_three_independent_runs_emit_ready_and_distilled() -> None:
    run_events = MemoryRunEventStore()
    evolution_store = InMemorySkillEvolutionStore()
    observer = EvolutionObservability()
    processor = EvolutionPipelineProcessor(
        event_store=run_events,
        evolution_store=evolution_store,
        app_config_provider=_config,
        extractor_factory=lambda _config: _FakeExtractor(),
        confirmer_factory=lambda _config: _AcceptingConfirmer(),
        new_skill_distiller_factory=lambda _config: _FakeDistiller(),
        observability=observer,
    )

    for index in range(1, 4):
        run_id = f"run-{index}"
        thread_id = f"thread-{index}"
        snapshot_hash = f"{index}" * 64
        created_at = datetime(
            2026,
            8,
            17,
            index,
            tzinfo=UTC,
        )
        await run_events.put(
            thread_id=thread_id,
            run_id=run_id,
            event_type=EVOLUTION_TRACE_EVENT.event_type,
            category=EVOLUTION_TRACE_EVENT.category,
            content=_snapshot_payload(
                run_id=run_id,
                thread_id=thread_id,
                snapshot_hash=snapshot_hash,
                created_at=created_at,
                runtime=f"python3.{10 + index}",
                os_name=("macos", "linux", "windows")[index - 1],
            ),
            metadata={},
        )
        await processor(
            _job(
                run_id=run_id,
                thread_id=thread_id,
                snapshot_hash=snapshot_hash,
            )
        )

    kinds = [event.kind for event in observer.recent_events()]
    assert EvolutionLifecycleKind.ready in kinds
    assert kinds[-1] is EvolutionLifecycleKind.distilled
    ready = next(event for event in observer.recent_events() if event.kind is EvolutionLifecycleKind.ready)
    assert ready.event_count == 3
    assert ready.distinct_run_count == 3
    proposals = [
        await evolution_store.get_proposal_by_cluster(
            "user-1",
            ready.cluster_id,
        )
    ]
    assert proposals[0] is not None
    assert proposals[0].status is ProposalStatus.staged


@pytest.mark.anyio
async def test_direct_mode_publishes_distilled_proposal_without_evaluation() -> None:
    run_events = MemoryRunEventStore()
    evolution_store = InMemorySkillEvolutionStore()
    publisher = _DirectPublisher()
    processor = EvolutionPipelineProcessor(
        event_store=run_events,
        evolution_store=evolution_store,
        app_config_provider=lambda: _config(
            publication_mode="direct",
        ),
        extractor_factory=lambda _config: _FakeExtractor(),
        confirmer_factory=lambda _config: _AcceptingConfirmer(),
        new_skill_distiller_factory=lambda _config: _FakeDistiller(),
        direct_publisher=publisher,
    )

    for index in range(1, 4):
        run_id = f"run-{index}"
        thread_id = f"thread-{index}"
        snapshot_hash = f"{index}" * 64
        await run_events.put(
            thread_id=thread_id,
            run_id=run_id,
            event_type=EVOLUTION_TRACE_EVENT.event_type,
            category=EVOLUTION_TRACE_EVENT.category,
            content=_snapshot_payload(
                run_id=run_id,
                thread_id=thread_id,
                snapshot_hash=snapshot_hash,
                created_at=datetime(
                    2026,
                    8,
                    17,
                    index,
                    tzinfo=UTC,
                ),
                runtime=f"python3.{10 + index}",
                os_name=("macos", "linux", "windows")[index - 1],
            ),
            metadata={},
        )
        await processor(
            _job(
                run_id=run_id,
                thread_id=thread_id,
                snapshot_hash=snapshot_hash,
            )
        )

    proposals = await evolution_store.list_proposal_page(
        "user-1",
        limit=10,
        offset=0,
    )
    assert len(proposals) == 1
    proposal = proposals[0]
    assert publisher.calls == [
        (
            proposal.user_id,
            proposal.proposal_id,
        )
    ]


class _FailingExtractor:
    async def extract_and_persist(self, extraction, store):
        return None


class _FailingCreditRecorder:
    async def record_verified_run(self, snapshot, outcome):
        raise RuntimeError("sensitive credit failure")


@pytest.mark.anyio
async def test_credit_failure_does_not_block_extraction() -> None:
    run_events = MemoryRunEventStore()
    evolution_store = InMemorySkillEvolutionStore()
    extractor = _FakeExtractor()
    await run_events.put(
        thread_id="thread-1",
        run_id="run-1",
        event_type=EVOLUTION_TRACE_EVENT.event_type,
        category=EVOLUTION_TRACE_EVENT.category,
        content=_snapshot_payload(),
        metadata={},
    )
    processor = EvolutionPipelineProcessor(
        event_store=run_events,
        evolution_store=evolution_store,
        app_config_provider=_config,
        extractor_factory=lambda _config: extractor,
        confirmer_factory=lambda _config: _FakeConfirmer(),
        credit_recorder=_FailingCreditRecorder(),
    )

    await processor(_job())

    assert extractor.calls == 1
    assert len(await evolution_store.list_events("user-1")) == 1


@pytest.mark.anyio
async def test_extraction_failure_is_observed_without_raw_content() -> None:
    run_events = MemoryRunEventStore()
    observer = EvolutionObservability()
    await run_events.put(
        thread_id="thread-1",
        run_id="run-1",
        event_type=EVOLUTION_TRACE_EVENT.event_type,
        category=EVOLUTION_TRACE_EVENT.category,
        content=_snapshot_payload(),
        metadata={},
    )
    processor = EvolutionPipelineProcessor(
        event_store=run_events,
        evolution_store=InMemorySkillEvolutionStore(),
        app_config_provider=_config,
        extractor_factory=lambda _config: _FailingExtractor(),
        observability=observer,
    )

    await processor(_job())

    metrics = observer.snapshot()
    assert metrics.extraction_failures == 1
    rejected = observer.recent_events()[-1]
    assert rejected.kind is EvolutionLifecycleKind.rejected
    assert rejected.stage == "structured_extraction"
    assert rejected.reason_codes == ["structured_extraction_failed"]
