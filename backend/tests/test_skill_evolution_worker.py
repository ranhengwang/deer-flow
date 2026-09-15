from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from deerflow.config.app_config import AppConfig
from deerflow.config.paths import Paths
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.skill_evolution_config import SkillEvolutionConfig
from deerflow.runtime.events.catalog import EVOLUTION_TRACE_EVENT
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.runs.manager import RunManager
from deerflow.runtime.runs.schemas import RunStatus
from deerflow.runtime.runs.worker import RunContext, run_agent
from deerflow.skill_evolution.models import EvolutionTraceSnapshot


def _bridge() -> SimpleNamespace:
    return SimpleNamespace(
        publish=AsyncMock(),
        publish_end=AsyncMock(),
        cleanup=AsyncMock(),
    )


def _app_config(*, enabled: bool) -> AppConfig:
    return AppConfig(
        sandbox=SandboxConfig(use="test"),
        skill_evolution=SkillEvolutionConfig(enabled=enabled),
    )


async def _trace_events(
    store: MemoryRunEventStore,
    *,
    thread_id: str,
    run_id: str,
) -> list[dict]:
    return await store.list_events(
        thread_id,
        run_id,
        event_types=[EVOLUTION_TRACE_EVENT.event_type],
    )


@pytest.mark.anyio
async def test_completed_run_persists_one_redacted_evolution_trace(
    tmp_path,
    monkeypatch,
) -> None:
    paths = Paths(base_dir=tmp_path)
    monkeypatch.setattr(
        "deerflow.runtime.runs.worker.get_paths",
        lambda: paths,
    )
    run_manager = RunManager()
    record = await run_manager.create("thread-1", user_id="user-1")
    record.model_name = "test-model"
    store = MemoryRunEventStore()
    host_workspace = str(
        paths.sandbox_work_dir(
            "thread-1",
            user_id="user-1",
        )
    )

    class DummyAgent:
        async def astream(
            self,
            graph_input,
            config=None,
            stream_mode=None,
            subgraphs=False,
        ):
            await store.put(
                thread_id="thread-1",
                run_id=record.run_id,
                event_type="llm.human.input",
                category="message",
                content={
                    "type": "human",
                    "content": (f"Process {host_workspace}/input.csv with request-secret"),
                    "additional_kwargs": {},
                },
                metadata={"caller": "lead_agent"},
            )
            await store.put(
                thread_id="thread-1",
                run_id=record.run_id,
                event_type="llm.ai.response",
                category="message",
                content={
                    "type": "ai",
                    "content": "Completed.",
                    "additional_kwargs": {},
                    "tool_calls": [],
                },
                metadata={"caller": "lead_agent"},
            )
            yield {"messages": [AIMessage(content="Completed.")]}

    await run_agent(
        _bridge(),
        run_manager,
        record,
        ctx=RunContext(
            checkpointer=None,
            event_store=store,
            app_config=_app_config(enabled=True),
        ),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={
            "context": {
                "secrets": {
                    "API_TOKEN": "request-secret",
                },
            }
        },
    )

    events = await _trace_events(
        store,
        thread_id="thread-1",
        run_id=record.run_id,
    )
    assert len(events) == 1
    snapshot = EvolutionTraceSnapshot.model_validate(events[0]["content"])
    assert snapshot.run_status.value == "success"
    assert snapshot.user_id == "user-1"
    assert snapshot.model_name == "test-model"
    assert snapshot.task_input == ("Process /mnt/user-data/workspace/input.csv with [redacted]")
    assert snapshot.final_answer == "Completed."
    serialized = snapshot.model_dump_json()
    assert "request-secret" not in serialized
    assert host_workspace not in serialized
    assert events[0]["metadata"] == {
        "schema_version": snapshot.schema_version,
        "snapshot_hash": snapshot.snapshot_hash,
    }


@pytest.mark.anyio
async def test_evolution_trace_write_is_idempotent() -> None:
    run_manager = RunManager()
    record = await run_manager.create("thread-1", user_id="user-1")
    store = MemoryRunEventStore()

    class DummyAgent:
        async def astream(
            self,
            graph_input,
            config=None,
            stream_mode=None,
            subgraphs=False,
        ):
            yield {"messages": [AIMessage(content="Completed.")]}

    run_context = RunContext(
        checkpointer=None,
        event_store=store,
        app_config=_app_config(enabled=True),
    )
    await run_agent(
        _bridge(),
        run_manager,
        record,
        ctx=run_context,
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    first = await _trace_events(
        store,
        thread_id="thread-1",
        run_id=record.run_id,
    )
    assert len(first) == 1

    from deerflow.runtime.runs.worker import _persist_evolution_trace_snapshot

    await _persist_evolution_trace_snapshot(
        event_store=store,
        record=record,
        runtime_context={},
    )

    second = await _trace_events(
        store,
        thread_id="thread-1",
        run_id=record.run_id,
    )
    assert len(second) == 1
    assert second[0]["content"]["snapshot_hash"] == (first[0]["content"]["snapshot_hash"])


@pytest.mark.anyio
async def test_disabled_evolution_does_not_persist_trace() -> None:
    run_manager = RunManager()
    record = await run_manager.create("thread-1", user_id="user-1")
    store = MemoryRunEventStore()

    class DummyAgent:
        async def astream(
            self,
            graph_input,
            config=None,
            stream_mode=None,
            subgraphs=False,
        ):
            yield {"messages": [AIMessage(content="Completed.")]}

    await run_agent(
        _bridge(),
        run_manager,
        record,
        ctx=RunContext(
            checkpointer=None,
            event_store=store,
            app_config=_app_config(enabled=False),
        ),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    assert not await _trace_events(
        store,
        thread_id="thread-1",
        run_id=record.run_id,
    )


@pytest.mark.anyio
async def test_evolution_trace_failure_does_not_change_run_outcome() -> None:
    class EvolutionFailingStore(MemoryRunEventStore):
        async def put_if_absent(self, **kwargs):
            if kwargs["event_type"] == EVOLUTION_TRACE_EVENT.event_type:
                raise RuntimeError("evolution storage unavailable")
            return await super().put_if_absent(**kwargs)

    run_manager = RunManager()
    record = await run_manager.create("thread-1", user_id="user-1")
    store = EvolutionFailingStore()

    class DummyAgent:
        async def astream(
            self,
            graph_input,
            config=None,
            stream_mode=None,
            subgraphs=False,
        ):
            yield {"messages": [AIMessage(content="Completed.")]}

    await run_agent(
        _bridge(),
        run_manager,
        record,
        ctx=RunContext(
            checkpointer=None,
            event_store=store,
            app_config=_app_config(enabled=True),
        ),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    assert record.status is RunStatus.success
    assert not await _trace_events(
        store,
        thread_id="thread-1",
        run_id=record.run_id,
    )


@pytest.mark.anyio
async def test_persisted_trace_is_enqueued_after_terminal_status() -> None:
    run_manager = RunManager()
    record = await run_manager.create("thread-1", user_id="user-1")
    store = MemoryRunEventStore()
    enqueue = AsyncMock()

    class DummyAgent:
        async def astream(
            self,
            graph_input,
            config=None,
            stream_mode=None,
            subgraphs=False,
        ):
            yield {"messages": [AIMessage(content="Completed.")]}

    await run_agent(
        _bridge(),
        run_manager,
        record,
        ctx=RunContext(
            checkpointer=None,
            event_store=store,
            app_config=_app_config(enabled=True),
            enqueue_evolution_job=enqueue,
        ),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    enqueue.assert_awaited_once()
    snapshot = enqueue.await_args.args[0]
    assert isinstance(snapshot, EvolutionTraceSnapshot)
    assert snapshot.run_id == record.run_id
    assert snapshot.user_id == "user-1"
    assert record.status is RunStatus.success


@pytest.mark.anyio
async def test_evolution_enqueue_failure_does_not_change_run_outcome() -> None:
    run_manager = RunManager()
    record = await run_manager.create("thread-1", user_id="user-1")
    store = MemoryRunEventStore()
    enqueue = AsyncMock(side_effect=RuntimeError("coordinator unavailable"))

    class DummyAgent:
        async def astream(
            self,
            graph_input,
            config=None,
            stream_mode=None,
            subgraphs=False,
        ):
            yield {"messages": [AIMessage(content="Completed.")]}

    await run_agent(
        _bridge(),
        run_manager,
        record,
        ctx=RunContext(
            checkpointer=None,
            event_store=store,
            app_config=_app_config(enabled=True),
            enqueue_evolution_job=enqueue,
        ),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    assert record.status is RunStatus.success
    assert (
        len(
            await _trace_events(
                store,
                thread_id="thread-1",
                run_id=record.run_id,
            )
        )
        == 1
    )
