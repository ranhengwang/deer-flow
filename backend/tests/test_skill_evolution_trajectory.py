from __future__ import annotations

from datetime import UTC, datetime, timedelta

from deerflow.skill_evolution.models import TraceRunStatus
from deerflow.skill_evolution.trajectory import (
    RedactionSpec,
    build_evolution_trace_snapshot,
)

_CREATED = datetime(2026, 8, 16, tzinfo=UTC)
_HOST_WORKSPACE = "/Users/alice/deer-flow/workspace"
_VIRTUAL_WORKSPACE = "/mnt/user-data/workspace"


def _event(
    seq: int,
    event_type: str,
    content,
    *,
    category: str = "message",
    metadata: dict | None = None,
) -> dict:
    return {
        "thread_id": "thread-1",
        "run_id": "run-1",
        "seq": seq,
        "event_type": event_type,
        "category": category,
        "content": content,
        "metadata": metadata or {},
        "created_at": (_CREATED + timedelta(seconds=seq)).isoformat(),
    }


def _source_events() -> list[dict]:
    return [
        _event(
            1,
            "llm.human.input",
            {
                "type": "human",
                "content": (f"Process {_HOST_WORKSPACE}/input.csv with supersecret"),
                "additional_kwargs": {},
            },
            metadata={"caller": "lead_agent"},
        ),
        _event(
            2,
            "llm.ai.response",
            {
                "type": "ai",
                "content": "",
                "additional_kwargs": {},
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "bash",
                        "args": {
                            "command": (f"python {_HOST_WORKSPACE}/run.py --token supersecret"),
                            "api_key": "unlisted-secret",
                        },
                    }
                ],
            },
            metadata={"caller": "lead_agent"},
        ),
        _event(
            3,
            "llm.tool.result",
            {
                "type": "tool",
                "name": "bash",
                "tool_call_id": "call-1",
                "status": "success",
                "content": (f"Wrote {_HOST_WORKSPACE}/output.csv using supersecret"),
                "additional_kwargs": {
                    "deerflow_tool_meta": {
                        "status": "success",
                        "error_type": None,
                    }
                },
            },
        ),
        _event(
            4,
            "llm.ai.response",
            {
                "type": "ai",
                "content": "",
                "additional_kwargs": {},
                "tool_calls": [
                    {
                        "id": "call-2",
                        "name": "read_file",
                        "args": {"path": ("/mnt/skills/public/data-analysis/SKILL.md")},
                    }
                ],
            },
            metadata={"caller": "lead_agent"},
        ),
        _event(
            5,
            "llm.tool.result",
            {
                "type": "tool",
                "name": "read_file",
                "tool_call_id": "call-2",
                "status": "success",
                "content": ("---\nname: data-analysis\ndescription: Analyze tabular data.\n---\nUse Python to validate the result."),
                "additional_kwargs": {
                    "skill_context_entry": {
                        "path": ("/mnt/skills/public/data-analysis/SKILL.md"),
                        "description": "Analyze tabular data.",
                    }
                },
            },
        ),
        _event(
            6,
            "middleware:skill_activation",
            {
                "name": "SkillActivationMiddleware",
                "hook": "awrap_model_call",
                "action": "activate",
                "changes": {
                    "skill_name": "deep-research",
                    "category": "public",
                    "path": "/mnt/skills/public/deep-research/SKILL.md",
                    "content_hash": "c" * 64,
                },
            },
            category="middleware",
        ),
        _event(
            7,
            "workspace_changes",
            "Workspace files changed",
            category="workspace",
            metadata={
                "workspace_changes": {
                    "files": [
                        {
                            "path": "output.csv",
                            "root": "outputs",
                            "status": "created",
                        }
                    ]
                }
            },
        ),
        _event(
            8,
            "run.delivery",
            {
                "presented": 1,
                "paths": ["/mnt/user-data/outputs/output.csv"],
                "by_tool": {"present_files": ["/mnt/user-data/outputs/output.csv"]},
            },
            category="outputs",
        ),
        _event(
            9,
            "llm.ai.response",
            {
                "type": "ai",
                "content": "Completed the data transformation.",
                "additional_kwargs": {},
                "tool_calls": [],
            },
            metadata={"caller": "lead_agent"},
        ),
    ]


def _build(events: list[dict], *, max_events: int = 256):
    return build_evolution_trace_snapshot(
        events,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        model_name="qwen3-local",
        run_status=TraceRunStatus.success,
        stop_reason=None,
        environment={
            "os": "macos",
            "shell": "zsh",
            "runtime": "python3.12",
        },
        redaction=RedactionSpec(
            secret_values=("supersecret",),
            path_mappings={
                _HOST_WORKSPACE: _VIRTUAL_WORKSPACE,
            },
        ),
        max_events=max_events,
    )


def test_snapshot_pairs_tools_captures_skills_and_artifacts() -> None:
    snapshot = _build(_source_events())

    assert snapshot.task_input == ("Process /mnt/user-data/workspace/input.csv with [redacted]")
    assert snapshot.final_answer == "Completed the data transformation."
    assert [event.tool_name for event in snapshot.tool_events] == [
        "bash",
        "read_file",
    ]
    assert snapshot.tool_events[0].tool_call_id == "call-1"
    assert snapshot.tool_events[0].status == "success"
    assert "/mnt/user-data/workspace/run.py" in (snapshot.tool_events[0].arguments)
    assert "[redacted]" in snapshot.tool_events[0].arguments
    assert "[redacted]" in snapshot.tool_events[0].result

    skill_by_name = {skill.skill_name: skill for skill in snapshot.skill_events}
    assert skill_by_name["deep-research"].activation_source == "slash"
    assert skill_by_name["deep-research"].content_hash == "c" * 64
    assert skill_by_name["data-analysis"].activation_source == "read"
    assert len(skill_by_name["data-analysis"].content_hash) == 64
    assert "Analyze tabular data" not in snapshot.model_dump_json()
    assert "skill content omitted" in snapshot.tool_events[1].result
    assert snapshot.artifacts == [
        "/mnt/user-data/outputs/output.csv",
    ]


def test_snapshot_contains_no_secret_or_host_path() -> None:
    events = _source_events()
    events[0]["content"]["content"] += " and codex7y9value"
    snapshot = build_evolution_trace_snapshot(
        events,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        model_name="qwen3-local",
        run_status=TraceRunStatus.success,
        stop_reason=None,
        environment={
            "os": "macos",
            "shell": "zsh",
            "runtime": "python3.12",
        },
        redaction=RedactionSpec(
            secret_values=("supersecret", "x7y9"),
            path_mappings={
                _HOST_WORKSPACE: _VIRTUAL_WORKSPACE,
            },
        ),
    )
    serialized = snapshot.model_dump_json()

    assert "supersecret" not in serialized
    assert "x7y9" not in serialized
    assert "unlisted-secret" not in serialized
    assert _HOST_WORKSPACE not in serialized
    assert "/mnt/user-data/workspace" in serialized


def test_snapshot_excludes_hidden_human_context_and_tracks_corrections() -> None:
    events = _source_events()
    events.insert(
        1,
        _event(
            10,
            "llm.human.input",
            {
                "type": "human",
                "content": "hidden framework context",
                "additional_kwargs": {"hide_from_ui": True},
            },
            metadata={"caller": "lead_agent"},
        ),
    )
    events.insert(
        2,
        _event(
            11,
            "llm.human.input",
            {
                "type": "human",
                "content": "Use the project environment instead.",
                "additional_kwargs": {},
            },
            metadata={"caller": "lead_agent"},
        ),
    )

    snapshot = _build(events)

    assert "hidden framework context" not in snapshot.model_dump_json()
    assert snapshot.user_corrections == ["Use the project environment instead."]


def test_snapshot_is_bounded_and_digest_is_deterministic() -> None:
    events = [
        _event(
            0,
            "run.start",
            {"chain": "lead_agent"},
            category="trace",
        ),
        _event(
            1,
            "llm.human.input",
            {
                "type": "human",
                "content": "Start task",
                "additional_kwargs": {},
            },
            metadata={"caller": "lead_agent"},
        ),
    ]
    for seq in range(2, 20):
        events.append(
            _event(
                seq,
                "llm.ai.response",
                {
                    "type": "ai",
                    "content": f"intermediate-{seq}",
                    "additional_kwargs": {},
                    "tool_calls": [],
                },
                metadata={"caller": "lead_agent"},
            )
        )
    events.append(
        _event(
            20,
            "llm.ai.response",
            {
                "type": "ai",
                "content": "Final answer",
                "additional_kwargs": {},
                "tool_calls": [],
            },
            metadata={"caller": "lead_agent"},
        )
    )

    first = _build(events, max_events=5)
    second = _build(list(reversed(events)), max_events=5)

    assert first.source_event_count == 21
    assert first.included_event_count == 5
    assert first.truncated is True
    assert first.task_input == "Start task"
    assert first.final_answer == "Final answer"
    assert first.snapshot_hash == second.snapshot_hash
    assert first == second
