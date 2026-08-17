"""Build bounded, redacted skill-evolution snapshots from run events."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from deerflow.skill_evolution.models import (
    EVOLUTION_TRACE_SCHEMA_VERSION,
    EnvironmentSignature,
    EvolutionTraceSnapshot,
    TraceRunStatus,
    TraceSkillEvent,
    TraceToolEvent,
)

_REDACTED = "[redacted]"
_SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|secret|token|password|passwd|credential|authorization|auth)(?:$|[_-])",
    re.IGNORECASE,
)
_SKILL_CONTEXT_ENTRY_KEY = "skill_context_entry"
_SKILL_ACTIVATION_EVENT = "middleware:skill_activation"
_EVOLUTION_TRACE_EVENT = "skill_evolution.trace"
_MAX_TASK_INPUT_CHARS = 8_000
_MAX_FINAL_ANSWER_CHARS = 12_000
_MAX_CORRECTION_CHARS = 4_000
_MAX_ARGUMENT_CHARS = 16_000
_MAX_RESULT_CHARS = 32_000
_MIN_MAX_EVENTS = 2


@dataclass(frozen=True, slots=True)
class RedactionSpec:
    """Sensitive values and host-to-virtual path mappings for one run."""

    secret_values: tuple[str, ...] = ()
    path_mappings: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _MutableToolEvent:
    sequence: int
    tool_call_id: str
    tool_name: str
    arguments: str
    result: str = ""
    status: str = "unknown"
    error_type: str | None = None
    result_truncated: bool = False
    raw_arguments: Mapping[str, Any] | None = None
    raw_result: str = ""
    result_message: Mapping[str, Any] | None = None


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    if value is None:
        return ""
    return _stable_json(value)


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    marker = f"\n... [truncated {len(value) - limit} chars]"
    kept = max(0, limit - len(marker))
    return f"{value[:kept]}{marker}"[:limit], True


def _replace_secret(value: str, secret: str) -> str:
    if not secret:
        return value
    return value.replace(secret, _REDACTED)


def _redact_text(value: str, spec: RedactionSpec) -> str:
    result = value
    for host_path, virtual_path in sorted(
        spec.path_mappings.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if host_path:
            result = result.replace(host_path, virtual_path)
    for secret in sorted(
        {item for item in spec.secret_values if item},
        key=len,
        reverse=True,
    ):
        result = _replace_secret(result, secret)
    return result


def _redact_value(value: Any, spec: RedactionSpec) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            redacted[key] = _REDACTED if _SECRET_KEY_RE.search(key) else _redact_value(item, spec)
        return redacted
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_redact_value(item, spec) for item in value]
    if isinstance(value, str):
        return _redact_text(value, spec)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value), spec)


def _event_seq(event: Mapping[str, Any]) -> int:
    value = event.get("seq")
    return value if isinstance(value, int) and value >= 0 else 0


def _visible_lead_message(
    event: Mapping[str, Any],
    expected_type: str,
) -> Mapping[str, Any] | None:
    metadata = event.get("metadata")
    if isinstance(metadata, Mapping):
        caller = metadata.get("caller")
        if caller not in {None, "lead_agent"}:
            return None
    content = event.get("content")
    if not isinstance(content, Mapping):
        return None
    if content.get("type") != expected_type:
        return None
    additional = content.get("additional_kwargs")
    if isinstance(additional, Mapping) and additional.get("hide_from_ui") is True:
        return None
    if content.get("name") == "summary":
        return None
    return content


def _bounded_events(
    events: Sequence[Mapping[str, Any]],
    max_events: int,
) -> tuple[list[Mapping[str, Any]], bool]:
    if max_events < _MIN_MAX_EVENTS:
        raise ValueError(f"max_events must be at least {_MIN_MAX_EVENTS}")
    ordered = sorted(events, key=_event_seq)
    if len(ordered) <= max_events:
        return ordered, False
    task_input_event = next(
        (event for event in ordered if event.get("event_type") == "llm.human.input" and _visible_lead_message(event, "human") is not None),
        ordered[0],
    )
    selected = [task_input_event]
    for event in reversed(ordered):
        if event is task_input_event:
            continue
        selected.append(event)
        if len(selected) >= max_events:
            break
    return sorted(selected, key=_event_seq), True


def _tool_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _tool_status(message: Mapping[str, Any], result_text: str) -> str:
    status = message.get("status")
    if status in {"success", "error"}:
        return str(status)
    return "error" if result_text.lstrip().startswith("Error:") else "unknown"


def _tool_error_type(message: Mapping[str, Any]) -> str | None:
    additional = message.get("additional_kwargs")
    if not isinstance(additional, Mapping):
        return None
    metadata = additional.get("deerflow_tool_meta")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("error_type")
    return str(value) if value else None


def _skill_from_read(
    tool: _MutableToolEvent,
) -> TraceSkillEvent | None:
    if tool.tool_name != "read_file" or tool.status != "success":
        return None
    arguments = tool.raw_arguments
    if not isinstance(arguments, Mapping):
        return None
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str):
        return None
    path = raw_path.replace("\\", "/")
    if not path.startswith("/mnt/skills/") or posixpath.basename(path) != "SKILL.md":
        return None

    result_message = tool.result_message
    additional = result_message.get("additional_kwargs") if isinstance(result_message, Mapping) else None
    metadata = additional.get(_SKILL_CONTEXT_ENTRY_KEY) if isinstance(additional, Mapping) else None
    if isinstance(metadata, Mapping) and isinstance(
        metadata.get("path"),
        str,
    ):
        path = str(metadata["path"]).replace("\\", "/")

    skill_name = posixpath.basename(posixpath.dirname(path))
    if not skill_name or not tool.raw_result:
        return None
    return TraceSkillEvent(
        skill_name=skill_name,
        skill_path=path,
        content_hash=hashlib.sha256(tool.raw_result.encode("utf-8")).hexdigest(),
        activation_source="read",
    )


def _skill_from_activation(
    event: Mapping[str, Any],
) -> TraceSkillEvent | None:
    if event.get("event_type") != _SKILL_ACTIVATION_EVENT:
        return None
    content = event.get("content")
    if not isinstance(content, Mapping) or content.get("action") != "activate":
        return None
    changes = content.get("changes")
    if not isinstance(changes, Mapping):
        return None
    name = changes.get("skill_name")
    path = changes.get("path")
    content_hash = changes.get("content_hash")
    if not all(isinstance(item, str) and item for item in (name, path, content_hash)):
        return None
    category = changes.get("category")
    return TraceSkillEvent(
        skill_name=str(name),
        skill_path=str(path),
        content_hash=str(content_hash),
        activation_source="slash",
        category=str(category) if category else None,
    )


def _workspace_artifacts(
    event: Mapping[str, Any],
    spec: RedactionSpec,
) -> list[str]:
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    changes = metadata.get("workspace_changes")
    if not isinstance(changes, Mapping):
        return []
    files = changes.get("files")
    if not isinstance(files, list):
        return []
    artifacts: list[str] = []
    roots = {
        "workspace": "/mnt/user-data/workspace",
        "outputs": "/mnt/user-data/outputs",
    }
    for item in files:
        if not isinstance(item, Mapping):
            continue
        path = item.get("path")
        root = item.get("root")
        status = item.get("status")
        if not isinstance(path, str) or root not in roots or status not in {"created", "modified", "symlink_created"}:
            continue
        artifacts.append(
            _redact_text(
                f"{roots[str(root)]}/{path.lstrip('/')}",
                spec,
            )
        )
    return artifacts


def _delivery_artifacts(
    event: Mapping[str, Any],
    spec: RedactionSpec,
) -> list[str]:
    if event.get("event_type") != "run.delivery":
        return []
    content = event.get("content")
    if not isinstance(content, Mapping):
        return []
    paths = content.get("paths")
    if not isinstance(paths, list):
        return []
    return [_redact_text(path, spec) for path in paths if isinstance(path, str) and path]


def _event_created_at(events: Sequence[Mapping[str, Any]]) -> datetime:
    parsed: list[datetime] = []
    for event in events:
        value = event.get("created_at")
        if not isinstance(value, str):
            continue
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        parsed.append(timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC))
    return max(parsed) if parsed else datetime.fromtimestamp(0, UTC)


def _dedupe_strings(values: Sequence[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _dedupe_skills(
    values: Sequence[TraceSkillEvent],
) -> list[TraceSkillEvent]:
    result: list[TraceSkillEvent] = []
    seen: set[tuple[str, str, str, str]] = set()
    for value in values:
        key = (
            value.skill_name,
            value.skill_path,
            value.content_hash,
            value.activation_source,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= 32:
            break
    return result


def build_evolution_trace_snapshot(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    thread_id: str,
    user_id: str,
    model_name: str | None,
    run_status: TraceRunStatus,
    stop_reason: str | None,
    environment: Mapping[str, Any],
    redaction: RedactionSpec | None = None,
    max_events: int = 256,
) -> EvolutionTraceSnapshot:
    """Materialize one deterministic snapshot from a run's persisted events."""
    spec = redaction or RedactionSpec()
    # A retry reads the already-persisted snapshot from the same event stream.
    # Excluding it keeps source counts and the content hash stable.
    all_events = sorted(
        (event for event in events if event.get("event_type") != _EVOLUTION_TRACE_EVENT),
        key=_event_seq,
    )
    included, truncated = _bounded_events(all_events, max_events)

    human_messages: list[str] = []
    final_answer = ""
    tools: list[_MutableToolEvent] = []
    tools_by_id: dict[str, _MutableToolEvent] = {}
    skills: list[TraceSkillEvent] = []
    artifacts: list[str] = []

    for event in included:
        event_type = event.get("event_type")
        if event_type == "llm.human.input":
            message = _visible_lead_message(event, "human")
            if message is not None:
                text = _redact_text(
                    _message_text(message.get("content")),
                    spec,
                )
                text, _ = _truncate(text, _MAX_CORRECTION_CHARS)
                if text:
                    human_messages.append(text)
            continue

        if event_type == "llm.ai.response":
            message = _visible_lead_message(event, "ai")
            if message is None:
                continue
            calls = _tool_calls(message)
            for index, call in enumerate(calls):
                tool_call_id = str(call.get("id") or f"unpaired-{_event_seq(event)}-{index}")
                tool_name = str(call.get("name") or "unknown_tool")
                raw_arguments = call.get("args")
                argument_mapping = raw_arguments if isinstance(raw_arguments, Mapping) else {}
                arguments = _stable_json(_redact_value(raw_arguments or {}, spec))
                arguments, _ = _truncate(arguments, _MAX_ARGUMENT_CHARS)
                tool = _MutableToolEvent(
                    sequence=_event_seq(event),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_arguments=argument_mapping,
                )
                tools.append(tool)
                tools_by_id[tool_call_id] = tool
            if not calls:
                text = _redact_text(
                    _message_text(message.get("content")),
                    spec,
                )
                text, _ = _truncate(text, _MAX_FINAL_ANSWER_CHARS)
                if text:
                    final_answer = text
            continue

        if event_type == "llm.tool.result":
            message = event.get("content")
            if not isinstance(message, Mapping):
                continue
            tool_call_id = str(message.get("tool_call_id") or f"unpaired-result-{_event_seq(event)}")
            result_raw = _message_text(message.get("content"))
            result = _redact_text(result_raw, spec)
            result, result_truncated = _truncate(
                result,
                _MAX_RESULT_CHARS,
            )
            tool = tools_by_id.get(tool_call_id)
            if tool is None:
                tool = _MutableToolEvent(
                    sequence=_event_seq(event),
                    tool_call_id=tool_call_id,
                    tool_name=str(message.get("name") or "unknown_tool"),
                    arguments="{}",
                )
                tools.append(tool)
                tools_by_id[tool_call_id] = tool
            tool.result = result
            tool.raw_result = result_raw
            tool.result_message = message
            tool.result_truncated = result_truncated
            tool.status = _tool_status(message, result)
            tool.error_type = _tool_error_type(message)
            continue

        skill = _skill_from_activation(event)
        if skill is not None:
            skills.append(skill)
        artifacts.extend(_workspace_artifacts(event, spec))
        artifacts.extend(_delivery_artifacts(event, spec))

    for tool in tools:
        skill = _skill_from_read(tool)
        if skill is not None:
            skills.append(skill)
            tool.result = f"[skill content omitted; sha256={skill.content_hash}]"
            tool.result_truncated = False

    task_input = human_messages[0] if human_messages else ""
    task_input, _ = _truncate(task_input, _MAX_TASK_INPUT_CHARS)
    corrections = [_truncate(item, _MAX_CORRECTION_CHARS)[0] for item in human_messages[1:17]]

    tool_events = [
        TraceToolEvent(
            sequence=tool.sequence,
            tool_call_id=tool.tool_call_id,
            tool_name=tool.tool_name,
            arguments=tool.arguments,
            result=tool.result,
            status=tool.status,
            error_type=tool.error_type,
            result_truncated=tool.result_truncated,
        )
        for tool in sorted(
            tools,
            key=lambda item: (item.sequence, item.tool_call_id),
        )[:256]
    ]
    skill_events = _dedupe_skills(skills)
    artifact_paths = _dedupe_strings(artifacts, limit=256)
    created_at = _event_created_at(all_events)
    environment_model = EnvironmentSignature.model_validate(_redact_value(dict(environment), spec))

    content = {
        "schema_version": EVOLUTION_TRACE_SCHEMA_VERSION,
        "run_id": run_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "model_name": model_name,
        "run_status": run_status,
        "stop_reason": stop_reason,
        "task_input": task_input,
        "final_answer": final_answer,
        "tool_events": [item.model_dump(mode="json") for item in tool_events],
        "skill_events": [item.model_dump(mode="json") for item in skill_events],
        "user_corrections": corrections,
        "artifacts": artifact_paths,
        "environment": environment_model.model_dump(mode="json"),
        "source_event_count": len(all_events),
        "included_event_count": len(included),
        "truncated": truncated,
        "created_at": created_at.isoformat(),
    }
    snapshot_hash = hashlib.sha256(_stable_json(content).encode("utf-8")).hexdigest()
    return EvolutionTraceSnapshot(
        snapshot_hash=snapshot_hash,
        **content,
    )
