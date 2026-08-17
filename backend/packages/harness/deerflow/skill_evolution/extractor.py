"""Deterministic pre-extraction for eligible skill-evolution traces."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from deerflow.skill_evolution.eligibility import (
    EligibilityBranch,
    EligibilityDecision,
)
from deerflow.skill_evolution.models import (
    ComplexitySignals,
    DetailText,
    EnvironmentSignature,
    EvolutionEventKind,
    EvolutionModel,
    EvolutionTraceSnapshot,
    Identifier,
    OutcomeEvidence,
    Sha256,
    SkillUsage,
    ToolSignature,
    TraceToolEvent,
)

DETERMINISTIC_EXTRACTION_SCHEMA_VERSION = "deerflow.skill-evolution.pre-extraction.v1"
DETERMINISTIC_EXTRACTOR_VERSION = "deterministic-v1"

_MAX_SEGMENT_CHARS = 2_000
_MAX_ARGUMENT_CHARS = 700
_MAX_RESULT_CHARS = 1_000
_DEFAULT_MAX_TOOL_SEGMENTS = 32
_DEFAULT_MAX_CORRECTION_SEGMENTS = 8
_DEFAULT_MAX_ARTIFACT_SEGMENTS = 16
_MAX_CANDIDATE_SEGMENTS = 64
_DATA_URI_RE = re.compile(
    r"data:[^,]{0,256};base64,",
    re.IGNORECASE,
)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

ArtifactPath = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1_024,
    ),
]


class IneligibleTraceError(ValueError):
    """Raised when pre-extraction receives a rejected trace."""


class CandidateSegmentKind(StrEnum):
    task = "task"
    tool = "tool"
    correction = "correction"
    artifact = "artifact"
    final_answer = "final_answer"


class CandidateSegment(EvolutionModel):
    """One bounded, hash-addressed excerpt for semantic extraction."""

    segment_id: Identifier
    kind: CandidateSegmentKind
    source_index: int | None = Field(default=None, ge=0)
    sequence: int | None = Field(default=None, ge=0)
    content: DetailText
    content_hash: Sha256
    truncated: bool = False
    omitted: bool = False
    tool_name: Identifier | None = None
    tool_status: Literal["success", "error", "unknown"] | None = None
    error_type: Identifier | None = None

    @model_validator(mode="after")
    def _validate_tool_fields(self) -> Self:
        tool_fields = (
            self.tool_name,
            self.tool_status,
            self.sequence,
        )
        if self.kind is CandidateSegmentKind.tool:
            if any(value is None for value in tool_fields):
                raise ValueError("tool segment requires tool fields and sequence")
        elif any(value is not None for value in tool_fields):
            raise ValueError("non-tool segment must not include tool fields")
        return self


class ToolSequenceEntry(EvolutionModel):
    """Content-free structural record for every observed tool step."""

    index: int = Field(ge=0, le=255)
    sequence: int = Field(ge=0)
    tool_call_id: Identifier
    tool_name: Identifier
    status: Literal["success", "error", "unknown"]
    error_type: Identifier | None = None
    arguments_hash: Sha256
    result_hash: Sha256
    result_truncated: bool = False


class DeterministicExtraction(EvolutionModel):
    """Reproducible intermediate representation for Phase 3.2."""

    schema_version: Literal["deerflow.skill-evolution.pre-extraction.v1"] = DETERMINISTIC_EXTRACTION_SCHEMA_VERSION
    extraction_hash: Sha256
    extractor_version: Identifier
    source_snapshot_hash: Sha256
    run_id: Identifier
    thread_id: Identifier
    user_id: Identifier
    task_input_hash: Sha256
    event_kind: EvolutionEventKind
    environment: EnvironmentSignature
    outcome: OutcomeEvidence
    eligibility: EligibilityDecision
    complexity: ComplexitySignals
    tool_signature: ToolSignature
    tool_sequence: list[ToolSequenceEntry] = Field(
        default_factory=list,
        max_length=256,
    )
    observed_skill_usages: list[SkillUsage] = Field(
        default_factory=list,
        max_length=32,
    )
    artifact_paths: list[ArtifactPath] = Field(
        default_factory=list,
        max_length=64,
    )
    candidate_segments: list[CandidateSegment] = Field(
        min_length=1,
        max_length=_MAX_CANDIDATE_SEGMENTS,
    )
    source_tool_count: int = Field(ge=0, le=256)
    included_tool_count: int = Field(ge=0, le=32)
    segments_truncated: bool
    created_at: datetime

    @model_validator(mode="after")
    def _validate_extraction(self) -> Self:
        if not self.eligibility.eligible:
            raise ValueError("pre-extraction requires eligible decision")
        if self.complexity != self.eligibility.complexity:
            raise ValueError("complexity must match eligibility decision")
        if self.outcome.status is not self.eligibility.outcome_status or self.outcome.confidence != self.eligibility.outcome_confidence:
            raise ValueError("outcome must match eligibility decision")
        if self.included_tool_count > self.source_tool_count:
            raise ValueError("included tool count cannot exceed source count")
        if self.source_tool_count != len(self.tool_sequence):
            raise ValueError("source tool count must match tool sequence")
        if self.included_tool_count < self.source_tool_count and not self.segments_truncated:
            raise ValueError("omitted tool segments require truncated marker")
        if self.event_kind is EvolutionEventKind.new_skill_evidence:
            if self.observed_skill_usages:
                raise ValueError("new-skill extraction cannot include Skill usage")
            if self.eligibility.branch is not EligibilityBranch.no_skill:
                raise ValueError("new-skill extraction requires no-skill branch")
        else:
            if not self.observed_skill_usages:
                raise ValueError("patch extraction requires observed Skill usage")
            if self.eligibility.branch is not EligibilityBranch.skill_used:
                raise ValueError("patch extraction requires skill-used branch")
        canonical = self.model_dump(
            mode="json",
            exclude={"extraction_hash"},
        )
        if self.extraction_hash != _sha256(_stable_json(canonical)):
            raise ValueError("extraction_hash does not match extraction payload")
        return self


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _segment_id(prefix: str, value: str) -> str:
    segment_id = f"{prefix}:{value}"
    if len(segment_id) <= 128:
        return segment_id
    digest = _sha256(segment_id)[:12]
    return f"{segment_id[:115]}-{digest}"


def _bounded_text(
    value: str,
    limit: int,
) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    marker = f"\n... [truncated {len(value) - limit} chars]\n"
    remaining = max(0, limit - len(marker))
    head = remaining // 2
    tail = remaining - head
    return (
        f"{value[:head]}{marker}{value[-tail:]}"[:limit],
        True,
    )


def _is_probable_encoded_or_binary(value: str) -> bool:
    if not value:
        return False
    if "\x00" in value or _DATA_URI_RE.search(value):
        return True
    compact = "".join(value.split())
    if len(compact) >= 512 and len(compact) % 4 == 0 and _BASE64_RE.fullmatch(compact) is not None:
        character_classes = sum(
            (
                any(char.islower() for char in compact),
                any(char.isupper() for char in compact),
                any(char.isdigit() for char in compact),
                any(char in "+/=" for char in compact),
            )
        )
        if character_classes >= 3:
            return True
    controls = sum(1 for char in value[:4_096] if ord(char) < 32 and char not in "\n\r\t")
    return controls >= 8


def _compact_field(
    value: str,
    *,
    limit: int,
) -> tuple[str, bool, bool]:
    if _is_probable_encoded_or_binary(value):
        return (
            (f"[encoded or binary content omitted; sha256={_sha256(value)}]"),
            True,
            True,
        )
    bounded, truncated = _bounded_text(value, limit)
    return bounded, truncated, False


def _text_segment(
    *,
    segment_id: str,
    kind: CandidateSegmentKind,
    value: str,
    source_index: int | None = None,
) -> CandidateSegment:
    content, truncated = _bounded_text(
        value,
        _MAX_SEGMENT_CHARS,
    )
    return CandidateSegment(
        segment_id=segment_id,
        kind=kind,
        source_index=source_index,
        content=content,
        content_hash=_sha256(value),
        truncated=truncated,
    )


def _tool_segment(
    source_index: int,
    tool: TraceToolEvent,
) -> CandidateSegment:
    arguments, args_truncated, args_omitted = _compact_field(
        tool.arguments,
        limit=_MAX_ARGUMENT_CHARS,
    )
    result, result_truncated, result_omitted = _compact_field(
        tool.result,
        limit=_MAX_RESULT_CHARS,
    )
    raw_content = _stable_json(
        {
            "arguments": tool.arguments,
            "error_type": tool.error_type,
            "result": tool.result,
            "status": tool.status,
            "tool_name": tool.tool_name,
        }
    )
    content = _stable_json(
        {
            "arguments": arguments,
            "error_type": tool.error_type,
            "result": result,
            "status": tool.status,
            "tool_name": tool.tool_name,
        }
    )
    content, content_truncated = _bounded_text(
        content,
        _MAX_SEGMENT_CHARS,
    )
    return CandidateSegment(
        segment_id=_segment_id("tool", tool.tool_call_id),
        kind=CandidateSegmentKind.tool,
        source_index=source_index,
        sequence=tool.sequence,
        content=content,
        content_hash=_sha256(raw_content),
        truncated=(tool.result_truncated or args_truncated or result_truncated or content_truncated),
        omitted=args_omitted or result_omitted,
        tool_name=tool.tool_name,
        tool_status=tool.status,
        error_type=tool.error_type,
    )


def _head_tail_indices(
    length: int,
    limit: int,
) -> list[int]:
    if length <= limit:
        return list(range(length))
    head = limit // 2
    tail = limit - head
    return [
        *range(head),
        *range(length - tail, length),
    ]


def _select_tool_indices(
    tools: list[TraceToolEvent],
    limit: int,
) -> list[int]:
    if len(tools) <= limit:
        return list(range(len(tools)))
    error_indices = [index for index, tool in enumerate(tools) if tool.status == "error"]
    if len(error_indices) >= limit:
        selected_error_positions = _head_tail_indices(
            len(error_indices),
            limit,
        )
        return sorted(error_indices[position] for position in selected_error_positions)
    selected = set(error_indices)
    for index in range(len(tools) - 1, -1, -1):
        selected.add(index)
        if len(selected) >= limit:
            break
    return sorted(selected)


def _skill_usages(
    snapshot: EvolutionTraceSnapshot,
) -> list[SkillUsage]:
    return [
        SkillUsage(
            used=True,
            skill_name=skill.skill_name,
            skill_path=skill.skill_path,
            content_hash=skill.content_hash,
            activation_source=skill.activation_source,
        )
        for skill in snapshot.skill_events
    ]


def _tool_signature(
    tools: list[TraceToolEvent],
) -> ToolSignature:
    error_types: list[str] = []
    seen_errors: set[str] = set()
    for tool in tools:
        if tool.status != "error":
            continue
        error_type = tool.error_type or "unknown"
        if error_type in seen_errors:
            continue
        seen_errors.add(error_type)
        error_types.append(error_type)
        if len(error_types) >= 64:
            break
    return ToolSignature(
        tool_names=[tool.tool_name for tool in tools],
        error_types=error_types,
    )


def _tool_sequence(
    tools: list[TraceToolEvent],
) -> list[ToolSequenceEntry]:
    return [
        ToolSequenceEntry(
            index=index,
            sequence=tool.sequence,
            tool_call_id=tool.tool_call_id,
            tool_name=tool.tool_name,
            status=tool.status,
            error_type=tool.error_type,
            arguments_hash=_sha256(tool.arguments),
            result_hash=_sha256(tool.result),
            result_truncated=tool.result_truncated,
        )
        for index, tool in enumerate(tools)
    ]


def _validate_segment_limits(
    *,
    max_tool_segments: int,
    max_correction_segments: int,
    max_artifact_segments: int,
) -> None:
    limits = (
        ("max_tool_segments", max_tool_segments, 32),
        (
            "max_correction_segments",
            max_correction_segments,
            8,
        ),
        (
            "max_artifact_segments",
            max_artifact_segments,
            16,
        ),
    )
    for name, value, maximum in limits:
        if value < 1 or value > maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")


def pre_extract_evolution(
    snapshot: EvolutionTraceSnapshot,
    outcome: OutcomeEvidence,
    eligibility: EligibilityDecision,
    *,
    extractor_version: str = DETERMINISTIC_EXTRACTOR_VERSION,
    max_tool_segments: int = _DEFAULT_MAX_TOOL_SEGMENTS,
    max_correction_segments: int = (_DEFAULT_MAX_CORRECTION_SEGMENTS),
    max_artifact_segments: int = _DEFAULT_MAX_ARTIFACT_SEGMENTS,
) -> DeterministicExtraction:
    """Build a reproducible, bounded input for semantic extraction."""
    if not eligibility.eligible:
        raise IneligibleTraceError("trace is not eligible for pre-extraction")
    if outcome.status is not eligibility.outcome_status or outcome.confidence != eligibility.outcome_confidence:
        raise ValueError("outcome does not match eligibility decision")
    expected_branch = EligibilityBranch.skill_used if snapshot.skill_events else EligibilityBranch.no_skill
    if eligibility.branch is not expected_branch:
        raise ValueError("eligibility branch does not match snapshot Skill usage")
    _validate_segment_limits(
        max_tool_segments=max_tool_segments,
        max_correction_segments=max_correction_segments,
        max_artifact_segments=max_artifact_segments,
    )
    tools = sorted(
        snapshot.tool_events,
        key=lambda tool: (tool.sequence, tool.tool_call_id),
    )

    segments: list[CandidateSegment] = []
    if snapshot.task_input:
        segments.append(
            _text_segment(
                segment_id="task:input",
                kind=CandidateSegmentKind.task,
                value=snapshot.task_input,
            )
        )

    selected_tool_indices = _select_tool_indices(
        tools,
        max_tool_segments,
    )
    segments.extend(_tool_segment(index, tools[index]) for index in selected_tool_indices)

    correction_indices = _head_tail_indices(
        len(snapshot.user_corrections),
        max_correction_segments,
    )
    segments.extend(
        _text_segment(
            segment_id=f"correction:{index}",
            kind=CandidateSegmentKind.correction,
            value=snapshot.user_corrections[index],
            source_index=index,
        )
        for index in correction_indices
    )

    artifact_indices = _head_tail_indices(
        len(snapshot.artifacts),
        max_artifact_segments,
    )
    artifact_paths = [
        snapshot.artifacts[index]
        for index in _head_tail_indices(
            len(snapshot.artifacts),
            64,
        )
    ]
    segments.extend(
        _text_segment(
            segment_id=f"artifact:{index}",
            kind=CandidateSegmentKind.artifact,
            value=snapshot.artifacts[index],
            source_index=index,
        )
        for index in artifact_indices
    )

    if snapshot.final_answer:
        segments.append(
            _text_segment(
                segment_id="final:answer",
                kind=CandidateSegmentKind.final_answer,
                value=snapshot.final_answer,
            )
        )

    if not segments:
        raise ValueError("eligible snapshot produced no candidate segments")
    if len(segments) > _MAX_CANDIDATE_SEGMENTS:
        raise ValueError("candidate segment budget exceeded")

    event_kind = EvolutionEventKind.skill_patch_evidence if expected_branch is EligibilityBranch.skill_used else EvolutionEventKind.new_skill_evidence
    segments_truncated = (
        len(selected_tool_indices) < len(tools)
        or len(correction_indices) < len(snapshot.user_corrections)
        or len(artifact_indices) < len(snapshot.artifacts)
        or len(artifact_paths) < len(snapshot.artifacts)
        or any(segment.truncated for segment in segments)
    )
    payload = {
        "schema_version": (DETERMINISTIC_EXTRACTION_SCHEMA_VERSION),
        "extractor_version": extractor_version,
        "source_snapshot_hash": snapshot.snapshot_hash,
        "run_id": snapshot.run_id,
        "thread_id": snapshot.thread_id,
        "user_id": snapshot.user_id,
        "task_input_hash": _sha256(snapshot.task_input),
        "event_kind": event_kind,
        "environment": snapshot.environment.model_dump(mode="json"),
        "outcome": outcome.model_dump(mode="json"),
        "eligibility": eligibility.model_dump(mode="json"),
        "complexity": eligibility.complexity.model_dump(mode="json"),
        "tool_signature": _tool_signature(tools).model_dump(mode="json"),
        "tool_sequence": [entry.model_dump(mode="json") for entry in _tool_sequence(tools)],
        "observed_skill_usages": [usage.model_dump(mode="json") for usage in _skill_usages(snapshot)],
        "artifact_paths": artifact_paths,
        "candidate_segments": [segment.model_dump(mode="json") for segment in segments],
        "source_tool_count": len(tools),
        "included_tool_count": len(selected_tool_indices),
        "segments_truncated": segments_truncated,
        "created_at": snapshot.created_at.isoformat().replace(
            "+00:00",
            "Z",
        ),
    }
    extraction_hash = _sha256(_stable_json(payload))
    return DeterministicExtraction(
        extraction_hash=extraction_hash,
        **payload,
    )
