"""Deterministic fingerprints and candidate grouping for evolution events."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from deerflow.config.skill_evolution_config import (
    SkillEvolutionGroupingConfig,
)
from deerflow.skill_evolution.models import (
    ClusterStatus,
    DetailText,
    EvolutionCluster,
    EvolutionEvent,
    EvolutionEventKind,
    EvolutionModel,
    GroupingEvidence,
    Identifier,
    Sha256,
)

EVENT_FINGERPRINT_SCHEMA_VERSION = "deerflow.skill-evolution.fingerprint.v1"
_MAX_GROUPING_EVENTS = 1_024
_MAX_CLUSTER_MEMBERS = 64
_WORD_RE = re.compile(
    r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]",
    re.IGNORECASE,
)
_NON_NAME_RE = re.compile(r"[^a-z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "with",
    }
)


class EventFingerprint(EvolutionModel):
    """Normalized workflow features for one evolution event."""

    schema_version: Literal["deerflow.skill-evolution.fingerprint.v1"] = EVENT_FINGERPRINT_SCHEMA_VERSION
    fingerprint_hash: Sha256
    event_id: Identifier
    run_id: Identifier
    user_id: Identifier
    task_input_hash: Sha256
    event_kind: EvolutionEventKind
    target_skill_name: Identifier | None = None
    target_skill_hash: Sha256 | None = None
    task_signature: Identifier
    task_signature_tokens: list[Identifier] = Field(
        max_length=64,
    )
    task_goal_tokens: list[Identifier] = Field(max_length=64)
    tool_sequence: list[Identifier] = Field(max_length=256)
    error_types: list[Identifier] = Field(max_length=64)
    environment_family: Identifier
    canonical_signature: str = Field(
        min_length=1,
        max_length=512,
    )

    @model_validator(mode="after")
    def _validate_hash(self) -> Self:
        payload = _fingerprint_payload(
            user_id=self.user_id,
            event_kind=self.event_kind,
            target_skill_name=self.target_skill_name,
            target_skill_hash=self.target_skill_hash,
            task_signature=self.task_signature,
            task_signature_tokens=self.task_signature_tokens,
            task_goal_tokens=self.task_goal_tokens,
            tool_sequence=self.tool_sequence,
            error_types=self.error_types,
            environment_family=self.environment_family,
        )
        if self.fingerprint_hash != _sha256(_stable_json(payload)):
            raise ValueError("fingerprint_hash does not match normalized features")
        return self


class FingerprintComparison(EvolutionModel):
    compatible: bool
    score: float = Field(ge=0.0, le=1.0)
    hard_mismatch: Identifier | None = None
    components: dict[Identifier, float] = Field(
        max_length=8,
    )
    reason: DetailText

    @model_validator(mode="after")
    def _validate_mismatch(self) -> Self:
        if self.hard_mismatch is not None:
            if self.compatible or self.score != 0.0:
                raise ValueError("hard mismatch must be incompatible with zero score")
        return self


class DeduplicationReason(StrEnum):
    same_event_id = "same_event_id"
    same_run = "same_run"
    same_task_input = "same_task_input"
    identical_fingerprint = "identical_fingerprint"


class DeduplicationRecord(EvolutionModel):
    duplicate_event_id: Identifier
    canonical_event_id: Identifier
    reason: DeduplicationReason


class DeterministicGroupingResult(EvolutionModel):
    fingerprints: list[EventFingerprint] = Field(
        max_length=_MAX_GROUPING_EVENTS,
    )
    clusters: list[EvolutionCluster] = Field(max_length=512)
    duplicates: list[DeduplicationRecord] = Field(
        max_length=_MAX_GROUPING_EVENTS,
    )


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


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()
    return _NON_NAME_RE.sub("-", normalized).strip("-") or "unknown"


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()
    result: list[str] = []
    seen: set[str] = set()
    for token in _WORD_RE.findall(normalized):
        if token in _STOP_WORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 64:
            break
    return result


def _os_family(value: str) -> str:
    normalized = _normalize_name(value)
    if normalized in {"darwin", "mac", "macos", "os-x"}:
        return "macos"
    if "windows" in normalized or normalized.startswith("win"):
        return "windows"
    if "linux" in normalized:
        return "linux"
    return normalized


def _runtime_family(value: str | None) -> str:
    if not value:
        return "unknown"
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()
    match = re.search(
        r"(python|node(?:js)?|go|java|ruby|rust)[^0-9]*(\d+)?",
        normalized,
    )
    if match is None:
        return _normalize_name(normalized)
    language = "node" if match.group(1).startswith("node") else match.group(1)
    major = match.group(2)
    return f"{language}:{major}" if major else language


def _shell_family(value: str | None) -> str:
    normalized = _normalize_name(value or "")
    if normalized in {"bash", "fish", "sh", "zsh"}:
        return "posix"
    if "powershell" in normalized or normalized in {"pwsh"}:
        return "powershell"
    if normalized in {"cmd", "cmd-exe"}:
        return "cmd"
    return normalized


def _environment_family(event: EvolutionEvent) -> str:
    return "|".join(
        (
            _os_family(event.environment.os),
            _runtime_family(event.environment.runtime),
            _shell_family(event.environment.shell),
        )
    )


def _bounded_signature(value: str) -> str:
    if len(value) <= 512:
        return value
    return f"{value[:495]}|{_sha256(value)[:16]}"


def _fingerprint_payload(
    *,
    user_id: str,
    event_kind: EvolutionEventKind,
    target_skill_name: str | None,
    target_skill_hash: str | None,
    task_signature: str,
    task_signature_tokens: list[str],
    task_goal_tokens: list[str],
    tool_sequence: list[str],
    error_types: list[str],
    environment_family: str,
) -> dict[str, object]:
    return {
        "user_id": user_id,
        "event_kind": event_kind.value,
        "target_skill_name": target_skill_name,
        "target_skill_hash": target_skill_hash,
        "task_signature": task_signature,
        "task_signature_tokens": task_signature_tokens,
        "task_goal_tokens": task_goal_tokens,
        "tool_sequence": tool_sequence,
        "error_types": error_types,
        "environment_family": environment_family,
    }


def build_event_fingerprint(
    event: EvolutionEvent,
) -> EventFingerprint:
    """Build a reproducible normalized fingerprint for one event."""
    task_signature = _normalize_name(event.task_signature)
    task_signature_tokens = _tokens(event.task_signature.replace("-", " "))
    task_goal_tokens = _tokens(event.task_goal)
    tool_sequence = [_normalize_name(tool) for tool in event.tool_signature.tool_names]
    error_types = sorted({_normalize_name(error) for error in event.tool_signature.error_types})
    target_name = event.target_skill.name if event.target_skill is not None else None
    target_hash = event.target_skill.content_hash if event.target_skill is not None else None
    environment_family = _environment_family(event)
    payload = _fingerprint_payload(
        user_id=event.user_id,
        event_kind=event.event_kind,
        target_skill_name=target_name,
        target_skill_hash=target_hash,
        task_signature=task_signature,
        task_signature_tokens=task_signature_tokens,
        task_goal_tokens=task_goal_tokens,
        tool_sequence=tool_sequence,
        error_types=error_types,
        environment_family=environment_family,
    )
    fingerprint_hash = _sha256(_stable_json(payload))
    canonical_signature = _bounded_signature(
        "|".join(
            (
                event.event_kind.value,
                target_name or "-",
                task_signature,
                ">".join(tool_sequence) or "-",
                ",".join(error_types) or "-",
                environment_family,
                fingerprint_hash[:16],
            )
        )
    )
    return EventFingerprint(
        fingerprint_hash=fingerprint_hash,
        event_id=event.event_id,
        run_id=event.run_id,
        user_id=event.user_id,
        task_input_hash=event.task_input_hash,
        event_kind=event.event_kind,
        target_skill_name=target_name,
        target_skill_hash=target_hash,
        task_signature=task_signature,
        task_signature_tokens=task_signature_tokens,
        task_goal_tokens=task_goal_tokens,
        tool_sequence=tool_sequence,
        error_types=error_types,
        environment_family=environment_family,
        canonical_signature=canonical_signature,
    )


def _jaccard(
    left: list[str],
    right: list[str],
) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union)


def _sequence_similarity(
    left: list[str],
    right: list[str],
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1] / max(len(left), len(right))


def _environment_similarity(
    left: str,
    right: str,
) -> float:
    if left == right:
        return 1.0
    return 0.5 if left.split("|", 1)[0] == right.split("|", 1)[0] else 0.0


def _hard_mismatch(
    left: EventFingerprint,
    right: EventFingerprint,
) -> str | None:
    if left.user_id != right.user_id:
        return "user_id"
    if left.event_kind is not right.event_kind:
        return "event_kind"
    if left.event_kind is EvolutionEventKind.skill_patch_evidence and left.target_skill_name != right.target_skill_name:
        return "target_skill"
    return None


def compare_event_fingerprints(
    left: EventFingerprint,
    right: EventFingerprint,
    *,
    threshold: float = 0.65,
) -> FingerprintComparison:
    """Score whether two fingerprints describe the same workflow."""
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0 and 1")
    mismatch = _hard_mismatch(left, right)
    if mismatch is not None:
        return FingerprintComparison(
            compatible=False,
            score=0.0,
            hard_mismatch=mismatch,
            components={},
            reason=f"Hard grouping boundary differs: {mismatch}.",
        )

    task_signature = (
        1.0
        if left.task_signature == right.task_signature
        else _jaccard(
            left.task_signature_tokens,
            right.task_signature_tokens,
        )
    )
    tools = _sequence_similarity(
        left.tool_sequence,
        right.tool_sequence,
    )
    goal = _jaccard(
        left.task_goal_tokens,
        right.task_goal_tokens,
    )
    errors = _jaccard(left.error_types, right.error_types)
    environment = _environment_similarity(
        left.environment_family,
        right.environment_family,
    )
    components = {
        "task_signature": round(task_signature, 6),
        "tool_sequence": round(tools, 6),
        "task_goal": round(goal, 6),
        "error_types": round(errors, 6),
        "environment": round(environment, 6),
    }
    score = round(
        0.35 * task_signature + 0.30 * tools + 0.20 * goal + 0.10 * errors + 0.05 * environment,
        6,
    )
    workflow_anchor = task_signature == 1.0 or (tools >= 0.75 and goal >= 0.35)
    compatible = score >= threshold and workflow_anchor
    reason = f"Deterministic score={score:.3f}; threshold={threshold:.3f}; workflow_anchor={str(workflow_anchor).lower()}."
    return FingerprintComparison(
        compatible=compatible,
        score=score,
        components=components,
        reason=reason,
    )


def _partition_key(
    fingerprint: EventFingerprint,
) -> tuple[str, EvolutionEventKind, str | None]:
    return (
        fingerprint.user_id,
        fingerprint.event_kind,
        fingerprint.target_skill_name,
    )


def _cluster_id(
    prototype: EventFingerprint,
) -> str:
    value = _stable_json(
        {
            "user_id": prototype.user_id,
            "event_kind": prototype.event_kind.value,
            "target_skill": prototype.target_skill_name,
            "prototype": prototype.fingerprint_hash,
        }
    )
    return f"cluster-{_sha256(value)[:32]}"


def _deduplicate(
    events: list[EvolutionEvent],
    fingerprints: dict[str, EventFingerprint],
) -> tuple[
    list[EvolutionEvent],
    list[DeduplicationRecord],
]:
    unique: list[EvolutionEvent] = []
    duplicates: list[DeduplicationRecord] = []
    seen_event_ids: dict[
        tuple[str, EvolutionEventKind, str | None, str],
        str,
    ] = {}
    seen_runs: dict[
        tuple[str, EvolutionEventKind, str | None, str],
        str,
    ] = {}
    seen_inputs: dict[
        tuple[str, EvolutionEventKind, str | None, str],
        str,
    ] = {}
    seen_fingerprints: dict[
        tuple[str, str],
        str,
    ] = {}

    for event in events:
        fingerprint = fingerprints[event.event_id]
        partition = _partition_key(fingerprint)
        checks = (
            (
                (
                    *partition,
                    event.event_id,
                ),
                seen_event_ids,
                DeduplicationReason.same_event_id,
            ),
            (
                (*partition, event.run_id),
                seen_runs,
                DeduplicationReason.same_run,
            ),
            (
                (*partition, event.task_input_hash),
                seen_inputs,
                DeduplicationReason.same_task_input,
            ),
            (
                (
                    event.user_id,
                    fingerprint.fingerprint_hash,
                ),
                seen_fingerprints,
                DeduplicationReason.identical_fingerprint,
            ),
        )
        duplicate_of: str | None = None
        duplicate_reason: DeduplicationReason | None = None
        for key, seen, reason in checks:
            duplicate_of = seen.get(key)
            if duplicate_of is not None:
                duplicate_reason = reason
                break
        if duplicate_of is not None:
            duplicates.append(
                DeduplicationRecord(
                    duplicate_event_id=event.event_id,
                    canonical_event_id=duplicate_of,
                    reason=duplicate_reason,
                )
            )
            continue

        unique.append(event)
        seen_event_ids[(*partition, event.event_id)] = event.event_id
        seen_runs[(*partition, event.run_id)] = event.event_id
        seen_inputs[(*partition, event.task_input_hash)] = event.event_id
        seen_fingerprints[(event.user_id, fingerprint.fingerprint_hash)] = event.event_id
    return unique, duplicates


def group_evolution_events(
    events: list[EvolutionEvent],
    *,
    config: SkillEvolutionGroupingConfig | None = None,
) -> DeterministicGroupingResult:
    """Build deterministic collecting clusters from structured events."""
    if len(events) > _MAX_GROUPING_EVENTS:
        raise ValueError(f"at most {_MAX_GROUPING_EVENTS} events may be grouped")
    resolved_config = config or SkillEvolutionGroupingConfig()
    ordered = sorted(
        events,
        key=lambda event: (
            event.created_at,
            event.event_id,
        ),
    )
    events_by_id: dict[str, EvolutionEvent] = {}
    for event in ordered:
        existing = events_by_id.get(event.event_id)
        if existing is not None and existing != event:
            raise ValueError("event ID contains conflicting grouping payloads")
        events_by_id[event.event_id] = event
    fingerprint_by_id = {event.event_id: build_event_fingerprint(event) for event in ordered}
    unique, duplicates = _deduplicate(
        ordered,
        fingerprint_by_id,
    )

    groups: list[
        tuple[
            EvolutionEvent,
            EventFingerprint,
            list[tuple[EvolutionEvent, FingerprintComparison]],
        ]
    ] = []
    for event in unique:
        fingerprint = fingerprint_by_id[event.event_id]
        candidates: list[
            tuple[
                float,
                str,
                int,
                FingerprintComparison,
            ]
        ] = []
        for index, (_, prototype, _) in enumerate(groups):
            comparison = compare_event_fingerprints(
                prototype,
                fingerprint,
                threshold=(resolved_config.deterministic_threshold),
            )
            if comparison.compatible:
                candidates.append(
                    (
                        comparison.score,
                        prototype.fingerprint_hash,
                        index,
                        comparison,
                    )
                )
        if not candidates:
            prototype_comparison = FingerprintComparison(
                compatible=True,
                score=1.0,
                components={
                    "task_signature": 1.0,
                    "tool_sequence": 1.0,
                    "task_goal": 1.0,
                    "error_types": 1.0,
                    "environment": 1.0,
                },
                reason="Prototype event.",
            )
            groups.append(
                (
                    event,
                    fingerprint,
                    [(event, prototype_comparison)],
                )
            )
            continue
        _, _, group_index, comparison = max(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )
        groups[group_index][2].append((event, comparison))

    clusters: list[EvolutionCluster] = []
    for prototype_event, prototype, members in groups:
        if len(members) > _MAX_CLUSTER_MEMBERS:
            raise ValueError("deterministic cluster exceeds 64 members")
        member_events = [event for event, _ in members]
        clusters.append(
            EvolutionCluster(
                cluster_id=_cluster_id(prototype),
                user_id=prototype.user_id,
                event_kind=prototype.event_kind,
                target_skill=prototype_event.target_skill,
                canonical_signature=(prototype.canonical_signature),
                member_event_ids=[event.event_id for event in member_events],
                independent_run_count=len({event.run_id for event in member_events}),
                status=ClusterStatus.collecting,
                grouping_evidence=[
                    GroupingEvidence(
                        method="deterministic",
                        score=comparison.score,
                        reason=comparison.reason,
                    )
                    for _, comparison in members
                ],
                created_at=min(event.created_at for event in member_events),
                updated_at=max(event.created_at for event in member_events),
            )
        )
    clusters.sort(
        key=lambda cluster: (
            cluster.user_id,
            cluster.event_kind.value,
            (cluster.target_skill.name if cluster.target_skill is not None else ""),
            cluster.canonical_signature,
        )
    )
    return DeterministicGroupingResult(
        fingerprints=[fingerprint_by_id[event.event_id] for event in ordered],
        clusters=clusters,
        duplicates=duplicates,
    )
