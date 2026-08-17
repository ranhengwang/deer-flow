"""Replay task contracts and isolated evaluation workspaces."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import posixpath
import re
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, StringConstraints, model_validator

from deerflow.config.skill_evolution_config import (
    SkillEvolutionQualityConfig,
)
from deerflow.skill_evolution.models import (
    DetailText,
    EvaluationArtifact,
    EvaluationDecision,
    EvaluationMetrics,
    EvolutionEvent,
    EvolutionModel,
    EvolutionTraceSnapshot,
    Identifier,
    OutcomeStatus,
    ProposalOperation,
    ProposalStatus,
    Sha256,
    ShortText,
    SkillEvaluation,
    SkillProposal,
    TaskEvaluationResult,
    TraceRunStatus,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
    PutResult,
    SkillEvolutionStore,
)

REPLAY_TASK_SCHEMA_VERSION = "deerflow.skill-evolution.replay-task.v1"
REPLAY_FIXTURE_SCHEMA_VERSION = "deerflow.skill-evolution.replay-fixture.v1"
NEW_SKILL_EVALUATOR_VERSION = "new-skill-evaluator-v1"
_MAX_FIXTURE_FILES = 128
_MAX_FIXTURE_FILE_BYTES = 4 * 1024 * 1024
_MAX_FIXTURE_TOTAL_BYTES = 16 * 1024 * 1024
_SECRET_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Replayability(StrEnum):
    automatic = "automatic"
    manual_review = "manual_review"


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode())


def _normalize_relative_path(path: str) -> str:
    raw = path.replace("\\", "/").strip()
    if not raw:
        raise ValueError("path must not be empty")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise ValueError("absolute paths are not allowed")
    normalized = posixpath.normpath(raw)
    if normalized in {"", "."} or normalized != raw or any(part in {"", ".."} for part in PurePosixPath(normalized).parts):
        raise ValueError("path must be normalized without parent traversal")
    return normalized


class ReplayFixtureFile(EvolutionModel):
    path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1_024),
    ]
    content_base64: Annotated[
        str,
        StringConstraints(max_length=6_000_000),
    ]
    size_bytes: int = Field(ge=0, le=_MAX_FIXTURE_FILE_BYTES)
    content_hash: Sha256

    @model_validator(mode="after")
    def _validate_file(self) -> Self:
        _normalize_relative_path(self.path)
        try:
            content = base64.b64decode(
                self.content_base64,
                validate=True,
            )
        except (ValueError, binascii.Error) as exc:
            raise ValueError("fixture content is not valid base64") from exc
        if len(content) != self.size_bytes:
            raise ValueError("fixture size does not match decoded content")
        if _sha256_bytes(content) != self.content_hash:
            raise ValueError("fixture content hash does not match")
        return self

    def decode_content(self) -> bytes:
        return base64.b64decode(
            self.content_base64,
            validate=True,
        )


def _fixture_hash_payload(
    files: list[ReplayFixtureFile],
    *,
    complete: bool,
    omitted_paths: list[str],
) -> dict[str, object]:
    return {
        "schema_version": REPLAY_FIXTURE_SCHEMA_VERSION,
        "files": [
            {
                "path": item.path,
                "size_bytes": item.size_bytes,
                "content_hash": item.content_hash,
            }
            for item in files
        ],
        "complete": complete,
        "omitted_paths": omitted_paths,
    }


class ReplayFixtureSnapshot(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.replay-fixture.v1"] = REPLAY_FIXTURE_SCHEMA_VERSION
    snapshot_hash: Sha256
    files: list[ReplayFixtureFile] = Field(max_length=_MAX_FIXTURE_FILES)
    total_bytes: int = Field(ge=0, le=_MAX_FIXTURE_TOTAL_BYTES)
    complete: bool = True
    omitted_paths: list[
        Annotated[
            str,
            StringConstraints(
                strip_whitespace=True,
                min_length=1,
                max_length=1_024,
            ),
        ]
    ] = Field(
        default_factory=list,
        max_length=_MAX_FIXTURE_FILES,
    )

    @model_validator(mode="after")
    def _validate_snapshot(self) -> Self:
        paths = [item.path for item in self.files]
        if paths != sorted(paths):
            raise ValueError("fixture files must be sorted by path")
        if len(set(paths)) != len(paths):
            raise ValueError("fixture paths must be unique")
        if self.total_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("fixture total_bytes does not match files")
        if len(set(self.omitted_paths)) != len(self.omitted_paths):
            raise ValueError("fixture omitted paths must be unique")
        expected = _sha256_text(
            _stable_json(
                _fixture_hash_payload(
                    self.files,
                    complete=self.complete,
                    omitted_paths=self.omitted_paths,
                )
            )
        )
        if self.snapshot_hash != expected:
            raise ValueError("fixture snapshot_hash does not match")
        if self.complete and self.omitted_paths:
            raise ValueError("complete fixture cannot contain omitted paths")
        return self


def build_replay_fixture_snapshot(
    files: Mapping[str, bytes],
    *,
    complete: bool = True,
    omitted_paths: list[str] | None = None,
) -> ReplayFixtureSnapshot:
    """Build a deterministic binary-safe fixture snapshot."""
    if len(files) > _MAX_FIXTURE_FILES:
        raise ValueError("fixture contains too many files")
    records: list[ReplayFixtureFile] = []
    total = 0
    for path, content in sorted(files.items()):
        normalized = _normalize_relative_path(path)
        if not isinstance(content, bytes):
            raise TypeError("fixture content must be bytes")
        if len(content) > _MAX_FIXTURE_FILE_BYTES:
            raise ValueError("fixture file exceeds size limit")
        total += len(content)
        if total > _MAX_FIXTURE_TOTAL_BYTES:
            raise ValueError("fixture exceeds total size limit")
        encoded = base64.b64encode(content).decode("ascii")
        records.append(
            ReplayFixtureFile(
                path=normalized,
                content_base64=encoded,
                size_bytes=len(content),
                content_hash=_sha256_bytes(content),
            )
        )
    omitted = sorted({_normalize_relative_path(path) for path in (omitted_paths or [])})
    payload = _fixture_hash_payload(
        records,
        complete=complete,
        omitted_paths=omitted,
    )
    return ReplayFixtureSnapshot(
        snapshot_hash=_sha256_text(_stable_json(payload)),
        files=records,
        total_bytes=total,
        complete=complete,
        omitted_paths=omitted,
    )


def _capture_fixture_sync(
    source_root: Path,
    paths: list[str],
) -> ReplayFixtureSnapshot:
    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError("fixture source root must be a directory")
    captured: dict[str, bytes] = {}
    for raw_path in paths:
        path = _normalize_relative_path(raw_path)
        candidate = source_root
        for part in PurePosixPath(path).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("fixture paths must not contain symlinks")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("fixture path escapes source root") from exc
        if not resolved.is_file():
            raise ValueError("fixture path must identify a regular file")
        captured[path] = resolved.read_bytes()
    return build_replay_fixture_snapshot(captured)


async def capture_replay_fixture_snapshot(
    source_root: str | Path,
    paths: list[str],
) -> ReplayFixtureSnapshot:
    """Capture selected source files off the event loop."""
    return await asyncio.to_thread(
        _capture_fixture_sync,
        Path(source_root),
        paths,
    )


class ReplayEnvironmentRequirements(EvolutionModel):
    os: Identifier
    shell: Identifier | None = None
    runtime: Identifier | None = None
    required_commands: list[Identifier] = Field(
        default_factory=list,
        max_length=32,
    )
    required_secret_names: list[Identifier] = Field(
        default_factory=list,
        max_length=32,
    )
    network_required: bool = False

    @model_validator(mode="after")
    def _validate_requirements(self) -> Self:
        for values in (
            self.required_commands,
            self.required_secret_names,
        ):
            if len(set(values)) != len(values):
                raise ValueError("environment requirements must be unique")
        if any(not _SECRET_NAME_RE.fullmatch(name) for name in self.required_secret_names):
            raise ValueError("required secret names must be environment identifiers")
        return self


class ReplayCommandVerifier(EvolutionModel):
    kind: Literal["command"] = "command"
    command: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
    ]
    expected_exit_code: int = Field(default=0, ge=0, le=255)
    timeout_seconds: int = Field(default=60, ge=1, le=600)


class ReplayArtifactVerifier(EvolutionModel):
    kind: Literal["artifact"] = "artifact"
    path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1_024),
    ]
    must_exist: bool = True
    expected_sha256: Sha256 | None = None
    contains_text: DetailText | None = None

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        _normalize_relative_path(self.path)
        if not self.must_exist and (self.expected_sha256 is not None or self.contains_text is not None):
            raise ValueError("absent artifact verifier cannot assert content")
        return self


ReplayVerifier = Annotated[
    ReplayCommandVerifier | ReplayArtifactVerifier,
    Field(discriminator="kind"),
]


class ReplaySideEffectPolicy(EvolutionModel):
    writable_roots: list[Literal["workspace", "outputs"]] = Field(
        default_factory=lambda: ["workspace", "outputs"],
        min_length=1,
        max_length=2,
    )
    network_allowed: bool = False
    skill_library_writes_allowed: bool = False
    max_changed_files: int = Field(default=128, ge=0, le=10_000)
    max_written_bytes: int = Field(
        default=16 * 1024 * 1024,
        ge=0,
        le=1024 * 1024 * 1024,
    )

    @model_validator(mode="after")
    def _validate_roots(self) -> Self:
        if len(set(self.writable_roots)) != len(self.writable_roots):
            raise ValueError("writable roots must be unique")
        if self.skill_library_writes_allowed:
            raise ValueError("evaluation cannot allow production Skill writes")
        return self


class ReplayTaskSpec(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.replay-task.v1"] = REPLAY_TASK_SCHEMA_VERSION
    task_id: Identifier
    source_event_id: Identifier
    source_run_id: Identifier
    source_snapshot_hash: Sha256
    user_id: Identifier
    task_family: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=256,
        ),
    ]
    task_input: Annotated[
        str,
        StringConstraints(max_length=8_000),
    ]
    fixture: ReplayFixtureSnapshot
    environment: ReplayEnvironmentRequirements
    verifiers: list[ReplayVerifier] = Field(max_length=16)
    timeout_seconds: int = Field(default=300, ge=1, le=3_600)
    side_effect_policy: ReplaySideEffectPolicy = Field(
        default_factory=ReplaySideEffectPolicy,
    )
    replayability: Replayability
    manual_review_reasons: list[Identifier] = Field(max_length=16)
    created_at: datetime

    @model_validator(mode="after")
    def _validate_replayability(self) -> Self:
        if self.replayability is Replayability.automatic:
            if self.manual_review_reasons:
                raise ValueError("automatic replay cannot have manual review reasons")
            if not self.task_input.strip():
                raise ValueError("automatic replay requires task input")
            if not self.verifiers:
                raise ValueError("automatic replay requires deterministic verifier")
            if not self.fixture.complete:
                raise ValueError("automatic replay requires complete fixture")
            if self.environment.network_required and not self.side_effect_policy.network_allowed:
                raise ValueError("automatic replay cannot require blocked network")
            if self.environment.required_secret_names:
                raise ValueError("automatic replay cannot require external credentials")
        elif not self.manual_review_reasons:
            raise ValueError("manual replay requires at least one reason")
        if len(set(self.manual_review_reasons)) != len(self.manual_review_reasons):
            raise ValueError("manual review reasons must be unique")
        return self


def build_replay_task_spec(
    snapshot: EvolutionTraceSnapshot,
    event: EvolutionEvent,
    *,
    fixture: ReplayFixtureSnapshot,
    verifiers: list[ReplayCommandVerifier | ReplayArtifactVerifier],
    environment: ReplayEnvironmentRequirements | None = None,
    side_effect_policy: ReplaySideEffectPolicy | None = None,
    timeout_seconds: int = 300,
) -> ReplayTaskSpec:
    """Build a replay spec and conservatively classify replayability."""
    if snapshot.run_id != event.run_id:
        raise ValueError("snapshot and event run IDs do not match")
    if snapshot.user_id != event.user_id:
        raise ValueError("snapshot and event users do not match")
    if snapshot.snapshot_hash != event.source_snapshot_hash:
        raise ValueError("snapshot hash does not match event source")
    if _sha256_text(snapshot.task_input) != event.task_input_hash:
        raise ValueError("snapshot task input hash does not match event")
    if snapshot.run_status is not TraceRunStatus.success or event.outcome.status is not OutcomeStatus.success:
        raise ValueError("replay specs require successful source evidence")

    resolved_environment = environment or ReplayEnvironmentRequirements(
        os=snapshot.environment.os,
        shell=snapshot.environment.shell,
        runtime=snapshot.environment.runtime,
    )
    policy = side_effect_policy or ReplaySideEffectPolicy()
    reasons: list[str] = []
    if not snapshot.task_input.strip():
        reasons.append("missing_task_input")
    if not verifiers:
        reasons.append("no_deterministic_verifier")
    if snapshot.truncated:
        reasons.append("source_trace_truncated")
    if not fixture.complete:
        reasons.append("incomplete_fixture")
    if resolved_environment.network_required and not policy.network_allowed:
        reasons.append("network_required_but_disallowed")
    if resolved_environment.required_secret_names:
        reasons.append("external_credentials_required")
    replayability = Replayability.automatic if not reasons else Replayability.manual_review
    id_payload = {
        "source_event_id": event.event_id,
        "source_snapshot_hash": snapshot.snapshot_hash,
        "task_family": event.task_signature,
        "fixture_hash": fixture.snapshot_hash,
        "environment": resolved_environment.model_dump(mode="json"),
        "verifiers": [verifier.model_dump(mode="json") for verifier in verifiers],
        "timeout_seconds": timeout_seconds,
        "side_effect_policy": policy.model_dump(mode="json"),
    }
    task_id = f"replay-{_sha256_text(_stable_json(id_payload))[:32]}"
    return ReplayTaskSpec(
        task_id=task_id,
        source_event_id=event.event_id,
        source_run_id=event.run_id,
        source_snapshot_hash=snapshot.snapshot_hash,
        user_id=event.user_id,
        task_family=event.task_signature,
        task_input=snapshot.task_input,
        fixture=fixture,
        environment=resolved_environment,
        verifiers=verifiers,
        timeout_seconds=timeout_seconds,
        side_effect_policy=policy,
        replayability=replayability,
        manual_review_reasons=reasons,
        created_at=snapshot.created_at,
    )


class ReplayWorkspacePaths(EvolutionModel):
    root: Path
    workspace: Path
    outputs: Path
    skills: Path


class ReplaySideEffectReport(EvolutionModel):
    changed_paths: list[str] = Field(max_length=10_000)
    violations: list[str] = Field(max_length=10_000)
    changed_file_count: int = Field(ge=0)
    written_bytes: int = Field(ge=0)
    within_policy: bool


def _safe_join(root: Path, relative_path: str) -> Path:
    normalized = _normalize_relative_path(relative_path)
    target = root.joinpath(*PurePosixPath(normalized).parts)
    resolved_root = root.resolve()
    resolved_target = target.resolve(strict=False)
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("materialized path escapes replay root") from exc
    return target


def _scan_files(root: Path) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            relative = path.relative_to(root).as_posix()
            result[relative] = ("symlink", 0)
            continue
        if not path.is_file():
            continue
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        result[relative] = (_sha256_bytes(content), len(content))
    return result


def _materialize_replay_workspace(
    spec: ReplayTaskSpec,
    proposal: SkillProposal | None,
    parent_dir: Path | None,
) -> tuple[ReplayWorkspacePaths, dict[str, tuple[str, int]]]:
    if proposal is not None:
        if proposal.user_id != spec.user_id:
            raise ValueError("proposal user does not match replay task")
        if proposal.status not in {
            ProposalStatus.staged,
            ProposalStatus.validating,
        }:
            raise ValueError("only staged or validating proposals can be replayed")
    if parent_dir is not None:
        parent_dir.mkdir(parents=True, exist_ok=True)
    root = Path(
        tempfile.mkdtemp(
            prefix=f"{spec.task_id}-",
            dir=str(parent_dir) if parent_dir is not None else None,
        )
    )
    paths = ReplayWorkspacePaths(
        root=root,
        workspace=root / "workspace",
        outputs=root / "outputs",
        skills=root / "skills",
    )
    paths.workspace.mkdir()
    paths.outputs.mkdir()
    paths.skills.mkdir()
    try:
        for fixture_file in spec.fixture.files:
            target = _safe_join(
                paths.workspace,
                fixture_file.path,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(fixture_file.decode_content())
        if proposal is not None:
            skill_root = paths.skills / proposal.skill_name
            skill_root.mkdir(parents=True)
            for proposed_file in proposal.proposed_files:
                target = _safe_join(
                    skill_root,
                    proposed_file.path,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    proposed_file.content,
                    encoding="utf-8",
                )
                target.chmod(0o444)
        return paths, _scan_files(root)
    except Exception:
        _cleanup_replay_workspace(root)
        raise


def _cleanup_replay_workspace(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        try:
            if path.is_symlink():
                continue
            if path.is_dir():
                path.chmod(0o755)
            else:
                path.chmod(0o644)
        except OSError:
            pass
    shutil.rmtree(root)


@dataclass(slots=True)
class ReplayWorkspace:
    spec: ReplayTaskSpec
    paths: ReplayWorkspacePaths
    _initial_files: dict[str, tuple[str, int]]

    async def audit_side_effects(self) -> ReplaySideEffectReport:
        current = await asyncio.to_thread(
            _scan_files,
            self.paths.root,
        )
        changed = sorted(path for path in set(self._initial_files) | set(current) if self._initial_files.get(path) != current.get(path))
        policy = self.spec.side_effect_policy
        allowed_roots = set(policy.writable_roots)
        if policy.skill_library_writes_allowed:
            allowed_roots.add("skills")
        violations = [path for path in changed if path.split("/", 1)[0] not in allowed_roots]
        written_bytes = sum(current.get(path, ("", 0))[1] for path in changed)
        if len(changed) > policy.max_changed_files:
            violations.append("policy:max_changed_files")
        if written_bytes > policy.max_written_bytes:
            violations.append("policy:max_written_bytes")
        violations = sorted(set(violations))
        return ReplaySideEffectReport(
            changed_paths=changed,
            violations=violations,
            changed_file_count=len(changed),
            written_bytes=written_bytes,
            within_policy=not violations,
        )


@asynccontextmanager
async def isolated_replay_workspace(
    spec: ReplayTaskSpec,
    *,
    proposal: SkillProposal | None = None,
    parent_dir: str | Path | None = None,
) -> AsyncIterator[ReplayWorkspace]:
    """Materialize an isolated replay tree and always remove it."""
    paths, initial_files = await asyncio.to_thread(
        _materialize_replay_workspace,
        spec,
        proposal,
        Path(parent_dir) if parent_dir is not None else None,
    )
    workspace = ReplayWorkspace(
        spec=spec,
        paths=paths,
        _initial_files=initial_files,
    )
    try:
        yield workspace
    finally:
        await asyncio.to_thread(
            _cleanup_replay_workspace,
            paths.root,
        )


ReplayCondition = Literal[
    "no_skill",
    "candidate_skill",
]
ReplaySplit = Literal[
    "source",
    "held_out",
]


class ReplayAgentExecution(EvolutionModel):
    """Redacted metrics returned by the isolated agent runtime."""

    tool_calls: int = Field(ge=0, le=10_000)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    errors: list[Identifier] = Field(
        default_factory=list,
        max_length=32,
    )

    @model_validator(mode="after")
    def _validate_errors(self) -> Self:
        if len(set(self.errors)) != len(self.errors):
            raise ValueError("agent execution errors must be unique")
        if any(
            re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_.:-]{0,127}",
                error,
            )
            is None
            for error in self.errors
        ):
            raise ValueError("agent execution errors must be stable codes")
        return self


class ReplayCommandExecution(EvolutionModel):
    """Redacted result of a verifier command run inside the replay runtime."""

    exit_code: int = Field(ge=0, le=255)
    errors: list[Identifier] = Field(
        default_factory=list,
        max_length=16,
    )

    @model_validator(mode="after")
    def _validate_errors(self) -> Self:
        if len(set(self.errors)) != len(self.errors):
            raise ValueError("command execution errors must be unique")
        if any(
            re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_.:-]{0,127}",
                error,
            )
            is None
            for error in self.errors
        ):
            raise ValueError("command execution errors must be stable codes")
        return self


@dataclass(frozen=True, slots=True)
class ReplayAgentRequest:
    spec: ReplayTaskSpec
    paths: ReplayWorkspacePaths
    condition: ReplayCondition
    candidate_skill_path: Path | None


@dataclass(frozen=True, slots=True)
class ReplayCommandRequest:
    spec: ReplayTaskSpec
    paths: ReplayWorkspacePaths
    condition: ReplayCondition
    command: str
    timeout_seconds: int


@runtime_checkable
class ReplayRuntime(Protocol):
    """Runs agents and commands in a sandbox rooted at the supplied replay tree."""

    async def run_agent(
        self,
        request: ReplayAgentRequest,
    ) -> ReplayAgentExecution:
        """Run one task, force-activating the candidate only when supplied."""

    async def run_command(
        self,
        request: ReplayCommandRequest,
    ) -> ReplayCommandExecution:
        """Run one deterministic verifier command in the same isolated runtime."""


def _dedupe_bounded(
    values: list[str],
    *,
    limit: int = 32,
) -> list[str]:
    return list(dict.fromkeys(values))[:limit]


def _artifact_target(
    paths: ReplayWorkspacePaths,
    artifact_path: str,
) -> tuple[str, Path]:
    normalized = _normalize_relative_path(artifact_path)
    pure = PurePosixPath(normalized)
    first = pure.parts[0]
    if first == "skills":
        raise ValueError("production or candidate Skill paths cannot be artifact verifiers")
    if first in {"workspace", "outputs"}:
        return normalized, _safe_join(paths.root, normalized)
    display_path = f"workspace/{normalized}"
    return display_path, _safe_join(
        paths.workspace,
        normalized,
    )


def _inspect_artifact_file(
    path: Path,
    *,
    contains_text: str | None,
) -> tuple[str, int, bool]:
    digest = hashlib.sha256()
    size = 0
    needle = contains_text.encode("utf-8") if contains_text is not None else None
    overlap = b""
    found = needle is None
    with path.open("rb") as source:
        while chunk := source.read(64 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if needle is not None and not found:
                window = overlap + chunk
                found = needle in window
                if len(needle) > 1:
                    overlap = window[-(len(needle) - 1) :]
    return digest.hexdigest(), size, found


async def _evaluate_artifact_verifier(
    paths: ReplayWorkspacePaths,
    verifier: ReplayArtifactVerifier,
) -> tuple[bool, str | None, EvaluationArtifact]:
    try:
        display_path, target = _artifact_target(
            paths,
            verifier.path,
        )
    except ValueError:
        return (
            False,
            "artifact_path_forbidden",
            EvaluationArtifact(
                path=verifier.path,
                kind="missing",
            ),
        )
    if target.is_symlink():
        return (
            False,
            "artifact_symlink_forbidden",
            EvaluationArtifact(
                path=display_path,
                kind="symlink",
            ),
        )
    if not target.is_file():
        passed = not verifier.must_exist and verifier.expected_sha256 is None and verifier.contains_text is None
        return (
            passed,
            None if passed else "artifact_missing",
            EvaluationArtifact(
                path=display_path,
                kind="missing",
            ),
        )
    content_hash, size_bytes, contains_text = await asyncio.to_thread(
        _inspect_artifact_file,
        target,
        contains_text=verifier.contains_text,
    )
    artifact = EvaluationArtifact(
        path=display_path,
        kind="file",
        content_hash=content_hash,
        size_bytes=size_bytes,
    )
    if not verifier.must_exist:
        return False, "artifact_unexpectedly_exists", artifact
    if verifier.expected_sha256 is not None and content_hash != verifier.expected_sha256:
        return False, "artifact_hash_mismatch", artifact
    if verifier.contains_text is not None and not contains_text:
        return False, "artifact_contains_text_mismatch", artifact
    return True, None, artifact


async def _artifact_from_changed_path(
    paths: ReplayWorkspacePaths,
    relative_path: str,
) -> EvaluationArtifact | None:
    first = relative_path.split("/", 1)[0]
    if first not in {"workspace", "outputs"}:
        return None
    normalized = _normalize_relative_path(relative_path)
    target = paths.root.joinpath(
        *PurePosixPath(normalized).parts,
    )
    if target.is_symlink():
        return EvaluationArtifact(
            path=normalized,
            kind="symlink",
        )
    target = _safe_join(paths.root, normalized)
    if not target.is_file():
        return EvaluationArtifact(
            path=normalized,
            kind="missing",
        )
    content_hash, size_bytes, _ = await asyncio.to_thread(
        _inspect_artifact_file,
        target,
        contains_text=None,
    )
    return EvaluationArtifact(
        path=normalized,
        kind="file",
        content_hash=content_hash,
        size_bytes=size_bytes,
    )


def _evaluation_id(
    proposal: SkillProposal,
    source_tasks: list[ReplayTaskSpec],
    held_out_tasks: list[ReplayTaskSpec],
    quality_config: SkillEvolutionQualityConfig,
) -> str:
    payload = {
        "evaluator_version": NEW_SKILL_EVALUATOR_VERSION,
        "proposal_id": proposal.proposal_id,
        "source_tasks": [task.task_id for task in source_tasks],
        "held_out_tasks": [task.task_id for task in held_out_tasks],
        "quality": quality_config.model_dump(mode="json"),
    }
    return f"evaluation-{_sha256_text(_stable_json(payload))[:32]}"


def _success_rate(
    results: list[TaskEvaluationResult],
) -> float:
    if not results:
        return 0.0
    return sum(result.success for result in results) / len(results)


def _validate_new_skill_suite(
    proposal: SkillProposal,
    source_tasks: list[ReplayTaskSpec],
    held_out_tasks: list[ReplayTaskSpec],
) -> None:
    if proposal.operation is not ProposalOperation.create:
        raise ValueError("new-Skill evaluation requires a create proposal")
    if proposal.status not in {
        ProposalStatus.staged,
        ProposalStatus.validating,
    }:
        raise ValueError("new-Skill evaluation requires a staged or validating proposal")
    source_event_ids = [task.source_event_id for task in source_tasks]
    if len(set(source_event_ids)) != len(source_event_ids):
        raise ValueError("source replay tasks must be unique by source event")
    if set(source_event_ids) != set(proposal.supporting_event_ids):
        raise ValueError("source replay tasks must cover every proposal supporting events")
    source_families = {task.task_family for task in source_tasks}
    if len(source_families) != 1:
        raise ValueError("source replay tasks must share one task family")
    held_out_event_ids = [task.source_event_id for task in held_out_tasks]
    if len(set(held_out_event_ids)) != len(held_out_event_ids):
        raise ValueError("held-out replay tasks must be unique by source event")
    if set(source_event_ids) & set(held_out_event_ids):
        raise ValueError("source and held-out replay tasks must be independent")
    if any(task.task_family not in source_families for task in held_out_tasks):
        raise ValueError("held-out replay tasks must match the source task family")
    task_ids = [
        task.task_id
        for task in [
            *source_tasks,
            *held_out_tasks,
        ]
    ]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("replay task IDs must be unique across the evaluation suite")
    if any(
        task.user_id != proposal.user_id
        for task in [
            *source_tasks,
            *held_out_tasks,
        ]
    ):
        raise ValueError("all replay tasks must belong to the proposal user")


@dataclass(frozen=True, slots=True)
class NewSkillProposalEvaluator:
    """Runs paired candidate/no-Skill replays for one create Proposal."""

    runtime: ReplayRuntime
    quality_config: SkillEvolutionQualityConfig = field(
        default_factory=SkillEvolutionQualityConfig,
    )

    async def _run_task(
        self,
        proposal: SkillProposal,
        spec: ReplayTaskSpec,
        *,
        split: ReplaySplit,
        condition: ReplayCondition,
        parent_dir: Path | None,
    ) -> TaskEvaluationResult:
        candidate = proposal if condition == "candidate_skill" else None
        async with isolated_replay_workspace(
            spec,
            proposal=candidate,
            parent_dir=parent_dir,
        ) as workspace:
            candidate_skill_path = workspace.paths.skills / proposal.skill_name if candidate is not None else None
            request = ReplayAgentRequest(
                spec=spec,
                paths=workspace.paths,
                condition=condition,
                candidate_skill_path=candidate_skill_path,
            )
            execution: ReplayAgentExecution | None = None
            errors: list[str] = []
            started = time.perf_counter()
            try:
                async with asyncio.timeout(spec.timeout_seconds):
                    execution = await self.runtime.run_agent(
                        request,
                    )
            except TimeoutError:
                errors.append("agent_timeout")
            except Exception as exc:
                errors.append(f"agent_runtime_error:{type(exc).__name__}")
            latency_seconds = time.perf_counter() - started
            if execution is not None:
                errors.extend(execution.errors)

            artifacts: dict[str, EvaluationArtifact] = {}
            for verifier in spec.verifiers:
                if isinstance(verifier, ReplayCommandVerifier):
                    command_request = ReplayCommandRequest(
                        spec=spec,
                        paths=workspace.paths,
                        condition=condition,
                        command=verifier.command,
                        timeout_seconds=verifier.timeout_seconds,
                    )
                    try:
                        async with asyncio.timeout(verifier.timeout_seconds):
                            command_result = await self.runtime.run_command(
                                command_request,
                            )
                    except TimeoutError:
                        errors.append("command_verifier_timeout")
                    except Exception as exc:
                        errors.append(f"command_runtime_error:{type(exc).__name__}")
                    else:
                        errors.extend(command_result.errors)
                        if command_result.exit_code != verifier.expected_exit_code:
                            errors.append("command_exit_code_mismatch")
                else:
                    passed, error, artifact = await _evaluate_artifact_verifier(
                        workspace.paths,
                        verifier,
                    )
                    artifacts[artifact.path] = artifact
                    if not passed and error is not None:
                        errors.append(error)

            side_effects = await workspace.audit_side_effects()
            if not side_effects.within_policy:
                errors.append("side_effect_policy_violation")
            for path in side_effects.changed_paths:
                artifact = await _artifact_from_changed_path(
                    workspace.paths,
                    path,
                )
                if artifact is not None:
                    artifacts[artifact.path] = artifact

            bounded_errors = _dedupe_bounded(errors)
            success = not bounded_errors
            return TaskEvaluationResult(
                task_id=spec.task_id,
                split=split,
                condition=condition,
                success=success,
                metrics=EvaluationMetrics(
                    tool_calls=(execution.tool_calls if execution is not None else 0),
                    input_tokens=(execution.input_tokens if execution is not None else 0),
                    output_tokens=(execution.output_tokens if execution is not None else 0),
                    latency_seconds=latency_seconds,
                ),
                failure_reason=(bounded_errors[0] if bounded_errors else None),
                errors=bounded_errors,
                artifacts=[artifacts[path] for path in sorted(artifacts)],
            )

    async def evaluate(
        self,
        proposal: SkillProposal,
        *,
        source_tasks: list[ReplayTaskSpec],
        held_out_tasks: list[ReplayTaskSpec],
        parent_dir: str | Path | None = None,
    ) -> SkillEvaluation:
        """Run paired source and held-out evaluations without publishing."""
        _validate_new_skill_suite(
            proposal,
            source_tasks,
            held_out_tasks,
        )
        evaluation_id = _evaluation_id(
            proposal,
            source_tasks,
            held_out_tasks,
            self.quality_config,
        )
        all_tasks = [
            *source_tasks,
            *held_out_tasks,
        ]
        blocked = [task for task in all_tasks if task.replayability is Replayability.manual_review]
        if blocked:
            return SkillEvaluation(
                evaluation_id=evaluation_id,
                proposal_id=proposal.proposal_id,
                user_id=proposal.user_id,
                source_replay_results=[],
                held_out_results=[],
                baseline_results=[],
                candidate_results=[],
                regression_results=[],
                safety_results={
                    "replayability": "manual_review_required",
                    "quality_score": "deferred_phase_6_4",
                },
                quality_score=0.0,
                decision=EvaluationDecision.manual_review,
                created_at=datetime.now(UTC),
            )

        resolved_parent = Path(parent_dir) if parent_dir is not None else None
        source_results: list[TaskEvaluationResult] = []
        held_out_results: list[TaskEvaluationResult] = []
        baseline_results: list[TaskEvaluationResult] = []
        candidate_results: list[TaskEvaluationResult] = []
        for split, tasks in (
            ("source", source_tasks),
            ("held_out", held_out_tasks),
        ):
            for task in tasks:
                baseline = await self._run_task(
                    proposal,
                    task,
                    split=split,
                    condition="no_skill",
                    parent_dir=resolved_parent,
                )
                candidate = await self._run_task(
                    proposal,
                    task,
                    split=split,
                    condition="candidate_skill",
                    parent_dir=resolved_parent,
                )
                baseline_results.append(baseline)
                candidate_results.append(candidate)
                if split == "source":
                    source_results.append(candidate)
                else:
                    held_out_results.append(candidate)

        source_rate = _success_rate(source_results)
        held_out_rate = _success_rate(held_out_results)
        side_effect_violation = any(
            "side_effect_policy_violation" in result.errors
            for result in [
                *baseline_results,
                *candidate_results,
            ]
        )
        if not held_out_tasks:
            decision = EvaluationDecision.manual_review
        elif side_effect_violation:
            decision = EvaluationDecision.reject
        elif proposal.requires_manual_review:
            decision = EvaluationDecision.manual_review
        elif source_rate < self.quality_config.min_source_replay_success_rate or held_out_rate < self.quality_config.min_held_out_success_rate:
            decision = EvaluationDecision.reject
        else:
            decision = EvaluationDecision.approve
        safety_results: dict[str, ShortText] = {
            "source_replay": (f"{sum(result.success for result in source_results)}/{len(source_results)}"),
            "held_out": (f"{sum(result.success for result in held_out_results)}/{len(held_out_results)}" if held_out_results else "missing"),
            "side_effects": ("violation" if side_effect_violation else "allow"),
            "quality_score": "deferred_phase_6_4",
        }
        if proposal.requires_manual_review:
            safety_results["proposal_review"] = "manual_review_required"
        return SkillEvaluation(
            evaluation_id=evaluation_id,
            proposal_id=proposal.proposal_id,
            user_id=proposal.user_id,
            source_replay_results=source_results,
            held_out_results=held_out_results,
            baseline_results=baseline_results,
            candidate_results=candidate_results,
            regression_results=[],
            safety_results=safety_results,
            quality_score=0.0,
            decision=decision,
            created_at=datetime.now(UTC),
        )

    async def evaluate_and_persist(
        self,
        proposal: SkillProposal,
        *,
        source_tasks: list[ReplayTaskSpec],
        held_out_tasks: list[ReplayTaskSpec],
        store: SkillEvolutionStore,
        parent_dir: str | Path | None = None,
    ) -> PutResult[SkillEvaluation]:
        """Evaluate once, persist raw results, and reject failed Proposals."""
        _validate_new_skill_suite(
            proposal,
            source_tasks,
            held_out_tasks,
        )
        evaluation_id = _evaluation_id(
            proposal,
            source_tasks,
            held_out_tasks,
            self.quality_config,
        )
        stored = await store.get_proposal(
            proposal.user_id,
            proposal.proposal_id,
        )
        if stored is None:
            raise ValueError("proposal must be persisted before evaluation")
        provided_payload = proposal.model_dump(
            mode="json",
            exclude={"status"},
        )
        stored_payload = stored.model_dump(
            mode="json",
            exclude={"status"},
        )
        if provided_payload != stored_payload:
            raise ValueError("persisted proposal does not match evaluation input")

        existing = await store.get_evaluation(
            proposal.user_id,
            evaluation_id,
        )
        if stored.status is ProposalStatus.staged:
            try:
                stored = await store.transition_proposal(
                    user_id=stored.user_id,
                    proposal_id=stored.proposal_id,
                    expected_status=ProposalStatus.staged,
                    new_status=ProposalStatus.validating,
                )
            except EvolutionStoreConflict:
                refreshed = await store.get_proposal(
                    stored.user_id,
                    stored.proposal_id,
                )
                if refreshed is None or refreshed.status is not ProposalStatus.validating:
                    raise
                stored = refreshed
        elif stored.status is ProposalStatus.rejected and existing is not None and existing.decision is EvaluationDecision.reject:
            return PutResult(existing, created=False)
        elif stored.status is not ProposalStatus.validating:
            raise ValueError("proposal is not eligible for evaluation")
        if existing is not None:
            if existing.decision is EvaluationDecision.reject:
                await self._reject_validating_proposal(
                    store,
                    stored,
                )
            return PutResult(existing, created=False)

        evaluation = await self.evaluate(
            stored,
            source_tasks=source_tasks,
            held_out_tasks=held_out_tasks,
            parent_dir=parent_dir,
        )
        try:
            put_result = await store.put_evaluation(
                evaluation,
            )
        except EvolutionStoreConflict:
            winner = await store.get_evaluation(
                proposal.user_id,
                evaluation_id,
            )
            if winner is None:
                raise
            put_result = PutResult(
                winner,
                created=False,
            )
        if put_result.value.decision is EvaluationDecision.reject:
            await self._reject_validating_proposal(
                store,
                stored,
            )
        return put_result

    async def _reject_validating_proposal(
        self,
        store: SkillEvolutionStore,
        proposal: SkillProposal,
    ) -> None:
        current = await store.get_proposal(
            proposal.user_id,
            proposal.proposal_id,
        )
        if current is None or current.status is ProposalStatus.rejected:
            return
        if current.status is not ProposalStatus.validating:
            raise EvolutionStoreConflict("failed evaluation requires a validating proposal")
        try:
            await store.transition_proposal(
                user_id=current.user_id,
                proposal_id=current.proposal_id,
                expected_status=ProposalStatus.validating,
                new_status=ProposalStatus.rejected,
            )
        except EvolutionStoreConflict:
            refreshed = await store.get_proposal(
                current.user_id,
                current.proposal_id,
            )
            if refreshed is None or refreshed.status is not ProposalStatus.rejected:
                raise
