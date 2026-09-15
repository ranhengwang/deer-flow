"""Shared security and persistence boundary for custom Skill mutations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, NoReturn
from weakref import WeakValueDictionary

from deerflow.agents.lead_agent.prompt import (
    refresh_user_skills_system_prompt_cache_async,
)
from deerflow.skills.package import (
    SkillPackageFile,
    compute_skill_package_hash,
    materialize_skill_package,
)
from deerflow.skills.security_scanner import (
    scan_skill_content,
)
from deerflow.skills.security_static_scanner import (
    StaticFinding,
    StaticScanBlockedError,
    StaticScannerError,
    enforce_static_scan,
)
from deerflow.skills.storage import (
    get_or_new_user_skill_storage,
)
from deerflow.skills.storage.skill_storage import (
    SkillStorage,
    SkillStorageConflict,
)
from deerflow.skills.types import SKILL_MD_FILE

MutationAction = Literal[
    "create",
    "patch",
    "edit",
    "delete",
    "write_file",
    "remove_file",
]
MutationAuthor = Literal["agent", "evolution"]
PackageMutationAction = Literal[
    "evolution_publish",
    "evolution_rollback",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_skill_locks: WeakValueDictionary[
    tuple[str, str],
    asyncio.Lock,
] = WeakValueDictionary()


class SkillMutationConflict(ValueError):
    """Raised when a mutation compare-and-swap token is stale."""


@dataclass(frozen=True, slots=True)
class SkillMutationRequest:
    user_id: str
    action: MutationAction | str
    name: str
    content: str | None = None
    path: str | None = None
    find: str | None = None
    replace: str | None = None
    expected_count: int | None = None
    expected_base_hash: str | None = None
    author: MutationAuthor = "agent"
    thread_id: str | None = None
    proposal_id: str | None = None
    evaluation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("user_id must not be empty")
        if self.expected_base_hash is not None and not _SHA256_RE.fullmatch(self.expected_base_hash):
            raise ValueError("expected_base_hash must be a lowercase SHA-256")
        if (self.proposal_id is None) != (self.evaluation_id is None):
            raise ValueError("proposal_id and evaluation_id must be provided together")
        if self.author == "evolution" and self.proposal_id is None:
            raise ValueError("evolution mutations require proposal_id and evaluation_id")
        for label, value in (
            ("thread_id", self.thread_id),
            ("proposal_id", self.proposal_id),
            ("evaluation_id", self.evaluation_id),
        ):
            if value is not None and not _ID_RE.fullmatch(value):
                raise ValueError(f"{label} is invalid")
        if self.expected_count is not None and self.expected_count < 1:
            raise ValueError("expected_count must be positive")


@dataclass(frozen=True, slots=True)
class SkillMutationResult:
    action: MutationAction
    name: str
    file_path: str
    message: str
    previous_skill_hash: str | None
    resulting_skill_hash: str | None


@dataclass(frozen=True, slots=True)
class SkillPackageMutationRequest:
    user_id: str
    action: PackageMutationAction
    name: str
    files: tuple[SkillPackageFile, ...]
    moderation_paths: tuple[str, ...]
    expected_base_hash: str | None
    expected_package_hash: str | None
    require_absent: bool
    proposal_id: str
    evaluation_id: str | None
    publication_id: str
    actor_id: str | None = None

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("user_id must not be empty")
        for label, value in (
            ("proposal_id", self.proposal_id),
            ("evaluation_id", self.evaluation_id),
            ("publication_id", self.publication_id),
            ("actor_id", self.actor_id),
        ):
            if value is not None and not _ID_RE.fullmatch(value):
                raise ValueError(f"{label} is invalid")
        for label, value in (
            ("expected_base_hash", self.expected_base_hash),
            ("expected_package_hash", self.expected_package_hash),
        ):
            if value is not None and not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if self.require_absent and self.expected_base_hash is not None:
            raise ValueError("require_absent cannot be combined with expected_base_hash")
        if self.files and not any(item.path == SKILL_MD_FILE for item in self.files):
            raise ValueError("Skill package mutation requires SKILL.md")
        if not self.files and self.action != "evolution_rollback":
            raise ValueError("Only rollback may delete a complete Skill package")
        file_paths = {item.path for item in self.files}
        if not set(self.moderation_paths) <= file_paths:
            raise ValueError("moderation paths must exist in the Skill package")


@dataclass(frozen=True, slots=True)
class SkillPackageMutationResult:
    action: PackageMutationAction
    name: str
    previous_skill_hash: str | None
    previous_package_hash: str | None
    resulting_skill_hash: str | None
    resulting_package_hash: str


ContentScanner = Callable[
    ...,
    Awaitable[dict[str, Any]],
]
StaticCandidateScanner = Callable[
    [str, dict[str, str], SkillStorage | None],
    Awaitable[list[StaticFinding]],
]
PackageCandidateScanner = Callable[
    [str, tuple[SkillPackageFile, ...]],
    Awaitable[list[StaticFinding]],
]
StorageFactory = Callable[[str], SkillStorage]
CacheRefresher = Callable[[str], Awaitable[None]]


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _get_lock(
    user_id: str,
    name: str,
) -> asyncio.Lock:
    key = (user_id, name)
    lock = _skill_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _skill_locks[key] = lock
    return lock


async def _to_thread(
    func,
    /,
    *args,
    **kwargs,
):
    return await asyncio.to_thread(
        func,
        *args,
        **kwargs,
    )


def _raise_static_block(
    error: StaticScanBlockedError,
) -> NoReturn:
    payload = {
        "skill_name": error.skill_name,
        "findings": error.findings,
    }
    raise ValueError(f"{error} Findings: {json.dumps(payload, ensure_ascii=False)}") from error


def _raise_static_scan_failure(
    name: str,
    error: StaticScannerError,
) -> NoReturn:
    raise ValueError(f"Static security scan failed for skill '{name}': {error}") from error


async def scan_static_candidate_or_raise(
    name: str,
    updates: dict[str, str],
    skill_storage: SkillStorage | None = None,
    *,
    scanner: Callable[..., list[StaticFinding]] = enforce_static_scan,
) -> list[StaticFinding]:
    """Run native SkillScan against a staged full candidate package."""

    def _scan_candidate() -> list[StaticFinding]:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / name
            if skill_storage is None:
                skill_dir.mkdir(parents=True)
            else:
                shutil.copytree(
                    skill_storage.get_custom_skill_dir(name),
                    skill_dir,
                )
            for relative_path, content in updates.items():
                target = skill_dir / relative_path
                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                target.write_text(
                    content,
                    encoding="utf-8",
                )
            return scanner(
                skill_dir,
                skill_name=name,
            )

    try:
        return await _to_thread(_scan_candidate)
    except StaticScanBlockedError as exc:
        _raise_static_block(exc)
    except StaticScannerError as exc:
        _raise_static_scan_failure(name, exc)


async def scan_static_package_or_raise(
    name: str,
    files: tuple[SkillPackageFile, ...],
    *,
    scanner: Callable[..., list[StaticFinding]] = enforce_static_scan,
) -> list[StaticFinding]:
    """Run native SkillScan against one complete binary-safe package."""

    def _scan_package() -> list[StaticFinding]:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / name
            materialize_skill_package(
                skill_dir,
                files,
            )
            return scanner(
                skill_dir,
                skill_name=name,
            )

    try:
        return await _to_thread(_scan_package)
    except StaticScanBlockedError as exc:
        _raise_static_block(exc)
    except StaticScannerError as exc:
        _raise_static_scan_failure(name, exc)


async def scan_content_or_raise(
    content: str,
    *,
    executable: bool,
    location: str,
    static_findings: list[StaticFinding] | None = None,
) -> dict[str, Any]:
    """Run LLM moderation after native SkillScan succeeds."""
    result = await scan_skill_content(
        content,
        executable=executable,
        location=location,
        static_findings=static_findings or [],
        attach_tracing=False,
    )
    if result.decision == "block":
        raise ValueError(f"Security scan blocked the write: {result.reason}")
    if executable and result.decision != "allow":
        raise ValueError(f"Security scan rejected executable content: {result.reason}")
    return {
        "decision": result.decision,
        "reason": result.reason,
        "static_findings": static_findings or [],
    }


def _history_record(
    *,
    request: SkillMutationRequest,
    action: MutationAction,
    file_path: str,
    prev_content: str | None,
    new_content: str | None,
    scanner: dict[str, Any],
    previous_skill_hash: str | None,
    resulting_skill_hash: str | None,
) -> dict[str, Any]:
    return {
        "action": action,
        "author": request.author,
        "thread_id": request.thread_id,
        "proposal_id": request.proposal_id,
        "evaluation_id": request.evaluation_id,
        "expected_base_hash": request.expected_base_hash,
        "previous_skill_hash": previous_skill_hash,
        "resulting_skill_hash": resulting_skill_hash,
        "file_path": file_path,
        "prev_content": prev_content,
        "new_content": new_content,
        "scanner": scanner,
    }


class SkillMutationService:
    """Single reusable path for validated, scanned custom Skill writes."""

    def __init__(
        self,
        *,
        storage_factory: StorageFactory = get_or_new_user_skill_storage,
        static_candidate_scanner: StaticCandidateScanner = (scan_static_candidate_or_raise),
        package_candidate_scanner: PackageCandidateScanner = (scan_static_package_or_raise),
        content_scanner: ContentScanner = scan_content_or_raise,
        refresh_cache: CacheRefresher = (refresh_user_skills_system_prompt_cache_async),
    ) -> None:
        self._storage_factory = storage_factory
        self._static_candidate_scanner = static_candidate_scanner
        self._package_candidate_scanner = package_candidate_scanner
        self._content_scanner = content_scanner
        self._refresh_cache = refresh_cache

    async def _assert_expected_base(
        self,
        storage: SkillStorage,
        request: SkillMutationRequest,
    ) -> str | None:
        try:
            return await _to_thread(
                storage.assert_expected_base_hash,
                request.name,
                request.expected_base_hash,
            )
        except SkillStorageConflict as exc:
            raise SkillMutationConflict(str(exc)) from None

    async def _write(
        self,
        storage: SkillStorage,
        request: SkillMutationRequest,
        *,
        path: str,
        content: str,
        require_absent: bool = False,
    ) -> None:
        try:
            await _to_thread(
                storage.write_custom_skill,
                request.name,
                path,
                content,
                expected_base_hash=request.expected_base_hash,
                require_absent=require_absent,
            )
        except SkillStorageConflict as exc:
            raise SkillMutationConflict(str(exc)) from None

    async def _append_history(
        self,
        storage: SkillStorage,
        request: SkillMutationRequest,
        *,
        action: MutationAction,
        path: str,
        prev_content: str | None,
        new_content: str | None,
        scanner: dict[str, Any],
        previous_skill_hash: str | None,
        resulting_skill_hash: str | None,
    ) -> None:
        await _to_thread(
            storage.append_history,
            request.name,
            _history_record(
                request=request,
                action=action,
                file_path=path,
                prev_content=prev_content,
                new_content=new_content,
                scanner=scanner,
                previous_skill_hash=previous_skill_hash,
                resulting_skill_hash=resulting_skill_hash,
            ),
        )

    async def mutate(
        self,
        request: SkillMutationRequest,
    ) -> SkillMutationResult:
        name = SkillStorage.validate_skill_name(request.name)
        if name != request.name:
            request = replace(
                request,
                name=name,
            )
        if request.action not in {
            "create",
            "patch",
            "edit",
            "delete",
            "write_file",
            "remove_file",
        }:
            storage = self._storage_factory(request.user_id)
            if await _to_thread(
                storage.public_skill_exists,
                name,
            ):
                raise ValueError(f"'{name}' is a read-only skill (built-in or legacy shared). To customise it, create your own version with the same name.")
            raise ValueError(f"Unsupported action '{request.action}'.")
        action: MutationAction = request.action  # type: ignore[assignment]
        storage = self._storage_factory(request.user_id)
        lock = _get_lock(
            request.user_id,
            name,
        )
        async with lock:
            result = await self._mutate_locked(
                storage,
                request,
                action,
            )
            await self._refresh_cache(request.user_id)
            return result

    async def replace_package(
        self,
        request: SkillPackageMutationRequest,
    ) -> SkillPackageMutationResult:
        name = SkillStorage.validate_skill_name(request.name)
        storage = self._storage_factory(request.user_id)
        lock = _get_lock(
            request.user_id,
            name,
        )
        async with lock:
            result = await self._replace_package_locked(
                storage,
                request,
                name,
            )
            await self._refresh_cache(request.user_id)
            return result

    async def validate_package(
        self,
        *,
        user_id: str,
        name: str,
        files: tuple[SkillPackageFile, ...],
        moderation_paths: tuple[str, ...],
    ) -> dict[str, Any]:
        """Run the complete package security boundary without mutating storage."""
        normalized_name = SkillStorage.validate_skill_name(name)
        storage = self._storage_factory(user_id)
        static_findings, scan_records, resulting_skill_hash = await self._validate_package_candidate(
            storage,
            normalized_name,
            files,
            moderation_paths,
        )
        return {
            "decision": "allow",
            "static_findings": static_findings,
            "files": scan_records,
            "resulting_skill_hash": resulting_skill_hash,
            "resulting_package_hash": compute_skill_package_hash(files),
        }

    async def _validate_package_candidate(
        self,
        storage: SkillStorage,
        name: str,
        files: tuple[SkillPackageFile, ...],
        moderation_paths: tuple[str, ...],
    ) -> tuple[list[StaticFinding], list[dict[str, Any]], str | None]:
        if not files:
            return [], [], None
        skill_file = next(
            (item for item in files if item.path == SKILL_MD_FILE),
            None,
        )
        if skill_file is None:
            raise ValueError("Skill package mutation requires SKILL.md")
        try:
            skill_content = skill_file.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("SKILL.md must be valid UTF-8") from exc
        await _to_thread(
            storage.validate_skill_markdown_content,
            name,
            skill_content,
        )
        static_findings = await self._package_candidate_scanner(
            name,
            files,
        )
        required_moderation_paths = {
            *moderation_paths,
            SKILL_MD_FILE,
            *(item.path for item in files if item.executable),
        }
        file_paths = {item.path for item in files}
        if not required_moderation_paths <= file_paths:
            raise ValueError("moderation paths must exist in the Skill package")
        scan_records: list[dict[str, Any]] = []
        for item in files:
            if item.path not in required_moderation_paths:
                continue
            try:
                content = item.content.decode("utf-8")
            except UnicodeDecodeError:
                if item.executable:
                    raise ValueError(f"Executable package file '{item.path}' must be valid UTF-8") from None
                continue
            scan = await self._content_scanner(
                content,
                executable=item.executable,
                location=f"{name}/{item.path}",
                static_findings=static_findings,
            )
            scan_records.append(
                {
                    "path": item.path,
                    **scan,
                }
            )
        return (
            static_findings,
            scan_records,
            _sha256_bytes(skill_file.content),
        )

    async def _replace_package_locked(
        self,
        storage: SkillStorage,
        request: SkillPackageMutationRequest,
        name: str,
    ) -> SkillPackageMutationResult:
        static_findings, scan_records, resulting_skill_hash = await self._validate_package_candidate(
            storage,
            name,
            request.files,
            request.moderation_paths,
        )

        resulting_package_hash = compute_skill_package_hash(
            request.files,
        )
        history_record = {
            "action": request.action,
            "author": "evolution",
            "thread_id": None,
            "proposal_id": request.proposal_id,
            "evaluation_id": request.evaluation_id,
            "publication_id": request.publication_id,
            "actor_id": request.actor_id,
            "expected_base_hash": request.expected_base_hash,
            "expected_package_hash": request.expected_package_hash,
            "previous_skill_hash": request.expected_base_hash,
            "previous_package_hash": request.expected_package_hash,
            "resulting_skill_hash": resulting_skill_hash,
            "resulting_package_hash": resulting_package_hash,
            "file_path": "<skill-package>",
            "prev_content": None,
            "new_content": None,
            "scanner": {
                "decision": "allow",
                "static_findings": static_findings,
                "files": scan_records,
            },
        }
        try:
            await _to_thread(
                storage.replace_custom_skill_package,
                name,
                request.files,
                expected_base_hash=request.expected_base_hash,
                expected_package_hash=request.expected_package_hash,
                require_absent=request.require_absent,
                history_record=history_record,
            )
        except SkillStorageConflict as exc:
            raise SkillMutationConflict(str(exc)) from None
        return SkillPackageMutationResult(
            action=request.action,
            name=name,
            previous_skill_hash=request.expected_base_hash,
            previous_package_hash=request.expected_package_hash,
            resulting_skill_hash=resulting_skill_hash,
            resulting_package_hash=resulting_package_hash,
        )

    async def _mutate_locked(
        self,
        storage: SkillStorage,
        request: SkillMutationRequest,
        action: MutationAction,
    ) -> SkillMutationResult:
        name = request.name
        if action == "create":
            if request.expected_base_hash is not None:
                raise ValueError("create does not accept expected_base_hash")
            if await _to_thread(
                storage.custom_skill_exists,
                name,
            ):
                raise ValueError(f"Custom skill '{name}' already exists.")
            if request.content is None:
                raise ValueError("content is required for create.")
            await _to_thread(
                storage.validate_skill_markdown_content,
                name,
                request.content,
            )
            static_findings = await self._static_candidate_scanner(
                name,
                {SKILL_MD_FILE: request.content},
                None,
            )
            scan = await self._content_scanner(
                request.content,
                executable=False,
                location=f"{name}/{SKILL_MD_FILE}",
                static_findings=static_findings,
            )
            resulting_hash = _sha256(request.content)
            await self._write(
                storage,
                request,
                path=SKILL_MD_FILE,
                content=request.content,
                require_absent=True,
            )
            await self._append_history(
                storage,
                request,
                action=action,
                path=SKILL_MD_FILE,
                prev_content=None,
                new_content=request.content,
                scanner=scan,
                previous_skill_hash=None,
                resulting_skill_hash=resulting_hash,
            )
            return SkillMutationResult(
                action=action,
                name=name,
                file_path=SKILL_MD_FILE,
                message=f"Created custom skill '{name}'.",
                previous_skill_hash=None,
                resulting_skill_hash=resulting_hash,
            )

        await _to_thread(
            storage.ensure_custom_skill_is_editable,
            name,
        )
        previous_hash = await self._assert_expected_base(
            storage,
            request,
        )

        if action == "edit":
            if request.content is None:
                raise ValueError("content is required for edit.")
            await _to_thread(
                storage.validate_skill_markdown_content,
                name,
                request.content,
            )
            static_findings = await self._static_candidate_scanner(
                name,
                {SKILL_MD_FILE: request.content},
                storage,
            )
            scan = await self._content_scanner(
                request.content,
                executable=False,
                location=f"{name}/{SKILL_MD_FILE}",
                static_findings=static_findings,
            )
            prev_content = await _to_thread(
                storage.read_custom_skill,
                name,
            )
            resulting_hash = _sha256(request.content)
            await self._write(
                storage,
                request,
                path=SKILL_MD_FILE,
                content=request.content,
            )
            await self._append_history(
                storage,
                request,
                action=action,
                path=SKILL_MD_FILE,
                prev_content=prev_content,
                new_content=request.content,
                scanner=scan,
                previous_skill_hash=previous_hash,
                resulting_skill_hash=resulting_hash,
            )
            return SkillMutationResult(
                action=action,
                name=name,
                file_path=SKILL_MD_FILE,
                message=f"Updated custom skill '{name}'.",
                previous_skill_hash=previous_hash,
                resulting_skill_hash=resulting_hash,
            )

        if action == "patch":
            if request.find is None or request.replace is None:
                raise ValueError("find and replace are required for patch.")
            prev_content = await _to_thread(
                storage.read_custom_skill,
                name,
            )
            occurrences = prev_content.count(request.find)
            if occurrences == 0:
                raise ValueError("Patch target not found in SKILL.md.")
            if request.expected_count is not None and occurrences != request.expected_count:
                raise ValueError(f"Expected {request.expected_count} replacements but found {occurrences}.")
            replacement_count = request.expected_count if request.expected_count is not None else 1
            new_content = prev_content.replace(
                request.find,
                request.replace,
                replacement_count,
            )
            await _to_thread(
                storage.validate_skill_markdown_content,
                name,
                new_content,
            )
            static_findings = await self._static_candidate_scanner(
                name,
                {SKILL_MD_FILE: new_content},
                storage,
            )
            scan = await self._content_scanner(
                new_content,
                executable=False,
                location=f"{name}/{SKILL_MD_FILE}",
                static_findings=static_findings,
            )
            resulting_hash = _sha256(new_content)
            await self._write(
                storage,
                request,
                path=SKILL_MD_FILE,
                content=new_content,
            )
            await self._append_history(
                storage,
                request,
                action=action,
                path=SKILL_MD_FILE,
                prev_content=prev_content,
                new_content=new_content,
                scanner=scan,
                previous_skill_hash=previous_hash,
                resulting_skill_hash=resulting_hash,
            )
            message = f"Patched custom skill '{name}' ({replacement_count} replacement(s) applied, {occurrences} match(es) found)."
            return SkillMutationResult(
                action=action,
                name=name,
                file_path=SKILL_MD_FILE,
                message=message,
                previous_skill_hash=previous_hash,
                resulting_skill_hash=resulting_hash,
            )

        if action == "delete":
            scanner = {
                "decision": "allow",
                "reason": "Deletion requested.",
            }
            history = _history_record(
                request=request,
                action=action,
                file_path=SKILL_MD_FILE,
                prev_content=None,
                new_content=None,
                scanner=scanner,
                previous_skill_hash=previous_hash,
                resulting_skill_hash=None,
            )
            try:
                await _to_thread(
                    storage.delete_custom_skill,
                    name,
                    history_meta=history,
                    expected_base_hash=(request.expected_base_hash),
                )
            except SkillStorageConflict as exc:
                raise SkillMutationConflict(str(exc)) from None
            return SkillMutationResult(
                action=action,
                name=name,
                file_path=SKILL_MD_FILE,
                message=f"Deleted custom skill '{name}'.",
                previous_skill_hash=previous_hash,
                resulting_skill_hash=None,
            )

        if action == "write_file":
            if request.path is None or request.content is None:
                raise ValueError("path and content are required for write_file.")
            target = await _to_thread(
                storage.ensure_safe_support_path,
                name,
                request.path,
            )
            exists = await _to_thread(target.exists)
            prev_content = (
                await _to_thread(
                    target.read_text,
                    encoding="utf-8",
                )
                if exists
                else None
            )
            executable = "scripts/" in request.path or request.path.startswith("scripts/")
            static_findings = await self._static_candidate_scanner(
                name,
                {request.path: request.content},
                storage,
            )
            scan = await self._content_scanner(
                request.content,
                executable=executable,
                location=f"{name}/{request.path}",
                static_findings=static_findings,
            )
            await self._write(
                storage,
                request,
                path=request.path,
                content=request.content,
            )
            await self._append_history(
                storage,
                request,
                action=action,
                path=request.path,
                prev_content=prev_content,
                new_content=request.content,
                scanner=scan,
                previous_skill_hash=previous_hash,
                resulting_skill_hash=previous_hash,
            )
            return SkillMutationResult(
                action=action,
                name=name,
                file_path=request.path,
                message=(f"Wrote '{request.path}' for custom skill '{name}'."),
                previous_skill_hash=previous_hash,
                resulting_skill_hash=previous_hash,
            )

        if request.path is None:
            raise ValueError("path is required for remove_file.")
        try:
            prev_content = await _to_thread(
                storage.remove_custom_skill_file,
                name,
                request.path,
                expected_base_hash=request.expected_base_hash,
            )
        except SkillStorageConflict as exc:
            raise SkillMutationConflict(str(exc)) from None
        scanner = {
            "decision": "allow",
            "reason": "Deletion requested.",
        }
        await self._append_history(
            storage,
            request,
            action=action,
            path=request.path,
            prev_content=prev_content,
            new_content=None,
            scanner=scanner,
            previous_skill_hash=previous_hash,
            resulting_skill_hash=previous_hash,
        )
        return SkillMutationResult(
            action=action,
            name=name,
            file_path=request.path,
            message=(f"Removed '{request.path}' from custom skill '{name}'."),
            previous_skill_hash=previous_hash,
            resulting_skill_hash=previous_hash,
        )
