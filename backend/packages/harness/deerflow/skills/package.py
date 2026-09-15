"""Binary-safe helpers for complete custom-Skill packages."""

from __future__ import annotations

import hashlib
import os
import posixpath
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_SKILL_PACKAGE_FILES = 256
MAX_SKILL_PACKAGE_FILE_BYTES = 8 * 1024 * 1024
MAX_SKILL_PACKAGE_TOTAL_BYTES = 32 * 1024 * 1024


def normalize_package_path(path: str) -> str:
    raw = path.replace("\\", "/").strip()
    if not raw:
        raise ValueError("package path must not be empty")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise ValueError("package path must be relative")
    normalized = posixpath.normpath(raw)
    if normalized in {"", "."}:
        raise ValueError("package path must identify a file")
    if len(normalized) > 1_024:
        raise ValueError("package path is too long")
    if any(part in {"", ".."} for part in PurePosixPath(normalized).parts):
        raise ValueError("package path must not contain parent-directory traversal")
    return normalized


@dataclass(frozen=True, slots=True)
class SkillPackageFile:
    path: str
    content: bytes
    executable: bool = False

    def __post_init__(self) -> None:
        normalized = normalize_package_path(self.path)
        if normalized != self.path.replace("\\", "/"):
            raise ValueError("package path must already be normalized")
        if len(self.content) > MAX_SKILL_PACKAGE_FILE_BYTES:
            raise ValueError(f"package file '{self.path}' exceeds the size limit")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def compute_skill_package_hash(
    files: tuple[SkillPackageFile, ...] | list[SkillPackageFile],
) -> str:
    records: list[bytes] = []
    seen: set[str] = set()
    for item in files:
        if item.path in seen:
            raise ValueError("package file paths must be unique")
        seen.add(item.path)
        record = b"\0".join(
            [
                item.path.encode("utf-8"),
                str(len(item.content)).encode("ascii"),
                item.content_hash.encode("ascii"),
                b"1" if item.executable else b"0",
            ]
        )
        records.append(record)
    digest = hashlib.sha256()
    for record in sorted(records):
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    return digest.hexdigest()


def read_skill_package(
    root: Path,
    *,
    allow_absent: bool = False,
) -> tuple[SkillPackageFile, ...]:
    if root.is_symlink():
        raise ValueError("Skill package root must not be a symlink")
    if not root.exists():
        if allow_absent:
            return ()
        raise FileNotFoundError(f"Skill package '{root.name}' was not found.")
    if not root.is_dir():
        raise ValueError("Skill package root must be a directory")

    files: list[SkillPackageFile] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Skill package contains a symlink: {path.relative_to(root).as_posix()}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Skill package contains an unsupported entry: {path.relative_to(root).as_posix()}")
        if len(files) >= MAX_SKILL_PACKAGE_FILES:
            raise ValueError("Skill package exceeds the file-count limit")
        relative_path = normalize_package_path(path.relative_to(root).as_posix())
        size = path.stat().st_size
        if size > MAX_SKILL_PACKAGE_FILE_BYTES:
            raise ValueError(f"package file '{relative_path}' exceeds the size limit")
        total_bytes += size
        if total_bytes > MAX_SKILL_PACKAGE_TOTAL_BYTES:
            raise ValueError("Skill package exceeds the total size limit")
        content = path.read_bytes()
        if len(content) != size:
            raise ValueError(f"package file '{relative_path}' changed while being read")
        mode = stat.S_IMODE(path.stat().st_mode)
        files.append(
            SkillPackageFile(
                path=relative_path,
                content=content,
                executable=bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
            )
        )
    if len(files) > MAX_SKILL_PACKAGE_FILES:
        raise ValueError("Skill package exceeds the file-count limit")
    if "SKILL.md" not in {item.path for item in files}:
        raise ValueError("Skill package requires SKILL.md")
    return tuple(files)


def materialize_skill_package(
    root: Path,
    files: tuple[SkillPackageFile, ...] | list[SkillPackageFile],
) -> None:
    if root.exists():
        raise FileExistsError(f"Package staging path already exists: {root}")
    if not files:
        raise ValueError("A materialized Skill package requires files")
    if "SKILL.md" not in {item.path for item in files}:
        raise ValueError("Skill package requires SKILL.md")
    if sum(len(item.content) for item in files) > MAX_SKILL_PACKAGE_TOTAL_BYTES:
        raise ValueError("Skill package exceeds the total size limit")

    root.mkdir(parents=True)
    seen: set[str] = set()
    for item in sorted(
        files,
        key=lambda value: value.path,
    ):
        if item.path in seen:
            raise ValueError("package file paths must be unique")
        seen.add(item.path)
        target = root.joinpath(*PurePosixPath(item.path).parts)
        resolved_parent = target.parent.resolve()
        resolved_parent.relative_to(root.resolve())
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        target.write_bytes(item.content)
        os.chmod(
            target,
            0o755 if item.executable else 0o644,
        )
