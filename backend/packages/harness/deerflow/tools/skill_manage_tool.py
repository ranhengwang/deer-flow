"""Tool for creating and evolving custom skills."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, NoReturn

from langchain.tools import tool

from deerflow.agents.lead_agent.prompt import refresh_user_skills_system_prompt_cache_async
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.skills.mutation import (
    SkillMutationRequest,
    SkillMutationService,
)
from deerflow.skills.security_scanner import scan_skill_content
from deerflow.skills.security_static_scanner import (
    StaticFinding,
    StaticScanBlockedError,
    StaticScannerError,
    enforce_static_scan,
)
from deerflow.skills.storage import get_or_new_user_skill_storage
from deerflow.skills.storage.skill_storage import SkillStorage
from deerflow.tools.sync import make_sync_tool_wrapper
from deerflow.tools.types import Runtime


def _get_thread_id(runtime: Runtime | None) -> str | None:
    if runtime is None:
        return None
    if runtime.context and runtime.context.get("thread_id"):
        return runtime.context.get("thread_id")
    return runtime.config.get("configurable", {}).get("thread_id")


async def _scan_or_raise(content: str, *, executable: bool, location: str, static_findings: list[StaticFinding] | None = None) -> dict[str, Any]:
    # In-graph: the graph root already attached tracing (see the INVARIANT in
    # agents/lead_agent/agent.py), so the scan model must not attach it again.
    result = await scan_skill_content(content, executable=executable, location=location, static_findings=static_findings or [], attach_tracing=False)
    if result.decision == "block":
        raise ValueError(f"Security scan blocked the write: {result.reason}")
    if executable and result.decision != "allow":
        raise ValueError(f"Security scan rejected executable content: {result.reason}")
    return {
        "decision": result.decision,
        "reason": result.reason,
        "static_findings": static_findings or [],
    }


def _raise_static_block(error: StaticScanBlockedError) -> NoReturn:
    payload = {
        "skill_name": error.skill_name,
        "findings": error.findings,
    }
    raise ValueError(f"{error} Findings: {json.dumps(payload, ensure_ascii=False)}") from error


def _raise_static_scan_failure(name: str, error: StaticScannerError) -> NoReturn:
    raise ValueError(f"Static security scan failed for skill '{name}': {error}") from error


async def _scan_static_candidate_or_raise(name: str, updates: dict[str, str], skill_storage: SkillStorage | None = None) -> list[StaticFinding]:
    def _scan_candidate() -> list[StaticFinding]:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / name
            if skill_storage is None:
                skill_dir.mkdir(parents=True)
            else:
                shutil.copytree(skill_storage.get_custom_skill_dir(name), skill_dir)
            for relative_path, content in updates.items():
                target = skill_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            return enforce_static_scan(skill_dir, skill_name=name)

    try:
        return await _to_thread(_scan_candidate)
    except StaticScanBlockedError as e:
        _raise_static_block(e)
    except StaticScannerError as e:
        _raise_static_scan_failure(name, e)


async def _to_thread(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def _skill_manage_impl(
    runtime: Runtime,
    action: str,
    name: str,
    content: str | None = None,
    path: str | None = None,
    find: str | None = None,
    replace: str | None = None,
    expected_count: int | None = None,
) -> str:
    """Manage custom skills under skills/custom/.

    Args:
        action: One of create, patch, edit, delete, write_file, remove_file.
        name: Skill name in hyphen-case.
        content: New file content for create, edit, or write_file.
        path: Supporting file path for write_file or remove_file.
        find: Existing text to replace for patch.
        replace: Replacement text for patch.
        expected_count: Optional expected number of replacements for patch.
    """
    user_id = resolve_runtime_user_id(runtime)
    service = SkillMutationService(
        storage_factory=get_or_new_user_skill_storage,
        static_candidate_scanner=(_scan_static_candidate_or_raise),
        content_scanner=_scan_or_raise,
        refresh_cache=(refresh_user_skills_system_prompt_cache_async),
    )
    result = await service.mutate(
        SkillMutationRequest(
            user_id=user_id,
            action=action,
            name=name,
            content=content,
            path=path,
            find=find,
            replace=replace,
            expected_count=expected_count,
            author="agent",
            thread_id=_get_thread_id(runtime),
        )
    )
    return result.message


@tool("skill_manage", parse_docstring=True)
async def skill_manage_tool(
    runtime: Runtime,
    action: str,
    name: str,
    content: str | None = None,
    path: str | None = None,
    find: str | None = None,
    replace: str | None = None,
    expected_count: int | None = None,
) -> str:
    """Manage custom skills under skills/custom/.

    Args:
        action: One of create, patch, edit, delete, write_file, remove_file.
        name: Skill name in hyphen-case.
        content: New file content for create, edit, or write_file.
        path: Supporting file path for write_file or remove_file.
        find: Existing text to replace for patch.
        replace: Replacement text for patch.
        expected_count: Optional expected number of replacements for patch.
    """
    return await _skill_manage_impl(
        runtime=runtime,
        action=action,
        name=name,
        content=content,
        path=path,
        find=find,
        replace=replace,
        expected_count=expected_count,
    )


skill_manage_tool.func = make_sync_tool_wrapper(_skill_manage_impl, "skill_manage")
