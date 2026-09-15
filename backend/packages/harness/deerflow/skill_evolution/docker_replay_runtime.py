"""Docker-isolated production ReplayRuntime for deterministic evaluations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from deerflow.config.app_config import AppConfig
from deerflow.models import create_chat_model
from deerflow.skill_evolution.evaluator import (
    ReplayAgentExecution,
    ReplayAgentRequest,
    ReplayCommandExecution,
    ReplayCommandRequest,
    ReplayWorkspacePaths,
)

logger = logging.getLogger(__name__)

DOCKER_REPLAY_RUNTIME_VERSION = "docker-replay-runtime-v1"
DEFAULT_REPLAY_IMAGE = "python:3.12-slim"
_MAX_TOOL_OUTPUT_CHARS = 20_000
_MAX_SKILL_PROMPT_CHARS = 262_144
_AGENT_TOOL_NAMES = (
    "bash",
    "read_file",
    "write_file",
    "write_json",
)


class DockerReplayUnavailable(RuntimeError):
    """Docker cannot provide the required isolated replay boundary."""


@dataclass(frozen=True, slots=True)
class _Container:
    name: str
    root: Path


def _docker_binary(explicit: str | None = None) -> str:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise DockerReplayUnavailable(f"Docker executable is unavailable at {candidate}")
    discovered = shutil.which("docker")
    if discovered:
        return discovered
    macos = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
    if macos.is_file() and os.access(macos, os.X_OK):
        return str(macos)
    raise DockerReplayUnavailable("Docker CLI is required for isolated production replay")


def _safe_container_path(
    raw: str,
    *,
    writable: bool,
) -> str:
    normalized = raw.replace("\\", "/").strip()
    pure = PurePosixPath(normalized)
    if not pure.is_absolute() or ".." in pure.parts:
        raise ValueError("replay path must be an absolute safe path")
    allowed = ("/workspace", "/outputs")
    if not writable:
        allowed = (*allowed, "/skills")
    if not any(normalized == root or normalized.startswith(f"{root}/") for root in allowed):
        raise ValueError("replay path is outside allowed mounts")
    if writable and normalized.startswith("/skills"):
        raise ValueError("replay Skill paths are read-only")
    return normalized


class DockerReplayRuntime:
    """Run each agent task inside a disposable network-disabled container."""

    def __init__(
        self,
        *,
        app_config: AppConfig,
        model_name: str,
        image: str = DEFAULT_REPLAY_IMAGE,
        docker_binary: str | None = None,
        thinking_enabled: bool = False,
        recursion_limit: int = 30,
        memory: str = "1g",
        cpus: float = 1.0,
        model_overrides: dict[str, Any] | None = None,
        allowed_agent_tools: tuple[str, ...] | None = None,
    ) -> None:
        self._app_config = app_config
        self.model_name = model_name
        self.image = image
        self.runtime_version = DOCKER_REPLAY_RUNTIME_VERSION
        self._docker = _docker_binary(docker_binary)
        self._thinking_enabled = thinking_enabled
        self._recursion_limit = recursion_limit
        self._memory = memory
        self._cpus = cpus
        self._model_overrides = dict(model_overrides or {})
        resolved_tools = _AGENT_TOOL_NAMES if allowed_agent_tools is None else allowed_agent_tools
        if not resolved_tools or len(set(resolved_tools)) != len(resolved_tools):
            raise ValueError("replay agent tools must be non-empty and unique")
        unknown_tools = sorted(set(resolved_tools) - set(_AGENT_TOOL_NAMES))
        if unknown_tools:
            raise ValueError(
                f"unknown replay agent tool: {unknown_tools[0]}",
            )
        self._allowed_agent_tools = tuple(resolved_tools)
        self._containers: dict[Path, _Container] = {}
        self._lock = threading.Lock()
        self._model = None

    @property
    def docker_binary(self) -> str:
        return self._docker

    def preflight(self) -> None:
        result = self._run_host(
            [self._docker, "info", "--format", "{{.ServerVersion}}"],
            timeout=15,
        )
        if result.returncode != 0:
            raise DockerReplayUnavailable("Docker daemon is not available for isolated replay")

    def _run_host(
        self,
        args: list[str],
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "DOCKER_HOST": os.environ.get("DOCKER_HOST", ""),
            },
        )

    @staticmethod
    def _root_key(paths: ReplayWorkspacePaths) -> Path:
        return paths.root.resolve()

    def _container_for(
        self,
        paths: ReplayWorkspacePaths,
    ) -> _Container:
        key = self._root_key(paths)
        with self._lock:
            container = self._containers.get(key)
        if container is None:
            raise DockerReplayUnavailable("replay container has not been started")
        return container

    def _start_container(
        self,
        paths: ReplayWorkspacePaths,
    ) -> _Container:
        key = self._root_key(paths)
        with self._lock:
            existing = self._containers.get(key)
        if existing is not None:
            return existing
        for path in (
            paths.workspace,
            paths.outputs,
            paths.skills,
        ):
            resolved = path.resolve()
            try:
                resolved.relative_to(key)
            except ValueError as exc:
                raise ValueError("replay mount escapes its isolated root") from exc
        name = f"deerflow-replay-{uuid.uuid4().hex[:20]}"
        args = [
            self._docker,
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--memory",
            self._memory,
            "--cpus",
            str(self._cpus),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            (f"type=bind,src={paths.workspace.resolve()},dst=/workspace"),
            "--mount",
            (f"type=bind,src={paths.outputs.resolve()},dst=/outputs"),
            "--mount",
            (f"type=bind,src={paths.skills.resolve()},dst=/skills,readonly"),
            "--workdir",
            "/workspace",
            self.image,
            "sleep",
            "infinity",
        ]
        result = self._run_host(args, timeout=180)
        if result.returncode != 0:
            raise DockerReplayUnavailable("failed to start isolated replay container")
        container = _Container(name=name, root=key)
        with self._lock:
            raced = self._containers.setdefault(key, container)
        if raced is not container:
            self._stop_container(container)
            return raced
        return container

    def _stop_container(self, container: _Container) -> None:
        self._run_host(
            [self._docker, "rm", "--force", container.name],
            timeout=30,
        )

    async def close_paths(
        self,
        paths: ReplayWorkspacePaths,
    ) -> None:
        key = self._root_key(paths)
        with self._lock:
            container = self._containers.pop(key, None)
        if container is not None:
            await asyncio.to_thread(
                self._stop_container,
                container,
            )

    async def close_all(self) -> None:
        with self._lock:
            containers = list(self._containers.values())
            self._containers.clear()
        await asyncio.gather(
            *[
                asyncio.to_thread(
                    self._stop_container,
                    container,
                )
                for container in containers
            ]
        )

    def _exec(
        self,
        container: _Container,
        command: str,
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self._run_host(
            [
                self._docker,
                "exec",
                *(["--interactive"] if input_text is not None else []),
                container.name,
                "sh",
                "-lc",
                command,
            ],
            timeout=timeout,
            input_text=input_text,
        )

    @staticmethod
    def _bounded_output(result: subprocess.CompletedProcess[str]) -> str:
        output = result.stdout
        if result.stderr:
            output += f"\nStd Error:\n{result.stderr}" if output else result.stderr
        if result.returncode != 0:
            output += f"\nExit Code: {result.returncode}"
        output = output or "(no output)"
        if len(output) > _MAX_TOOL_OUTPUT_CHARS:
            half = _MAX_TOOL_OUTPUT_CHARS // 2
            output = output[:half] + "\n...[truncated]...\n" + output[-half:]
        return output

    def _tools(self, container: _Container):
        @tool
        def bash(command: str) -> str:
            """Run one shell command inside the isolated replay container."""
            try:
                result = self._exec(
                    container,
                    command,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                return "Error: command_timeout"
            return self._bounded_output(result)

        @tool
        def read_file(path: str) -> str:
            """Read a UTF-8 file from workspace, outputs, or active skills."""
            try:
                resolved = _safe_container_path(
                    path,
                    writable=False,
                )
            except ValueError:
                return "Error: path_forbidden"
            result = self._exec(
                container,
                f"cat -- {shlex.quote(resolved)}",
                timeout=30,
            )
            return self._bounded_output(result)

        @tool
        def write_file(path: str, content: str) -> str:
            """Write one UTF-8 file under /workspace or /outputs."""
            try:
                resolved = _safe_container_path(
                    path,
                    writable=True,
                )
            except ValueError:
                return "Error: path_forbidden"
            parent = str(PurePosixPath(resolved).parent)
            command = f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(resolved)}"
            result = self._exec(
                container,
                command,
                timeout=30,
                input_text=content,
            )
            return "ok" if result.returncode == 0 else self._bounded_output(result)

        @tool
        def write_json(path: str, value: dict[str, Any]) -> str:
            """Serialize one JSON object under /workspace or /outputs."""
            try:
                resolved = _safe_container_path(
                    path,
                    writable=True,
                )
            except ValueError:
                return "Error: path_forbidden"
            parent = str(PurePosixPath(resolved).parent)
            command = f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(resolved)}"
            result = self._exec(
                container,
                command,
                timeout=30,
                input_text=(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ),
            )
            return "ok" if result.returncode == 0 else self._bounded_output(result)

        available = {
            replay_tool.name: replay_tool
            for replay_tool in [
                bash,
                read_file,
                write_file,
                write_json,
            ]
        }
        return [available[name] for name in self._allowed_agent_tools]

    def _active_skill(
        self,
        request: ReplayAgentRequest,
    ) -> tuple[str | None, str | None]:
        if request.active_skill_path is None:
            return None, None
        root = request.paths.skills.resolve()
        skill_path = request.active_skill_path.resolve()
        try:
            relative = skill_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("active Skill path escapes replay Skill root") from exc
        skill_file = skill_path / "SKILL.md"
        content = skill_file.read_text(encoding="utf-8")
        if len(content) > _MAX_SKILL_PROMPT_CHARS:
            raise ValueError("active Skill exceeds replay prompt bound")
        container_path = PurePosixPath("/skills").joinpath(*relative.parts, "SKILL.md").as_posix()
        return container_path, content

    def _model_instance(self):
        if self._model is None:
            self._model = create_chat_model(
                name=self.model_name,
                thinking_enabled=self._thinking_enabled,
                app_config=self._app_config,
                attach_tracing=False,
                model_overrides=self._model_overrides,
            )
        return self._model

    @staticmethod
    def _execution_from_messages(
        messages: list[Any],
    ) -> ReplayAgentExecution:
        tool_calls = 0
        input_tokens = 0
        output_tokens = 0
        errors: list[str] = []
        for message in messages:
            if isinstance(message, AIMessage):
                tool_calls += len(message.tool_calls or [])
                usage = message.usage_metadata or {}
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
            elif isinstance(message, ToolMessage):
                if getattr(message, "status", None) == "error":
                    errors.append("tool_error")
                elif isinstance(message.content, str) and message.content.startswith("Error:"):
                    errors.append("tool_error")
        return ReplayAgentExecution(
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            errors=list(dict.fromkeys(errors)),
        )

    async def run_agent(
        self,
        request: ReplayAgentRequest,
    ) -> ReplayAgentExecution:
        container = await asyncio.to_thread(
            self._start_container,
            request.paths,
        )
        skill_path, skill_content = await asyncio.to_thread(
            self._active_skill,
            request,
        )
        skill_block = (
            "No Skill is active. Solve from the task and fixture only."
            if skill_content is None
            else (f"The following replay Skill is explicitly active and must be used. Its read-only path is {skill_path}.\n<active_skill>\n{skill_content}\n</active_skill>")
        )
        system_prompt = (
            "You are running one deterministic evaluation task in an isolated "
            "network-disabled container. Use the provided tools for all file "
            "and command work. The only writable roots are /workspace and "
            "/outputs; /skills is read-only. Do not claim completion until the "
            "requested output exists. The final /outputs/result.json must be "
            "written with write_json by passing a JSON object as value; do not "
            "encode that object as a string or write the final file with bash "
            "or write_file. Do not ask questions.\n\n"
            f"{skill_block}"
        )
        agent = create_agent(
            model=self._model_instance(),
            tools=self._tools(container),
            system_prompt=system_prompt,
        )
        state: dict[str, Any] = {"messages": []}
        try:
            async for update in agent.astream(
                {
                    "messages": [
                        HumanMessage(
                            content=request.spec.task_input,
                        )
                    ]
                },
                config={"recursion_limit": self._recursion_limit},
                stream_mode="values",
            ):
                state = update
                request.progress.update(
                    self._execution_from_messages(
                        state.get("messages", []),
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Replay agent invocation failed: %s",
                type(exc).__name__,
            )
            partial = request.progress.snapshot()
            return ReplayAgentExecution(
                tool_calls=(partial.tool_calls if partial is not None else 0),
                input_tokens=(partial.input_tokens if partial is not None else 0),
                output_tokens=(partial.output_tokens if partial is not None else 0),
                errors=list(
                    dict.fromkeys(
                        [
                            *(partial.errors if partial is not None else []),
                            f"agent_runtime_error:{type(exc).__name__}",
                        ]
                    )
                ),
            )
        messages = state.get("messages", [])
        execution = self._execution_from_messages(messages)
        request.progress.update(execution)
        return execution

    async def run_command(
        self,
        request: ReplayCommandRequest,
    ) -> ReplayCommandExecution:
        container = self._container_for(request.paths)
        try:
            result = await asyncio.to_thread(
                self._exec,
                container,
                request.command,
                timeout=request.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ReplayCommandExecution(
                exit_code=124,
                errors=["command_timeout"],
            )
        return ReplayCommandExecution(
            exit_code=max(0, min(255, result.returncode)),
        )

    def environment_fingerprint(self) -> str:
        payload = f"{self.runtime_version}\0{self.image}\0{self.model_name}\0network:none\0rootfs:readonly\0tools:{','.join(self._allowed_agent_tools)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
