"""Production-shaped deterministic ReplayTask suite for Phase 11."""

from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from deerflow.skill_evolution.evaluator import (
    Replayability,
    ReplayArtifactVerifier,
    ReplayCommandVerifier,
    ReplayEnvironmentRequirements,
    ReplaySideEffectPolicy,
    ReplaySkillPackage,
    ReplayTaskSpec,
    build_replay_fixture_snapshot,
    build_replay_skill_package,
)
from deerflow.skill_evolution.experiment import (
    BenchmarkTask,
    ExperimentManifest,
)

REPLAY_SUITE_VERSION = "skill-evolution-replay-suite-v1"
_CREATED_AT = datetime(2026, 8, 18, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class MaterializedReplayCase:
    benchmark_task: BenchmarkTask
    spec: ReplayTaskSpec
    candidate_skill: ReplaySkillPackage
    base_skill: ReplaySkillPackage | None
    reusable_lesson: str
    skill_gap: str | None
    recovered_error_evidence: bool


@dataclass(frozen=True, slots=True)
class _TaskPayload:
    input_files: dict[str, bytes]
    expected: Any
    instructions: str
    skill_guidance: str
    base_guidance: str


def _json_file(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8")


def _normalize_safe_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    for raw in paths:
        normalized = posixpath.normpath(raw.replace("\\", "/"))
        if normalized.startswith("/") or normalized in {"", ".", ".."} or any(part == ".." for part in normalized.split("/")):
            continue
        if normalized not in result:
            result.append(normalized)
    return sorted(result)


def _payload_repo_preflight(variant: int) -> _TaskPayload:
    checks = [
        {
            "name": f"lint-{variant}",
            "enabled": True,
            "priority": 20,
        },
        {
            "name": "unit-tests",
            "enabled": True,
            "priority": 10,
        },
        {
            "name": f"legacy-{variant}",
            "enabled": False,
            "priority": 1,
        },
    ]
    expected = {
        "enabled_checks": [
            item["name"]
            for item in sorted(
                (item for item in checks if item["enabled"]),
                key=lambda item: (
                    item["priority"],
                    item["name"],
                ),
            )
        ]
    }
    return _TaskPayload(
        input_files={"input.json": _json_file({"checks": checks})},
        expected=expected,
        instructions=('Read input.json. Keep enabled checks only, order by ascending priority then name, and write only their name strings as {"enabled_checks": ["name", ...]}.'),
        skill_guidance=("Filter disabled checks before sorting. Sort by numeric priority then lexical name. Emit only the ordered name strings under enabled_checks."),
        base_guidance=("Sort every check alphabetically and include disabled checks."),
    )


def _payload_repo_paths(variant: int) -> _TaskPayload:
    if variant <= 3:
        paths = [
            "assets/logo.png",
            "docs/guide.md",
            f"src/module-{variant}.py",
        ]
    else:
        paths = [
            f"src/./module-{variant}.py",
            "docs//guide.md",
            "../secret.txt",
            "/etc/passwd",
            f"src/module-{variant}.py",
            "assets/../assets/logo.png",
        ]
    return _TaskPayload(
        input_files={"input.json": _json_file({"paths": paths})},
        expected={"safe_paths": _normalize_safe_paths(paths)},
        instructions=('Read input.json. Normalize slash-separated relative paths, reject absolute or parent-escaping paths, deduplicate, sort the final strings in ascending lexical order, and write exactly {"safe_paths": [...]}.'),
        skill_guidance=("Normalize backslashes to slashes and apply POSIX normpath. Reject absolute paths and any remaining '..' segment. Deduplicate, then sort the final path strings in ascending lexical order."),
        base_guidance=("Remove literal '../' text and accept the resulting path."),
    )


def _payload_repo_retry(variant: int) -> _TaskPayload:
    attempts = [
        {"status": "error", "code": f"transient-{variant}"},
        {"status": "error", "code": "retryable"},
        {"status": "ok", "value": f"result-{variant}"},
        {"status": "ok", "value": "late-result"},
    ]
    expected = {
        "attempts": 3,
        "result": f"result-{variant}",
    }
    return _TaskPayload(
        input_files={"input.json": _json_file({"attempts": attempts})},
        expected=expected,
        instructions=('Read input.json. Select the first successful attempt and report its one-based attempt count and value as {"attempts": N, "result": value}.'),
        skill_guidance=("Scan in source order and stop at the first status='ok'. The attempt count is one-based and later successes must be ignored."),
        base_guidance="Use the final attempt regardless of earlier success.",
    )


def _payload_repo_config(variant: int) -> _TaskPayload:
    legacy = {
        "service": f"worker-{variant}",
        "retries": variant + 1,
        "timeout_ms": 1500 + variant * 500,
    }
    expected = {
        "retry": {"max_attempts": legacy["retries"]},
        "service": {"name": legacy["service"]},
        "timeout_seconds": legacy["timeout_ms"] / 1000,
        "version": 2,
    }
    return _TaskPayload(
        input_files={"input.json": _json_file(legacy)},
        expected=expected,
        instructions=('Migrate input.json to version 2 and write exactly {"version": 2, "service": {"name": SERVICE}, "retry": {"max_attempts": RETRIES}, "timeout_seconds": TIMEOUT_MS / 1000}. Preserve the values from input.json.'),
        skill_guidance=("Use the exact top-level keys version, service, retry, timeout_seconds. service must be {'name': original service}; retry must be {'max_attempts': original retries}; divide timeout_ms by exactly 1000."),
        base_guidance=("Rename timeout_ms to timeout_seconds without unit conversion."),
    )


def _payload_csv(variant: int) -> _TaskPayload:
    rows = [
        (f" Alpha {variant} ", str(variant * 10)),
        ("beta", f"{variant * 10 + 2}.0"),
        (" GAMMA ", str(variant * 10 + 1)),
    ]
    csv = "name,amount\n" + "".join(f"{name},{amount}\n" for name, amount in rows)
    expected = {
        "rows": sorted(
            [
                {
                    "amount": int(float(amount)),
                    "name": name.strip().lower(),
                }
                for name, amount in rows
            ],
            key=lambda item: item["name"],
        )
    }
    return _TaskPayload(
        input_files={"input.csv": csv.encode("utf-8")},
        expected=expected,
        instructions=(
            "Read input.csv with the standard library. Trim only surrounding "
            "whitespace and lowercase each complete name, preserving digits "
            "and internal spaces. Convert numeric amounts to integers, sort by "
            'the complete normalized name, and write {"rows": [...]}.'
        ),
        skill_guidance=("Use csv.DictReader. For name, apply only strip().lower(); never remove numeric suffixes or internal spaces. Parse amount through float then int, and sort normalized records by the complete name."),
        base_guidance="Keep original name whitespace and CSV row order.",
    )


def _payload_aggregation(variant: int) -> _TaskPayload:
    records = [
        {"category": "b", "value": variant},
        {"category": "a", "value": variant + 1},
        {"category": "b", "value": variant + 2},
    ]
    expected = {
        "totals": {
            "a": variant + 1,
            "b": variant * 2 + 2,
        }
    }
    return _TaskPayload(
        input_files={"input.json": _json_file({"records": records})},
        expected=expected,
        instructions=('Sum input.json records by category and write {"totals": {category: sum}} with categories sorted.'),
        skill_guidance=("Accumulate integer values in a dictionary, then construct totals in sorted category order."),
        base_guidance="Keep only the last value for each category.",
    )


def _payload_schema(variant: int) -> _TaskPayload:
    if variant <= 3:
        rows = [
            {"id": f"ok-{variant}", "value": variant},
            {"id": f"ok2-{variant}", "value": variant + 2},
        ]
        expected = {
            "rejected_ids": [],
            "valid_ids": [
                f"ok-{variant}",
                f"ok2-{variant}",
            ],
        }
    else:
        rows = [
            {"id": f"ok-{variant}", "value": variant},
            {"id": f"bad-negative-{variant}", "value": -1},
            {"id": "", "value": variant + 1},
            {"id": f"ok2-{variant}", "value": variant + 2},
        ]
        expected = {
            "rejected_ids": [
                f"bad-negative-{variant}",
                "<missing>",
            ],
            "valid_ids": [
                f"ok-{variant}",
                f"ok2-{variant}",
            ],
        }
    return _TaskPayload(
        input_files={"input.json": _json_file({"rows": rows})},
        expected=expected,
        instructions=(
            "Validate rows from input.json. A valid row has a non-empty string "
            "id and a non-negative integer value. Preserve source order. In "
            "rejected_ids, replace every blank or missing id with the exact "
            "literal '<missing>'; never emit an empty string. Write exactly "
            "valid_ids and rejected_ids."
        ),
        skill_guidance=("Validate both ID shape and value type/range. Keep source order. Map every blank or missing rejected ID to the exact string '<missing>', never ''."),
        base_guidance="Accept every row whose value key exists.",
    )


def _payload_dedup(variant: int) -> _TaskPayload:
    records = [
        {"key": "alpha", "version": 1, "value": "old"},
        {
            "key": "beta",
            "version": variant + 1,
            "value": f"beta-{variant}",
        },
        {
            "key": "alpha",
            "version": variant + 2,
            "value": f"alpha-{variant}",
        },
    ]
    expected = {
        "records": [
            {
                "key": "alpha",
                "value": f"alpha-{variant}",
                "version": variant + 2,
            },
            {
                "key": "beta",
                "value": f"beta-{variant}",
                "version": variant + 1,
            },
        ]
    }
    return _TaskPayload(
        input_files={"input.json": _json_file({"records": records})},
        expected=expected,
        instructions=('Deduplicate input.json records by key, keeping the greatest numeric version, then sort by key and write {"records": [...]}.'),
        skill_guidance=("Compare numeric version per key, replace only with a greater version, and sort final records lexically by key."),
        base_guidance="Keep the first record seen for each key.",
    )


def _payload_bootstrap(variant: int) -> _TaskPayload:
    available = ["python"] if variant <= 3 else (["uv", "python"] if variant % 2 else ["python", "pip"])
    manager = "uv" if "uv" in available else "pip"
    expected = {
        "manager": manager,
        "steps": (
            ["uv venv", "uv pip install -r requirements.txt"]
            if manager == "uv"
            else [
                "python -m venv .venv",
                "python -m pip install -r requirements.txt",
            ]
        ),
    }
    return _TaskPayload(
        input_files={"input.json": _json_file({"available": available})},
        expected=expected,
        instructions=("Read input.json and plan, but do not execute, any bootstrap command. Choose uv when available, otherwise pip. Emit manager and the portable bootstrap steps for requirements.txt."),
        skill_guidance=("This is a planning task: never execute the listed commands. Prefer uv: ['uv venv','uv pip install -r requirements.txt']. Fallback to ['python -m venv .venv','python -m pip install -r requirements.txt']."),
        base_guidance="Always run sudo pip install globally.",
    )


def _payload_archive(variant: int) -> _TaskPayload:
    entries = [
        {
            "path": f"data/file-{variant}.txt",
            "checksum_ok": True,
        },
        {"path": "../escape.txt", "checksum_ok": True},
        {"path": "data/bad.bin", "checksum_ok": False},
        {"path": "docs/readme.md", "checksum_ok": True},
    ]
    expected = {
        "accepted": [
            f"data/file-{variant}.txt",
            "docs/readme.md",
        ],
        "rejected": ["../escape.txt", "data/bad.bin"],
    }
    return _TaskPayload(
        input_files={"input.json": _json_file({"entries": entries})},
        expected=expected,
        instructions=(
            "Read input.json and classify every archive entry in source order. "
            "Accept only relative paths with no '..' segment and "
            "checksum_ok=true. Write exactly {'accepted': [accepted path "
            "strings], 'rejected': [rejected path strings]}; do not inspect "
            "the real filesystem."
        ),
        skill_guidance=("Read entries from input.json, not the filesystem. Reject absolute/traversal paths before checksum handling; then reject checksum failures. Preserve source order in both path-string lists."),
        base_guidance="Accept every entry whose checksum is valid.",
    )


def _payload_pipeline(variant: int) -> _TaskPayload:
    markers = (
        "zero",
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
    )
    marker = markers[variant] if 0 <= variant < len(markers) else f"variant {variant}"
    stages = (
        [
            f"extract {marker}",
            "transform",
            "verify",
        ]
        if variant <= 3
        else [
            f" Extract {marker} ",
            "TRANSFORM",
            " verify ",
        ]
    )
    expected = {
        "artifact": "::".join(stage.strip().lower() for stage in stages),
        "stage_count": len(stages),
    }
    return _TaskPayload(
        input_files={"input.json": _json_file({"stages": stages})},
        expected=expected,
        instructions=(
            "Read input.json. Trim only surrounding whitespace and lowercase each complete stage, preserving digits and internal spaces. Join in source order with the exact delimiter '::', and write exactly artifact plus stage_count."
        ),
        skill_guidance=(
            "Preserve stage order and all non-whitespace characters. Apply "
            "strip().lower() to every full stage before joining with exactly "
            "'::'. The final artifact must equal artifact.lower(); uppercase "
            "letters are always an error. Count all stages."
        ),
        base_guidance="Sort stages alphabetically and join with '-'.",
    )


def _payload_fallback(variant: int) -> _TaskPayload:
    available = ["python", "sed"] if variant <= 3 else (["sed", "python"] if variant % 2 else ["awk", "python"])
    preference = ["jq", "python", "awk", "sed"]
    selected = next(item for item in preference if item in available)
    return _TaskPayload(
        input_files={
            "input.json": _json_file(
                {
                    "available": available,
                    "preference": preference,
                }
            )
        },
        expected={
            "selected": selected,
            "used_fallback": selected != preference[0],
        },
        instructions=("Read input.json. Choose the first string from preference that also appears in available. Write exactly {'selected': STRING, 'used_fallback': BOOLEAN}, where used_fallback is true when selected is not preference[0]."),
        skill_guidance=("Iterate preference order, not available order. Select the first exact string intersection. Use only the keys selected and used_fallback; set used_fallback from the selected preference index."),
        base_guidance="Choose the first item in available.",
    )


_PAYLOAD_BUILDERS = {
    "repo-preflight": _payload_repo_preflight,
    "repo-path-normalization": _payload_repo_paths,
    "repo-async-retry": _payload_repo_retry,
    "repo-config-migration": _payload_repo_config,
    "data-csv-normalization": _payload_csv,
    "data-json-aggregation": _payload_aggregation,
    "data-schema-validation": _payload_schema,
    "data-deduplication": _payload_dedup,
    "shell-portable-bootstrap": _payload_bootstrap,
    "shell-archive-verification": _payload_archive,
    "shell-artifact-pipeline": _payload_pipeline,
    "shell-command-fallback": _payload_fallback,
}


def _verifier_command(expected: Any) -> str:
    expected_json = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "python - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        f"expected=json.loads({expected_json!r})\n"
        "path=Path('/outputs/result.json')\n"
        "if not path.is_file(): raise SystemExit(2)\n"
        "try: actual=json.loads(path.read_text('utf-8'))\n"
        "except Exception: raise SystemExit(3)\n"
        "raise SystemExit(0 if actual == expected else 4)\n"
        "PY"
    )


def _skill_package(
    task: BenchmarkTask,
    payload: _TaskPayload,
    *,
    candidate: bool,
) -> ReplaySkillPackage:
    guidance = payload.skill_guidance if candidate else payload.base_guidance
    label = "Verified" if candidate else "Legacy"
    content = (
        "---\n"
        f"name: {task.family_id}\n"
        f"description: {label} workflow for {task.family_id} benchmark tasks.\n"
        "---\n\n"
        f"# {label} {task.family_id}\n\n"
        "When active, solve the current task using only /workspace inputs and "
        "write the requested JSON to /outputs/result.json.\n\n"
        f"{guidance}\n\n"
        "Always emit valid JSON through write_json. Do not read or modify other "
        "Skill packages. Read the fixture once and write the final result once.\n"
    )
    return build_replay_skill_package(
        user_id="benchmark-user",
        skill_name=task.family_id,
        files={"SKILL.md": content},
    )


def materialize_replay_case(
    task: BenchmarkTask,
) -> MaterializedReplayCase:
    try:
        payload = _PAYLOAD_BUILDERS[task.family_id](task.variant_index)
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark family {task.family_id!r}") from exc
    task_input = f"Replay case {task.task_id}. {payload.instructions} Work only inside the isolated replay workspace. The input is under /workspace and the final file must be /outputs/result.json. Do not use network access."
    fixture = build_replay_fixture_snapshot(payload.input_files)
    source_hash = hashlib.sha256(f"{REPLAY_SUITE_VERSION}:{task.task_id}".encode()).hexdigest()
    spec = ReplayTaskSpec(
        task_id=f"replay-{task.task_id}",
        source_event_id=f"event-{task.task_id}",
        source_run_id=f"run-{task.task_id}",
        source_snapshot_hash=source_hash,
        user_id="benchmark-user",
        task_family=task.family_id,
        task_input=task_input,
        fixture=fixture,
        environment=ReplayEnvironmentRequirements(
            os="linux",
            shell="sh",
            runtime="python3.12",
            required_commands=["python"],
            network_required=False,
        ),
        verifiers=[
            ReplayCommandVerifier(
                command=_verifier_command(payload.expected),
                expected_exit_code=0,
                timeout_seconds=30,
            ),
            ReplayArtifactVerifier(
                path="outputs/result.json",
                must_exist=True,
            ),
        ],
        timeout_seconds=180,
        side_effect_policy=ReplaySideEffectPolicy(
            writable_roots=["workspace", "outputs"],
            network_allowed=False,
            skill_library_writes_allowed=False,
            max_changed_files=32,
            max_written_bytes=1024 * 1024,
        ),
        replayability=Replayability.automatic,
        manual_review_reasons=[],
        created_at=_CREATED_AT,
    )
    return MaterializedReplayCase(
        benchmark_task=task,
        spec=spec,
        candidate_skill=_skill_package(
            task,
            payload,
            candidate=True,
        ),
        base_skill=(_skill_package(task, payload, candidate=False) if task.branch == "patch" else None),
        reusable_lesson=payload.instructions,
        skill_gap=(f"Legacy guidance was unsafe or incomplete: {payload.base_guidance} The verified task requirement was: {payload.instructions}" if task.branch == "patch" else None),
        recovered_error_evidence=(task.split == "evidence" and task.variant_index == 2),
    )


def materialize_replay_suite(
    manifest: ExperimentManifest,
) -> list[MaterializedReplayCase]:
    cases = [materialize_replay_case(task) for task in manifest.tasks]
    if len({case.spec.task_id for case in cases}) != len(cases):
        raise ValueError("materialized replay task IDs must be unique")
    return cases
