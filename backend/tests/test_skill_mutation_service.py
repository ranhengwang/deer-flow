from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from deerflow.config.paths import Paths
from deerflow.skills.mutation import (
    SkillMutationConflict,
    SkillMutationRequest,
    SkillMutationService,
    SkillPackageMutationRequest,
)
from deerflow.skills.package import (
    SkillPackageFile,
    compute_skill_package_hash,
    read_skill_package,
)
from deerflow.skills.storage.user_scoped_skill_storage import (
    UserScopedSkillStorage,
)


def _skill_content(
    name: str,
    description: str = "Demo skill",
) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ServiceHarness:
    def __init__(
        self,
        storage: UserScopedSkillStorage,
    ) -> None:
        self.storage = storage
        self.static_calls: list[tuple[str, dict[str, str], bool]] = []
        self.package_static_calls: list[tuple[str, tuple[str, ...]]] = []
        self.content_calls: list[tuple[str, bool, str]] = []
        self.refresh_calls: list[str] = []
        self.on_content_scan = None

    async def static_scan(
        self,
        name: str,
        updates: dict[str, str],
        storage: UserScopedSkillStorage | None,
    ) -> list:
        self.static_calls.append(
            (
                name,
                updates,
                storage is not None,
            )
        )
        return []

    async def content_scan(
        self,
        content: str,
        *,
        executable: bool,
        location: str,
        static_findings: list,
    ) -> dict:
        self.content_calls.append(
            (
                content,
                executable,
                location,
            )
        )
        if self.on_content_scan is not None:
            self.on_content_scan()
        return {
            "decision": "allow",
            "reason": "ok",
            "static_findings": static_findings,
        }

    async def package_static_scan(
        self,
        name: str,
        files: tuple[SkillPackageFile, ...],
    ) -> list:
        self.package_static_calls.append(
            (
                name,
                tuple(item.path for item in files),
            )
        )
        return []

    async def refresh(self, user_id: str) -> None:
        history = self.storage.read_history("demo-skill")
        assert history
        self.refresh_calls.append(user_id)

    def service(self) -> SkillMutationService:
        return SkillMutationService(
            storage_factory=lambda user_id: self.storage,
            static_candidate_scanner=self.static_scan,
            package_candidate_scanner=self.package_static_scan,
            content_scanner=self.content_scan,
            refresh_cache=self.refresh,
        )


@pytest.fixture
def service_harness(
    monkeypatch,
    tmp_path: Path,
) -> ServiceHarness:
    paths = Paths(base_dir=tmp_path)
    monkeypatch.setattr(
        "deerflow.config.paths.get_paths",
        lambda: paths,
    )
    storage = UserScopedSkillStorage(
        "user-1",
        host_path=str(tmp_path / "skills"),
    )
    return ServiceHarness(storage)


@pytest.mark.asyncio
async def test_service_create_and_patch_preserve_security_history_and_refresh(
    service_harness: ServiceHarness,
) -> None:
    service = service_harness.service()
    initial = _skill_content("demo-skill")

    created = await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
            author="evolution",
            thread_id="thread-1",
            proposal_id="proposal-1",
            evaluation_id="evaluation-1",
        )
    )
    patched_content = initial.replace(
        "Demo skill",
        "Patched skill",
    )
    patched = await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="patch",
            name="demo-skill",
            find="Demo skill",
            replace="Patched skill",
            expected_count=1,
            expected_base_hash=_sha256(initial),
            author="evolution",
            thread_id="thread-1",
            proposal_id="proposal-1",
            evaluation_id="evaluation-1",
        )
    )

    assert created.message == "Created custom skill 'demo-skill'."
    assert created.previous_skill_hash is None
    assert created.resulting_skill_hash == _sha256(initial)
    assert "Patched custom skill" in patched.message
    assert patched.previous_skill_hash == _sha256(initial)
    assert patched.resulting_skill_hash == _sha256(patched_content)
    assert service_harness.storage.read_custom_skill("demo-skill") == patched_content
    assert len(service_harness.static_calls) == 2
    assert len(service_harness.content_calls) == 2
    assert service_harness.refresh_calls == [
        "user-1",
        "user-1",
    ]

    history = service_harness.storage.read_history("demo-skill")
    record = history[-1]
    assert record["action"] == "patch"
    assert record["author"] == "evolution"
    assert record["thread_id"] == "thread-1"
    assert record["proposal_id"] == "proposal-1"
    assert record["evaluation_id"] == "evaluation-1"
    assert record["expected_base_hash"] == _sha256(initial)
    assert record["previous_skill_hash"] == _sha256(initial)
    assert record["resulting_skill_hash"] == _sha256(patched_content)
    assert record["scanner"]["decision"] == "allow"


@pytest.mark.asyncio
async def test_stale_base_hash_fails_before_scanners_or_write(
    service_harness: ServiceHarness,
) -> None:
    service = service_harness.service()
    initial = _skill_content("demo-skill")
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
        )
    )
    service_harness.static_calls.clear()
    service_harness.content_calls.clear()
    service_harness.refresh_calls.clear()
    history_count = len(service_harness.storage.read_history("demo-skill"))

    with pytest.raises(
        SkillMutationConflict,
        match="base hash",
    ):
        await service.mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="edit",
                name="demo-skill",
                content=_skill_content(
                    "demo-skill",
                    "new",
                ),
                expected_base_hash="0" * 64,
                proposal_id="proposal-1",
                evaluation_id="evaluation-1",
                author="evolution",
            )
        )

    assert service_harness.storage.read_custom_skill("demo-skill") == initial
    assert service_harness.static_calls == []
    assert service_harness.content_calls == []
    assert service_harness.refresh_calls == []
    assert len(service_harness.storage.read_history("demo-skill")) == history_count


@pytest.mark.asyncio
async def test_atomic_cas_catches_drift_during_security_scan(
    service_harness: ServiceHarness,
) -> None:
    service = service_harness.service()
    initial = _skill_content("demo-skill")
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
        )
    )
    competing = _skill_content(
        "demo-skill",
        "competing update",
    )

    def _concurrent_write() -> None:
        service_harness.storage.write_custom_skill(
            "demo-skill",
            "SKILL.md",
            competing,
        )

    service_harness.on_content_scan = _concurrent_write
    history_count = len(service_harness.storage.read_history("demo-skill"))

    with pytest.raises(
        SkillMutationConflict,
        match="base hash",
    ):
        await service.mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="edit",
                name="demo-skill",
                content=_skill_content(
                    "demo-skill",
                    "candidate update",
                ),
                expected_base_hash=_sha256(initial),
                proposal_id="proposal-1",
                evaluation_id="evaluation-1",
                author="evolution",
            )
        )

    assert service_harness.storage.read_custom_skill("demo-skill") == competing
    assert len(service_harness.storage.read_history("demo-skill")) == history_count
    assert service_harness.refresh_calls == ["user-1"]


@pytest.mark.asyncio
async def test_create_cas_does_not_overwrite_concurrent_create(
    service_harness: ServiceHarness,
) -> None:
    service = service_harness.service()
    competing = _skill_content(
        "demo-skill",
        "competing create",
    )

    def _concurrent_create() -> None:
        service_harness.storage.write_custom_skill(
            "demo-skill",
            "SKILL.md",
            competing,
        )

    service_harness.on_content_scan = _concurrent_create

    with pytest.raises(
        SkillMutationConflict,
        match="already exists",
    ):
        await service.mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="create",
                name="demo-skill",
                content=_skill_content(
                    "demo-skill",
                    "candidate create",
                ),
            )
        )

    assert service_harness.storage.read_custom_skill("demo-skill") == competing
    assert service_harness.storage.read_history("demo-skill") == []
    assert service_harness.refresh_calls == []


@pytest.mark.asyncio
async def test_static_scan_failure_stops_llm_write_history_and_refresh(
    service_harness: ServiceHarness,
) -> None:
    async def _blocked_static(*args, **kwargs):
        raise ValueError("static blocked")

    service = SkillMutationService(
        storage_factory=lambda user_id: service_harness.storage,
        static_candidate_scanner=_blocked_static,
        content_scanner=service_harness.content_scan,
        refresh_cache=service_harness.refresh,
    )

    with pytest.raises(ValueError, match="static blocked"):
        await service.mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="create",
                name="demo-skill",
                content=_skill_content("demo-skill"),
            )
        )

    assert service_harness.content_calls == []
    assert service_harness.refresh_calls == []
    assert not service_harness.storage.custom_skill_exists("demo-skill")


@pytest.mark.asyncio
async def test_executable_support_file_uses_both_scanners(
    service_harness: ServiceHarness,
) -> None:
    service = service_harness.service()
    initial = _skill_content("demo-skill")
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
        )
    )
    service_harness.static_calls.clear()
    service_harness.content_calls.clear()

    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="write_file",
            name="demo-skill",
            path="scripts/run.sh",
            content="#!/bin/sh\nprintf ok\n",
            expected_base_hash=_sha256(initial),
            proposal_id="proposal-1",
            evaluation_id="evaluation-1",
            author="evolution",
        )
    )

    assert service_harness.static_calls == [
        (
            "demo-skill",
            {"scripts/run.sh": "#!/bin/sh\nprintf ok\n"},
            True,
        )
    ]
    assert service_harness.content_calls[0][1] is True
    history = service_harness.storage.read_history("demo-skill")
    assert history[-1]["proposal_id"] == "proposal-1"
    assert history[-1]["evaluation_id"] == "evaluation-1"
    assert history[-1]["previous_skill_hash"] == _sha256(initial)
    assert history[-1]["resulting_skill_hash"] == _sha256(initial)


@pytest.mark.asyncio
async def test_service_rejects_support_path_traversal(
    service_harness: ServiceHarness,
) -> None:
    service = service_harness.service()
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=_skill_content("demo-skill"),
        )
    )

    with pytest.raises(
        ValueError,
        match="parent-directory traversal",
    ):
        await service.mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="write_file",
                name="demo-skill",
                path="references/../SKILL.md",
                content="overwrite",
            )
        )


@pytest.mark.asyncio
async def test_create_rejects_expected_base_hash(
    service_harness: ServiceHarness,
) -> None:
    with pytest.raises(
        ValueError,
        match="create.*expected_base_hash",
    ):
        await service_harness.service().mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="create",
                name="demo-skill",
                content=_skill_content("demo-skill"),
                expected_base_hash="0" * 64,
            )
        )


def test_request_rejects_partial_evolution_provenance() -> None:
    with pytest.raises(
        ValueError,
        match="proposal_id and evaluation_id",
    ):
        SkillMutationRequest(
            user_id="user-1",
            action="edit",
            name="demo-skill",
            content=_skill_content("demo-skill"),
            author="evolution",
            proposal_id="proposal-1",
        )


@pytest.mark.asyncio
async def test_delete_cas_and_history_keep_evolution_provenance(
    service_harness: ServiceHarness,
) -> None:
    service = service_harness.service()
    initial = _skill_content("demo-skill")
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
        )
    )

    result = await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="delete",
            name="demo-skill",
            expected_base_hash=_sha256(initial),
            author="evolution",
            proposal_id="proposal-1",
            evaluation_id="evaluation-1",
        )
    )

    assert result.previous_skill_hash == _sha256(initial)
    assert result.resulting_skill_hash is None
    assert not service_harness.storage.custom_skill_exists("demo-skill")
    record = service_harness.storage.read_history("demo-skill")[-1]
    assert record["action"] == "delete"
    assert record["proposal_id"] == "proposal-1"
    assert record["evaluation_id"] == "evaluation-1"
    assert record["previous_skill_hash"] == _sha256(initial)
    assert record["resulting_skill_hash"] is None


@pytest.mark.asyncio
async def test_remove_file_stale_cas_preserves_support_file(
    service_harness: ServiceHarness,
) -> None:
    service = service_harness.service()
    initial = _skill_content("demo-skill")
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
        )
    )
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="write_file",
            name="demo-skill",
            path="references/guide.md",
            content="guide",
        )
    )

    with pytest.raises(SkillMutationConflict):
        await service.mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="remove_file",
                name="demo-skill",
                path="references/guide.md",
                expected_base_hash="0" * 64,
                author="evolution",
                proposal_id="proposal-1",
                evaluation_id="evaluation-1",
            )
        )

    assert (service_harness.storage.get_custom_skill_dir("demo-skill") / "references" / "guide.md").read_text(encoding="utf-8") == "guide"


@pytest.mark.asyncio
async def test_same_user_skill_mutations_remain_serialized(
    service_harness: ServiceHarness,
) -> None:
    initial = _skill_content("demo-skill")
    await service_harness.service().mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
        )
    )
    active = 0
    max_active = 0

    async def _slow_scan(
        content: str,
        *,
        executable: bool,
        location: str,
        static_findings: list,
    ) -> dict:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return {
            "decision": "allow",
            "reason": "ok",
            "static_findings": static_findings,
        }

    service = SkillMutationService(
        storage_factory=lambda user_id: service_harness.storage,
        static_candidate_scanner=(service_harness.static_scan),
        content_scanner=_slow_scan,
        refresh_cache=service_harness.refresh,
    )

    await asyncio.gather(
        service.mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="edit",
                name="demo-skill",
                content=_skill_content(
                    "demo-skill",
                    "first",
                ),
            )
        ),
        service.mutate(
            SkillMutationRequest(
                user_id="user-1",
                action="edit",
                name="demo-skill",
                content=_skill_content(
                    "demo-skill",
                    "second",
                ),
            )
        ),
    )

    assert max_active == 1


@pytest.mark.asyncio
async def test_package_history_failure_restores_original_package(
    service_harness: ServiceHarness,
    monkeypatch,
) -> None:
    service = service_harness.service()
    initial = _skill_content("demo-skill")
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
        )
    )
    root = service_harness.storage.get_custom_skill_dir("demo-skill")
    guide = root / "references" / "guide.md"
    guide.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    guide.write_text(
        "original guide",
        encoding="utf-8",
    )
    original_files = read_skill_package(root)
    original_hash = compute_skill_package_hash(original_files)
    candidate = (
        SkillPackageFile(
            path="SKILL.md",
            content=_skill_content(
                "demo-skill",
                "published",
            ).encode(),
        ),
        SkillPackageFile(
            path="references/new.md",
            content=b"candidate",
        ),
    )
    append_history = service_harness.storage.append_history

    def _fail_publication_history(name: str, record: dict) -> None:
        if record.get("action") == "evolution_publish":
            raise OSError("history unavailable")
        append_history(
            name,
            record,
        )

    monkeypatch.setattr(
        service_harness.storage,
        "append_history",
        _fail_publication_history,
    )

    with pytest.raises(
        OSError,
        match="history unavailable",
    ):
        await service.replace_package(
            SkillPackageMutationRequest(
                user_id="user-1",
                action="evolution_publish",
                name="demo-skill",
                files=candidate,
                moderation_paths=(
                    "SKILL.md",
                    "references/new.md",
                ),
                expected_base_hash=_sha256(initial),
                expected_package_hash=original_hash,
                require_absent=False,
                proposal_id="proposal-1",
                evaluation_id="evaluation-1",
                publication_id="publication-1",
            )
        )

    restored_files = read_skill_package(root)
    assert compute_skill_package_hash(restored_files) == original_hash
    assert (root / "SKILL.md").read_text(encoding="utf-8") == initial
    assert guide.read_text(encoding="utf-8") == "original guide"
    assert not (root / "references" / "new.md").exists()
    assert service_harness.refresh_calls == ["user-1"]


@pytest.mark.asyncio
async def test_package_permission_failure_after_swap_restores_original_package(
    service_harness: ServiceHarness,
    monkeypatch,
) -> None:
    service = service_harness.service()
    initial = _skill_content("demo-skill")
    await service.mutate(
        SkillMutationRequest(
            user_id="user-1",
            action="create",
            name="demo-skill",
            content=initial,
        )
    )
    root = service_harness.storage.get_custom_skill_dir("demo-skill")
    original_files = read_skill_package(root)
    original_hash = compute_skill_package_hash(original_files)

    def _fail_permissions(path: Path) -> None:
        raise OSError("permission repair failed")

    monkeypatch.setattr(
        "deerflow.skills.storage.local_skill_storage.make_skill_tree_sandbox_readable",
        _fail_permissions,
    )

    with pytest.raises(
        OSError,
        match="permission repair failed",
    ):
        await service.replace_package(
            SkillPackageMutationRequest(
                user_id="user-1",
                action="evolution_publish",
                name="demo-skill",
                files=(
                    SkillPackageFile(
                        path="SKILL.md",
                        content=_skill_content(
                            "demo-skill",
                            "published",
                        ).encode(),
                    ),
                ),
                moderation_paths=("SKILL.md",),
                expected_base_hash=_sha256(initial),
                expected_package_hash=original_hash,
                require_absent=False,
                proposal_id="proposal-1",
                evaluation_id="evaluation-1",
                publication_id="publication-1",
            )
        )

    restored_files = read_skill_package(root)
    assert compute_skill_package_hash(restored_files) == original_hash
    assert (root / "SKILL.md").read_text(encoding="utf-8") == initial
    assert service_harness.storage.read_history("demo-skill")[-1]["action"] == "create"
