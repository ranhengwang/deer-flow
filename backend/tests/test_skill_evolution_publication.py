from __future__ import annotations

import hashlib
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from deerflow.config.paths import Paths
from deerflow.skill_evolution.models import (
    EvaluationDecision,
    EvaluationMetrics,
    ProposalOperation,
    ProposalStatus,
    ProposedSkillFile,
    SkillEvaluation,
    SkillPatchOperation,
    SkillProposal,
    TaskEvaluationResult,
)
from deerflow.skill_evolution.observability import (
    EvolutionLifecycleKind,
    EvolutionObservability,
)
from deerflow.skill_evolution.publication import (
    PublicationStatus,
    SkillPublicationService,
    capture_skill_package_snapshot,
)
from deerflow.skill_evolution.store.memory import (
    InMemorySkillEvolutionStore,
)
from deerflow.skills.mutation import SkillMutationService
from deerflow.skills.storage.user_scoped_skill_storage import (
    UserScopedSkillStorage,
)

_NOW = datetime(2026, 8, 17, tzinfo=UTC)
_BASE_SKILL = """---
name: fixture-transform
description: Transform fixtures.
---

# Fixture Transform

Use the legacy workflow.
"""
_PATCHED_SKILL = _BASE_SKILL.replace(
    "legacy workflow",
    "portable workflow",
)
_CREATE_SKILL = """---
name: generated-helper
description: Run the generated helper.
---

# Generated Helper
"""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _CreditRecorder:
    def __init__(self) -> None:
        self.publications = []
        self.rollbacks = []

    async def record_publication(
        self,
        proposal,
        publication,
    ):
        self.publications.append((proposal, publication))

    async def record_rollback(
        self,
        publication,
        *,
        rolled_back_at,
    ):
        self.rollbacks.append((publication, rolled_back_at))


class PublicationHarness:
    def __init__(
        self,
        storage: UserScopedSkillStorage,
    ) -> None:
        self.storage = storage
        self.package_scan_calls: list[tuple[str, tuple[str, ...]]] = []
        self.content_scan_calls: list[tuple[str, bool, str]] = []
        self.refresh_calls: list[str] = []
        self.on_package_scan = None
        self.block_package_scan_after: int | None = None

    async def package_scan(
        self,
        name,
        files,
    ) -> list:
        paths = tuple(item.path for item in files)
        self.package_scan_calls.append(
            (
                name,
                paths,
            )
        )
        if self.on_package_scan is not None:
            self.on_package_scan(
                len(self.package_scan_calls),
            )
        if self.block_package_scan_after is not None and len(self.package_scan_calls) >= self.block_package_scan_after:
            raise ValueError("package scan blocked")
        return []

    async def content_scan(
        self,
        content: str,
        *,
        executable: bool,
        location: str,
        static_findings: list,
    ) -> dict:
        self.content_scan_calls.append(
            (
                content,
                executable,
                location,
            )
        )
        return {
            "decision": "allow",
            "reason": "ok",
            "static_findings": static_findings,
        }

    async def refresh(self, user_id: str) -> None:
        self.refresh_calls.append(user_id)

    def mutation_service(self) -> SkillMutationService:
        return SkillMutationService(
            storage_factory=lambda user_id: self.storage,
            package_candidate_scanner=self.package_scan,
            content_scanner=self.content_scan,
            refresh_cache=self.refresh,
        )

    def publication_service(
        self,
        store: InMemorySkillEvolutionStore,
        *,
        observability: EvolutionObservability | None = None,
        credit_recorder=None,
    ) -> SkillPublicationService:
        return SkillPublicationService(
            store=store,
            storage_factory=lambda user_id: self.storage,
            mutation_service=self.mutation_service(),
            observability=observability,
            credit_recorder=credit_recorder,
        )


@pytest.fixture
def publication_harness(
    monkeypatch,
    tmp_path: Path,
) -> PublicationHarness:
    paths = Paths(base_dir=tmp_path)
    monkeypatch.setattr(
        "deerflow.config.paths.get_paths",
        lambda: paths,
    )
    storage = UserScopedSkillStorage(
        "user-1",
        host_path=str(tmp_path / "skills"),
    )
    return PublicationHarness(storage)


def _write_base_package(
    storage: UserScopedSkillStorage,
) -> None:
    storage.write_custom_skill(
        "fixture-transform",
        "SKILL.md",
        _BASE_SKILL,
        require_absent=True,
    )
    root = storage.get_custom_skill_dir("fixture-transform")
    script = root / "scripts" / "legacy.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes(b"#!/bin/sh\nprintf legacy\n")
    script.chmod(0o755)
    asset = root / "assets" / "fixture.bin"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"\x00\xfffixture")


def _patch_proposal() -> SkillProposal:
    return SkillProposal(
        proposal_id="proposal-patch-1",
        cluster_id="cluster-patch-1",
        user_id="user-1",
        operation=ProposalOperation.patch,
        skill_name="fixture-transform",
        base_skill_hash=_sha256(_BASE_SKILL),
        proposed_files=[
            ProposedSkillFile(
                path="SKILL.md",
                content=_PATCHED_SKILL,
                executable=False,
            ),
            ProposedSkillFile(
                path="references/new.md",
                content="Use the portable workflow.\n",
                executable=False,
            ),
        ],
        supporting_event_ids=[
            "event-1",
            "event-2",
            "event-3",
        ],
        rationale="Replace the legacy workflow.",
        expected_improvements=["Work across supported systems."],
        patch_operations=[
            SkillPatchOperation(
                find="legacy workflow",
                replace="portable workflow",
                reason="The portable workflow passed evaluation.",
                supporting_event_ids=[
                    "event-1",
                    "event-2",
                    "event-3",
                ],
            )
        ],
        source_skill_hashes=[_sha256(_BASE_SKILL)],
        status=ProposalStatus.approved,
        created_at=_NOW - timedelta(days=1),
        expires_at=_NOW + timedelta(days=30),
    )


def _create_proposal() -> SkillProposal:
    return SkillProposal(
        proposal_id="proposal-create-1",
        cluster_id="cluster-create-1",
        user_id="user-1",
        operation=ProposalOperation.create,
        skill_name="generated-helper",
        proposed_files=[
            ProposedSkillFile(
                path="SKILL.md",
                content=_CREATE_SKILL,
                executable=False,
            ),
            ProposedSkillFile(
                path="scripts/run.sh",
                content="#!/bin/sh\nprintf generated\n",
                executable=True,
            ),
        ],
        supporting_event_ids=[
            "event-1",
            "event-2",
            "event-3",
        ],
        rationale="Create the evaluated helper.",
        expected_improvements=["Reuse the verified helper."],
        status=ProposalStatus.approved,
        created_at=_NOW - timedelta(days=1),
        expires_at=_NOW + timedelta(days=30),
    )


def _evaluation(
    proposal: SkillProposal,
) -> SkillEvaluation:
    result = TaskEvaluationResult(
        task_id="held-out-1",
        split="held_out",
        condition="candidate_skill",
        success=True,
        metrics=EvaluationMetrics(
            tool_calls=2,
            input_tokens=100,
            output_tokens=20,
            latency_seconds=1.0,
        ),
    )
    return SkillEvaluation(
        evaluation_id=f"evaluation-{proposal.proposal_id}",
        proposal_id=proposal.proposal_id,
        user_id=proposal.user_id,
        source_replay_results=[],
        held_out_results=[result],
        baseline_results=[],
        candidate_results=[result],
        regression_results=[],
        safety_results={
            "held_out": "1/1",
            "side_effects": "allow",
        },
        quality_score=0.8,
        decision=EvaluationDecision.approve,
        created_at=_NOW - timedelta(hours=1),
    )


async def _persist_candidate(
    store: InMemorySkillEvolutionStore,
    proposal: SkillProposal,
) -> SkillEvaluation:
    evaluation = _evaluation(proposal)
    await store.put_proposal(proposal)
    await store.put_evaluation(evaluation)
    return evaluation


@pytest.mark.asyncio
async def test_snapshot_is_binary_safe_complete_and_rejects_symlinks(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)

    snapshot = await capture_skill_package_snapshot(
        publication_harness.storage,
        user_id="user-1",
        skill_name="fixture-transform",
        created_at=_NOW,
    )

    assert snapshot.exists is True
    assert snapshot.skill_md_hash == _sha256(_BASE_SKILL)
    assert [item.path for item in snapshot.files] == [
        "SKILL.md",
        "assets/fixture.bin",
        "scripts/legacy.sh",
    ]
    binary = next(item for item in snapshot.files if item.path == "assets/fixture.bin")
    assert binary.content_bytes == b"\x00\xfffixture"
    script = next(item for item in snapshot.files if item.path == "scripts/legacy.sh")
    assert script.executable is True

    symlink = publication_harness.storage.get_custom_skill_dir("fixture-transform") / "references" / "outside"
    symlink.parent.mkdir(parents=True, exist_ok=True)
    symlink.symlink_to("/etc/hosts")
    with pytest.raises(ValueError, match="symlink"):
        await capture_skill_package_snapshot(
            publication_harness.storage,
            user_id="user-1",
            skill_name="fixture-transform",
            created_at=_NOW,
        )

    broken_root = publication_harness.storage.get_custom_skill_dir("broken-skill")
    broken_root.symlink_to(broken_root.parent / "missing-target")
    with pytest.raises(ValueError, match="root must not be a symlink"):
        await capture_skill_package_snapshot(
            publication_harness.storage,
            user_id="user-1",
            skill_name="broken-skill",
            created_at=_NOW,
            allow_absent=True,
        )


@pytest.mark.asyncio
async def test_patch_publication_and_rollback_restore_exact_package(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)
    store = InMemorySkillEvolutionStore()
    proposal = _patch_proposal()
    evaluation = await _persist_candidate(
        store,
        proposal,
    )
    observer = EvolutionObservability()
    credit_recorder = _CreditRecorder()
    service = publication_harness.publication_service(
        store,
        observability=observer,
        credit_recorder=credit_recorder,
    )

    published = await service.publish(
        user_id="user-1",
        proposal_id=proposal.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        now=_NOW,
    )

    assert published.changed is True
    assert published.publication.status is PublicationStatus.published
    assert published.publication.base_skill_hash == _sha256(_BASE_SKILL)
    proposed_skill_content = next(item.content for item in proposal.proposed_files if item.path == "SKILL.md")
    assert published.publication.published_skill_hash == _sha256(proposed_skill_content)
    assert published.publication.base_snapshot.package_hash != published.publication.published_snapshot.package_hash
    assert published.proposal.status is ProposalStatus.published
    assert published.proposal.status_history[-1].publication_id == published.publication.publication_id
    root = publication_harness.storage.get_custom_skill_dir("fixture-transform")
    assert (root / "SKILL.md").read_text(encoding="utf-8") == proposed_skill_content
    assert (root / "assets" / "fixture.bin").read_bytes() == b"\x00\xfffixture"
    assert (root / "scripts" / "legacy.sh").read_bytes() == b"#!/bin/sh\nprintf legacy\n"
    proposed_reference = next(item.content for item in proposal.proposed_files if item.path == "references/new.md")
    assert (root / "references" / "new.md").read_text(encoding="utf-8") == proposed_reference
    assert publication_harness.storage.read_history("fixture-transform")[-1]["action"] == "evolution_publish"

    rolled_back = await service.rollback(
        user_id="user-1",
        publication_id=published.publication.publication_id,
        actor_id="reviewer-1",
        now=_NOW + timedelta(hours=1),
    )

    assert rolled_back.changed is True
    assert rolled_back.publication.status is PublicationStatus.rolled_back
    assert rolled_back.proposal.status is ProposalStatus.rolled_back
    assert rolled_back.proposal.status_history[-1].publication_id == published.publication.publication_id
    assert rolled_back.publication.rollback_snapshot.package_hash == rolled_back.publication.base_snapshot.package_hash
    assert (root / "SKILL.md").read_text(encoding="utf-8") == _BASE_SKILL
    assert (root / "assets" / "fixture.bin").read_bytes() == b"\x00\xfffixture"
    assert (root / "scripts" / "legacy.sh").read_bytes() == b"#!/bin/sh\nprintf legacy\n"
    assert stat.S_IMODE((root / "scripts" / "legacy.sh").stat().st_mode) & stat.S_IXUSR
    assert not (root / "references" / "new.md").exists()
    history = publication_harness.storage.read_history("fixture-transform")
    assert history[-1]["action"] == "evolution_rollback"
    assert history[-1]["publication_id"] == published.publication.publication_id
    assert any(executable and location.endswith("scripts/legacy.sh") for _, executable, location in publication_harness.content_scan_calls)
    second_rollback = await service.rollback(
        user_id="user-1",
        publication_id=published.publication.publication_id,
        actor_id="reviewer-1",
        now=_NOW + timedelta(hours=2),
    )
    assert second_rollback.changed is False
    assert second_rollback.publication == rolled_back.publication
    assert [event.kind for event in observer.recent_events()] == [
        EvolutionLifecycleKind.published,
        EvolutionLifecycleKind.rolled_back,
    ]
    assert len(credit_recorder.publications) == 3
    assert len(credit_recorder.rollbacks) == 2


@pytest.mark.asyncio
async def test_create_publication_rollback_deletes_created_skill(
    publication_harness: PublicationHarness,
) -> None:
    store = InMemorySkillEvolutionStore()
    proposal = _create_proposal()
    evaluation = await _persist_candidate(
        store,
        proposal,
    )
    service = publication_harness.publication_service(store)

    published = await service.publish(
        user_id="user-1",
        proposal_id=proposal.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        now=_NOW,
    )
    assert publication_harness.storage.custom_skill_exists("generated-helper")
    assert published.publication.base_snapshot.exists is False

    rolled_back = await service.rollback(
        user_id="user-1",
        publication_id=published.publication.publication_id,
        actor_id="reviewer-1",
        now=_NOW + timedelta(hours=1),
    )

    assert rolled_back.publication.rollback_snapshot.exists is False
    assert not publication_harness.storage.custom_skill_exists("generated-helper")
    history = publication_harness.storage.read_history("generated-helper")
    assert history[-1]["action"] == "evolution_rollback"


@pytest.mark.asyncio
async def test_direct_publication_skips_evaluation_and_remains_rollbackable(
    publication_harness: PublicationHarness,
) -> None:
    store = InMemorySkillEvolutionStore()
    proposal = SkillProposal.model_validate(
        {
            **_create_proposal().model_dump(mode="python"),
            "status": ProposalStatus.staged,
        }
    )
    await store.put_proposal(proposal)
    credit_recorder = _CreditRecorder()
    service = publication_harness.publication_service(
        store,
        credit_recorder=credit_recorder,
    )

    published = await service.publish_direct(
        user_id=proposal.user_id,
        proposal_id=proposal.proposal_id,
        now=_NOW,
    )

    assert published.changed is True
    assert published.publication.schema_version == "deerflow.skill-evolution.publication.v2"
    assert published.publication.evaluation_id is None
    assert published.proposal.status is ProposalStatus.published
    assert (
        await store.list_evaluation_page(
            proposal.user_id,
            limit=10,
            offset=0,
        )
        == []
    )
    assert publication_harness.storage.custom_skill_exists(proposal.skill_name)
    history = publication_harness.storage.read_history(proposal.skill_name)
    assert history[-1]["action"] == "evolution_publish"
    assert history[-1]["evaluation_id"] is None

    repeated = await service.publish_direct(
        user_id=proposal.user_id,
        proposal_id=proposal.proposal_id,
        now=_NOW + timedelta(minutes=1),
    )
    assert repeated.changed is False
    assert repeated.publication == published.publication

    rolled_back = await service.rollback(
        user_id=proposal.user_id,
        publication_id=published.publication.publication_id,
        actor_id="operator-1",
        now=_NOW + timedelta(hours=1),
    )

    assert rolled_back.changed is True
    assert rolled_back.publication.evaluation_id is None
    assert rolled_back.proposal.status is ProposalStatus.rolled_back
    assert not publication_harness.storage.custom_skill_exists(proposal.skill_name)


@pytest.mark.asyncio
async def test_direct_patch_preserves_unmodified_package_files(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)
    store = InMemorySkillEvolutionStore()
    proposal = SkillProposal.model_validate(
        {
            **_patch_proposal().model_dump(mode="python"),
            "status": ProposalStatus.staged,
        }
    )
    await store.put_proposal(proposal)
    service = publication_harness.publication_service(
        store,
        credit_recorder=_CreditRecorder(),
    )

    published = await service.publish_direct(
        user_id=proposal.user_id,
        proposal_id=proposal.proposal_id,
        now=_NOW,
    )

    root = publication_harness.storage.get_custom_skill_dir(proposal.skill_name)
    proposed_skill_content = next(item.content for item in proposal.proposed_files if item.path == "SKILL.md")
    assert published.publication.evaluation_id is None
    assert published.proposal.status is ProposalStatus.published
    assert (root / "SKILL.md").read_text(encoding="utf-8") == proposed_skill_content
    assert (root / "assets" / "fixture.bin").read_bytes() == b"\x00\xfffixture"
    assert (root / "scripts" / "legacy.sh").read_bytes() == (b"#!/bin/sh\nprintf legacy\n")
    proposed_reference = next(item.content for item in proposal.proposed_files if item.path == "references/new.md")
    assert (root / "references" / "new.md").read_text(encoding="utf-8") == proposed_reference


@pytest.mark.asyncio
async def test_direct_publication_security_failure_restores_staged_proposal(
    publication_harness: PublicationHarness,
) -> None:
    store = InMemorySkillEvolutionStore()
    proposal = SkillProposal.model_validate(
        {
            **_create_proposal().model_dump(mode="python"),
            "status": ProposalStatus.staged,
        }
    )
    await store.put_proposal(proposal)
    publication_harness.block_package_scan_after = 2
    service = publication_harness.publication_service(store)

    with pytest.raises(ValueError, match="package scan blocked"):
        await service.publish_direct(
            user_id=proposal.user_id,
            proposal_id=proposal.proposal_id,
            now=_NOW,
        )

    stored = await store.get_proposal(
        proposal.user_id,
        proposal.proposal_id,
    )
    assert stored is not None
    assert stored.status is ProposalStatus.staged
    assert not publication_harness.storage.custom_skill_exists(proposal.skill_name)
    publication = await store.get_publication(
        proposal.user_id,
        f"publication-{proposal.proposal_id}",
    )
    assert publication is not None
    assert publication.status is PublicationStatus.preparing

    publication_harness.block_package_scan_after = None
    retried = await service.publish_direct(
        user_id=proposal.user_id,
        proposal_id=proposal.proposal_id,
        now=_NOW + timedelta(minutes=1),
    )
    assert retried.changed is True
    assert retried.proposal.status is ProposalStatus.published


@pytest.mark.asyncio
async def test_rollback_security_failure_preserves_published_package(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)
    store = InMemorySkillEvolutionStore()
    proposal = _patch_proposal()
    evaluation = await _persist_candidate(
        store,
        proposal,
    )
    service = publication_harness.publication_service(store)
    published = await service.publish(
        user_id="user-1",
        proposal_id=proposal.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        now=_NOW,
    )
    publication_harness.block_package_scan_after = 2

    with pytest.raises(ValueError, match="package scan blocked"):
        await service.rollback(
            user_id="user-1",
            publication_id=published.publication.publication_id,
            actor_id="reviewer-1",
            now=_NOW + timedelta(hours=1),
        )

    root = publication_harness.storage.get_custom_skill_dir("fixture-transform")
    proposed_skill_content = next(item.content for item in proposal.proposed_files if item.path == "SKILL.md")
    assert (root / "SKILL.md").read_text(encoding="utf-8") == proposed_skill_content
    record = await store.get_publication(
        "user-1",
        published.publication.publication_id,
    )
    assert record is not None
    assert record.status is PublicationStatus.published
    stored_proposal = await store.get_proposal(
        "user-1",
        proposal.proposal_id,
    )
    assert stored_proposal is not None
    assert stored_proposal.status is ProposalStatus.published
    assert publication_harness.storage.read_history("fixture-transform")[-1]["action"] == "evolution_publish"


@pytest.mark.asyncio
async def test_publication_preflight_failure_persists_no_snapshot_record(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)
    store = InMemorySkillEvolutionStore()
    proposal = _patch_proposal()
    evaluation = await _persist_candidate(
        store,
        proposal,
    )
    publication_harness.block_package_scan_after = 1
    service = publication_harness.publication_service(store)

    with pytest.raises(ValueError, match="package scan blocked"):
        await service.publish(
            user_id="user-1",
            proposal_id=proposal.proposal_id,
            evaluation_id=evaluation.evaluation_id,
            now=_NOW,
        )

    assert (
        await store.get_publication(
            "user-1",
            f"publication-{proposal.proposal_id}",
        )
        is None
    )
    stored_proposal = await store.get_proposal(
        "user-1",
        proposal.proposal_id,
    )
    assert stored_proposal is not None
    assert stored_proposal.status is ProposalStatus.approved
    assert publication_harness.storage.read_custom_skill("fixture-transform") == _BASE_SKILL


@pytest.mark.asyncio
async def test_package_drift_during_scan_is_not_overwritten(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)
    store = InMemorySkillEvolutionStore()
    proposal = _patch_proposal()
    evaluation = await _persist_candidate(
        store,
        proposal,
    )
    service = publication_harness.publication_service(store)
    root = publication_harness.storage.get_custom_skill_dir("fixture-transform")

    def _drift(call_count: int) -> None:
        if call_count == 1:
            (root / "assets" / "fixture.bin").write_bytes(b"competing")

    publication_harness.on_package_scan = _drift

    with pytest.raises(ValueError, match="package hash conflict"):
        await service.publish(
            user_id="user-1",
            proposal_id=proposal.proposal_id,
            evaluation_id=evaluation.evaluation_id,
            now=_NOW,
        )

    assert (root / "SKILL.md").read_text(encoding="utf-8") == _BASE_SKILL
    assert (root / "assets" / "fixture.bin").read_bytes() == b"competing"
    stored_proposal = await store.get_proposal(
        "user-1",
        proposal.proposal_id,
    )
    assert stored_proposal is not None
    assert stored_proposal.status is ProposalStatus.approved
    publication = await store.get_publication(
        "user-1",
        f"publication-{proposal.proposal_id}",
    )
    assert publication is not None
    assert publication.status is PublicationStatus.preparing


@pytest.mark.asyncio
async def test_publication_is_idempotent_and_user_scoped(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)
    store = InMemorySkillEvolutionStore()
    proposal = _patch_proposal()
    evaluation = await _persist_candidate(
        store,
        proposal,
    )
    service = publication_harness.publication_service(store)

    first = await service.publish(
        user_id="user-1",
        proposal_id=proposal.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        now=_NOW,
    )
    scan_count = len(publication_harness.package_scan_calls)
    second = await service.publish(
        user_id="user-1",
        proposal_id=proposal.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        now=_NOW,
    )

    assert first.publication == second.publication
    assert second.changed is False
    assert len(publication_harness.package_scan_calls) == scan_count
    with pytest.raises(ValueError, match="not found"):
        await service.rollback(
            user_id="other-user",
            publication_id=first.publication.publication_id,
            actor_id="reviewer-1",
            now=_NOW + timedelta(hours=1),
        )


@pytest.mark.asyncio
async def test_retry_repairs_proposal_status_after_publication_record_commits(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)
    first_store = InMemorySkillEvolutionStore()
    proposal = _patch_proposal()
    evaluation = await _persist_candidate(
        first_store,
        proposal,
    )
    first_service = publication_harness.publication_service(first_store)
    completed = await first_service.publish(
        user_id="user-1",
        proposal_id=proposal.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        now=_NOW,
    )

    recovery_store = InMemorySkillEvolutionStore()
    recovering_proposal = proposal.model_copy(
        update={
            "status": ProposalStatus.publishing,
        }
    )
    await recovery_store.put_proposal(recovering_proposal)
    await recovery_store.put_evaluation(evaluation)
    await recovery_store.put_publication(completed.publication)
    recovery_service = publication_harness.publication_service(recovery_store)
    scan_count = len(publication_harness.package_scan_calls)

    recovered = await recovery_service.publish(
        user_id="user-1",
        proposal_id=proposal.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        now=_NOW + timedelta(minutes=1),
    )

    assert recovered.changed is True
    assert recovered.proposal.status is ProposalStatus.published
    assert recovered.publication == completed.publication
    assert len(publication_harness.package_scan_calls) == scan_count


@pytest.mark.asyncio
async def test_publication_rejects_proposed_file_outside_safe_support_roots(
    publication_harness: PublicationHarness,
) -> None:
    _write_base_package(publication_harness.storage)
    store = InMemorySkillEvolutionStore()
    proposal = _patch_proposal().model_copy(
        update={
            "proposed_files": [
                *_patch_proposal().proposed_files,
                ProposedSkillFile(
                    path="other/unsafe.txt",
                    content="unsafe",
                    executable=False,
                ),
            ]
        }
    )
    evaluation = await _persist_candidate(
        store,
        proposal,
    )
    service = publication_harness.publication_service(store)

    with pytest.raises(
        ValueError,
        match="Supporting files must live under",
    ):
        await service.publish(
            user_id="user-1",
            proposal_id=proposal.proposal_id,
            evaluation_id=evaluation.evaluation_id,
            now=_NOW,
        )

    assert publication_harness.storage.read_custom_skill("fixture-transform") == _BASE_SKILL
    assert (
        await store.get_publication(
            "user-1",
            f"publication-{proposal.proposal_id}",
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expires_at", "message"),
    [
        (
            ProposalStatus.validating,
            _NOW + timedelta(days=1),
            "approved",
        ),
        (
            ProposalStatus.approved,
            _NOW,
            "expired",
        ),
    ],
)
async def test_unapproved_or_expired_proposal_creates_no_publication_record(
    publication_harness: PublicationHarness,
    status: ProposalStatus,
    expires_at: datetime,
    message: str,
) -> None:
    _write_base_package(publication_harness.storage)
    store = InMemorySkillEvolutionStore()
    proposal = _patch_proposal().model_copy(
        update={
            "status": status,
            "expires_at": expires_at,
        }
    )
    evaluation = await _persist_candidate(
        store,
        proposal,
    )
    service = publication_harness.publication_service(store)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        await service.publish(
            user_id="user-1",
            proposal_id=proposal.proposal_id,
            evaluation_id=evaluation.evaluation_id,
            now=_NOW,
        )

    assert (
        await store.get_publication(
            "user-1",
            f"publication-{proposal.proposal_id}",
        )
        is None
    )
