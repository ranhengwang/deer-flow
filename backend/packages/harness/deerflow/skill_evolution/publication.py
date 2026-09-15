"""Versioned publication and snapshot-based rollback for approved Skills."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from deerflow.skill_evolution.credit import CreditRecorder
from deerflow.skill_evolution.models import (
    EvaluationDecision,
    ProposalOperation,
    ProposalStatus,
    ProposalStatusSource,
    ProposalStatusTransition,
    PublicationStatus,
    SkillEvaluation,
    SkillPackageSnapshot,
    SkillPackageSnapshotFile,
    SkillProposal,
    SkillPublication,
)
from deerflow.skill_evolution.observability import (
    EvolutionLifecycleKind,
    EvolutionObservability,
    get_evolution_observability,
    safe_emit_evolution_event,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
    SkillEvolutionStore,
)
from deerflow.skills.mutation import (
    SkillMutationService,
    SkillPackageMutationRequest,
)
from deerflow.skills.package import (
    SkillPackageFile,
    compute_skill_package_hash,
    read_skill_package,
)
from deerflow.skills.storage import (
    get_or_new_user_skill_storage,
)
from deerflow.skills.storage.skill_storage import SkillStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SkillPublicationResult:
    publication: SkillPublication
    proposal: SkillProposal
    changed: bool


def _snapshot_from_files(
    *,
    user_id: str,
    skill_name: str,
    files: tuple[SkillPackageFile, ...],
    exists: bool,
    created_at: datetime,
) -> SkillPackageSnapshot:
    snapshot_files = [
        SkillPackageSnapshotFile(
            path=item.path,
            content_base64=base64.b64encode(item.content).decode("ascii"),
            content_hash=item.content_hash,
            size_bytes=len(item.content),
            executable=item.executable,
        )
        for item in sorted(
            files,
            key=lambda value: value.path,
        )
    ]
    skill_md = next(
        (item for item in files if item.path == "SKILL.md"),
        None,
    )
    return SkillPackageSnapshot(
        snapshot_hash=compute_skill_package_hash(files),
        user_id=user_id,
        skill_name=skill_name,
        exists=exists,
        skill_md_hash=(skill_md.content_hash if skill_md is not None else None),
        files=snapshot_files,
        created_at=created_at,
    )


async def capture_skill_package_snapshot(
    storage: SkillStorage,
    *,
    user_id: str,
    skill_name: str,
    created_at: datetime | None = None,
    allow_absent: bool = False,
) -> SkillPackageSnapshot:
    """Capture every package file without following symlinks."""
    root = storage.get_custom_skill_dir(skill_name)
    files = await asyncio.to_thread(
        read_skill_package,
        root,
        allow_absent=allow_absent,
    )
    return _snapshot_from_files(
        user_id=user_id,
        skill_name=skill_name,
        files=files,
        exists=bool(files),
        created_at=created_at or datetime.now(UTC),
    )


def build_candidate_package_snapshot(
    proposal: SkillProposal,
    base_snapshot: SkillPackageSnapshot,
    *,
    created_at: datetime,
) -> SkillPackageSnapshot:
    if proposal.user_id != base_snapshot.user_id or proposal.skill_name != base_snapshot.skill_name:
        raise ValueError("Proposal and base snapshot identity do not match")
    if proposal.operation is ProposalOperation.create:
        if base_snapshot.exists:
            raise ValueError("create Proposal requires an absent base Skill")
        files: dict[str, SkillPackageFile] = {}
    elif proposal.operation is ProposalOperation.patch:
        if not base_snapshot.exists:
            raise ValueError("patch Proposal requires an existing base Skill")
        if proposal.base_skill_hash != base_snapshot.skill_md_hash:
            raise ValueError("Proposal base Skill hash does not match the package snapshot")
        files = {item.path: item.to_package_file() for item in base_snapshot.files}
    else:
        raise ValueError("publication supports only create and patch Proposals")

    for item in proposal.proposed_files:
        files[item.path] = SkillPackageFile(
            path=item.path,
            content=item.content.encode("utf-8"),
            executable=item.executable,
        )
    records = tuple(files[path] for path in sorted(files))
    return _snapshot_from_files(
        user_id=proposal.user_id,
        skill_name=proposal.skill_name,
        files=records,
        exists=True,
        created_at=created_at,
    )


class SkillPublicationService:
    def __init__(
        self,
        *,
        store: SkillEvolutionStore,
        storage_factory: Callable[[str], SkillStorage] = get_or_new_user_skill_storage,
        mutation_service: SkillMutationService | None = None,
        credit_recorder: CreditRecorder | None = None,
        observability: EvolutionObservability | None = None,
    ) -> None:
        self._store = store
        self._storage_factory = storage_factory
        self._mutation_service = mutation_service or SkillMutationService(
            storage_factory=storage_factory,
        )
        self._credit_recorder = credit_recorder or CreditRecorder(store)
        self._observability = observability or get_evolution_observability()

    async def _record_publication_credit(
        self,
        proposal: SkillProposal,
        publication: SkillPublication,
    ) -> None:
        try:
            await self._credit_recorder.record_publication(
                proposal,
                publication,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to persist distillation Credit for publication %s (non-fatal)",
                publication.publication_id,
            )

    async def _record_rollback_credit(
        self,
        proposal: SkillProposal,
        publication: SkillPublication,
        *,
        rolled_back_at: datetime,
    ) -> None:
        try:
            await self._credit_recorder.record_publication(
                proposal,
                publication,
            )
            await self._credit_recorder.record_rollback(
                publication,
                rolled_back_at=rolled_back_at,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to close distillation Credit for publication %s (non-fatal)",
                publication.publication_id,
            )

    async def _load_pair(
        self,
        *,
        user_id: str,
        proposal_id: str,
        evaluation_id: str,
    ) -> tuple[SkillProposal, SkillEvaluation]:
        proposal = await self._store.get_proposal(
            user_id,
            proposal_id,
        )
        if proposal is None:
            raise ValueError("proposal was not found")
        evaluation = await self._store.get_evaluation(
            user_id,
            evaluation_id,
        )
        if evaluation is None:
            raise ValueError("evaluation was not found")
        if evaluation.proposal_id != proposal.proposal_id:
            raise ValueError("evaluation does not belong to proposal")
        if evaluation.user_id != proposal.user_id:
            raise ValueError("evaluation user does not match proposal owner")
        if evaluation.decision is EvaluationDecision.reject:
            raise ValueError("rejected evaluation cannot be published")
        return proposal, evaluation

    async def _load_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
    ) -> SkillProposal:
        proposal = await self._store.get_proposal(
            user_id,
            proposal_id,
        )
        if proposal is None:
            raise ValueError("proposal was not found")
        return proposal

    async def _transition_proposal(
        self,
        proposal: SkillProposal,
        *,
        new_status: ProposalStatus,
        reason_code: str,
        reason: str,
        occurred_at: datetime,
        source: ProposalStatusSource,
        evaluation_id: str | None,
        publication_id: str,
        actor_id: str | None = None,
    ) -> SkillProposal:
        transition = ProposalStatusTransition(
            from_status=proposal.status,
            to_status=new_status,
            source=source,
            reason_code=reason_code,
            reason=reason,
            occurred_at=occurred_at,
            evaluation_id=evaluation_id,
            publication_id=publication_id,
            actor_id=actor_id,
        )
        try:
            return await self._store.transition_proposal(
                user_id=proposal.user_id,
                proposal_id=proposal.proposal_id,
                expected_status=proposal.status,
                new_status=new_status,
                transition=transition,
            )
        except EvolutionStoreConflict:
            winner = await self._store.get_proposal(
                proposal.user_id,
                proposal.proposal_id,
            )
            if winner is not None and winner.status is new_status and winner.status_history and winner.status_history[-1] == transition:
                return winner
            raise

    async def _prepare_publication(
        self,
        proposal: SkillProposal,
        *,
        evaluation_id: str | None,
        now: datetime,
    ) -> tuple[SkillPublication, bool]:
        publication_id = f"publication-{proposal.proposal_id}"
        existing = await self._store.get_publication(
            proposal.user_id,
            publication_id,
        )
        if existing is not None:
            if existing.proposal_id != proposal.proposal_id or existing.evaluation_id != evaluation_id:
                raise EvolutionStoreConflict("publication identity does not match Proposal and evaluation mode")
            return existing, False

        storage = self._storage_factory(proposal.user_id)
        for item in proposal.proposed_files:
            if item.path != "SKILL.md":
                await asyncio.to_thread(
                    storage.ensure_safe_support_path,
                    proposal.skill_name,
                    item.path,
                )
        base_snapshot = await capture_skill_package_snapshot(
            storage,
            user_id=proposal.user_id,
            skill_name=proposal.skill_name,
            created_at=now,
            allow_absent=(proposal.operation is ProposalOperation.create),
        )
        candidate_snapshot = build_candidate_package_snapshot(
            proposal,
            base_snapshot,
            created_at=now,
        )
        await self._mutation_service.validate_package(
            user_id=proposal.user_id,
            name=proposal.skill_name,
            files=candidate_snapshot.package_files(),
            moderation_paths=tuple(item.path for item in proposal.proposed_files),
        )
        publication = SkillPublication(
            publication_id=publication_id,
            user_id=proposal.user_id,
            proposal_id=proposal.proposal_id,
            evaluation_id=evaluation_id,
            skill_name=proposal.skill_name,
            operation=proposal.operation,
            status=PublicationStatus.preparing,
            base_snapshot=base_snapshot,
            candidate_snapshot=candidate_snapshot,
            base_skill_hash=base_snapshot.skill_md_hash,
            created_at=now,
        )
        put_result = await self._store.put_publication(publication)
        return put_result.value, put_result.created

    async def publish(
        self,
        *,
        user_id: str,
        proposal_id: str,
        evaluation_id: str,
        now: datetime | None = None,
    ) -> SkillPublicationResult:
        proposal, evaluation = await self._load_pair(
            user_id=user_id,
            proposal_id=proposal_id,
            evaluation_id=evaluation_id,
        )
        return await self._publish_loaded(
            proposal=proposal,
            evaluation_id=evaluation.evaluation_id,
            start_status=ProposalStatus.approved,
            prerequisite="approved",
            now=now,
        )

    async def publish_direct(
        self,
        *,
        user_id: str,
        proposal_id: str,
        now: datetime | None = None,
    ) -> SkillPublicationResult:
        """Publish one staged Proposal without quality evaluation or approval."""
        proposal = await self._load_proposal(
            user_id=user_id,
            proposal_id=proposal_id,
        )
        return await self._publish_loaded(
            proposal=proposal,
            evaluation_id=None,
            start_status=ProposalStatus.staged,
            prerequisite="staged for direct publication",
            now=now,
        )

    async def _publish_loaded(
        self,
        *,
        proposal: SkillProposal,
        evaluation_id: str | None,
        start_status: ProposalStatus,
        prerequisite: str,
        now: datetime | None,
    ) -> SkillPublicationResult:
        published_at = now or datetime.now(UTC)
        user_id = proposal.user_id
        proposal_id = proposal.proposal_id
        if proposal.status not in {
            start_status,
            ProposalStatus.publishing,
            ProposalStatus.published,
            ProposalStatus.rolled_back,
        }:
            raise ValueError(f"Proposal must be {prerequisite} before publication")
        if proposal.status is start_status and proposal.expires_at is not None and published_at >= proposal.expires_at:
            raise ValueError(f"{start_status.value} Proposal is expired")
        publication, _ = await self._prepare_publication(
            proposal,
            evaluation_id=evaluation_id,
            now=published_at,
        )
        if publication.status in {
            PublicationStatus.published,
            PublicationStatus.rolled_back,
        }:
            stored_proposal = await self._store.get_proposal(
                user_id,
                proposal_id,
            )
            if stored_proposal is None:
                raise ValueError("proposal was not found")
            changed = False
            if publication.status is PublicationStatus.published and stored_proposal.status is ProposalStatus.publishing:
                stored_proposal = await self._transition_proposal(
                    stored_proposal,
                    new_status=ProposalStatus.published,
                    reason_code="publication_completed",
                    reason=f"Publication {publication.publication_id} completed.",
                    occurred_at=publication.published_at or published_at,
                    source=ProposalStatusSource.publisher,
                    evaluation_id=evaluation_id,
                    publication_id=publication.publication_id,
                )
                changed = True
            elif publication.status is PublicationStatus.rolled_back and stored_proposal.status is ProposalStatus.published:
                stored_proposal = await self._transition_proposal(
                    stored_proposal,
                    new_status=ProposalStatus.rolled_back,
                    reason_code="rollback_completed",
                    reason=f"Publication {publication.publication_id} was rolled back.",
                    occurred_at=publication.rolled_back_at or published_at,
                    source=ProposalStatusSource.rollback,
                    evaluation_id=evaluation_id,
                    publication_id=publication.publication_id,
                    actor_id=publication.rollback_actor_id,
                )
                changed = True
            if publication.status is PublicationStatus.published:
                await self._record_publication_credit(
                    stored_proposal,
                    publication,
                )
            else:
                await self._record_rollback_credit(
                    stored_proposal,
                    publication,
                    rolled_back_at=(publication.rolled_back_at or published_at),
                )
            return SkillPublicationResult(
                publication=publication,
                proposal=stored_proposal,
                changed=changed,
            )
        if proposal.status is start_status:
            direct = start_status is ProposalStatus.staged
            proposal = await self._transition_proposal(
                proposal,
                new_status=ProposalStatus.publishing,
                reason_code=("direct_publication_started" if direct else "publication_started"),
                reason=(f"Publication {publication.publication_id} reserved the staged Proposal for direct publication." if direct else f"Publication {publication.publication_id} reserved the approved Proposal."),
                occurred_at=published_at,
                source=ProposalStatusSource.publisher,
                evaluation_id=evaluation_id,
                publication_id=publication.publication_id,
            )
        elif proposal.status is not ProposalStatus.publishing:
            raise ValueError(f"Proposal must be {prerequisite} before publication")

        storage = self._storage_factory(user_id)
        current = await capture_skill_package_snapshot(
            storage,
            user_id=user_id,
            skill_name=proposal.skill_name,
            created_at=published_at,
            allow_absent=not publication.base_snapshot.exists,
        )
        try:
            if current.snapshot_hash != publication.candidate_snapshot.snapshot_hash:
                if current.snapshot_hash != publication.base_snapshot.snapshot_hash:
                    raise ValueError(f"Skill package hash conflict for '{proposal.skill_name}'.")
                await self._mutation_service.replace_package(
                    SkillPackageMutationRequest(
                        user_id=user_id,
                        action="evolution_publish",
                        name=proposal.skill_name,
                        files=publication.candidate_snapshot.package_files(),
                        moderation_paths=tuple(item.path for item in proposal.proposed_files),
                        expected_base_hash=publication.base_skill_hash,
                        expected_package_hash=(publication.base_snapshot.snapshot_hash if publication.base_snapshot.exists else None),
                        require_absent=not publication.base_snapshot.exists,
                        proposal_id=proposal.proposal_id,
                        evaluation_id=evaluation_id,
                        publication_id=publication.publication_id,
                    )
                )
            actual = await capture_skill_package_snapshot(
                storage,
                user_id=user_id,
                skill_name=proposal.skill_name,
                created_at=published_at,
            )
            if actual.snapshot_hash != publication.candidate_snapshot.snapshot_hash:
                raise ValueError("published Skill package does not match the persisted candidate snapshot")
        except Exception:
            current_proposal = await self._store.get_proposal(
                user_id,
                proposal_id,
            )
            if current_proposal is not None and current_proposal.status is ProposalStatus.publishing:
                await self._transition_proposal(
                    current_proposal,
                    new_status=start_status,
                    reason_code="publication_failed",
                    reason=f"Publication {publication.publication_id} failed before completion.",
                    occurred_at=published_at,
                    source=ProposalStatusSource.publisher,
                    evaluation_id=evaluation_id,
                    publication_id=publication.publication_id,
                )
            safe_emit_evolution_event(
                self._observability,
                kind=EvolutionLifecycleKind.rejected,
                stage="publication",
                user_id=user_id,
                skill_name=proposal.skill_name,
                proposal_id=proposal_id,
                evaluation_id=evaluation_id,
                publication_id=publication.publication_id,
                snapshot_hash=(publication.candidate_snapshot.snapshot_hash),
                reason_codes=["publication_failed"],
                occurred_at=published_at,
            )
            raise

        completed = SkillPublication.model_validate(
            {
                **publication.model_dump(mode="python"),
                "status": PublicationStatus.published,
                "published_snapshot": actual,
                "published_skill_hash": actual.skill_md_hash,
                "published_at": published_at,
            }
        )
        publication_transitioned = False
        try:
            completed = await self._store.transition_publication(
                completed,
                expected_status=PublicationStatus.preparing,
            )
            publication_transitioned = True
        except EvolutionStoreConflict:
            winner = await self._store.get_publication(
                user_id,
                publication.publication_id,
            )
            if winner is None or winner.status not in {
                PublicationStatus.published,
                PublicationStatus.rolled_back,
            }:
                raise
            completed = winner
        if publication_transitioned:
            safe_emit_evolution_event(
                self._observability,
                kind=EvolutionLifecycleKind.published,
                stage="publication",
                user_id=user_id,
                skill_name=proposal.skill_name,
                proposal_id=proposal_id,
                evaluation_id=evaluation_id,
                publication_id=completed.publication_id,
                snapshot_hash=(completed.published_snapshot.snapshot_hash),
                occurred_at=completed.published_at,
            )
        current_proposal = await self._store.get_proposal(
            user_id,
            proposal_id,
        )
        if current_proposal is None:
            raise ValueError("proposal was not found")
        if current_proposal.status is ProposalStatus.publishing:
            current_proposal = await self._transition_proposal(
                current_proposal,
                new_status=ProposalStatus.published,
                reason_code="publication_completed",
                reason=f"Publication {publication.publication_id} completed.",
                occurred_at=published_at,
                source=ProposalStatusSource.publisher,
                evaluation_id=evaluation_id,
                publication_id=publication.publication_id,
            )
        await self._record_publication_credit(
            current_proposal,
            completed,
        )
        return SkillPublicationResult(
            publication=completed,
            proposal=current_proposal,
            changed=True,
        )

    async def rollback(
        self,
        *,
        user_id: str,
        publication_id: str,
        actor_id: str,
        now: datetime | None = None,
    ) -> SkillPublicationResult:
        rolled_back_at = now or datetime.now(UTC)
        publication = await self._store.get_publication(
            user_id,
            publication_id,
        )
        if publication is None:
            raise ValueError("publication was not found")
        proposal = await self._store.get_proposal(
            user_id,
            publication.proposal_id,
        )
        if proposal is None:
            raise ValueError("proposal was not found")
        if publication.status is PublicationStatus.rolled_back:
            if proposal.status is ProposalStatus.published:
                proposal = await self._transition_proposal(
                    proposal,
                    new_status=ProposalStatus.rolled_back,
                    reason_code="rollback_completed",
                    reason=f"Publication {publication_id} was rolled back.",
                    occurred_at=publication.rolled_back_at or rolled_back_at,
                    source=ProposalStatusSource.rollback,
                    evaluation_id=publication.evaluation_id,
                    publication_id=publication.publication_id,
                    actor_id=publication.rollback_actor_id,
                )
            await self._record_rollback_credit(
                proposal,
                publication,
                rolled_back_at=(publication.rolled_back_at or rolled_back_at),
            )
            return SkillPublicationResult(
                publication=publication,
                proposal=proposal,
                changed=False,
            )
        if publication.status is not PublicationStatus.published:
            raise ValueError("publication is not ready for rollback")
        if proposal.status is not ProposalStatus.published:
            raise ValueError("Proposal is not in published state")

        storage = self._storage_factory(user_id)
        current = await capture_skill_package_snapshot(
            storage,
            user_id=user_id,
            skill_name=publication.skill_name,
            created_at=rolled_back_at,
            allow_absent=not publication.published_snapshot.exists,
        )
        if current.snapshot_hash != publication.base_snapshot.snapshot_hash:
            if current.snapshot_hash != publication.published_snapshot.snapshot_hash:
                raise ValueError(f"Skill package hash conflict for '{publication.skill_name}'.")
            await self._mutation_service.replace_package(
                SkillPackageMutationRequest(
                    user_id=user_id,
                    action="evolution_rollback",
                    name=publication.skill_name,
                    files=publication.base_snapshot.package_files(),
                    moderation_paths=tuple(item.path for item in publication.base_snapshot.files if item.path == "SKILL.md" or item.executable),
                    expected_base_hash=publication.published_skill_hash,
                    expected_package_hash=publication.published_snapshot.snapshot_hash,
                    require_absent=False,
                    proposal_id=publication.proposal_id,
                    evaluation_id=publication.evaluation_id,
                    publication_id=publication.publication_id,
                    actor_id=actor_id,
                )
            )
        restored = await capture_skill_package_snapshot(
            storage,
            user_id=user_id,
            skill_name=publication.skill_name,
            created_at=rolled_back_at,
            allow_absent=not publication.base_snapshot.exists,
        )
        if restored.snapshot_hash != publication.base_snapshot.snapshot_hash:
            raise ValueError("rollback result does not match the persisted base snapshot")
        rolled_back = SkillPublication.model_validate(
            {
                **publication.model_dump(mode="python"),
                "status": PublicationStatus.rolled_back,
                "rollback_snapshot": restored,
                "rolled_back_at": rolled_back_at,
                "rollback_actor_id": actor_id,
            }
        )
        rolled_back = await self._store.transition_publication(
            rolled_back,
            expected_status=PublicationStatus.published,
        )
        safe_emit_evolution_event(
            self._observability,
            kind=EvolutionLifecycleKind.rolled_back,
            stage="rollback",
            user_id=user_id,
            skill_name=publication.skill_name,
            proposal_id=publication.proposal_id,
            evaluation_id=publication.evaluation_id,
            publication_id=publication.publication_id,
            snapshot_hash=(rolled_back.rollback_snapshot.snapshot_hash),
            occurred_at=rolled_back_at,
        )
        proposal = await self._transition_proposal(
            proposal,
            new_status=ProposalStatus.rolled_back,
            reason_code="rollback_completed",
            reason=f"Publication {publication_id} was rolled back from its persisted base snapshot.",
            occurred_at=rolled_back_at,
            source=ProposalStatusSource.rollback,
            evaluation_id=publication.evaluation_id,
            publication_id=publication.publication_id,
            actor_id=actor_id,
        )
        await self._record_rollback_credit(
            proposal,
            rolled_back,
            rolled_back_at=rolled_back_at,
        )
        return SkillPublicationResult(
            publication=rolled_back,
            proposal=proposal,
            changed=True,
        )
