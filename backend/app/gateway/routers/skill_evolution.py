"""Authenticated, owner-scoped APIs for Skill evolution records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Query, Request

from app.gateway.authz import require_permission
from app.gateway.deps import (
    get_config,
    get_skill_evolution_store,
    get_skill_publication_service,
    require_admin_user,
)
from app.gateway.skill_evolution_models import (
    EvolutionClusterPage,
    EvolutionEventPage,
    ProposalPublishRequest,
    ProposalReviewRequest,
    ProposalReviewResponse,
    SkillEvaluationPage,
    SkillProposalPage,
    SkillPublicationMutationResponse,
    SkillVersionPage,
    compact_approval_assessment,
    compact_cluster,
    compact_evaluation,
    compact_event,
    compact_proposal,
    compact_version,
)
from deerflow.skill_evolution.approval import (
    ApprovalOutcome,
    ManualApprovalRequest,
    ProposalApprovalPolicy,
)
from deerflow.skill_evolution.models import (
    ClusterStatus,
    EvaluationDecision,
    EvolutionEventKind,
    ProposalStatus,
    PublicationStatus,
)
from deerflow.skill_evolution.observability import (
    get_evolution_observability,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
)

router = APIRouter(
    prefix="/api/skill-evolution",
    tags=["skill-evolution"],
)

_CROSS_USER_ADMIN_DETAIL = "Admin access is required to read another user's Skill evolution records."


async def _resolve_owner_user_id(
    request: Request,
    requested_user_id: str | None,
) -> str:
    auth = request.state.auth
    current_user_id = str(auth.require_user().id)
    if requested_user_id is None or requested_user_id == current_user_id:
        return current_user_id

    await require_admin_user(
        request,
        detail=_CROSS_USER_ADMIN_DETAIL,
    )
    return requested_user_id


def _page_metadata(
    rows: list[Any],
    *,
    limit: int,
    offset: int,
) -> tuple[list[Any], bool, int | None]:
    has_more = len(rows) > limit
    return (
        rows[:limit],
        has_more,
        offset + limit if has_more else None,
    )


def _get_approval_policy(
    request: Request,
) -> ProposalApprovalPolicy:
    observability = getattr(
        request.app.state,
        "skill_evolution_observability",
        None,
    )
    return ProposalApprovalPolicy(
        config=get_config().skill_evolution.publication,
        observability=(observability or get_evolution_observability()),
    )


def _actor_user_id(request: Request) -> str:
    return str(request.state.auth.require_user().id)


async def _require_proposal_and_evaluation(
    request: Request,
    *,
    user_id: str,
    proposal_id: str,
    evaluation_id: str,
) -> tuple[Any, Any]:
    store = get_skill_evolution_store(request)
    proposal = await store.get_proposal(
        user_id,
        proposal_id,
    )
    if proposal is None:
        raise HTTPException(
            status_code=404,
            detail="Skill Proposal not found",
        )
    evaluation = await store.get_evaluation(
        user_id,
        evaluation_id,
    )
    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail="Skill Evaluation not found",
        )
    if evaluation.proposal_id != proposal.proposal_id:
        raise HTTPException(
            status_code=409,
            detail=("Skill Evaluation does not belong to the requested Proposal"),
        )
    return proposal, evaluation


@router.get("/events", response_model=EvolutionEventPage)
@require_permission("skill_evolution", "read")
async def list_events(
    request: Request,
    user_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    event_kind: EvolutionEventKind | None = None,
) -> EvolutionEventPage:
    owner_user_id = await _resolve_owner_user_id(
        request,
        user_id,
    )
    rows = await get_skill_evolution_store(request).list_event_page(
        owner_user_id,
        limit=limit + 1,
        offset=offset,
        event_kind=event_kind,
    )
    page, has_more, next_offset = _page_metadata(
        rows,
        limit=limit,
        offset=offset,
    )
    return EvolutionEventPage(
        user_id=owner_user_id,
        data=[compact_event(item) for item in page],
        limit=limit,
        offset=offset,
        has_more=has_more,
        next_offset=next_offset,
    )


@router.get("/clusters", response_model=EvolutionClusterPage)
@require_permission("skill_evolution", "read")
async def list_clusters(
    request: Request,
    user_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: ClusterStatus | None = None,
) -> EvolutionClusterPage:
    owner_user_id = await _resolve_owner_user_id(
        request,
        user_id,
    )
    rows = await get_skill_evolution_store(request).list_cluster_page(
        owner_user_id,
        limit=limit + 1,
        offset=offset,
        status=status,
    )
    page, has_more, next_offset = _page_metadata(
        rows,
        limit=limit,
        offset=offset,
    )
    return EvolutionClusterPage(
        user_id=owner_user_id,
        data=[compact_cluster(item) for item in page],
        limit=limit,
        offset=offset,
        has_more=has_more,
        next_offset=next_offset,
    )


@router.get("/proposals", response_model=SkillProposalPage)
@require_permission("skill_evolution", "read")
async def list_proposals(
    request: Request,
    user_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: ProposalStatus | None = None,
) -> SkillProposalPage:
    owner_user_id = await _resolve_owner_user_id(
        request,
        user_id,
    )
    rows = await get_skill_evolution_store(request).list_proposal_page(
        owner_user_id,
        limit=limit + 1,
        offset=offset,
        status=status,
    )
    page, has_more, next_offset = _page_metadata(
        rows,
        limit=limit,
        offset=offset,
    )
    return SkillProposalPage(
        user_id=owner_user_id,
        data=[compact_proposal(item) for item in page],
        limit=limit,
        offset=offset,
        has_more=has_more,
        next_offset=next_offset,
    )


@router.get(
    "/evaluations",
    response_model=SkillEvaluationPage,
)
@require_permission("skill_evolution", "read")
async def list_evaluations(
    request: Request,
    user_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    decision: EvaluationDecision | None = None,
) -> SkillEvaluationPage:
    owner_user_id = await _resolve_owner_user_id(
        request,
        user_id,
    )
    rows = await get_skill_evolution_store(request).list_evaluation_page(
        owner_user_id,
        limit=limit + 1,
        offset=offset,
        decision=decision,
    )
    page, has_more, next_offset = _page_metadata(
        rows,
        limit=limit,
        offset=offset,
    )
    return SkillEvaluationPage(
        user_id=owner_user_id,
        data=[compact_evaluation(item) for item in page],
        limit=limit,
        offset=offset,
        has_more=has_more,
        next_offset=next_offset,
    )


@router.get("/versions", response_model=SkillVersionPage)
@require_permission("skill_evolution", "read")
async def list_versions(
    request: Request,
    user_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: PublicationStatus | None = None,
) -> SkillVersionPage:
    owner_user_id = await _resolve_owner_user_id(
        request,
        user_id,
    )
    rows = await get_skill_evolution_store(request).list_publication_page(
        owner_user_id,
        limit=limit + 1,
        offset=offset,
        status=status,
    )
    page, has_more, next_offset = _page_metadata(
        rows,
        limit=limit,
        offset=offset,
    )
    return SkillVersionPage(
        user_id=owner_user_id,
        data=[compact_version(item) for item in page],
        limit=limit,
        offset=offset,
        has_more=has_more,
        next_offset=next_offset,
    )


@router.post(
    "/proposals/{proposal_id}/review",
    response_model=ProposalReviewResponse,
)
@require_permission("skill_evolution", "review")
async def review_proposal(
    proposal_id: Annotated[
        str,
        Path(min_length=1, max_length=128),
    ],
    body: ProposalReviewRequest,
    request: Request,
    user_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=128,
    ),
) -> ProposalReviewResponse:
    owner_user_id = await _resolve_owner_user_id(
        request,
        user_id,
    )
    await _require_proposal_and_evaluation(
        request,
        user_id=owner_user_id,
        proposal_id=proposal_id,
        evaluation_id=body.evaluation_id,
    )
    manual_request = ManualApprovalRequest(
        decision=(ApprovalOutcome.approved if body.decision == "approve" else ApprovalOutcome.rejected),
        reviewer_id=_actor_user_id(request),
        reason=body.reason,
        decided_at=datetime.now(UTC),
    )
    try:
        result = await _get_approval_policy(request).review_and_persist(
            user_id=owner_user_id,
            proposal_id=proposal_id,
            evaluation_id=body.evaluation_id,
            request=manual_request,
            store=get_skill_evolution_store(request),
        )
    except EvolutionStoreConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="Skill Proposal review state conflict",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=("Skill Proposal cannot be reviewed in its current state"),
        ) from exc
    return ProposalReviewResponse(
        user_id=owner_user_id,
        proposal=compact_proposal(result.proposal),
        assessment=compact_approval_assessment(result.assessment),
        changed=result.changed,
    )


@router.post(
    "/proposals/{proposal_id}/publish",
    response_model=SkillPublicationMutationResponse,
)
@require_permission("skill_evolution", "publish")
async def publish_proposal(
    proposal_id: Annotated[
        str,
        Path(min_length=1, max_length=128),
    ],
    body: ProposalPublishRequest,
    request: Request,
    user_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=128,
    ),
) -> SkillPublicationMutationResponse:
    owner_user_id = await _resolve_owner_user_id(
        request,
        user_id,
    )
    await _require_proposal_and_evaluation(
        request,
        user_id=owner_user_id,
        proposal_id=proposal_id,
        evaluation_id=body.evaluation_id,
    )
    try:
        result = await get_skill_publication_service(request).publish(
            user_id=owner_user_id,
            proposal_id=proposal_id,
            evaluation_id=body.evaluation_id,
        )
    except EvolutionStoreConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="Skill publication state conflict",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=("Skill Proposal cannot be published in its current state"),
        ) from exc
    return SkillPublicationMutationResponse(
        user_id=owner_user_id,
        proposal=compact_proposal(result.proposal),
        version=compact_version(result.publication),
        changed=result.changed,
    )


@router.post(
    "/versions/{publication_id}/rollback",
    response_model=SkillPublicationMutationResponse,
)
@require_permission("skill_evolution", "rollback")
async def rollback_version(
    publication_id: Annotated[
        str,
        Path(min_length=1, max_length=128),
    ],
    request: Request,
    user_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=128,
    ),
) -> SkillPublicationMutationResponse:
    owner_user_id = await _resolve_owner_user_id(
        request,
        user_id,
    )
    publication = await get_skill_evolution_store(request).get_publication(
        owner_user_id,
        publication_id,
    )
    if publication is None:
        raise HTTPException(
            status_code=404,
            detail="Skill version not found",
        )
    try:
        result = await get_skill_publication_service(request).rollback(
            user_id=owner_user_id,
            publication_id=publication_id,
            actor_id=_actor_user_id(request),
        )
    except EvolutionStoreConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="Skill rollback state conflict",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=("Skill version cannot be rolled back in its current state"),
        ) from exc
    return SkillPublicationMutationResponse(
        user_id=owner_user_id,
        proposal=compact_proposal(result.proposal),
        version=compact_version(result.publication),
        changed=result.changed,
    )
