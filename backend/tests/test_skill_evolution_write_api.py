from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from _router_auth_helpers import make_authed_test_app
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.gateway.auth.models import User
from app.gateway.authz import AuthContext, Permissions
from app.gateway.csrf_middleware import (
    CSRF_COOKIE_NAME,
    CSRF_HEADER_NAME,
    CSRFMiddleware,
)
from app.gateway.routers import skill_evolution
from deerflow.skill_evolution.approval import ApprovalOutcome

_NOW = datetime(2026, 8, 17, 12, tzinfo=UTC)
_ALICE_ID = UUID("00000000-0000-0000-0000-000000000001")
_BOB_ID = UUID("00000000-0000-0000-0000-000000000002")
_CSRF_TOKEN = "csrf-token-for-skill-evolution-tests"
_SECRET = "TOP-SECRET-WRITE-CONTENT"


def _user(
    user_id: UUID,
    *,
    role: str = "user",
) -> User:
    return User(
        id=user_id,
        email=f"{role}-{user_id.int}@example.com",
        password_hash="x",
        system_role=role,
    )


def _proposal(
    *,
    status: str,
    proposal_id: str = "proposal-1",
):
    return SimpleNamespace(
        proposal_id=proposal_id,
        cluster_id="cluster-1",
        operation=SimpleNamespace(value="patch"),
        skill_name="verified-workflow",
        base_skill_hash="a" * 64,
        status=SimpleNamespace(value=status),
        proposed_files=[
            SimpleNamespace(
                path="SKILL.md",
                content=_SECRET,
                executable=False,
            )
        ],
        supporting_event_ids=["event-1", "event-2", "event-3"],
        risks=[],
        requires_manual_review=False,
        review_reasons=[],
        expires_at=_NOW,
        created_at=_NOW,
    )


def _evaluation():
    return SimpleNamespace(
        evaluation_id="evaluation-1",
        proposal_id="proposal-1",
    )


def _publication(*, status: str):
    return SimpleNamespace(
        publication_id="publication-proposal-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        skill_name="verified-workflow",
        operation=SimpleNamespace(value="patch"),
        status=SimpleNamespace(value=status),
        base_snapshot=SimpleNamespace(
            snapshot_hash="b" * 64,
            files=[SimpleNamespace(content_base64=_SECRET)],
        ),
        published_snapshot=SimpleNamespace(
            snapshot_hash="c" * 64,
            files=[SimpleNamespace(content_base64=_SECRET)],
        ),
        base_skill_hash="a" * 64,
        published_skill_hash="d" * 64,
        rollback_snapshot=(
            SimpleNamespace(
                snapshot_hash="b" * 64,
                files=[SimpleNamespace(content_base64=_SECRET)],
            )
            if status == "rolled_back"
            else None
        ),
        created_at=_NOW,
        published_at=_NOW,
        rolled_back_at=(_NOW if status == "rolled_back" else None),
    )


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def get_proposal(
        self,
        user_id: str,
        proposal_id: str,
    ):
        self.calls.append(("proposal", user_id, proposal_id))
        if user_id not in {str(_ALICE_ID), str(_BOB_ID)}:
            return None
        return _proposal(
            status="validating",
            proposal_id=proposal_id,
        )

    async def get_evaluation(
        self,
        user_id: str,
        evaluation_id: str,
    ):
        self.calls.append(("evaluation", user_id, evaluation_id))
        if user_id not in {str(_ALICE_ID), str(_BOB_ID)}:
            return None
        return _evaluation()

    async def get_publication(
        self,
        user_id: str,
        publication_id: str,
    ):
        self.calls.append(("publication", user_id, publication_id))
        if user_id not in {str(_ALICE_ID), str(_BOB_ID)}:
            return None
        return _publication(status="published")


class _ApprovalPolicy:
    def __init__(self) -> None:
        self.calls = []

    async def review_and_persist(self, **kwargs):
        self.calls.append(kwargs)
        outcome = kwargs["request"].decision
        status = "approved" if outcome is ApprovalOutcome.approved else "rejected"
        return SimpleNamespace(
            proposal=_proposal(status=status),
            assessment=SimpleNamespace(
                policy_version="skill-approval-v1",
                outcome=outcome,
                reason_codes=["manual_approval" if outcome is ApprovalOutcome.approved else "manual_rejection"],
                auto_publish_eligible=False,
                expires_at=_NOW,
                assessed_at=kwargs["request"].decided_at,
            ),
            changed=True,
        )


class _PublicationService:
    def __init__(self) -> None:
        self.publish_calls = []
        self.rollback_calls = []

    async def publish(self, **kwargs):
        self.publish_calls.append(kwargs)
        return SimpleNamespace(
            proposal=_proposal(status="published"),
            publication=_publication(status="published"),
            changed=True,
        )

    async def rollback(self, **kwargs):
        self.rollback_calls.append(kwargs)
        return SimpleNamespace(
            proposal=_proposal(status="rolled_back"),
            publication=_publication(status="rolled_back"),
            changed=True,
        )


def _app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    user: User,
):
    store = _Store()
    approval_policy = _ApprovalPolicy()
    publication_service = _PublicationService()
    app = make_authed_test_app(user_factory=lambda: user)
    app.add_middleware(CSRFMiddleware)
    app.state.skill_evolution_store = store
    app.state.skill_publication_service = publication_service
    monkeypatch.setattr(
        skill_evolution,
        "_get_approval_policy",
        lambda request: approval_policy,
    )
    app.include_router(skill_evolution.router)
    return app, store, approval_policy, publication_service


def _post(
    client: TestClient,
    path: str,
    *,
    json: dict | None = None,
    params: dict | None = None,
):
    client.cookies.set(CSRF_COOKIE_NAME, _CSRF_TOKEN)
    return client.post(
        path,
        json=json,
        params=params,
        headers={CSRF_HEADER_NAME: _CSRF_TOKEN},
    )


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [
        ("approve", "approved"),
        ("reject", "rejected"),
    ],
)
def test_owner_can_review_proposal_with_server_owned_actor(
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    expected_status: str,
) -> None:
    app, _, policy, _ = _app(
        monkeypatch,
        user=_user(_ALICE_ID),
    )

    with TestClient(app) as client:
        response = _post(
            client,
            "/api/skill-evolution/proposals/proposal-1/review",
            json={
                "evaluation_id": "evaluation-1",
                "decision": decision,
                "reason": "Reviewed against the replay evidence.",
            },
        )

    assert response.status_code == 200
    assert response.json()["proposal"]["status"] == expected_status
    assert response.json()["assessment"]["outcome"] == expected_status
    assert _SECRET not in response.text
    manual_request = policy.calls[0]["request"]
    assert manual_request.reviewer_id == str(_ALICE_ID)
    assert manual_request.reason == ("Reviewed against the replay evidence.")
    assert manual_request.decided_at.tzinfo is UTC


def test_owner_can_publish_and_rollback_compact_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, service = _app(
        monkeypatch,
        user=_user(_ALICE_ID),
    )

    with TestClient(app) as client:
        published = _post(
            client,
            "/api/skill-evolution/proposals/proposal-1/publish",
            json={"evaluation_id": "evaluation-1"},
        )
        rolled_back = _post(
            client,
            ("/api/skill-evolution/versions/publication-proposal-1/rollback"),
        )

    assert published.status_code == 200
    assert published.json()["version"]["status"] == "published"
    assert rolled_back.status_code == 200
    assert rolled_back.json()["version"]["status"] == "rolled_back"
    assert _SECRET not in published.text
    assert _SECRET not in rolled_back.text
    assert service.publish_calls[0]["user_id"] == str(_ALICE_ID)
    assert service.rollback_calls[0]["user_id"] == str(_ALICE_ID)
    assert service.rollback_calls[0]["actor_id"] == str(_ALICE_ID)


def test_non_admin_cross_user_mutation_stops_before_store_and_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store, policy, service = _app(
        monkeypatch,
        user=_user(_ALICE_ID),
    )

    with TestClient(app) as client:
        response = _post(
            client,
            "/api/skill-evolution/proposals/proposal-1/review",
            params={"user_id": str(_BOB_ID)},
            json={
                "evaluation_id": "evaluation-1",
                "decision": "approve",
                "reason": "Attempted cross-user review.",
            },
        )

    assert response.status_code == 403
    assert store.calls == []
    assert policy.calls == []
    assert service.publish_calls == []


def test_admin_cross_user_publish_keeps_admin_actor_and_bob_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, service = _app(
        monkeypatch,
        user=_user(_ALICE_ID, role="admin"),
    )

    with TestClient(app) as client:
        response = _post(
            client,
            "/api/skill-evolution/proposals/proposal-1/publish",
            params={"user_id": str(_BOB_ID)},
            json={"evaluation_id": "evaluation-1"},
        )

    assert response.status_code == 200
    assert service.publish_calls[0]["user_id"] == str(_BOB_ID)


def test_csrf_blocks_write_before_domain_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store, policy, service = _app(
        monkeypatch,
        user=_user(_ALICE_ID),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/skill-evolution/proposals/proposal-1/review",
            json={
                "evaluation_id": "evaluation-1",
                "decision": "approve",
                "reason": "No CSRF token.",
            },
        )

    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]
    assert store.calls == []
    assert policy.calls == []
    assert service.publish_calls == []


@pytest.mark.parametrize(
    "body",
    [
        {
            "evaluation_id": "evaluation-1",
            "decision": "approve",
            "reason": "Reviewed.",
            "reviewer_id": str(_BOB_ID),
        },
        {
            "evaluation_id": "evaluation-1",
            "decision": "approve",
            "reason": "Reviewed.",
            "decided_at": _NOW.isoformat(),
        },
        {
            "evaluation_id": "evaluation-1",
            "decision": "approve",
            "reason": "   ",
        },
    ],
)
def test_review_rejects_client_owned_audit_fields_and_blank_reason(
    monkeypatch: pytest.MonkeyPatch,
    body: dict,
) -> None:
    app, store, policy, _ = _app(
        monkeypatch,
        user=_user(_ALICE_ID),
    )

    with TestClient(app) as client:
        response = _post(
            client,
            "/api/skill-evolution/proposals/proposal-1/review",
            json=body,
        )

    assert response.status_code == 422
    assert store.calls == []
    assert policy.calls == []


def test_domain_error_is_redacted_at_http_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, policy, _ = _app(
        monkeypatch,
        user=_user(_ALICE_ID),
    )

    async def fail_review(**kwargs):
        raise ValueError(_SECRET)

    monkeypatch.setattr(
        policy,
        "review_and_persist",
        fail_review,
    )
    with TestClient(app) as client:
        response = _post(
            client,
            "/api/skill-evolution/proposals/proposal-1/review",
            json={
                "evaluation_id": "evaluation-1",
                "decision": "approve",
                "reason": "Reviewed.",
            },
        )

    assert response.status_code == 409
    assert _SECRET not in response.text
    assert response.json()["detail"] == ("Skill Proposal cannot be reviewed in its current state")


def test_publication_security_error_is_redacted_at_http_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, service = _app(
        monkeypatch,
        user=_user(_ALICE_ID),
    )

    async def fail_publication(**kwargs):
        raise ValueError(_SECRET)

    monkeypatch.setattr(
        service,
        "publish",
        fail_publication,
    )
    with TestClient(app) as client:
        response = _post(
            client,
            "/api/skill-evolution/proposals/proposal-1/publish",
            json={"evaluation_id": "evaluation-1"},
        )

    assert response.status_code == 409
    assert _SECRET not in response.text
    assert response.json()["detail"] == ("Skill Proposal cannot be published in its current state")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "permission", "kwargs"),
    [
        (
            skill_evolution.review_proposal,
            "skill_evolution:review",
            {
                "proposal_id": "proposal-1",
                "body": SimpleNamespace(),
                "user_id": None,
            },
        ),
        (
            skill_evolution.publish_proposal,
            "skill_evolution:publish",
            {
                "proposal_id": "proposal-1",
                "body": SimpleNamespace(),
                "user_id": None,
            },
        ),
        (
            skill_evolution.rollback_version,
            "skill_evolution:rollback",
            {
                "publication_id": "publication-1",
                "user_id": None,
            },
        ),
    ],
)
async def test_each_write_route_requires_its_dedicated_permission(
    handler,
    permission: str,
    kwargs: dict,
) -> None:
    user = _user(_ALICE_ID)
    request = SimpleNamespace(
        state=SimpleNamespace(
            user=user,
            auth=AuthContext(
                user=user,
                permissions=[Permissions.SKILL_EVOLUTION_READ],
            ),
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler(request=request, **kwargs)

    assert exc_info.value.status_code == 403
    assert permission in str(exc_info.value.detail)
