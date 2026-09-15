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
from app.gateway.routers import skill_evolution

_NOW = datetime(2026, 8, 17, tzinfo=UTC)
_ALICE_ID = UUID("00000000-0000-0000-0000-000000000001")
_BOB_ID = UUID("00000000-0000-0000-0000-000000000002")
_SECRET = "TOP-SECRET-EVOLUTION-CONTENT"


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


def _event(index: int):
    return SimpleNamespace(
        event_id=f"event-{index}",
        run_id=f"run-{index}",
        thread_id=f"thread-{index}",
        event_kind=SimpleNamespace(value="new_skill_evidence"),
        task_signature=f"{_SECRET}-{index}",
        target_skill=None,
        source_snapshot_hash=f"{index}" * 64,
        outcome=SimpleNamespace(
            status=SimpleNamespace(value="success"),
            confidence=0.95,
        ),
        complexity=SimpleNamespace(
            tool_calls=7,
            had_recoverable_errors=True,
            had_user_correction=False,
            non_trivial_workflow=True,
            explicit_remember_request=False,
        ),
        task_goal=_SECRET,
        successful_path=[_SECRET],
        created_at=_NOW,
    )


def _cluster(index: int):
    return SimpleNamespace(
        cluster_id=f"cluster-{index}",
        event_kind=SimpleNamespace(value="new_skill_evidence"),
        target_skill=None,
        canonical_signature=_SECRET,
        member_event_ids=["event-1", "event-2", "event-3"],
        independent_run_count=3,
        status=SimpleNamespace(value="ready"),
        contradictory=False,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _proposal(index: int):
    return SimpleNamespace(
        proposal_id=f"proposal-{index}",
        cluster_id="cluster-1",
        operation=SimpleNamespace(value="create"),
        skill_name="verified-workflow",
        base_skill_hash=None,
        status=SimpleNamespace(value="staged"),
        proposed_files=[
            SimpleNamespace(
                path="SKILL.md",
                content=_SECRET,
                executable=False,
            )
        ],
        supporting_event_ids=["event-1", "event-2", "event-3"],
        risks=[_SECRET],
        requires_manual_review=False,
        review_reasons=[],
        expires_at=None,
        created_at=_NOW,
    )


def _evaluation(index: int):
    return SimpleNamespace(
        evaluation_id=f"evaluation-{index}",
        proposal_id="proposal-1",
        decision=SimpleNamespace(value="approve"),
        quality_score=0.8,
        quality=SimpleNamespace(
            designation=SimpleNamespace(value="high_quality"),
            sample_sufficient=True,
            sample_counts={
                "candidate_tasks": 5,
                "held_out_tasks": 2,
            },
            blockers=[],
        ),
        candidate_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(success=False),
        ],
        regression_results=[],
        safety_results={"private": _SECRET},
        created_at=_NOW,
    )


def _publication(index: int):
    return SimpleNamespace(
        publication_id=f"publication-{index}",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        skill_name="verified-workflow",
        operation=SimpleNamespace(value="create"),
        status=SimpleNamespace(value="published"),
        base_snapshot=SimpleNamespace(
            snapshot_hash="a" * 64,
            files=[
                SimpleNamespace(
                    content_base64=_SECRET,
                )
            ],
        ),
        published_snapshot=SimpleNamespace(
            snapshot_hash="b" * 64,
            files=[
                SimpleNamespace(
                    content_base64=_SECRET,
                )
            ],
        ),
        base_skill_hash=None,
        published_skill_hash="c" * 64,
        rollback_snapshot=None,
        created_at=_NOW,
        published_at=_NOW,
        rolled_back_at=None,
    )


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.records = {
            str(_ALICE_ID): {
                "events": [_event(3), _event(2), _event(1)],
                "clusters": [_cluster(1)],
                "proposals": [_proposal(1)],
                "evaluations": [_evaluation(1)],
                "publications": [_publication(1)],
            },
            str(_BOB_ID): {
                "events": [_event(9)],
                "clusters": [_cluster(9)],
                "proposals": [_proposal(9)],
                "evaluations": [_evaluation(9)],
                "publications": [_publication(9)],
            },
        }

    def _page(
        self,
        collection: str,
        user_id: str,
        *,
        limit: int,
        offset: int,
        filter_value=None,
    ):
        self.calls.append(
            (
                collection,
                user_id,
                limit,
                offset,
                filter_value,
            )
        )
        return self.records.get(user_id, {}).get(
            collection,
            [],
        )[offset : offset + limit]

    async def list_event_page(
        self,
        user_id,
        *,
        limit,
        offset,
        event_kind=None,
    ):
        return self._page(
            "events",
            user_id,
            limit=limit,
            offset=offset,
            filter_value=event_kind,
        )

    async def list_cluster_page(
        self,
        user_id,
        *,
        limit,
        offset,
        status=None,
    ):
        return self._page(
            "clusters",
            user_id,
            limit=limit,
            offset=offset,
            filter_value=status,
        )

    async def list_proposal_page(
        self,
        user_id,
        *,
        limit,
        offset,
        status=None,
    ):
        return self._page(
            "proposals",
            user_id,
            limit=limit,
            offset=offset,
            filter_value=status,
        )

    async def list_evaluation_page(
        self,
        user_id,
        *,
        limit,
        offset,
        decision=None,
    ):
        return self._page(
            "evaluations",
            user_id,
            limit=limit,
            offset=offset,
            filter_value=decision,
        )

    async def list_publication_page(
        self,
        user_id,
        *,
        limit,
        offset,
        status=None,
    ):
        return self._page(
            "publications",
            user_id,
            limit=limit,
            offset=offset,
            filter_value=status,
        )


def _app(user: User, store: _Store):
    app = make_authed_test_app(user_factory=lambda: user)
    app.state.skill_evolution_store = store
    app.include_router(skill_evolution.router)
    return app


def test_owner_scoped_event_pagination() -> None:
    store = _Store()
    app = _app(_user(_ALICE_ID), store)

    with TestClient(app) as client:
        first = client.get(
            "/api/skill-evolution/events",
            params={"limit": 2},
        )
        second = client.get(
            "/api/skill-evolution/events",
            params={"limit": 2, "offset": 2},
        )

    assert first.status_code == 200
    assert first.json()["user_id"] == str(_ALICE_ID)
    assert [item["event_id"] for item in first.json()["data"]] == ["event-3", "event-2"]
    assert first.json()["has_more"] is True
    assert first.json()["next_offset"] == 2
    assert [item["event_id"] for item in second.json()["data"]] == ["event-1"]
    assert second.json()["has_more"] is False
    assert store.calls[0][:4] == (
        "events",
        str(_ALICE_ID),
        3,
        0,
    )


@pytest.mark.parametrize(
    "path",
    [
        "events",
        "clusters",
        "proposals",
        "evaluations",
        "versions",
    ],
)
def test_all_read_collections_are_compact_and_redacted(
    path: str,
) -> None:
    store = _Store()
    app = _app(_user(_ALICE_ID), store)

    with TestClient(app) as client:
        response = client.get(
            f"/api/skill-evolution/{path}",
        )

    assert response.status_code == 200
    serialized = response.text
    assert _SECRET not in serialized
    assert "task_goal" not in serialized
    assert "successful_path" not in serialized
    assert "proposed_files" not in serialized
    assert "content_base64" not in serialized
    assert '"files"' not in serialized


def test_non_admin_cannot_query_another_user() -> None:
    store = _Store()
    app = _app(_user(_ALICE_ID), store)

    with TestClient(app) as client:
        response = client.get(
            "/api/skill-evolution/events",
            params={"user_id": str(_BOB_ID)},
        )

    assert response.status_code == 403
    assert store.calls == []


def test_admin_can_query_another_user() -> None:
    store = _Store()
    app = _app(
        _user(_ALICE_ID, role="admin"),
        store,
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/skill-evolution/events",
            params={"user_id": str(_BOB_ID)},
        )

    assert response.status_code == 200
    assert response.json()["user_id"] == str(_BOB_ID)
    assert response.json()["data"][0]["event_id"] == "event-9"


@pytest.mark.asyncio
async def test_route_requires_dedicated_read_permission() -> None:
    user = _user(_ALICE_ID)
    request = SimpleNamespace(
        state=SimpleNamespace(
            user=user,
            auth=AuthContext(
                user=user,
                permissions=[Permissions.RUNS_READ],
            ),
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await skill_evolution.list_events(
            request=request,
            user_id=None,
            limit=50,
            offset=0,
            event_kind=None,
        )

    assert exc_info.value.status_code == 403
    assert "skill_evolution:read" in str(exc_info.value.detail)


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
    ],
)
def test_pagination_query_is_bounded(params: dict) -> None:
    store = _Store()
    app = _app(_user(_ALICE_ID), store)

    with TestClient(app) as client:
        response = client.get(
            "/api/skill-evolution/events",
            params=params,
        )

    assert response.status_code == 422
    assert store.calls == []
