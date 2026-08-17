from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from deerflow.config.skill_evolution_config import (
    SkillEvolutionEmbeddingConfig,
    SkillEvolutionGroupingConfig,
    SkillEvolutionVectorStoreConfig,
)
from deerflow.skill_evolution.models import (
    ComplexitySignals,
    EnvironmentSignature,
    EvolutionEvent,
    EvolutionEventKind,
    OutcomeEvidence,
    OutcomeStatus,
    SkillGap,
    SkillGapCategory,
    SkillTarget,
    SkillUsage,
    ToolSignature,
)
from deerflow.skill_evolution.semantic_retrieval import (
    EmbeddingBatch,
    OllamaEmbeddingProvider,
    QdrantSemanticVectorStore,
    SemanticFallbackReason,
    SemanticRetrievalMode,
    SemanticSearchHit,
    retrieve_event_candidates,
)


def _event(
    event_id: str,
    *,
    user_id: str = "user-1",
    run_id: str | None = None,
    task_input_hash: str | None = None,
    task_signature: str = "python-package-install",
    task_goal: str = "Install a Python package in the project environment.",
    tool_names: list[str] | None = None,
    target_skill: str | None = None,
) -> EvolutionEvent:
    patch = target_skill is not None
    return EvolutionEvent(
        event_id=event_id,
        run_id=run_id or f"run-{event_id}",
        thread_id=f"thread-{event_id}",
        user_id=user_id,
        extractor_version="structured-v1:test-model",
        source_snapshot_hash="a" * 64,
        task_input_hash=task_input_hash or f"{int(event_id[-1], 36) % 16:x}" * 64,
        event_kind=(EvolutionEventKind.skill_patch_evidence if patch else EvolutionEventKind.new_skill_evidence),
        task_signature=task_signature,
        task_goal=task_goal,
        environment=EnvironmentSignature(
            os="macOS",
            shell="zsh",
            runtime="Python 3.12",
        ),
        outcome=OutcomeEvidence(
            status=OutcomeStatus.success,
            confidence=0.95,
            sources=["tests"],
        ),
        complexity=ComplexitySignals(tool_calls=6),
        tool_signature=ToolSignature(
            tool_names=tool_names or ["read_file", "bash", "bash"],
        ),
        skill_usage=(
            SkillUsage(
                used=True,
                skill_name=target_skill,
                skill_path=f"/mnt/skills/custom/{target_skill}/SKILL.md",
                content_hash="b" * 64,
                activation_source="read",
            )
            if patch
            else SkillUsage(used=False)
        ),
        successful_path=["Inspect the environment.", "Run and verify the workflow."],
        reusable_lessons=["Inspect the environment before execution."],
        skill_gaps=(
            [
                SkillGap(
                    category=SkillGapCategory.missing_prerequisite,
                    evidence="The environment check was missing.",
                    recommended_change="Inspect the environment first.",
                )
            ]
            if patch
            else []
        ),
        target_skill=(
            SkillTarget(
                name=target_skill,
                content_hash="b" * 64,
            )
            if patch
            else None
        ),
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


class _FakeEmbeddingProvider:
    def __init__(
        self,
        vectors: dict[str, list[float]],
        *,
        fail: bool = False,
    ) -> None:
        self._vectors = vectors
        self._fail = fail

    async def embed_texts(
        self,
        texts: list[str],
    ) -> EmbeddingBatch:
        if self._fail:
            raise RuntimeError("embedding unavailable")
        vectors = [
            self._vectors.get(
                json.loads(text)["task_signature"],
                [0.0, 1.0, 0.0],
            )
            for text in texts
        ]
        return EmbeddingBatch(
            model_name="test-embed",
            model_version="sha256:test-v1",
            vectors=vectors,
        )


class _FakeVectorStore:
    def __init__(
        self,
        hits: list[SemanticSearchHit] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.hits = hits or []
        self.fail = fail
        self.records: list[Any] = []
        self.query: Any = None

    async def upsert(
        self,
        records: list[Any],
    ) -> None:
        if self.fail:
            raise RuntimeError("vector store unavailable")
        self.records = records

    async def search(
        self,
        query: Any,
    ) -> list[SemanticSearchHit]:
        if self.fail:
            raise RuntimeError("vector store unavailable")
        self.query = query
        return self.hits


@pytest.mark.asyncio
async def test_disabled_semantic_retrieval_uses_deterministic_candidates() -> None:
    query = _event("event-1")
    same_family = _event(
        "event-2",
        task_goal="Add a Python dependency to the project environment.",
    )
    unrelated = _event(
        "event-3",
        task_signature="csv-report-generation",
        task_goal="Generate a CSV report from sales records.",
        tool_names=["read_file", "write_file"],
    )

    result = await retrieve_event_candidates(
        query,
        [same_family, unrelated],
    )

    assert result.mode is SemanticRetrievalMode.deterministic
    assert result.fallback_reason is SemanticFallbackReason.disabled
    assert [match.event_id for match in result.matches] == ["event-2"]
    assert result.matches[0].sources == ["deterministic"]
    assert result.embedding_model_name is None


@pytest.mark.asyncio
async def test_semantic_retrieval_expands_deterministic_candidates() -> None:
    query = _event("event-1")
    semantic_only = _event(
        "event-2",
        task_signature="repair-project-dependency",
        task_goal="Repair a broken dependency inside a project.",
        tool_names=["inspect_environment", "install_dependency", "verify"],
    )
    provider = _FakeEmbeddingProvider(
        {
            "python-package-install": [1.0, 0.0, 0.0],
            "repair-project-dependency": [0.99, 0.01, 0.0],
        }
    )
    store = _FakeVectorStore(
        [
            SemanticSearchHit(
                event_id="event-2",
                score=0.96,
            )
        ]
    )
    config = SkillEvolutionGroupingConfig(
        embedding=SkillEvolutionEmbeddingConfig(
            enabled=True,
            model_name="test-embed",
        )
    )

    result = await retrieve_event_candidates(
        query,
        [semantic_only],
        config=config,
        embedding_provider=provider,
        vector_store=store,
    )

    assert result.mode is SemanticRetrievalMode.hybrid
    assert result.fallback_reason is None
    assert [match.event_id for match in result.matches] == ["event-2"]
    assert result.matches[0].sources == ["semantic"]
    assert result.matches[0].semantic_score == 0.96
    assert result.embedding_model_name == "test-embed"
    assert result.embedding_model_version == "sha256:test-v1"
    assert {record.event_id for record in store.records} == {
        "event-1",
        "event-2",
    }
    assert all(record.embedding_model_name == "test-embed" for record in store.records)
    assert all(record.embedding_model_version == "sha256:test-v1" for record in store.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_fail", "store_fail", "reason"),
    [
        (
            True,
            False,
            SemanticFallbackReason.embedding_unavailable,
        ),
        (
            False,
            True,
            SemanticFallbackReason.vector_store_unavailable,
        ),
    ],
)
async def test_semantic_failure_falls_back_to_deterministic_candidates(
    provider_fail: bool,
    store_fail: bool,
    reason: SemanticFallbackReason,
) -> None:
    query = _event("event-1")
    same_family = _event("event-2")
    provider = _FakeEmbeddingProvider(
        {},
        fail=provider_fail,
    )
    store = _FakeVectorStore(fail=store_fail)
    config = SkillEvolutionGroupingConfig(
        embedding=SkillEvolutionEmbeddingConfig(
            enabled=True,
            model_name="test-embed",
        )
    )

    result = await retrieve_event_candidates(
        query,
        [same_family],
        config=config,
        embedding_provider=provider,
        vector_store=store,
    )

    assert result.mode is SemanticRetrievalMode.deterministic
    assert result.fallback_reason is reason
    assert [match.event_id for match in result.matches] == ["event-2"]
    assert result.matches[0].sources == ["deterministic"]


@pytest.mark.asyncio
async def test_semantic_retrieval_enforces_user_partition_before_embedding() -> None:
    query = _event("event-1", user_id="alice")
    other_user = _event("event-2", user_id="bob")
    provider = _FakeEmbeddingProvider({})
    store = _FakeVectorStore()
    config = SkillEvolutionGroupingConfig(
        embedding=SkillEvolutionEmbeddingConfig(
            enabled=True,
            model_name="test-embed",
        )
    )

    result = await retrieve_event_candidates(
        query,
        [other_user],
        config=config,
        embedding_provider=provider,
        vector_store=store,
    )

    assert result.matches == []
    assert {record.event_id for record in store.records} == {"event-1"}
    assert store.query.user_id == "alice"


@pytest.mark.asyncio
async def test_semantic_retrieval_enforces_patch_target_before_embedding() -> None:
    query = _event(
        "event-1",
        target_skill="package-repair",
    )
    same_target = _event(
        "event-2",
        target_skill="package-repair",
    )
    other_target = _event(
        "event-3",
        target_skill="environment-setup",
    )
    provider = _FakeEmbeddingProvider({})
    store = _FakeVectorStore(
        [
            SemanticSearchHit(
                event_id="event-2",
                score=0.94,
            )
        ]
    )
    config = SkillEvolutionGroupingConfig(
        embedding=SkillEvolutionEmbeddingConfig(
            enabled=True,
            model_name="test-embed",
        )
    )

    result = await retrieve_event_candidates(
        query,
        [same_target, other_target],
        config=config,
        embedding_provider=provider,
        vector_store=store,
    )

    assert [match.event_id for match in result.matches] == ["event-2"]
    assert {record.event_id for record in store.records} == {
        "event-1",
        "event-2",
    }
    assert store.query.target_skill_name == "package-repair"


@pytest.mark.asyncio
async def test_ollama_provider_records_resolved_model_digest() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "nomic-embed-text:latest",
                            "model": "nomic-embed-text:latest",
                            "digest": "d" * 64,
                        }
                    ]
                },
            )
        if request.url.path == "/api/embed":
            return httpx.Response(
                200,
                json={
                    "model": "nomic-embed-text:latest",
                    "embeddings": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                },
            )
        return httpx.Response(404)

    provider = OllamaEmbeddingProvider(
        base_url="http://ollama.test",
        model_name="nomic-embed-text:latest",
        transport=httpx.MockTransport(handler),
    )

    batch = await provider.embed_texts(["one", "two"])

    assert batch.model_name == "nomic-embed-text:latest"
    assert batch.model_version == f"sha256:{'d' * 64}"
    assert batch.dimension == 3
    assert [request.url.path for request in requests] == [
        "/api/tags",
        "/api/embed",
    ]


@pytest.mark.asyncio
async def test_qdrant_store_persists_model_metadata_and_hard_filters() -> None:
    requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append(
            (
                request.method,
                request.url.path,
                body,
            )
        )
        if request.method == "GET":
            return httpx.Response(404, json={"status": {"error": "missing"}})
        if request.method == "PUT" and "/points" not in request.url.path:
            return httpx.Response(200, json={"result": True, "status": "ok"})
        if request.method == "PUT":
            return httpx.Response(200, json={"result": {"status": "completed"}, "status": "ok"})
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {
                                "id": "00000000-0000-0000-0000-000000000001",
                                "score": 0.93,
                                "payload": {"event_id": "event-2"},
                            }
                        ]
                    },
                    "status": "ok",
                },
            )
        return httpx.Response(500)

    provider = _FakeEmbeddingProvider({})
    store = QdrantSemanticVectorStore(
        url="http://qdrant.test",
        collection_name="deerflow_skill_evolution",
        transport=httpx.MockTransport(handler),
    )
    config = SkillEvolutionGroupingConfig(
        semantic_threshold=0.82,
        semantic_top_k=7,
        embedding=SkillEvolutionEmbeddingConfig(
            enabled=True,
            model_name="test-embed",
        ),
    )

    result = await retrieve_event_candidates(
        _event("event-1"),
        [_event("event-2")],
        config=config,
        embedding_provider=provider,
        vector_store=store,
    )

    assert result.matches[0].semantic_score == 0.93
    create_request = next(item for item in requests if item[0] == "PUT" and "/points" not in item[1])
    assert create_request[2] == {
        "vectors": {
            "size": 3,
            "distance": "Cosine",
        }
    }
    upsert_request = next(item for item in requests if item[0] == "PUT" and "/points" in item[1])
    payload = upsert_request[2]["points"][0]["payload"]
    assert payload["embedding_model_name"] == "test-embed"
    assert payload["embedding_model_version"] == "sha256:test-v1"
    query_request = next(item for item in requests if item[0] == "POST")
    assert query_request[2]["limit"] == 7
    assert query_request[2]["score_threshold"] == 0.82
    must_filters = query_request[2]["filter"]["must"]
    assert {"key": "user_id", "match": {"value": "user-1"}} in must_filters
    assert {
        "key": "event_kind",
        "match": {"value": "new_skill_evidence"},
    } in must_filters
    assert {
        "key": "embedding_model_version",
        "match": {"value": "sha256:test-v1"},
    } in must_filters


def test_semantic_config_is_optional_by_default() -> None:
    config = SkillEvolutionGroupingConfig()

    assert config.semantic_threshold == 0.82
    assert config.semantic_top_k == 16
    assert config.embedding.enabled is False
    assert config.embedding.model_name is None
    assert config.vector_store == SkillEvolutionVectorStoreConfig()


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("DEER_FLOW_RUN_QDRANT_TESTS") != "1",
    reason="requires explicit live Ollama/Qdrant opt-in",
)
@pytest.mark.asyncio
async def test_live_ollama_qdrant_semantic_retrieval() -> None:
    qdrant_url = os.getenv(
        "DEER_FLOW_TEST_QDRANT_URL",
        "http://127.0.0.1:6333",
    )
    ollama_url = os.getenv(
        "DEER_FLOW_TEST_OLLAMA_URL",
        "http://127.0.0.1:11434",
    )
    collection_base = f"deerflow_skill_evolution_smoke_{uuid.uuid4().hex[:8]}"
    provider = OllamaEmbeddingProvider(
        base_url=ollama_url,
        model_name="nomic-embed-text",
    )
    store = QdrantSemanticVectorStore(
        url=qdrant_url,
        collection_name=collection_base,
    )
    config = SkillEvolutionGroupingConfig(
        semantic_threshold=0.0,
        semantic_top_k=4,
        embedding=SkillEvolutionEmbeddingConfig(
            enabled=True,
            model_name="nomic-embed-text",
        ),
    )
    try:
        result = await retrieve_event_candidates(
            _event("event-1"),
            [
                _event(
                    "event-2",
                    task_signature="repair-python-dependency",
                    task_goal="Repair a Python dependency in the project environment.",
                )
            ],
            config=config,
            embedding_provider=provider,
            vector_store=store,
        )

        assert result.mode is SemanticRetrievalMode.hybrid
        assert result.embedding_model_version is not None
        assert result.embedding_model_version.startswith("sha256:")
        assert [match.event_id for match in result.matches] == ["event-2"]
        assert result.matches[0].sources == ["semantic"]
    finally:
        async with httpx.AsyncClient(
            base_url=qdrant_url,
            timeout=10.0,
        ) as client:
            collections_response = await client.get("/collections")
            collections_response.raise_for_status()
            collections = collections_response.json()["result"]["collections"]
            for collection in collections:
                name = collection["name"]
                if not name.startswith(f"{collection_base}_"):
                    continue
                response = await client.delete(f"/collections/{name}")
                response.raise_for_status()
