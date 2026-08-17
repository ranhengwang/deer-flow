"""Optional semantic candidate retrieval with deterministic fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import uuid
from enum import StrEnum
from typing import Any, Literal, Protocol, Self

import httpx
from pydantic import Field, model_validator

from deerflow.config.skill_evolution_config import SkillEvolutionGroupingConfig
from deerflow.skill_evolution.grouping import (
    build_event_fingerprint,
    compare_event_fingerprints,
)
from deerflow.skill_evolution.models import (
    EvolutionEvent,
    EvolutionEventKind,
    EvolutionModel,
    Identifier,
    Sha256,
)

logger = logging.getLogger(__name__)

_MAX_RETRIEVAL_EVENTS = 1_024
_MAX_EMBEDDING_DIMENSION = 16_384
_MAX_SEMANTIC_DOCUMENT_CHARS = 8_192
_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot produce a valid batch."""


class SemanticVectorStoreError(RuntimeError):
    """Raised when semantic vectors cannot be persisted or queried."""


class SemanticRetrievalMode(StrEnum):
    deterministic = "deterministic"
    hybrid = "hybrid"


class SemanticFallbackReason(StrEnum):
    disabled = "disabled"
    embedding_unavailable = "embedding_unavailable"
    vector_store_unavailable = "vector_store_unavailable"


class EmbeddingBatch(EvolutionModel):
    """One versioned embedding response."""

    model_name: Identifier
    model_version: Identifier
    vectors: list[list[float]] = Field(min_length=1, max_length=_MAX_RETRIEVAL_EVENTS + 1)

    @model_validator(mode="after")
    def _validate_vectors(self) -> Self:
        dimension = len(self.vectors[0])
        if dimension < 1 or dimension > _MAX_EMBEDDING_DIMENSION:
            raise ValueError("embedding dimension is outside the supported range")
        for vector in self.vectors:
            if len(vector) != dimension:
                raise ValueError("embedding vectors must have one consistent dimension")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("embedding vectors must contain only finite values")
        return self

    @property
    def dimension(self) -> int:
        return len(self.vectors[0])


class EmbeddingProvider(Protocol):
    """Provider-neutral async embedding contract."""

    async def embed_texts(
        self,
        texts: list[str],
    ) -> EmbeddingBatch:
        """Embed texts and return immutable model identity with the vectors."""


class SemanticVectorRecord(EvolutionModel):
    event_id: Identifier
    user_id: Identifier
    event_kind: EvolutionEventKind
    target_skill_name: Identifier | None = None
    document_hash: Sha256
    embedding_model_name: Identifier
    embedding_model_version: Identifier
    vector: list[float] = Field(
        min_length=1,
        max_length=_MAX_EMBEDDING_DIMENSION,
    )


class SemanticSearchQuery(EvolutionModel):
    query_event_id: Identifier
    user_id: Identifier
    event_kind: EvolutionEventKind
    target_skill_name: Identifier | None = None
    embedding_model_name: Identifier
    embedding_model_version: Identifier
    vector: list[float] = Field(
        min_length=1,
        max_length=_MAX_EMBEDDING_DIMENSION,
    )
    score_threshold: float = Field(ge=0.0, le=1.0)
    limit: int = Field(ge=1, le=64)


class SemanticSearchHit(EvolutionModel):
    event_id: Identifier
    score: float = Field(ge=0.0, le=1.0)


class SemanticVectorStore(Protocol):
    """Persistence/search contract for versioned event vectors."""

    async def upsert(
        self,
        records: list[SemanticVectorRecord],
    ) -> None:
        """Persist one model-version batch."""

    async def search(
        self,
        query: SemanticSearchQuery,
    ) -> list[SemanticSearchHit]:
        """Return same-partition semantic candidates."""


class SemanticCandidateMatch(EvolutionModel):
    event_id: Identifier
    sources: list[Literal["deterministic", "semantic"]] = Field(
        min_length=1,
        max_length=2,
    )
    deterministic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    semantic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def _validate_sources(self) -> Self:
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("candidate match sources must be unique")
        if "deterministic" in self.sources and self.deterministic_score is None:
            raise ValueError("deterministic source requires a deterministic score")
        if "semantic" in self.sources and self.semantic_score is None:
            raise ValueError("semantic source requires a semantic score")
        return self


class SemanticCandidateRetrievalResult(EvolutionModel):
    query_event_id: Identifier
    mode: SemanticRetrievalMode
    matches: list[SemanticCandidateMatch] = Field(
        max_length=_MAX_RETRIEVAL_EVENTS,
    )
    fallback_reason: SemanticFallbackReason | None = None
    embedding_model_name: Identifier | None = None
    embedding_model_version: Identifier | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> Self:
        if self.mode is SemanticRetrievalMode.deterministic:
            if self.fallback_reason is None:
                raise ValueError("deterministic retrieval requires a fallback reason")
            if self.embedding_model_name is not None or self.embedding_model_version is not None:
                raise ValueError("deterministic retrieval must not report embedding metadata")
        else:
            if self.fallback_reason is not None:
                raise ValueError("hybrid retrieval must not report a fallback reason")
            if self.embedding_model_name is None or self.embedding_model_version is None:
                raise ValueError("hybrid retrieval requires embedding metadata")
        return self


class OllamaEmbeddingProvider:
    """Embedding provider backed by Ollama's native HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        model_version: str | None = None,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._model_version = model_version
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def _resolve_model_version(
        self,
        client: httpx.AsyncClient,
    ) -> str:
        if self._model_version is not None:
            return self._model_version
        response = await client.get("/api/tags")
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            raise EmbeddingProviderError("Ollama model catalog is malformed")
        requested_base = self._model_name.split(":", 1)[0]
        for model in models:
            if not isinstance(model, dict):
                continue
            names = {
                str(model.get("name") or ""),
                str(model.get("model") or ""),
            }
            matched = self._model_name in names or (":" not in self._model_name and any(name.split(":", 1)[0] == requested_base for name in names if name))
            if not matched:
                continue
            digest = model.get("digest")
            if not isinstance(digest, str) or not digest.strip():
                raise EmbeddingProviderError("Ollama model digest is unavailable")
            normalized = digest.strip()
            return normalized if normalized.startswith("sha256:") else f"sha256:{normalized}"
        raise EmbeddingProviderError("Ollama embedding model is not installed")

    async def embed_texts(
        self,
        texts: list[str],
    ) -> EmbeddingBatch:
        if not texts or len(texts) > _MAX_RETRIEVAL_EVENTS + 1:
            raise EmbeddingProviderError("embedding batch size is outside the supported range")
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                model_version = await self._resolve_model_version(client)
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": self._model_name,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError("Ollama embedding request failed") from exc

        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingProviderError("Ollama returned an invalid embedding count")
        try:
            return EmbeddingBatch(
                model_name=self._model_name,
                model_version=model_version,
                vectors=vectors,
            )
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError("Ollama returned invalid embedding vectors") from exc


class QdrantSemanticVectorStore:
    """Minimal Qdrant REST adapter for versioned evolution-event vectors."""

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not _COLLECTION_NAME_RE.fullmatch(collection_name):
            raise ValueError("Qdrant collection_name contains unsupported characters")
        self._url = url.rstrip("/")
        self._collection_name = collection_name
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._ready_collections: set[str] = set()
        self._collection_lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        return {"api-key": self._api_key} if self._api_key else {}

    def _physical_collection(
        self,
        model_name: str,
        model_version: str,
    ) -> str:
        identity = hashlib.sha256(
            f"{model_name}\0{model_version}".encode(),
        ).hexdigest()[:16]
        return f"{self._collection_name}_{identity}"

    async def _validate_existing_collection(
        self,
        response: httpx.Response,
        *,
        dimension: int,
    ) -> None:
        try:
            vectors = response.json()["result"]["config"]["params"]["vectors"]
            size = vectors["size"]
            distance = vectors["distance"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SemanticVectorStoreError("Qdrant collection metadata is malformed") from exc
        if size != dimension or str(distance).casefold() != "cosine":
            raise SemanticVectorStoreError("Qdrant collection vector schema does not match")

    async def _ensure_collection(
        self,
        client: httpx.AsyncClient,
        *,
        collection: str,
        dimension: int,
    ) -> None:
        if collection in self._ready_collections:
            return
        async with self._collection_lock:
            if collection in self._ready_collections:
                return
            response = await client.get(f"/collections/{collection}")
            if response.status_code == 200:
                await self._validate_existing_collection(
                    response,
                    dimension=dimension,
                )
            elif response.status_code == 404:
                create = await client.put(
                    f"/collections/{collection}",
                    json={
                        "vectors": {
                            "size": dimension,
                            "distance": "Cosine",
                        }
                    },
                )
                if create.status_code == 409:
                    retry = await client.get(f"/collections/{collection}")
                    retry.raise_for_status()
                    await self._validate_existing_collection(
                        retry,
                        dimension=dimension,
                    )
                else:
                    create.raise_for_status()
            else:
                response.raise_for_status()
            self._ready_collections.add(collection)

    async def upsert(
        self,
        records: list[SemanticVectorRecord],
    ) -> None:
        if not records:
            return
        first = records[0]
        dimension = len(first.vector)
        if any(len(record.vector) != dimension or record.embedding_model_name != first.embedding_model_name or record.embedding_model_version != first.embedding_model_version for record in records):
            raise SemanticVectorStoreError("Qdrant upsert batch must use one model and dimension")
        collection = self._physical_collection(
            first.embedding_model_name,
            first.embedding_model_version,
        )
        points = []
        for record in records:
            point_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                "\0".join(
                    (
                        "deerflow.skill-evolution",
                        record.user_id,
                        record.event_id,
                        record.embedding_model_name,
                        record.embedding_model_version,
                    )
                ),
            )
            points.append(
                {
                    "id": str(point_id),
                    "vector": record.vector,
                    "payload": {
                        "event_id": record.event_id,
                        "user_id": record.user_id,
                        "event_kind": record.event_kind.value,
                        "target_skill_name": record.target_skill_name,
                        "document_hash": record.document_hash,
                        "embedding_model_name": record.embedding_model_name,
                        "embedding_model_version": record.embedding_model_version,
                    },
                }
            )
        try:
            async with httpx.AsyncClient(
                base_url=self._url,
                headers=self._headers(),
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                await self._ensure_collection(
                    client,
                    collection=collection,
                    dimension=dimension,
                )
                response = await client.put(
                    f"/collections/{collection}/points",
                    params={"wait": "true"},
                    json={"points": points},
                )
                response.raise_for_status()
        except SemanticVectorStoreError:
            raise
        except Exception as exc:
            raise SemanticVectorStoreError("Qdrant vector upsert failed") from exc

    async def search(
        self,
        query: SemanticSearchQuery,
    ) -> list[SemanticSearchHit]:
        collection = self._physical_collection(
            query.embedding_model_name,
            query.embedding_model_version,
        )
        must: list[dict[str, Any]] = [
            {
                "key": "user_id",
                "match": {"value": query.user_id},
            },
            {
                "key": "event_kind",
                "match": {"value": query.event_kind.value},
            },
            {
                "key": "embedding_model_name",
                "match": {"value": query.embedding_model_name},
            },
            {
                "key": "embedding_model_version",
                "match": {"value": query.embedding_model_version},
            },
        ]
        if query.target_skill_name is not None:
            must.append(
                {
                    "key": "target_skill_name",
                    "match": {"value": query.target_skill_name},
                }
            )
        try:
            async with httpx.AsyncClient(
                base_url=self._url,
                headers=self._headers(),
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"/collections/{collection}/points/query",
                    json={
                        "query": query.vector,
                        "filter": {"must": must},
                        "limit": query.limit,
                        "score_threshold": query.score_threshold,
                        "with_payload": True,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise SemanticVectorStoreError("Qdrant vector search failed") from exc
        points = payload.get("result", {}).get("points")
        if not isinstance(points, list):
            raise SemanticVectorStoreError("Qdrant search response is malformed")
        hits: list[SemanticSearchHit] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            point_payload = point.get("payload")
            event_id = point_payload.get("event_id") if isinstance(point_payload, dict) else None
            score = point.get("score")
            if not isinstance(event_id, str) or not isinstance(score, int | float):
                continue
            if event_id == query.query_event_id:
                continue
            try:
                hits.append(
                    SemanticSearchHit(
                        event_id=event_id,
                        score=float(score),
                    )
                )
            except ValueError:
                continue
        return hits


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def build_semantic_event_document(
    event: EvolutionEvent,
) -> str:
    """Build bounded structured text without event/user identity."""
    value = _stable_json(
        {
            "task_signature": event.task_signature,
            "task_goal": event.task_goal,
            "tool_sequence": event.tool_signature.tool_names,
            "error_types": event.tool_signature.error_types,
            "environment": event.environment.model_dump(mode="json"),
            "successful_path": event.successful_path,
            "reusable_lessons": event.reusable_lessons,
            "skill_gaps": [
                {
                    "category": gap.category.value,
                    "evidence": gap.evidence,
                    "recommended_change": gap.recommended_change,
                }
                for gap in event.skill_gaps
            ],
        }
    )
    if len(value) <= _MAX_SEMANTIC_DOCUMENT_CHARS:
        return value
    return f"{value[: _MAX_SEMANTIC_DOCUMENT_CHARS - 81]}|sha256:{_sha256(value)}"


def _target_skill_name(
    event: EvolutionEvent,
) -> str | None:
    return event.target_skill.name if event.target_skill is not None else None


def _deterministic_candidates(
    query_event: EvolutionEvent,
    candidate_events: list[EvolutionEvent],
    *,
    threshold: float,
) -> tuple[
    dict[str, EvolutionEvent],
    dict[str, float],
    dict[str, float],
]:
    query_fingerprint = build_event_fingerprint(query_event)
    eligible: dict[str, EvolutionEvent] = {}
    deterministic_scores: dict[str, float] = {}
    compatible_scores: dict[str, float] = {}
    for event in candidate_events:
        if event.event_id == query_event.event_id:
            continue
        comparison = compare_event_fingerprints(
            query_fingerprint,
            build_event_fingerprint(event),
            threshold=threshold,
        )
        if comparison.hard_mismatch is not None:
            continue
        existing = eligible.get(event.event_id)
        if existing is not None and existing != event:
            raise ValueError("event ID contains conflicting retrieval payloads")
        eligible[event.event_id] = event
        deterministic_scores[event.event_id] = comparison.score
        if comparison.compatible:
            compatible_scores[event.event_id] = comparison.score
    return eligible, deterministic_scores, compatible_scores


def _deterministic_result(
    query_event: EvolutionEvent,
    compatible_scores: dict[str, float],
    *,
    reason: SemanticFallbackReason,
) -> SemanticCandidateRetrievalResult:
    matches = [
        SemanticCandidateMatch(
            event_id=event_id,
            sources=["deterministic"],
            deterministic_score=score,
        )
        for event_id, score in sorted(
            compatible_scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    ]
    return SemanticCandidateRetrievalResult(
        query_event_id=query_event.event_id,
        mode=SemanticRetrievalMode.deterministic,
        matches=matches,
        fallback_reason=reason,
    )


def _build_default_provider(
    config: SkillEvolutionGroupingConfig,
) -> EmbeddingProvider:
    embedding = config.embedding
    if embedding.model_name is None:
        raise EmbeddingProviderError("embedding model is not configured")
    return OllamaEmbeddingProvider(
        base_url=embedding.base_url,
        model_name=embedding.model_name,
        model_version=embedding.model_version,
        timeout_seconds=embedding.timeout_seconds,
    )


def _build_default_vector_store(
    config: SkillEvolutionGroupingConfig,
) -> SemanticVectorStore:
    vector_store = config.vector_store
    api_key = os.getenv(vector_store.api_key_env) if vector_store.api_key_env is not None else None
    return QdrantSemanticVectorStore(
        url=vector_store.url,
        collection_name=vector_store.collection_name,
        api_key=api_key,
        timeout_seconds=vector_store.timeout_seconds,
    )


async def retrieve_event_candidates(
    query_event: EvolutionEvent,
    candidate_events: list[EvolutionEvent],
    *,
    config: SkillEvolutionGroupingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: SemanticVectorStore | None = None,
) -> SemanticCandidateRetrievalResult:
    """Retrieve candidate events semantically, failing open to deterministic scoring."""
    if len(candidate_events) > _MAX_RETRIEVAL_EVENTS:
        raise ValueError(f"at most {_MAX_RETRIEVAL_EVENTS} candidate events may be retrieved")
    resolved_config = config or SkillEvolutionGroupingConfig()
    eligible, deterministic_scores, compatible_scores = _deterministic_candidates(
        query_event,
        candidate_events,
        threshold=resolved_config.deterministic_threshold,
    )
    if not resolved_config.embedding.enabled:
        return _deterministic_result(
            query_event,
            compatible_scores,
            reason=SemanticFallbackReason.disabled,
        )

    try:
        provider = embedding_provider or _build_default_provider(resolved_config)
        ordered_events = [
            query_event,
            *sorted(
                eligible.values(),
                key=lambda event: (
                    event.created_at,
                    event.event_id,
                ),
            ),
        ]
        documents = [build_semantic_event_document(event) for event in ordered_events]
        batch = await provider.embed_texts(documents)
        if len(batch.vectors) != len(ordered_events):
            raise EmbeddingProviderError("embedding provider returned the wrong vector count")
    except Exception as exc:
        logger.warning(
            "Semantic event retrieval fell back after embedding failure: %s",
            type(exc).__name__,
        )
        return _deterministic_result(
            query_event,
            compatible_scores,
            reason=SemanticFallbackReason.embedding_unavailable,
        )

    records = [
        SemanticVectorRecord(
            event_id=event.event_id,
            user_id=event.user_id,
            event_kind=event.event_kind,
            target_skill_name=_target_skill_name(event),
            document_hash=_sha256(document),
            embedding_model_name=batch.model_name,
            embedding_model_version=batch.model_version,
            vector=vector,
        )
        for event, document, vector in zip(
            ordered_events,
            documents,
            batch.vectors,
            strict=True,
        )
    ]
    try:
        store = vector_store or _build_default_vector_store(resolved_config)
        await store.upsert(records)
        hits = await store.search(
            SemanticSearchQuery(
                query_event_id=query_event.event_id,
                user_id=query_event.user_id,
                event_kind=query_event.event_kind,
                target_skill_name=_target_skill_name(query_event),
                embedding_model_name=batch.model_name,
                embedding_model_version=batch.model_version,
                vector=batch.vectors[0],
                score_threshold=resolved_config.semantic_threshold,
                limit=resolved_config.semantic_top_k,
            )
        )
    except Exception as exc:
        logger.warning(
            "Semantic event retrieval fell back after vector-store failure: %s",
            type(exc).__name__,
        )
        return _deterministic_result(
            query_event,
            compatible_scores,
            reason=SemanticFallbackReason.vector_store_unavailable,
        )

    semantic_scores: dict[str, float] = {}
    for hit in hits:
        if hit.event_id not in eligible:
            continue
        semantic_scores[hit.event_id] = max(
            semantic_scores.get(hit.event_id, 0.0),
            hit.score,
        )
    matches: list[SemanticCandidateMatch] = []
    for event_id in set(compatible_scores) | set(semantic_scores):
        sources: list[Literal["deterministic", "semantic"]] = []
        if event_id in compatible_scores:
            sources.append("deterministic")
        if event_id in semantic_scores:
            sources.append("semantic")
        matches.append(
            SemanticCandidateMatch(
                event_id=event_id,
                sources=sources,
                deterministic_score=deterministic_scores.get(event_id),
                semantic_score=semantic_scores.get(event_id),
            )
        )
    matches.sort(
        key=lambda match: (
            -(match.semantic_score or 0.0),
            -(match.deterministic_score or 0.0),
            match.event_id,
        )
    )
    return SemanticCandidateRetrievalResult(
        query_event_id=query_event.event_id,
        mode=SemanticRetrievalMode.hybrid,
        matches=matches,
        embedding_model_name=batch.model_name,
        embedding_model_version=batch.model_version,
    )
