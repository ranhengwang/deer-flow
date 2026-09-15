"""Low-cardinality, content-free observability for Skill evolution."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import Counter, OrderedDict, deque
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from deerflow.skill_evolution.models import (
    EvolutionModel,
    Identifier,
    Sha256,
    SkillEvaluation,
)

logger = logging.getLogger(__name__)

EVOLUTION_OBSERVABILITY_SCHEMA_VERSION = "deerflow.skill-evolution.observation.v1"

_COUNTER_NAMES = frozenset({"extraction_failures"})
_GAUGE_NAMES = frozenset({"queue_depth", "active_jobs"})
_DISTRIBUTION_RANGES = {
    "cluster_purity": (0.0, 1.0),
    "regression_rate": (0.0, 1.0),
    "quality_lift": (-1.0, 1.0),
}


class EvolutionLifecycleKind(StrEnum):
    admitted = "admitted"
    rejected = "rejected"
    extracted = "extracted"
    clustered = "clustered"
    ready = "ready"
    distilled = "distilled"
    evaluated = "evaluated"
    approved = "approved"
    published = "published"
    rolled_back = "rolled_back"


class EvolutionLifecycleEvent(EvolutionModel):
    """Strict content-free lifecycle record suitable for structured logs."""

    schema_version: Literal["deerflow.skill-evolution.observation.v1"] = EVOLUTION_OBSERVABILITY_SCHEMA_VERSION
    observation_id: Identifier
    kind: EvolutionLifecycleKind
    stage: Identifier
    user_id_hash: Sha256
    skill_name_hash: Sha256 | None = None
    run_id: Identifier | None = None
    job_id: Identifier | None = None
    evolution_event_id: Identifier | None = None
    cluster_id: Identifier | None = None
    proposal_id: Identifier | None = None
    evaluation_id: Identifier | None = None
    publication_id: Identifier | None = None
    snapshot_hash: Sha256 | None = None
    decision: Identifier | None = None
    reason_codes: list[Identifier] = Field(
        default_factory=list,
        max_length=16,
    )
    event_count: int | None = Field(default=None, ge=0)
    distinct_run_count: int | None = Field(default=None, ge=0)
    candidate_count: int | None = Field(default=None, ge=0)
    accepted_count: int | None = Field(default=None, ge=0)
    occurred_at: datetime


class EvolutionDistributionSnapshot(EvolutionModel):
    count: int = Field(ge=0)
    total: float
    minimum: float | None = None
    maximum: float | None = None
    average: float | None = None
    last: float | None = None


class EvolutionMetricsSnapshot(EvolutionModel):
    """Process-local metrics with no user- or Skill-cardinality labels."""

    queue_depth: int = Field(ge=0)
    active_jobs: int = Field(ge=0)
    extraction_failures: int = Field(ge=0)
    lifecycle_counts: dict[str, int]
    proposal_pass_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    publication_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    cluster_purity: EvolutionDistributionSnapshot
    regression_rate: EvolutionDistributionSnapshot
    quality_lift: EvolutionDistributionSnapshot
    generated_at: datetime


class _Distribution:
    __slots__ = ("count", "total", "minimum", "maximum", "last")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.last: float | None = None

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.last = value

    def snapshot(self) -> EvolutionDistributionSnapshot:
        return EvolutionDistributionSnapshot(
            count=self.count,
            total=round(self.total, 6),
            minimum=self.minimum,
            maximum=self.maximum,
            average=(round(self.total / self.count, 6) if self.count else None),
            last=self.last,
        )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


class EvolutionObservability:
    """Thread-safe bounded event buffer and process-local metric registry."""

    def __init__(
        self,
        *,
        max_recent_events: int = 512,
        max_deduplication_keys: int = 4_096,
    ) -> None:
        if max_recent_events < 1 or max_recent_events > 10_000:
            raise ValueError("max_recent_events must be between 1 and 10000")
        if max_deduplication_keys < max_recent_events or max_deduplication_keys > 100_000:
            raise ValueError("max_deduplication_keys must be at least max_recent_events and at most 100000")
        self._lock = threading.RLock()
        self._recent: deque[EvolutionLifecycleEvent] = deque(maxlen=max_recent_events)
        self._seen: OrderedDict[
            str,
            EvolutionLifecycleEvent,
        ] = OrderedDict()
        self._max_deduplication_keys = max_deduplication_keys
        self._lifecycle_counts: Counter[str] = Counter()
        self._counters: Counter[str] = Counter()
        self._gauges = {
            "queue_depth": 0,
            "active_jobs": 0,
        }
        self._distributions = {name: _Distribution() for name in _DISTRIBUTION_RANGES}
        self._metric_seen: OrderedDict[
            tuple[str, str],
            None,
        ] = OrderedDict()
        self._evaluation_total = 0
        self._evaluation_approved = 0
        self._distillation_total = 0
        self._publication_total = 0

    def emit(
        self,
        *,
        kind: EvolutionLifecycleKind | str,
        stage: str,
        user_id: str,
        skill_name: str | None = None,
        run_id: str | None = None,
        job_id: str | None = None,
        evolution_event_id: str | None = None,
        cluster_id: str | None = None,
        proposal_id: str | None = None,
        evaluation_id: str | None = None,
        publication_id: str | None = None,
        snapshot_hash: str | None = None,
        decision: str | None = None,
        reason_codes: list[str] | tuple[str, ...] = (),
        event_count: int | None = None,
        distinct_run_count: int | None = None,
        candidate_count: int | None = None,
        accepted_count: int | None = None,
        occurred_at: datetime | None = None,
    ) -> EvolutionLifecycleEvent:
        resolved_kind = EvolutionLifecycleKind(kind)
        identity = {
            "kind": resolved_kind.value,
            "stage": stage,
            "user_id_hash": _sha256(user_id),
            "skill_name_hash": (_sha256(skill_name) if skill_name is not None else None),
            "run_id": run_id,
            "job_id": job_id,
            "evolution_event_id": evolution_event_id,
            "cluster_id": cluster_id,
            "proposal_id": proposal_id,
            "evaluation_id": evaluation_id,
            "publication_id": publication_id,
            "snapshot_hash": snapshot_hash,
            "decision": decision,
            "reason_codes": list(reason_codes),
            "event_count": event_count,
            "distinct_run_count": distinct_run_count,
            "candidate_count": candidate_count,
            "accepted_count": accepted_count,
        }
        digest = _sha256(_stable_json(identity))
        event = EvolutionLifecycleEvent(
            observation_id=f"observation-{digest[:32]}",
            kind=resolved_kind,
            stage=stage,
            user_id_hash=identity["user_id_hash"],
            skill_name_hash=identity["skill_name_hash"],
            run_id=run_id,
            job_id=job_id,
            evolution_event_id=evolution_event_id,
            cluster_id=cluster_id,
            proposal_id=proposal_id,
            evaluation_id=evaluation_id,
            publication_id=publication_id,
            snapshot_hash=snapshot_hash,
            decision=decision,
            reason_codes=list(reason_codes),
            event_count=event_count,
            distinct_run_count=distinct_run_count,
            candidate_count=candidate_count,
            accepted_count=accepted_count,
            occurred_at=occurred_at or datetime.now(UTC),
        )
        with self._lock:
            existing = self._seen.get(event.observation_id)
            if existing is not None:
                self._seen.move_to_end(event.observation_id)
                return existing
            self._seen[event.observation_id] = event
            if len(self._seen) > self._max_deduplication_keys:
                self._seen.popitem(last=False)
            self._recent.append(event)
            self._lifecycle_counts[event.kind.value] += 1
            if event.kind is EvolutionLifecycleKind.evaluated:
                self._evaluation_total += 1
                if event.decision == "approve":
                    self._evaluation_approved += 1
            elif event.kind is EvolutionLifecycleKind.distilled:
                self._distillation_total += 1
            elif event.kind is EvolutionLifecycleKind.published:
                self._publication_total += 1
        logger.info(
            "skill_evolution_lifecycle %s",
            event.model_dump_json(),
        )
        return event

    def increment(
        self,
        name: str,
        amount: int = 1,
    ) -> None:
        if name not in _COUNTER_NAMES:
            raise ValueError(f"unsupported evolution counter {name!r}")
        if amount < 0:
            raise ValueError("counter amount must be non-negative")
        with self._lock:
            self._counters[name] += amount

    def set_gauge(
        self,
        name: str,
        value: int,
    ) -> None:
        if name not in _GAUGE_NAMES:
            raise ValueError(f"unsupported evolution gauge {name!r}")
        if value < 0:
            raise ValueError("gauge value must be non-negative")
        with self._lock:
            self._gauges[name] = value

    def observe(
        self,
        name: str,
        value: float,
        *,
        deduplication_key: str | None = None,
    ) -> None:
        bounds = _DISTRIBUTION_RANGES.get(name)
        if bounds is None:
            raise ValueError(f"unsupported evolution distribution {name!r}")
        numeric = float(value)
        if not bounds[0] <= numeric <= bounds[1]:
            raise ValueError(f"{name} must be between {bounds[0]} and {bounds[1]}")
        if deduplication_key is not None and (not deduplication_key or len(deduplication_key) > 512):
            raise ValueError("metric deduplication_key must contain 1 to 512 characters")
        with self._lock:
            if deduplication_key is not None:
                key = (name, deduplication_key)
                if key in self._metric_seen:
                    self._metric_seen.move_to_end(key)
                    return
                self._metric_seen[key] = None
                if len(self._metric_seen) > self._max_deduplication_keys:
                    self._metric_seen.popitem(last=False)
            self._distributions[name].observe(numeric)

    def recent_events(self) -> list[EvolutionLifecycleEvent]:
        with self._lock:
            return list(self._recent)

    def snapshot(self) -> EvolutionMetricsSnapshot:
        with self._lock:
            pass_rate = self._evaluation_approved / self._evaluation_total if self._evaluation_total else None
            publication_rate = self._publication_total / self._distillation_total if self._distillation_total else None
            return EvolutionMetricsSnapshot(
                queue_depth=self._gauges["queue_depth"],
                active_jobs=self._gauges["active_jobs"],
                extraction_failures=self._counters["extraction_failures"],
                lifecycle_counts=dict(sorted(self._lifecycle_counts.items())),
                proposal_pass_rate=(round(pass_rate, 6) if pass_rate is not None else None),
                publication_rate=(round(min(1.0, publication_rate), 6) if publication_rate is not None else None),
                cluster_purity=self._distributions["cluster_purity"].snapshot(),
                regression_rate=self._distributions["regression_rate"].snapshot(),
                quality_lift=self._distributions["quality_lift"].snapshot(),
                generated_at=datetime.now(UTC),
            )


_DEFAULT_OBSERVABILITY = EvolutionObservability()


def get_evolution_observability() -> EvolutionObservability:
    return _DEFAULT_OBSERVABILITY


def safe_emit_evolution_event(
    observer: Any,
    **kwargs: Any,
) -> EvolutionLifecycleEvent | None:
    try:
        return observer.emit(**kwargs)
    except Exception:
        logger.warning(
            "Skill evolution observability operation failed (operation=emit, kind=%s, stage=%s)",
            kwargs.get("kind"),
            kwargs.get("stage"),
        )
        return None


def safe_increment_evolution_metric(
    observer: Any,
    name: str,
    amount: int = 1,
) -> None:
    try:
        observer.increment(name, amount)
    except Exception:
        logger.warning(
            "Skill evolution observability operation failed (operation=increment, metric=%s)",
            name,
        )


def safe_set_evolution_gauge(
    observer: Any,
    name: str,
    value: int,
) -> None:
    try:
        observer.set_gauge(name, value)
    except Exception:
        logger.warning(
            "Skill evolution observability operation failed (operation=set_gauge, metric=%s)",
            name,
        )


def safe_observe_evolution_metric(
    observer: Any,
    name: str,
    value: float,
    *,
    deduplication_key: str | None = None,
) -> None:
    try:
        observer.observe(
            name,
            value,
            deduplication_key=deduplication_key,
        )
    except Exception:
        logger.warning(
            "Skill evolution observability operation failed (operation=observe, metric=%s)",
            name,
        )


def observe_evaluation(
    observer: Any,
    evaluation: SkillEvaluation,
    *,
    skill_name: str | None = None,
) -> None:
    safe_emit_evolution_event(
        observer,
        kind=EvolutionLifecycleKind.evaluated,
        stage="proposal_evaluation",
        user_id=evaluation.user_id,
        skill_name=skill_name,
        proposal_id=evaluation.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        decision=evaluation.decision.value,
        occurred_at=evaluation.created_at,
    )
    if evaluation.decision.value == "reject":
        safe_emit_evolution_event(
            observer,
            kind=EvolutionLifecycleKind.rejected,
            stage="proposal_evaluation",
            user_id=evaluation.user_id,
            skill_name=skill_name,
            proposal_id=evaluation.proposal_id,
            evaluation_id=evaluation.evaluation_id,
            decision=evaluation.decision.value,
            reason_codes=["evaluation_rejected"],
            occurred_at=evaluation.created_at,
        )
    quality = evaluation.quality
    if quality is None:
        return
    regression = quality.dimensions.get("regression_resistance")
    if regression is not None and regression.available and regression.raw_value is not None:
        safe_observe_evolution_metric(
            observer,
            "regression_rate",
            regression.raw_value,
            deduplication_key=evaluation.evaluation_id,
        )
    lift = quality.dimensions.get("success_lift")
    if lift is not None and lift.available and lift.raw_value is not None:
        safe_observe_evolution_metric(
            observer,
            "quality_lift",
            lift.raw_value,
            deduplication_key=evaluation.evaluation_id,
        )
