from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from deerflow.skill_evolution.observability import (
    EvolutionLifecycleKind,
    EvolutionObservability,
    safe_emit_evolution_event,
    safe_increment_evolution_metric,
    safe_observe_evolution_metric,
    safe_set_evolution_gauge,
)


def test_lifecycle_event_hashes_user_and_skill_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observer = EvolutionObservability()
    caplog.set_level(
        logging.INFO,
        logger="deerflow.skill_evolution.observability",
    )

    event = observer.emit(
        kind=EvolutionLifecycleKind.extracted,
        stage="structured_extraction",
        user_id="private-user@example.com",
        skill_name="private-customer-workflow",
        run_id="run-1",
        job_id="job-1",
        evolution_event_id="event-1",
        snapshot_hash="a" * 64,
        occurred_at=datetime(2026, 8, 17, tzinfo=UTC),
    )

    assert event is not None
    assert event.user_id_hash != "private-user@example.com"
    assert event.skill_name_hash is not None
    assert event.run_id == "run-1"
    assert event.snapshot_hash == "a" * 64
    assert "private-user@example.com" not in caplog.text
    assert "private-customer-workflow" not in caplog.text
    assert "event-1" in caplog.text


def test_duplicate_lifecycle_event_is_counted_once() -> None:
    observer = EvolutionObservability()
    kwargs = {
        "kind": EvolutionLifecycleKind.ready,
        "stage": "cluster_confirmation",
        "user_id": "user-1",
        "cluster_id": "cluster-1",
        "event_count": 3,
        "distinct_run_count": 3,
        "occurred_at": datetime(2026, 8, 17, tzinfo=UTC),
    }

    first = observer.emit(**kwargs)
    second = observer.emit(**kwargs)

    assert first is not None
    assert second == first
    assert observer.recent_events() == [first]
    snapshot = observer.snapshot()
    assert snapshot.lifecycle_counts["ready"] == 1


def test_recent_lifecycle_events_are_bounded() -> None:
    observer = EvolutionObservability(max_recent_events=2)

    for index in range(3):
        observer.emit(
            kind=EvolutionLifecycleKind.clustered,
            stage="deterministic_grouping",
            user_id="user-1",
            cluster_id=f"cluster-{index}",
            event_count=index + 1,
        )

    assert [event.cluster_id for event in observer.recent_events()] == ["cluster-1", "cluster-2"]
    assert observer.snapshot().lifecycle_counts["clustered"] == 3


def test_metrics_snapshot_has_fixed_low_cardinality_aggregates() -> None:
    observer = EvolutionObservability()

    observer.set_gauge("queue_depth", 7)
    observer.set_gauge("active_jobs", 2)
    observer.increment("extraction_failures")
    observer.observe("cluster_purity", 0.75)
    observer.observe("cluster_purity", 1.0)
    observer.observe("regression_rate", 0.25)
    observer.observe("quality_lift", 0.4)
    observer.emit(
        kind=EvolutionLifecycleKind.distilled,
        stage="proposal_distillation",
        user_id="user-1",
        proposal_id="proposal-1",
    )
    observer.emit(
        kind=EvolutionLifecycleKind.evaluated,
        stage="proposal_evaluation",
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        decision="approve",
    )
    observer.emit(
        kind=EvolutionLifecycleKind.approved,
        stage="approval_policy",
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
    )
    observer.emit(
        kind=EvolutionLifecycleKind.published,
        stage="publication",
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        publication_id="publication-1",
    )

    snapshot = observer.snapshot()

    assert snapshot.queue_depth == 7
    assert snapshot.active_jobs == 2
    assert snapshot.extraction_failures == 1
    assert snapshot.proposal_pass_rate == 1.0
    assert snapshot.publication_rate == 1.0
    assert snapshot.cluster_purity.count == 2
    assert snapshot.cluster_purity.average == 0.875
    assert snapshot.regression_rate.last == 0.25
    assert snapshot.quality_lift.last == 0.4
    assert "user-1" not in snapshot.model_dump_json()


class _FailingObserver:
    def emit(self, **_kwargs):
        raise RuntimeError("secret observer failure")

    def increment(self, *_args, **_kwargs):
        raise RuntimeError("secret observer failure")

    def observe(self, *_args, **_kwargs):
        raise RuntimeError("secret observer failure")

    def set_gauge(self, *_args, **_kwargs):
        raise RuntimeError("secret observer failure")


def test_safe_observability_helpers_never_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observer = _FailingObserver()
    caplog.set_level(
        logging.WARNING,
        logger="deerflow.skill_evolution.observability",
    )

    assert (
        safe_emit_evolution_event(
            observer,
            kind=EvolutionLifecycleKind.rejected,
            stage="eligibility",
            user_id="user-1",
            run_id="run-1",
            reason_codes=["outcome_not_success"],
        )
        is None
    )
    safe_increment_evolution_metric(
        observer,
        "extraction_failures",
    )
    safe_observe_evolution_metric(
        observer,
        "cluster_purity",
        0.5,
    )
    safe_set_evolution_gauge(
        observer,
        "queue_depth",
        1,
    )

    assert "secret observer failure" not in caplog.text
    assert "observability operation failed" in caplog.text


@pytest.mark.parametrize(
    ("method", "name"),
    [
        ("increment", "unknown_counter"),
        ("observe", "unknown_distribution"),
        ("set_gauge", "unknown_gauge"),
    ],
)
def test_unknown_metric_names_are_rejected(
    method: str,
    name: str,
) -> None:
    observer = EvolutionObservability()

    with pytest.raises(ValueError, match="unsupported"):
        getattr(observer, method)(name, 1)
