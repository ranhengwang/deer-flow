"""Deterministic Credit logging for future Skill policy training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from deerflow.skill_evolution.models import (
    CreditCost,
    CreditKind,
    CreditOutcomeSample,
    CreditToolCall,
    DistillationCredit,
    DistillationCreditStatus,
    EvolutionTraceSnapshot,
    OutcomeEvidence,
    OutcomeStatus,
    SelectionCandidate,
    SelectionCredit,
    SelectionDecisionSource,
    SkillProposal,
    SkillPublication,
    TraceSkillEvent,
    UtilizationCredit,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
    SkillEvolutionStore,
)

SELECTION_CREDIT_FORMULA_VERSION = "selection-ewma-v1"
UTILIZATION_CREDIT_FORMULA_VERSION = "verified-task-reward-v1"
DISTILLATION_CREDIT_FORMULA_VERSION = "post-publication-marginal-utility-v1"
SELECTION_POLICY_PROMPT_VERSION = "implicit-skill-selection-v1"
MINIMUM_FUTURE_DISTILLATION_SAMPLES = 3
_SELECTION_EWMA_ALPHA = 0.3


@dataclass(frozen=True, slots=True)
class VerifiedRunCreditResult:
    selection: SelectionCredit
    selection_created: bool
    utilizations: tuple[UtilizationCredit, ...]
    utilizations_created: tuple[bool, ...]


def _credit_id(kind: CreditKind, *parts: str) -> str:
    digest = hashlib.sha256("\0".join((kind.value, *parts)).encode("utf-8")).hexdigest()
    return f"credit-{kind.value}-{digest[:32]}"


def _reward(outcome: OutcomeEvidence) -> float | None:
    if outcome.final_reward is not None:
        return outcome.final_reward
    if outcome.status is OutcomeStatus.success:
        return 1.0
    if outcome.status is OutcomeStatus.failure:
        return 0.0
    return None


def _environment_fingerprint(
    snapshot: EvolutionTraceSnapshot,
) -> str:
    payload = snapshot.environment.model_dump(
        mode="json",
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ewma(values: list[float]) -> float | None:
    if not values:
        return None
    current = values[0]
    for value in values[1:]:
        current = _SELECTION_EWMA_ALPHA * value + (1.0 - _SELECTION_EWMA_ALPHA) * current
    return current


def _primary_skill(
    skills: list[TraceSkillEvent],
) -> TraceSkillEvent | None:
    return next(
        (skill for skill in skills if skill.activation_source == "slash"),
        skills[0] if skills else None,
    )


def _observed_search_query(
    snapshot: EvolutionTraceSnapshot,
) -> str | None:
    for tool_call in reversed(snapshot.tool_events):
        if tool_call.tool_name != "describe_skill":
            continue
        try:
            arguments = json.loads(tool_call.arguments)
        except (TypeError, ValueError):
            continue
        if not isinstance(arguments, dict):
            continue
        query = arguments.get("name")
        if isinstance(query, str) and query.strip():
            return query.strip()[:512]
    return None


def _decision_source(
    skills: list[TraceSkillEvent],
) -> SelectionDecisionSource:
    sources = {skill.activation_source for skill in skills}
    if not sources:
        return SelectionDecisionSource.no_skill
    if sources == {"slash"}:
        return SelectionDecisionSource.user_slash
    if sources == {"read"}:
        return SelectionDecisionSource.model_read
    return SelectionDecisionSource.mixed


def _outcome_sample(
    *,
    run_id: str,
    outcome: OutcomeEvidence,
    observed_at: datetime,
    environment_fingerprint: str | None = None,
) -> CreditOutcomeSample:
    return CreditOutcomeSample(
        run_id=run_id,
        reward=_reward(outcome),
        outcome_status=outcome.status,
        environment_fingerprint=environment_fingerprint,
        observed_at=observed_at,
    )


def _mean_known(
    samples: list[CreditOutcomeSample],
) -> float | None:
    rewards = [sample.reward for sample in samples if sample.reward is not None]
    if not rewards:
        return None
    return sum(rewards) / len(rewards)


class CreditRecorder:
    """Build and persist content-free, retry-safe Credit records."""

    def __init__(
        self,
        store: SkillEvolutionStore,
    ) -> None:
        self._store = store

    async def _selection_history(
        self,
        *,
        user_id: str,
        skill_name: str | None,
    ) -> list[SelectionCredit]:
        records = await self._store.list_credits(
            user_id,
            kind=CreditKind.selection,
            skill_name=skill_name,
            limit=256,
        )
        selections = [item for item in records if isinstance(item, SelectionCredit) and ((skill_name is None and item.no_skill_selected) or item.selected_skill_name == skill_name)]
        return list(reversed(selections))

    async def _build_selection(
        self,
        snapshot: EvolutionTraceSnapshot,
        outcome: OutcomeEvidence,
    ) -> SelectionCredit:
        skills = list(snapshot.skill_events)
        primary = _primary_skill(skills)
        credit_id = _credit_id(
            CreditKind.selection,
            snapshot.user_id,
            snapshot.run_id,
            snapshot.snapshot_hash,
        )
        existing = await self._store.get_credit(
            snapshot.user_id,
            credit_id,
        )
        if isinstance(existing, SelectionCredit):
            return existing

        history = await self._selection_history(
            user_id=snapshot.user_id,
            skill_name=(primary.skill_name if primary is not None else None),
        )
        current_reward = _reward(outcome)
        rewards = [item.outcome_reward for item in history if item.outcome_reward is not None]
        if current_reward is not None:
            rewards.append(current_reward)

        source = _decision_source(skills)
        slash_selected = primary is not None and primary.activation_source == "slash"
        observed_query = _observed_search_query(snapshot)
        search_query = f"select:{primary.skill_name}" if slash_selected else observed_query
        policy_model_name = None if slash_selected else snapshot.model_name
        return SelectionCredit(
            credit_id=credit_id,
            user_id=snapshot.user_id,
            thread_id=snapshot.thread_id,
            run_id=snapshot.run_id,
            snapshot_hash=snapshot.snapshot_hash,
            search_query=search_query,
            query_unavailable_reason=(None if search_query is not None else "selection_query_not_observed"),
            candidates=[
                SelectionCandidate(
                    skill_name=skill.skill_name,
                    content_hash=skill.content_hash,
                    selected=True,
                    rank=None,
                    score=None,
                )
                for skill in skills
            ],
            selected_skill_name=(primary.skill_name if primary is not None else None),
            selected_skill_hash=(primary.content_hash if primary is not None else None),
            no_skill_selected=primary is None,
            decision_source=source,
            policy_model_name=policy_model_name,
            policy_prompt_version=(None if slash_selected else SELECTION_POLICY_PROMPT_VERSION),
            policy_metadata_unavailable_reason=("user_selected_skill" if slash_selected else ("policy_model_unavailable" if policy_model_name is None else None)),
            log_probability=None,
            log_probability_unavailable_reason=("provider_logprob_unavailable"),
            outcome_reward=current_reward,
            credit_value=_ewma(rewards),
            formula_version=(SELECTION_CREDIT_FORMULA_VERSION),
            utility_sample_count=len(rewards),
            created_at=snapshot.created_at,
        )

    @staticmethod
    def _build_utilization(
        snapshot: EvolutionTraceSnapshot,
        outcome: OutcomeEvidence,
        skill: TraceSkillEvent,
    ) -> UtilizationCredit:
        reward = _reward(outcome)
        tool_calls = [
            CreditToolCall(
                tool_call_id=item.tool_call_id,
                tool_name=item.tool_name,
                status=item.status,
                error_type=item.error_type,
            )
            for item in snapshot.tool_events
        ]
        deviations: list[str] = []
        for item in snapshot.tool_events:
            if item.status != "error":
                continue
            code = item.error_type or "tool_error"
            if code not in deviations:
                deviations.append(code)
            if len(deviations) >= 64:
                break
        return UtilizationCredit(
            credit_id=_credit_id(
                CreditKind.utilization,
                snapshot.user_id,
                snapshot.run_id,
                snapshot.snapshot_hash,
                skill.skill_name,
                skill.content_hash,
                skill.activation_source,
            ),
            user_id=snapshot.user_id,
            thread_id=snapshot.thread_id,
            run_id=snapshot.run_id,
            snapshot_hash=snapshot.snapshot_hash,
            skill_name=skill.skill_name,
            skill_content_hash=skill.content_hash,
            activation_source=skill.activation_source,
            relevant_tool_calls=tool_calls,
            tool_attribution_scope="run_level",
            deviation_codes=deviations,
            outcome_status=outcome.status,
            outcome_confidence=outcome.confidence,
            outcome_reward=reward,
            credit_value=reward,
            formula_version=(UTILIZATION_CREDIT_FORMULA_VERSION),
            cost=CreditCost(
                tool_call_count=len(snapshot.tool_events),
                error_tool_call_count=sum(item.status == "error" for item in snapshot.tool_events),
                input_tokens=None,
                output_tokens=None,
                latency_seconds=None,
                unavailable_fields=[
                    "token_usage",
                    "latency",
                ],
            ),
            instruction_adherence=None,
            adherence_unavailable_reason=("model_feature_not_computed"),
            created_at=snapshot.created_at,
        )

    async def record_verified_run(
        self,
        snapshot: EvolutionTraceSnapshot,
        outcome: OutcomeEvidence,
    ) -> VerifiedRunCreditResult:
        selection = await self._build_selection(
            snapshot,
            outcome,
        )
        selection_result = await self._store.put_credit(selection)
        utilizations: list[UtilizationCredit] = []
        utilization_created: list[bool] = []
        for skill in snapshot.skill_events:
            utilization = self._build_utilization(
                snapshot,
                outcome,
                skill,
            )
            result = await self._store.put_credit(utilization)
            utilizations.append(result.value)
            utilization_created.append(result.created)
            if result.created:
                await self._record_future_utilization(result.value)
        return VerifiedRunCreditResult(
            selection=selection_result.value,
            selection_created=selection_result.created,
            utilizations=tuple(utilizations),
            utilizations_created=tuple(utilization_created),
        )

    async def record_publication(
        self,
        proposal: SkillProposal,
        publication: SkillPublication,
    ) -> DistillationCredit:
        if publication.user_id != proposal.user_id or publication.proposal_id != proposal.proposal_id:
            raise ValueError("Proposal and publication identity do not match")
        if publication.published_at is None or publication.published_skill_hash is None:
            raise ValueError("distillation credit requires a published version")
        credit_id = _credit_id(
            CreditKind.distillation,
            publication.user_id,
            publication.publication_id,
            publication.published_skill_hash,
        )
        existing = await self._store.get_credit(
            publication.user_id,
            credit_id,
        )
        if isinstance(existing, DistillationCredit):
            return existing

        source_events = []
        for event_id in proposal.supporting_event_ids:
            event = await self._store.get_event(
                proposal.user_id,
                event_id,
            )
            if event is None:
                raise ValueError("distillation source event was not found")
            source_events.append(event)
        baseline = [
            _outcome_sample(
                run_id=event.run_id,
                outcome=event.outcome,
                observed_at=event.created_at,
                environment_fingerprint=hashlib.sha256(
                    json.dumps(
                        event.environment.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            )
            for event in source_events
        ]
        credit = DistillationCredit(
            credit_id=credit_id,
            user_id=publication.user_id,
            proposal_id=proposal.proposal_id,
            publication_id=publication.publication_id,
            skill_name=publication.skill_name,
            published_skill_hash=(publication.published_skill_hash),
            source_event_ids=list(proposal.supporting_event_ids),
            baseline_outcomes=baseline,
            future_outcomes=[],
            minimum_future_samples=(MINIMUM_FUTURE_DISTILLATION_SAMPLES),
            status=DistillationCreditStatus.collecting,
            formula_version=(DISTILLATION_CREDIT_FORMULA_VERSION),
            credit_value=None,
            created_at=publication.published_at,
            updated_at=publication.published_at,
        )
        return (await self._store.put_credit(credit)).value

    async def _record_future_utilization(
        self,
        utilization: UtilizationCredit,
    ) -> None:
        records = await self._store.list_credits(
            utilization.user_id,
            kind=CreditKind.distillation,
            skill_name=utilization.skill_name,
            limit=128,
        )
        for record in records:
            if not isinstance(record, DistillationCredit):
                continue
            if record.status is DistillationCreditStatus.rolled_back or record.published_skill_hash != utilization.skill_content_hash or utilization.created_at <= record.created_at:
                continue
            await self._append_future_outcome(
                record,
                utilization,
            )

    async def _append_future_outcome(
        self,
        initial: DistillationCredit,
        utilization: UtilizationCredit,
    ) -> DistillationCredit:
        current = initial
        for _ in range(3):
            if any(item.run_id == utilization.run_id for item in current.future_outcomes):
                return current
            future = [
                *current.future_outcomes,
                CreditOutcomeSample(
                    run_id=utilization.run_id,
                    reward=utilization.outcome_reward,
                    outcome_status=utilization.outcome_status,
                    observed_at=utilization.created_at,
                ),
            ]
            known = [item for item in future if item.reward is not None]
            baseline_mean = _mean_known(current.baseline_outcomes)
            future_mean = _mean_known(future)
            mature = len(known) >= current.minimum_future_samples and baseline_mean is not None and future_mean is not None
            updated = DistillationCredit.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "future_outcomes": future,
                    "status": (DistillationCreditStatus.mature if mature else DistillationCreditStatus.collecting),
                    "credit_value": (future_mean - baseline_mean if mature and future_mean is not None and baseline_mean is not None else None),
                    "updated_at": utilization.created_at,
                    "revision": current.revision + 1,
                }
            )
            try:
                return await self._store.replace_credit(
                    updated,
                    expected_revision=current.revision,
                )
            except EvolutionStoreConflict:
                winner = await self._store.get_credit(
                    current.user_id,
                    current.credit_id,
                )
                if not isinstance(
                    winner,
                    DistillationCredit,
                ):
                    raise
                current = winner
        raise EvolutionStoreConflict("distillation credit update retry exhausted")

    async def record_rollback(
        self,
        publication: SkillPublication,
        *,
        rolled_back_at: datetime,
    ) -> DistillationCredit | None:
        if publication.published_skill_hash is None:
            return None
        credit_id = _credit_id(
            CreditKind.distillation,
            publication.user_id,
            publication.publication_id,
            publication.published_skill_hash,
        )
        current = await self._store.get_credit(
            publication.user_id,
            credit_id,
        )
        if not isinstance(current, DistillationCredit):
            return None
        if current.status is DistillationCreditStatus.rolled_back:
            return current
        updated = DistillationCredit.model_validate(
            {
                **current.model_dump(mode="python"),
                "status": (DistillationCreditStatus.rolled_back),
                "updated_at": rolled_back_at,
                "revision": current.revision + 1,
            }
        )
        return await self._store.replace_credit(
            updated,
            expected_revision=current.revision,
        )
