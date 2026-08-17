# Self-Evolving Skills Development Plan

> Status: Draft implementation plan
>
> Scope owner: DeerFlow harness/backend
>
> Primary goal: evolve persistent Agent Skills from verified task trajectories without
> updating model parameters in the first implementation stage.

## 1. Goal

Build a controlled skill-evolution lifecycle on top of DeerFlow:

1. Capture a completed agent run as a bounded, redacted trajectory.
2. Admit only verified successful runs with useful learning evidence.
3. Convert eligible trajectories into structured evolution events.
4. Group events that represent the same reusable workflow or the same skill defect.
5. After enough independent evidence has accumulated, distill a new skill or an
   update proposal for an existing custom skill.
6. Evaluate the proposal against source tasks and held-out tasks.
7. Publish only approved proposals through the existing skill storage and security
   boundaries.
8. Record selection, utilization, and distillation evidence for later RL training.

The first production milestone ends at external skill-library evolution. Policy
training, Verl integration, and skill internalization are explicitly deferred.

## 2. Current Baseline

DeerFlow currently provides:

- `skill_evolution.enabled` to expose the `skill_manage` tool.
- Prompt guidance that asks the lead model to consider skill creation or patching.
- `skill_manage` operations for create, patch, edit, delete, supporting-file write,
  and supporting-file removal.
- Per-user custom skill storage and mutation history.
- Deterministic SkillScan plus LLM-based security review before writes.
- Prompt-cache invalidation after successful skill mutations.

The current baseline is single-run and model-initiated:

```text
task trajectory
  -> lead model decides to evolve a skill
  -> lead model generates skill content
  -> skill_manage validates and writes it
```

It does not yet provide:

- reliable post-run success verification;
- structured trajectory events;
- durable cross-run evidence collection;
- same-task-family clustering;
- evidence thresholds before distillation;
- candidate skill staging;
- replay, A/B evaluation, regression checks, or rollback;
- skill quality scores;
- selection/utilization/distillation credit records;
- model-parameter internalization.

## 3. Research Questions

The implementation and experiments should answer:

1. Does multi-trajectory distillation produce more reliable skills than immediate
   single-trajectory creation?
2. Does evidence-based patching improve an existing skill without regressing tasks
   it previously solved?
3. How does the minimum evidence threshold (`K=1, 3, 5`) affect quality, latency,
   and false-positive evolution?
4. Does hybrid grouping (deterministic features plus LLM confirmation) outperform
   either deterministic or LLM-only grouping?
5. Do evolved skills improve task success, tool efficiency, token cost, and
   robustness on held-out tasks?
6. Can logged selection, utilization, and distillation evidence later support
   Skill1-style policy optimization and SKILL0-style skill internalization?

## 4. Non-Goals for the First Implementation

- Fine-tuning or reinforcement learning of the policy model.
- Verl integration.
- Automatic parameter internalization.
- Automatically editing built-in public skills.
- Learning from unverified task outcomes.
- Treating model self-reported success as authoritative.
- Synchronously distilling and evaluating skills on the user request path.
- Fully autonomous publication of executable skill packages.

## 5. Core Invariants

1. **Verified outcomes only.** A run is not eligible because the model says it
   succeeded. Eligibility requires verifier evidence and a confidence score.
2. **Successful recovery is useful evidence.** Failed steps inside a finally
   successful trajectory may be distilled; an unverified failed run may not create
   or patch a skill automatically.
3. **Independent evidence.** The threshold counts distinct task runs, not repeated
   attempts from one run.
4. **No direct public-skill mutation.** Public and integration skills remain
   read-only. Updates create a per-user custom shadow or a separate candidate.
5. **No direct publication from the distiller.** Distillation produces a staged
   proposal. Evaluation and approval precede publication.
6. **`skill_manage` remains the mutation boundary.** The new subsystem may reuse
   its validation/publishing service, but must not bypass path validation,
   SkillScan, LLM moderation, history, or cache refresh.
7. **Per-user isolation.** Events, clusters, proposals, evaluations, and published
   custom skills are scoped by user.
8. **Evidence provenance.** Every generated rule or patch must reference the events
   that support it.
9. **Redaction before persistence/model calls.** Secrets, credentials, and raw
   host paths must not enter evolution records or distillation prompts.
10. **Online latency isolation.** Run completion may enqueue evolution work, but
    extraction, clustering, distillation, and replay must run asynchronously.
11. **Version and rollback.** Every published update records its base version,
    candidate version, evaluation, and a reversible snapshot.
12. **Configuration off means no behavior change.**

## 6. Target Architecture

```text
Lead Agent Run
  |
  v
RunJournal + final state + environment result
  |
  v
Evolution Trace Snapshot
  |
  v
Outcome Verifier
  |
  v
Eligibility Detector
  |
  v
Trajectory Event Extractor
  |
  v
Evolution Event Store
  |
  v
Hybrid Grouper / Clusterer
  |
  |  independent event count >= K
  v
Cross-Trajectory Distiller
  |
  v
Staged Skill Proposal
  |
  v
Replay + A/B + Regression Evaluator
  |
  v
Approval Policy
  |
  v
Skill Publisher
  |
  v
Per-user Custom Skill + Version History
```

Recommended package structure:

```text
backend/packages/harness/deerflow/skill_evolution/
├── __init__.py
├── models.py
├── redaction.py
├── trajectory.py
├── verifier.py
├── eligibility.py
├── extractor.py
├── grouping.py
├── distiller.py
├── evaluator.py
├── quality.py
├── credit.py
├── publisher.py
├── coordinator.py
└── store/
    ├── base.py
    ├── memory.py
    └── sql.py
```

The package must not import `app.*`.

## 7. Domain Model

### 7.1 Evolution Trace Snapshot

A bounded post-run input assembled from existing runtime data:

```python
EvolutionTraceSnapshot(
    run_id,
    thread_id,
    user_id,
    model_name,
    task_input,
    final_answer,
    final_status,
    stop_reason,
    tool_events,
    skill_events,
    user_corrections,
    artifacts,
    environment,
    verifier_hints,
)
```

Requirements:

- Bound every text and collection field.
- Preserve tool order and tool-call/result pairing.
- Record configured virtual paths, not raw host paths.
- Record skill name, category, canonical path, and content hash at activation.
- Store references to large artifacts instead of embedding their contents.

### 7.2 Outcome Evidence

```python
OutcomeEvidence(
    status="success" | "failure" | "unknown",
    confidence=0.0..1.0,
    sources=[...],
    checks=[...],
    final_reward=None | float,
)
```

Evidence priority:

1. deterministic task-specific verifier;
2. tests, command exit codes, artifact validation, or environment reward;
3. explicit user acceptance/correction outcome;
4. LLM judge as supporting evidence only.

### 7.3 Structured Evolution Event

```python
EvolutionEvent(
    event_id,
    run_id,
    user_id,
    event_kind="new_skill_evidence" | "skill_patch_evidence",
    task_signature,
    task_goal,
    environment_signature,
    outcome,
    complexity_signals,
    tool_signature,
    skill_usage,
    successful_path,
    failed_attempts,
    user_corrections,
    reusable_lessons,
    skill_gaps,
    target_skill,
    provenance,
    created_at,
)
```

Supported skill-gap categories:

```text
outdated_instruction
os_incompatibility
missing_prerequisite
uncovered_failure_mode
weak_verification
wrong_tool_guidance
ambiguous_trigger
unnecessary_step
```

### 7.4 Event Cluster

```python
EvolutionCluster(
    cluster_id,
    user_id,
    event_kind,
    target_skill,
    canonical_signature,
    member_event_ids,
    independent_run_count,
    status,
    grouping_evidence,
)
```

### 7.5 Skill Proposal

```python
SkillProposal(
    proposal_id,
    cluster_id,
    operation="create" | "patch" | "edit" | "write_file",
    skill_name,
    base_skill_hash,
    proposed_files,
    supporting_event_ids,
    rationale,
    expected_improvements,
    risks,
    status,
)
```

### 7.6 Evaluation Result

```python
SkillEvaluation(
    proposal_id,
    source_replay_results,
    held_out_results,
    baseline_results,
    candidate_results,
    regression_results,
    safety_results,
    quality_score,
    decision="approve" | "reject" | "manual_review",
)
```

## 8. Configuration

Extend `SkillEvolutionConfig` incrementally. Proposed final shape:

```yaml
skill_evolution:
  enabled: false
  mode: immediate | evidence
  moderation_model_name: null
  extraction_model_name: null
  distillation_model_name: null
  evaluation_model_name: null
  security_fail_closed: true

  evidence:
    min_success_confidence: 0.8
    min_cluster_events: 3
    min_distinct_runs: 3
    max_events_per_cluster: 20
    tool_call_complexity_threshold: 5
    accept_recovered_errors: true
    accept_user_corrections: true
    accept_explicit_remember_requests: true

  grouping:
    strategy: hybrid
    deterministic_threshold: 0.6
    semantic_threshold: 0.82
    llm_confirmation: true

  publication:
    mode: manual
    allow_non_executable_auto_publish: false
    allow_executable_auto_publish: false
    require_held_out_evaluation: true

  quality:
    min_source_replay_success_rate: 1.0
    min_held_out_success_rate: 0.8
    max_regression_rate: 0.0
```

Implementation requirements:

- Add fields only in the phase that consumes them.
- Bump `config_version` for schema changes.
- Update `config.example.yaml` and architecture documentation in the same change.
- Preserve the current immediate mode for compatibility until evidence mode is ready.

## 9. Task Plan

Every implementation task follows RED -> GREEN -> regression proof. Backend TDD is
mandatory.

### Phase 0: Freeze Contracts and Experimental Protocol

#### Task 0.1: Record design decisions

Files:

- Modify this plan.
- Create implementation notes only when implementation starts.

- [x] Confirm that evolution is per-user.
- [x] Confirm that only successful/high-confidence runs are admitted.
- [x] Confirm whether user confirmation is required for new skill creation.
- [x] Confirm default publication mode is `manual`.
- [x] Confirm `K=3` is a configurable default, not a hard-coded constant.
- [x] Define what counts as an independent task run.
- [x] Define supported verifier types for the first benchmark.
- [x] Define retention rules for raw traces and structured events.

Frozen decisions:

- Evolution records and custom skills are scoped by the runtime-resolved `user_id`.
- Only runs with `OutcomeEvidence.status == "success"` and confidence at or above
  `min_success_confidence` are admitted automatically.
- In interactive runs, creating a new skill requires explicit user confirmation.
  An explicit user request to create or remember a skill counts as confirmation.
  Non-interactive runs may create proposals but may not publish new skills.
- Publication defaults to `manual`, including every executable-file proposal.
- `K=3` means three independent evidence runs and is a configurable default.
- Evidence runs are independent only when all of the following hold:
  - their `run_id` values differ;
  - their normalized task-input hashes differ;
  - they are not regenerate/edit-replay/branch descendants of the same source turn;
  - one run is not a retry or continuation of another.
- The first benchmark supports three authoritative deterministic verifier families:
  unit-test/command verifiers, structured artifact schema/value verifiers, and shell
  exit-code plus expected-artifact verifiers.
- Raw redacted trace snapshots are retained for 7 days by default. Structured events,
  rejected proposals, and evaluations are retained for 180 days. Provenance for a
  published skill version is retained with that version until the skill is deleted
  under the deployment's retention policy.

Acceptance:

- All open decisions are resolved or explicitly marked deferred.

#### Task 0.2: Define benchmark and baselines

- [x] Select at least three reproducible task domains.
- [x] Create task families with train/evidence and held-out variants.
- [x] Give every task a deterministic verifier where possible.
- [x] Define baselines:
  - no skill evolution;
  - current immediate single-trajectory `skill_manage`;
  - single-trajectory staged evolution;
  - proposed multi-trajectory `K=3` evolution.
- [x] Define primary metrics and statistical procedure.

Suggested domains:

- repository/code repair with unit-test verifiers;
- structured data transformation with schema/value verifiers;
- shell/environment workflows with exit-code and artifact verifiers.

Frozen benchmark protocol:

- Use three domains: repository repair, structured data transformation, and
  shell/environment workflow.
- Define at least four task families per domain.
- Each family contains three evidence variants and two held-out variants, for a
  minimum of 60 task variants.
- Repository tasks are verified by focused tests plus required file-state checks.
- Data tasks are verified by schema, row/value invariants, and deterministic output
  hashes where appropriate.
- Shell tasks are verified by exit status, expected artifacts, and explicit
  postcondition commands.
- Run each baseline on identical task variants and seeds. Use at least three seeds
  for stochastic model runs in the final experiment.
- Primary metrics are task success-rate lift, held-out success, regression rate,
  incorrect-evolution rate, and cluster purity.
- Secondary metrics are tool calls, token usage, latency, evolution delay, proposal
  acceptance, and security rejection.
- Report paired bootstrap 95% confidence intervals for metric deltas. Use McNemar's
  test for paired binary success outcomes when sample counts permit, and report raw
  per-family results alongside aggregate results.

Acceptance:

- The experiment can be rerun without manual interpretation of task success.

### Phase 1: Data Contracts and Persistence

#### Task 1.1: Add typed domain models

Files:

- Create `deerflow/skill_evolution/models.py`.
- Create `backend/tests/test_skill_evolution_models.py`.

- [x] Write validation tests for every model and enum.
- [x] Reject oversized fields, invalid confidence, malformed provenance, and
      unsupported event kinds.
- [x] Implement JSON-safe serialization.
- [x] Add schema version to persisted records.

Acceptance:

- Models round-trip through JSON without losing provenance.
- Invalid or oversized records fail before persistence.

#### Task 1.2: Add store contract and memory implementation

Files:

- Create `deerflow/skill_evolution/store/base.py`.
- Create `deerflow/skill_evolution/store/memory.py`.
- Create focused store contract tests.

- [x] Define async CRUD/query methods for events, clusters, proposals, evaluations,
      and credit records.
- [x] Define idempotent upsert by `(user_id, run_id, extractor_version)`.
- [x] Add cluster readiness query based on distinct runs.
- [x] Add optimistic proposal-state transitions.
- [x] Run the same contract tests against every backend.

Acceptance:

- Duplicate run processing does not create duplicate evidence.
- Concurrent proposal workers cannot publish the same proposal twice.

#### Task 1.3: Add SQL persistence

Files:

- Create persistence models/repository under `deerflow/persistence/`.
- Add an Alembic migration.
- Add SQLite and Postgres-compatible repository tests.

- [x] Store bounded JSON payloads and indexed lookup fields separately.
- [x] Index user, status, task signature, target skill, and creation time.
- [x] Preserve provenance and proposal version relationships.
- [x] Add migration downgrade only if repository migration policy permits it.

Acceptance:

- Events survive process restart.
- SQLite is sufficient for single-node development.
- SQL behavior matches the memory store contract.

### Phase 2: Trace Capture and Verification

#### Task 2.1: Build bounded evolution trace snapshots

Files:

- Create `deerflow/skill_evolution/trajectory.py`.
- Modify the run completion path in `runtime/runs/worker.py`.
- Add run-worker and trajectory tests.

- [x] Materialize ordered tool calls/results from run-scoped data.
- [x] Capture active skill name/path/hash and slash/read activation source.
- [x] Capture user correction messages without hidden framework context.
- [x] Capture final status, stop reason, output artifacts, and workspace changes.
- [x] Redact secrets and host paths.
- [x] Bound event count and content sizes.
- [x] Make snapshot creation idempotent.

Acceptance:

- A completed run produces one deterministic redacted snapshot.
- No API key, injected secret, or raw host path appears in the snapshot.

#### Task 2.2: Implement outcome verifier framework

Files:

- Create `deerflow/skill_evolution/verifier.py`.
- Add verifier unit and integration tests.

- [x] Define pluggable deterministic verifier protocol.
- [x] Implement signals for tests, command exit codes, artifact existence/schema,
      environment rewards, and explicit user acceptance.
- [x] Support `success`, `failure`, and `unknown`.
- [x] Add confidence aggregation with evidence provenance.
- [x] Keep LLM judging optional and non-authoritative by default.

Acceptance:

- Model final text alone cannot mark a run successful.
- Conflicting evidence yields `unknown` or reduced confidence.

#### Task 2.3: Implement eligibility and complexity detection

Files:

- Create `deerflow/skill_evolution/eligibility.py`.
- Add focused tests for every signal combination.

- [x] Admit only successful runs above configured confidence.
- [x] Detect `tool_calls >= threshold`.
- [x] Detect an error ToolMessage followed by a successful alternative path.
- [x] Detect explicit remember/create-skill requests.
- [x] Accept user correction and non-trivial workflow flags from extraction.
- [x] Distinguish no-skill and skill-used branches.
- [x] Exclude safety-, loop-, token-, and subagent-limit-capped runs by default.

Acceptance:

- Simple successful requests are rejected.
- Complex recovered runs are admitted.
- Failed and unknown runs are not automatically distilled.

### Phase 3: Structured Event Extraction

#### Task 3.1: Add deterministic pre-extraction

Files:

- Create `deerflow/skill_evolution/extractor.py`.
- Add fixture-based tests.

- [x] Derive tool sequence, error categories, OS/runtime signature, skill usage,
      artifact evidence, and deterministic complexity signals without an LLM.
- [x] Produce bounded candidate segments for LLM extraction.
- [x] Never include secrets or full binary/large-file content.

Acceptance:

- Deterministic fields are reproducible across repeated extraction.

#### Task 3.2: Add structured LLM extraction

- [x] Define a strict JSON output schema.
- [x] Ask the extraction model for task goal, successful path, failed attempts,
      corrections, reusable lessons, skill gaps, and candidate target.
- [x] Validate output against typed models.
- [x] Retry only malformed/transient responses with a bounded budget.
- [x] Fail closed to "no evolution event" after repeated malformed output.
- [x] Persist extractor model, prompt version, and source snapshot hash.

Acceptance:

- Extraction never writes a skill.
- Every lesson is linked to trace evidence.
- Unparseable output does not enter the event store.

### Phase 4: Event Grouping and Cluster Readiness

#### Task 4.1: Implement deterministic fingerprints

Files:

- Create `deerflow/skill_evolution/grouping.py`.
- Add grouping fixtures and negative tests.

- [x] Build normalized fingerprints from goal, tool signature, error signature,
      environment family, event kind, and target skill.
- [x] Prevent new-skill and patch events from sharing a cluster.
- [x] Prevent patch events for different target skills from sharing a cluster.
- [x] Deduplicate near-identical runs.

Acceptance:

- Obvious same-family tasks group together.
- Tasks sharing words but requiring different workflows remain separate.

#### Task 4.2: Add optional semantic retrieval

- [x] Define an embedding-provider interface.
- [x] Keep embeddings optional for local development.
- [x] Store embedding model/version with vectors.
- [x] Fall back to deterministic candidate retrieval when unavailable.

Acceptance:

- The subsystem works without an embedding service.

#### Task 4.3: Add LLM cluster confirmation

- [x] Compare candidate event with cluster prototype.
- [x] Require a structured same-workflow decision and reason.
- [x] Detect environment-specific conditional branches rather than splitting
      compatible workflows unnecessarily.
- [x] Mark contradictory evidence and defer readiness.
- [x] Trigger readiness only after `K` distinct runs.

Acceptance:

- Cluster membership has deterministic and semantic provenance.
- A single repeated run cannot satisfy `K=3`.

### Phase 5: Cross-Trajectory Distillation

#### Task 5.1: Distill new-skill proposals

Files:

- Create `deerflow/skill_evolution/distiller.py`.
- Add golden fixtures and schema tests.

- [x] Input only structured events plus bounded provenance excerpts.
- [x] Separate common rules, conditional environment rules, conflicts, and
      unsupported one-off observations.
- [x] Generate complete `SKILL.md` frontmatter and body.
- [x] Generate supporting files only when repeated evidence justifies them.
- [x] Include evidence mapping for every generated section.
- [x] Stage proposal without publishing.

Acceptance:

- A proposal cannot contain unsupported mandatory steps.
- Conflicting events produce manual review instead of forced synthesis.

#### Task 5.2: Distill existing-skill patches

- [x] Load the exact base skill version/hash used by source trajectories.
- [x] Produce structured patch operations before rendering new files.
- [x] Prefer targeted patch over full edit.
- [x] Preserve unaffected instructions.
- [x] Add OS/runtime conditions where evidence is environment-specific.
- [x] Detect stale base versions and restart distillation against the current version.

Acceptance:

- Concurrent skill updates cannot silently overwrite each other.
- Every change links to at least one supporting event.

### Phase 6: Candidate Evaluation and Quality

#### Task 6.1: Add replayable task specification

Files:

- Create `deerflow/skill_evolution/evaluator.py`.
- Add deterministic replay harness tests.

- [x] Define task input, fixture snapshot, environment requirements, verifier,
      timeout, and allowed side effects.
- [x] Mark non-replayable tasks as manual-review-only.
- [x] Isolate replay workspaces and clean them after evaluation.

Acceptance:

- Replays cannot mutate the source run workspace or production skill library.

#### Task 6.2: Evaluate new skill proposals

- [x] Replay all source tasks with the candidate skill.
- [x] Run held-out same-family tasks.
- [x] Run paired no-skill baselines.
- [x] Record success, tool calls, tokens, latency, errors, and artifacts.
- [x] Reject candidates that pass source tasks but fail held-out requirements.

#### Task 6.3: Evaluate skill updates

- [ ] Run old-skill and new-skill paired evaluations.
- [ ] Include historical regression tasks for the base skill.
- [ ] Reject updates above configured regression rate.
- [ ] Require manual review for executable supporting-file changes.

#### Task 6.4: Compute skill quality

Initial quality dimensions:

```text
success-rate lift over no-skill baseline
held-out generalization
tool-call reduction
token reduction
latency reduction
robustness across environments
regression rate
safety risk
```

- [ ] Store raw metrics separately from aggregate quality score.
- [ ] Version the scoring formula.
- [ ] Require minimum sample counts before assigning "high quality".

Acceptance:

- A skill cannot become high quality from only its three source runs.

### Phase 7: Approval, Publication, and Rollback

#### Task 7.1: Refactor reusable skill mutation service

Files:

- Extract common mutation logic from `tools/skill_manage_tool.py`.
- Keep the LangChain tool as a thin adapter.
- Add mutation-service tests plus existing tool regressions.

- [ ] Preserve validation, per-user locking, SkillScan, LLM moderation, history,
      and cache refresh.
- [ ] Add expected base hash/version for compare-and-swap publication.
- [ ] Add proposal and evaluation IDs to mutation history.

Acceptance:

- Manual `skill_manage` behavior remains backward compatible.
- Evolution publisher cannot bypass existing security controls.

#### Task 7.2: Add proposal approval policy

- [ ] Default to manual approval.
- [ ] Allow future auto-publication only for non-executable, low-risk changes with
      passing held-out evaluation.
- [ ] Never auto-publish executable files in the first version.
- [ ] Add explicit rejection reason and proposal expiration.

#### Task 7.3: Add version snapshots and rollback

- [ ] Snapshot all skill package files before publication.
- [ ] Record base and published hashes.
- [ ] Add rollback operation that creates a new history entry.
- [ ] Re-run security checks before restoring executable content.

Acceptance:

- Every published proposal can be rolled back without reconstructing state from
  model output.

### Phase 8: Asynchronous Coordination and Operations

#### Task 8.1: Add evolution coordinator

Files:

- Create `deerflow/skill_evolution/coordinator.py`.
- Create `deerflow/skill_evolution/worker.py`.
- Add lifecycle and shutdown tests.

- [ ] Enqueue eligible run IDs after durable run finalization.
- [ ] Keep extraction/distillation off the request event loop.
- [ ] Use bounded queues, retries, backoff, and idempotency keys.
- [ ] Persist state before acknowledging work.
- [ ] Drain or safely abandon work during shutdown according to configured timeout.

Acceptance:

- Skill evolution failure cannot change the completed user run result.
- Restart resumes pending durable work without duplication.

#### Task 8.2: Add observability

- [ ] Emit events for admitted, rejected, extracted, clustered, ready, distilled,
      evaluated, approved, published, rejected, and rolled back.
- [ ] Add metrics for queue depth, extraction failures, cluster purity samples,
      proposal pass rate, publication rate, regression rate, and quality lift.
- [ ] Log IDs and hashes, not raw sensitive content.

### Phase 9: API and Minimal Review Surface

#### Task 9.1: Add read-only APIs

- [ ] List events, clusters, proposals, evaluations, and skill versions per user.
- [ ] Enforce ownership and admin rules.
- [ ] Paginate all collections.
- [ ] Return compact/redacted data by default.

#### Task 9.2: Add approval and rollback APIs

- [ ] Approve/reject proposals.
- [ ] Publish approved proposals.
- [ ] Roll back published versions.
- [ ] Protect all mutations with CSRF/authz/ownership checks.

Frontend UI is optional for the first research milestone; APIs plus CLI are sufficient.

### Phase 10: Credit Logging for Future RL

#### Task 10.1: Record selection actions

- [ ] Record generated search query, candidate skill IDs/hashes, rank scores,
      selected skill, and no-skill decision.
- [ ] Record policy model and prompt version.
- [ ] Record log probabilities only when the provider exposes them reliably.

#### Task 10.2: Record utilization evidence

- [ ] Record activation source, loaded skill version, relevant tool calls,
      deviations, outcome reward, and cost.
- [ ] Do not infer instruction adherence from text matching alone; retain it as
      an optional model-derived feature.

#### Task 10.3: Record distillation evidence

- [ ] Link proposal to source events and future tasks using the published version.
- [ ] Compute delayed marginal utility after sufficient future evaluations.
- [ ] Preserve raw outcome series for alternative credit formulas.

Initial non-training signals:

```text
utilization credit = verified current-task reward
selection credit = smoothed skill-conditioned utility trend
distillation credit = post-publication marginal utility change
```

Acceptance:

- Data can be exported without reconstructing decisions from free-form logs.

### Phase 11: Experimental Evaluation

#### Task 11.1: Run ablations

- [ ] `K=1` vs `K=3` vs `K=5`.
- [ ] deterministic grouping vs LLM grouping vs hybrid.
- [ ] immediate publish vs staged evaluation.
- [ ] success path only vs success path plus recovered failures.
- [ ] no-skill creation branch vs existing-skill patch branch.

#### Task 11.2: Report metrics

Primary:

- task success-rate lift;
- held-out success rate;
- regression rate;
- incorrect evolution rate;
- cluster precision/purity.

Secondary:

- tool-call count;
- token usage;
- latency;
- proposal acceptance rate;
- time/evidence required to evolve;
- security rejection rate.

#### Task 11.3: Statistical analysis

- [ ] Use paired tasks/seeds for old/new/no-skill comparisons.
- [ ] Report confidence intervals.
- [ ] Separate task-family and aggregate results.
- [ ] Publish failed and rejected proposal counts, not only successful cases.

### Phase 12: Deferred RL and Skill Internalization

This phase begins only after the external skill-evolution pipeline is stable.

#### Task 12.1: Export Verl-ready datasets

- [ ] Export trajectories, policy actions, candidate skills, rewards, and credit
      records with stable schemas.
- [ ] Exclude secrets and non-redistributable artifacts.
- [ ] Version tokenizer, model, prompt, environment, and verifier.

#### Task 12.2: Skill1-style unified policy training

- [ ] Train selection, utilization, and distillation behavior from shared task
      outcomes and recorded temporal signals.
- [ ] Compare shared-signal training with separate reward baselines.
- [ ] Keep online production skill publication disabled during training runs.

#### Task 12.3: SKILL0-style internalization

- [ ] Select only high-quality, stable skills with enough paired evaluations.
- [ ] Construct full-skill, compressed-skill, name-only, and no-skill curricula.
- [ ] Gradually withdraw external skill context.
- [ ] Measure whether no-skill performance approaches skill-conditioned performance.
- [ ] Keep the external skill until internalization is verified; never delete it
      solely because training completed.

## 10. Milestones

### Milestone A: Evidence Capture

Includes Phases 0-3.

Definition of done:

- verified successful complex runs produce redacted structured events;
- skill-used and no-skill branches are distinguishable;
- duplicate run processing is idempotent.

### Milestone B: Multi-Trajectory Proposal

Includes Phases 4-5.

Definition of done:

- three independent same-family events create one staged proposal;
- new-skill and existing-skill proposals are supported;
- proposal content contains evidence provenance.

### Milestone C: Safe Evolution Loop

Includes Phases 6-8.

Definition of done:

- proposals are replayed and regression tested;
- approved proposals publish through existing safety boundaries;
- published versions can be rolled back;
- evolution work does not block user runs.

### Milestone D: Research Evaluation

Includes Phases 9-11.

Definition of done:

- all baselines and ablations run reproducibly;
- primary metrics and confidence intervals are reported;
- limitations and rejected cases are documented.

### Milestone E: RL Extension

Phase 12, optional/deferred.

## 11. Test Strategy

Required test layers:

1. Pure unit tests for schemas, eligibility, fingerprints, scoring, and redaction.
2. Store contract tests shared by memory and SQL implementations.
3. Mock-model tests for extraction, grouping confirmation, and distillation.
4. Integration tests for run completion -> event persistence.
5. Replay tests with deterministic task fixtures.
6. Mutation regression tests proving `skill_manage` remains compatible.
7. Security tests for prompt injection, malicious supporting scripts, path traversal,
   secret leakage, cross-user access, stale-version publication, and rollback.
8. Blocking-I/O tests for every async runtime path that touches disk or SQL.

Every production behavior change must include a negative mutation proof: temporarily
remove or bypass the new wiring and confirm at least one focused test fails.

## 12. Open Decisions

Resolve before the corresponding phase:

- [ ] Which success verifiers are authoritative for open-ended research tasks?
- [x] Should explicit user requests bypass the complexity threshold but still require
      successful task evidence?
- [x] Is manual approval required for every new skill in the research prototype?
- [x] Which embedding provider, if any, is part of the reproducible baseline?
- [x] How long are raw snapshots retained after structured extraction?
- [ ] What minimum held-out sample count is required for a high-quality designation?
- [ ] Can a custom skill shadow a public skill automatically, or only after approval?
- [ ] What is the exact formula for aggregate skill quality?

## 13. Decision Log

Append decisions; do not rewrite historical entries.

### 2026-08-16 — Initial Plan

- **Decision:** Implement external skill-library evolution before policy RL.
- **Decision:** Use verified successful trajectories; retain failed attempts only when
  the overall task eventually succeeds.
- **Decision:** Default evidence threshold is three independent runs and remains
  configurable.
- **Decision:** Separate new-skill creation and existing-skill patch evidence.
- **Decision:** Distillation generates staged proposals, never direct writes.
- **Decision:** Reuse DeerFlow SkillScan, LLM moderation, per-user storage, history,
  and cache refresh for publication.
- **Decision:** Default publication mode is manual.
- **Decision:** Keep executable skill changes manual-review-only in the first version.
- **Decision:** Record future RL evidence from the start, but defer Verl integration
  and parameter internalization.
- **Decision:** An explicit remember/create request bypasses the complexity threshold,
  but the underlying task still requires verified successful evidence.
- **Decision:** Every new skill requires manual approval in the research prototype;
  non-interactive runs can only stage proposals.
- **Decision:** Raw redacted snapshots default to 7-day retention, structured events
  and unpublished proposal/evaluation records to 180 days, and published-version
  provenance to the lifetime of that version.
- **Decision:** The initial benchmark contains repository repair, structured data
  transformation, and shell/environment workflows, with at least four task families
  per domain, three evidence variants and two held-out variants per family.

### 2026-08-16 — Phase 0 and Task 1.1

- **Completed:** Frozen the core evidence, approval, independence, retention, and
  benchmark contracts.
- **Completed:** Added strict immutable Pydantic contracts for evolution events,
  clusters, proposals, and evaluations under `deerflow.skill_evolution.models`.
- **Completed:** Added schema versions, bounded fields/collections, JSON round trips,
  branch invariants, safe proposal paths, and operation/version constraints.
- **Evidence:** 12 focused model tests pass; Ruff lint/format pass; temporarily
  widening the outcome confidence bound caused the intended regression test to fail.

### 2026-08-16 — Phase 1 Task 1.2 / In-Memory Store

- **Completed:** Added an async store interface and process-local implementation.
- **Completed:** Added event upsert idempotency by
  `(user_id, run_id, extractor_version)` with payload-drift conflicts.
- **Completed:** Added per-user event, cluster, proposal, and evaluation isolation.
- **Completed:** Added ready-cluster threshold queries and compare-and-swap proposal
  status transitions.
- **Deferred within Task 1.2:** Running the same contract suite against every backend
  remains pending until the SQL implementation in Task 1.3 exists.
- **Evidence:** 21 focused tests pass; Ruff lint/format pass; removing the ready-status
  predicate caused the intended cluster-query regression test to fail.

### 2026-08-16 — Phase 1 Task 1.3 / SQL Persistence

- **Completed:** Added SQLAlchemy rows for events, clusters, proposals, and
  evaluations, with indexed lookup fields plus complete versioned JSON payloads.
- **Completed:** Added a SQLite/PostgreSQL-compatible SQL store implementing the same
  async contract as the memory store.
- **Completed:** Added revision `0012_skill_evolution` with idempotent table/index
  creation and downgrade support.
- **Completed:** Updated bootstrap head expectations and registered all ORM rows for
  create-all and Alembic autogeneration.
- **Evidence:** 36 evolution tests pass across model, memory, SQL, migration, restart
  durability, and PostgreSQL DDL compilation; 98 persistence bootstrap/migration tests
  pass; removing one migration index caused the intended migration test failure.

### 2026-08-16 — Phase 2 Task 2.1 / Trace Capture

- **Completed:** Added immutable, versioned `EvolutionTraceSnapshot`,
  `TraceToolEvent`, and `TraceSkillEvent` contracts.
- **Completed:** Added a deterministic builder that keeps the first visible task
  input and bounded event tail, pairs tool calls/results, captures corrections,
  artifacts, workspace changes, final status, and stop reason.
- **Completed:** Captured slash/read Skill usage by canonical path and content hash;
  loaded `SKILL.md` bodies are replaced by hash references.
- **Completed:** Added recursive sensitive-key redaction, exact request/active-secret
  replacement, and host-to-virtual path masking.
- **Completed:** Added the `skill_evolution.trace` / `evolution` run event to the
  runtime catalog and public stream contract.
- **Completed:** Wired enabled runs into terminal worker finalization with
  `put_if_absent` idempotency and fail-open isolation from the user run outcome.
- **Evidence:** Trajectory and worker tests cover deterministic hashes, event bounds,
  hidden-context exclusion, short/exact secret redaction, host paths, idempotency,
  disabled mode, and non-fatal storage failure.

### 2026-08-16 — Phase 2 Task 2.2 / Outcome Verification

- **Completed:** Added the bounded `OutcomeVerifier` plugin protocol and immutable
  verification context/signal contracts.
- **Completed:** Added deterministic verifiers for terminal run failure, latest
  recognized test commands, explicit command expectations, reported artifact
  existence/schema/value checks, environment rewards, and explicit user acceptance.
- **Completed:** Added conservative priority-aware aggregation. Equal-priority
  success/failure conflicts return `unknown`; lower-priority opposition reduces
  confidence.
- **Completed:** Persisted per-check confidence, authority, priority, detail, and
  source provenance in `OutcomeEvidence`.
- **Completed:** Added optional precomputed LLM judgment as supporting-only evidence;
  it cannot establish success without authoritative evidence.
- **Completed:** Isolated verifier exceptions as bounded `unknown` evidence without
  exposing exception messages.
- **Evidence:** Unit and real run-event -> snapshot -> verifier integration tests cover
  all three statuses, conflicts, truncation, provenance, rewards, plugin failures, and
  the model-self-report rejection invariant.

### 2026-08-16 — Phase 2 Task 2.3 / Eligibility and Complexity

- **Completed:** Added immutable eligibility hints/decision contracts and explicit
  `no_skill` / `skill_used` branches.
- **Completed:** Enforced verified `success` plus configurable minimum confidence
  before any trajectory can be admitted.
- **Completed:** Added deterministic complexity signals for tool-call count, error
  followed by a successful path, snapshot/extractor user corrections, trusted
  non-trivial workflow hints, and explicit remember/create-skill requests.
- **Completed:** Added English/Chinese explicit-request matching with negation guards
  for instructions such as "do not create a skill" and "不要记住这个流程".
- **Completed:** Rejected `safety_capped`, `loop_capped`, `token_capped`, and
  `subagent_limit_capped` runs regardless of apparent success.
- **Completed:** Added `skill_evolution.evidence` admission settings and bumped the
  example configuration schema from version 33 to 34.
- **Evidence:** Focused tests cover every signal, configurable signal gates, exact
  confidence boundary, simple-task rejection, recovered-run admission, both Skill
  branches, capped runs, config upgrade, and config hot reload.

### 2026-08-16 — Phase 3 Task 3.1 / Deterministic Pre-Extraction

- **Completed:** Added a versioned, hash-validated deterministic pre-extraction
  contract for eligible traces.
- **Completed:** Derived ordered content-free tool steps, tool/error signatures,
  environment, verified outcome, eligibility/complexity, observed Skill identities,
  artifact paths, event branch, and task-input hash without an LLM.
- **Completed:** Added at most 64 hash-addressed candidate segments covering task
  input, selected tool steps, corrections, artifact references, and final answer.
- **Completed:** Preserved every tool step structurally through parameter/result
  hashes while prioritizing error steps and the execution tail for bounded semantic
  excerpts.
- **Completed:** Added head/tail text truncation and omission of probable binary,
  control-byte, data-URI, and encoded payloads from both tool arguments and results.
- **Completed:** Rejected ineligible traces and mismatched outcome/eligibility/Skill
  branch inputs before extraction.
- **Evidence:** Fixture-based recovered-run tests cover deterministic JSON
  round-trips, extraction-hash integrity, both event branches, tool budgets, error/tail
  preservation, redaction, artifacts, and encoded/large-content handling.

### 2026-08-16 — Phase 3 Task 3.2 / Structured LLM Extraction

- **Completed:** Added a provider-neutral async structured extractor using a strict
  Pydantic JSON schema embedded in the prompt and validated after every response.
- **Completed:** Extracted task signature/goal, successful path, failed attempts,
  effective corrections, reusable lessons, Skill gaps, and candidate target.
- **Completed:** Required every semantic item to cite existing candidate segment IDs;
  persisted per-item evidence links plus deduplicated hash-addressed provenance.
- **Completed:** Validated new-skill vs. patch branch semantics and required patch
  targets to match an observed Skill name/hash.
- **Completed:** Retried only malformed output and recognized transient provider
  failures with a bounded 1-5 attempt budget. Permanent errors and exhausted malformed
  responses fail closed to no event.
- **Completed:** Added optional `extract_and_persist()`; the Store is called only after
  strict schema, provenance, branch, and final `EvolutionEvent` validation succeeds.
- **Completed:** Persisted extraction model name, Prompt version, deterministic
  pre-extraction hash, source snapshot hash, and a model+Prompt extractor version used
  by the event idempotency key.
- **Completed:** Added `skill_evolution.extraction_model_name` and bumped example/local
  configuration from version 34 to 35.
- **Evidence:** Fake-model tests cover strict schema, both branches, provenance retry,
  malformed exhaustion, transient/permanent errors, Markdown rejection, target
  mismatch, prompt bounds, event persistence, and extractor metadata.

### 2026-08-16 — Phase 4 Task 4.2 / Optional Semantic Retrieval

- **Decision:** The reproducible local semantic baseline is Ollama embeddings plus
  Qdrant over HTTP. No provider SDK is required.
- **Decision:** Embeddings remain disabled by default. An unavailable embedding model
  or vector store must return deterministic candidates instead of failing the online
  task.
- **Completed:** Added provider-neutral embedding and vector-store interfaces, an
  Ollama provider, and a Qdrant adapter that isolates physical collections by
  embedding model/version.
- **Completed:** Persisted model name, immutable model digest, document hash, user,
  event branch, and patch target with each vector; Qdrant queries enforce those hard
  partitions.
- **Completed:** Added hybrid candidate retrieval that unions deterministic and
  semantic candidates while retaining both scores and source provenance.
- **Completed:** Added optional semantic threshold/top-K/provider/vector-store config
  and bumped example/local configuration from version 36 to 37.
- **Evidence:** Unit tests cover disabled mode, semantic expansion, model metadata,
  user/branch/patch-target isolation, malformed service behavior, and both fallback
  paths. An opt-in integration test passed against local Ollama `nomic-embed-text`
  and Qdrant 1.19.

### 2026-08-16 — Phase 4 Task 4.3 / Cluster Confirmation and Readiness

- **Decision:** Candidate retrieval never establishes membership by itself. Every
  non-prototype member requires deterministic or semantic retrieval provenance plus
  a strict structured LLM same-workflow decision.
- **Decision:** `K=3` means both at least three confirmed events and at least three
  distinct `run_id` values. Replays or repeated extraction from one run cannot
  satisfy readiness.
- **Completed:** Added bounded prototype/candidate prompts, strict JSON-schema output,
  malformed/transient retry limits, and fail-closed unconfirmed candidates.
- **Completed:** Added explicit same-workflow, conditional-environment-branch, and
  different-workflow relationships. Compatible OS/runtime branches remain in one
  cluster with their condition preserved.
- **Completed:** Added member-level deterministic/semantic/LLM evidence, confirmation
  model/Prompt metadata, contradiction flags, rejected/unconfirmed candidate lists,
  and readiness blockers.
- **Completed:** Contradictory or unconfirmed evidence keeps a cluster collecting;
  only blocker-free clusters meeting both event and distinct-run thresholds become
  ready.
- **Completed:** Added readiness thresholds and confirmation model settings, bumping
  example/local configuration from version 37 to 38.
- **Evidence:** Tests cover ready persistence/query, same-run replay rejection,
  environment branches, contradictions, different workflows, semantic provenance,
  missing provenance, malformed output, bounded prompts, and config upgrade.

### 2026-08-16 — Phase 5 Task 5.1 / New-Skill Proposal Distillation

- **Decision:** Distillation can only create a staged Proposal. It cannot call
  `skill_manage`, write a Skill directory, approve, or publish.
- **Decision:** Mandatory common/verification rules and supporting files require
  evidence from at least two distinct runs. Conditional rules may preserve a
  single-environment observation as explicitly conditional.
- **Decision:** Conflicts are never resolved implicitly. They remain Proposal risks
  and force manual review; unsupported one-off observations are excluded from
  `SKILL.md`.
- **Completed:** Added strict JSON-schema distillation over structured events and
  bounded provenance excerpts, with malformed/transient retry and fail-closed output.
- **Completed:** Deterministically rendered complete `SKILL.md` frontmatter, overview,
  common workflow, environment guidance, and verification sections.
- **Completed:** Added repeated-evidence-gated supporting files and executable-file
  manual review.
- **Completed:** Added per-file/per-section event mappings, source Cluster hash,
  distiller model/Prompt metadata, deterministic Proposal IDs, and idempotent Store
  persistence.
- **Completed:** Added `skill_evolution.distillation_model_name`, bumping example/local
  configuration from version 38 to 39.
- **Evidence:** Tests cover complete rendering, repeated-run evidence, unknown IDs,
  conflict review, executable review, one-off exclusion, prompt bounds, ineligible
  clusters, deterministic IDs, Store idempotency, and config upgrade.

### 2026-08-16 — Phase 5 Task 5.2 / Existing-Skill Patch Distillation

- **Decision:** The source hash recorded by trajectories and the current publication
  base hash are distinct concepts. Source hash proves provenance; Proposal
  `base_skill_hash` is the future publication CAS token.
- **Decision:** Existing-Skill updates use exact `find/replace/expected_count=1`
  operations only. Full-file replacement is rejected during distillation.
- **Completed:** Added current/history Skill version resolution by SHA-256 using the
  existing `read_custom_skill()` and JSONL `prev_content`/`new_content` APIs.
- **Completed:** Added strict targeted patch output, per-operation evidence,
  deterministic in-memory application, final frontmatter/name validation, and
  unaffected-content preservation.
- **Completed:** Added explicit environment conditions for one-environment changes;
  general changes require repeated distinct-run evidence.
- **Completed:** Added double-read current hash checks. If the Skill changes while the
  model runs, the result is discarded and regenerated against the new current version.
- **Completed:** Patch Proposals retain structured operations, source Skill hashes,
  current base hash, rendered candidate `SKILL.md`, source Cluster hash, and
  per-operation evidence mappings.
- **Completed:** Conflicts remain risks and force manual review. Distillation and Store
  persistence never write the Skill.
- **Evidence:** Tests cover historical version recovery, stale-base restart, exact
  target counts, full-edit rejection, mixed-version rejection, environment conditions,
  conflict review, unaffected instructions, Store idempotency, and no-write behavior.

### 2026-08-16 — Phase 6 Task 6.1 / Replay Task Specification

- **Decision:** Automatic replay is fail-closed. It requires a successful source
  snapshot and event, non-empty task input, a complete bounded fixture, at least one
  deterministic verifier, no external credentials, and no network dependency that
  the side-effect policy blocks.
- **Decision:** Replay workspaces are disposable host-side evaluation roots, not the
  source run workspace or production Skill library. Candidate files are materialized
  read-only under a temporary `skills/` tree and are never published from this layer.
- **Completed:** Added versioned, hash-validated binary fixture and replay task
  contracts covering environment requirements, command/artifact verifiers, timeout,
  replayability, manual-review reasons, and allowed side effects.
- **Completed:** Added off-loop fixture capture with normalized relative paths,
  symlink/traversal rejection, per-file/total size bounds, and deterministic hashes.
- **Completed:** Added isolated `workspace/`, `outputs/`, and `skills/` materialization,
  exception-safe cleanup, staged-Proposal/user validation, and side-effect auditing
  with changed-file and written-byte limits.
- **Completed:** Added conservative automatic/manual-review classification for
  truncated traces, incomplete fixtures, missing verifiers or task input, blocked
  network requirements, and required external secrets.
- **Evidence:** Fourteen focused tests cover stable/binary fixtures, traversal and
  symlink rejection, source identity checks, replayability classification, model
  invariant rejection, candidate isolation, source/production immutability, cleanup,
  and side-effect policy enforcement.
- **Evidence:** Worker/config/event-contract regression passed with 101 tests. The
  complete backend offline suite passed with 11,414 tests and 77 skips.
- **Boundary:** This task defines and materializes replay inputs only. It does not run
  an agent, execute verifiers, score a candidate, approve a Proposal, or publish a
  Skill; those begin in Tasks 6.2 and 6.3.

### 2026-08-17 — Phase 6 Task 6.2 / New-Skill Proposal Evaluation

- **Decision:** Agent and command execution are delegated to a provider-neutral
  `ReplayRuntime` that must enforce the supplied replay root. The evaluator never
  invokes a host shell directly, and candidate runs receive an explicit candidate
  path so Skill use does not depend on model selection.
- **Decision:** Source tasks must exactly cover Proposal supporting events. Source and
  held-out tasks must be independent, user-scoped, uniquely identified, and share the
  same persisted task family.
- **Decision:** Every automatic task runs twice in separate disposable workspaces:
  once with no Skill and once with the staged candidate. Manual-only tasks block the
  automatic suite before any runtime call.
- **Completed:** Added paired source/held-out/baseline orchestration with whole-agent
  and per-command timeouts, deterministic command/artifact verification, redacted
  stable error codes, tool/token/latency metrics, and hash/size-only artifact records.
- **Completed:** Added changed-file/byte policy enforcement, fail-closed symlink
  handling, and cleanup that never follows a replay symlink during permission repair.
- **Completed:** Added configurable source and held-out pass thresholds under
  `skill_evolution.quality`, bumping example/local configuration from version 39 to
  40. Held-out failure can reject a candidate even when every source replay passes.
- **Completed:** Added deterministic evaluation IDs, Store idempotency, staged to
  validating transitions, and validating to rejected transitions. Passing and
  manual-review evaluations remain validating for Phase 7 approval policy.
- **Completed:** Raw `SkillEvaluation` results are persisted separately from aggregate
  quality; `quality_score=0.0` is explicitly marked deferred to Task 6.4.
- **Evidence:** Eighteen focused tests cover pairing, candidate activation, family and
  evidence coverage, thresholds, metrics, artifacts, errors, timeout, manual review,
  side effects, symlink isolation, idempotency, and Proposal state transitions.
- **Evidence:** All Skill evolution/config tests passed with 206 tests and one skip;
  worker/config/event-contract regression passed with 102 tests; the complete backend
  offline suite passed with 11,433 tests and 77 skips.
- **Boundary:** Evaluation is explicitly callable when a conforming `ReplayRuntime` is
  supplied. The worker/background coordinator and production runtime adapter are not
  wired yet. Existing-Skill regression evaluation starts in Task 6.3, aggregate
  quality remains Task 6.4, and no Skill is approved or published here.

## 14. Final Completion Criteria

The non-RL project is complete when:

- a successful complex run can be deterministically admitted or rejected;
- eligible trajectories become validated structured events;
- three independent same-family events produce a staged skill proposal;
- both new-skill and existing-skill update branches work;
- proposals are replayed against source and held-out tasks;
- unsafe or regressive proposals are rejected;
- approved proposals publish to per-user custom storage with version history;
- rollback works;
- online runs remain unaffected by evolution worker failures;
- experiments demonstrate the effect of multi-trajectory aggregation against the
  defined baselines;
- all focused, backend, security, and blocking-I/O tests pass.
