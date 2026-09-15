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
- Making direct publication the distributed default; it remains an explicit
  operator opt-in because it bypasses effectiveness evaluation and approval.

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
5. **No Skill writes inside the distiller.** Distillation always persists a staged
   proposal first. `manual` and `eligible_auto` keep evaluation/approval gates;
   explicit `direct` mode hands the staged Proposal to the separate publisher.
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
    candidate version, optional evaluation identity, and a reversible snapshot.
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
    mode: manual # manual | eligible_auto | direct
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

- [x] Run old-skill and new-skill paired evaluations.
- [x] Include historical regression tasks for the base skill.
- [x] Reject updates above configured regression rate.
- [x] Require manual review for executable supporting-file changes.

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

- [x] Store raw metrics separately from aggregate quality score.
- [x] Version the scoring formula.
- [x] Require minimum sample counts before assigning "high quality".

Acceptance:

- A skill cannot become high quality from only its three source runs.

### Phase 7: Approval, Publication, and Rollback

#### Task 7.1: Refactor reusable skill mutation service

Files:

- Extract common mutation logic from `tools/skill_manage_tool.py`.
- Keep the LangChain tool as a thin adapter.
- Add mutation-service tests plus existing tool regressions.

- [x] Preserve validation, per-user locking, SkillScan, LLM moderation, history,
      and cache refresh.
- [x] Add expected base hash/version for compare-and-swap publication.
- [x] Add proposal and evaluation IDs to mutation history.

Acceptance:

- Manual `skill_manage` behavior remains backward compatible.
- Evolution publisher cannot bypass existing security controls.

#### Task 7.2: Add proposal approval policy

- [x] Default to manual approval.
- [x] Allow future auto-publication only for non-executable, low-risk changes with
      passing held-out evaluation.
- [x] Never auto-publish executable files in the first version.
- [x] Add explicit rejection reason and proposal expiration.

#### Task 7.3: Add version snapshots and rollback

- [x] Snapshot all skill package files before publication.
- [x] Record base and published hashes.
- [x] Add rollback operation that creates a new history entry.
- [x] Re-run security checks before restoring executable content.

Acceptance:

- Every published proposal can be rolled back without reconstructing state from
  model output.

### Phase 8: Asynchronous Coordination and Operations

#### Task 8.1: Add evolution coordinator

Files:

- Create `deerflow/skill_evolution/coordinator.py`.
- Create `deerflow/skill_evolution/worker.py`.
- Add lifecycle and shutdown tests.

- [x] Enqueue eligible run IDs after durable run finalization.
- [x] Keep extraction/distillation off the request event loop.
- [x] Use bounded queues, retries, backoff, and idempotency keys.
- [x] Persist state before acknowledging work.
- [x] Drain or safely abandon work during shutdown according to configured timeout.

Acceptance:

- Skill evolution failure cannot change the completed user run result.
- Restart resumes pending durable work without duplication.

#### Task 8.2: Add observability

- [x] Emit events for admitted, rejected, extracted, clustered, ready, distilled,
      evaluated, approved, published, rejected, and rolled back.
- [x] Add metrics for queue depth, extraction failures, cluster purity samples,
      proposal pass rate, publication rate, regression rate, and quality lift.
- [x] Log IDs and hashes, not raw sensitive content.

### Phase 9: API and Minimal Review Surface

#### Task 9.1: Add read-only APIs

- [x] List events, clusters, proposals, evaluations, and skill versions per user.
- [x] Enforce ownership and admin rules.
- [x] Paginate all collections.
- [x] Return compact/redacted data by default.

#### Task 9.2: Add approval and rollback APIs

- [x] Approve/reject proposals.
- [x] Publish approved proposals.
- [x] Roll back published versions.
- [x] Protect all mutations with CSRF/authz/ownership checks.

Frontend UI is optional for the first research milestone; APIs plus CLI are sufficient.

### Phase 10: Credit Logging for Future RL

#### Task 10.1: Record selection actions

- [x] Record generated search query, candidate skill IDs/hashes, rank scores,
      selected skill, and no-skill decision.
- [x] Record policy model and prompt version.
- [x] Record log probabilities only when the provider exposes them reliably.

#### Task 10.2: Record utilization evidence

- [x] Record activation source, loaded skill version, relevant tool calls,
      deviations, outcome reward, and cost.
- [x] Do not infer instruction adherence from text matching alone; retain it as
      an optional model-derived feature.

#### Task 10.3: Record distillation evidence

- [x] Link proposal to source events and future tasks using the published version.
- [x] Compute delayed marginal utility after sufficient future evaluations.
- [x] Preserve raw outcome series for alternative credit formulas.

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

- [x] `K=1` vs `K=3` vs `K=5`.
- [x] deterministic grouping vs LLM grouping vs hybrid.
- [x] immediate publish vs staged evaluation.
- [x] success path only vs success path plus recovered failures.
- [x] no-skill creation branch vs existing-skill patch branch.

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

- [x] Use paired tasks/seeds for old/new/no-skill comparisons.
- [x] Report confidence intervals.
- [x] Separate task-family and aggregate results.
- [x] Publish failed and rejected proposal counts, not only successful cases.

Implementation status:

- The frozen 60-task manifest, 12-condition ablation matrix, executor protocol,
  JSONL result contract, paired bootstrap intervals, exact McNemar test, and
  aggregate/per-family reports are implemented.
- A full 2,160-row `deterministic_smoke` matrix validates the experiment plumbing.
- A production suite now materializes all 60 tasks with hidden deterministic
  verifiers. `DockerReplayRuntime` runs model-selected tools in disposable,
  network-disabled/read-only-rootfs containers with read-only Skill mounts.
- The generated-candidate executor uses real deterministic/LLM/hybrid grouping,
  Qwen distillation for both create and patch branches, staged source replay, and
  held-out execution. Cohort-level checkpoints make the long run resumable.
- The completed `production_replay` matrix contains 2,160 unique rows for Qwen3 8B
  digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.
  Its report contains 12 conditions, 12 family sections, and 11 paired
  comparisons. Smoke and production rows remain schema-separated.
- The result does not demonstrate an aggregate K=3 lift over no evolution on this
  suite: both held-out rates are `0.6667`. It does show that immediate publication
  is worse than K=3 staged evaluation by `-0.0833` held-out success
  (95% paired-bootstrap CI `[-0.1528, -0.0278]`, exact McNemar `p=0.03125`).

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
- [x] What minimum held-out sample count is required for a high-quality designation?
- [ ] Can a custom skill shadow a public skill automatically, or only after approval?
- [x] What is the exact formula for aggregate skill quality?

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

### 2026-08-30 — Explicit Direct Publication Mode

- **Decision:** Add `skill_evolution.publication.mode=direct` for operators who
  intentionally defer effectiveness measurement to offline experiments.
- **Decision:** Direct mode publishes a staged K-ready Proposal immediately and
  creates no synthetic `SkillEvaluation` or approval record.
- **Decision:** Direct mode still uses `SkillPublicationService` and
  `SkillMutationService`; path/package validation, SkillScan, LLM security
  moderation, base/package CAS, atomic replacement, history, snapshots, Credit,
  and rollback remain mandatory.
- **Decision:** `manual` remains the distributed default. The local development
  configuration may opt into `direct`.
- **Decision:** A direct publication persists `evaluation_id=null` so audit APIs
  distinguish a skipped gate from an evaluated approval.

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

### 2026-08-17 — Phase 6 Task 6.3 / Existing-Skill Patch Evaluation

- **Decision:** Patch evaluation requires a complete, same-user `ReplaySkillPackage`
  whose exact `SKILL.md` hash equals the Proposal `base_skill_hash`. Replay files use
  content-preserving validation so leading/trailing bytes cannot silently change the
  CAS identity.
- **Decision:** The candidate package is built in memory by overlaying Proposal files
  on the complete base package. Old and candidate packages are materialized read-only
  in separate disposable workspaces; production Skill storage is never mounted or
  modified.
- **Decision:** Regression rate is the number of historical tasks where the old Skill
  succeeded and the candidate failed, divided by historical tasks where the old Skill
  succeeded. Old-Skill failures do not count as candidate regressions, and a suite
  with no successful old-Skill regression sample requires manual review.
- **Completed:** Added `base_skill` replay condition and shared source/regression
  runner orchestration. Source tasks exactly cover supporting evidence; historical
  regression tasks are independent, unique, and user-scoped.
- **Completed:** Added configurable `max_regression_rate` under
  `skill_evolution.quality` with fail-closed default `0.0`, bumping example/local
  configuration from version 40 to 41.
- **Completed:** Added patch evaluation IDs, raw base/candidate/source/regression
  results, source pass rate, regression sample/rate, artifact/metric/error retention,
  Store idempotency, and rejected Proposal transitions.
- **Completed:** Missing regression tasks, incomplete base packages, no successful
  old-Skill regression sample, Proposal review flags, and executable supporting-file
  changes require manual review. Source failure, side-effect violation, and excessive
  regression reject.
- **Completed:** `SkillEvaluation` approval now accepts either held-out results for a
  new Skill or regression results for an existing-Skill patch; aggregate quality
  remains deferred.
- **Evidence:** Sixteen focused tests cover exact content/hash preservation, package
  merge, user/hash mismatch, old/new source and regression pairs, configured
  thresholds, old-failure exclusion, incomplete/missing samples, manual-only tasks,
  executable changes, idempotency, and Proposal state transitions.
- **Evidence:** All Skill evolution/config tests passed with 224 tests and one skip;
  worker/config/event-contract regression passed with 103 tests; the complete backend
  offline suite passed with 11,451 tests and 77 skips.
- **Boundary:** Patch evaluation is explicitly callable with a complete base package
  and conforming `ReplayRuntime`. The worker/coordinator still does not assemble
  suites or invoke it automatically. Aggregate quality, approval, publication, and
  rollback remain later phases.

### 2026-08-17 — Phase 6 Task 6.4 / Versioned Skill Quality

- **Decision:** `skill-quality-v1` is deterministic and model-free. It stores eight
  independent dimensions: success lift (`0.25`), held-out generalization (`0.20`),
  tool-call reduction (`0.10`), token reduction (`0.10`), latency reduction (`0.10`),
  environment robustness (`0.10`), regression resistance (`0.10`), and safety
  (`0.05`).
- **Decision:** Success lift and efficiency deltas in `[-1, 1]` use symmetric
  normalization `(value + 1) / 2`. Held-out success and worst-environment success are
  direct scores. Regression resistance is `1 - regression_rate`; safety is
  `1 - risk`. Unavailable dimensions are excluded from the weight denominator rather
  than treated as zero.
- **Decision:** A high-quality designation requires an approved evaluation, aggregate
  score at least `0.75`, no safety/review blocker, at least five candidate tasks, two
  held-out tasks for a new Skill or two old-success regression pairs for a Patch, and
  at least two distinct environment fingerprints. Three source runs alone can never
  qualify.
- **Completed:** Added immutable `SkillQualityDimension` and `SkillQualityReport`
  contracts with formula version, raw derived value, normalized score, weight, sample
  count, aggregate, designation, sample counts, and explicit blockers.
- **Completed:** Added deterministic `compute_skill_quality()` and
  `apply_skill_quality()`. Raw task metrics/results remain unchanged and separate;
  only the report and indexed `quality_score` are added to the evaluation.
- **Completed:** Added environment fingerprints to raw task results and integrated
  scoring into both new-Skill and Patch evaluators, including manual/rejected results.
  Evaluator identity versions advanced to v2 and include the quality formula version.
- **Completed:** Failed candidate/base pairs cannot earn efficiency gains. Malformed
  unpaired raw results fail closed. Model validation prevents forged high-quality
  reports with insufficient samples, blockers, low score, or non-approved decisions.
- **Completed:** Added high-quality threshold and sample gates under
  `skill_evolution.quality`, bumping example/local configuration from version 41 to
  42.
- **Evidence:** Thirteen focused quality tests cover all dimensions, both Proposal
  branches, exact formula metadata, high-quality assignment, source-only/held-out/
  environment insufficiency, regression, safety, failed-pair efficiency, raw-result
  immutability, malformed pairing, model forgery, and configurable threshold.
- **Evidence:** Quality/evaluator/model/store/config focused tests passed with 99
  tests; all Skill evolution/config tests passed with 238 tests and one skip;
  worker/config/event-contract regression passed with 104 tests; the complete backend
  offline suite passed with 11,465 tests and 77 skips.
- **Boundary:** Quality is calculated per persisted evaluation and does not itself
  approve or publish a Skill. Cross-evaluation longitudinal stability and RL
  internalization remain later work; the worker/coordinator still does not invoke the
  evolution pipeline automatically.

### 2026-08-17 — Phase 7 Task 7.1 / Reusable Skill Mutation Service

- **Decision:** `deerflow.skills.mutation.SkillMutationService` is the shared security
  and persistence boundary for both manual agent mutations and future Evolution
  publication. Callers may not write Proposal content directly through storage.
- **Decision:** A supplied `expected_base_hash` is checked before security scanning
  for fast stale-Proposal rejection and checked again inside the final
  cross-process Skill projection/storage write lock. The second check is authoritative
  and closes the scan-time TOCTOU window. Create uses the same locked boundary with
  `require_absent=True`.
- **Decision:** Manual `skill_manage` requests retain their existing tool schema and
  do not require CAS metadata. Evolution-authored requests require paired
  `proposal_id` and `evaluation_id` values, which are written with the expected,
  previous, and resulting Skill hashes into mutation history.
- **Completed:** Extracted create, patch, edit, delete, support-file write, and
  support-file removal into the reusable service while preserving Skill-name/content/
  path validation, per-user/per-Skill serialization, full-candidate SkillScan, LLM
  moderation, history, projection rebuilds, and prompt-cache refresh.
- **Completed:** Reduced `skill_manage_tool.py` to a runtime adapter that resolves the
  user/thread and delegates to the service. Existing scanner injection points and
  response/error behavior remain compatible with the existing tool tests.
- **Completed:** Added storage CAS inputs and `SkillStorageConflict` handling to local
  and user-scoped writes and deletes. Atomic replacement, CAS validation, and
  projection mutation now share the same write lock.
- **Evidence:** Twelve mutation-service tests and ten existing `skill_manage` tests
  pass. The focused Skill security/router/storage/projection suite passes with 277
  tests, and the complete backend offline suite passes with 11,477 tests and 77 skips.
- **Evidence:** Removing the final storage-layer CAS caused
  `test_atomic_cas_catches_drift_during_security_scan` to fail and allowed a competing
  write to be silently overwritten. Restoring the locked CAS returned the test to
  green; a separate concurrent-create test proves `require_absent=True`.
- **Boundary:** This task provides the safe mutation primitive only. No approval
  policy or Evolution publisher calls it yet, and no Proposal is automatically
  published. Approval policy begins in Task 7.2; package snapshots and rollback begin
  in Task 7.3.

### 2026-08-17 — Phase 7 Task 7.2 / Proposal Approval Policy

- **Decision:** `skill-approval-v1` is deterministic and model-free. The default
  `skill_evolution.publication.mode=manual` leaves an evaluated Proposal in
  `validating` until an identified human reviewer approves or rejects it.
- **Decision:** The opt-in `eligible_auto` mode can only move a Patch Proposal to
  `approved`; it never publishes. Eligibility requires the non-executable auto flag,
  no executable file, no Proposal risk or manual-review marker, no evaluation safety
  blocker, an approved evaluation, and candidate held-out results that all succeed.
  New-Skill creation remains manual in v1.
- **Decision:** `allow_executable_auto_publish` is typed as the literal `false`, and
  `require_held_out_evaluation` as the literal `true`. Configuration cannot weaken
  either v1 safety invariant.
- **Decision:** New Proposals persist `expires_at=created_at+180 days`; legacy
  Proposals use the configured 180-day fallback. Expiration is evaluated before
  automatic or manual approval, and an approved-but-unpublished Proposal can still
  expire.
- **Completed:** Added immutable approval assessments, manual approval requests, and
  idempotent policy results. Manual approval can resolve evaluation/manual-review
  blockers, including inspected executable content, but cannot override a rejected
  evaluation.
- **Completed:** Added append-only `ProposalStatusTransition` metadata to Proposal
  JSON payloads. Each transition records source, reason code/detail, timestamp,
  evaluation ID, reviewer ID when applicable, and policy version. Store transitions
  retain status CAS in both memory and SQL implementations; no schema migration was
  required.
- **Completed:** Evaluators now record `evaluation_started` and explicit
  `evaluation_rejected` transitions. Identical concurrent approvals collapse to one
  write and one idempotent winner read; conflicting outcomes still fail CAS.
- **Completed:** Added `skill_evolution.publication` configuration and bumped example
  and local config from version 42 to 43.
- **Evidence:** Eleven approval-policy tests cover default manual behavior, safe opt-in
  approval, new-Skill/manual and executable/manual invariants, risk and held-out
  blockers, rejection reasons, pending/approved expiration, human approve/reject,
  ownership, and concurrent idempotency. Approval plus memory/SQL Store tests pass
  with 32 tests, including legacy terminal payload compatibility.
- **Evidence:** The complete backend offline suite passes with 11,491 tests and 77
  skips.
- **Evidence:** Removing the executable-file gate caused
  `test_executable_change_can_never_be_auto_approved` to fail because an executable
  Proposal became `approved`. Restoring the gate returned the test to green.
- **Boundary:** Phase 7.2 persists approval state only. There is no approval API,
  background approval invocation, publication snapshot, rollback, or mutation-service
  call. `approved` is not `published`; Phase 7.3 owns package snapshots, publication,
  and rollback.

### 2026-08-17 — Phase 7 Task 7.3 / Versioned Publication and Rollback

- **Decision:** Version snapshots are binary-safe complete packages, not only
  `SKILL.md`. Each sorted file record stores normalized path, exact base64 bytes,
  byte size, SHA-256, and executable bit. The snapshot stores `SKILL.md` and package
  hashes and rejects symlinks, unsupported entries, traversal, more than 256 files,
  files over 8 MiB, or packages over 32 MiB.
- **Decision:** Candidate security preflight completes before snapshot bytes enter
  the SQL publication record. Final package mutation repeats the same security
  boundary before storage CAS, so scanner failure cannot persist a sensitive or
  unsafe candidate and scan-time package drift cannot be overwritten.
- **Decision:** Publication uses durable states `preparing -> published ->
  rolled_back`. The Proposal separately reserves `approved -> publishing` before
  mutation, then becomes `published`; rollback records `published -> rolled_back`.
  Reservation prevents expiration/rejection from racing an active publisher.
- **Decision:** The persisted `preparing` record contains both base and deterministic
  candidate snapshots before any Skill write. A create base is an explicit absent
  snapshot. This makes publication and rollback restart-recoverable without model
  output.
- **Completed:** Added package-level mutation to `SkillMutationService`: full-package
  SkillScan, bounded moderation of Proposal files plus mandatory `SKILL.md` and every
  executable, safe support-path validation, base `SKILL.md` hash CAS, complete package
  hash CAS, one storage-lock directory exchange, history, projection rebuild, and
  prompt-cache refresh.
- **Completed:** Storage stages the complete candidate beside the target, moves the
  old package to a hidden backup under the projection lock, and restores that backup
  on candidate rename, permission repair, or history-write failure. Create and delete
  use the same package operation.
- **Completed:** `SkillPublicationService.publish()` accepts only non-expired approved
  create/patch Proposals. It persists preflighted snapshots, reserves publication,
  writes through the mutation service, captures the actual published package, records
  base/published Skill and package hashes, and finalizes both Store records.
- **Completed:** `rollback()` reads only the persisted base snapshot. It reruns full
  SkillScan and mandatory executable moderation, requires the live package to match
  the published hash, atomically restores every file and execution bit (or deletes a
  newly created Skill), appends `evolution_rollback` history, and records reviewer
  identity.
- **Completed:** Identical publication/rollback retries are idempotent. If process
  exit occurs after the package or publication row commits but before Proposal state,
  the next call recognizes the exact snapshot and repairs the remaining status
  transition without rerunning mutation.
- **Completed:** Added per-user `skill_evolution_publications` SQL persistence and
  Alembic revision `0013_skill_publications`; fresh, 0011-chain, and direct 0012
  upgrades are covered. No config schema change was required.
- **Evidence:** Eleven publication tests and the mutation/Store/migration suites pass
  with 56 focused tests. Skill evolution/mutation/config regression passes with 292
  tests and one skip; bootstrap/migration regression passes with 36 tests.
- **Evidence:** The complete backend offline suite passes with 11,509 tests and 77
  skips.
- **Evidence:** Bypassing full-package SkillScan caused
  `test_rollback_security_failure_preserves_published_package` to fail because unsafe
  rollback content was restored. Restoring the scan returned the test to green.
- **Boundary:** Publication and rollback are explicit service calls only. Phase 8
  still must coordinate the full background pipeline, and Phase 9 must expose
  ownership/authz/CSRF-protected approval, publication, and rollback APIs. No worker
  automatically publishes a Skill.

### 2026-08-17 — Phase 8 Task 8.1 / Durable Evolution Coordinator

- **Decision:** One redacted run snapshot maps to one deterministic
  `skill-evolution-pipeline-v1` job. Its idempotency key hashes user, run, snapshot,
  and pipeline version; duplicate run finalization returns the existing mutable job
  rather than creating duplicate evidence.
- **Decision:** Job state is `pending -> running -> completed`, with `retry` and
  terminal `dead` branches. Every claim receives a unique token, expiry, incremented
  attempt count, and revision. Completion, retry, and lease renewal require that
  token plus revision CAS, so an expired worker cannot overwrite a recovery worker.
- **Decision:** The bounded `asyncio.Queue` contains wake-up hints only. Enqueue
  commits the SQL/memory Store row first; queue saturation is ignored because the
  periodic database poll recovers pending, due-retry, and expired-running rows.
- **Decision:** The production background processor advances verification,
  eligibility, deterministic pre-extraction, strict structured extraction,
  deterministic/semantic grouping, strict K=3 confirmation, and new/Patch Proposal
  distillation. It stops at `staged`: production Replay-suite assembly and an
  isolated `ReplayRuntime` adapter are still absent, so no evaluation is fabricated.
- **Decision:** The coordinator never invokes approval, publication, rollback, or
  Skill mutation. Those operations remain explicit trusted services and retain the
  Phase 7 security/CAS boundaries.
- **Completed:** Added immutable `EvolutionJob`, Memory/SQL Store operations, indexed
  `skill_evolution_jobs`, and Alembic `0014_skill_evolution_jobs`. Fresh databases and
  upgrades from 0011, 0012, and 0013 are covered.
- **Completed:** Added renewable lease execution, deterministic exponential backoff,
  bounded exception-class error codes, max-attempt dead state, startup recovery,
  bounded local concurrency, and configured shutdown drain/cancellation recovery.
- **Completed:** Added process-local striped Cluster continuation. A restart that
  finds a persisted ready Cluster without a completed Proposal resumes distillation;
  a losing cross-worker Cluster CAS leaves continuation to the winner's durable job.
- **Completed:** Run finalization now awaits durable enqueue only after terminal
  status and the redacted trace are persisted. Trace/enqueue/coordinator failures are
  separately caught and cannot change the completed user run result.
- **Completed:** Gateway lifecycle creates a shared evolution Store and processor,
  starts the coordinator when evolution is enabled at startup, and stops it before
  database teardown. Enqueues received after local stop remain durable for restart.
- **Completed:** Added `skill_evolution.coordinator` operational settings and bumped
  example/local config from version 43 to 44.
- **Evidence:** Durable Store/coordinator/pipeline/worker tests cover idempotent
  enqueue, queue saturation, exclusive claims, stale claim rejection, lease renewal,
  retry due times, dead state, SQL recreation, startup recovery, shutdown drain,
  timeout recovery, snapshot identity, extraction restart idempotency, and non-fatal
  run enqueue failure.
- **Evidence:** Phase 8.1 plus migration/bootstrap/Gateway focused regression passes
  with 130 tests. Skill evolution and mutation regression passes with 288 tests and
  one skip.
- **Evidence:** The complete backend offline suite passes with 11,537 tests, 77
  skips, and 39 warnings.
- **Boundary:** Phase 8.2 still owns lifecycle event/metric observability. Phase 9
  owns authenticated read, review, publication, and rollback APIs. Replay suite
  synthesis and a production isolated runtime adapter must be completed before the
  background processor can evaluate a staged Proposal.

### 2026-08-17 — Phase 8 Task 8.2 / Content-Free Observability

- **Decision:** `deerflow.skill-evolution.observation.v1` is a strict structured
  event schema. It permits only lifecycle kind/stage, reason codes, bounded counts,
  run/job/event/Cluster/Proposal/evaluation/publication IDs, snapshot hashes, and
  SHA-256 hashes of user identity and Skill name. It has no free-text content,
  exception text, task input, model output, file path, or Proposal body field.
- **Decision:** Lifecycle records are process-local observability, not a second
  domain Store. They are written as structured INFO logs and retained in a bounded
  recent-event ring for later exporters/API integration. Deterministic observation
  IDs suppress same-process retry duplicates; domain durability remains in the
  existing event/Cluster/Proposal/evaluation/publication/job tables.
- **Decision:** Metrics are fixed-name, low-cardinality process aggregates. No user,
  Skill, task family, model output, reason text, or exception is a metric label.
  Distributions support bounded deduplication keys so job retries do not resample the
  same Cluster revision or evaluation.
- **Decision:** `proposal_pass_rate` is approved evaluations divided by all persisted
  evaluations. `publication_rate` is newly published versions divided by newly
  distilled Proposals observed in the process. `quality_lift` is the v1
  `success_lift` raw dimension; `regression_rate` is the v1 regression-resistance raw
  dimension; Cluster purity is confirmed members over all considered candidates.
- **Completed:** Added lifecycle kinds `admitted`, `rejected`, `extracted`,
  `clustered`, `ready`, `distilled`, `evaluated`, `approved`, `published`, and
  `rolled_back`, with one `rejected` kind differentiated by its fixed stage.
- **Completed:** Added durable backlog and active-worker gauges, extraction-failure
  counter, lifecycle counters, proposal/publication rates, and bounded
  Cluster-purity/regression/quality-lift distributions.
- **Completed:** Coordinator polls Memory/SQL job counts for `pending + retry`
  durable queue depth. Queue metrics are not derived from the wake-up hint queue.
- **Completed:** Pipeline emits admission/rejection, extraction, grouping,
  confirmation/readiness, and distillation events. A real three-independent-run test
  proves `ready -> distilled` after K=3.
- **Completed:** Both replay evaluators emit once after evaluation persistence and
  sample quality/regression metrics. Approval emits only after status CAS.
  Publication/rollback emit only after publication-row CAS; failed publication emits
  a content-free rejection reason code.
- **Completed:** Gateway owns the process observer on `app.state` and passes the same
  instance to coordinator and pipeline. Explicit evaluators, approval policies, and
  publication services default to that same process singleton.
- **Completed:** All observer calls use fail-open wrappers. Wrapper warnings include
  only operation, fixed metric name, lifecycle kind, and stage; the original
  exception message and traceback are deliberately omitted.
- **Completed:** No config schema or persistence schema changed; config remains
  version 44 and Alembic head remains `0014_skill_evolution_jobs`.
- **Evidence:** Observability, Skill evolution, mutation, and Gateway lifespan
  focused regression passes with 313 tests and one skip. Ruff and diff whitespace
  checks pass.
- **Evidence:** The complete backend offline suite passes with 11,550 tests, 77
  skips, and 39 warnings.
- **Boundary:** Metrics reset on process restart and no `/metrics` or read API is
  added in this task because the repository has no generic Prometheus exporter
  infrastructure. Phase 9 may expose compact owner/admin-safe snapshots; a future
  deployment exporter can consume the fixed snapshot without changing lifecycle
  producers.

## 14. Final Completion Criteria

The non-RL project is complete when:

- a successful complex run can be deterministically admitted or rejected;
- eligible trajectories become validated structured events;
- three independent same-family events produce a staged skill proposal;
- both new-skill and existing-skill update branches work;
- evaluated modes can replay proposals against source and held-out tasks and reject
  unsafe or regressive candidates;
- direct mode can publish staged proposals without Evaluation or approval;
- approved or explicitly direct proposals publish to per-user custom storage with
  version history;
- rollback works;
- online runs remain unaffected by evolution worker failures;
- experiments demonstrate the effect of multi-trajectory aggregation against the
  defined baselines;
- all focused, backend, security, and blocking-I/O tests pass.
