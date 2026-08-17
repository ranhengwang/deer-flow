# Self-Evolving Skills Implementation Notes

This is the append-only implementation record for
`2026-08-16-self-evolving-skills-development-plan.md`.

## Continuity Checklist

Before each implementation task:

- Read the development plan and these notes.
- Inspect the latest code and focused tests.
- Follow RED -> GREEN -> regression proof.
- Update completed checkboxes and append evidence here.
- Do not start deferred RL/Verl work before the external skill-evolution loop is
  stable.

## 2026-08-16 — Phase 0 and Task 1.1

### Decisions Frozen

- Evolution is per-user.
- Automatic admission requires verified success above the configured confidence.
- New skill publication requires manual approval in the research prototype.
- Non-interactive runs may stage proposals but may not publish new skills.
- The default evidence threshold is three independent runs.
- Raw redacted snapshots default to 7-day retention; structured evolution records
  default to 180-day retention.
- Initial benchmarks cover repository repair, structured data transformation, and
  shell/environment workflows.

### Implementation

- Added `deerflow.skill_evolution.models`.
- Added versioned strict contracts for:
  - outcome evidence;
  - new-skill and patch evidence events;
  - event clusters;
  - staged skill proposals;
  - task and skill evaluations.
- Added per-branch invariants:
  - new-skill evidence cannot claim a used skill;
  - patch evidence requires a matching used/target skill and at least one gap;
  - patch/edit/write proposals require a base skill hash;
  - create proposals reject a base skill hash.
- Added bounded text/collections, normalized relative proposal paths, UTC timestamps,
  immutable Pydantic models, and JSON round-trip support.

### Verification

```text
Focused tests: 12 passed
Ruff check: passed
Ruff format --check: passed
Negative mutation: widening confidence upper bound to 2.0 caused the intended test
to fail for confidence=1.01; restoring 1.0 returned the suite to green.
```

### Next Task

Phase 1 Task 1.2: define the async store contract and implement the in-memory store
with shared contract tests, idempotent event upsert, cluster readiness queries, and
optimistic proposal transitions.

## 2026-08-16 — Phase 1 Task 1.2

### Contract Adjustment

Before persistence, the event contract was extended with:

- `extractor_version`;
- `source_snapshot_hash`;
- `task_input_hash`.

Proposal and evaluation records now carry `user_id` directly. These fields make the
idempotency and isolation contracts enforceable without joining through unrelated
records.

### Implementation

- Added `deerflow.skill_evolution.store.base.SkillEvolutionStore`.
- Added `deerflow.skill_evolution.store.memory.InMemorySkillEvolutionStore`.
- Event writes are idempotent by `(user_id, run_id, extractor_version)`.
- Reusing an idempotency key or record ID with a changed payload raises
  `EvolutionStoreConflict`.
- The same run can be processed by a newer extractor version.
- All record keys are scoped by user.
- Ready-cluster queries require both `status=ready` and the configured independent-run
  threshold.
- Proposal status transitions use compare-and-swap and a fixed transition graph.

### Verification

```text
Focused model + memory-store tests: 21 passed
Ruff check: passed
Ruff format --check: passed
Negative mutation: removing the ready-status predicate returned a collecting cluster
and caused the intended focused test to fail; restoring it returned the suite to green.
Full backend suite: reached 100% with no observed failures after Task 1.1.
```

### Deferred

The store contract tests currently exercise the memory backend. Task 1.3 must add the
SQL backend and run the same behavioral contract against both implementations before
the cross-backend checkbox is complete.

### Next Task

Phase 1 Task 1.3: design SQL persistence models and migration, then parameterize the
store contract tests across memory and SQL backends.

## 2026-08-16 — Phase 1 Task 1.3

### Implementation

- Added four SQLAlchemy tables:
  - `skill_evolution_events`;
  - `skill_evolution_clusters`;
  - `skill_evolution_proposals`;
  - `skill_evolution_evaluations`.
- Each table stores complete versioned JSON payloads plus separately indexed lookup
  fields.
- Event rows enforce the unique idempotency key
  `(user_id, run_id, extractor_version)`.
- Added `SqlSkillEvolutionStore` with the same semantics as the memory backend:
  idempotent puts, payload-drift conflicts, user isolation, ready-cluster queries,
  optimistic proposal transitions, and evaluation persistence.
- Added Alembic revision `0012_skill_evolution`.
- Registered ORM rows through `deerflow.persistence.models`.
- Updated persistence tests whose expected migration head was `0011_mcp_tasks`.

### Verification

```text
Evolution model/store/migration tests: 36 passed
Persistence bootstrap/migration/autogenerate tests: 98 passed
Ruff check: passed
Ruff format --check: passed
SQLite durability: event survives engine disposal and recreation
PostgreSQL compatibility: every table and index compiles with the PostgreSQL dialect
Negative mutation: removing the event target-skill index caused the migration schema
test to fail; restoring the index returned the suite to green.
Full backend offline suite: 11273 passed, 76 skipped, 39 warnings.
```

### Behavioral Finding

The first SQL implementation validated the requested transition before checking the
stored proposal status. Shared contract tests caught the mismatch: a stale CAS attempt
returned "invalid transition" instead of "expected status mismatch". SQL now follows the
memory backend's order: read current status, reject stale expectations, then validate the
requested transition, with the final update still guarded atomically in SQL.

### Next Task

Phase 2 Task 2.1: define and test the bounded, redacted evolution trace snapshot before
wiring it into run completion.

## 2026-08-16 — Phase 2 Task 2.1

### Implementation

- Added immutable `EvolutionTraceSnapshot`, `TraceToolEvent`, and
  `TraceSkillEvent` contracts under `deerflow.skill_evolution.models`.
- Added `build_evolution_trace_snapshot()` with:
  - stable run-event ordering and tool-call/result pairing;
  - first visible task-input retention plus a bounded event tail;
  - hidden framework-message exclusion and user-correction capture;
  - slash/read Skill path, source, category, and content-hash capture;
  - `SKILL.md` body omission after hashing;
  - artifact extraction from workspace changes and delivery receipts;
  - recursive sensitive-key redaction, exact request/active-secret replacement,
    and host-to-virtual path masking;
  - deterministic snapshot hashing and self-event exclusion on retries.
- Added `skill_evolution.trace` in the `evolution` run-event category and updated
  the public run-event stream contract.
- Wired the owning run worker to create the snapshot only when
  `skill_evolution.enabled=true`, after journal/workspace/delivery/terminal
  persistence.
- Used `RunEventStore.put_if_absent` for one snapshot per run. Snapshot listing,
  building, validation, and persistence are wrapped as a non-fatal finalization
  step and cannot change the user run outcome.

### Verification

```text
Trajectory + worker tests: 8 passed
All evolution model/store/migration/trajectory/worker tests: 44 passed
Worker delivery/rollback/subagent persistence tests: 119 passed
Run event contract/store/journal tests: 181 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11281 passed, 76 skipped, 39 warnings
```

The first full-suite invocation inherited a PATH without `~/.local/bin`, so the
existing config-upgrade subprocess test could not find `uv`; the other 11280 tests
passed. Re-running that test and then the full suite with the installed `uv` on PATH
produced the clean result above.

### Regression Proof

- A store that raises only for `skill_evolution.trace` leaves the run in
  `success`.
- Re-running snapshot persistence returns the original event and hash.
- Disabled evolution produces no snapshot.
- Tests explicitly reject request secrets (including short embedded values), raw
  host paths, hidden context, and loaded Skill bodies in serialized snapshots.

### Next Task

Phase 2 Task 2.2: implement the pluggable outcome-verifier framework, starting with
deterministic evidence and no LLM judge dependency.

## 2026-08-16 — Phase 2 Task 2.2

### Implementation

- Added `deerflow.skill_evolution.verifier` with a bounded, synchronous
  `OutcomeVerifier` protocol. Verifiers receive only an immutable redacted snapshot
  plus explicit bounded evidence inputs.
- Added normalized verification signals and evidence inputs for:
  - latest recognized test command and its exit status;
  - task-specific command expectations;
  - prevalidated artifact existence, schema, and value checks;
  - environment rewards with explicit success/failure thresholds;
  - explicit user acceptance or rejection;
  - optional precomputed LLM judgment.
- A successful Run status and final answer text provide no positive evidence.
  Non-success terminal Run statuses are authoritative failures.
- Added priority-aware aggregation:
  - task-specific deterministic checks outrank generic automated tests;
  - explicit user evidence is lower priority;
  - equal-priority success/failure conflicts become `unknown`;
  - lower-priority conflicts reduce confidence;
  - LLM judgments are supporting-only and cannot establish success.
- Extended `OutcomeCheck` with per-signal confidence, authority, and priority so the
  aggregate decision remains auditable.
- Bounded signal collection at 256, persisted sources at 32, and checks at 64.
  Verifier exceptions become generic `unknown` evidence and never expose exception
  text.
- Exported the stable verification context, evidence inputs, protocol, signal, and
  `verify_outcome()` API from `deerflow.skill_evolution`.

### Verification

```text
Outcome verifier tests: 15 passed
All evolution + run-event contract tests: 93 passed
Worker and run-event regression tests: 142 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11296 passed, 76 skipped, 39 warnings
```

### Regression Proof

Temporarily changing `LlmJudgmentVerifier` from `authoritative=False` to
`authoritative=True` caused
`test_llm_judgment_alone_cannot_mark_success` to fail: the result changed from
`unknown` to `success`. Restoring supporting-only authority returned the verifier
suite to green.

### Boundary

The worker does not execute outcome verification yet. Phase 2.2 provides the pure,
tested snapshot-to-outcome boundary; Phase 2.3 will consume `OutcomeEvidence` for
admission and complexity eligibility without adding verifier work to the user request
path.

### Next Task

Phase 2 Task 2.3: implement eligibility and complexity detection, including the
configured minimum success confidence and capped-run exclusions.

## 2026-08-16 — Phase 2 Task 2.3

### Implementation

- Added `deerflow.skill_evolution.eligibility` with immutable
  `EligibilityHints`, `EligibilityDecision`, and `EligibilityBranch` contracts.
- `evaluate_evolution_eligibility()` requires:
  - `OutcomeEvidence.status == success`;
  - confidence at or above `min_success_confidence`;
  - at least one enabled complexity signal;
  - no excluded terminal cap reason.
- Complexity detection now covers:
  - tool-call count at the configured threshold;
  - an error tool event followed by a successful tool path;
  - user corrections captured in the snapshot or supplied by the extractor;
  - a trusted non-trivial workflow hint;
  - explicit English/Chinese remember/create-skill requests, with negation
    filtering.
- Skill usage deterministically selects the `skill_used` branch; an empty Skill set
  selects `no_skill`.
- Runs stopped by `safety_capped`, `loop_capped`, `token_capped`, or
  `subagent_limit_capped` are rejected even when the Run status and verifier result
  say success.
- Added `SkillEvolutionEvidenceConfig` with the Phase 2.3 settings only:
  `min_success_confidence`, `tool_call_complexity_threshold`, and four signal gates.
- Updated `config.example.yaml` to config version 34. The real config-upgrade test
  verifies that old files receive the nested evidence defaults without overwriting
  existing database settings.

### Verification

```text
Eligibility tests: 25 passed
Evolution + config focused tests: 95 passed
Worker/config reload regression tests: 190 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11322 passed, 76 skipped, 39 warnings
```

### Regression Proof

Temporarily removing the minimum-confidence rejection admitted a successful,
six-tool-call trace with confidence `0.79` under the default `0.8` threshold.
`test_success_below_configured_confidence_is_rejected` failed immediately. Restoring
the gate returned the eligibility suite to green.

### Boundary

Eligibility remains a pure snapshot/outcome decision API. The worker still only
persists the redacted trace; it does not synchronously verify, admit, extract, group,
or distill evidence.

### Next Task

Phase 3 Task 3.1: add deterministic pre-extraction from admitted traces.

## 2026-08-16 — Phase 3 Task 3.1

### Implementation

- Added `deerflow.skill_evolution.extractor` with versioned
  `DeterministicExtraction`, `ToolSequenceEntry`, and `CandidateSegment`
  contracts.
- Pre-extraction accepts only an eligible decision whose outcome and Skill branch
  match the supplied snapshot.
- Derived deterministic fields without an LLM:
  - new-skill vs. skill-patch event kind;
  - task-input and extraction hashes;
  - environment and verified outcome;
  - eligibility complexity;
  - ordered tool names and unique error categories;
  - all tool steps as content-free parameter/result hashes;
  - observed Skill path/name/hash/source identities;
  - artifact path evidence.
- Candidate input for Phase 3.2 is bounded to 64 segments:
  - up to 32 tool excerpts, preserving errors and the execution tail;
  - up to 8 user corrections;
  - up to 16 artifact references;
  - task input and final answer.
- Every segment carries a deterministic content hash. Large plain text is head/tail
  truncated; probable data-URI/base64/control-byte payloads in either tool arguments
  or results are replaced by an omission marker plus SHA-256.
- `extraction_hash` is recomputed during model validation, so modified payloads with
  stale hashes are rejected.
- Added a recovered package-install fixture with a failed global install, successful
  project-local recovery, verification, correction, and artifacts.

### Verification

```text
Extractor fixture tests: 9 passed
Evolution + config focused tests: 104 passed
Worker/config reload regression tests: 190 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11331 passed, 76 skipped, 39 warnings
```

### Regression Proof

Temporarily removing data-URI detection caused
`test_probable_binary_or_base64_tool_content_is_omitted` to fail because encoded
image content was truncated but no longer marked omitted. Restoring the rule returned
the extractor suite to green.

### Boundary

Pre-extraction is a pure deterministic API. It does not call an LLM, persist an
`EvolutionEvent`, invoke `skill_manage`, or run synchronously from the worker.

### Next Task

Phase 3 Task 3.2: add strict structured LLM extraction over the bounded candidate
segments.

## 2026-08-16 — Phase 3 Task 3.2

### Implementation

- Added `deerflow.skill_evolution.structured_extractor` with:
  - provider-neutral `StructuredExtractionModel` protocol;
  - strict `StructuredExtractionOutput` JSON schema;
  - evidence-linked path, failed-attempt, correction, lesson, and Skill-gap models;
  - bounded async `StructuredEvolutionExtractor`.
- The prompt treats candidate text as untrusted data, embeds the generated JSON
  schema, requires JSON-only output, and includes only the bounded deterministic
  pre-extraction payload.
- Parsing accepts a JSON object, provider text content, or provider text blocks. It
  rejects prose and Markdown-fenced JSON rather than extracting a permissive
  substring.
- Semantic validation requires:
  - every evidence segment ID exists;
  - failed attempts cite failed tool segments;
  - user corrections cite correction segments;
  - new-skill output has no target/gaps;
  - patch output has at least one gap and targets an observed Skill name/hash.
- Added per-item `SemanticEvidenceLink` persistence. Structured events must link task
  goal, every path step, failed attempt, correction, reusable lesson, Skill gap, and
  target Skill to hash-addressed provenance.
- Retry policy:
  - malformed JSON/schema/semantic output is retried;
  - timeout, connection, recognized provider transient classes, and HTTP
    408/409/425/429/5xx transient statuses are retried;
  - arbitrary permanent exceptions are not retried;
  - attempts are bounded to 1-5 and exhaustion returns no event.
- Valid output is converted into a deterministic-ID `EvolutionEvent`. Optional
  `extract_and_persist()` calls `SkillEvolutionStore.upsert_event()` only after all
  validation succeeds.
- Events persist `extractor_model_name`, `extractor_prompt_version`,
  `source_extraction_hash`, `source_snapshot_hash`, and a model+Prompt
  `extractor_version`.
- Added optional `skill_evolution.extraction_model_name`; null selects the primary
  configured model. Updated example and local config to version 35.

### Verification

```text
Structured extraction tests: 11 passed
Structured extraction + model/store/config tests: 55 passed
Evolution/config focused tests: 117 passed
Worker/config reload regression tests: 190 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11344 passed, 76 skipped, 39 warnings
```

### Regression Proof

Temporarily bypassing `_validate_semantics()` caused
`test_unknown_provenance_retries_then_succeeds` to fail: the unknown segment reached
event construction, failed closed as a permanent lookup error, and never consumed the
second corrected model response. Restoring semantic validation returned the suite to
green and proved malformed provenance follows the bounded retry path.

### Boundary

The extractor can be called explicitly and can persist a validated
`EvolutionEvent`, but no worker/coordinator automatically invokes verification,
eligibility, pre-extraction, or structured extraction yet. No Skill is written.

### Next Task

Phase 4 Task 4.1: implement deterministic event fingerprints and cluster candidate
grouping.

## 2026-08-16 — Phase 4 Task 4.1

### Implementation

- Added `deerflow.skill_evolution.grouping` with reproducible, hash-validated
  `EventFingerprint` records derived from:
  - normalized task signature and task-goal tokens;
  - ordered tool sequence and error categories;
  - OS, runtime-major, and shell families;
  - event kind and target Skill identity.
- Added deterministic similarity scoring with explicit component provenance:
  task signature 0.35, tool sequence 0.30, goal 0.20, errors 0.10, and
  environment 0.05. Membership also requires an exact task signature or a strong
  tool-sequence/goal workflow anchor.
- Enforced hard partitions by user, event kind, and patch target Skill. New-Skill
  evidence cannot share a cluster with patch evidence, and patches for different
  Skills cannot share a cluster.
- Added ordered deduplication for repeated event IDs, runs, task-input hashes, and
  identical fingerprints. Reusing an event ID with conflicting payloads is rejected.
- Grouping is stable across input order, bounded to 1,024 input events and 64 members
  per cluster, and creates deterministic `collecting` clusters with per-member
  `GroupingEvidence`.
- Added `skill_evolution.grouping.deterministic_threshold`, defaulting to `0.65`,
  and updated example and local config to version 36.

### Verification

```text
Grouping tests: 12 passed
Grouping + config focused tests: 25 passed
All Skill evolution tests: 117 passed
Worker/config/event-contract regression tests: 98 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11357 passed, 76 skipped, 39 warnings
```

### Regression Proof

Temporarily removing the new-Skill/patch event-kind hard boundary caused
`test_new_skill_and_patch_events_never_group` to fail: otherwise identical events
received a score of `1.0` and were incorrectly treated as compatible. Restoring the
hard boundary returned the grouping suite to green.

### Boundary

This phase only builds deterministic candidate clusters in `collecting` state. It
does not perform embedding retrieval, LLM membership confirmation, `K=3` readiness,
proposal generation, or Skill publication. No worker/coordinator invokes grouping
automatically yet.

### Next Task

Phase 4 Task 4.2: add optional semantic retrieval with a deterministic-only fallback.

## 2026-08-16 — Phase 4 Task 4.2

### Implementation

- Added `deerflow.skill_evolution.semantic_retrieval` with provider-neutral
  `EmbeddingProvider` and `SemanticVectorStore` protocols.
- Added `OllamaEmbeddingProvider` using `/api/tags` and `/api/embed`. When no version
  is configured, it resolves and records the installed model's immutable digest.
- Added a dependency-free `QdrantSemanticVectorStore` over the existing `httpx`
  dependency:
  - physical collections are isolated by embedding model name/version;
  - vector size and cosine distance are validated;
  - deterministic UUID point IDs make repeated upserts idempotent;
  - payloads retain event/user/branch/target, document hash, model, and version;
  - searches enforce user, event-kind, model-version, and patch-target filters.
- Added bounded semantic documents from structured event fields only. Event IDs and
  user IDs are excluded from embedding text; long documents are hash-suffixed and
  capped at 8,192 characters.
- Added hybrid candidate retrieval that:
  - applies deterministic hard partitions before embedding;
  - unions deterministic and semantic candidates;
  - retains deterministic/semantic scores and source provenance;
  - discards Qdrant hits outside the caller-supplied event set;
  - falls back to deterministic results on embedding or Qdrant failure.
- Added optional `semantic_threshold`, `semantic_top_k`, Ollama embedding, and Qdrant
  settings under `skill_evolution.grouping`. Example config remains disabled by
  default; local config uses `nomic-embed-text` and Qdrant on port 6333. Config version
  is now 37.
- Installed local Ollama `nomic-embed-text` (768 dimensions) and verified its digest,
  then passed an opt-in live Ollama -> Qdrant -> filtered-search integration test.

### Verification

```text
Semantic retrieval tests: 9 passed, 1 skipped
Live Ollama/Qdrant integration: 1 passed
All Skill evolution + config tests: 140 passed, 1 skipped
Worker/config/event-contract regression tests: 99 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11367 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily replacing the embedding-error fallback with `raise` caused
`test_semantic_failure_falls_back_to_deterministic_candidates` to fail with the
provider exception. Restoring the fallback returned the suite to green and proved an
embedding outage cannot remove deterministic retrieval.

### Boundary

Semantic retrieval only broadens the candidate set. It does not confirm cluster
membership, alter `EvolutionCluster.status`, count `K=3`, generate proposals, or
publish Skills. No worker/coordinator invokes it automatically yet.

### Next Task

Phase 4 Task 4.3: add structured LLM cluster confirmation and distinct-run readiness.

## 2026-08-16 — Phase 4 Task 4.3

### Implementation

- Added `deerflow.skill_evolution.cluster_confirmation` with:
  - provider-neutral `ClusterConfirmationModel`;
  - strict `ClusterConfirmationOutput`;
  - bounded `StructuredClusterConfirmer`;
  - `confirm_cluster_readiness()` aggregation.
- The prompt compares one prototype/candidate pair, treats event content as untrusted
  data, embeds the Pydantic JSON Schema, and accepts JSON-only output.
- Confirmation distinguishes:
  - the same workflow;
  - a compatible environment-specific branch with a required condition;
  - a different workflow that is excluded from the cluster.
- Malformed output and recognized transient model errors retry within a 1-5 attempt
  budget. Exhaustion returns an unconfirmed candidate and keeps the cluster
  `collecting`.
- Added member-level `ClusterMemberEvidence` with run ID, environment relationship,
  optional condition, contradiction marker, and unique deterministic/semantic/LLM
  evidence.
- LLM agreement alone cannot establish membership. Every candidate must already have
  deterministic or semantic retrieval provenance.
- Readiness requires all of:
  - at least `min_cluster_events` confirmed members, default 3;
  - at least `min_distinct_runs` distinct run IDs, default 3;
  - no contradictory accepted evidence;
  - no unconfirmed candidates in the bounded confirmation set.
- Different-workflow candidates are rejected without blocking readiness except through
  the remaining evidence count. Conditional environment branches remain accepted.
- Confirmed snapshots retain model/Prompt metadata and can be persisted through the
  existing immutable `put_cluster()` boundary. Ready snapshots are returned by
  `list_ready_clusters(min_distinct_runs=3)`.
- Added `min_cluster_events`, `min_distinct_runs`, `max_events_per_cluster`,
  `llm_confirmation`, and `confirmation_model_name`; config version is now 38.

### Verification

```text
Cluster confirmation tests: 11 passed
Confirmation/store/config focused tests: 45 passed
All Skill evolution + config tests: 152 passed, 1 skipped
Worker/config/event-contract regression tests: 100 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11379 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily counting accepted events instead of unique `run_id` values caused
`test_single_repeated_run_cannot_satisfy_k_three` to fail: three events from one run
were incorrectly marked ready. Restoring set-based run counting returned the suite to
green.

### Boundary

Phase 4 now supports explicit snapshot -> event -> deterministic/semantic candidate
retrieval -> LLM confirmation -> K=3 readiness. No worker/coordinator invokes this
pipeline automatically yet, and readiness only authorizes Phase 5 proposal
distillation; it never writes or publishes a Skill.

### Next Task

Phase 5 Task 5.1: distill ready new-Skill clusters into evidence-linked staged
proposals.

## 2026-08-16 — Phase 5 Task 5.1

### Implementation

- Added `deerflow.skill_evolution.distiller` with:
  - provider-neutral `DistillationModel`;
  - strict `NewSkillDistillationOutput`;
  - bounded `NewSkillDistiller`;
  - deterministic Proposal rendering and persistence.
- The distillation prompt includes only confirmed structured event fields and at most
  eight 500-character provenance excerpts per event. The complete serialized input is
  capped at 96,000 characters and treated as untrusted data.
- Structured output separates:
  - repeated common workflow steps;
  - environment-specific conditional rules;
  - repeated verification steps;
  - unresolved conflicts;
  - unsupported one-off observations;
  - optional supporting files.
- Mandatory common/verification guidance, conflicts, and supporting files must cite at
  least two distinct runs. Unknown IDs and single-run mandatory evidence retry and
  eventually fail closed.
- Deterministic rendering creates valid `SKILL.md` frontmatter plus Overview,
  Workflow, optional Environment-Specific Guidance, and Verification sections.
  Unsupported observations and conflict text are not rendered into instructions.
- Supporting files use normalized relative paths and are emitted only after repeated
  evidence validation. Executable support files force manual review.
- Added `ProposalEvidenceMapping` for every rendered semantic section and support
  file. Distilled Proposals also retain source Cluster hash, model/Prompt versions,
  deterministic Proposal ID, manual-review reasons, and bounded deduplicated risks.
- Conflicts stage a Proposal with `requires_manual_review=true` and
  `unresolved_conflicts`; they are never silently synthesized into a mandatory rule.
- `distill_and_persist()` only calls `SkillEvolutionStore.put_proposal()` after all
  validation and always creates `operation=create`, `status=staged`. It never calls
  `skill_manage` or writes the Skill library.
- Added `skill_evolution.distillation_model_name`; config version is now 39.

### Verification

```text
New-Skill distillation tests: 10 passed
Distillation/model/store/config focused tests: 58 passed
All Skill evolution + config tests: 163 passed, 1 skipped
Worker/config/event-contract regression tests: 101 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11390 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily removing the repeated distinct-run evidence check caused
`test_unsupported_mandatory_step_is_retried_then_rejected` to fail: a single event's
"delete the global package cache" observation was incorrectly rendered as mandatory
Skill guidance. Restoring the check returned the suite to green.

### Boundary

Ready new-Skill clusters can now explicitly become evidence-linked staged Proposals.
No background coordinator invokes the pipeline automatically, and no Proposal is
evaluated, approved, written, or published in this phase.

### Next Task

Phase 5 Task 5.2: distill ready patch clusters into targeted, stale-base-safe Skill
patch Proposals.

## 2026-08-16 — Phase 5 Task 5.2

### Implementation

- Added `deerflow.skill_evolution.patch_distiller` with:
  - `SkillVersionSource` protocol;
  - off-loop `SkillStorageVersionSource`;
  - strict `PatchSkillDistillationOutput`;
  - stale-aware `PatchSkillDistiller`;
  - deterministic `apply_structured_patch()`.
- Exact source content is resolved by SHA-256 from the current `SKILL.md` or existing
  JSONL history `prev_content` / `new_content`. Missing source versions fail before
  any model call.
- A patch cluster must contain one target Skill name and one source content hash.
  Mixed source versions are rejected instead of being silently merged.
- The model receives both:
  - `observed_source_content`, the exact version source trajectories used;
  - `patch_base_content`, the current version that operations must target.
- Output is restricted to exact `SKILL.md` `find` / `replace` operations with
  `expected_count=1`. Full-file replacement, missing/ambiguous targets, changed Skill
  names, invalid frontmatter, and empty bodies fail closed.
- General operations require evidence from at least two distinct runs.
  Environment-specific operations may use one event only with an explicit condition.
- Operations are applied sequentially in memory, preserving all unaffected content.
  The resulting full `SKILL.md` is retained only inside the staged Proposal.
- The current content hash is read again after model output. Drift discards the output
  and restarts against the latest content, bounded to three rebase attempts.
- Patch Proposals retain:
  - source trajectory Skill hash;
  - current `base_skill_hash` for future publication CAS;
  - structured `SkillPatchOperation` records;
  - rendered candidate file;
  - per-operation event mappings;
  - source Cluster and distiller metadata.
- Conflicts force manual review and are excluded from the rendered Skill.
  `distill_and_persist()` only stores a staged Proposal and never writes a Skill.

### Verification

```text
Patch distillation tests: 10 passed
Patch/new distillation/model/store focused tests: 52 passed
All Skill evolution + config tests: 173 passed, 1 skipped
Worker/config/event-contract regression tests: 101 passed
Ruff check: passed
Ruff format --check: passed
Full backend offline suite: 11400 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily removing the post-model current-hash check caused
`test_stale_current_version_restarts_distillation` to fail: the model ran once and
returned a Proposal based on a version that changed during generation. Restoring the
check forced a second model call against the latest base.

### Boundary

Both new-Skill and existing-Skill ready clusters can now produce evidence-linked
staged Proposals. No Proposal is evaluated, approved, applied, or published, and the
runtime worker still does not invoke the pipeline automatically.

### Next Task

Phase 6 Task 6.1: define replayable task specifications and isolated evaluation
workspaces.

## 2026-08-16 — Phase 6 Task 6.1

### Implementation

- Added `deerflow.skill_evolution.evaluator` with versioned replay-task and
  replay-fixture contracts.
- `ReplayFixtureSnapshot` stores bounded binary files as validated base64 with
  per-file hashes, a deterministic aggregate hash, completeness metadata, and omitted
  paths.
- Fixture capture runs off the event loop and rejects absolute, non-normalized,
  traversal, missing, non-file, and symlink-backed paths.
- `ReplayTaskSpec` binds one successful `EvolutionTraceSnapshot` to its
  `EvolutionEvent` and records:
  - task input and source identities;
  - fixture snapshot;
  - OS, shell, runtime, command, secret, and network requirements;
  - deterministic command and artifact verifiers;
  - timeout and side-effect policy;
  - automatic/manual-review classification and reasons.
- Automatic replay requires a complete source and fixture, task input, at least one
  verifier, no external credentials, and no disallowed network requirement.
  Incomplete or environment-dependent tasks remain explicit manual-review inputs.
- `isolated_replay_workspace()` creates disposable `workspace/`, `outputs/`, and
  `skills/` roots. Fixtures are copied only into the temporary workspace; a staged
  candidate Proposal is copied into a read-only temporary Skill package.
- Workspace creation validates Proposal ownership and state. Materialization failure,
  normal exit, and exceptional exit all remove the temporary tree off the event loop.
- `ReplayWorkspace.audit_side_effects()` compares the materialized baseline with the
  final tree and reports changed paths, changed-file count, written bytes, and policy
  violations. Production Skill writes are invalid in every policy.

### Verification

```text
Replay specification tests: 14 passed
All Skill evolution + config tests: 187 passed, 1 skipped
Worker/config/event-contract regression tests: 101 passed
Full backend offline suite: 11414 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily removing the context manager's `finally` cleanup caused
`test_workspace_cleans_up_after_exception` to fail and leave the replay root present.
Restoring unconditional cleanup returned the suite to green.

### Boundary

Phase 6.1 creates trustworthy replay inputs and disposable filesystem isolation. It
does not yet run the agent, execute verifiers, compare a candidate to a baseline,
calculate quality, approve a Proposal, or publish a Skill. The runtime worker still
only records the redacted trace and does not invoke this pipeline automatically.

### Next Task

Phase 6 Task 6.2: execute source, held-out, and no-Skill baseline replays for staged
new-Skill Proposals and persist their raw evaluation metrics.

## 2026-08-17 — Phase 6 Task 6.2

### Implementation

- Extended `ReplayTaskSpec` with the source event's `task_family`; deterministic task
  IDs now include that family.
- Added provider-neutral `ReplayRuntime` with separate async agent and verifier-command
  methods. A runtime receives only the disposable replay paths, task contract,
  condition, and explicit candidate path; `evaluator.py` never starts a host shell.
- Added redacted runtime result contracts:
  - `ReplayAgentExecution` for tool calls, input/output tokens, and stable error codes;
  - `ReplayCommandExecution` for exit code and stable error codes.
- Added `NewSkillProposalEvaluator` validation:
  - only create Proposals in staged/validating state;
  - source tasks exactly cover all supporting event IDs;
  - no duplicate or overlapping source/held-out tasks;
  - every task belongs to the Proposal user;
  - source and held-out tasks share one task family;
  - manual-only tasks stop automatic evaluation before any runtime call.
- Every source and held-out task runs in two independent workspaces:
  - `no_skill`, without candidate files;
  - `candidate_skill`, with the candidate explicitly materialized and exposed to the
    runtime for forced activation.
- Agent execution uses the task timeout; command verifiers use their own timeout.
  Artifact verifiers check regular-file existence, absence, SHA-256, and bounded
  streaming text containment without persisting content.
- `TaskEvaluationResult` now stores raw success, tool calls, tokens, measured latency,
  stable errors, and deduplicated artifact path/kind/hash/size metadata.
- Changed files under workspace/outputs become artifact metadata. Skill changes,
  disallowed roots, file-count overflow, and written-byte overflow fail the run.
- Replay symlinks never satisfy artifact checks. Cleanup skips chmod on symlinks so it
  cannot mutate an external target before removing the temporary tree.
- Source and held-out pass rates use `skill_evolution.quality` thresholds. Missing
  held-out tasks and risky Proposals remain manual review; side-effect violations and
  threshold failures reject.
- `evaluate_and_persist()`:
  - derives one deterministic evaluation ID from Proposal/task/config identity;
  - reuses an existing evaluation without rerunning the model;
  - moves a staged Proposal to validating before execution;
  - persists the raw `SkillEvaluation`;
  - moves failed validating Proposals to rejected;
  - leaves passed/manual-review Proposals validating for later approval.
- Added `SkillEvolutionQualityConfig` with source `1.0` and held-out `0.8` defaults;
  config version is now 40.
- Aggregate multi-dimensional quality is intentionally not computed in this task.
  Stored evaluations use `quality_score=0.0` plus
  `quality_score=deferred_phase_6_4` safety metadata.

### Verification

```text
New-Skill evaluation tests: 18 passed
Evaluation/replay/model/store/config focused tests: 80 passed
All Skill evolution + config tests: 206 passed, 1 skipped
Worker/config/event-contract regression tests: 102 passed
Full backend offline suite: 11433 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily removing the held-out success-rate condition caused
`test_held_out_failure_rejects_candidate_after_source_passes` to fail: a candidate
that passed all three source tasks but failed its held-out task was incorrectly marked
`approve`. Restoring the condition returned the test to green.

The first full-suite run also exposed cleanup following an output symlink during
`chmod`. Cleanup now skips symlinks; the focused regression proves the external
target's content and `0600` mode remain unchanged, and the repeated full suite exits
cleanly.

### Boundary

Phase 6.2 can explicitly evaluate and persist a new-Skill Proposal when the caller
supplies a conforming isolated `ReplayRuntime`. The normal worker does not yet build
task suites or provide the production runtime adapter. Existing-Skill regression
evaluation, aggregate quality scoring, approval, publication, and rollback remain
unimplemented.

### Next Task

Phase 6 Task 6.3: evaluate existing-Skill patches against paired old/new Skill runs
and historical regression tasks.
