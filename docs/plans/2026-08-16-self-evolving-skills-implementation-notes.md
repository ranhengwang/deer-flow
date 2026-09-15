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
- Provider compatibility keeps the deterministic branch authoritative before
  semantic validation: new-Skill output discards Patch-only target/gap fields,
  and optional failed-attempt or correction claims lacking their required typed
  evidence are omitted. Required path/goal/lesson evidence and Patch target/gap
  validation remain fail-closed.
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

## 2026-08-17 — Phase 6 Task 6.3

### Implementation

- Added `ReplaySkillFile` and `ReplaySkillPackage`:
  - preserve exact Skill file content without global whitespace trimming;
  - require normalized unique paths and `SKILL.md`;
  - validate exact `SKILL.md` SHA-256;
  - retain executable metadata;
  - explicitly record completeness and omitted paths;
  - enforce Skill-name and user isolation.
- Added `build_replay_skill_package()` for deterministic base packages and
  `build_patch_candidate_skill_package()` for in-memory Proposal overlay. Candidate
  files replace matching base paths while unchanged support files remain available.
- Extended replay conditions with `base_skill`. `ReplayAgentRequest` now exposes one
  explicit `active_skill_path` for base/candidate activation while preserving the
  candidate-only compatibility property.
- Extracted the common agent/command/verifier/artifact/side-effect path into one shared
  `_run_replay_task()` used by both new-Skill and Patch evaluators.
- Added `PatchSkillProposalEvaluator` validation:
  - only Patch Proposals in staged/validating state;
  - exact base package user/name/hash match;
  - source tasks exactly cover supporting events and share one family;
  - source and historical regression tasks are unique, independent, and same-user;
  - manual-only tasks block execution before runtime calls.
- Each source and historical task runs twice in separate disposable workspaces:
  - `base_skill`, containing the exact old package;
  - `candidate_skill`, containing the in-memory merged candidate package.
- Regression rate counts only paired historical cases where base succeeds and
  candidate fails, divided by historical base successes. Base failures are excluded
  instead of being credited as regressions.
- Decision policy:
  - reject source pass-rate failure, side-effect violations, or regression above the
    configured maximum;
  - require manual review for missing historical tasks, incomplete base packages, no
    successful base regression sample, Proposal review flags, or executable support
    changes;
  - otherwise recommend approval while leaving publication untouched.
- `evaluate_and_persist()` uses a deterministic Patch evaluation ID, preserves Store
  idempotency, transitions staged to validating, persists raw paired results, and
  transitions rejected evaluations to rejected Proposal state.
- Added `skill_evolution.quality.max_regression_rate` with default `0.0`; config
  version is now 41.
- `SkillEvaluation` approval accepts held-out results for new Skills or regression
  results for patches. Aggregate quality remains `0.0` with deferred metadata until
  Task 6.4.
- Tightened Phase 6.2 ordering: a risky/manual-review Proposal that also fails source
  or held-out requirements is rejected rather than merely sent to review.

### Verification

```text
Patch evaluation tests: 16 passed
Patch/new evaluation/replay/model/store/config focused tests: 99 passed
All Skill evolution + config tests: 224 passed, 1 skipped
Worker/config/event-contract regression tests: 103 passed
Full backend offline suite: 11451 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily removing the configured regression-rate condition caused
`test_regression_above_threshold_rejects_patch` to fail: a Patch that preserved all
source tasks but broke one of two previously passing historical tasks was incorrectly
marked `approve`. Restoring the condition returned the test to green.

### Boundary

Phase 6.3 explicitly evaluates existing-Skill patches when supplied a complete
hash-bound base package, historical tasks, and an isolated `ReplayRuntime`. The normal
worker still does not assemble suites or invoke evaluation automatically. Aggregate
quality scoring, approval, publication, version history, and rollback are not
performed here.

### Next Task

Phase 6 Task 6.4: compute versioned aggregate Skill quality from persisted raw
candidate, baseline, held-out, and regression metrics.

## 2026-08-17 — Phase 6 Task 6.4

### Implementation

- Added `deerflow.skill_evolution.quality` with deterministic formula version
  `skill-quality-v1`.
- Added persisted quality contracts:
  - `QualityDesignation`: `insufficient_evidence`, `evaluated`, `high_quality`;
  - `SkillQualityDimension`: raw derived value, normalized score, fixed formula
    weight, sample count, availability;
  - `SkillQualityReport`: formula version, all dimensions, aggregate score, threshold,
    designation, sample sufficiency, sample counts, and blockers.
- `SkillEvaluation` now stores the report separately from raw task results and keeps
  `quality_score` synchronized with the report aggregate for SQL indexing.
- `TaskEvaluationResult` now stores a SHA-256 environment fingerprint derived from the
  replay environment contract. No environment secret or raw host path is added.
- Formula v1 dimensions and fixed weights:

```text
success-rate lift          0.25
held-out generalization    0.20
tool-call reduction        0.10
token reduction            0.10
latency reduction          0.10
environment robustness     0.10
regression resistance      0.10
safety                     0.05
```

- Success lift and efficiency reductions are clamped to `[-1, 1]` then mapped with
  `(value + 1) / 2`. Equal performance is neutral (`0.5`), improvement is above
  neutral, and degradation is below neutral.
- Held-out score is candidate held-out success rate. Environment robustness is the
  worst candidate success rate across observed environment fingerprints. Regression
  score is `1 - old_success_new_failure_rate`. Safety score is `1 - risk`, where
  hard side-effect violations are `1.0` risk and manual-review states are `0.5`.
- Efficiency uses only pairs where baseline and candidate both succeeded, preventing
  failed low-cost attempts from appearing efficient.
- Unavailable dimensions carry no score and are removed from the aggregate weight
  denominator. Their missing evidence remains visible through sample blockers.
- Default high-quality requirements:
  - approved evaluation;
  - aggregate score at least `0.75`;
  - no safety/review blockers;
  - at least five candidate tasks;
  - at least two held-out tasks for a new Skill, or two old-success historical
    regression pairs for a Patch;
  - at least two distinct environments.
- Added `compute_skill_quality()` for report generation and `apply_skill_quality()` for
  immutable evaluation enrichment. Raw results are copied unchanged.
- Both Proposal evaluators now apply quality before persistence. Their deterministic
  evaluator IDs include `skill-quality-v1`, and evaluator versions are now v2.
- Model validation rejects high-quality reports with insufficient samples, blockers,
  aggregate below threshold, or a non-approved parent evaluation.
- Added five quality configuration fields and bumped config version from 41 to 42.

### Verification

```text
Skill quality tests: 13 passed
Quality/evaluator/model/store/config focused tests: 99 passed
All Skill evolution + config tests: 238 passed, 1 skipped
Worker/config/event-contract regression tests: 104 passed
Full backend offline suite: 11465 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily forcing `sample_sufficient=True` caused
`test_three_source_runs_alone_can_never_be_high_quality` to fail: the report no longer
classified the three-source-only evaluation as insufficient. Restoring the blocker-
derived sample flag returned the test to green.

### Boundary

Phase 6.4 computes quality for one immutable evaluation. It does not aggregate quality
across repeated evaluation campaigns, approve a Proposal, publish a Skill, or trigger
RL/internalization. The normal worker still does not invoke the evolution pipeline.

### Next Task

Phase 7 Task 7.1: extract the existing Skill mutation/security path into a reusable
service and add compare-and-swap publication inputs.

## 2026-08-17 — Phase 7 Task 7.1

### Implementation

- Added `deerflow.skills.mutation` with:
  - immutable `SkillMutationRequest` and `SkillMutationResult` contracts;
  - `SkillMutationService` as the shared custom-Skill mutation boundary;
  - explicit `SkillMutationConflict` for stale compare-and-swap requests;
  - reusable default native SkillScan and LLM moderation adapters.
- The service supports the existing `create`, `patch`, `edit`, `delete`,
  `write_file`, and `remove_file` operations while preserving:
  - Skill-name, `SKILL.md`, and support-path validation;
  - per-user/per-Skill async serialization;
  - staged full-package native SkillScan before LLM moderation;
  - stricter executable support-file moderation;
  - mutation history and prompt-cache refresh;
  - storage-owned projection rebuilding.
- `SkillMutationRequest.expected_base_hash` is the current `SKILL.md` SHA-256 CAS
  token. Existing-Skill mutations reject a stale token before scanners run, then pass
  the same token to storage for an authoritative second check immediately before the
  filesystem mutation.
- `SkillStorage`, `LocalSkillStorage`, and `UserScopedSkillStorage` now accept CAS
  inputs for writes, support-file removals, and package deletion. The final hash check,
  atomic file replacement or removal, and projection mutation execute under the same
  cross-process projection lock, closing the security-scan TOCTOU window.
- Create does not accept a base hash. It passes `require_absent=True` into the final
  locked storage write, so another process cannot create the same Skill during
  scanning and then be overwritten.
- Evolution-authored requests require paired `proposal_id` and `evaluation_id`
  values. History records now retain:

```text
action
author
thread_id
proposal_id
evaluation_id
expected_base_hash
previous_skill_hash
resulting_skill_hash
file_path
prev_content
new_content
scanner
```

- `tools/skill_manage_tool.py` is now a thin runtime adapter. It resolves user/thread
  identity, constructs an agent-authored request, and returns the service message.
  Its public LangChain tool signature is unchanged. The existing scanner wrapper
  functions remain injectable so tracing behavior, error text, and regression-test
  monkeypatch points stay compatible.
- Blocking storage and filesystem work remains off the event loop through
  `asyncio.to_thread`. No config version or database migration was required.

### Verification

```text
Mutation service tests: 12 passed
Mutation service + existing skill_manage tests: 22 passed
Skill security/router/storage/projection tests: 277 passed
Full backend offline suite: 11477 passed, 77 skipped, 39 warnings
Ruff check and format check: passed before documentation close-out
```

### Regression Proof

Temporarily removing `expected_base_hash` from the final storage write caused
`test_atomic_cas_catches_drift_during_security_scan` to fail: a competing update made
while moderation was running was silently overwritten. Restoring the storage-layer
CAS rejected the stale candidate and preserved the competing content. A separate
`test_create_cas_does_not_overwrite_concurrent_create` regression proves the locked
absence check for create.

### Boundary

Phase 7.1 supplies a reusable, provenance-aware mutation primitive. It does not decide
whether a Proposal is approved and nothing in the Evolution pipeline invokes it
automatically. Versioned package snapshots and rollback are also still absent.

### Next Task

Phase 7 Task 7.2: add an explicit Proposal approval policy, defaulting to manual
approval and keeping executable changes out of any initial auto-publication path.

## 2026-08-17 — Phase 7 Task 7.2

### Implementation

- Added `deerflow.skill_evolution.approval` with deterministic policy version
  `skill-approval-v1`.
- Added immutable approval contracts:
  - `ApprovalOutcome`: `manual_review`, `approved`, `rejected`, or `expired`;
  - `ProposalApprovalAssessment`: policy/evaluation identity, reason codes and detail,
    automatic-publication eligibility, deadline, and assessment time;
  - `ManualApprovalRequest`: reviewer identity, explicit approve/reject decision,
    reason, and decision time;
  - `ProposalApprovalResult`: resulting Proposal plus whether this caller won the
    state change.
- Default policy is manual. A passing evaluation produces a manual-review assessment
  and leaves the Proposal in `validating`; no persistence write is needed until a
  reviewer decides.
- Opt-in automatic approval requires both:
  - `publication.mode=eligible_auto`;
  - `allow_non_executable_auto_publish=true`.
- Automatic eligibility then additionally requires:
  - a Patch rather than a new-Skill Proposal;
  - no executable proposed file;
  - no Proposal risks or distiller manual-review flag;
  - an `approve` evaluation with no safety/manual-review blocker;
  - at least one candidate held-out result, with every held-out candidate succeeding.
- Automatic policy success transitions only `validating -> approved`. No branch in
  this module transitions to `published` or calls `SkillMutationService`.
- `allow_executable_auto_publish` is `Literal[False]` and
  `require_held_out_evaluation` is `Literal[True]`, so configuration cannot weaken
  either first-version invariant.
- New distilled Proposals persist an explicit 180-day deadline. Legacy Proposal
  payloads without `expires_at` use `publication.proposal_ttl_days` (default 180).
  Expiration is checked before automatic or manual approval; an approved Proposal
  that has not been published also transitions to expired after its deadline.
- Added `ProposalStatusSource` and append-only `ProposalStatusTransition` records.
  Store status changes now retain:

```text
from_status
to_status
source
reason_code
reason
occurred_at
evaluation_id
actor_id
policy_version
```

- In-memory and SQL Store implementations validate transition metadata, append it to
  the Proposal JSON payload, and update status under the existing CAS. This is an
  additive JSON contract change and requires no Alembic migration.
- Legacy Proposal payloads in any status may omit history, allowing pre-7.2 terminal
  and validating records to load after upgrade. The first post-upgrade transition
  starts the audit chain; every recorded transition after that point must be
  continuous.
- Both evaluators now add `evaluation_started`, and rejected evaluations add an
  explicit `evaluation_rejected` reason with bounded safety-result details.
  Evaluator input equivalence excludes mutable status/history while retaining every
  candidate-content and provenance field.
- Identical concurrent policy/reviewer requests use Store CAS; the loser reads and
  accepts only an exactly matching winning transition. A different reviewer, reason,
  or outcome remains a conflict.
- Human reviewers may approve review-only or inspected executable candidates, but
  cannot override an evaluation whose decision is `reject`.
- Added `skill_evolution.publication` to configuration and bumped example/local
  config version from 42 to 43.

### Verification

```text
Approval policy tests: 11 passed
Approval policy + memory/SQL Store contract tests: 32 passed
Skill evolution + config tests: 252 passed, 1 skipped
Full backend offline suite: 11491 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily removing the executable-file check from automatic eligibility caused
`test_executable_change_can_never_be_auto_approved` to fail: the policy changed an
executable Proposal from `validating` to `approved`. Restoring the gate returned the
test to green. The config model separately rejects
`allow_executable_auto_publish=true`.

### Boundary

Phase 7.2 records approval decisions and expiration only. It has no Gateway API or
background coordinator integration and performs no Skill write. In particular,
`approved` does not imply `published`; Phase 7.3 must snapshot the complete package,
publish through `SkillMutationService`, and implement rollback.

### Next Task

Phase 7 Task 7.3: add complete package version snapshots, safe publication, and
rollback with security revalidation for executable content.

## 2026-08-17 — Phase 7 Task 7.3

### Implementation

- Added binary-safe package primitives in `deerflow.skills.package`:
  - normalized package-relative paths;
  - exact bytes and executable-bit records;
  - stable package SHA-256 over path, size, content hash, and executable state;
  - bounded reads (256 files, 8 MiB/file, 32 MiB/package);
  - symlink, traversal, unsupported-entry, and read-drift rejection;
  - deterministic materialization with restored executable permissions.
- Added persisted evolution contracts:
  - `SkillPackageSnapshotFile`;
  - `SkillPackageSnapshot`;
  - `PublicationStatus`: `preparing`, `published`, `rolled_back`;
  - `SkillPublication`, containing base, candidate, actual published, and rollback
    snapshots plus base/published Skill hashes.
- Snapshot file bytes are base64-encoded with byte-size and SHA-256 validation.
  Snapshots validate sorted unique paths, exact `SKILL.md` identity, complete package
  hash, owner, and Skill name.
- Added `publishing` and `rolled_back` Proposal states. Publication status transitions
  include `publication_id`; rollback transitions additionally require reviewer
  identity.
- Added `SkillEvolutionStore` publication methods:
  - idempotent `put_publication()`;
  - user-scoped `get_publication()`;
  - status-CAS `transition_publication()`.
- Added memory and SQL implementations. The SQL row indexes user/status, user/Skill,
  and user/update time, and enforces one publication per user/Proposal.
- Added `skill_evolution_publications` through Alembic revision
  `0013_skill_publications`. Updated every fixed-head bootstrap test from 0012 to
  0013 and added direct 0012-to-0013 plus full 0011-chain migration tests.
- New `SkillPublicationService` publication sequence:
  1. Load same-user Proposal/evaluation and require a non-expired approved Proposal.
  2. Validate every proposed support path through `SkillStorage`.
  3. Capture the complete base package, or an explicit absent snapshot for create.
  4. Overlay Proposal files in memory to produce the complete candidate snapshot.
  5. Run full SkillScan and bounded moderation before persisting candidate bytes.
  6. Persist a `preparing` record with base and candidate snapshots.
  7. CAS `approved -> publishing`.
  8. Repeat security validation and atomically replace the package through
     `SkillMutationService`.
  9. Capture and hash the actual published package.
  10. CAS the publication record to `published` and Proposal to `published`.
- `SkillMutationService` now exposes a complete-package path in addition to existing
  per-file operations:
  - `validate_package()` performs read-only preflight;
  - `replace_package()` owns final validation, package/base CAS, storage commit,
    history, projection, and cache refresh;
  - full SkillScan sees every binary/text file;
  - LLM moderation is limited to Proposal-selected paths while always including
    `SKILL.md` and every executable file.
- `LocalSkillStorage.replace_custom_skill_package()` stages beside the target so
  renames stay on one filesystem. Under the projection lock it validates both current
  `SKILL.md` and complete package hashes, moves the old directory to a hidden backup,
  installs or deletes the package, repairs sandbox-readable modes, and writes history.
  Candidate rename, permission repair, and history failure all restore the backup.
  `UserScopedSkillStorage` inherits the same operation on its per-user root.
- Publication history records Proposal/evaluation/publication IDs, actor, expected and
  resulting Skill/package hashes, and scanner results under
  `action=evolution_publish`.
- Rollback sequence:
  - require a `published` version and exact live published package hash;
  - reconstruct no content from Proposal/model output;
  - use only `base_snapshot.package_files()`;
  - rerun full SkillScan, `SKILL.md` moderation, and all executable moderation;
  - atomically restore the full base package, or delete a create publication;
  - verify the restored package hash;
  - append `action=evolution_rollback`;
  - CAS publication and Proposal to `rolled_back`.
- Retry recovery handles:
  - `preparing` record with unchanged base;
  - package already equal to candidate after a process exit;
  - publication row committed while Proposal remains `publishing`;
  - rollback package restored while publication row remains `published`;
  - publication row rolled back while Proposal remains `published`.
- Package/preparing records are never created for validating, expired, cross-user,
  unsafe-path, or preflight-blocked inputs.
- Publication service remains an explicit submodule import so persistence/config
  initialization does not pull in mutation/agent prompt code and recreate a harness
  circular import.
- No `config.yaml` field changed, so config version remains 43.

### Verification

```text
Publication tests: 11 passed
Publication + mutation + Store + migration focused tests: 56 passed
Skill evolution + mutation + config tests: 292 passed, 1 skipped
Bootstrap and migration regression tests: 36 passed
Full backend offline suite: 11509 passed, 77 skipped, 39 warnings
```

### Regression Proof

Temporarily bypassing `_package_candidate_scanner` in the complete-package mutation
path caused `test_rollback_security_failure_preserves_published_package` to fail:
rollback restored the persisted base package even though the configured scanner
blocked it. Restoring SkillScan made the test pass and left blocked rollback attempts
in the published state.

Two additional atomicity regressions force history append and permission-repair
failures after directory exchange. Both prove that the original package hash and
bytes are restored and no candidate file remains.

### Boundary

Phase 7.3 provides explicit, durable publication and rollback services. It does not
add Gateway endpoints or invoke them from the run worker. The normal worker still
only writes `skill_evolution.trace`; Phase 8 must coordinate the pipeline, while
Phase 9 owns authenticated review/publication/rollback APIs.

### Next Task

Phase 8 Task 8.1: add a bounded, restart-safe asynchronous coordinator that invokes
the already explicit verification-through-publication stages outside the user request
event loop.

## 2026-08-17 - Phase 8.1 Durable Evolution Coordinator

### Scope

Phase 8.1 turns the post-run trace hook into recoverable background work without
moving LLM, Qdrant, or Skill distillation onto the user run's completion path.

### Durable Job Contract

- Added `EvolutionJob` schema `deerflow.skill-evolution.job.v1`.
- Stable identity is SHA-256 over:
  - `user_id`;
  - `run_id`;
  - redacted `snapshot_hash`;
  - `skill-evolution-pipeline-v1`.
- State machine:

```text
pending -> running -> completed
             |
             +-> retry -> running
             |
             +-> dead
```

- Claims carry:
  - process owner;
  - unique per-claim token;
  - renewable expiry;
  - incremented attempt count;
  - monotonic row revision.
- Renewal/completion/retry require the active token and revision CAS. Completion also
  rejects an already expired lease. A stale worker cannot commit after another worker
  reclaims the row.
- Error persistence stores only a bounded exception class name, not provider text or
  trajectory content.

### Store and Migration

- Extended both `InMemorySkillEvolutionStore` and `SqlSkillEvolutionStore` with:
  - idempotent enqueue;
  - owner-scoped get;
  - due/expired claim;
  - lease renewal;
  - completion;
  - retry/dead transition.
- Added `skill_evolution_jobs` with indexed due and lease scans plus a unique
  `(user_id, idempotency_key)` constraint.
- Added Alembic `0014_skill_evolution_jobs`, now the repository head.
- Verified fresh and sequential upgrades from 0011, 0012, and 0013.

### Coordinator and Worker

- `EvolutionCoordinator.enqueue_snapshot()` commits the Store row before publishing a
  local wake-up.
- Its bounded queue stores `None` wake-up hints, never canonical jobs. `QueueFull` is
  safe because polling SQL is the recovery path.
- The dispatcher claims only available local capacity. The default per-process limit
  is two jobs.
- `EvolutionJobWorker` renews the lease every third of its duration while processing.
- Processing errors use deterministic exponential backoff:

```text
min(retry_max_delay, retry_base_delay * 2 ** (attempt - 1))
```

- Shutdown stops new claims, waits up to
  `skill_evolution.coordinator.shutdown_timeout_seconds`, then cancels remaining local
  processors. Cancellation releases the row to immediate retry when possible; if
  release persistence is interrupted, the running lease remains recoverable.

### Background Pipeline

`EvolutionPipelineProcessor` performs:

```text
load same-user trace and verify full identity/hash
-> deterministic outcome verification
-> eligibility/complexity gate
-> deterministic pre-extraction
-> strict structured LLM extraction + event upsert
-> deterministic grouping
-> optional Ollama/Qdrant retrieval
-> strict cluster confirmation and K=3 readiness
-> ready Cluster persistence
-> new-Skill or Patch Proposal distillation
-> staged Proposal
```

Persisted events are reused after restart rather than invoking extraction again.
Ready-Cluster continuation uses 64 bounded process-local lock stripes. A ready Cluster
found after restart is treated as a potentially interrupted stage and distillation is
resumed. Before invoking the model, the worker checks for an existing same-user
Proposal by Cluster and reuses it, closing the Proposal-written/job-not-completed
restart window. A peer that loses Cluster CAS does not repeat the winner's
continuation.

The processor intentionally stops at `staged`. The evaluator contracts exist, but
production ReplayTaskSpec suite assembly and an isolated `ReplayRuntime` adapter do
not. The coordinator therefore does not create a fake evaluation and never invokes
approval, publication, rollback, or Skill mutation.

### Run and Gateway Wiring

- The run worker persists delivery receipt and terminal status first.
- It then persists the redacted trace.
- Only after that succeeds does it call the injected enqueue callback.
- Trace and enqueue have separate fail-open exception boundaries.
- `langgraph_runtime()` creates the SQL or memory evolution Store beside the other
  application repositories.
- Gateway lifespan creates the processor/coordinator and starts polling when the
  startup config has evolution enabled.
- Gateway shutdown stops the coordinator before the persistence engine closes.
- In-flight runs that finalize after local coordinator stop may still persist jobs;
  they are recovered on the next enabled startup.

### Configuration

Config version 44 adds:

```yaml
skill_evolution:
  coordinator:
    queue_capacity: 64
    max_concurrent_jobs: 2
    poll_interval_seconds: 1.0
    lease_seconds: 120.0
    max_attempts: 5
    retry_base_delay_seconds: 5.0
    retry_max_delay_seconds: 300.0
    shutdown_timeout_seconds: 10.0
```

### Verification

```text
Durable Store + coordinator tests: 17 passed
Pipeline processor tests: 3 passed
Run-worker evolution tests: 6 passed
Phase 8.1 + migration/bootstrap/Gateway focused regression: 130 passed
Skill evolution + mutation regression: 288 passed, 1 skipped
Full backend offline suite: 11537 passed, 77 skipped, 39 warnings
```

### Boundary

Phase 8.2 adds compact lifecycle events and metrics. Phase 9 adds ownership/authz/
CSRF-protected APIs. Production replay-suite synthesis and runtime isolation remain
required before automatic evaluation can be enabled.

## 2026-08-17 - Phase 8.2 Content-Free Observability

### Event Contract

Added `skill_evolution/observability.py` with schema
`deerflow.skill-evolution.observation.v1`.

Allowed fields are fixed:

```text
kind + stage
user_id_hash + optional skill_name_hash
run/job/evolution-event/cluster/proposal/evaluation/publication IDs
snapshot hash
decision + reason codes
event/distinct-run/candidate/accepted counts
timestamp
```

There is intentionally no generic metadata map or free-text detail. Callers cannot
attach task input, final answer, tool arguments/results, paths, Proposal content,
model output, reviewer reason, exception message, or traceback.

User identity and Skill name are SHA-256 hashed before model construction. Structured
INFO logs serialize only the validated event. Deterministic observation IDs omit the
timestamp and deduplicate equivalent process-local retry emissions. Recent events use
a bounded ring; deduplication keys use a separate bounded LRU.

Lifecycle kinds:

```text
admitted
rejected (stage differentiates eligibility/extraction/confirmation/evaluation/
          approval/publication)
extracted
clustered
ready
distilled
evaluated
approved
published
rolled_back
```

### Metrics

`EvolutionMetricsSnapshot` contains only fixed global aggregates:

```text
queue_depth
active_jobs
extraction_failures
lifecycle_counts
proposal_pass_rate
publication_rate
cluster_purity distribution
regression_rate distribution
quality_lift distribution
```

Definitions:

- queue depth is the SQL/memory count of `pending + retry` jobs, not local wake-up
  queue size;
- proposal pass rate is persisted `approve` evaluations / all persisted evaluations;
- publication rate is new published versions / new distilled Proposals observed in
  this process;
- Cluster purity is accepted confirmed members / all considered candidates;
- regression rate comes from `skill-quality-v1.regression_resistance.raw_value`;
- quality lift comes from `skill-quality-v1.success_lift.raw_value`.

Counters/gauges reject negative values and unknown names. Distribution names and
ranges are fixed. Optional bounded deduplication keys prevent the same Cluster
revision or evaluation from being sampled twice after a job retry.

### Wiring

- Gateway stores the process observer as
  `app.state.skill_evolution_observability`.
- Coordinator refreshes durable backlog before/after claims and reports active task
  count on dispatch/completion.
- Pipeline emits:
  - eligibility admitted/rejected;
  - extraction success/failure;
  - deterministic Cluster candidate;
  - confirmation rejection or ready;
  - Proposal distillation success/failure.
- Both evaluators emit only when a new evaluation row is created, after any rejection
  transition has committed.
- Approval emits only when Proposal status CAS changes state.
- Publication emits after `preparing -> published`; rollback emits after
  `published -> rolled_back`.
- A failed package mutation emits content-free `publication_failed` after Proposal
  recovery, then preserves the original exception.
- Idempotent reads/retries with no durable state change do not emit another lifecycle
  event.

Every integration uses `safe_*` wrappers. Observer exceptions are swallowed and the
warning deliberately omits exception text and traceback.

### Verification

```text
Observability core + Memory/SQL job-count contract: 21 passed
Lifecycle integration including real K=3 ready/distilled: passed
Skill evolution + mutation + Gateway focused regression: 313 passed, 1 skipped
Full backend offline suite: 11550 passed, 77 skipped, 39 warnings
```

### Boundary

Metrics and recent-event buffers are process-local and reset on restart. This task
did not add a Prometheus dependency, `/metrics`, or user-facing read route. Phase 9.1
now exposes compact authenticated persisted records; external metrics exporters can
consume the fixed observability snapshot contract later.

## 2026-08-17 - Phase 9.1 Authenticated Read APIs

### Surface

Gateway now mounts:

```text
GET /api/skill-evolution/events
GET /api/skill-evolution/clusters
GET /api/skill-evolution/proposals
GET /api/skill-evolution/evaluations
GET /api/skill-evolution/versions
```

All five routes require the independent route permission
`skill_evolution:read`. Authorization-disabled deployments retain the legacy
all-permissions behavior. An enabled provider evaluates this permission separately;
`runs:read` does not imply it, and an explicit route allowlist that omits it denies
the new endpoints.

### Ownership and Pagination

The authenticated user is the default and mandatory Store partition. Supplying a
different `user_id` is rejected before Store access unless the caller passes the
existing hard-coded admin gate. Admin callers still need `skill_evolution:read`.

Each Store implementation applies owner and optional enum filters before
newest-first ordering, then executes `offset` plus `limit`. Stable ID ordering breaks
timestamp ties. Routes request `limit + 1`, return at most 100 records, and derive
`has_more` and `next_offset` without loading the full owner collection.

### Compact Boundary

The API returns identifiers, enum states, fixed counts, hashes, and timestamps.
Event task signatures and Cluster canonical signatures are SHA-256 hashed.
Responses exclude:

```text
task goal and successful/failed path prose
Proposal file content and rationale
raw replay errors, artifacts, and safety payloads
publication package files and base64 snapshots
```

### Verification

```text
Read API + Memory/SQL Store + route authorization: 53 passed
Skill evolution + authorization + Gateway focused regression: 488 passed, 1 skipped
Focused Ruff checks: passed
Full backend offline suite: 11564 passed, 77 skipped, 39 warnings
```

### Boundary

Phase 9.1 itself is read-only. Phase 9.2 now provides the explicit mutations while
preserving CSRF, route permission, owner/admin, status-CAS, and
`SkillMutationService` publication boundaries.

## 2026-08-17 - Phase 9.2 Review, Publication, and Rollback APIs

### Surface and Permissions

Gateway now mounts:

```text
POST /api/skill-evolution/proposals/{proposal_id}/review
     permission: skill_evolution:review
POST /api/skill-evolution/proposals/{proposal_id}/publish
     permission: skill_evolution:publish
POST /api/skill-evolution/versions/{publication_id}/rollback
     permission: skill_evolution:rollback
```

Review accepts only `evaluation_id`, `decision=approve|reject`, and a bounded reason.
Publish accepts only `evaluation_id`. Rollback has no client-owned audit fields.
Reviewer/rollback actor IDs come from the authenticated user and decision times come
from the server clock.

### Security Boundaries

- The authenticated user remains the default Store partition.
- A different `user_id` is rejected before Store/service access unless the caller
  also passes the hard-coded admin gate.
- Admin callers still need the operation-specific route permission.
- The existing Gateway `CSRFMiddleware` protects every POST.
- Request DTOs forbid client-supplied reviewer, actor, and timestamp fields.
- Domain `ValueError` and Store CAS conflicts become fixed 409 responses; scanner,
  model, package, or exception text is never reflected.
- Responses reuse compact Proposal/version DTOs and never include Proposal files,
  package snapshot files/base64, safety payloads, or raw replay errors.

### Service Wiring

Gateway lifespan owns one `SkillPublicationService` bound to the process Store and
observer. Keeping it process-scoped preserves the `SkillMutationService` instance and
its per-user/per-Skill async locks across concurrent HTTP requests.

Review delegates to `ProposalApprovalPolicy.review_and_persist()`. Publication and
rollback delegate only to `SkillPublicationService`; the Router never writes Skill
storage. The existing pipeline remains:

```text
route auth/owner/CSRF
→ approval/publication state CAS
→ persisted package snapshot
→ SkillMutationService validation and lock
→ SkillScan + moderation
→ atomic package mutation
→ publication/proposal terminal CAS
```

Manual-review retries with the same decision, evaluation, reviewer, and reason now
reuse the first persisted decision time. This makes an HTTP retry idempotent without
letting the client choose the audit timestamp.

### Verification

```text
Write/read API + approval/publication + Store/auth/CSRF/lifespan: 172 passed
Skill evolution + authorization + Gateway focused regression: 573 passed, 1 skipped
Focused Ruff checks: passed
Production route mount check: 5 GET + 3 POST routes
Full backend offline suite: 11578 passed, 77 skipped, 39 warnings
```

### Boundary

The background coordinator still stops at a staged Proposal and cannot invoke these
routes or services automatically. Production replay-suite synthesis remains required
before automatic evaluation can feed the manual review surface.

## 2026-08-18 - Phase 10 Structured Credit Logging

### Durable Contract

Added three strict immutable/versioned models:

```text
deerflow.skill-evolution.selection-credit.v1
deerflow.skill-evolution.utilization-credit.v1
deerflow.skill-evolution.distillation-credit.v1
```

All records share deterministic `credit-*` IDs and the per-user
`skill_evolution_credits` Store. Alembic `0015_skill_credits` adds the table with
owner/kind/Skill/publication indexes. Selection and utilization revisions remain
zero. Distillation future samples update under revision CAS; Memory and SQL stores
share the same idempotency and owner-isolation contract.

### Selection Credit

The recorder uses only evidence present in the redacted trace:

- a real `describe_skill(name=...)` query when one exists;
- `select:<skill>` for explicit slash selection;
- observed Skill names/content hashes as candidates;
- primary selected Skill or explicit no-Skill decision;
- model/prompt metadata only for model-driven read/no-Skill decisions;
- provider log probability only when supplied reliably (currently null with an
  explicit unavailable code);
- verified current-run reward and `selection-ewma-v1`.

Candidate rank/score remain null because the current Skill catalog path does not
persist model scores. The implementation does not invent them.

### Utilization Credit

One record is written per distinct observed Skill version/activation source in a run.
It includes:

```text
activation source + exact Skill content hash
run-level tool call IDs/names/status/error types
deterministic deviation codes from tool errors
verified outcome status/confidence/reward
tool/error counts
explicitly unavailable token/latency fields
```

Tool arguments/results, task text, final answer, and Skill content are excluded.
Tool attribution is labelled `run_level`; it is not claimed to prove per-instruction
causality. Instruction adherence remains null with
`model_feature_not_computed`; no text-matching heuristic is used.

### Distillation Credit

Successful publication creates a record linking:

```text
Proposal + publication + source EvolutionEvent IDs
published Skill hash
raw source outcome series
raw future utilization outcome series
```

Only utilization records whose exact Skill hash equals the published hash and whose
run occurred after publication are accepted. Run IDs are unique. After three known
future rewards:

```text
post-publication-marginal-utility-v1
  = mean(future rewards) - mean(source baseline rewards)
```

Unknown rewards remain in the raw series but do not satisfy the sample gate.
Rollback marks the corresponding record `rolled_back` without deleting its series.

### Wiring and Failure Policy

- `EvolutionPipelineProcessor` runs the deterministic verifier, then records
  selection/utilization before eligibility/extraction.
- Existing-event retries still run Credit recording, whose deterministic IDs make
  crash recovery idempotent.
- New utilization samples advance matching Distillation Credit records.
- `SkillPublicationService` creates/repairs publication Credit and closes it on
  rollback, including retry-recovery paths.
- Credit failures log only run/publication IDs and remain non-fatal. Cancellation
  still propagates.

### Verification

```text
Credit model/Memory/SQL/formula tests: 9 passed
Credit + pipeline/publication/migration/bootstrap focused regression: 73 passed
Skill evolution + persistence focused regression: 363 passed, 1 skipped
Focused Ruff checks: passed
Full backend offline suite: 11590 passed, 77 skipped, 39 warnings
```

### Boundary

This phase records external Skill-policy evidence only. It does not train a model,
request provider log probabilities, infer instruction adherence, synthesize missing
rank scores, or export Verl datasets. Phase 12 owns training-ready export and
tokenizer/model/environment version packaging.

## 2026-08-18 - Phase 11 Experiment Harness and Statistical Evaluation

### Frozen Manifest

`skill_evolution/experiment.py` materializes the Phase 0 protocol as a strict
versioned manifest:

```text
3 domains
× 4 task families per domain
× (3 evidence + 2 held-out variants)
= 60 task variants
```

Every task has a stable ID, split, create/patch branch, fixture ID, objective, and
deterministic command/artifact/invariant verifier declaration. The domains are
repository repair, structured data transformation, and shell workflow.

### Conditions

The default matrix contains 12 conditions:

```text
baseline: no evolution
baseline: immediate single trajectory
baseline: staged single trajectory
reference: proposed K=3 hybrid
K: 1, 3, 5
grouping: deterministic, LLM, hybrid
publication: immediate, staged
evidence: success only, success plus recovered failures
branch: create, patch
```

Reference-equivalent values are shared rather than duplicated as separate rows. With
three seeds the frozen matrix contains 2,160 result slots.

### Result Provenance

Every JSONL result records:

```text
experiment/condition/task/family/split/variant/seed identity
production_replay | deterministic_smoke mode
executor name/version
model name/version
prompt version
environment fingerprint
primary and secondary metric facts
completed/failed/rejected/skipped status + stable error code
```

`production_replay` requires immutable model identity. Reports reject mixed execution
modes. `run.py` also writes a metadata sidecar containing the full manifest,
condition objects, seeds, executor path, concurrency, and result count, so a result
set never depends on reconstructing configuration from logs.

### Metrics and Statistics

Condition summaries report:

- task and held-out success rates;
- regression and incorrect-evolution rates;
- Cluster precision/purity;
- tool calls, input/output tokens, and latency;
- Proposal acceptance, evidence count, evolution delay, and security rejection;
- explicit completed, failed, rejected, and skipped counts.

Comparisons join rows by exact `(task_id, seed)`. Primary deltas use deterministic
paired bootstrap 95% confidence intervals. Binary success additionally uses an exact
two-sided McNemar test. Reports contain both aggregate and per-family summaries while
raw JSONL retains every row.

### CLI and Smoke Execution

```bash
cd backend
PYTHONPATH=. uv run python scripts/benchmark/skill_evolution/run.py \
  --validate-only --conditions all --seeds 1,2,3

PYTHONPATH=. uv run python scripts/benchmark/skill_evolution/run.py \
  --experiment-id skill-evolution-phase11-smoke \
  --executor scripts.benchmark.skill_evolution.smoke_executor:create_executor \
  --conditions all --seeds 1,2,3 \
  --output /tmp/deerflow-phase11-smoke.jsonl

PYTHONPATH=. uv run python scripts/benchmark/skill_evolution/summarize.py \
  /tmp/deerflow-phase11-smoke.jsonl \
  --reference baseline-no-evolution \
  --output /tmp/deerflow-phase11-smoke-report.json
```

The deterministic smoke run produced exactly 2,160 rows, 12 condition summaries, 12
family sections, and 11 paired comparisons. Its execution mode is
`deterministic_smoke`; no metric value from that report is an empirical model result.

### Verification

```text
Experiment manifest/runner/statistics tests: 8 passed
All Skill evolution focused regression: 335 passed, 1 skipped
60-task × 12-condition × 3-seed smoke matrix: 2160 rows
Smoke report: 12 conditions, 12 families, 11 paired comparisons
Focused Ruff checks: passed
Full backend offline suite: 11598 passed, 77 skipped, 39 warnings
```

### Boundary

The experiment and statistics infrastructure is complete, but Task 11.1 production
ablations remain open. DeerFlow still needs production ReplayTask suite
materialization and a conforming isolated `ReplayRuntime` executor before real model
runs can begin. This phase deliberately does not fabricate those results or mark
smoke metrics as thesis evidence.

## 2026-08-19 - Phase 11 Production Replay and Real Ablations

### Production Suite and Isolation

`replay_suite.py` now materializes all 60 frozen task slots as automatic replay
cases. Every case has a bounded fixture, task-specific transformation objective,
hidden semantic command verifier, artifact verifier, side-effect policy, and
create/Patch candidate boundary. Evidence task prompts carry unique case IDs so
three independent runs are not collapsed by task-input deduplication. Regular
evidence fixtures and edge-oriented held-out fixtures are separated explicitly.

`DockerReplayRuntime` runs business tools only through disposable
`python:3.12-slim` containers:

```text
--network none
--read-only
--cap-drop ALL
--security-opt no-new-privileges
--pids-limit 128
--memory 1g
--cpus 1
workspace/output bind mounts writable
Skill bind mount read-only
```

The production benchmark exposes only `read_file` and structured `write_json` to
the model. The runtime forces the selected candidate `SKILL.md` into the system
request, disables Ollama reasoning when `thinking_enabled=false`, preserves
partial tool/token progress across timeout/runtime errors, bounds returned
artifacts without weakening symlink fail-closed behavior, and always removes the
container/workspace.

The isolation probe verified:

```text
verifier command: exit 0
Skill write attempt: exit 2
network attempt: exit 1
```

### Generated Candidate Adapter

`production_evolution.py` is benchmark-only and never writes a Store or Skill
library. It converts successful verifier-backed replay outcomes into
`EvolutionEvent`s without copying oracle candidate guidance, then reuses the
production components:

```text
deterministic grouping
→ optional Ollama embedding/Qdrant retrieval
→ StructuredClusterConfirmer
→ K readiness
→ NewSkillDistiller or PatchSkillDistiller
→ in-memory read-only ReplaySkillPackage
```

K=1/single-trajectory baselines use a separate strict Qwen JSON distillation
prompt rather than the oracle package. The oracle builder remains only an
explicit test fixture. Generated candidates are cached only across conditions
with the same seed, evidence IDs, and grouping strategy; task replays and their
latency/token metrics are never cached.

Real create and Patch pilots both completed:

```text
3/3 evidence replay successes
→ ready cluster
→ Qwen-generated Proposal
→ staged candidate package
→ held-out replay success
```

### Recoverable Experiment Execution

The CLI accepts cohort-only executors and `--resume`. Results are atomically
checkpointed after each complete five-task family cohort. Resume rejects partial
cohorts or rows outside the requested experiment/condition/task/seed matrix and
reruns only missing cohorts. The metadata sidecar records
`in_progress|completed`, completed/planned result counts, full manifest,
conditions, seeds, executor, and concurrency.

### Completed Production Matrix

The real matrix completed with:

```text
60 tasks × 12 conditions × 3 seeds = 2,160 rows
432 complete family cohorts
12 condition summaries
12 family sections
11 paired comparisons
execution_mode: production_replay
executor: docker-replay-generated@production-evolution-generated-v1
model: qwen3-local
model digest:
  500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41
all runtime statuses: completed
security rejections/runtime error codes: 0
```

Local evidence files:

```text
.deer-flow/benchmarks/skill-evolution/production-phase11-v1.jsonl
.deer-flow/benchmarks/skill-evolution/production-phase11-v1.jsonl.metadata.json
.deer-flow/benchmarks/skill-evolution/production-phase11-v1.report.json
.deer-flow/benchmarks/skill-evolution/production-phase11-v1.proposed-reference.report.json
```

### Result Interpretation

The real result is neutral for aggregate K=3 benefit and negative for immediate
publication:

```text
baseline no evolution held-out: 0.6667
proposed K=3 hybrid held-out:   0.6667

immediate publication vs K=3 staged held-out delta:
  -0.0833
paired bootstrap 95% CI:
  [-0.1528, -0.0278]
exact two-sided McNemar:
  p = 0.03125
```

K=1 and staged-single were `-0.0278` below K=3 on held-out, but their confidence
interval `[-0.0694, 0.0]` includes zero. Deterministic, LLM, and hybrid grouping
were identical on these deliberately homogeneous family cohorts. K=5 and
success-only produced no proposals because the frozen manifest contains only
three evidence variants and success-only excludes the recovered-error variant;
they therefore measure the expected no-evolution fallback rather than a
five-evidence learned candidate. Create-only and Patch-only ablations also
matched the no-evolution held-out rate outside their enabled branch.

These results support the staging safety claim, not a general performance-lift
claim. Four families have zero no-Skill held-out success and also fail to reach
K=3 reliably because their evidence executions are not consistently successful;
future benchmark expansion needs more independent successful evidence variants
before it can test whether evolution improves those hard families.

## 2026-08-30 - Explicit Direct Publication

The online pipeline now supports the operator-selected
`skill_evolution.publication.mode=direct` path:

```text
ready K=3 Cluster
→ staged create/Patch Proposal
→ SkillPublicationService.publish_direct()
→ SkillMutationService
→ published custom Skill
```

This mode intentionally creates no `SkillEvaluation` and performs no approval
transition. Its publication and mutation history persist `evaluation_id=null`.
The distributed example remains `manual`; the local development config opts into
`direct`.

Direct mode also suppresses the lead agent's active self-evolution prompt and
does not register `skill_manage` in its toolset. This prevents one task from
writing a Skill before the three-run evidence threshold; only the durable
K-ready pipeline may invoke direct publication. `manual` and `eligible_auto`
retain the existing explicit agent-managed Skill path.

Direct mode does not bypass the trusted mutation boundary. It still captures the
complete base/candidate package, validates support paths, runs native SkillScan
and LLM security moderation, enforces base/package CAS, atomically replaces the
package, refreshes caches, records publication Credit, and retains exact rollback
snapshots. Publication failures return `publishing` to `staged`; the durable run
job retries and completed mutations are idempotently recovered.

Alembic revision `0016_direct_skill_publication` makes
`skill_evolution_publications.evaluation_id` nullable. Downgrade is rejected while
direct-publication rows exist rather than inventing an Evaluation identity.

Direct-mode verification:

```text
Focused evolution/publication/persistence regression:
  442 passed, 1 skipped
Full backend offline pytest:
  11,626 passed, 77 skipped, 17 warnings
Focused Ruff checks:
  passed
```

The full `make test` pytest run completed without test failures. The surrounding
TRAE command wrapper returned non-zero afterward because a sandbox hook attempted
to read Docker Desktop's host log outside the workspace; this did not alter the
pytest result.

### Qwen Distillation Compatibility

A three-run production Cluster exposed a provider behavior where Qwen emitted
the absolute source path `/mnt/user-data/workspace/test_normalize_config.py` as
an optional supporting file. The response was otherwise strict JSON, but the
safe-path validator correctly rejected the absolute path and both bounded
distillation attempts returned no Proposal.

New-Skill parsing now discards only optional supporting-file entries that
violate the existing normalized-relative-path policy before validating the
rest of the response. Mandatory workflow and verification evidence remains
fail closed. Retained support files still require two distinct source runs and
still pass the publication path, SkillScan, and LLM moderation boundaries.

The same persisted Cluster then completed the real direct path:

```text
3 EvolutionEvents / 3 independent runs
→ ready Cluster
→ staged Proposal
→ published Skill

proposal: proposal-2f04b282b5c14a051c3fccd4206dfe02
publication: publication-proposal-2f04b282b5c14a051c3fccd4206dfe02
skill: legacy-json-config-to-v2-validation
evaluation_id: null
```

### Verification

```text
Production replay/model/experiment focused regression: 128 passed
Resume/checkpoint and production adapter focused tests: 13 passed
Focused Ruff checks: passed
Full backend offline suite: 11,616 passed, 77 skipped, 39 warnings
Production matrix integrity:
  2,160 unique result IDs
  2,160 unique condition/task/seed keys
  one production execution mode
  one immutable model digest
  no leaked replay containers
```
