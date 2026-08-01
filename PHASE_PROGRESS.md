# Phase Progress

## Phase 0 — Pre-check
Status: done
Confirmed all core files exist and read for context (src/pipeline.py, schemas.py,
gemma_client.py, orbital.py, src/ingestion/ adapters, src/logging_utils.py,
NOTES_FOR_MORNING.md, tests/). 16/16 tests passing at baseline, GEMMA_BACKEND/
GEMMA_MODEL=gemma4:e4b confirmed in .env + .env.example + config.py, LICENSE
confirmed unmodified original waterloodev MIT text.

## Phase 1 — Maneuver calculation
Status: done
Added src/maneuver.py (compute_avoidance_maneuver) and ManeuverPlan schema.
Simplified linear displacement model, clearly commented as not real
astrodynamics (fixed radial-outward direction, target clearance = 30km base +
velocity margin). 3 new tests, not yet wired into pipeline.py. Suite: 19/19.

## Phase 2 — Re-verification
Status: done
Added verify_maneuver -> VerifiedClearance (independently re-derives distance
forward from delta-v rather than echoing the plan's own target). 5 new tests.
Suite: 24/24.

## Phase 3 — Wire into decide_node + LaTeX fix
Status: done
decide_node now computes+attaches maneuver_plan/verified_clearance for CRITICAL
severity (Decision schema extended, nullable). CRITICAL Gemma prompt rewritten
to narrate completed autonomous action, past tense; WATCH/WARNING/HOLD language
untouched. Added "plain prose only, no LaTeX/markdown" instruction to both
Gemma system prompts (both previously lacked it). 2 new tests. Verified via a
real end-to-end run against Ollama/gemma4:e4b with a synthetic 3.2km fixture.
Suite: 26/26.

## Phase 4 — Visible local/cloud failover
Status: done
Confirmed GemmaClient previously only retried the SAME backend (no
cross-backend fallback). Added fallback: after primary backend's existing
retry is exhausted, attempts the other backend once if configured, before
falling through to the deterministic fallback text. GemmaProvenance.model_used
now names the responding backend when it differs from the configured one.
5 new tests (mocked, no network), including one full pipeline-level check that
provenance reflects the fallback correctly. Suite: 31/31.
ADDENDUM (post-Phase-5, real key wired up): found and fixed a real security
bug - the hosted API call put the key in the URL query string, so a failed
request's error text leaked it (this happened once in conversation before
the fix). Switched to header-based auth (x-goog-api-key). Also split
gemma_model into gemma_model_api (Ollama tags and hosted model ids use
incompatible naming) - now gemma4:e4b locally, gemma-4-26b-a4b-it via the
real hosted API, confirmed against the key's actual available-models list.
Verified BOTH real fallback directions with genuine network calls (not
mocks): cloud-primary-fails-to-local, and local-genuinely-broken-falls-to-
real-cloud. Chain-of-thought finding: fixed. Checked whether the API supports disabling
"thinking" via generationConfig.thinkingConfig - real request confirmed
"Thinking budget is not supported for this model" (both available hosted
models exhibit the same verbose behavior; not a toggleable mode). Fix
applied in code instead: new _extract_final_answer() in pipeline.py, a
single choke-point both _conjunction_description and _decision_rationale
already funnel through - takes the last non-empty paragraph of any Gemma
response. No-op for Ollama's already-clean single-paragraph replies;
strips the reasoning trace down to just the real answer for the hosted
model. Verified against a real live fallback call (broken ollama_host ->
real cloud): rationale went from ~2800 chars of scratch-work down to a
clean 145-char sentence. Refined once more after a second real call
exposed a gap: the model doesn't always separate its label from the real
answer with a blank line (sometimes just "Final Polish:\n    <answer>" on
consecutive single-newline lines) - the original \n\n-paragraph-based
extraction missed that case. Switched to "last non-empty line" instead of
"last paragraph", which correctly handles both patterns; re-verified
clean across 3 more consecutive real live calls. 4 new tests (was 3).
Suite: 48/48.
Wrote DEMO.md - a 9-stage, copy-paste walkthrough of everything built so
far (Phases 0-5 plus this failover/cleanup work), each command tested for
real before being included.

## Phase 5 — Credibility/gap-closing polish
Status: done
Confidence: analyze_node now derives AnomalyFinding.confidence from real TLE
epoch age (new parse_tle_epoch in orbital.py, wired through
CelesTrakAdapter's tle_epoch_age_hours) when available, falling back to a
clearly-commented placeholder constant otherwise. Delta-v budget: new
DeltaVBudgetTracker (config-level DELTA_V_BUDGET_M_S, default 5.0)
decrements per executed CRITICAL maneuver; insufficient budget now sets
Decision.budget_insufficient=True, skips verify_maneuver, and reframes the
Gemma rationale as "calculated but not executed - escalate for review"
instead of falsely claiming success. Human-review: DecisionLogger gained
find_entry/mark_reviewed (rewrites the matching JSONL line in place - the
one deliberate exception to append-only), new reviewed_by field on
DecisionLogEntry, plus scripts/mark_reviewed.py CLI - verified with a real
(non-mocked) log-then-mark run, not just unit tests. Suite: 44/44.

## Phase 6 — Terminal output styling
Status: not started
