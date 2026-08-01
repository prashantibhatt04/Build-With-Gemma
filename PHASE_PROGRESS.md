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
Status: done
Added `rich` to requirements.txt. New src/display.py (render_entry/
render_entries): color-coded severity badges (green NOMINAL, yellow WATCH,
dark_orange WARNING, bold red CRITICAL), one-line summary per event
(object names, distance, severity, rationale), and a bordered Panel when a
CRITICAL maneuver was computed - one style for "executed" (red border,
direction/delta-v/verified clearance) and a distinct one for
"budget-insufficient" (yellow border, "NOT executed... escalate for
review"). Only src/pipeline.py's `if __name__` block was touched - log_node/
DecisionLogger untouched, confirmed by grepping the actual log file for
styling artifacts (rich markup, box-drawing characters) after a real run:
zero matches. 5 new tests (Console(record=True), no real terminal needed).
Suite: 53/53. Real run output pasted in chat for review.

## Phase 7 — Preflight check + automatic demo runner
Status: done
Not user-specified yet at time of building (user said "implement next
phase" while setting up the laptop demo) - judgment call: since exactly
that laptop setup was the live task, built the thing most useful for it
right now. New src/preflight.py: check_config (validates GEMMA_BACKEND +
that GEMMA_API_KEY is set when backend=api), check_log_dir_writable,
check_gemma_reachable (one real call, reports which backend actually
answered - flags if it silently fell back). New scripts/run_demo.py: runs
preflight as a rich Table, then a live CelesTrak scan, then the same
synthetic CRITICAL/budget-depletion scenario as DEMO.md Stage 5, all via
src/display.py, ending in a rich summary table (counts by severity, Gemma
vs fallback split, maneuvers executed vs budget-blocked). One command,
no copy-pasting DEMO.md snippets. 9 new tests (Gemma connectivity mocked,
config/filesystem checks real). Suite: 62/62. Ran the real script
end-to-end - confirmed log file untouched by styling (same check as Phase
6).
REVISED after user feedback: wanted ONE reusable, step-by-step,
self-explanatory demo script instead of a straight-through batch run -
rewrote scripts/run_demo.py entirely around a `Step` list (phase, title,
explanation, action), each one printing a plain-language explanation
(written for a cold GitHub reader, not just someone who built it) and
pausing for Enter/skip before running (rich Confirm.ask; `--auto` skips
pauses for CI). Now covers 8 steps: preflight, live CelesTrak data,
CRITICAL maneuver+verification+budget depletion, live local/cloud
failover, human review, raw audit log readback, test suite, summary.
Designed so adding a future phase means appending one Step, not writing
a new file. Caught and fixed a real bug while re-running it live:
synthetic event_ids were hardcoded per-index (not per-run), so a second
demo run's mark_reviewed/find_entry matched the FIRST same-id entry
(from an earlier run) instead of the current run's - audit-trail step
showed human_reviewed=false right after marking it true. Fixed by
including a per-run uuid in the synthetic event_ids; re-verified correct
across 3 consecutive real runs. Suite unaffected (62/62, no src/ logic
changed, only scripts/run_demo.py).

## Phase 8 — Human-approval gating for CRITICAL maneuvers (local vs cloud)
Status: done
User's proposed design, confirmed sound before implementing: maneuver
PHYSICS stays fully deterministic (unchanged) - what's new is an approval
gate around COMMITTING an already-computed maneuver. The configured Gemma
backend doubles as a stand-in for "is ground control reachable": local
(ollama) = unreachable -> self-approve immediately based on the existing
deterministic severity/physics checks, no human in the loop (same
behavior as before, now explicitly labeled autonomous). Cloud (api) =
reachable -> maneuver is calculated and budget-checked but held pending;
a human must explicitly approve or reject it before verify_maneuver()
ever runs. Gemma's role is unchanged in spirit - narrator only, never
decides the physics - just now narrates a 3rd possible state ("proposed,
awaiting approval") alongside the existing "executed" and
"budget-insufficient" ones.
New: schemas.ManeuverApproval (mode: autonomous/human, approved,
approved_by, approved_at, reason) and Decision.awaiting_human_approval.
decide_node branches on client.settings.gemma_backend after the existing
budget check (which still takes priority - unaffordable maneuvers never
reach the approval branch on either backend). New
DecisionLogger.approve_maneuver() (+ scripts/approve_maneuver.py CLI,
same in-place-rewrite pattern as mark_reviewed): on approval, actually
runs verify_maneuver() for the first time; on rejection, records that and
leaves nothing executed. Existing FakeGemmaClient/FailingGemmaClient test
stubs updated to default gemma_backend="ollama" so all pre-Phase-8
CRITICAL tests keep exercising the autonomous path they were written
against, unchanged.
scripts/run_demo.py's CRITICAL step now resolves any pending approvals
live - interactive prompt per pending maneuver (auto-approved in --auto
mode) - and the summary step reports autonomous vs human-approved vs
rejected vs still-pending counts.
Caught and fixed a real crash while verifying live against the real cloud
backend (not just local): src/display.py's render_entry assumed
verified_clearance was always set whenever budget wasn't insufficient -
true before this phase, false now that "awaiting approval" is a 3rd
state with both budget_insufficient=False and verified_clearance=None.
AttributeError on entry.decision.verified_clearance.new_min_distance_km.
Rewrote render_entry to branch on all 5 real states (budget-insufficient
/ awaiting-approval / autonomous-executed / human-approved-executed /
rejected) instead of assuming only 2. 3 new display tests, one explicitly
a regression test for the exact crash. Verified real end-to-end runs on
BOTH backends (not just unit tests): local (ollama) - confirmed
unchanged autonomous behavior, 3 executed + 1 budget-blocked as before;
cloud (api) - confirmed proposals correctly held pending, real Gemma
narration reads "Maneuver proposed: awaiting human approval before
execution", and (after the display fix) the full run completes without
crashing through to the summary. 6 new tests total across
test_pipeline_smoke.py/test_logging_utils.py/test_display.py. Suite:
71/71.
