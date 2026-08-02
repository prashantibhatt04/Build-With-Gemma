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
FOLLOW-UP after user feedback + open-issue review:
1) scripts/run_demo.py's failover step (Step 4) always deliberately broke
local Ollama and made a real cloud call, regardless of the demo's
configured backend - meaning a LOCAL-only demo run was silently touching
the cloud. Fixed: that step now skips entirely (with an explanation
printed, not silent) whenever GEMMA_BACKEND=ollama. Verified with a fresh
full local run: step correctly skipped, rest of the 8-step walkthrough
unaffected, summary unchanged (3 autonomous, 1 budget-blocked).
2) Resolved the previously-flagged "Ok, let's go."-style extraction bug:
_extract_final_answer now searches backward for the last line that's
actually substantive (>= 20 chars) instead of blindly taking the literal
last line, so a short throwaway remark after the real answer no longer
gets selected instead of it. 2 new tests, including one reproducing the
exact real failure pattern. Re-verified against 5 fresh real cloud calls -
all returned clean, substantive rationale text (135-162 chars), no
regressions.
No other open issues found (checked for TODO/FIXME/XXX across src/,
scripts/, tests/ - none). Suite: 73/73.

## Phase 9 — Gemma as autonomous verifier (veto gate)
Status: done
Built exactly to the design scoped earlier in this file: on the
autonomous (local/Ollama) path only, after compute_avoidance_maneuver +
verify_maneuver independently confirm a CRITICAL maneuver is safe, Gemma
now gets those same numbers and issues a real GO/NO-GO verdict via a new
_maneuver_veto_check (src/pipeline.py), standing in for the unavailable
human. Reuses the existing ManeuverApproval schema exactly as planned -
mode="autonomous" now carries Gemma's actual verdict instead of a
hardcoded True: approved=True/approved_by=None when Gemma affirms (or is
unreachable - see fail-safe below); approved=False/approved_by="Gemma
(autonomous safety review)" when Gemma explicitly vetoes or gives an
unparseable answer. verified_clearance stays None whenever vetoed - same
"nothing was actually applied" invariant as budget_insufficient/
awaiting_human_approval - so a vetoed maneuver can never be mistaken for
an executed one downstream. The two fail-safe defaults from the original
design are both implemented and tested: an unparseable verdict defaults
to NO-GO (escalate, not a free pass); Gemma being unreachable is NOT
treated as a veto and falls back to physics-only autonomy (an LLM outage
alone shouldn't block an already-verified-safe maneuver) - the response
text guarantees this deterministically rather than relying on parsing.
Verdict parsing (_parse_veto_verdict) scans the FULL response for
GO/NO-GO tokens and takes the last one found, not just the last line -
this deliberately does NOT reuse _extract_final_answer's "last
substantive line" heuristic, since a short verdict token given first (as
instructed) would get discarded by that heuristic in favor of a longer
justification sentence after it; _call_gemma_with_provenance gained an
optional postprocess param so the veto call can skip that extraction
instead of duplicating the whole helper. _decision_rationale gained a
4th CRITICAL branch (maneuver_vetoed) so Gemma's own narration call
correctly says "Maneuver vetoed: blocked pending review" instead of
falling through to the generic recommendation text. display.py now
renders a vetoed maneuver under its own "MANEUVER VETOED - AUTONOMOUS
SAFETY REVIEW" title, kept visually distinct from a human rejecting a
cloud-pending proposal (same underlying "REJECTED" condition, different
mode) - existing human-rejected test/text left untouched. Added
veto_provenance (GemmaProvenance, nullable) to DecisionLogEntry/
PipelineState for the same audit-trail parity as description/rationale
provenance, threaded through make_log_node and run_once. scripts/
run_demo.py's CRITICAL step explanation and summary counts updated -
"autonomous" execution count is now correctly restricted to
approved=True (a vetoed maneuver has mode="autonomous" but wasn't
executed), and the summary separates "Vetoed by Gemma" from "Rejected by
human" instead of lumping both into one count. Delta-v budget is NOT
refunded when Gemma vetoes a maneuver it already consumed budget for -
kept consistent with the pre-existing behavior for human rejection
(budget is spent at proposal time regardless of outcome), not changed as
part of this phase. 14 new tests (parsing edge cases, explicit NO-GO,
unparseable-defaults-to-NO-GO, unreachable-fail-safe-GO via the existing
FailingGemmaClient test strengthened with new assertions, GO-affirmed
provenance, display panel distinction). Suite: 87/87. Verified live
end-to-end against real local Ollama (gemma4:e4b) twice: once through the
full analyze->decide pipeline (real GO verdict, maneuver executed,
correct narration), and once calling the new vetoed-narration branch
directly (real model output: "Maneuver vetoed: blocked pending review...")
since a genuinely unsafe/vetoed maneuver essentially never occurs in
normal operation by design, so the narration branch needed a targeted
live check rather than relying on the pipeline to produce one naturally.

## Phase 10 — Real cross-group conjunction screening
Status: done
Previously CelesTrakAdapter fetched one fixed group
(cosmos-2251-debris, sample_size=30 by default, 15 in the demo) and
screened only within-group pairs - real data, but a small, narrow slice.
Rebuilt to screen across MULTIPLE real CelesTrak groups at once - default
DEFAULT_GROUPS=("stations", "cosmos-2251-debris") - answering the
actually-motivating question ("is a real active spacecraft at risk from
real tracked debris?") instead of debris-vs-debris alone. Benchmarked
first before optimizing (not assumed): the naive approach - calling
orbital.find_closest_approach per pair, unchanged - took ~9.6s for 1770
pairs from a real 60-object sample, confirming naive brute force doesn't
scale to a meaningfully larger real pool. Root cause: each pair
independently recomputed its own coarse-pass propagation from scratch, so
cost was O(pairs) expensive Skyfield calls, not O(objects). Fixed with two
changes verified independently by direct benchmark before combining:
1. src/orbital.py decomposed into build_coarse_times/
   compute_coarse_positions/coarse_min_distance/refine_closest_approach,
   with find_closest_approach kept as a behavior-identical single-pair
   convenience wrapper (existing test unchanged, still passes, plus a new
   test proving the decomposed path produces bit-identical results).
   CelesTrakAdapter now calls compute_coarse_positions ONCE per satellite
   and reuses it across every pair that satellite appears in - cut the
   same 1770-pair coarse screen from ~9.6s to ~0.02s in isolated
   benchmarking.
2. Rank ALL pairs by that cached coarse distance, then only run the
   expensive fine-pass refinement (orbital.refine_closest_approach) on
   the closest `refine_top_k` candidates (default 80) - sound because the
   coarse pass can only ever OVERESTIMATE the true closest approach (see
   orbital.py's module docstring), so it's a safe proxy for true ranking,
   not a guess.
Combined: fetching+screening the new default (121 real objects, 7050
pairs) end-to-end, including the live network fetch, measured at ~0.6-1s
- fast enough for a live demo step.
Two real issues caught only by actually running this live against
CelesTrak (not just unit tests, matching this project's established
practice) and fixed before calling it done:
- CelesTrak's real "stations" group includes crewed stations AND their
  currently-docked visiting vehicles (Soyuz, Progress, Cygnus, Crew
  Dragon, ...) - these sit at ~0.00km separation from each other, which
  is real, correct physics (they're physically attached) but not an
  operational conjunction risk, and it dominated the top-ranked results,
  crowding out anything more interesting. Fixed with
  exclude_within_group (default: {"stations"}) - pairs within the same
  excluded group are skipped before ever costing coarse-ranking or
  refinement effort, not filtered post-hoc.
- Even after that fix, a dense single-origin debris field
  (cosmos-2251-debris - many fragments from one 2009 breakup, naturally
  close to each other) filled the ENTIRE refine_top_k ranking on its own
  in live testing - zero cross-group ("asset vs. debris") pairs got
  refined, silently defeating the actual point of screening multiple
  groups. Fixed with min_cross_group_refine (default 20): the closest
  that many CROSS-group pairs by coarse distance are unioned into the
  refinement set regardless of the overall ranking, guaranteeing
  cross-group representation rather than leaving it to chance.
last_scan_stats (groups, total_objects, total_pairs_screened,
pairs_refined, cross_group_pairs_refined) exposed on the adapter instance
so callers (scripts/run_demo.py) can report exactly what was screened
without threading extra return values through fetch_batch. raw_data
gained object_a_group/object_b_group for traceability. run_demo.py's
orbital-data step and DEMO.md Stage 2 updated to match, plus a new Stage
2b walking through the performance approach with a runnable benchmark
snippet. 9 new tests (decomposition equivalence, cross-group screening,
refine_top_k bounding, exclude_within_group, min_cross_group_refine
guarantee). Suite: 93/93. Verified live end-to-end against real CelesTrak
multiple times during development, including through a full
scripts/run_demo.py --auto run against real local Ollama - confirmed fast
(~0.6s), confirmed cross-group results present, confirmed no docked-
vehicle noise in the top results.
