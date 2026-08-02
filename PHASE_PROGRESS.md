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

## Phase 11 — Live mission-ops dashboard
Status: done
Added scripts/dashboard.py (Streamlit): a read/act layer over the exact
same append-only audit log the CLI writes to, not a second copy of the
pipeline. Run with `streamlit run scripts/dashboard.py`. Shows a metrics
row (totals by state - executed/vetoed/rejected/awaiting/budget-blocked),
a live decision table, a "pending human approval" inbox with real
Approve/Reject buttons wired to DecisionLogger.approve_maneuver, and an
inspect/mark-reviewed panel wired to DecisionLogger.mark_reviewed.
Sidebar buttons generate real new activity from inside the dashboard
itself: "Fetch live CelesTrak conjunctions" (Phase 10's cross-group
screening) and "Run synthetic CRITICAL scenario" (src/ingestion/
synthetic_adapter.py) - both go through the real src/pipeline.run_once,
nothing bypassed.
Two refactors done first, both to avoid duplicating logic the dashboard
needed a second copy of:
1. SyntheticCriticalAdapter was previously defined inline inside
   scripts/run_demo.py - promoted to src/ingestion/synthetic_adapter.py
   (with a configurable id_prefix so run_demo.py's existing
   "conj-run-demo-*" event ids are unchanged) so the dashboard could reuse
   it instead of duplicating ~30 lines of fixture code. Verified
   byte-identical event_id output before/after.
2. display.py's render_entry had six mutually-exclusive maneuver-state
   branches (budget-insufficient/awaiting-approval/executed-autonomous/
   executed-human-approved/vetoed/rejected) inlined directly in the Rich
   rendering code. Extracted the branching itself into
   classify_decision_status(decision) -> Optional[str], with render_entry
   rewritten to switch on it - single source of truth so the dashboard's
   table/metrics can never disagree with the CLI about what state a
   decision is in. All 9 pre-existing display tests pass unchanged,
   confirming the refactor didn't alter any rendered output.
Also added DecisionLogger.load_all_entries() (reads every decisions-*.jsonl
under log_dir, not just one) and src/dashboard_data.py - the dashboard's
non-UI logic (row/metric transforms) kept Streamlit-free specifically so
it's directly unit-testable without simulating a UI, mirroring why
src/preflight.py's checks are separated from scripts/run_demo.py's
printing.
19 new tests: dashboard_data's transforms/aggregation (5, pure, no
Streamlit), DecisionLogger.load_all_entries (3), classify_decision_status
direct coverage (7), and 3 streamlit.testing.v1.AppTest-based smoke tests
for the actual app (loads without exception, shows expected metrics,
sidebar controls present) - the two sidebar buttons that make real
network/Gemma calls are deliberately NOT exercised by automated tests,
consistent with this project's existing rule that the test suite never
needs live network. Suite: 111/111.
Live-verified for real: ran the dashboard against the real accumulated
audit log (144 entries at the time) in an actual browser. Confirmed
correct metrics, a working 3-item pending-approval inbox with real
conjunction data, and - critically - clicked a real "Approve" button and
confirmed via the raw log file afterward that DecisionLogger.approve_maneuver
actually ran (maneuver_approval.mode="human", approved_by="dashboard-
operator", verified_clearance populated) and the UI correctly dropped to
"Pending human approval (2)" immediately after.

## Phase 12 — Historical replay/backtest
Status: done
Added src/ingestion/historical_adapter.py (HistoricalReplayAdapter):
replays a real, documented historical conjunction through the exact same
pipeline live data goes through, unmodified - default event is the 2009
Iridium 33/Cosmos 2251 collision, the first confirmed accidental
collision between two intact satellites, and thematically the same
"cosmos-2251-debris" story already used everywhere else in this project.
Investigated genuine historical TLE propagation first rather than
assuming it wasn't possible: confirmed by directly querying it that
CelesTrak's public gp.php endpoint ignores an EPOCH query parameter and
always returns the CURRENT TLE regardless - real historical TLE archives
require Space-Track.org, which needs a real account/credentials this
project doesn't have and can't obtain on a user's behalf. Rather than
fake historical propagation or invent numbers, this replays the REAL
closest-approach prediction exactly as it was documented at the time.
Every number is sourced and independently corroborated, not invented:
the 584m predicted closest approach and the report timing (final report
issued 2009-02-10 15:02 UTC, predicted closest approach ~16:56 UTC the
same day) come directly from CelesTrak's own historical account
(celestrak.org/events/collision/) - including that SOCRATES genuinely
predicted this exact conjunction in all 14 reports issued that week
(range 117m-1.812km across those reports) but it ranked only #152
overall in the final report and was never prioritized or acted on - a
real, documented TRIAGE failure, not a detection failure (directly on-theme
for "Triage in Light Speed"). NORAD catalog numbers (Iridium 33: 24946,
Cosmos 2251: 22675), the ~11.7 km/s relative velocity, and the ~789 km
collision altitude were independently verified via NASA/Wikipedia
sources, not taken from a single origin.
TelemetryEvent.source="historical-replay" so it's never mistaken for
live data anywhere downstream (audit log, dashboard table, display) -
raw_data also carries historical_event/historical_source/
historical_actual_outcome fields so the citation and real-world outcome
travel with the record itself, not just in documentation. Wired into
scripts/run_demo.py as a new step (prints the full citation/outcome in
its own panel before showing the system's response) and
scripts/dashboard.py as a new sidebar button - both go through the real
src/pipeline.run_once, nothing bypassed. run_id-based event ids follow
the same uniqueness convention as SyntheticCriticalAdapter.
5 new tests: the real documented values themselves (NORAD ids, distance,
velocity, timestamp), event-id uniqueness across runs, limit handling,
and - the core claim of this phase - an integration test proving the
real 584m number classifies as CRITICAL and produces a verified maneuver
through the actual analyze_node/decide_node, not a mocked shortcut.
Suite: 116/116. Live-verified against real local Ollama twice: a direct
analyze_node/decide_node call (confirmed CRITICAL classification, a real
Gemma GO verdict with real reasoning referencing the actual 41.7km
verified clearance) and a full scripts/run_demo.py --auto run
end-to-end (confirmed correct integration into the 9-step walkthrough,
correct summary totals). One honest, minor imperfection noted rather
than hidden: Gemma's own narration describes the event in present/future
tense ("a close encounter is predicted...") since neither analyze_node's
nor decide_node's prompts are historical-replay-aware - the surrounding
demo step's own panel (title, citation, real outcome) and the persisted
source="historical-replay" field both make the historical framing clear
regardless, so this wasn't judged worth adding replay-specific prompt
branching for.

## QA / gap-analysis / fresh-eyes review pass
Status: done
Every phase (0-12) was already "done" - this was a deliberate second pass
treating that as a starting point, not a conclusion: full manual QA (live
demo runs on both backends, live dashboard click-through), a gap analysis
against docs and repo hygiene, and an independent fresh-eyes code review
by an agent with no memory of this project's history, briefed only on
what the system is supposed to do. Found and fixed 7 real issues,
verified 4 more were clean.

**1. `verify_maneuver` was a mathematical tautology, not an independent
check (found by the fresh-eyes review).** `compute_avoidance_maneuver`
solves `delta_v` from `target_clearance_km - min_distance_km`;
`verify_maneuver` recomputed `new_min_distance_km` from that same
`delta_v` - the algebraic inverse of the original solve. Given the same
`min_distance_km` (which every real call site always passes), the two are
*mathematically guaranteed* to agree - `cleared` could never be `False`
for any CRITICAL-range input, since `target_clearance_km` is always
`>= 30km`, comfortably above the 5km threshold. This directly contradicted
repeated "independently re-verified, not just trusting the number it was
solved for" language across README/PROJECT_OVERVIEW/KAGGLE_WRITEUP, and
mattered more given Phase 9 feeds this "verified" number to Gemma's veto
prompt as evidence of safety. Fixed with a check that's actually
independent: `MAX_PLAUSIBLE_DELTA_V_M_S` (50 m/s) - a real-world
plausibility bound the original solve never touched, so it genuinely can
fail (e.g. a corrupted/nonsensical `min_distance_km`), unlike the
recompute (kept - it's still a real regression guard against the two
formulas drifting apart after a future edit, just not independent proof
of safety on its own). `verify_maneuver`'s docstring, the test that
asserted the tautology without naming it as one, and all three docs'
"independently re-verified" language were all corrected to be honest
about which part is which. 2 new tests (implausible delta-v, nonpositive
distance both now correctly fail).

**2. `CelesTrakAdapter` event_ids could collide across separate real
scans (found by the fresh-eyes review).** Unlike its two siblings
(`SyntheticCriticalAdapter`, `HistoricalReplayAdapter` - both fixed for
this during their own phases), `CelesTrakAdapter`'s event_id was derived
only from the object pair (`conj-{a}-{b}`), with no per-scan uniqueness.
The same real object pair very plausibly stays the closest pair across
two scans within the same 1h TLE cache window (e.g. two dashboard clicks
in a row) - and `DecisionLogger.find_entry`/`mark_reviewed`/
`approve_maneuver` all match an event_id's FIRST logged occurrence, so a
second scan's genuinely-pending entry could be indistinguishable from an
already-resolved earlier one, silently misdirecting an Approve/Reject
click to the wrong entry. Fixed with the same `run_id` treatment as its
siblings - auto-generated per adapter instance if not given explicitly,
so every caller gets the fix by default with no code changes required at
the call sites. 2 new tests.

**3. Gemma's veto verdict could be misread from a negated later mention
(found by the fresh-eyes review).** `_parse_veto_verdict` took the LAST
GO/NO-GO token found anywhere in the response. A response that correctly
leads with "GO" (as instructed) but later reasons "...this is clearly
not a NO-GO situation, so proceed" would have that later "NO-GO"
substring win, misreading an affirmed maneuver as vetoed - undercutting
the veto prompt's own instruction not to second-guess a maneuver the
numbers already show is safe. Fixed by checking the FIRST token
authoritatively (the prompt instructs the verdict to come first) before
falling back to the existing last-match scan for responses that don't
lead with a clear token. 2 new regression tests.

**4. Fallback provenance reported the wrong model for the api backend
(found by the fresh-eyes review).** `_call_gemma_with_provenance`'s
except-branch unconditionally set `model_used = client.settings.gemma_model`
(the Ollama tag) even when `GEMMA_BACKEND=api` and both backends had
failed - a cloud-only deployment's fallback log entries would claim
"gemma4:e4b" responded when nothing running that tag was ever involved.
Fixed to report `gemma_model_api` when the configured backend is "api".
1 new test.

**5. Stale field comment (found by the fresh-eyes review).**
`ManeuverApproval.approved_by`'s inline comment still said "None for
autonomous approvals" - no longer true since Phase 9, where an autonomous
Gemma veto sets `approved_by="Gemma (autonomous safety review)"`.
Corrected.

**6. Two hardcoded 15s Gemma timeouts were too short for real hosted-API
latency (found by manual QA, not the fresh-eyes review - it correctly
avoided making real network calls).** Running the full cloud-backend demo
live surfaced `preflight.check_gemma_reachable` and
`run_demo._step_failover` both using `timeout=15`. Measured directly: the
same trivial prompt against the real hosted API took anywhere from ~2s to
35s+ across repeated calls during this session - a real key, correctly
configured, with a 15s timeout would intermittently and misleadingly
report "unreachable." Bumped both to 45s. Not a correctness bug in the
sense of wrong output, but a real flakiness/false-negative risk a judge
running this demo could hit through no fault of their own setup.

**7. `GemmaClient.generate()` silently discarded the real reason a
fallback attempt failed (found chasing #6 live).** When both backends
failed, the code raised only the PRIMARY's error and discarded the
fallback's via a bare `except GemmaClientError: pass`. In practice the
primary is often *expected* to be down (e.g. this project's own failover
demo step deliberately breaks it) - so it's almost always the fallback's
error that actually explains a real failure, and it was invisible. Caught
live: a real run showed "Both backends unreachable" with only the
intentionally-broken local connection's error visible; the actual cause
(a transient 500 from the hosted API) was silently dropped. Fixed to
raise a combined error naming both backends' actual failures. 1 test
strengthened into an explicit regression test for this exact behavior.

**Verified clean, not just assumed:** no TODO/FIXME/XXX anywhere in
src/scripts/tests; pyflakes reports zero unused imports or undefined
names across the whole codebase; every file path referenced across all 5
docs actually exists; `.env.example` exactly matches every env var
`config.py` actually reads (no more, no less); no secrets anywhere in
git history (re-confirmed); `NOTES_FOR_MORNING.md` untouched this
session, consistent with it being an intentionally-preserved historical
record; LICENSE unmodified.

Full local-backend and cloud-backend `scripts/run_demo.py --auto` runs
were executed live end-to-end as part of this pass (not just unit
tests) - the cloud run is what surfaced #6 and #7. The dashboard was
re-launched and its Approve, Reject, and "Replay historical event"
controls were each exercised for real against the live accumulated audit
log, with the resulting writes confirmed directly against the raw log
file afterward, not just trusted from the UI. Suite: 123/123.

## Phase 13 — Visual orbit plot
Status: done
Added src/orbit_plot_data.py + a new "Orbit plot" section in the
dashboard's inspect panel: for any celestrak-sourced event, re-fetches
both objects' CURRENT TLEs by NORAD catalog number and re-propagates
using the exact same coarse-pass functions (orbital.build_coarse_times,
orbital.compute_coarse_positions) the real screening pipeline already
uses - no duplicated physics. Renders two real Plotly charts: a 3D view
(Earth to scale, both objects' actual propagated paths over the next 48h,
a marker at the closest-approach point) and a distance-vs-time line chart
with this project's own severity thresholds (5/25/100km) drawn as
reference lines, so it's visually obvious when/whether the real curve
crosses into risk territory.
Deliberately scoped to celestrak-sourced events only, not synthetic or
historical ones - synthetic fixtures have no real orbital elements at
all, and (as Phase 12 already established) CelesTrak's public API only
ever serves CURRENT TLEs, not archival ones, so a historical event has no
real trajectory to re-propagate either. The panel says so explicitly
for both cases rather than silently doing nothing or, worse, fabricating
a plausible-looking plot from synthetic numbers.
Honest caveat surfaced in the UI itself, not just here: since this
re-fetches CURRENT TLEs rather than the exact TLE epoch that produced the
originally-logged min_distance_km, the propagation window starts from
now - orbital elements update over time, so the plotted closest approach
can differ from what was logged when the decision was first made. This
is a live recomputation, not a replay of the original numbers.
6 new tests (TLE-by-catalog-number fetch/parse using the same real
Vanguard 1/ISS fixtures test_orbital.py already uses, real propagation
end-to-end, and figure-structure checks for both chart types) - dashboard
wiring itself covered by the existing AppTest smoke tests (still pass
unchanged with the new "Orbit plot" section added). Suite: 128/128.
Live-verified in a real browser: selected a real celestrak-sourced event
(COSMOS 2251 DEB vs COSMOS 2251 DEB), clicked "Generate orbit plot", and
confirmed a real interactive 3D Plotly chart rendered - Earth, both
objects' actual propagated paths, the closest-approach marker - and that
rotating it (a real drag interaction, not scripted) revealed genuine
elliptical orbital paths, not placeholder geometry. The distance-vs-time
chart's structure was verified via an isolated live smoke test (real
fetch + real propagation, outside the browser) plus its dedicated unit
tests, rather than fighting this environment's flaky browser-scroll
behavior on an already-proven code path.

## Phase 14 — A second real hazard type: orbital decay / re-entry risk
Status: done
Every prior phase was conjunction-specific, even though schemas.py's own
docstring always said the telemetry/finding/decision shapes were
"intentionally idea-agnostic." This phase makes that real: a second,
independently real hazard type - orbital decay/re-entry risk - screened
through the exact same analyze_node -> decide_node -> log_node pipeline,
using data this project already fetches (no new data source, no new
credentials).
Verified feasibility before designing anything: confirmed directly that
Skyfield's SGP4 model (the same EarthSatellite object already used for
conjunction propagation) exposes real, already-parsed perigee/apogee
altitude (`sat.model.altp`/`alta`) and the BSTAR drag term
(`sat.model.bstar`) - checked against ISS's real current TLE and got
~414km, the correct real ISS altitude. This meant src/decay.py needed no
separate TLE-column parser at all - just reads values Skyfield already
computed.
Added:
- src/ingestion/tle_source.py: extracted CelesTrakAdapter's fetch-with-
  disk-caching and TLE-block-parsing logic into shared module-level
  functions (fetch_tle_group_text, parse_tle_blocks), refactored
  CelesTrakAdapter to use them (behavior-preserving - all 8 of its
  existing tests pass unchanged after just retargeting their
  `requests.get` mock path). A genuine second real consumer
  (DecayRiskAdapter) justified this promotion, matching the same
  reasoning Phase 11 used for SyntheticCriticalAdapter.
- src/decay.py: assess_decay_risk() builds a raw_data dict from
  Skyfield's own orbital elements. Explicitly NOT a real atmospheric-
  drag/decay-rate model (needs solar flux + atmospheric density tables) -
  perigee altitude alone is the signal, since it's real, well-established,
  uncontested orbital mechanics on its own (an object with a perigee
  below ~200km reliably reenters within days to weeks regardless of other
  factors), not a precise reentry-time predictor. Same "simplified,
  clearly labeled as simplified, not flight software" spirit as
  src/maneuver.py.
- src/ingestion/decay_adapter.py: DecayRiskAdapter screens objects
  INDIVIDUALLY (not pairs) from a real CelesTrak group (default:
  cosmos-2251-debris, the same real debris field CelesTrakAdapter already
  uses for conjunctions), ranked by ascending perigee (most at-risk
  first). source="celestrak-decay" (not "celestrak") so it's always
  distinguishable downstream from conjunction-pair data without needing
  to inspect raw_data shape.
- pipeline.py: classify_decay_severity (same threshold-check design as
  classify_conjunction_severity - 200/300/500km bands), _decay_description
  (a new Gemma prompt), analyze_node gained a third branch
  (conjunction-shaped / decay-shaped / generic-placeholder, keyed off
  which raw_data fields are present - same pattern already used to tell
  conjunctions from DummyAdapter's generic payload). decide_node's
  maneuver-computation block is now gated on `"object_a_id" in raw`, not
  severity alone - a CRITICAL decay finding gets a real deterministic
  action (ABORT) and real Gemma narration through the existing generic
  rationale branch, but deliberately NO maneuver/budget/veto/approval
  machinery: an avoidance burn doesn't mean anything for "your perigee is
  too low," and building a second simplified-maneuver model (a reboost/
  deorbit planner, with its own budget and approval semantics) was
  explicitly scoped OUT of this phase, not overlooked - a real, separate
  problem for a future phase if ever needed. The HOLD-action instruction
  in _decision_rationale is now hazard-aware too, so a decay WARNING isn't
  told to suggest a conjunction-flavored "along-track burn."
- display.py / dashboard_data.py: both gained a decay-aware subject-line
  branch (object_name + perigee, not a fallback to the raw event_id) -
  same fix applied in both places since they duplicate this formatting
  logic for their respective outputs (terminal vs. table).
- scripts/run_demo.py (new step) and scripts/dashboard.py (new sidebar
  button) both wired to the real pipeline, nothing bypassed.
19 new tests: decay.py's real-data assessment (2, using the same real
Vanguard 1/ISS fixtures test_orbital.py already uses), tle_source.py
directly (5), DecayRiskAdapter (4), pipeline.py's new classification/
analyze/decide branches (8, including a dedicated test proving the
CRITICAL-decay guard doesn't KeyError trying to read object_a_id off
decay-shaped raw_data), plus small display.py/dashboard_data.py coverage
for the new subject-line branch. Suite: 149/149.
Live-verified end-to-end against real local Ollama three times: a normal
real decay screen (WATCH classification, accurate narration that
correctly avoided stating a specific reentry date as instructed), a
directly-constructed CRITICAL-range decay scenario (confirmed ABORT
action, no maneuver_plan, sensible real narration with no
conjunction-flavored language leaking in), and a full
scripts/run_demo.py --auto run (confirmed correct integration into the
10-step walkthrough and correct summary totals). Also live-verified via
the dashboard: clicked "Screen for orbital decay risk" for real, and
confirmed via the raw log file - not just the UI - that exactly 5 new
real entries appeared with a fresh run_id, matching the button's own
limit.
One honest finding surfaced rather than tuned around: with the default
group (cosmos-2251-debris) and these thresholds, real data currently
tops out at WATCH (lowest real perigee found across the full 598-object
group: ~313km) - the lowest-perigee fragments from the 2009 breakup have
already decayed away over the intervening years, so WARNING/CRITICAL
don't occur "on demand" from this specific real source right now. Same
"real data rarely produces the most severe case live" pattern already
established and accepted for CRITICAL conjunctions (see Phase 5) - no
synthetic decay fixture was built to paper over this, unlike conjunctions
where one was, to keep this phase's scope controlled; CRITICAL/WARNING
classification is still directly covered by dedicated unit tests either way.

## Phase 15 — Live satellite tracking view
Status: done
A repo-hygiene phase (CI workflow, .gitignore cleanup, linking the
existing LICENSE) landed between Phase 14 and this one but wasn't given
its own numbered entry here at the user's request - it's not a pipeline
feature, so PROJECT_OVERVIEW.md covers it instead of this file.
This phase adds the first dashboard view that answers "where are the
real assets right now" rather than "what did a triage run find." Every
prior visualization (orbit plot, decision table) is event-driven - it
only shows objects that were screened and produced a logged finding.
Prompted by the user asking whether the UI showed live catalog status at
all; agreed scope explicitly in conversation before building anything:
NOT an attempt at rendering CelesTrak's full ~20,000-object public
catalog (that's a different, unbounded feature), just the same
`stations` group (real crewed stations - ISS, Tiangong - plus their
currently-docked visiting vehicles) CelesTrakAdapter already treats as
the actual payload worth protecting - a small, named, recognizable set
that reads as real, not an anonymous debris point-cloud.
Added:
- src/live_positions.py: fetch_live_positions() reuses
  src/ingestion/tle_source.py's fetch/cache/parse (no new fetch logic)
  to get real current TLEs for the `stations` group, then computes each
  object's real current position with ONE Skyfield/SGP4 evaluation at
  ts.now() - the exact same Earth-centered (GCRS) frame and km units
  orbital.py's compute_coarse_positions already uses, just evaluated at a
  single instant instead of propagated forward. No new physics.
  build_live_globe_figure() renders it on the same plain-sphere Earth
  style orbit_plot_data.build_3d_trajectory_figure already established,
  for visual consistency between the two 3D views.
- scripts/dashboard.py: new "Live tracking: real crewed stations, right
  now" section, button-gated (a real network call on every click, same
  pattern as every other live-activity button) rather than fetched on
  every page load.
Deliberately NOT wired into scripts/run_demo.py or the audit log: unlike
every other data source in this project, a live position snapshot
produces no TelemetryEvent/AnomalyFinding/Decision - there's no triage
question being answered ("is this a risk?"), just a real, honest picture
of where things are. Forcing it through the analyze/decide/log pipeline
to make it "consistent" with the rest of the demo would have meant
inventing a finding/decision for data that doesn't have one - dashboard-
only is the honest scope, not an oversight.
3 new tests (tests/test_live_positions.py): real position computation
from real, fixed TLE fixtures (same ISS ZARYA / Vanguard 1 fixtures used
elsewhere in this suite) with the network call mocked, the disk-cache
reuse behavior, and the figure's trace/marker structure. Suite: 152/152.
Live-verified in a real browser: clicked "Show live positions" against
the real running dashboard, confirmed a real network fetch (visible in
the spinner + real elapsed time, not instant) returned 21 real objects
from CelesTrak's live `stations` group, rendered as labeled markers
(including real names like "KNACKSAT-2") on the 3D globe.
