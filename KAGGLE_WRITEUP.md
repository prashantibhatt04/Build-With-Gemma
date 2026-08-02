# Deep Space Navigation — Orbital Collision Avoidance with Gemma

*Track 2 — Build with Gemma: Triage in Light Speed*

## Overview

Low Earth orbit is crowded with tracked satellites and debris, and close
approaches ("conjunctions") happen constantly. A mission controller can't
manually evaluate every one in real time — but an automated system making
opaque or unreliable calls in this domain is genuinely dangerous. This
project pulls real satellite tracking data, runs real orbital mechanics to
predict close approaches, classifies risk with deterministic physics-based
thresholds, and uses Gemma to turn those findings into plain-language
explanations a human can actually act on — with a human-approval workflow
for the most severe cases when a human is actually reachable to give one.
A live browser dashboard (`streamlit run scripts/dashboard.py`) puts that
approval workflow - and the full decision history, plus a real 3D orbit
plot for any live conjunction - in front of a real person instead of only
a terminal. And it isn't only tested against live or synthetic data:
replaying the real, documented 2009 Iridium 33/Cosmos 2251 collision -
the first confirmed accidental collision between two intact satellites -
through this system's unmodified pipeline shows it would have classified
the real 584m SOCRATES prediction as CRITICAL, against a real event where
that same warning existed but was never acted on. A triage failure, not a
detection failure - which is exactly the problem this track is named for.
And it isn't conjunction-only either: a second, independently real hazard
type - orbital decay/re-entry risk, using real orbital elements already
parsed from the same TLE data - runs through the exact same pipeline,
proving the "idea-agnostic" design `schemas.py` always claimed.

## Architecture

```
CelesTrak (live TLE data, multiple real groups: stations + debris)
        │
        ├──▶ CelesTrakAdapter — cross-group PAIRWISE screening, Skyfield/
        │     SGP4 propagation, two-pass coarse/fine search (cached coarse
        │     pass + top-K refinement for scale) -> conjunction risk
        │
        └──▶ DecayRiskAdapter — per-OBJECT screening using Skyfield's own
              already-parsed perigee altitude + BSTAR -> decay/re-entry risk
        │
        ▼
analyze_node — deterministic severity classification (distance thresholds
             for conjunctions, perigee-altitude bands for decay risk)
             — Gemma: plain-language description of the finding
        │
        ▼
decide_node — deterministic action mapping (severity -> action)
            — for CRITICAL conjunctions only: deterministic maneuver calc
              + independent verification + delta-v budget check (decay
              CRITICAL gets a real action + narration, no maneuver -
              conjunction-specific scope, not overlooked)
            — local backend -> Gemma's own GO/NO-GO veto check, then
              self-approve (bounded: can only veto, never approve
              what physics hasn't already cleared)
            — cloud backend -> held pending, awaits human approval
            — Gemma: never computes the physics; explains the current
              state everywhere, and additionally renders the local-path
              veto verdict itself
        │
        ▼
log_node — append-only JSON-lines audit log (every decision, its
           provenance, and - for CRITICAL - its approval history)
```

Built on **LangGraph** for the `analyze -> decide -> log` pipeline,
**Pydantic** for schema validation throughout, **Skyfield/SGP4** for real
orbital mechanics, and a backend-agnostic **GemmaClient** that talks to
either a local Ollama instance (`gemma4:e4b`) or a hosted Gemini-style API
(`gemma-4-26b-a4b-it`) through the same interface, with automatic failover
between them.

Key modules: `src/orbital.py` (TLE parsing, closest-approach search),
`src/ingestion/tle_source.py` (shared CelesTrak fetch + cache + TLE-block
parsing, used by both hazard adapters below),
`src/ingestion/celestrak_adapter.py` (real multi-group CelesTrak fetch +
scalable cross-group screening for conjunctions), `src/decay.py`
(pulls real perigee/apogee altitude and BSTAR straight out of Skyfield's
own SGP4 model - no separate parser needed), `src/ingestion/decay_adapter.py`
(screens a real CelesTrak debris group for low-perigee objects and ranks
them by decay risk), `src/maneuver.py` (maneuver math, independent
verification, delta-v budget), `src/gemma_client.py` (backend-agnostic
Gemma access + failover), `src/pipeline.py` (the LangGraph nodes, now
branching on raw-data shape to classify either conjunction or decay
risk), `src/logging_utils.py` (audit log, human-review, and
maneuver-approval workflows), `src/display.py` (terminal rendering),
`src/preflight.py` (environment health checks), `scripts/run_demo.py`
(the guided end-to-end demo), `scripts/dashboard.py` +
`src/dashboard_data.py` (the live browser dashboard - UI wiring and its
Streamlit-free data logic kept in separate files specifically so the
latter stays directly unit-testable), `src/ingestion/historical_adapter.py`
(replays a real, documented past conjunction through the same unmodified
pipeline), and `src/orbit_plot_data.py` (real 3D trajectory + distance-
over-time Plotly charts, built by re-propagating live TLE data with the
same physics `src/orbital.py` already uses).

## How Gemma was used

Gemma is used as an **explainer** everywhere in this system, with one
narrow, deliberately bounded exception where it acts as a real,
fail-safe-first decision-maker:

- **Finding descriptions** (`analyze_node`): turns raw conjunction numbers
  (objects, distance, relative velocity, time of closest approach) into a
  1-2 sentence plain-language summary for a mission controller.
- **Decision rationale** (`decide_node`): explains a deterministically
  already-decided action. The exact instruction changes by state — a
  routine "continue/hold" recommendation, a completed autonomous maneuver
  narrated in past tense, a maneuver *vetoed* by Gemma's own safety
  review, a maneuver *proposed and awaiting human approval* narrated as
  not-yet-executed, or a maneuver that couldn't be afforded, escalated for
  review. Gemma is explicitly instructed never to hedge on, second-guess,
  or contradict the severity/action it's given.
- **An autonomous maneuver veto-check** (`_maneuver_veto_check`, local
  backend only): this is the one place Gemma genuinely decides something,
  not just narrates it — but it's tightly bounded. A deterministic physics
  check independently verifies a CRITICAL maneuver is safe first; only
  then does Gemma get those same numbers and issue a real GO/NO-GO
  verdict, standing in for the unavailable human. Gemma can only make the
  outcome *more* conservative than the physics check, never less: it can
  veto an already-verified-safe maneuver, but it can never approve one the
  physics hasn't already cleared. Two fail-safe defaults matter here: an
  unparseable verdict defaults to NO-GO (escalate, not a free pass), while
  Gemma being *unreachable* is explicitly NOT treated as a veto — an LLM
  outage alone shouldn't block a maneuver already proven safe, so that
  case still executes autonomously on the physics check alone.
- **Cross-backend by design, with three tiers of resilience**: the same
  prompts run against either the local Ollama model (`gemma4:e4b`) or the
  hosted API model (`gemma-4-26b-a4b-it`), chosen by a `GEMMA_BACKEND` env
  var. A single call first retries once on the same backend (transient
  hiccups shouldn't force escalation); if that still fails, `GemmaClient`
  fails over to the *other* backend and retries there; if both are
  unreachable, the pipeline falls through to a deterministic templated
  sentence with no AI involved, so a finding is never silently dropped.
  Every log entry records exactly which tier actually produced the text
  (`GemmaProvenance.source`: `"gemma"` or `"fallback"`, plus `model_used`
  naming the real backend/model that responded) — never silently hidden.
- **A deliberate product decision, not just infrastructure**: which
  backend is configured is used as a stand-in for whether ground control
  is currently reachable. Local (Ollama) -> a CRITICAL maneuver goes
  through Gemma's autonomous safety review above, because a real probe
  often can't wait for a human (light-delay, communication blackout).
  Cloud (hosted API) -> ground control is reachable, so the same maneuver
  is calculated but held pending a real human's explicit approval or
  rejection before it executes.

## Engineering hurdles overcome

**Hosted model chain-of-thought leaking into user-facing output.** The
hosted Gemma model would return its full reasoning trace - draft
attempts, self-checklists, "wait, let me reconsider" - instead of a clean
final answer, unlike the local model which responded cleanly. Fixed with
a post-processing step (`_extract_final_answer`) applied uniformly to
both backends. This took two iterations to get right: the first version
took the last blank-line-separated paragraph, which missed cases where
the model separated a label from its real answer with only a single
newline. The second, more serious bug was caught only by running the full
demo live and repeatedly against the real hosted API (not just unit
tests): a response's literal last line was occasionally a short throwaway
remark like "Ok, let's go." with the real content one line above it. The
final version searches backward for the last line that's actually
substantive (length-based heuristic) rather than trusting positional
"last line" alone.

**API key exposure via URL query parameter.** The hosted API call
originally embedded the API key directly in the request URL
(`?key=...`). When a request failed during testing, the resulting error
message - which included the full URL - got surfaced in a terminal,
exposing the raw key. Root-caused and fixed by switching to header-based
auth (`x-goog-api-key`), which Google's API supports specifically to
avoid this failure mode; verified the fix by forcing a failure and
confirming the key no longer appears anywhere in the resulting error text.

**Local and hosted model naming are not interchangeable.** The Ollama tag
used locally (`gemma4:e4b`) isn't a valid model id on the hosted API,
which uses an entirely different naming scheme. Attempting to reuse a
single config field for both silently failed. Resolved by querying the
hosted API's own models-list endpoint to discover which model ids the
account's key actually had access to, then splitting configuration into
separate `GEMMA_MODEL` (local) and `GEMMA_MODEL_API` (cloud) fields, with
provenance logging that names the correct model regardless of which
backend actually answered.

**An unanticipated new system state crashed the renderer.** Adding the
human-approval workflow introduced a third possible maneuver state
("awaiting approval": not yet executed, not blocked by budget) that the
terminal display code hadn't been written to expect - it assumed a
maneuver was always either budget-blocked or already verified. This
wasn't caught by the existing unit test suite; it only surfaced when
running the full pipeline live end-to-end against the real cloud backend.
Fixed by rewriting the renderer to explicitly handle every real state
(budget-blocked / awaiting-approval / autonomous-executed /
human-approved-executed / rejected) instead of assuming only two, with
regression tests added for the exact failure.

**Repeated demo runs silently updated the wrong log entry.** The guided
demo script used a fixed, hardcoded event id for its synthetic scenario
(`conj-run-demo-0`, `-1`, etc.). The audit log's `mark_reviewed`/
`approve_maneuver` operations match an entry by its first occurrence of a
given event id — so on a *second* run of the demo, marking "this run's"
event as reviewed silently updated a stale entry from an *earlier* run
instead, while the entry that was actually just created stayed untouched.
Not caught by unit tests (each test uses an isolated temp directory with
no repeat-run collisions); only surfaced by actually running the demo
multiple times in a row, the way a presenter rehearsing for a live demo
naturally would. Fixed by including a per-run unique id in the synthetic
event ids, and re-verified correct across three consecutive real runs.

**Naive pairwise screening didn't scale past a handful of real objects.**
The original CelesTrak adapter screened one small group in isolation by
calling the two-pass closest-approach search independently for every
pair. Benchmarked (not assumed) before optimizing: 1770 pairs from a real
60-object sample took ~9.6s - already too slow for a live demo, and
quadratically worse for a meaningfully larger real pool. Root cause: each
pair recomputed its own coarse-pass orbital propagation from scratch, so
cost scaled with the number of *pairs*, not the number of *objects*.
Fixed by caching each object's coarse-pass position once and reusing it
across every pair it appears in (cut the same 1770-pair screen to ~0.02s
in isolated benchmarking), then only running the expensive precise
refinement on the closest candidates by that cached estimate - sound
because a coarse pass can only ever overestimate the true closest
approach, never underestimate it. Two further issues surfaced only by
actually running the resulting cross-group screen live against real
CelesTrak data, not unit tests: real crewed-station groups include
currently-docked vehicles sitting at ~0km apart (real physics, not a
collision risk, but it dominated the top results); and a dense
single-origin debris field could fill the entire refinement budget on its
own, leaving zero cross-group ("asset vs. debris") results even though
that's the actual point. Both fixed and verified live - see
`PHASE_PROGRESS.md` Phase 10 for the full detail.

**No public, free API for historical TLE data - so don't fake it.** For
the historical-replay backtest, the obvious approach would be pulling the
actual Iridium 33/Cosmos 2251 TLEs from around 2009-02-10 and propagating
them with the same Skyfield/SGP4 code used elsewhere. Checked directly
rather than assumed: querying CelesTrak's public `gp.php` endpoint with a
historical `EPOCH` parameter silently ignores it and returns today's TLE
regardless - genuine historical archives require Space-Track.org, which
needs a real account this project doesn't have and can't create on a
user's behalf. Rather than fake historical propagation with current-day
TLEs relabeled as 2009 data (which would misrepresent what's actually
being computed), the replay instead uses the real, independently-sourced
closest-approach number CelesTrak's own SOCRATES system actually reported
at the time (584m) - honest about being a documented-record replay, not
a re-derived one, and clearly labeled as such (`source=
"historical-replay"`) everywhere it surfaces.

## Design choices

- **Severity, action, and maneuver physics are 100% deterministic
  thresholds and closed-form math - never AI output.** In a
  collision-avoidance context, reliability has to come first; Gemma's
  primary value is making already-reliable decisions understandable, not
  making them. The one deliberate exception - the local-path veto check -
  is designed so Gemma can only ever narrow what's already been verified
  safe, never widen it: it can say no to a maneuver the physics cleared,
  it can never say yes to one the physics didn't.
- **A maneuver is re-verified, not just trusted - and that verification
  is honest about what it can and can't catch.** The re-check re-derives
  the resulting clearance forward from the computed delta-v, rather than
  echoing the number the plan was algebraically solved for - but a QA
  pass found that recompute is the algebraic inverse of the original
  solve, so on its own it's mathematically guaranteed to agree (a
  regression guard against the two formulas drifting apart, not
  independent proof of safety). Fixed by adding a genuinely independent
  plausibility bound (`MAX_PLAUSIBLE_DELTA_V_M_S`) that uses information
  the original solve never touched, so it actually can fail - see
  `PHASE_PROGRESS.md`'s QA pass entry.
- **A limited delta-v budget prevents unlimited silent autonomy.** Even
  in the fully autonomous (local) path, the system can't execute an
  unbounded number of maneuvers - when the budget runs out, it says so
  explicitly and escalates, rather than continuing to act.
- **Every decision is auditable after the fact.** An append-only
  JSON-lines log records the raw data, the deterministic finding, the
  full decision, and explicit provenance (real Gemma output vs.
  deterministic fallback, which backend, and - for CRITICAL cases -
  whether a maneuver was autonomous, human-approved, vetoed by Gemma,
  rejected by a human, or blocked by budget) for every single event,
  reconstructable independently of this system.
- **Confidence is a real signal, not a hardcoded placeholder.** Early on,
  `AnomalyFinding.confidence` was a flat constant everywhere. It's now
  derived from how stale the underlying TLE tracking data actually is
  (epoch age), with a clearly-labeled placeholder used only for telemetry
  shapes that genuinely have no real signal to derive it from - honest
  about what's real and what isn't, rather than a number that only looks
  computed.
- **A second UI must not be able to disagree with the first.** Adding the
  dashboard meant a second surface (Streamlit) now had to interpret the
  same six mutually-exclusive maneuver states the terminal renderer
  already handled. Rather than reimplementing that branching, it was
  extracted into one function (`classify_decision_status`) that both
  consume - the two views are structurally incapable of drifting out of
  sync, instead of relying on remembering to update both whenever a new
  state is added (the way Phase 8 originally required, and the way
  Phase 9's veto state was, in hindsight, an opportunity to do this
  extraction sooner).
- **The schemas were built hazard-agnostic from the start, and Phase 14
  is where that actually got tested.** `TelemetryEvent`/`AnomalyFinding`/
  `Decision` never assumed "conjunction" - they were documented as
  idea-agnostic from early on, but nothing had exercised that claim until
  a second real hazard type (orbital decay/re-entry risk) was added.
  `analyze_node`/`decide_node` branch on the *shape* of `raw_data`
  (`min_distance_km`+`object_a_id` vs. `perigee_altitude_km`+`object_id`)
  rather than a hardcoded hazard-type field, and no existing conjunction
  code path needed to change - a real test of whether the "idea-agnostic"
  design was actually true, not just asserted.

## Future work

Every phase originally scoped for this submission is now built - the
autonomous veto-gate, real cross-group catalog screening, a live
dashboard, the historical replay, a real 3D orbit visualization, and a
second real hazard type (orbital decay/re-entry risk, screened from a
real CelesTrak debris group using Skyfield's own SGP4 model) - plus a
full QA/gap-analysis/fresh-eyes review pass afterward. Decay risk
currently stops at NOMINAL/WATCH classification by design: real-world
severity depends on the deterministic thresholds alone, and the maneuver/
budget/veto machinery stays conjunction-specific since re-entry isn't
avoided with a delta-v burn the way a conjunction is. Extending
human-approval-style workflows to decay events (e.g. re-entry monitoring
tasks) remains an open idea, not a committed next step.

## Verification

149 automated tests (network-free, Gemma calls mocked, CelesTrak network
calls mocked, the dashboard tested via Streamlit's own AppTest harness)
cover orbital math (including the decomposed coarse/fine search used for
scalable screening), severity/confidence derivation, maneuver math
(including the plausibility bound added by the QA pass), budget tracking,
Gemma retry/fallback logic, the autonomous maneuver veto-check (including
its fail-safe defaults and a negation-parsing regression case), cross-group
conjunction screening, decay-risk classification and screening (including
real Vanguard 1/ISS TLE fixtures exercising Skyfield's own perigee/apogee/
BSTAR fields), the dashboard's data transforms and UI wiring, the
historical replay (including an integration test proving the real 584m
number classifies as CRITICAL through the actual pipeline), the orbit
plot's real TLE fetch/propagation and figure structure, the full pipeline,
and the human-approval/review workflows. Beyond unit tests, every major
path in this writeup was also run live end-to-end against a real local
Ollama instance, a real hosted API key, and real live CelesTrak data
during development - which is how several of the issues described above
were actually found, including two specific to real cross-group
screening (docked-vehicle noise, a dense debris field crowding out
cross-group results) and two more from a dedicated QA pass (hardcoded
Gemma timeouts too short for real hosted-API latency, and a fallback
error silently discarded). The dashboard specifically was verified in a
real browser against the real accumulated audit log, including clicking
a real Approve button and confirming via the raw log file afterward that
it actually executed, rotating the real 3D orbit plot to confirm it
showed genuine elliptical paths, not placeholder geometry, and clicking
the real decay-risk screening button to confirm live CelesTrak debris
data flows through to real Gemma narration. The historical
replay was verified live twice: directly (confirming CRITICAL
classification and a real Gemma GO verdict referencing the actual
verified clearance) and through a full guided-demo run (confirming
correct integration and summary totals). The decay hazard type was also
verified live end-to-end, including the honest finding that the real
`cosmos-2251-debris` group's lowest perigee (~313km) currently only
reaches WATCH severity, not CRITICAL - documented openly rather than
tuning thresholds to force a more dramatic result on demand.

See the [public repository](https://github.com/prashantibhatt04/Build-With-Gemma)
for full source, `PROJECT_OVERVIEW.md` for a diagram-based walkthrough,
`DEMO.md` for exact reproduction steps, and `PHASE_PROGRESS.md` for the
complete build history.
