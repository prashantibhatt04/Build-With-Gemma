# Live Demo Walkthrough

Run these in order from the repo root, with the venv active:

```bash
source .venv/bin/activate
```

Each stage below says what it proves and what to look for in the output.
Nothing here touches LICENSE, TRACKS.md, or git - it's all just running
the code that's already built (see `PHASE_PROGRESS.md` for the full
phase-by-phase history).

---

## Setup on a new machine

`.env` (your real API key) and `.venv/` are both gitignored and never get
pushed - `git log --all --full-history -- .env` on this repo returns
nothing, confirming the key was never in history. On the new box, after
`git clone` (or `git pull` if already cloned):

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Then edit `.env` on that machine and set `GEMMA_API_KEY=<your key>`.

**Important:** if that machine won't have Ollama running, also change
`GEMMA_BACKEND=ollama` to `GEMMA_BACKEND=api` in the same `.env` file. Local
(`ollama`) stays the default - if you leave it as-is with no Ollama
running, every Gemma call will still work (Phase 4's cross-backend
fallback catches the failure and falls through to the cloud key
automatically), but each call wastes a failed local attempt + retry first,
adding latency to the whole demo. Setting `api` as primary skips that. If
the new machine *does* have Ollama too and you just want cloud as a
backup, leave `GEMMA_BACKEND=ollama` as-is.

Confirm the environment is sound before running anything else:

```bash
python -m pytest -v
```

All tests should pass (check `PHASE_PROGRESS.md` for the current count) -
they're fully mocked, no network or Ollama required, so this works
identically on any machine with the deps installed. Then continue with
Stage 0 below, which is the first command that actually depends on which
Gemma backend is configured.

---

## Before a live/investor demo

Three real `.env` settings genuinely change what a first-time viewer
sees on the dashboard's very first screen - worth checking before
presenting live, not because anything is broken, but because each is a
real behavior change that's easy to forget you left at its zero-setup
default:

- **`OPERATOR_TOKENS`** - unset by default (the zero-setup dev
  experience), which means the dashboard's very first thing rendered,
  above all content, is a real `⚠️ Unauthenticated dashboard` warning
  (see `src/auth.py`). It's correct, intentional behavior - not a bug -
  but it's a rough first impression for an audience that can't parse
  what it means in the two seconds before you move on. Set
  `OPERATOR_TOKENS=name:token` so the dashboard opens straight into
  `Signed in as <name>` instead.
- **`GEMMA_BACKEND`** - already covered above ("Setup on a new
  machine"): if it's `ollama` but the machine has no Ollama models
  installed (or none running), every single Gemma call silently falls
  over to the real cloud API anyway (Phase 4's cross-backend fallback -
  see `GemmaClient`) - functionally correct, but ~10x slower per call
  with a wasted local-timeout delay first. Set it to `api` explicitly if
  that's really what's answering, both for speed and honesty about
  which backend is live.
- **`ALERT_WEBHOOK_URL`** - if you're planning to show real-time
  CRITICAL alerting (Stage 5c below) live, configure this beforehand and
  use the dashboard sidebar's real "Send test alert" button to confirm
  it's actually reaching your channel - the same kind of test-alert
  capability PagerDuty/Datadog/UptimeRobot already have, closing the gap
  where the first real signal a broken webhook gives you is silence
  during the actual CRITICAL-event demo.

---

## Quick start: the one-file guided demo

Everything below (Stages 0, 2, and 5) is also available as a single,
self-contained, step-by-step script - this is the recommended way to
demo the project, including to someone with no other context (e.g.
reading this repo cold on GitHub):

```bash
python scripts/run_demo.py
```

It pauses before each step, prints a plain-language explanation of what
that step demonstrates and why, then waits for you to press Enter (or
type `n` to skip that step) before running it. Covers: a preflight check
(config, log directory, real Gemma connectivity), a live CelesTrak scan,
the synthetic CRITICAL/budget-depletion/human-approval scenario (which of
the two - autonomous local execution vs. cloud proposal awaiting your
live approve/reject - depends entirely on this machine's `GEMMA_BACKEND`),
a replay of the real 2009 Iridium 33/Cosmos 2251 collision proving the
same severity threshold would have classified it CRITICAL, a real
decay/re-entry risk screen of a real CelesTrak debris group (a second,
non-conjunction hazard type, see Stage 2c), a local/cloud failover proof
(**skipped** on a local-only machine - `GEMMA_BACKEND=ollama` - since
demonstrating it would require a real cloud call, which a local-only demo
should never make), marking a decision human-reviewed, reading back the
raw audit log entry, running the test suite, and a summary table.

For a non-interactive run (CI / quick smoke-testing, no pauses):

```bash
python scripts/run_demo.py --auto
```

**Alternative: the live browser dashboard.** Everything the CLI demo logs
is also viewable (and, for pending approvals, actionable) in a browser:

```bash
streamlit run scripts/dashboard.py
```

Opens at `http://localhost:8501`: a "Show live positions" button renders
a real *current-position* view (not a triage result) for CelesTrak's real
crewed-stations group - ISS, Tiangong, and their currently-docked
visiting vehicles - on a 3D globe, independent of any logged event (see
Stage 2d below), an "Ask about the mission log" box for real
retrieval-augmented Q&A over the log itself (see Stage 2e below), and a
"Trends" section aggregating the accumulated log itself - severity mix
per day, recurring real objects, Gemma-vs-fallback narration mix over
time (see Stage 2g below). Below that: a metrics row, the full decision table,
a pending-human-approval inbox with real Approve/Reject buttons, sidebar
actions to fetch live CelesTrak conjunction data, screen a real CelesTrak
debris group for decay/re-entry risk, run the synthetic CRITICAL
scenario, run the synthetic attitude/pointing-loss scenario (Stage 2f
below), or replay the historical collision without leaving the
browser, and - for any real conjunction event selected in the inspect
panel - a real 3D orbit plot (Earth to scale, both objects' actual
propagated paths, a closest-approach marker) plus a distance-vs-time
chart with severity thresholds drawn in, built by re-fetching each
object's current TLE and re-propagating with the same physics Stage 2
above uses. Reads the exact same `logs/decisions-*.jsonl` audit log the
CLI writes to - run either one first (or both, in either order) and the
other will show the same data.

**This is the file to update every time a new phase is added** - append
a new `Step(...)` to the `STEPS` list in `scripts/run_demo.py` with its
own self-contained explanation, rather than writing a new script. The
stage-by-stage walkthrough below is still worth keeping around for
copy-pasting an individual piece in isolation, but `run_demo.py` is the
canonical, always-up-to-date demo.

---

## Stage 0 — Is Gemma actually reachable?

```bash
python scripts/check_gemma.py
```

**What it proves:** confirms which backend is configured (`GEMMA_BACKEND` in
`.env`) and that it can actually reach a model. Should print `REACHABLE`
and a real one-word response from `gemma4:e4b` via local Ollama. If Ollama
isn't running, it fails **gracefully** (prints `UNREACHABLE`, exit code 0)
rather than crashing - that graceful-failure behavior is itself part of
what was built.

---

## Stage 1 — The basic pipeline shape (no real orbital data yet)

```bash
python -m src.pipeline
```

**What it proves:** the `ingest -> analyze -> decide -> log` LangGraph
pipeline runs end to end. This uses `DummyAdapter` (fake telemetry, not
conjunction-shaped), so you'll see `severity: nominal` and a placeholder
description every time - that's expected. It's here to show the pipeline
skeleton works before real orbital data enters the picture. It also writes
3 lines to `logs/decisions-<today>.jsonl`.

---

## Stage 2 — Real orbital data: CelesTrak + Skyfield/SGP4, cross-group screening

```bash
python3 -c "
from src.pipeline import run_once
from src.ingestion.celestrak_adapter import CelesTrakAdapter
adapter = CelesTrakAdapter()  # default: real stations vs. real debris
entries = run_once(adapter=adapter, limit=2)
print('Scan stats:', adapter.last_scan_stats)
for e in entries:
    print(e.model_dump_json(indent=2))
"
```

**What it proves:** real TLE data fetched live from CelesTrak (cached to
`data/tle_cache/` for an hour) - by default two real, meaningfully
different groups: `stations` (ISS, Tiangong, ...) and `cosmos-2251-debris`
(real fragments from the 2009 Cosmos 2251/Iridium 33 collision) - screened
against EACH OTHER, not just within one group, so this actually answers
"is a real active spacecraft at risk from real tracked debris?" rather
than debris-vs-debris alone. Every pairwise conjunction across the
combined pool gets a real two-pass coarse/fine closest-approach search
(Skyfield/SGP4) over the next 48h. `adapter.last_scan_stats` shows exactly
what was screened: `total_objects`, `total_pairs_screened`, and
`pairs_refined` (see Stage 2b below for why refinement is capped). Look
for, in each printed entry:
- `min_distance_km`, `relative_velocity_km_s`, `time_of_closest_approach` -
  all real numbers from real physics, not placeholders.
- `object_a_group` / `object_b_group` - which real CelesTrak group each
  object actually came from.
- `tle_epoch_age_hours` - how stale the tracking data is.
- `finding.confidence` - now derived from that epoch age (fresher data =
  higher confidence, see Stage 4), not a flat constant.
- `description_provenance.source` / `rationale_provenance.source` - both
  should say `"gemma"` with `model_used: "gemma4:e4b"`.

Real conjunction data right now tends to land in `watch` or `nominal`
severity (min distance > 25km) - CRITICAL (<5km) is rare on any given real
fetch, which is why Stage 5 uses a synthetic fixture to demo that path
deterministically.

---

## Stage 2b — Why screening doesn't just brute-force every pair

Naively running the full two-pass search on every pair the way the
pre-Phase-10 adapter did doesn't scale: recomputing each object's coarse
trajectory from scratch for every pair it appears in is O(pairs) expensive
propagation calls, not O(objects). Measured directly during development,
calling `orbital.find_closest_approach` (the per-pair convenience
function) independently for all 1770 pairs from a real 60-object sample
took ~9.6s - already too slow for a live demo step, and it gets worse
quadratically. `CelesTrakAdapter` avoids this two ways (see its class
docstring in `src/ingestion/celestrak_adapter.py` for the exact numbers):
1. Each object's coarse-pass position is computed **once**
   (`orbital.compute_coarse_positions`) and reused for every pair it's in,
   instead of recomputed per pair - this alone cut the same 1770-pair
   coarse screen to ~0.02s in testing.
2. Only the `refine_top_k` closest pairs **by that cheap coarse estimate**
   get the expensive precise fine-pass refinement - a coarse pass can only
   ever *overestimate* the true closest approach (see `src/orbital.py`'s
   module docstring), so ranking by it and refining just the closest
   candidates is a sound way to skip pairs that are obviously nowhere near
   each other, without risking missing a genuinely close one.

See it run fast for real, on a much larger single-group sample than the
default:

```bash
python3 -c "
import time
from src.ingestion.celestrak_adapter import CelesTrakAdapter
a = CelesTrakAdapter(groups=['cosmos-2251-debris'], sample_size_per_group=100)
t0 = time.time()
a.fetch_batch(limit=5)
print(f'{a.last_scan_stats} in {time.time()-t0:.2f}s')
"
```

100 real objects -> 4950 real pairs screened, refined down to the
`refine_top_k` closest, in a fraction of a second (excluding the one-time
network fetch/cache).

---

## Stage 2c — A second real hazard type: orbital decay / re-entry risk

```bash
python3 -c "
from src.pipeline import run_once
from src.ingestion.decay_adapter import DecayRiskAdapter
adapter = DecayRiskAdapter(sample_size=200)  # real cosmos-2251-debris group
entries = run_once(adapter=adapter, limit=3)
for e in entries:
    print(e.model_dump_json(indent=2))
"
```

**What it proves:** conjunctions aren't the only thing this pipeline
screens for. `DecayRiskAdapter` fetches the same kind of real CelesTrak
TLE data (reusing `src/ingestion/tle_source.py`, the module extracted
from `CelesTrakAdapter` specifically so both hazard adapters share one
fetch/cache/parse path) but screens **per-object**, not pairwise: it pulls
perigee altitude, apogee altitude, and the BSTAR drag term straight out of
Skyfield's own SGP4 model (`sat.model.altp` / `.alta` / `.bstar` -
`src/decay.py`) and ranks objects by how low their perigee actually is.
No maneuver machinery applies here - re-entry isn't avoided with a
delta-v burn the way a conjunction is - so decay CRITICAL/WARNING/WATCH
findings get a real deterministic action and a real Gemma narration, same
as conjunctions, but no maneuver/verification/budget step. Look for the
same honesty markers as Stage 2: `perigee_altitude_km` and `bstar` are
real numbers pulled straight from Skyfield, not placeholders, and
`description_provenance.source` should say `"gemma"`.

Real data here currently tops out at `watch` severity - the
`cosmos-2251-debris` group's lowest real perigee is around 313km, above
the WARNING/CRITICAL bands below - because the very-low-perigee fragments
from that 2009 collision have already re-entered by now. Same "real data
rarely produces the most severe case on demand" situation already true
for CRITICAL conjunctions in Stage 2 - documented honestly rather than
tuning the thresholds to force a more dramatic result.

---

## Stage 2d — Live tracking: what's actually up there right now

```bash
python3 -c "
from src.live_positions import fetch_live_positions
positions = fetch_live_positions()  # real CelesTrak 'stations' group
for p in sorted(positions, key=lambda p: p.altitude_km):
    print(f'{p.name:30s} alt={p.altitude_km:7.1f} km  norad={p.norad_id}')
print(f'{len(positions)} real objects')
"
```

**What it proves:** not every real-data view in this project is a triage
result. Stages 2 and 2c both answer "is this a risk?" for a real
question with a real finding logged. This answers a different question -
"where are the real, named assets right now?" - independent of any
logged event: real current TLEs for CelesTrak's `stations` group (real
crewed stations - ISS, Tiangong - plus their currently-docked visiting
vehicles), each propagated with a single Skyfield/SGP4 evaluation at
`ts.now()` (`src/live_positions.py`) - the same physics, evaluated at an
instant instead of over a lookahead window. The dashboard's "Show live
positions" button (see above) renders this same data as a real 3D globe
with labeled markers instead of printing it.

Deliberately scoped to the `stations` group rather than CelesTrak's full
~20,000-object public catalog - the same "asset actually worth
protecting" set Stage 2's cross-group screening already uses, not an
attempt at an unbounded live map. And deliberately NOT wired into
`scripts/run_demo.py`: a position snapshot produces no
TelemetryEvent/AnomalyFinding/Decision, so forcing it through the guided
demo's log-writing steps would mean inventing a finding for data that
doesn't have one.

---

## Stage 2e — Ask about the mission log (retrieval-augmented, not fine-tuned)

```bash
python scripts/query_log.py "which CRITICAL events were vetoed and why?"
```

**What it proves:** real retrieval-augmented Q&A over the real audit log
- not fine-tuning, and every fact must be traceable to a real logged
entry. Every logged decision's real fields (subject, severity, action,
rationale) get embedded via a local Ollama embedding model
(`nomic-embed-text` by default, cached to disk keyed by `event_id` so an
unchanged log isn't re-embedded on every query - see `src/rag.py`), your
question gets embedded the same way, real cosine similarity ranks every
entry against it, and only the top-K most relevant real entries get
handed to Gemma as context - with an explicit instruction to answer ONLY
from those entries and say so plainly if they don't contain enough
information, rather than guessing. The printed output always shows which
real `event_id`s the answer was grounded in, so the grounding is
checkable, not just claimed.

Requires a reachable local Ollama for embeddings specifically
(`ollama pull nomic-embed-text`), even if `GEMMA_BACKEND=api` for
narration elsewhere - the hosted Gemini-style API has no embedding
endpoint wired up here. Same feature is in the dashboard too, as an "Ask
about the mission log" box.

Deliberately NOT model fine-tuning: real LoRA fine-tuning on this log's
(finding → rationale) pairs was a real option discussed and set aside -
it's a genuine separate ML effort (dataset curation, a training pass, an
evaluation harness), not something to casually bolt on. Retrieval gets
most of the practical benefit (real, checkable grounding in this
project's own history) using only data and infrastructure that already
exist.

---

## Stage 2f — A third hazard type: attitude / pointing loss (synthetic-only)

```bash
python3 -c "
import uuid
from src.ingestion.attitude_adapter import SyntheticAttitudeAdapter
from src.pipeline import run_once
adapter = SyntheticAttitudeAdapter(run_id=uuid.uuid4().hex[:8])
entries = run_once(adapter=adapter, limit=4)
for e in entries:
    print(e.finding.severity.value, '->', e.decision.rationale)
"
```

**What it proves:** a third hazard type, same idea-agnostic pipeline -
but a genuinely different situation from conjunctions and decay risk.
There is NO real, publicly-fetchable data source for spacecraft
ATTITUDE at all: TLEs encode only orbital position and velocity, never
orientation, and real attitude telemetry is normally proprietary to each
spacecraft's own operator, not published anywhere analogous to
CelesTrak. That's a structural absence, not "real data is rare on
demand" the way a CRITICAL conjunction is - so `SyntheticAttitudeAdapter`
(`src/ingestion/attitude_adapter.py`) is necessarily synthetic-only,
clearly labeled via `source="synthetic-attitude-fixture"`, the same
honesty standard `SyntheticCriticalAdapter` already set for conjunctions.
This was discussed and agreed explicitly before writing any code, not
discovered as a limitation afterward.

Four synthetic readings deliberately span NOMINAL/WATCH/WARNING/CRITICAL
in one run (unlike `SyntheticCriticalAdapter`'s all-CRITICAL design,
which exists specifically to demo delta-v budget depletion - not
applicable here, since attitude loss has no maneuver machinery either: a
tumbling spacecraft isn't fixed by an avoidance burn or a reboost, and
real attitude recovery - reaction-wheel desaturation, thruster-based
detumbling - is a genuinely separate, out-of-scope problem). Look for:
`classify_attitude_severity()` in `src/pipeline.py` classifying purely
from `pointing_error_deg` (< 5° NOMINAL, 5-15° WATCH, 15-45° WARNING,
>= 45° CRITICAL), with `angular_rate_deg_s` and `solar_panel_power_pct`
carried as real supporting signal in the description/rationale - and the
CRITICAL reading still getting a real deterministic action and real
Gemma narration, with no `maneuver_plan`.

---

## Stage 2g — Trends: what does the accumulated log actually say

```bash
python3 -c "
from src.logging_utils import DecisionLogger
from src.trends import is_real_live_source, rationale_source_counts_by_day, recurring_objects, severity_counts_by_day

entries = DecisionLogger().load_all_entries()
real_entries = [e for e in entries if is_real_live_source(e)]
print(f'{len(real_entries)} of {len(entries)} total entries are real, live CelesTrak scans')
print('Severity by day (real live scans only):', dict(sorted(severity_counts_by_day(real_entries).items())))
print('Rationale source by day (real live scans only):', dict(sorted(rationale_source_counts_by_day(real_entries).items())))
print('Top recurring objects (real and synthetic alike):')
for row in recurring_objects(entries, top_n=5):
    print(f\"  {row['object_name']} ({row['object_id']}): {row['count']} appearances, real={row['real']}\")
"
```

**What it proves:** every other view in this project shows one event or
one instant - this is the first that looks at the accumulated log's own
history. `src/trends.py` (pure data transforms, no Streamlit, no new
network/AI calls) buckets logged entries by real calendar day (keyed off
`decision.made_at`, set right after the Gemma call completes - not
`telemetry.timestamp`, set at ingestion before analyze/decide even run,
which could disagree with which `logs/decisions-YYYY-MM-DD.jsonl` file
an entry actually landed in if a slow Gemma call straddled a UTC-midnight
boundary) to show severity mix over time and how much narration came
from Gemma versus the deterministic fallback each day - filtered to
`is_real_live_source()` entries only (real live CelesTrak scans, not
synthetic fixtures or the repeated historical replay, which would
otherwise dominate a "real pattern over time" view just because a demo
was run more than once) - plus ranks EVERY object (real and synthetic
alike, each labeled `real`) by how many separate logged events they
appeared in - covering both conjunction pairs (each side counted
separately) and single-object hazards (decay, attitude). The dashboard's
"Trends" section renders the same data as two Plotly charts plus a table
- run the command above
first to see the raw numbers behind them.

---

## Stage 3 — Deterministic severity thresholds

No command needed to prove this - it's already visible in Stage 2's output
(`finding.severity`) and enforced by `classify_conjunction_severity()` in
`src/pipeline.py`:

| min_distance_km | severity   | action   |
|---|---|---|
| < 5              | CRITICAL   | abort (autonomous maneuver, see Stage 5) |
| 5 - 25           | WARNING    | hold |
| 25 - 100         | WATCH      | continue |
| >= 100           | NOMINAL    | continue |

The decay hazard type (Stage 2c) uses its own real, independent
thresholds - `classify_decay_severity()` in `src/pipeline.py` - based on
perigee altitude rather than distance, since it's a different physical
hazard:

| perigee_altitude_km | severity | action |
|---|---|---|
| < 200                | CRITICAL | abort (no maneuver - see Stage 2c) |
| 200 - 300            | WARNING  | hold |
| 300 - 500            | WATCH    | continue |
| >= 500               | NOMINAL  | continue |

Both are plain threshold checks, not Gemma-decided - the point being that
severity/action are reliable and reproducible; Gemma only explains them in
plain language.

---

## Stage 4 — Confidence from real TLE staleness

```bash
python3 -c "
from src.pipeline import compute_confidence
for age_hours in [12, 48, 100, 300]:
    print(f'{age_hours}h old -> confidence {compute_confidence({\"tle_epoch_age_hours\": age_hours})}')
print('no epoch data available -> confidence', compute_confidence({}))
"
```

**What it proves:** `AnomalyFinding.confidence` isn't a hardcoded `0.8`
anymore for conjunction events - it degrades as tracking data gets stale
(fresh <24h -> 0.9, down to >1 week -> 0.4), and falls back to a clearly
labeled placeholder only when no epoch signal exists at all (e.g.
`DummyAdapter`'s non-conjunction telemetry).

---

## Stage 5 — CRITICAL conjunction: maneuver, verification, budget, and approval (human or Gemma)

Real data rarely gives a CRITICAL case on demand, so this uses a synthetic
fixture (clearly labeled as such in `source`) to demo the full path
deterministically - **4 CRITICAL events sharing one 5.0 m/s delta-v
budget**, so you can watch it run out live:

```bash
python3 -c "
from datetime import datetime, timezone
from src.ingestion.base_adapter import DataSourceAdapter
from src.schemas import TelemetryEvent
from src.pipeline import run_once
from src.maneuver import DeltaVBudgetTracker

class SyntheticCriticalAdapter(DataSourceAdapter):
    def fetch_batch(self, limit):
        events = []
        for i in range(limit):
            raw = {
                'object_a_id': f'9900{i}', 'object_a_name': f'SYNTH-A-{i}',
                'object_b_id': f'9901{i}', 'object_b_name': f'SYNTH-B-{i}',
                'min_distance_km': 3.0,
                'time_of_closest_approach': '2026-08-01T20:00:00+00:00',
                'relative_velocity_km_s': 6.0,
            }
            events.append(TelemetryEvent(
                event_id=f'conj-critical-demo-{i}',
                timestamp=datetime.now(timezone.utc),
                source='synthetic-critical-fixture',
                raw_data=raw,
            ))
        return events

tracker = DeltaVBudgetTracker(starting_budget_m_s=5.0)
entries = run_once(adapter=SyntheticCriticalAdapter(), budget_tracker=tracker, limit=4)
for e in entries:
    d = e.decision
    approval = d.maneuver_approval
    print(f'{e.telemetry.event_id}: budget_insufficient={d.budget_insufficient} remaining={tracker.remaining_m_s:.3f}m/s')
    if approval is not None:
        print(f'  approval: mode={approval.mode} approved={approval.approved} by={approval.approved_by}')
    print(f'  -> {d.rationale}')
"
```

**What it proves:**
- Events 0-2: `compute_avoidance_maneuver()` + `verify_maneuver()` actually
  run. On the local backend, a deterministic physics check verifies the
  maneuver safe first, then Gemma itself gets those same numbers and
  issues a real GO/NO-GO verdict (Phase 9 - see `_maneuver_veto_check` in
  `src/pipeline.py`) standing in for the unavailable human - requested as
  real JSON-schema-constrained output from Ollama (Phase 17), not parsed
  out of free text, with the original free-text parser
  (`_parse_veto_verdict`) kept on as a fallback for the one case
  structured output can't cover (a cross-backend fallback response from
  the hosted API mid-call). When it
  affirms (the expected outcome for a maneuver the physics already
  verified safe), `decision.maneuver_plan` and `decision.verified_clearance`
  (new distance, `cleared: true`) are populated, `maneuver_approval.mode`
  reads `"autonomous"` with `approved=True`, and Gemma narrates it as a
  **completed autonomous action** ("Autonomous action taken: executed a
  radial-outward avoidance maneuver..."). Gemma can only make this *more*
  conservative than the physics check, never less - if it ever vetoes an
  already-verified-safe maneuver (or gives an unparseable answer, which
  fail-safes to a veto too), `verified_clearance` stays `None`,
  `maneuver_approval.approved=False`, and the rationale reads "Maneuver
  vetoed: blocked pending review" instead. Gemma being *unreachable*
  during this check is NOT treated as a veto - an LLM outage alone
  shouldn't block a maneuver the physics already verified safe, so that
  case still executes autonomously.
- Event 3: the shared budget runs out mid-batch. `budget_insufficient`
  flips to `True`, `verified_clearance` stays `None` (nothing was actually
  applied), and the rationale correctly says the maneuver was **calculated
  but not executed** and escalates for human review - it does not lie
  about having succeeded.

**Local vs. cloud changes what happens above (Phase 8-9):** the snippet
above doesn't pick a `client` explicitly, so it uses whatever this
machine's `.env` has configured. `GEMMA_BACKEND=ollama` (local) is treated
as "ground control unreachable" - events 0-2 go through Gemma's autonomous
safety review described above (affirm-and-execute in the normal case).
`GEMMA_BACKEND=api` (cloud) is treated as "ground control reachable" -
events 0-2 instead come back with `decision.awaiting_human_approval=True`
and `verified_clearance=None` (nothing executed yet), and the rationale
reads "Maneuver proposed: awaiting human approval before execution." To
resolve one:

```bash
python scripts/approve_maneuver.py <event_id> "your-name"            # approve
python scripts/approve_maneuver.py <event_id> "your-name" --reject   # reject
```

This calls `DecisionLogger.approve_maneuver()`, which (if approved) runs
`verify_maneuver()` for real at that point - not before - and rewrites the
log entry in place with `maneuver_approval` (`mode="human"`,
`approved_by=<name>`) and (if approved) `verified_clearance`. `run_demo.py`
does this same resolution automatically, prompting live for each pending
maneuver when run interactively.

**Re-running this exact snippet:** the event ids above (`conj-critical-demo-0`
etc.) are hardcoded, unlike `run_demo.py`'s synthetic fixture (which
includes a per-run unique id specifically to avoid this). `mark_reviewed`/
`approve_maneuver` match the *first* logged entry with a given event id -
so running this snippet a second time and then trying to approve/review
"this run's" event would silently update the stale entry from the first
run instead. Fine for a single demo pass; if you're rehearsing repeatedly,
either use `python scripts/run_demo.py` (handles this correctly) or vary
the event ids yourself between runs.

---

## Stage 5b — Historical replay: a real past collision, not live/synthetic data

Everything above uses live or synthetic data. This replays a REAL,
documented historical conjunction through the exact same pipeline,
unmodified:

```bash
python3 -c "
from src.pipeline import run_once
from src.ingestion.historical_adapter import HistoricalReplayAdapter
entries = run_once(adapter=HistoricalReplayAdapter(run_id='demo'), limit=1)
e = entries[0]
print(e.model_dump_json(indent=2))
"
```

**What it proves:** the 2009 Iridium 33/Cosmos 2251 collision - the first
confirmed accidental collision between two intact satellites - is
replayed using the REAL, documented SOCRATES prediction: 584m, from
CelesTrak's own account of the event
([celestrak.org/events/collision](https://celestrak.org/events/collision/)).
SOCRATES genuinely predicted this exact conjunction in all 14 reports
issued that week (range 117m-1.812km); the final report, issued
2009-02-10 15:02 UTC, predicted 584m at ~16:56 UTC the same day. It just
never made the priority list (rank #152 that day) and nobody acted on it
- a triage failure, not a detection failure. NORAD catalog numbers
(Iridium 33: 24946, Cosmos 2251: 22675), the ~11.7 km/s relative
velocity, and the ~789 km collision altitude were independently
corroborated against NASA/Wikipedia sources.

Feeding that real number into this system's ordinary, unmodified severity
threshold (<5km = CRITICAL) - look for `finding.severity: "critical"` in
the output, exactly like Stage 5's synthetic events, with no
special-casing for this being a replay. `source: "historical-replay"` and
the `historical_event`/`historical_source`/`historical_actual_outcome`
fields in `raw_data` keep this clearly labeled as a documented-record
replay everywhere it surfaces (audit log, dashboard table), never
mistaken for live tracking.

**Why not re-derive it from real historical TLEs via SGP4, like Stage 2?**
Checked directly, not assumed: CelesTrak's public `gp.php` endpoint
ignores a historical `EPOCH` query parameter and always returns today's
TLE regardless - genuine historical archives require Space-Track.org,
which needs a real account this project doesn't have. Rather than fake
historical propagation with current-day TLEs mislabeled as 2009 data,
this replays the real closest-approach number that was actually reported
at the time.

---

## Stage 5c — Real-time alerting for CRITICAL events

```bash
python3 -c "
import http.server, json, threading, uuid

received = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        received.append(json.loads(self.rfile.read(length)))
        self.send_response(200); self.end_headers()
    def log_message(self, *a): pass

server = http.server.HTTPServer(('localhost', 8765), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

from src.config import settings as base_settings, Settings
from src.gemma_client import GemmaClient
from src.ingestion.synthetic_adapter import SyntheticCriticalAdapter
from src.logging_utils import DecisionLogger
from src.pipeline import run_once

settings = Settings(
    gemma_backend=base_settings.gemma_backend, gemma_model=base_settings.gemma_model,
    ollama_host=base_settings.ollama_host, gemma_api_key=base_settings.gemma_api_key,
    gemma_model_api=base_settings.gemma_model_api, log_dir=base_settings.log_dir,
    delta_v_budget_m_s=base_settings.delta_v_budget_m_s,
    alert_webhook_url='http://localhost:8765/webhook',
)
adapter = SyntheticCriticalAdapter(run_id=uuid.uuid4().hex[:8])
run_once(adapter=adapter, client=GemmaClient(settings=settings), logger=DecisionLogger(settings=settings), limit=1)
print(f'{len(received)} real webhook call(s) received:')
print(received[0]['text'] if received else '(none)')
"
```

**What it proves:** a CRITICAL finding doesn't just sit in the log
waiting to be noticed - `src/alerting.py` fires a real HTTP POST the
moment one is logged, from any of the three hazard types. This command
stands up a genuine local HTTP receiver (not a mock) to prove the real
webhook call actually happens, with the real content: subject, event_id,
action, and the already-generated real Gemma rationale (no new Gemma
call for the alert itself). Slack Incoming Webhook compatible
(`{"text": ...}`), so it also works with Discord, Microsoft Teams, or
any custom receiver - just set `ALERT_WEBHOOK_URL`.

The firing condition is deterministic
(`finding.severity == Severity.CRITICAL`) - not Gemma's call, same
"Gemma narrates, never decides" principle as everywhere else. Disabled
by default (`ALERT_WEBHOOK_URL` unset is a no-op, not an error), and a
failed send is caught and reported, never raised - an alerting outage
must never block or crash the actual triage pipeline. Try a non-CRITICAL
event through the same receiver and confirm zero webhook calls arrive.

---

## Stage 6 — Human review, for real

Take the `event_id` from Stage 5's last (budget-insufficient) line and
mark it reviewed:

```bash
python scripts/mark_reviewed.py conj-critical-demo-3 "your-name-here"
```

**What it proves:** confirm it actually persisted by checking the real log
file - `human_reviewed`, `human_reviewed_at`, and `reviewed_by` should now
be filled in on that one line:

```bash
grep conj-critical-demo-3 logs/decisions-*.jsonl
```

---

## Stage 7 — Local/cloud failover

```bash
python3 -c "
from datetime import datetime, timezone
from src.config import load_settings, Settings
from src.gemma_client import GemmaClient
from src.pipeline import make_decide_node
from src.schemas import AnomalyFinding, Severity, TelemetryEvent

settings = load_settings()
# Simulate a dead local Ollama - doesn't touch your real .env or Ollama.
broken = Settings(
    gemma_backend='ollama', gemma_model=settings.gemma_model,
    ollama_host='http://localhost:1', gemma_api_key=settings.gemma_api_key,
    gemma_model_api=settings.gemma_model_api, log_dir=settings.log_dir,
    delta_v_budget_m_s=settings.delta_v_budget_m_s,
)
client = GemmaClient(settings=broken)
event = TelemetryEvent(
    event_id='conj-failover-demo', timestamp=datetime.now(timezone.utc), source='celestrak',
    raw_data={'object_a_id': '1', 'object_a_name': 'A', 'object_b_id': '2', 'object_b_name': 'B',
              'min_distance_km': 50.0, 'time_of_closest_approach': '2026-08-02T00:00:00+00:00',
              'relative_velocity_km_s': 5.0},
)
finding = AnomalyFinding(event_id=event.event_id, severity=Severity.WATCH, description='Test.', confidence=0.8)
result = make_decide_node(client)({'telemetry': event, 'finding': finding, 'decision': None, 'log_path': None})
print(f'backend/model: {result[\"rationale_provenance\"].model_used}')
print(f'rationale: {result[\"decision\"].rationale!r}')
"
```

**What it proves:** with local Ollama genuinely unreachable, the client
automatically falls over to the real hosted API and still returns a clean
answer (chain-of-thought stripped by `_extract_final_answer`, whichever
backend answers) - going through `decide_node` (not calling `generate()`
directly) so you see it exactly as it behaves in a real pipeline run.
`rationale_provenance.model_used` will read
`"gemma-4-26b-a4b-it (api, fallback from ollama)"` - visible proof of
which backend actually responded, not just which was configured.

(Only run this if you've added a real `GEMMA_API_KEY` to `.env` - otherwise
both backends fail and you'll see the deterministic fallback text instead,
which is also correct behavior, just less interesting to watch.)

---

## Stage 7b — Circuit breaker: an extended outage doesn't keep paying full price

```bash
python3 -c "
import time
from src.config import settings as base_settings, Settings
from src.gemma_client import CIRCUIT_BREAKER_THRESHOLD, GemmaClient, GemmaClientError

broken = Settings(
    gemma_backend='ollama', gemma_model=base_settings.gemma_model,
    ollama_host='http://localhost:1', gemma_api_key='',
    gemma_model_api=base_settings.gemma_model_api, log_dir=base_settings.log_dir,
    delta_v_budget_m_s=base_settings.delta_v_budget_m_s,
)
client = GemmaClient(settings=broken)

for i in range(CIRCUIT_BREAKER_THRESHOLD):
    t0 = time.monotonic()
    try:
        client.generate(prompt='test')
    except GemmaClientError:
        print(f'call {i+1}: real failed attempt in {time.monotonic()-t0:.4f}s')

t0 = time.monotonic()
try:
    client.generate(prompt='test')
except GemmaClientError as e:
    print(f'call {CIRCUIT_BREAKER_THRESHOLD+1}: {time.monotonic()-t0:.6f}s - {e}')
"
```

**What it proves:** the local/cloud failover above already handles a
transient hiccup, but if a backend stays down for an extended stretch,
every SINGLE subsequent event would otherwise still pay the full
retry+timeout cost against a backend already known to be unreachable.
`GemmaClient` tracks CONSECUTIVE failed `generate()` calls per backend
(one count per real-world event that found it down, not per raw HTTP
attempt); after `CIRCUIT_BREAKER_THRESHOLD` (3) in a row, that backend's
circuit "opens" and real attempts against it are skipped entirely for
`CIRCUIT_BREAKER_COOLDOWN_S` (60) seconds. The 4th call above should
fail near-instantly with an error explicitly naming the circuit
breaker, not a real (even if fast-failing) network attempt - this was
measured for real against a deliberately-unreachable host
(`http://localhost:1`, the same pattern the failover step above uses),
not just asserted from mocks: ~2500x faster in testing. A single
success resets the count to zero, and once the cooldown elapses the
next call gets a real "half-open" probe.

---

## Stage 8 — Full test suite

```bash
python -m pytest -v
```

**What it proves:** 221 tests, all green - orbital math (including the
decomposed coarse/fine search used for scalable screening), TLE parsing
and the shared `tle_source.py` fetch/cache module, the CelesTrak
adapter's cross-group conjunction screening (mocked network), the decay
hazard type's severity classification and screening (mocked network,
plus real Vanguard 1/ISS TLE fixtures exercising Skyfield's own
perigee/apogee/BSTAR fields), the attitude hazard type's severity
classification and synthetic fixture (including the decay-vs-attitude
subject-line disambiguation regression), the live tracking view's real
position computation and figure structure (mocked network, real fixed
TLE fixtures), the mission-log search's embedding cache/invalidation,
cosine-similarity ranking, and context-grounded prompt construction
(mocked Ollama calls), CRITICAL-event webhook alerting's text formatting
across all three hazard shapes, severity/URL gating, and fail-safe
network-error handling (mocked), the Trends view's day-bucketing and
recurring-objects ranking across all three hazard shapes, the circuit
breaker's open/reset/half-open-retry state machine (mocked time) and
prompt logging on both the Gemma-success and fallback paths, maneuver
math (including the QA pass's
plausibility bound), budget tracking, Gemma client retry/fallback
(mocked), Gemma's autonomous maneuver veto-check - both the structured-
JSON path and the free-text fallback path (mocked) - the historical
replay (including an integration test proving the real 584m number
classifies as CRITICAL through the actual pipeline), the orbit plot's
real TLE fetch/propagation and Plotly figure structure (mocked network),
terminal rendering for every maneuver state and all three hazard types,
the dashboard's data transforms and UI (via Streamlit's AppTest harness),
preflight checks, the full pipeline wiring, and the human-review/
maneuver-approval log rewrites - covering everything demoed above
without needing real network calls for CI/repeatability. (Check
`PHASE_PROGRESS.md` for the current count if this drifts again as more
gets added.)

---

## Stage 9 — The audit trail itself

```bash
tail -5 logs/decisions-*.jsonl | python3 -m json.tool
```

**What it proves:** every decision is self-contained and reconstructable -
what was detected (`telemetry.raw_data`), what was decided
(`finding.severity`, `decision.action`), why in plain language
(`finding.description`, `decision.rationale`), and whether that
explanation is trustworthy (`*_provenance.source`: `"gemma"` vs
`"fallback"`, and which backend/model actually produced it).
