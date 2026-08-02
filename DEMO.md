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
Stage 2d below). Below that: a metrics row, the full decision table,
a pending-human-approval inbox with real Approve/Reject buttons, sidebar
actions to fetch live CelesTrak conjunction data, screen a real CelesTrak
debris group for decay/re-entry risk, run the synthetic CRITICAL
scenario, or replay the historical collision without leaving the
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
  `src/pipeline.py`) standing in for the unavailable human. When it
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

## Stage 8 — Full test suite

```bash
python -m pytest -v
```

**What it proves:** 152 tests, all green - orbital math (including the
decomposed coarse/fine search used for scalable screening), TLE parsing
and the shared `tle_source.py` fetch/cache module, the CelesTrak
adapter's cross-group conjunction screening (mocked network), the decay
hazard type's severity classification and screening (mocked network,
plus real Vanguard 1/ISS TLE fixtures exercising Skyfield's own
perigee/apogee/BSTAR fields), the live tracking view's real position
computation and figure structure (mocked network, real fixed TLE
fixtures), maneuver math (including the QA pass's
plausibility bound), budget tracking, Gemma client retry/fallback
(mocked), Gemma's autonomous maneuver veto-check (mocked), the historical
replay (including an integration test proving the real 584m number
classifies as CRITICAL through the actual pipeline), the orbit plot's
real TLE fetch/propagation and Plotly figure structure (mocked network),
terminal rendering for every maneuver state and both hazard types, the
dashboard's data transforms and UI (via Streamlit's AppTest harness),
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
