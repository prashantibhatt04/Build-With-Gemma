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
a local/cloud failover proof (**skipped** on a local-only machine -
`GEMMA_BACKEND=ollama` - since demonstrating it would require a real
cloud call, which a local-only demo should never make), marking a
decision human-reviewed, reading back the raw audit log entry, running
the test suite, and a summary table.

For a non-interactive run (CI / quick smoke-testing, no pauses):

```bash
python scripts/run_demo.py --auto
```

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

## Stage 2 — Real orbital data: CelesTrak + Skyfield/SGP4

```bash
python3 -c "
from src.pipeline import run_once
from src.ingestion.celestrak_adapter import CelesTrakAdapter
entries = run_once(adapter=CelesTrakAdapter(sample_size=15), limit=2)
for e in entries:
    print(e.model_dump_json(indent=2))
"
```

**What it proves:** real TLE data fetched live from CelesTrak (cached to
`data/tle_cache/` for an hour), real orbital propagation (two-pass
coarse/fine closest-approach search over the next 48h), real Gemma-written
plain-language descriptions and rationales. Look for:
- `min_distance_km`, `relative_velocity_km_s`, `time_of_closest_approach` -
  all real numbers from real physics, not placeholders.
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

This is a plain threshold check, not Gemma-decided - the point being that
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

## Stage 5 — CRITICAL conjunction: maneuver, verification, budget, human approval

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
    print(f'{e.telemetry.event_id}: budget_insufficient={d.budget_insufficient} remaining={tracker.remaining_m_s:.3f}m/s')
    print(f'  -> {d.rationale}')
"
```

**What it proves:**
- Events 0-2: `compute_avoidance_maneuver()` + `verify_maneuver()` actually
  run - `decision.maneuver_plan` (direction, delta-v, target clearance) and
  `decision.verified_clearance` (new distance, `cleared: true`) are
  populated, and Gemma narrates it as a **completed autonomous action**
  ("Autonomous action taken: executed a radial-outward avoidance
  maneuver...").
- Event 3: the shared budget runs out mid-batch. `budget_insufficient`
  flips to `True`, `verified_clearance` stays `None` (nothing was actually
  applied), and the rationale correctly says the maneuver was **calculated
  but not executed** and escalates for human review - it does not lie
  about having succeeded.

**Local vs. cloud changes what happens above (Phase 8):** the snippet
above doesn't pick a `client` explicitly, so it uses whatever this
machine's `.env` has configured. `GEMMA_BACKEND=ollama` (local) is treated
as "ground control unreachable" - events 0-2 self-approve and execute
immediately, exactly as shown above. `GEMMA_BACKEND=api` (cloud) is
treated as "ground control reachable" - events 0-2 instead come back with
`decision.awaiting_human_approval=True` and `verified_clearance=None`
(nothing executed yet), and the rationale reads "Maneuver proposed:
awaiting human approval before execution." To resolve one:

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

**What it proves:** 73 tests, all green - orbital math, TLE parsing, the
CelesTrak adapter (mocked network), maneuver math, budget tracking,
Gemma client retry/fallback (mocked), terminal rendering for every
maneuver state, preflight checks, the full pipeline wiring, and the
human-review/maneuver-approval log rewrites - covering everything demoed
above without needing real network calls for CI/repeatability. (Check
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
