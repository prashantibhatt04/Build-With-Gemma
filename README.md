# Track 2: Deep Space Navigation — Orbital Collision Avoidance

*Build with Gemma: Triage in Light Speed*

[![Tests](https://github.com/prashantibhatt04/Build-With-Gemma/actions/workflows/tests.yml/badge.svg)](https://github.com/prashantibhatt04/Build-With-Gemma/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An orbital collision/conjunction predictor: pulls real satellite/debris
tracking data (TLEs) from CelesTrak — by default real crewed stations
cross-screened against a real debris field, not just one group in
isolation — runs real orbital mechanics (Skyfield/SGP4) to find close
approaches over the next 48 hours, classifies risk severity with
deterministic distance thresholds (**not** AI-decided — reliability
matters here), and uses Gemma (local via Ollama, or a hosted API) to
explain findings and decisions in plain language.

For the most severe (CRITICAL) conjunctions, a simplified avoidance
maneuver is computed and independently re-verified — deterministically,
never by the AI. Whether it executes autonomously or waits for a human to
approve it depends on which Gemma backend is configured: local (Ollama)
is treated as "ground control unreachable" — a deterministic physics
check verifies the maneuver safe, then Gemma itself issues a real
GO/NO-GO veto over it, standing in for the unavailable human (bounded so
it can only veto an already-verified-safe maneuver, never approve one the
physics hasn't cleared); the hosted API is treated as "ground control
reachable" (a real human must explicitly approve or reject before it
executes). A limited delta-v budget prevents an unlimited number of
maneuvers from executing silently.
Every decision — including whether its explanation came from Gemma or a
deterministic fallback, and who (if anyone) approved a maneuver — is
written to an append-only JSON-lines audit log.

It's also validated against a real historical failure, not just live or
synthetic data: replaying the real, documented 2009 Iridium 33/Cosmos
2251 collision — the first confirmed satellite-satellite collision —
through this system's unmodified pipeline shows it would have classified
the real 584m SOCRATES prediction as CRITICAL, against a real event where
that same warning existed but was never prioritized or acted on.

Conjunctions aren't the only hazard it screens for: a second real hazard
type — orbital decay/re-entry risk — screens a real CelesTrak debris
group for objects with dangerously low perigee altitude, using Skyfield's
own SGP4 model (no synthetic data, same deterministic-severity /
Gemma-narrates design as conjunctions).

**See [`DEMO.md`](DEMO.md) for a full stage-by-stage walkthrough**, or
just run the guided demo script directly (see below). **See
[`PHASE_PROGRESS.md`](PHASE_PROGRESS.md)** for what was built in each
phase and why.

## Project layout

- `src/config.py` — env-var driven settings
- `src/gemma_client.py` — swappable Gemma client (local Ollama or a hosted
  API), with automatic cross-backend fallback if the primary is unreachable
- `src/schemas.py` — telemetry / finding / decision / maneuver / approval models
- `src/orbital.py` — NORAD ID + TLE epoch parsing, two-pass closest-approach search
- `src/maneuver.py` — simplified deterministic avoidance-maneuver math,
  independent verification, and delta-v budget tracking
- `src/ingestion/tle_source.py` — shared CelesTrak TLE fetch + disk cache +
  TLE-block parsing, used by both hazard adapters below
- `src/ingestion/celestrak_adapter.py` — real cross-group orbital-mechanics
  conjunction screening
- `src/decay.py` — real decay/re-entry risk assessment, pulling perigee/
  apogee altitude and BSTAR straight out of Skyfield's own SGP4 model
- `src/ingestion/decay_adapter.py` — screens a real CelesTrak debris group
  for low-perigee objects, ranked by decay risk
- `src/ingestion/synthetic_adapter.py` — synthetic CRITICAL-range fixture
  (real data rarely produces one on demand), shared by the demo and dashboard
- `src/ingestion/historical_adapter.py` — replays a real, documented past
  conjunction (the 2009 Iridium 33/Cosmos 2251 collision) through the
  unmodified pipeline
- `src/pipeline.py` — the LangGraph pipeline: `analyze -> decide -> log`
- `src/logging_utils.py` — append-only JSON-lines decision audit log,
  plus human-review and maneuver-approval workflows
- `src/display.py` — rich-based color-coded terminal rendering (stdout
  only — never affects the log file); `classify_decision_status` here is
  the single source of truth for maneuver state, shared with the dashboard
- `src/dashboard_data.py` — the dashboard's data transforms, kept
  Streamlit-free so they're directly unit-testable
- `src/orbit_plot_data.py` — real 3D trajectory + distance-over-time
  Plotly charts, re-propagated from live TLE data with the same physics
  `src/orbital.py` already uses
- `src/live_positions.py` — real *current* positions (not a triage
  result) for CelesTrak's real crewed-stations group, on a 3D globe
- `src/preflight.py` — config/connectivity/filesystem health checks
- `scripts/run_demo.py` — **the guided, step-by-step CLI demo** (see below)
- `scripts/dashboard.py` — **the live browser dashboard** (see below)
- `scripts/check_gemma.py`, `scripts/mark_reviewed.py`,
  `scripts/approve_maneuver.py` — standalone CLI utilities
- `tests/` — full suite, no live network/Ollama required

## Setup

### Prerequisites

- Python 3.10+ (developed against 3.14)
- [git](https://git-scm.com/)
- Either [Ollama](https://ollama.com) (for a fully local setup) or a
  hosted Gemini-style API key (for the cloud path) — see below, you don't
  need both

### 1. Clone and install

```bash
git clone https://github.com/prashantibhatt04/Build-With-Gemma.git
cd Build-With-Gemma
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Gemma access

`.env` is gitignored — it's created locally from `.env.example` and never
committed. Pick one (or set up both and switch via `GEMMA_BACKEND`):

**Option A — Local, via Ollama (no API key, no cost, works offline):**

1. Install Ollama: [ollama.com/download](https://ollama.com/download)
2. Pull the model this project uses:
   ```bash
   ollama pull gemma4:e4b
   ```
3. Confirm it's running (Ollama typically starts automatically after
   install; if not, `ollama serve`). Default is `http://localhost:11434`,
   matching `OLLAMA_HOST` in `.env.example`.
4. In `.env`: `GEMMA_BACKEND=ollama`, `GEMMA_MODEL=gemma4:e4b` (already the
   defaults).

**Option B — Cloud, via a hosted Gemini-style API key:**

1. Generate a free API key at
   [Google AI Studio](https://aistudio.google.com/apikey).
2. In `.env`: `GEMMA_BACKEND=api`, `GEMMA_API_KEY=<your key>`.
3. Ollama tags (e.g. `gemma4:e4b`) and hosted model ids use **different
   naming schemes** — they're not interchangeable. Check which Gemma
   models your key actually has access to:
   ```bash
   curl -H "x-goog-api-key: $GEMMA_API_KEY" \
     https://generativelanguage.googleapis.com/v1beta/models | grep -i gemma
   ```
   Set `GEMMA_MODEL_API` to one of the returned model ids (e.g.
   `gemma-4-26b-a4b-it`).

`DELTA_V_BUDGET_M_S` controls how many CRITICAL maneuvers can execute
before the system starts explicitly flagging "insufficient budget"
instead of silently continuing — the default (5.0) is fine to start with.

**Security note:** never commit `.env`, never `cat`/`echo`/`print` it or
your API key on screen (including during recordings or screenshots), and
never paste a real key into a terminal you're recording. This project's
own code never prints the raw key anywhere — the only way it leaks is if
someone does so manually.

## Run the demo

```bash
python scripts/run_demo.py
```

A guided, self-explanatory, step-by-step walkthrough of everything above
— pauses before each step with a plain-language explanation, real
CelesTrak data, the CRITICAL/maneuver/budget/approval scenario, a
local/cloud failover proof (skipped automatically on a local-only
machine), human review, the raw audit trail, and the test suite. Add
`--auto` to run straight through with no pauses. See `DEMO.md` for more
detail on each step, plus copy-pasteable standalone snippets.

## Run the dashboard

```bash
streamlit run scripts/dashboard.py
```

A live, browser-based mission-ops view over the exact same audit log the
CLI writes to — metrics, a full decision table, a pending-approval inbox
with real Approve/Reject buttons, and (for any real conjunction event) a
real 3D orbit plot built by re-propagating live TLE data. A separate
"Show live positions" button renders a real *current-position* view (not
a triage result) for CelesTrak's real crewed-stations group — where those
assets actually are right now, independent of any logged event. Sidebar
buttons can generate real new activity (a live CelesTrak conjunction
scan, a real decay/re-entry risk screen, the synthetic CRITICAL scenario,
or a historical replay) without leaving the browser. Opens at
`http://localhost:8501` by default.

## Run the test suite

```bash
pytest
```

No live network or Ollama required — Gemma calls are mocked throughout.

## Other useful commands

```bash
python scripts/check_gemma.py          # connectivity check for the configured backend
python scripts/mark_reviewed.py <event_id> <name>              # post-hoc human review
python scripts/approve_maneuver.py <event_id> <name> [--reject]  # resolve a pending maneuver
```

## Links & references

- [Ollama](https://ollama.com) — local model runtime ([download](https://ollama.com/download), [model library](https://ollama.com/library))
- [Google AI Studio](https://aistudio.google.com/apikey) — generate a hosted Gemini-style API key
- [Generative Language API docs](https://ai.google.dev/gemini-api/docs) — hosted API reference (models list, request/response format)
- [CelesTrak](https://celestrak.org) — source of live TLE (satellite/debris tracking) data
- [Skyfield](https://rhodesmill.org/skyfield/) — orbital mechanics / SGP4 propagation library used here
- [LangGraph](https://langchain-ai.github.io/langgraph/) — pipeline orchestration
- [Pydantic](https://docs.pydantic.dev/) — schema validation
- [Rich](https://rich.readthedocs.io/) — terminal rendering

## More documentation

- [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) — problem statement, architecture diagrams, field glossary
- [`DEMO.md`](DEMO.md) — full stage-by-stage walkthrough with copy-pasteable commands
- [`PHASE_PROGRESS.md`](PHASE_PROGRESS.md) — complete build history, phase by phase
- [`KAGGLE_WRITEUP.md`](KAGGLE_WRITEUP.md) — submission writeup (architecture, Gemma usage, engineering hurdles, design choices)

## License

[MIT](LICENSE)
