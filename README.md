# Track 2: Deep Space Navigation — Orbital Collision Avoidance

*Build with Gemma: Triage in Light Speed*

An orbital collision/conjunction predictor: pulls real satellite/debris
tracking data (TLEs) from CelesTrak, runs real orbital mechanics
(Skyfield/SGP4) to find close approaches over the next 48 hours,
classifies risk severity with deterministic distance thresholds (**not**
AI-decided — reliability matters here), and uses Gemma (local via Ollama,
or a hosted API) to explain findings and decisions in plain language.

For the most severe (CRITICAL) conjunctions, a simplified avoidance
maneuver is computed and independently re-verified — deterministically,
never by the AI. Whether it executes autonomously or waits for a human to
approve it depends on which Gemma backend is configured: local (Ollama)
is treated as "ground control unreachable" (self-approve, no human in the
loop); the hosted API is treated as "ground control reachable" (a human
must explicitly approve or reject before it executes). A limited delta-v
budget prevents an unlimited number of maneuvers from executing silently.
Every decision — including whether its explanation came from Gemma or a
deterministic fallback, and who (if anyone) approved a maneuver — is
written to an append-only JSON-lines audit log.

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
- `src/ingestion/celestrak_adapter.py` — live CelesTrak TLE fetch + real
  orbital-mechanics conjunction ranking (with disk caching)
- `src/pipeline.py` — the LangGraph pipeline: `analyze -> decide -> log`
- `src/logging_utils.py` — append-only JSON-lines decision audit log,
  plus human-review and maneuver-approval workflows
- `src/display.py` — rich-based color-coded terminal rendering (stdout
  only — never affects the log file)
- `src/preflight.py` — config/connectivity/filesystem health checks
- `scripts/run_demo.py` — **the guided, step-by-step demo** (see below)
- `scripts/check_gemma.py`, `scripts/mark_reviewed.py`,
  `scripts/approve_maneuver.py` — standalone CLI utilities
- `tests/` — full suite, no live network/Ollama required

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`: `GEMMA_BACKEND=ollama` (local) or `api` (hosted). For the
hosted backend, set `GEMMA_API_KEY` and `GEMMA_MODEL_API` (Ollama tags and
hosted model ids use different naming schemes — check which models your
key actually has access to). `DELTA_V_BUDGET_M_S` controls how many
CRITICAL maneuvers can execute before the system starts explicitly
flagging "insufficient budget" instead of silently continuing.

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
