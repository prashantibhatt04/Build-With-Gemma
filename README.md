# Track 2 Starter Scaffolding — Deep Space Navigation

*Build with Gemma: Triage in Light Speed*

This is **Layer 1 scaffolding only**: idea-agnostic infrastructure for a
Track 2 (Deep Space Navigation) submission. No specific project idea is
encoded here — just the plumbing every version of the idea will need:

- `src/config.py` — env-var driven settings
- `src/gemma_client.py` — swappable Gemma client (local Ollama or a hosted
  Gemini-style API), same interface either way
- `src/schemas.py` — generic telemetry / anomaly / decision models
- `src/ingestion/base_adapter.py` — data-source adapter pattern + a
  `DummyAdapter` for smoke testing
- `src/pipeline.py` — a LangGraph pipeline skeleton wiring
  `analyze -> decide -> log`
- `src/logging_utils.py` — append-only JSON-lines decision audit log
- `scripts/check_gemma.py` — standalone Gemma backend connectivity check
- `tests/test_pipeline_smoke.py` — end-to-end smoke test with a stubbed
  Gemma client (no live network required)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if you want to point at a hosted Gemma API instead of a local
Ollama server (set `GEMMA_BACKEND=api` and fill in `GEMMA_API_KEY`).

## Run the smoke test

```bash
pytest
```

This runs the full pipeline (`ingest -> analyze -> decide -> log`) against
`DummyAdapter` and a stubbed `GemmaClient`, and asserts every event produces
a valid `AnomalyFinding` and `Decision`. It does not require a running
Ollama server or network access.

## Check Gemma connectivity

```bash
python scripts/check_gemma.py
```

Reports whether the configured backend (local Ollama or hosted API) is
reachable. If it isn't, it fails gracefully with a clear message instead of
crashing — useful for confirming the plumbing before a real model is
available.

## Run the pipeline directly

```bash
python -m src.pipeline
```

Generates a few dummy telemetry events, runs them through the pipeline, and
prints each resulting `DecisionLogEntry` as JSON. Entries are also appended
to `logs/decisions-<date>.jsonl`.

## Next step

`analyze_node` in `src/pipeline.py` currently runs a **placeholder**
anomaly check (just enough to produce a valid `AnomalyFinding`), and
`decide_node` produces a **placeholder** `Decision`. There is no real data
adapter yet beyond `DummyAdapter`. Once the actual Track 2 idea/prompt is
known, fill in:

- A real `DataSourceAdapter` subclass for the actual telemetry source
- Real anomaly-detection logic in `analyze_node` (using `GemmaClient` for
  real reasoning instead of a placeholder call)
- Real decision logic in `decide_node`
