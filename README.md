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
GO/NO-GO veto over it — requested as real JSON-schema-constrained output
from Ollama, not parsed out of free text, with the original free-text
parser kept on as a fallback for the one case structured output can't
cover — standing in for the unavailable human (bounded so it can only
veto an already-verified-safe maneuver, never approve one the physics
hasn't cleared); the hosted API is treated as "ground control reachable"
(a real human must explicitly approve or reject before it executes). A
limited delta-v budget prevents an unlimited number of maneuvers from
executing silently.
Every decision — including whether its explanation came from Gemma or a
deterministic fallback, and who (if anyone) approved a maneuver — is
written to an append-only JSON-lines audit log, along with the real
prompt text that was actually sent for that call, not just the
response. A circuit breaker also short-circuits repeated Gemma calls
against a backend already known to be down (3 consecutive failures
opens it for 60s), instead of every subsequent event paying the full
retry cost.

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
Gemma-narrates design as conjunctions). A third hazard type — attitude/
pointing loss — is necessarily synthetic-only and clearly labeled as
such: unlike orbital position, there's no real public data source for
spacecraft attitude at all (TLEs never encode orientation), so this one
is an honest exception to "real data wherever possible," not a quiet one.

You can also ask the system about its own history: real
retrieval-augmented search (local Ollama embeddings, real
cosine-similarity ranking) lets Gemma answer plain-English questions
about the audit log using only the real entries it retrieves — grounded
and checkable, not guessed.

A CRITICAL finding doesn't just sit in the log waiting to be noticed —
a real webhook (Slack/Discord/Teams-compatible, or any custom receiver)
fires the moment one is logged, reusing the already-generated real
Gemma rationale as the alert body. Disabled by default; opt in with
`ALERT_WEBHOOK_URL`.

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
- `src/ingestion/attitude_adapter.py` — synthetic attitude/pointing-loss
  fixture (no real public data source exists for spacecraft attitude at
  all, unlike orbital position), spanning all four severity bands
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
- `src/rag.py` — retrieval-augmented search over the real audit log:
  local Ollama embeddings, real cosine-similarity ranking, Gemma
  answering from only the retrieved entries
- `src/alerting.py` — real webhook alert (Slack/Discord/Teams-compatible)
  fired the moment a CRITICAL decision is logged; disabled by default
- `src/trends.py` — severity mix per day, recurring real objects, and
  Gemma-vs-fallback narration mix over time, aggregated from the real
  accumulated log
- `src/preflight.py` — config/connectivity/filesystem health checks
- `src/auth.py` — real operator token authentication for the dashboard
  (see [`ROADMAP_TO_PRODUCT.md`](ROADMAP_TO_PRODUCT.md) Phase 5)
- `src/catalog_screening.py` — a fast apogee/perigee altitude-range
  overlap filter (Phase 3) so conjunction screening scales past a
  curated sample toward a real catalog
- `src/pc_severity.py` — real probability-of-collision severity
  classification from a Space-Track CDM, when one is available (Phase 2)
- `src/postgres_logging.py` — a real Postgres-backed alternative to the
  JSONL audit log, for continuous/concurrent operation (Phase 4), plus
  real retention deletion (`delete_older_than`/`count_older_than`)
- `src/rate_limit.py` — a real token-bucket rate limiter for the REST API
- `src/metrics.py` — real Prometheus-format metrics for the REST API
- `src/ingestion/spacetrack_client.py` / `spacetrack_adapter.py` /
  `cdm_enrichment.py` — an alternative, credentialed data source to
  CelesTrak (Phase 1)
- `scripts/run_demo.py` — **the guided, step-by-step CLI demo** (see below)
- `scripts/dashboard.py` — **the live browser dashboard** (see below)
- `scripts/api.py` — a real REST API over the same audit log, for
  programmatic integration (Phase 6, see below)
- `scripts/scheduler.py` — continuous background screening loop for
  unattended/production operation (see below)
- `scripts/healthcheck_scheduler.py` — real Docker HEALTHCHECK for the
  scheduler service, since it has no HTTP server of its own to probe
- `scripts/retention_cleanup.py`, `scripts/backup_postgres.sh` — real
  retention/backup tooling for the Postgres backend (see below)
- `scripts/query_log.py` — standalone CLI for "ask about the mission log"
  (see `src/rag.py`) outside the browser
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
- [Docker](https://www.docker.com) — only needed for the Docker
  deployment path below; the local setup above needs neither Docker nor
  Postgres

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
5. Optional, only for "ask about the mission log" (see below): pull an
   embedding model —
   ```bash
   ollama pull nomic-embed-text
   ```
   This is required for mission-log search specifically, regardless of
   which `GEMMA_BACKEND` you're otherwise using — the hosted API has no
   embedding endpoint wired up here, so this always goes through Ollama.

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

### 3. Point it at your own satellite (optional)

Every screen defaults to CelesTrak's own demo groups — real objects, but
not *your* asset (`stations`, the crewed-stations group, is the closest
stand-in). If you operate a real satellite, set `WATCHED_NORAD_IDS` in
`.env` to its NORAD catalog ID (comma-separated for more than one, e.g.
`WATCHED_NORAD_IDS=25544,48274`) to monitor it specifically — every
conjunction/decay screen (dashboard buttons, `scripts/scheduler.py`) then
watches those exact objects, cross-screened against the real debris
field, instead of the demo placeholder. The dashboard sidebar shows which
mode is active ("Monitoring your own asset(s)" vs. the demo-group
notice). Leave unset to keep using the zero-setup demo groups.

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
assets actually are right now, independent of any logged event. An "Ask
about the mission log" box answers plain-English questions about the log
itself with real retrieval-augmented search (see below). A "Trends"
section aggregates the accumulated log itself — severity mix per day,
recurring real objects across scans, Gemma-vs-fallback narration mix
over time — the first view that looks at the log's history instead of
one event or one instant. Sidebar buttons can generate real new activity
(a live CelesTrak conjunction scan, a real decay/re-entry risk screen,
the synthetic CRITICAL scenario, the synthetic attitude/pointing-loss
scenario, or a historical replay) without leaving the browser. Opens at
`http://localhost:8501` by default. A sidebar notice shows whether it's
watching your own configured asset (`WATCHED_NORAD_IDS` — see Setup
above) or CelesTrak's demo `stations` group.

## Run the REST API (programmatic access)

```bash
uvicorn scripts.api:app --reload
```

The dashboard is for a human; this is for an operator's own
mission-control software. A real FastAPI service over the exact same
audit log (`GET /decisions`, `/decisions/{event_id}`,
`/decisions/pending-approval`, `/stats/summary`; `POST
/decisions/{event_id}/approve|reject|review`) — see
[`ROADMAP_TO_PRODUCT.md`](ROADMAP_TO_PRODUCT.md) Phase 6. Interactive
docs at `http://localhost:8000/docs` once running. Reads stay open if
`OPERATOR_TOKENS` isn't configured (same zero-setup default as the
dashboard, with a visible `X-Warning` response header); writes always
require a real token — `Authorization: Bearer <token>` — and refuse to
run at all (503) if none are configured, stricter than the dashboard's
free-text fallback since a programmatic caller has no human-readable-name
equivalent.

**Rate limited by default** — a real token-bucket limiter
(`src/rate_limit.py`), per authenticated operator or source IP,
`API_RATE_LIMIT_PER_MINUTE` (default 120/min, `0` disables it). Exceeding
it returns `429` with a real `Retry-After` header.

**Real Prometheus metrics at `GET /metrics`** (`src/metrics.py`) — the
conventional scrape path, no authentication required (a scraper is
infrastructure, not a human operator). Decision counts by severity and
maneuver status are computed fresh from the real audit log every scrape;
real HTTP request/rate-limit counters track the API process itself,
labeled by route *template* (`/decisions/{event_id}`, not the resolved
path) so real per-object ids never blow up label cardinality.

## Run the scheduler (continuous / unattended operation)

```bash
python scripts/scheduler.py                          # real cadence: ticks every ~8h, runs until Ctrl-C
python scripts/scheduler.py --interval-seconds 60     # faster cadence, for demo/testing
python scripts/scheduler.py --max-iterations 3        # stop after N ticks, for testing
```

Every other way to run this project is on-demand (open the dashboard and
click a button, or run the guided demo once). This runs the same real
conjunction and decay screens continuously, on a fixed interval, and
fires a distinct system-health alert (separate from a CRITICAL-finding
alert) if it fails several ticks in a row — see
[`ROADMAP_TO_PRODUCT.md`](ROADMAP_TO_PRODUCT.md) Phase 4/5. Uses
whichever `DecisionLogger` backend is configured (JSONL files by
default, or a real Postgres database via `DATABASE_URL` — JSONL's
whole-file rewrites aren't safe under the concurrent writes a scheduler
running alongside a dashboard/CLI operator would produce). Watches your
own `WATCHED_NORAD_IDS` (see Setup above) when configured — this is the
real, unattended, 24/7 form of "protect my satellite," not just the
interactive dashboard buttons.

## Production deployment (Docker)

```bash
cp .env.example .env   # fill in real values first
docker compose up -d
```

Brings up four real services: a Postgres database, the dashboard
(`http://localhost:8501`), the REST API (`http://localhost:8000`), and
the scheduler above — see `docker-compose.yml`/`Dockerfile`. Requires a
real `POSTGRES_PASSWORD` in `.env` (`docker compose up` refuses to start
without one, rather than silently using a guessable default). Secrets
come from `.env` via `env_file`, never baked into the image. Set
`OPERATOR_TOKENS` in `.env` before exposing the dashboard or API beyond
localhost — see `src/auth.py`; without it, the dashboard's "Operator
name" field is free text anyone with the page open can set to anything,
and the API's write endpoints refuse to run at all.

All four services report real Docker health status (`docker compose
ps`), not just Postgres: the dashboard via Streamlit's own
`/_stcore/health`, the API via its real `/health` (which actually
touches storage, not just "the process is up"), and the scheduler via
`scripts/healthcheck_scheduler.py`, which checks a heartbeat file
`scripts/scheduler.py` writes at startup and after every tick — the
only one of the three with no HTTP server to probe directly.

## Retention and backup (Postgres backend only)

```bash
python scripts/retention_cleanup.py --days 365          # delete audit rows older than 365 days
python scripts/retention_cleanup.py --days 365 --dry-run  # report what WOULD be deleted, no changes
./scripts/backup_postgres.sh                             # real pg_dump to ./backups/
```

Only applies when `DATABASE_URL` is set — the JSONL backend's own
retention/backup story is already simple (one file per real day under
`logs/`; delete old ones or copy the directory directly). Retention
requires an explicit `--days`/`RETENTION_DAYS` — there's no default
window, so a misconfigured cron job can't silently delete real audit
history. Restore a backup with `pg_restore --dbname="$DATABASE_URL"
--clean --if-exists backups/decisions-<timestamp>.dump`. Neither script
is wired into a scheduler automatically — run them from host cron, a
Kubernetes CronJob, or similar, matching how any other periodic
maintenance job would be operated in a real deployment.

## Ask about the mission log

```bash
python scripts/query_log.py "which CRITICAL events were vetoed and why?"
```

Real retrieval-augmented search over the real audit log, not fine-tuning:
every logged decision gets embedded via a local Ollama embedding model
(`nomic-embed-text` by default — cached to disk so an unchanged log isn't
re-embedded every query), your question gets embedded the same way, real
cosine similarity ranks every entry against it, and Gemma answers using
*only* the retrieved real entries — it's explicitly instructed to say so
if they don't contain enough information, rather than guessing. Prints
which real `event_id`s the answer was grounded in. Same feature is also
available in the dashboard. Requires a reachable local Ollama for
embeddings specifically, even if `GEMMA_BACKEND=api` for narration
elsewhere — see `src/rag.py`.

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
- [`ROADMAP_TO_PRODUCT.md`](ROADMAP_TO_PRODUCT.md) — what's missing to go from
  this submission to a real product, and progress against that plan
- [`KAGGLE_WRITEUP.md`](KAGGLE_WRITEUP.md) — submission writeup (architecture, Gemma usage, engineering hurdles, design choices)

## License

[MIT](LICENSE)
