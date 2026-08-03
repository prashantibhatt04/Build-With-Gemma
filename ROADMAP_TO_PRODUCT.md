# Roadmap: from hackathon submission to a real product

`PROJECT_OVERVIEW.md` documents what's built and why. This document is
the opposite direction — what's honestly still missing to make this
usable by a real satellite operator, not a demo audience, and the plan
for closing that gap. Written after an explicit gap-analysis discussion
(see chat), not guessed.

Unlike `PHASE_PROGRESS.md` (a retroactive build log, updated after each
phase lands), this file is written forward and kept as the live plan —
phases get marked done here too, but the point of this file is to state
the destination before building toward it.

## Why these phases, in this order

Four separate gaps stand between this project and a real product: the
**physics** (distance thresholds vs. real collision probability), the
**data** (CelesTrak's low-precision public TLEs vs. real tracking data),
the **scale** (pairwise screening vs. a full-catalog pipeline running
continuously), and **trust** (a demo audit log vs. something a real
operator's insurer/regulator would accept). They're ordered so each
phase either unblocks the next one or is independently valuable on its
own if the roadmap stops partway.

## Phase 1 — Space-Track.org integration
Status: done — live-verified against a real account, including two real
bugs found and fixed that only a live account could have surfaced

Replace CelesTrak as the primary data source with
[Space-Track.org](https://www.space-track.org) (the official US Space
Force catalog, free registration required — a human has to sign up,
this can't be automated on anyone's behalf). This unblocks two things
CelesTrak structurally cannot provide:

- **Real Conjunction Data Messages (CDMs)** — the actual per-conjunction
  predictions professional operators act on, including position/velocity
  **covariance**, which Phase 2 needs.
- **Real historical TLE archives** — Phase 12's historical replay
  currently has to reuse *today's* TLE for a 2009 event because
  CelesTrak's public endpoint ignores any epoch parameter; Space-Track
  can serve the TLE that was actually valid at the time, so a replay
  becomes a genuine historical propagation instead of a stand-in.

Scope: a new `SpaceTrackAdapter` alongside (not replacing)
`CelesTrakAdapter` — CelesTrak stays as the always-available fallback
requiring no credentials, consistent with this project's existing
local/cloud honest-fallback pattern. Requires `SPACETRACK_USERNAME` /
`SPACETRACK_PASSWORD` config, defaulted empty (feature no-ops without
them, same treatment as `ALERT_WEBHOOK_URL`).

## Phase 2 — Real probability-of-collision severity
Status: done — live-verified end to end, including a real correction to
an earlier, overly pessimistic finding about CDM access

Today's severity is a plain distance threshold (`< 5km = CRITICAL`).
Real conjunction assessment computes **Pc** (probability of collision)
from both objects' full position/velocity covariance — a tight, well-
tracked close pass can be far less dangerous than a loose one, something
a pure distance threshold structurally cannot see. This is the single
change that would most credibly move severity classification from "toy"
to "how it's actually done."

Depends on Phase 1: covariance only exists in real CDMs, not in TLEs.
Scope: for objects with a real CDM available, classify by Pc against the
industry-standard thresholds (e.g. Pc > 1e-4 is the common
maneuver-consideration threshold); for objects without one (most of the
catalog, most of the time), fall back to today's distance threshold —
same "degrade honestly, don't fail" pattern as the rest of this project.
The two paths must be clearly distinguishable in the audit log, not
silently blended.

## Phase 3 — Full-catalog screening pipeline
Status: done — the only phase so far needing no external credentials, so
it's fully live-verified, not just built

Today's screening is pairwise across a curated sample (~120 objects).
Space-Track's full catalog is 40,000+ tracked objects and growing. Add a
fast geometric pre-filter (apogee/perigee range overlap) across the
*entire* catalog before any expensive propagation runs, so only pairs
that could plausibly conjunct ever reach the existing coarse/fine
propagation search — the physics and severity logic from Phases 1-2
don't change, just what volume of data they can be run against.

## Phase 4 — Continuous background operation
Status: done — live-verified against a real local Postgres instance and
a real running scheduler process

Today this runs on demand — open the dashboard, click a button, get a
snapshot. A real operator needs continuous screening (Space-Track CDMs
refresh roughly every 8 hours) with proactive alerting, not "run it and
look." Scope: a scheduled worker process, and — the audit log's current
JSONL-file format won't hold up under continuous concurrent writes at
this volume — a real database (Postgres) behind `DecisionLogger`,
keeping its existing interface so nothing downstream (dashboard, RAG,
trends, alerting) needs to change, only its storage backend.

## Phase 5 — Operational trust
Status: partially done — dashboard auth, scheduler self-health alerting,
and a real deployment bundle are built and live-verified; historical
Pc-validation against a reference implementation remains blocked on the
full `cdm` class's real access requirement (confirmed live - see Phase 2)

Process and validation work, not a single feature: cross-checking Pc
results against a reference implementation (e.g. NASA CARA) on more
historical events than just 2009 Iridium/Cosmos, real deployment
(hosting, secrets management, dashboard auth so it isn't an open
localhost page), monitoring/alerting on the system's own health (not
just the conjunctions it finds), and explicit documentation that
Gemma's role stays advisory — never decisional — for anyone evaluating
this for real operational use. This is what turns "the code works" into
"an operator's insurer would accept this."

## Phase 6 — REST API layer
Status: done — live-verified standalone and inside the real Docker stack

Added after Phases 1-5: with Phase 1/2 blocked on a real Space-Track
account, this is real product work that needs no external credentials at
all. Every way to get data out of this system today is human-facing — the
dashboard UI, or reading the audit log/database directly. A real
operator's own mission-control software needs to query findings and
resolve pending approvals *programmatically*, not by having a person
click through a browser. This is the actual gap between "has a UI" and
"is a product other systems can integrate with."

Scope: a real FastAPI service (this project's schemas are already real
Pydantic v2 models — `DecisionLogEntry` and friends serve directly as
FastAPI response models, no translation layer needed) over the exact
same `DecisionLogger` every other surface already uses (JSONL or
Postgres, whichever is configured — nothing API-specific about storage).
Read endpoints for decisions/pending-approvals/metrics; write endpoints
(approve/reject/mark-reviewed) reuse the same `OPERATOR_TOKENS` real
authentication Phase 5 already built for the dashboard — not a second,
weaker auth scheme.

## Explicitly out of scope for this roadmap

- **Real attitude telemetry** — stays synthetic; it's proprietary
  per-operator data with no public source, not a gap this project can
  close generically (see `PROJECT_OVERVIEW.md`'s attitude section).
- **A second, real maneuver-planning model for decay/attitude** — a
  genuinely separate astrodynamics problem, deliberately scoped out
  when decay/attitude were first built, unchanged by this roadmap.

## Progress log

(Updated as phases complete — mirrors `PHASE_PROGRESS.md`'s style but
scoped to this roadmap only.)

### Phase 1 — built, then genuinely live-verified against a real account, two real bugs found and fixed

Added, all following this project's existing adapter/config conventions:
`src/config.py`/`.env.example` (`SPACETRACK_USERNAME`/`SPACETRACK_PASSWORD`,
empty-default no-op); `src/ingestion/spacetrack_client.py`'s
`SpaceTrackClient` (real session auth, rate-limit-aware query helpers,
`fetch_historical_tle_text`); `src/ingestion/tle_source.py`'s
`fetch_spacetrack_group_text`; `CelesTrakAdapter`'s injectable
`fetch_group_fn`/`SOURCE_LABEL`; `SpaceTrackAdapter` as a thin subclass
reusing the entire cross-group screening algorithm.

**A real account was created and credentials configured — here's what
actually happened testing against it, not what was assumed beforehand:**

Real authentication worked on the first attempt (`ajaxauth/login`, a real
`GET class/gp/NORAD_CAT_ID/25544` returned real, current ISS element
data). Two real, previously-undetectable bugs surfaced immediately after:

1. **Space-Track's `format=tle` returns bare 2-line elements with NO name
   line at all** — confirmed live (`repr()` of the raw response showed
   exactly two lines per object) — unlike CelesTrak's own `FORMAT=tle`,
   which always includes one, and unlike what this code originally
   assumed. Feeding that 2-line stream through `parse_tle_blocks`
   (written and tested against a 3-line format) silently corrupted
   results: of a real 160-object `~~ISS` query, every other object was
   dropped outright, and the rest had a neighboring object's orbital
   element line substituted for their name. Fixed by requesting
   Space-Track's `3le` format instead (confirmed live: it prefixes each
   name line with a literal `"0 "` per the real CCSDS/3LE convention,
   e.g. `"0 ISS (ZARYA)"` — stripped so the parsed name matches
   CelesTrak's own unprefixed convention exactly).
2. **Space-Track's raw GP class returns every historical elset ever
   published for a matching name, including objects with decades-stale
   epochs.** Confirmed live: of 160 real `~~ISS`-matching objects, 159
   had epochs ranging from 1998 to 2025 (real ISS debris fragments
   tracked historically, many since decayed) — propagating those forward
   to "now" isn't just inaccurate, SGP4 breaks down and silently returns
   NaN position, which would have flowed straight into `min_distance_km`
   with no exception raised. Fixed by adding `DECAY_DATE/null-val` +
   `EPOCH/>now-30` to the query — confirmed live: the same query then
   returned 13 real, current objects, all 13 propagating cleanly.

With both fixes in place, `SpaceTrackAdapter(client=...).fetch_batch(limit=10)`
was run live end to end: 10 real conjunction events, real named objects
(`COSMOS 2251 DEB` vs. `COSMOS 2251 DEB`), real sensible distances
(17–33km), zero NaN. `fetch_historical_tle_text` was also confirmed live
against the real 2009 Iridium 33 event (NORAD 24946): a real element set
with an epoch of day 40.78 of 2009 — i.e. February 9th, 2009, the day
before the actual collision — genuinely close to the real event, not a
current-day stand-in.

21 tests (`test_spacetrack_client.py`, `test_spacetrack_adapter.py`,
additions to `test_tle_source.py`), all updated to match the real,
live-confirmed shapes above rather than the original (partially wrong)
assumptions. Full suite: 307/307.

**Wired into the dashboard, and confirmed in an actual browser, not just
a script.** `EnrichedSpaceTrackAdapter` (`src/ingestion/spacetrack_adapter.py`)
composes `SpaceTrackAdapter` with Phase 2's real CDM enrichment
(`enrich_conjunction_events_with_pc`) into one adapter that drops
straight into `run_once` - no changes needed to `pipeline.py`. A new
dashboard sidebar button ("Fetch Space-Track conjunctions (real Pc when
available)") only appears when `SPACETRACK_USERNAME`/`SPACETRACK_PASSWORD`
are configured, with a visible caption explaining how to enable it
otherwise - same "explicit, not silent" gating this project uses for
every optional integration. Live-verified by actually starting the real
dashboard, clicking the real button in a real browser, and watching
"Total events" go from 246 to 251 - 5 real Space-Track conjunctions,
correctly labeled `source="spacetrack"`, correctly classified via the
distance-threshold path (`severity_source="distance-threshold"` - no CDM
match this run either, same honest expected outcome as Phase 2's own
scripted test), with real Gemma narration (`rationale_provenance.source
== "gemma"`) for every one of them, not fallback text. 3 new tests
(`EnrichedSpaceTrackAdapter` in `test_spacetrack_adapter.py`; the gated
button's presence/absence in `test_dashboard_app.py`, using the same
frozen-singleton `object.__setattr__` override pattern already
established for `OPERATOR_TOKENS` testing). Full suite: 312/312.

### Phase 2 — built, then live-verified with a real correction to the original finding

Added: `AnomalyFinding.severity_source`; `src/pc_severity.py`
(`classify_pc_severity` — real NASA ISS Pc thresholds; `extract_cdm_summary`);
`SpaceTrackClient.fetch_recent_cdms()`; `src/ingestion/cdm_enrichment.py`;
`pipeline.py`'s Pc-vs-distance dispatch; `severity_source` surfaced in
`display.py`/`dashboard_data.py`.

**The original finding (CDM access requires owner/operator privileges)
was half right, and live testing against the real account found the
better other half.** Querying the full `class/cdm` live returned a real
`{"error":"Your Class Does Not Exist "}` 400 response — confirming that
class genuinely is inaccessible to a generic account, exactly as the
original research concluded. But Space-Track also publishes a *separate*,
lower-detail `cdm_public` class — real, current conjunction data
(`SAT_1_ID`/`SAT_2_ID`/`PC`/`TCA`/`MIN_RNG`), accessible to *any*
registered account, no owner/operator status required. This was not
found during the original documentation research and is a materially
better outcome: confirmed live with a real, current row (`CZ-6A R/B` vs.
`CZ-6A DEB`, real `PC` of 0.0016). `extract_cdm_summary` and
`fetch_recent_cdms` were rewritten to match `cdm_public`'s real field
names (not the full `cdm` class's, which this code originally assumed).

Full real end-to-end integration test: fetched 20 real conjunctions via
`SpaceTrackAdapter`, fetched real current CDMs via `fetch_recent_cdms`,
ran `enrich_conjunction_events_with_pc` — 0 matches (expected and
correct: `cdm_public` covers the whole real catalog, not specifically
the narrow `stations`/`cosmos-2251-debris` groups this adapter screens),
and confirmed every event correctly fell through to the real
distance-threshold path with `severity_source="distance-threshold"`, no
errors anywhere in the chain. The Pc path itself is confirmed real and
reachable — not permanently dormant as first documented — it's just that
this specific screening scope rarely intersects with the specific pairs
`cdm_public` happens to cover on any given day, the same "real data
rarely produces the most severe case" pattern already established for
CRITICAL conjunctions and decay risk. A genuinely promising follow-up,
not pursued here to keep this verification pass scoped: screening
`cdm_public` more broadly (not just matching against an already-narrow
sample) could surface real Pc-classified findings directly, independent
of CelesTrak/Space-Track TLE screening entirely.

7 tests rewritten (`test_pc_severity.py`, `test_cdm_enrichment.py`,
`test_spacetrack_client.py`'s CDM test) to match the real, live-confirmed
`cdm_public` field names. Full suite: 307/307.

### Phase 3 — built and fully live-verified (no credentials needed)

Added: `orbital.perigee_apogee_altitude_km` (promoted out of
`src/decay.py`, which now reuses it — a second real consumer justified
sharing code, matching this project's own established practice);
`src/catalog_screening.py`'s `apogee_perigee_overlap_pairs` — a real
O(n log n + k) interval-overlap sweep (sort by perigee, sweep an
active set keyed by apogee) replacing the O(n²) `itertools.combinations`
every adapter previously used to enumerate candidate pairs before any
propagation ran. Two objects whose real altitude ranges never overlap
can never be close to each other regardless of where either is right
now — a real, standard technique (the same kind of "AP filter" CelesTrak's
own SOCRATES uses as a first pass), not a heuristic invented for this
project. `CelesTrakAdapter._rank_conjunctions` (and, by inheritance,
`SpaceTrackAdapter`) now runs every pair through this filter first;
`last_scan_stats` gained `total_possible_pairs`/`pairs_after_ap_filter`
so the filter's real effect is always visible, not just claimed. 9 new
tests for the filter itself (correctness checked directly against a
naive O(n²) reference, not just spot-checked, plus edge cases: touching
endpoints, full containment, empty/single-object input, a margin-based
gap-bridging case) — all in `test_catalog_screening.py`. The existing
adapter test fixtures (`test_celestrak_adapter.py`,
`test_spacetrack_adapter.py`) needed real rework here, not just a
mechanical update: their old hand-picked TLEs (Vanguard 1, ISS, a GTO-like
"third test sat", LAGEOS 1/2) spanned wildly different real altitude
bands and never actually overlapped — harmless before Phase 3 (screening
didn't check plausibility), but a real filter correctly stripped every
pair out once it did. Replaced with 5 fixtures sharing one ~400-408km LEO
band (small RAAN/mean-anomaly perturbations of the real ISS TLE, the same
way real debris from one breakup shares its parent's orbital regime) —
more physically honest test data, not a workaround. Full suite: 262/262.

**Live-verified at real, meaningful scale — the one phase so far that
needed no external account to prove for real:**

- Fetched CelesTrak's real `active` group live: **16,108 real tracked
  objects**, ~129.7 million possible pairs. Running just the AP filter
  (perigee/apogee extraction + the overlap sweep, no propagation) over
  all of them took **0.41 seconds** and cut candidates to **13.4 million
  (10.3% of total)** — a real ~90% reduction at real full-catalog-adjacent
  scale, confirming the filter itself doesn't become the bottleneck even
  at this size.
- Ran the full `CelesTrakAdapter.fetch_batch()` pipeline end-to-end
  against **3,000 real active objects** (30x this project's previous
  default sample size) — completed in **14.29 seconds**, with
  `last_scan_stats` showing 4,498,500 possible pairs reduced to 345,671
  AP-filtered candidates (7.7%), and a real top result surfaced
  (ISS (ZARYA) vs. ISS (UNITY) at 0.00km — two real CelesTrak catalog
  entries for parts of the same physical station, the same kind of
  "docked/co-located objects read as a false top result" noise Phase 10
  already documented for the `stations` group specifically; unsurprising
  here since `exclude_within_group` wasn't configured for this ad hoc
  single-group benchmark run).

**Honest scope boundary, not overstated:** the AP filter genuinely
solves the specific problem it targeted — the O(n²) pair-*enumeration*
wall that made anything past a few hundred objects impossible to even
attempt. It does **not**, by itself, make a full 16,000+-object catalog
screen instantly: at that real scale, ~13.4 million candidate pairs still
remain, and ranking them by coarse distance is still one Python-level
call per pair (the same loop structure Phase 10 already established) —
that becomes the next real bottleneck, not pair generation. Vectorizing
that ranking step (e.g. batching the numpy distance calculation across
many candidate pairs at once instead of one call each) is a natural
follow-on if a deployment ever needs to screen the entire real catalog
in one pass, not something this phase claims to have already solved.
This project's own default `sample_size_per_group` (100) is deliberately
left unchanged for the demo/dashboard's normal interactive use — Phase 3
is about the underlying capability being real and measured, not about
changing today's demo UX latency.

### Phase 4 — built and live-verified against real infrastructure

Two pieces, matching the phase's scope exactly:

**A swappable storage backend, not a rewrite.** `src/logging_utils.py`
was refactored around a `DecisionLogStore` interface
(`append`/`find`/`update`/`load_all`) — the original file logic became
`JSONLDecisionLogStore`, and `DecisionLogger` itself is now a thin
business-logic wrapper (the actual audit rules in `mark_reviewed`/
`approve_maneuver` are unchanged) that delegates storage to whichever
store it's given. `DecisionLogger`'s own public interface
(`log`/`find_entry`/`load_all_entries`/`mark_reviewed`/`approve_maneuver`)
is identical either way — every existing caller (`scripts/dashboard.py`,
`scripts/run_demo.py`, `pipeline.make_log_node`, RAG, trends, alerting)
needed zero changes. One real interface cleanup: `find_entry` used to
leak a JSONL-specific `(Path, line_index, entry)` tuple; since no
external caller ever used that shape (confirmed by grep — only this
project's own `logging_utils` tests did), it now returns a plain
`Optional[DecisionLogEntry]`, a cleaner contract a Postgres row has no
trouble satisfying too.

**`src/postgres_logging.py`'s `PostgresDecisionLogStore`** — real
`psycopg` (v3) queries against one JSONB-per-entry table (`entry_json`
holds the full entry so the schema can evolve without a migration for
every new field, exactly like JSONL never needed one either; an indexed
`event_id` column gives real fast lookup instead of JSONL's linear file
scan). `DATABASE_URL` (`src/config.py`, empty by default — same
absent-means-fallback pattern as every other optional integration here)
selects it; unset keeps the original JSONL behavior, zero setup required.

**`scripts/scheduler.py`** — a real continuous loop: one shared
`GemmaClient` and `DeltaVBudgetTracker` persist across every tick (not
reconstructed per tick — the exact QA-pass lesson `PHASE_PROGRESS.md`
already documents for the dashboard/`run_demo.py`), ticking on a
configurable interval (defaults to ~8h, matching Space-Track's real CDM
refresh cadence) and running the same real conjunction + decay screens
the dashboard already exposes. A single bad tick is caught and logged,
not fatal — a process meant to run indefinitely can't die because
CelesTrak was briefly unreachable once.

11 new tests: `test_postgres_logging.py` (7, run against a REAL local
Postgres — not mocked, since concurrent-safe storage is exactly the
property this module exists to prove, and a mock would prove nothing)
and `test_scheduler.py` (4, fast/offline via a duck-typed fake client and
`DummyAdapter`, covering `run_tick`'s real behavior — including an
explicit regression-style test that the client/budget tracker really are
reused across calls, not reconstructed). Full suite: 273/273.

**Live-verified, not just built — the two pieces working together for
real, start to finish:** created a dedicated local Postgres database
(`build_with_gemma`) on this machine, then ran `scripts/scheduler.py`
for real with `DATABASE_URL` pointed at it, a real local Ollama backend,
and `--interval-seconds 3 --max-iterations 2`. Both ticks completed
cleanly (exit code 0): tick 1 at `2026-08-03T01:50:41Z` logged 20 real
entries (`{'critical': 1, 'warning': 6, 'watch': 13}`), tick 2 fired
right on schedule ~4 minutes later (real local Gemma narration for ~20
events is what actually took the time, not the interval or the
database) and logged 20 more (`{'warning': 5, 'watch': 15}`) - the
scheduler's own stdout and a direct `psql SELECT COUNT(*)` against the
live table agree exactly: **40 real rows**. Tick 1's CRITICAL was a
genuine real conjunction (COSMOS 2251 DEB vs. COSMOS 2251 DEB, 4.45km,
action=abort), correctly triaged through the full maneuver pipeline and
persisted to Postgres, not synthesized for the test. `psql` queries
against the live table (not the dashboard UI or the script's own
stdout alone) were used throughout as the primary source of truth,
consistent with this project's standing practice of verifying against
raw persisted state rather than trusting a script's own claims about
what it did.

### Phase 5 — three real pieces built and live-verified; one genuinely blocked

Phase 5 is process/validation work, not a single feature (see the phase
description above) — this pass covers the three pieces that were
actually buildable and testable right now, and is explicit about the
one that isn't.

**Real dashboard authentication, closing the "open localhost page" gap
the roadmap named explicitly.** Before this, `scripts/dashboard.py`'s
"Operator name" was free text — anyone with the page open could type
"alice" and approve or reject a real maneuver, or mark a decision
reviewed, as her. `src/auth.py` (`parse_operator_tokens`/`authenticate`)
plus `OPERATOR_TOKENS` (`src/config.py`, `"name:token,name2:token2"`,
empty by default) fixes this: when configured, nothing else in the
dashboard renders until a valid token is entered, and `approved_by`/
`reviewed_by` become the identity the token actually maps to, not
whatever a visitor typed. Left unconfigured (the default, zero-setup
behavior), the dashboard still works exactly as before, but now shows an
explicit "⚠️ Unauthenticated" warning instead of a silent gap. 13 new
tests, including two real `AppTest` interaction tests (not just
unit-level checks of `src/auth.py`) — one confirming the login gate
genuinely blocks all content and rejects a wrong token, one confirming a
correct token unlocks the real dashboard and shows "Signed in as
**alice**".

**Scheduler self-health alerting, distinct from CRITICAL-conjunction
alerts.** `send_health_alert` (`src/alerting.py`) and
`scripts/scheduler.py`'s `update_consecutive_failures` fire a real
webhook (same `ALERT_WEBHOOK_URL`, a visibly different `"⚠️ SYSTEM
HEALTH:"` prefix so it's never confused with a real conjunction alert in
a shared channel) after `CONSECUTIVE_FAILURE_ALERT_THRESHOLD` (3,
matching `GemmaClient`'s own circuit-breaker threshold intentionally)
ticks fail in a row — once per outage, not spammed on every failure
after, and able to fire again after a real recovery. The gap this closes
is real: without it, an operator relying on the scheduler to run
unattended would only learn something was wrong by noticing an unusual
*absence* of new findings, not by being told. Live-verified against a
real local HTTP receiver (the same technique Phase 19's original
alerting used) — a real POST landed with the expected distinct prefix.

**A real deployment bundle, not just instructions.** `Dockerfile` +
`docker-compose.yml`: three services (a real Postgres, the dashboard,
the scheduler), secrets via `.env` + `env_file` (never baked into the
image), `DATABASE_URL` built from `POSTGRES_*` and pointed at the
`postgres` service's Docker-internal hostname (deliberately not reused
from a developer's own host-oriented `.env` value — the two environments
need different values for the same variable). Live-verified for real on
this machine, not just written and assumed correct: `docker compose
build` succeeded, `docker compose up -d` brought up all three containers
(Postgres reported healthy), the dashboard served a real HTTP 200, and —
the strongest proof — a direct `psql` query against the *containerized*
Postgres showed a real logged conjunction event
(`conj-33942-33971-...`, `source=celestrak`) that the *containerized*
scheduler had genuinely fetched from CelesTrak and written through the
real pipeline, entirely inside the compose network. Torn down and its
volumes removed afterward — this was a verification, not something left
running.

**Explicit, in one place: Gemma's role stays advisory, never
decisional.** This has been true in the code since Phase 0 (see
`PROJECT_OVERVIEW.md`'s "Reliability layers" section — severity, action,
and maneuver math are 100% deterministic; the one bounded exception, the
local-path veto, can only ever make an already-verified-safe outcome
*more* conservative, never less) but was never stated as its own
standalone claim for someone evaluating this system's trustworthiness
without reading the rest of the codebase first. Stating it here,
plainly, for that reader: **nothing in this project lets Gemma decide
whether a conjunction is dangerous, whether to maneuver, or what a
maneuver should be** — every one of those is a deterministic threshold
or a closed-form calculation Gemma never touches. Gemma explains
findings, answers questions about the audit log, and (on the
local/autonomous path only) can veto — never approve — a maneuver the
physics has already independently verified safe.

**Honestly blocked, not quietly dropped — and this held up even after
real credentials arrived:** cross-checking this project's severity/Pc
classifications against a reference implementation (e.g. NASA CARA)
across more historical events than the single 2009 Iridium/Cosmos replay
needs real covariance data, not just a Pc value - Space-Track's public
`cdm_public` class (confirmed live and reachable - see Phase 2's
progress-log entry) doesn't carry it; the full `cdm` class does, and a
real live query against it returned a genuine `{"error":"Your Class Does
Not Exist "}` - confirming this specific gap is real, not a documentation
assumption that would dissolve once an account existed. It requires
Space-Track "CDM privileges," generally granted only to a registered
spacecraft owner/operator, which this project structurally cannot
become. This isn't a "not started" gap this roadmap can schedule its way
out of with more engineering time — it's a real access boundary. If this
project (or whoever operates it) ever becomes a real registered
owner/operator, this validation becomes possible and should be treated
as a real follow-up, not before.

17 new tests total (`test_auth.py`, additions to `test_dashboard_app.py`,
`test_alerting.py`, and `test_scheduler.py`). Full suite: 290/290.

**A follow-up self-review pass found and fixed 3 real issues,
matching this project's established practice of not treating "built and
tested" as "done" without a second look (see `PHASE_PROGRESS.md`'s own
QA-pass entries):**

1. `src/auth.py`'s `authenticate()` compared tokens with plain `==` -
   real timing-attack surface (string comparison returns as soon as the
   first mismatched character is found, so response time can leak how
   many leading characters of a guess were right). Switched to
   `hmac.compare_digest`.
2. `docker-compose.yml` defaulted `POSTGRES_PASSWORD` to the literal
   string `"postgres"` if unset - exactly the kind of silent gap this
   phase exists to close, not repeat. Changed to `${POSTGRES_PASSWORD:?...}`,
   which makes `docker compose up` refuse to start with a clear error
   instead of silently running with a guessable password - confirmed
   both directions for real: fails with the intended message when unset,
   succeeds and builds the correct `DATABASE_URL` when set. `.env.example`
   gained the missing `POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD`
   entries it should have had from the start.
3. The scheduler service had no restart policy - a real crash (an
   unhandled error escaping `run_tick`'s own try/except, an OOM, a host
   reboot) would silently end "continuous" operation until a human
   noticed. Added `restart: unless-stopped`.

Full suite re-run after all three fixes: still 290/290 (none touched
tested code paths - two were deployment-config-only, one is covered by
existing `test_auth.py` coverage of `authenticate()`'s behavior, which
is unchanged from the caller's perspective).

### Phase 6 — built and live-verified, both standalone and in Docker

`scripts/api.py`: a real FastAPI service, deliberately thin - every
endpoint delegates to the exact same `DecisionLogger` (JSONL or
Postgres, whichever `DATABASE_URL` selects) and the exact same
`src/dashboard_data.py` transforms `scripts/dashboard.py` already uses,
serving this project's own real Pydantic v2 schemas
(`schemas.DecisionLogEntry` and friends) directly as response models -
no second copy of any aggregation/formatting logic, no translation
layer. Endpoints: `GET /health` (unauthenticated - a real liveness check
that actually touches storage, not just "the process is up"), `GET
/decisions` (filterable by severity/source, paginated), `GET
/decisions/{event_id}`, `GET /decisions/pending-approval`, `GET
/metrics`; `POST /decisions/{event_id}/approve|reject|review`.

Auth reuses `src/auth.py`'s `OPERATOR_TOKENS` scheme from Phase 5
exactly - not a second, weaker scheme. Reads stay open if unconfigured
(same zero-setup default as the dashboard), with a real `X-Warning`
response header standing in for the dashboard's visible banner (a
machine client has no UI to show a warning in). Writes are stricter than
the dashboard on purpose: they refuse to run at all (503) if
`OPERATOR_TOKENS` isn't configured, rather than falling back to anything
resembling free-text identity - a programmatic caller has no "a human
glances at a warning and decides" equivalent. `ValueError`s from
`DecisionLogger` are mapped to real, distinct HTTP semantics a
programmatic caller can branch on: 404 for "doesn't exist", 409 for
"exists but not in a state this action applies to" - collapsing both
into one status code (as the dashboard's single `st.error()` text
effectively does for a human reader) would lose information an
automated caller actually needs.

16 new tests (`test_api.py`), using FastAPI's real `TestClient` (in-
process ASGI calls, not mocked HTTP) against an isolated tmp_path-backed
JSONL store via `app.dependency_overrides` - routing, auth (all four
real outcomes: no header, wrong token, valid token, unconfigured-503),
filtering/pagination, and the 404-vs-409 error mapping are all exercised
for real. Full suite: 306/306.

**Live-verified twice - standalone and inside the real Docker stack:**
first, ran `uvicorn scripts.api:app` directly and confirmed
`scripts.api:app`'s module path resolves correctly from the repo root
(this matters - `uvicorn module:app` isn't guaranteed to find a
same-repo package without confirming it actually does) - a real `GET
/health` returned `200 {"status":"ok","storage_backend":"jsonl"}`.
Second, added a 4th `api` service to `docker-compose.yml` (same
`Dockerfile`, its own port 8000) and brought up the real containerized
stack: `GET /health` correctly reported `storage_backend: postgres`
(confirming `DATABASE_URL` wiring), then a real entry was written from
*inside* the `api` container via the real `DecisionLogger` and
confirmed to appear correctly through `GET /decisions`, `GET
/decisions/{event_id}`, and `GET /metrics` - full real round trip
through Postgres and back out as HTTP JSON, not asserted from a diagram.
Also confirmed live: an unauthenticated `POST .../review` against the
real containerized API correctly returned 503, not a silent no-op.
Torn down and its volumes removed afterward, same discipline as every
other Docker verification in this project.

## Post-Phase-6 improvement — real historical re-propagation, and an honest finding about TLE staleness

Phase 1's `fetch_historical_tle_text` (real `gp_history` queries) was
built and live-verified but deliberately not wired into anything - "an
independent follow-up," per its own docstring at the time. With a real
account in hand, that follow-up: `real_repropagate_event`
(`src/ingestion/historical_adapter.py`) fetches REAL historical element
sets for both objects in the 2009 Iridium 33/Cosmos 2251 collision and
re-runs this project's own real two-pass propagation search
(`orbital.find_closest_approach` - the exact same physics every live
screening call already uses) to independently derive a closest approach,
rather than only replaying the documented 584m SOCRATES number.
`HistoricalReplayAdapter` gained an optional `spacetrack_client` param -
when provided, each event gets this real cross-check attached to
`raw_data` under `real_repropagated_*` keys; a failure is caught and
recorded, never raised, since an optional enhancement must never block
the documented replay it's checking. Severity classification still runs
on the documented numbers, unchanged - this is a genuinely additive
cross-check, not a replacement. Wired into the dashboard's existing
"Replay historical event" button (auto-enabled when Space-Track
credentials are configured) and surfaced in `display.py`. 11 new tests
(`test_historical_adapter.py`). Full suite: 318/318.

**The real finding, run live against the actual account:** the
re-propagated closest approach came out to **181km** - not anywhere
close to the documented 584m - with the predicted time of closest
approach drifting **5 hours** later than what was actually reported in
2009. This is a real, explicable result, not a bug: `gp_history` only
returns whatever element set happened to be published, and for this
query that meant a Cosmos 2251 TLE **over a full day (29 hours) stale**
relative to the actual collision time (Iridium 33's was fresher, ~22
hours). SGP4 propagation error compounds significantly over a
day-plus window - a few seconds of accumulated along-track timing error,
at the ~11.7 km/s relative velocity involved, is enough to turn a
sub-kilometer real geometry into a multi-hundred-kilometer predicted
one. The real 584m SOCRATES prediction was possible specifically because
this conjunction was flagged as concerning enough that week to be
tracked with much fresher, more frequently updated data (14 SOCRATES
reports in the week leading up to it) - a real advantage a generic
historical query into whatever Space-Track happened to have on file
can't reproduce after the fact.

This is worth stating plainly rather than downplaying: **a two-pass
SGP4 search from a day-old TLE cannot reproduce an official sub-kilometer
conjunction prediction, and shouldn't be expected to.** It's the same
underlying limitation this project's own maneuver math and decay
assessment already document as "simplified, not flight-software-grade"
- and it's a genuine, concrete illustration of exactly why real
operational conjunction assessment needs frequently refreshed tracking
data (ideally same-day) and full covariance (Phase 2), not point-position
propagation from whatever archival element set happens to be available.
The real cross-check is still valuable specifically *because* it
surfaces this honestly instead of quietly repeating the documented
number and implying more independent confirmation than actually exists.

## Post-Phase-6 improvement — rate limiting, Prometheus metrics, Postgres retention & backup

Three real, independent hardening items, all applying to infrastructure
already live-verified in earlier phases rather than new surface area:

**Rate limiting** (`src/rate_limit.py`): a real in-process token-bucket
`RateLimiter`, one bucket per client (authenticated operator name, or
source IP when unauthenticated), with an injectable clock for
deterministic tests rather than sleeping real wall-clock time. Wired
into `scripts/api.py` as request middleware ahead of routing, returning
a real `429` with a `Retry-After` header computed from actual remaining
refill time - not a fixed guess. On by default (`API_RATE_LIMIT_PER_MINUTE`,
120/min) since a baseline abuse guard shouldn't need opt-in the way
optional features in this project do; `0` disables it. Live-verified
standalone: ran `uvicorn scripts.api:app` with
`API_RATE_LIMIT_PER_MINUTE=3` and confirmed the 4th request within a
minute really returned `429` with a real `Retry-After` value, while the
first 3 succeeded.

**Prometheus metrics** (`src/metrics.py`): a real `prometheus_client`
`CollectorRegistry` (a dedicated one, not the process-global default, so
tests don't leak state into each other), exposed at `GET /metrics` with
no auth (matching how Prometheus itself expects to scrape - a metrics
endpoint that needs a bearer token per-scrape target is unusual and
adds real operational friction for no benefit here). Two real
monotonic `Counter`s (`http_requests_total`, `rate_limited_requests_total`)
track process-level facts the audit log doesn't capture. A custom
`DecisionLogCollector` recomputes decision-derived gauges
(`mission_ops_decisions_total`, by-severity breakdown, Gemma-rationale
ratio, maneuver status) fresh on every real scrape rather than caching -
correctness over marginal scrape cost, consistent with this project's
"deterministic, not fabricated" ethos. Request counting labels by the
matched FastAPI **route template** (`request.scope["route"]`), not the
resolved path - deliberately, since labeling by real event IDs would
give Prometheus unbounded label cardinality in production. A real bug
was caught here by the tests, not shipped: the collector's first
version called `get_logger()` directly at scrape time, bypassing
FastAPI's `app.dependency_overrides` - the exact mechanism every test
uses to point the API at an isolated log - so a test asserting an
isolated 1-entry log instead saw the real, ambient 251-entry
`.env`-configured log leak in. Fixed by having the collector consult
`app.dependency_overrides.get(get_logger, get_logger)` at call time.
Live-verified standalone with `curl http://localhost:8124/metrics`
against a real running server before wiring into Docker; renamed the
old `/metrics` JSON summary endpoint to `/stats/summary` to free up the
real Prometheus exposition-format path.

**Postgres retention & backup** (`scripts/retention_cleanup.py`,
`scripts/backup_postgres.sh`): retention deletes by `created_at` (the
real insert timestamp Postgres itself assigns), not any timestamp
inside an entry's own payload - a historical replay's real 2009
collision timestamp must not make a row logged today look 16 years old.
Deliberately refuses to run without an explicit, positive `--days` or
`RETENTION_DAYS` - no default retention window, since a cron job
silently deleting real audit history because someone forgot to set a
value is a genuinely dangerous default to have. `--dry-run` reports a
real count via a new `PostgresDecisionLogStore.count_older_than()`
without deleting anything. Backup is a real `pg_dump --format=custom`
wrapper (not a placeholder), with a documented restore command via
`pg_restore --clean --if-exists`. Only applies to the Postgres backend;
both scripts no-op cleanly (exit 0) when `DATABASE_URL` isn't
configured, since the JSONL backend's retention/backup story is already
just "manage the date-stamped files directly."

**The real finding, from live-verifying the backup script:** `pg_dump`
failed on first run with "aborting because of server version mismatch"
- this machine's default `pg_dump` on `PATH` (Homebrew's postgresql@16,
16.14) was older than the actual running Postgres server (postgresql@17,
17.10), a genuine multi-version-install situation rather than a script
defect. Resolved for verification by pointing at the matching
`postgresql@17` binary directly; documented in the script's own header
comment so the next person hitting this doesn't mistake it for a bug.
Retention was live-verified with a real `--dry-run` against the actual
local Postgres audit table (40 real rows accumulated across this
session's testing), then a real (non-dry-run) delete confirmed against
a short cutoff, and the resulting row count reduction was directly
observed via `psql`.

18 new tests across `test_rate_limit.py` (7, real elapsed-time refill
behavior, not just capacity math), `test_metrics.py` (7, including the
dependency-override bug above and a route-template-not-raw-path
cardinality-safety check), and `test_postgres_logging.py` (4 new,
`count_older_than`/`delete_older_than` run against a real local
Postgres table, not mocked), plus API-level rate-limit/metrics tests
folded into `test_api.py`. Full suite: **344/344**, including all 11
Postgres-backed tests in `test_postgres_logging.py` running against a
real reachable local Postgres instance (not skipped).

## Post-Phase-6 improvement — real Docker healthchecks for all four services

A real operational gap, not a hypothetical one: `docker-compose.yml`
only ever had a `healthcheck` on the `postgres` service. `dashboard`,
`scheduler`, and `api` had none - Docker (or any orchestrator reading
its health status) had no way to tell a hung or silently-crashed-but-
still-running application process from a healthy one; only an outright
container exit was visible.

Closed for all three, each using infrastructure that already existed
rather than new surface area: **`api`** gets a real `python -c
"urllib.request.urlopen(...)"` probe against its own already-built
`GET /health` (Phase 6/REST API section above) - the same endpoint that
actually touches storage, not just confirms the process is up.
**`dashboard`** probes Streamlit's own built-in `/_stcore/health`
endpoint the same way - real and standard, not something this project
added. Neither service needed `curl` installed into the
`python:3.12-slim` image just for this; `python`'s own `urllib` was
already there.

**`scheduler`** has no HTTP server to probe at all, so it needed a real
new mechanism: `scripts/scheduler.py` now writes a small heartbeat file
(`data/scheduler_heartbeat.json` - the same relative `data/...`
convention `tle_source.py`/`rag.py`'s cache dirs already use) at startup
and after every tick, *whether that tick succeeded or failed* -
deliberately a liveness signal ("the loop is still alive and
progressing"), not a correctness one (the existing consecutive-failure
alerting from Phase 5 already covers "running but broken"). The
heartbeat carries its own `interval_seconds`, so freshness checking
never needs external knowledge of a given deployment's configured
`--interval-seconds`. A new `scripts/healthcheck_scheduler.py` reads it
and exits 0/1 - the real Docker `HEALTHCHECK CMD`. Freshness threshold
is `2 × interval_seconds + 5 minutes`: one full interval of slack for a
single legitimately slow tick (a slow Space-Track response must not
itself look like a hang, since this scheduler sleeps *after* each tick
rather than firing on a hard timer) plus a second interval of margin
before concluding the loop itself is actually stuck.

10 new tests (`test_scheduler.py`: heartbeat writing, freshness math at
both a 60s and the real 8-hour default interval, the "one slow tick is
fine" boundary; `test_healthcheck_scheduler.py`: the real 0/1 exit-code
contract for fresh/stale/missing/malformed heartbeat files). Full
suite: **354/354**.

**Live-verified against a real `docker compose up`, not just unit
tests:** built and brought up the full real four-service stack; within
its healthchecks' `start_period` windows, `docker compose ps` reported
all four services - not only `postgres` - as `(healthy)`. Directly
inspected the real heartbeat file inside the running `scheduler`
container (`{"timestamp": ..., "interval_seconds": 28800, "tick_number":
0}` - the real 8-hour default, tick 0 since no tick had completed yet
in that short a window) and ran `scripts/healthcheck_scheduler.py`
inside the container by hand, confirming a real exit 0. Then forced a
real negative case rather than only trusting the unit tests: overwrote
that same real heartbeat file's timestamp to a stale value inside the
running container and re-ran the healthcheck script, which correctly
exited 1 with an "unhealthy: heartbeat is stale" message - the exact
mechanism Docker's own `HEALTHCHECK` invokes, proven both ways against
a real container rather than only asserted in isolation. Torn down and
volumes removed afterward, same discipline as every other Docker
verification in this project.

## Post-Phase-6 improvement — a full QA pass, and 20 real defects fixed

A dedicated QA pass, not incidental to feature work: four parallel deep
reads covered every file in `src/`, `src/ingestion/`, and the ops/
dashboard layer against their existing tests, followed by a live pass
against the real running dashboard, REST API, scheduler, and CLI
scripts. Found and fixed **20 real defects** (4 fixed first as the highest-
priority/most-critical items, 16 more across High/Medium/Low severity
in a second pass), several reproduced live rather than only inferred
from reading code:

- A Pc-classified CRITICAL conjunction (large distance, tight
  covariance - the exact scenario Phase 2 exists for) could crash the
  whole pipeline with an uncaught `pydantic.ValidationError`.
- The shared TLE parser cascaded one corrupted line into dropping every
  object after it, not just the bad one.
- A TOCTOU race let concurrent approve/reject calls silently overwrite
  each other - reproduced and fixed against a real local Postgres with
  two racing threads.
- The webhook alert secret could leak into process logs on a failed
  send.
- A real Prometheus cardinality bug (429 responses bypassing route-
  template labeling) - reproduced live by deliberately rate-limiting
  against distinct dynamic paths and confirming the raw IDs leaked into
  label values.
- CDM enrichment matched a stale Pc onto a fresh, geometrically
  unrelated conjunction pass (object-pair match only, no TCA proximity
  check).
- The delta-v budget tracker silently reset to full on every dashboard
  button click instead of persisting across a real operator session.
- Real Space-Track scans were excluded from Trends and the orbit-plot
  feature (both still hardcoded to `source == "celestrak"` only).
- A dozen more medium/low-severity gaps across the ingestion layer
  (Space-Track session re-auth, per-hour throttle, cache-key collision),
  the API (`/health` bypassing dependency overrides, a bare `POST
  /review` 422ing instead of working, read endpoints 500ing instead of
  503ing on a storage blip), and smaller fixes to `gemma_client.py`'s
  response parsing, `rag.py`'s attitude-entry handling, `preflight.py`'s
  storage check, and `rate_limit.py`'s bucket eviction.

Every fix got a real regression test (several exercising the actual
race/failure condition against real infrastructure - real Postgres, a
real running API, a real Docker healthcheck - not just asserting the
fix exists). Full suite: 375/375 after both rounds.

## Post-Phase-6 improvement — real satellite watch list (`WATCHED_NORAD_IDS`)

A PM-level gap, not a bug: every screen in this project was locked to
CelesTrak's/Space-Track's own pre-curated *named groups* (`stations`,
`cosmos-2251-debris`) - there was no way to monitor a specific real
satellite by NORAD catalog ID. For nearly any real operator whose
satellite isn't a crewed station, that meant **no way to point this
project at their own asset at all** - the single largest gap between
"a working demo" and "something a real customer could actually use,"
bigger than any individual bug the QA pass above found.

Closed with `WATCHED_NORAD_IDS` (`src/config.py`) - a comma-separated
list of a real operator's own NORAD catalog ID(s). Real new fetch
capability added to `src/ingestion/tle_source.py`:
`fetch_tle_by_catalog_ids` (CelesTrak's real per-object `CATNR` query -
no bulk-by-ID endpoint exists, so one real request per watched object)
and `fetch_spacetrack_by_catalog_ids` (a real Space-Track bulk
`NORAD_CAT_ID` list query - one real request for the whole watch list,
a genuine efficiency advantage Space-Track's API provides that
CelesTrak's doesn't). Wired into `CelesTrakAdapter`/`SpaceTrackAdapter`
via a new `watched_norad_ids` param and `WATCHED_GROUP_LABEL` - "my
satellite" is fetched differently but then treated as just another real
group by the existing cross-group screening algorithm, no special-cased
logic needed. `DecayRiskAdapter` got the equivalent `catalog_ids` param
for real per-satellite decay/re-entry screening. Every real entry point
(dashboard buttons, `scripts/scheduler.py`'s continuous monitoring)
watches the real configured asset when set, falling back to the
original demo groups completely unchanged when it isn't.

**A real, live customer walkthrough found a second gap the fix itself
introduced**, closed in the same pass: with a watch list configured and
cross-screened against a real, naturally clustered debris field, the
debris field's own internal pairs are almost always closer to *each
other* than to an unrelated, well-separated satellite - a live test
(watching the real ISS, 25544) confirmed the default 5-result fetch
returned zero mentions of the watched asset at all, 100% debris-vs-
debris noise. Fixed by excluding debris-vs-debris pairs entirely once a
watch list is configured - every returned conjunction is guaranteed to
involve the customer's own asset (multi-satellite fleets still see
their own mutual conjunctions; only *unrelated* pairs are excluded).

**Live-verified against the real CelesTrak API, not just mocked
tests:** fetched the real ISS by NORAD ID (25544, no group membership
involved) via a real `CATNR` query, confirmed it parsed correctly, then
ran it through the real screening pipeline against the real
`cosmos-2251-debris` group - correctly labeled `my-assets`, correctly
cross-screened (a real 11,193 km real conjunction found, ISS vs. real
Cosmos 2251 debris), and confirmed real decay-risk screening reported
ISS's real current perigee (~414km, matching this project's own
earlier live-verified finding). Also live-verified through the real
dashboard UI end to end with a real `.env` configuration change
(reverted afterward) - the "Monitoring your own asset(s)" sidebar
notice, a real fetch, and real logged entries all confirmed working
together.

23 new tests across `tests/test_tle_source.py`,
`tests/test_celestrak_adapter.py`, `tests/test_spacetrack_adapter.py`,
`tests/test_decay_adapter.py`, `tests/test_scheduler.py`,
`tests/test_dashboard_app.py`, and a new `tests/test_config.py`. Full
suite: 393/393.

Also closed as part of the same PM pass: the feature was undiscoverable
without reading source code - a real customer would never have found
`WATCHED_NORAD_IDS` on their own. Documented in `.env.example` and a new
README "Point it at your own satellite" setup step.

## Post-Phase-6 improvement — configurable hazard-severity thresholds

A second PM-level gap from the same customer-review pass: every
CRITICAL/WARNING/WATCH/NOMINAL cutoff in this project (conjunction
distance, decay perigee altitude, attitude pointing error) was a
hardcoded literal inside `src/pipeline.py`'s `classify_*_severity`
functions. That's a reasonable single set of defaults for a demo, but a
real deployment isn't one-size-fits-all - a maneuverable, high-value
satellite reasonably wants a bigger CRITICAL buffer than a defunct
cubesat, and there's no one correct answer this project should hardcode
for every real operator.

Closed by adding nine new `Settings` fields (`src/config.py`) -
`conjunction_critical_km`/`warning_km`/`watch_km`,
`decay_critical_perigee_km`/`warning_perigee_km`/`watch_perigee_km`,
`attitude_critical_deg`/`warning_deg`/`watch_deg` - all defaulting to
exactly this project's original hardcoded values, so a deployment that
never touches these env vars sees zero behavior change. Every
`classify_*_severity` function in `pipeline.py` now accepts these as
optional parameters (still defaulting to the original literals for any
caller/test that doesn't pass them), and `analyze_node` reads the real
configured values off `client.settings` via the codebase's established
`getattr(settings, "field", default)` defensive-access pattern - the
same convention already used for `gemma_backend`, needed here because
`client.settings` is a real `Settings` dataclass in production but a
minimal `SimpleNamespace` in this project's own test doubles.

`conjunction_critical_km` also closes a real single-source-of-truth
risk: `src/maneuver.py`'s `CRITICAL_THRESHOLD_KM` module constant used
to be an independently-hardcoded duplicate of the pipeline's own
conjunction CRITICAL threshold, with no mechanism keeping the two in
sync. `verify_maneuver()` now takes an optional `critical_threshold_km`
parameter (still defaulting to the module constant for direct callers),
and both real call sites - `pipeline.py`'s `decide_node` and
`logging_utils.py`'s `approve_maneuver` (the human-approval path) - pass
the real configured `Settings.conjunction_critical_km` explicitly, so
the value that classified a conjunction CRITICAL in the first place is
the same value that verifies its avoidance maneuver.

**Live-verified against a real, unmocked `Settings` object** - not just
the test doubles: loaded `Settings` via `config.load_settings()` with
`CONJUNCTION_CRITICAL_KM=15.0` set as a real process environment
variable (real `os.getenv` parsing, no monkeypatching), ran a real
10km-conjunction `TelemetryEvent` through the real `analyze_node`, and
confirmed it classified CRITICAL - then reran the identical event with
no override and confirmed the same code classified it WARNING instead,
proving the configured value (not a coincidence, not a stale default)
is what actually drove the decision.

10 new tests across `tests/test_config.py` (defaults-preservation and
real env-var parsing) and `tests/test_pipeline_smoke.py`/
`tests/test_maneuver.py` (each `classify_*_severity` respecting
non-default thresholds, `analyze_node` end-to-end using a configured
client's real thresholds, `verify_maneuver` respecting a custom
`critical_threshold_km`). Full suite: 400/400.

Documented in `.env.example` (all nine new env vars) and a new README
"Tune hazard severity thresholds" setup step.

**A follow-up customer-discoverability check** (the same class of gap
already found for `WATCHED_NORAD_IDS`) asked: once an operator sets one
of these nine env vars, how do they confirm it actually took effect,
short of reading `src/pipeline.py`? Closed by adding a
"⚙️ Hazard severity thresholds" sidebar expander to
`scripts/dashboard.py`, right below the existing watch-list status
notice - labeled `(defaults)` or `(customized)` depending on whether
any of the nine differ from their original values, and always showing
the real currently-active cutoffs for all three hazard types. 2 new
tests (`tests/test_dashboard_app.py`), full suite: 402/402.
Live-verified in a real running dashboard: confirmed the expander
renders the exact real values from a real `Settings` singleton, both
collapsed (accessibility tree) and expanded (screenshot).

## Post-Phase-6 improvement — CRITICAL-alert cooldown (fixing alert fatigue)

A fresh PM-then-customer review (explicitly scoped to find gaps beyond
everything already documented above) surfaced a real defect in the
continuous-operation + alerting features that were each already shipped
individually but never tested *together* at real scheduler cadence:
`scripts/scheduler.py` re-detects the same still-unresolved conjunction
or at-risk object fresh on every tick (a new `event_id` each time - see
`CelesTrakAdapter`/`DecayRiskAdapter`), and `src/alerting.py`'s
`send_critical_alert` fired unconditionally on every CRITICAL-severity
log entry. Combined, one still-open CRITICAL hazard would re-page a real
operator every scheduler tick, indefinitely, until it happened to
resolve on its own - the fastest way to make anyone mute or distrust the
alert channel this project just built.

Closed with `hazard_key()` (`src/alerting.py`) - a stable identity for
the real underlying hazard, independent of the entry's own
run/tick-specific `event_id`: a sorted object-id pair for conjunctions,
`object_id` for decay/attitude. `send_critical_alert` gained an optional
`recent_critical_entries` parameter and a 24-hour `ALERT_COOLDOWN_HOURS`
- a CRITICAL finding sharing the same `hazard_key` as another CRITICAL
alert within the cooldown window is suppressed; `src/pipeline.py`'s
`log_node` fetches real recent history (`logger.load_all_entries()`,
only when the finding is actually CRITICAL, to avoid the cost for the
common case) and threads it through. A genuinely new/different hazard,
or the same hazard once the cooldown has elapsed, still alerts
normally - this isn't an acknowledge/resolve workflow (out of scope at
this product's scale), just a real "once a day per unresolved hazard"
ceiling.

Honest limitation: the cooldown check scans the full logged history via
`load_all_entries()` rather than a scoped store query - fine at this
project's real demo/small-fleet scale (the same pattern the dashboard's
own Trends view already uses), but a production deployment at real
operator scale would want `DecisionLogStore` to support a proper
indexed/filtered query instead.

**Live-verified against a real local Ollama-backed pipeline, a real
JSONL audit log, and a real local HTTP webhook receiver** (not mocked):
ran the same synthetic CRITICAL hazard (object pair `99000`/`99010`)
through `run_once()` twice with a shared `DecisionLogger`, simulating
two scheduler ticks re-detecting the same still-unresolved conjunction.
The real receiver logged exactly one POST total - the first tick's real
Gemma-authored alert text, confirmed suppressed on the second.

9 new tests across `tests/test_alerting.py` (`hazard_key` across all
three hazard shapes, order-independence for conjunction pairs, cooldown
suppression/expiry/non-interference, self-suppression exclusion) and
`tests/test_pipeline_smoke.py` (a real `JSONLDecisionLogStore`-backed
end-to-end test proving `log_node` actually wires real history into the
cooldown check, not just a unit-level mock). Full suite: 411/411.
