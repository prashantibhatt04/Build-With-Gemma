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

## Architecture

```
CelesTrak (live TLE data)
        │
        ▼
CelesTrakAdapter — Skyfield/SGP4 orbital propagation,
                    two-pass coarse/fine closest-approach search
        │
        ▼
analyze_node — deterministic severity classification (distance thresholds)
             — Gemma: plain-language description of the finding
        │
        ▼
decide_node — deterministic action mapping (severity -> action)
            — for CRITICAL: deterministic maneuver calc + independent
              verification + delta-v budget check
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
`src/maneuver.py` (maneuver math, independent verification, delta-v
budget), `src/gemma_client.py` (backend-agnostic Gemma access + failover),
`src/pipeline.py` (the LangGraph nodes), `src/logging_utils.py` (audit
log, human-review, and maneuver-approval workflows), `src/display.py`
(terminal rendering), `src/preflight.py` (environment health checks), and
`scripts/run_demo.py` (the guided end-to-end demo).

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

## Design choices

- **Severity, action, and maneuver physics are 100% deterministic
  thresholds and closed-form math - never AI output.** In a
  collision-avoidance context, reliability has to come first; Gemma's
  primary value is making already-reliable decisions understandable, not
  making them. The one deliberate exception - the local-path veto check -
  is designed so Gemma can only ever narrow what's already been verified
  safe, never widen it: it can say no to a maneuver the physics cleared,
  it can never say yes to one the physics didn't.
- **A maneuver is independently re-verified, not just trusted.** The
  verification step re-derives the resulting clearance forward from the
  computed delta-v, rather than echoing the number the plan was
  algebraically solved for - a deliberate cross-check, not a rubber stamp.
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

## Future work

With Gemma's autonomous veto-gate now built (see above), the next
candidates - not yet committed to, tracked in `PHASE_PROGRESS.md` - are:
screening the live CelesTrak catalog for real current conjunctions
instead of a small staged set, a visual dashboard for a judge-friendly
live risk board, and a historical replay/backtest against a real past
close-approach event.

## Verification

87 automated tests (network-free, Gemma calls mocked) cover orbital math,
severity/confidence derivation, maneuver math, budget tracking, Gemma
retry/fallback logic, the autonomous maneuver veto-check (including its
fail-safe defaults), the full pipeline, and the human-approval/review
workflows. Beyond unit tests, every major path in this writeup was also
run live end-to-end against a real local Ollama instance and a real
hosted API key during development - which is how the two bugs described
above were actually found.

See the [public repository](https://github.com/prashantibhatt04/Build-With-Gemma)
for full source, `PROJECT_OVERVIEW.md` for a diagram-based walkthrough,
`DEMO.md` for exact reproduction steps, and `PHASE_PROGRESS.md` for the
complete build history.
