# Notes for morning — track2-preevent-ok

Session covered Part A (design fixes) + Phase 5 (real end-to-end run) + commits.
Everything below is in `track2-preevent-ok` only. `track2-live-build` was never
touched this session.

## Review this first (< 10 min skim)

1. **A GitHub remote now exists on both repos** (`origin` -> `github.com/prashantibhatt04/track2-preevent-ok`
   and `.../track2-live-build`). I did not add these and have not pushed
   anything, per your explicit "no remotes, no pushes" instruction. This
   must have happened outside this session (you or something else). Please
   confirm this is intentional before anyone runs `git push` — right now
   both repos are commits-ahead-of-nothing-pushed, which is a safe state,
   but worth knowing before it isn't.
2. **Commit grouping is 2 commits, not 5.** Phases 3, 4, the model-default
   fix, and Part A all landed in the same files (`pipeline.py`, `schemas.py`,
   `gemma_client.py`, etc.) across many rounds with no intermediate commits
   made along the way. Rather than fabricate a fake per-phase history by
   reconstructing intermediate file states from memory, I grouped honestly:
   - `5b5c96a` — Phase 2 (CelesTrak adapter, fully self-contained, untouched since)
   - `cdef01e` — Phases 3-4 + Part A hardening (analyze/decide nodes,
     provenance, retry, timeouts, human-review fields) — see the commit body
     for the full breakdown, it's written to be skimmable.
3. **`scripts/sanity_check_celestrak.py` is still untracked**, sitting in the
   working tree, never committed. It was explicitly throwaway exploratory
   code from before Phase 1 was even named. Your call: commit it, delete it,
   or leave it — I didn't want to unilaterally decide.
4. **Real Gemma latency: ~17.5s/call average** (range 10.8s-24.2s across 6
   calls in the Phase 5 run), ~105s of real Gemma time for 3 events (each
   event costs 2 calls: description + rationale). If you run more events
   live tomorrow, budget roughly `N events x 2 x ~17.5s`. Warm-up itself is
   fast now (2.9s with `gemma4:e4b`, vs. the ~53s cold start we saw earlier
   with the larger `gemma4:12b`).
5. **Minor Gemma content quirk**: one description in the real log below
   contains stray LaTeX-ish formatting (`$\text{33766}$`, `$1.8$`). Cosmetic,
   not a pipeline bug — the model occasionally emits markup it shouldn't.
   Low-priority polish item if it recurs during the live demo.

## What got completed

**Part A (design fixes):**
1. Provenance tracking — new `GemmaProvenance` model (`source`: `"gemma"`|`"fallback"`,
   `model_used`, `latency_ms`) on `DecisionLogEntry` as `description_provenance`
   (nullable — analyze_node only attempts Gemma for conjunction-shaped data)
   and `rationale_provenance` (always populated — decide_node always attempts
   Gemma). Both nodes now measure and record real call latency.
2. `GemmaClient.generate()` retries once on `GemmaClientError` before letting
   the exception propagate to the caller's fallback. Retry wraps only the
   actual backend call (ollama/api), not the "unknown backend" config-error
   path — that's not transient, so it fails fast instead of retrying.
3. Checked `CelesTrakAdapter`'s `requests.get()` — already had an explicit
   `timeout=30` from Phase 2. Left as-is (satisfies the requirement; didn't
   shrink it to the suggested 10-15s since it wasn't asked to change, just
   checked).
4. Added `human_reviewed: bool = False` and `human_reviewed_at: Optional[datetime] = None`
   to `DecisionLogEntry` — no workflow wired to them yet, just the fields.

Ran the full suite after each addition; all green throughout, no regressions.

**Part B (Phase 5 — real end-to-end run):**
- Warmed up `gemma4:e4b` (2.9s), then ran the complete real pipeline —
  `CelesTrakAdapter(cosmos-2251-debris, sample_size=30).fetch_batch(3)` ->
  `analyze_node` -> `decide_node` -> `log_node` — via `run_once()`, using the
  real local Ollama client end to end (no stubs).
- **Note on the numbers**: the top 3 conjunctions this run were NOT the same
  as Phase 2's original report (33779/33825 @ 6.429km). Orbital geometry and
  TLE data both shift with time, so a fresh fetch against "now" naturally
  produces different top results. This run's top 3 all came back in the
  WATCH band (25-100km), so none hit HOLD/ABORT — that's real data behaving
  correctly, not a bug. If you want to see a HOLD/ABORT case live, either
  re-run closer to a real close-approach window or use a synthetic example
  like the ones from the earlier `decide_node` manual checks.
- `log_node` needed no changes and required no fixes during this run —
  `DecisionLogEntry` construction in `log_node`/`run_once` was updated in
  Part A to pass the new provenance fields, and that's the only touch it got.

**Test suite: 16/16 passing** (2 orbital, 2 adapter, 1 original DummyAdapter
smoke test, 6 severity boundary cases, 4 severity->action cases, 1 Gemma-
fallback case — provenance assertions added inline to the relevant existing
tests rather than as separate test functions).

## Real Phase 5 log output (from `logs/decisions-2026-08-01.jsonl`)

```json
{
  "telemetry": {
    "event_id": "conj-33765-33818",
    "timestamp": "2026-08-01T03:23:04.766669Z",
    "source": "celestrak",
    "raw_data": {
      "object_a_id": "33765", "object_a_name": "COSMOS 2251 DEB",
      "object_b_id": "33818", "object_b_name": "COSMOS 2251 DEB",
      "min_distance_km": 32.14097390688428,
      "time_of_closest_approach": "2026-08-02T03:33:12.070570+00:00",
      "relative_velocity_km_s": 1.304492135903492
    }
  },
  "finding": {
    "event_id": "conj-33765-33818", "severity": "watch",
    "description": "On August 2, 2026, at 03:33 UTC, the two COSMOS 2251 DEB satellites will undergo a close approach. The minimum distance between them during this conjunction is approximately 32 km.",
    "confidence": 0.8
  },
  "decision": {
    "action": "continue",
    "rationale": "Recommendation: continue. The predicted minimum distance of approximately 32 km is well above typical collision thresholds, and therefore no avoidance maneuver is required at this time. We are monitoring this conjunction as a \"Watch\" event.",
    "made_at": "2026-08-01T03:23:38.616905Z"
  },
  "description_provenance": {"source": "gemma", "model_used": "gemma4:e4b", "latency_ms": 17616.08},
  "rationale_provenance": {"source": "gemma", "model_used": "gemma4:e4b", "latency_ms": 16229.41},
  "human_reviewed": false, "human_reviewed_at": null
}
```
```json
{
  "telemetry": {
    "event_id": "conj-33766-33793",
    "raw_data": {
      "object_a_id": "33766", "object_a_name": "COSMOS 2251 DEB",
      "object_b_id": "33793", "object_b_name": "COSMOS 2251 DEB",
      "min_distance_km": 45.032735700574676,
      "time_of_closest_approach": "2026-08-02T04:49:12.070570+00:00",
      "relative_velocity_km_s": 1.8172442454156563
    }
  },
  "finding": {
    "severity": "watch",
    "description": "There is a conjunction risk for COSMOS satellites $\\text{33766}$ and $\\text{33793}$, which will pass at a minimum separation of approximately 45 km on August 2, 2026. This close approach occurs at 04:49 UTC with a relative velocity of $1.8$ km/s.",
    "confidence": 0.8
  },
  "decision": {
    "action": "continue",
    "rationale": "Recommendation: continue. The projected minimum separation distance of approximately 45 km on August 2, 2026, provides sufficient clearance and falls within acceptable limits for a Watch level event. Therefore, no avoidance maneuver is required.",
    "made_at": "2026-08-01T03:24:21.214896Z"
  },
  "description_provenance": {"source": "gemma", "model_used": "gemma4:e4b", "latency_ms": 24213.84},
  "rationale_provenance": {"source": "gemma", "model_used": "gemma4:e4b", "latency_ms": 18381.02},
  "human_reviewed": false, "human_reviewed_at": null
}
```
```json
{
  "telemetry": {
    "event_id": "conj-33764-33822",
    "raw_data": {
      "object_a_id": "33764", "object_a_name": "COSMOS 2251 DEB",
      "object_b_id": "33822", "object_b_name": "COSMOS 2251 DEB",
      "min_distance_km": 47.13865288058911,
      "time_of_closest_approach": "2026-08-01T21:47:32.070570+00:00",
      "relative_velocity_km_s": 7.267227909038828
    }
  },
  "finding": {
    "severity": "watch",
    "description": "Two COSMOS 2251 DEB satellites will experience a close approach on August 1, 2026, at 21:47 UTC. The minimum separation during this event is calculated to be approximately 47 kilometers.",
    "confidence": 0.8
  },
  "decision": {
    "action": "continue",
    "rationale": "Recommendation: continue. This is a 'Watch' level event, and current predictions show a minimum separation of about 47 kilometers during the close approach on August 1, 2026. No immediate action is necessary.",
    "made_at": "2026-08-01T03:24:50.028601Z"
  },
  "description_provenance": {"source": "gemma", "model_used": "gemma4:e4b", "latency_ms": 18051.26},
  "rationale_provenance": {"source": "gemma", "model_used": "gemma4:e4b", "latency_ms": 10759.54},
  "human_reviewed": false, "human_reviewed_at": null
}
```

**Reconstructability check**: yes — each entry alone tells you what was
detected (`telemetry.raw_data`: objects, distance, velocity, time), what was
decided (`finding.severity`, `decision.action`), why (`finding.description`,
`decision.rationale` — real prose, not just numbers), and whether that
explanation is trustworthy (`*_provenance.source` — all three came from real
Gemma calls here, `"fallback"` would show clearly if Ollama had been down).

**Total real Gemma time this run**: 105.3s across 6 calls (3 events x 2 calls
each). Full `run_once()` wall time was 108.0s (the remaining ~2.7s was the
CelesTrak fetch/435-pair physics computation + logging, not Gemma).

## Judgment calls made (flagging per your instruction, not blocking on them)

- **Provenance granularity**: tracked separately per node
  (`description_provenance` for analyze_node, `rationale_provenance` for
  decide_node) rather than one combined field, since the two Gemma calls
  can independently succeed or fail and I didn't want to lose that
  information. The task named the fields generically (`rationale_source`,
  `model_used`, `latency_ms`); I read "rationale_source" as tied to
  `Decision.rationale` specifically (hence `rationale_provenance`) and
  added a parallel field for the description call rather than dropping
  that data. If you wanted a single combined field instead, easy to
  collapse — just say so.
- **Commit grouping** — see item 2 above.
- **`scripts/sanity_check_celestrak.py`** — see item 3 above.
- **CelesTrak timeout** — left at 30s (already explicit), didn't shrink to
  10-15s since only asked to check it existed, not to tune the value.

## Scope discipline

No dashboard/frontend/UI work started, as instructed. No scope added beyond
Part A items 1-4 and Phase 5 items 5-6. Stopping here per your instruction
to stop after this file.
