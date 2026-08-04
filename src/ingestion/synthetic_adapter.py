"""Synthetic CRITICAL-range conjunction fixture, shared by scripts/run_demo.py
and scripts/dashboard.py.

Real orbital data rarely produces a CRITICAL (<5km) conjunction on demand
(see CelesTrakAdapter), so this exists to demo that path - maneuver
calculation, verification, budget depletion, and (depending on the
configured Gemma backend) autonomous execution/veto or human approval -
deterministically. Clearly labeled as synthetic via source/event_id, never
mistaken for real tracking data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..schemas import TelemetryEvent
from .base_adapter import DataSourceAdapter

# Matches src/maneuver.py's own ASSUMED_MANEUVER_LEAD_TIME_S - a
# plausible "there's still time for the computed maneuver to act before
# closest approach" window, not just an arbitrary future offset. Real
# bug this fixes: a fixed calendar date (e.g. "2026-08-01T20:00:00")
# reads as CRITICAL-and-urgent only until that date passes, after which
# every "Run synthetic CRITICAL scenario" demo click shows the real
# tca_urgency_label() feature (src/dashboard_data.py) rendering "TCA
# already passed Xh Ym ago - approving this maneuver no longer has any
# effect" on the project's own flagship demo fixture.
_ASSUMED_LEAD_TIME = timedelta(hours=6)


class SyntheticCriticalAdapter(DataSourceAdapter):
    """4 synthetic CRITICAL-range conjunctions sharing one budget tracker,
    so budget depletion is visible deterministically on the 4th event.

    event_id includes run_id so repeat runs (e.g. clicking a dashboard
    button twice, or re-running the demo script) don't collide on the same
    id - DecisionLogger.mark_reviewed/find_entry match on the FIRST logged
    entry with a given event_id, so a reused id from an earlier run would
    silently update that older entry instead of the new one.
    """

    def __init__(self, run_id: str, id_prefix: str = "conj-synthetic-critical"):
        self.run_id = run_id
        self.id_prefix = id_prefix

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        events = []
        tca = (datetime.now(timezone.utc) + _ASSUMED_LEAD_TIME).isoformat()
        for i in range(limit):
            raw = {
                "object_a_id": f"9900{i}", "object_a_name": f"SYNTH-A-{i}",
                "object_b_id": f"9901{i}", "object_b_name": f"SYNTH-B-{i}",
                "min_distance_km": 3.0,
                "time_of_closest_approach": tca,
                "relative_velocity_km_s": 6.0,
            }
            events.append(TelemetryEvent(
                event_id=f"{self.id_prefix}-{self.run_id}-{i}",
                timestamp=datetime.now(timezone.utc),
                source="synthetic-critical-fixture",
                raw_data=raw,
            ))
        return events
