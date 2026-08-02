"""Replays real, documented historical conjunction records through the
pipeline - proof that this system's deterministic classification and
decision logic would have flagged a real past near-collision correctly,
using the real numbers as they were actually reported at the time.

This is NOT live SGP4 propagation of archival TLEs. CelesTrak's public
gp.php endpoint only ever serves the CURRENT TLE for a given catalog
number - confirmed by querying it with a historical EPOCH parameter,
which is silently ignored and returns today's data regardless. Genuine
historical TLE archives require Space-Track.org, which needs a real
account/credentials this project doesn't have and can't obtain on a
user's behalf. Rather than fake historical propagation, this replays the
REAL closest-approach numbers exactly as they were documented at the
time, through the same pipeline live data goes through.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from ..schemas import TelemetryEvent
from .base_adapter import DataSourceAdapter


@dataclass(frozen=True)
class HistoricalEvent:
    id_slug: str
    object_a_id: str
    object_a_name: str
    object_b_id: str
    object_b_name: str
    min_distance_km: float
    time_of_closest_approach: str  # ISO 8601
    relative_velocity_km_s: float
    historical_event: str
    historical_source: str
    historical_actual_outcome: str


# Every number below is independently verifiable, not invented:
#   - The 584m predicted closest approach and the report timing (issued
#     2009-02-10 15:02 UTC, predicted closest approach ~16:56 UTC the same
#     day) come from CelesTrak's own account of this event
#     (https://celestrak.org/events/collision/), including the detail that
#     SOCRATES predicted this exact conjunction in all 14 reports issued
#     that week (range 117m-1.812km across those reports), but it ranked
#     only #152 overall in the final report and was never prioritized or
#     acted on.
#   - NORAD catalog numbers, the ~11.7 km/s relative velocity, and the
#     ~789 km collision altitude are well-documented and were
#     cross-checked against NASA/Wikipedia sources.
IRIDIUM_COSMOS_COLLISION = HistoricalEvent(
    id_slug="iridium33-cosmos2251",
    object_a_id="24946", object_a_name="IRIDIUM 33",
    object_b_id="22675", object_b_name="COSMOS 2251",
    min_distance_km=0.584,
    time_of_closest_approach="2009-02-10T16:56:00+00:00",
    relative_velocity_km_s=11.7,
    historical_event=(
        "Iridium 33 / Cosmos 2251, 2009-02-10 - the first confirmed "
        "accidental collision between two intact satellites in orbit, at "
        "~789 km altitude over Siberia."
    ),
    historical_source=(
        "CelesTrak's own historical account (https://celestrak.org/events/collision/): "
        "SOCRATES predicted this conjunction in all 14 reports issued that week "
        "(range 117m-1.812km); the final report, issued 2009-02-10 15:02 UTC, "
        "predicted 584m at ~16:56 UTC the same day. Relative velocity (~11.7 km/s) "
        "and altitude (~789 km) corroborated against NASA/Wikipedia sources."
    ),
    historical_actual_outcome=(
        "COLLISION. Despite being genuinely predicted all week, this conjunction "
        "ranked only #152 overall in the day's report (out of a much larger set of "
        "predicted conjunctions industry-wide) and was never prioritized or acted "
        "on - this was a triage failure, not a detection failure. No avoidance "
        "maneuver was performed."
    ),
)

DEFAULT_HISTORICAL_EVENTS: tuple[HistoricalEvent, ...] = (IRIDIUM_COSMOS_COLLISION,)


class HistoricalReplayAdapter(DataSourceAdapter):
    """Replays fixed historical conjunction records (default: the real
    2009 Iridium 33/Cosmos 2251 collision - see IRIDIUM_COSMOS_COLLISION)
    as TelemetryEvents, source="historical-replay" so they're never
    mistaken for live data downstream (audit log, dashboard, display all
    surface `source` as-is).

    run_id follows the same convention as SyntheticCriticalAdapter: included
    in event_id so repeat replays (re-running the demo, clicking a
    dashboard button twice) don't collide on the same id - DecisionLogger
    matches an event_id's FIRST logged occurrence, so a reused id would
    silently update a stale entry instead of the new run's.
    """

    def __init__(self, run_id: str, events: Sequence[HistoricalEvent] = DEFAULT_HISTORICAL_EVENTS):
        self.run_id = run_id
        self.events = list(events)

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        results = []
        for event in self.events[:limit]:
            raw = {
                "object_a_id": event.object_a_id, "object_a_name": event.object_a_name,
                "object_b_id": event.object_b_id, "object_b_name": event.object_b_name,
                "min_distance_km": event.min_distance_km,
                "time_of_closest_approach": event.time_of_closest_approach,
                "relative_velocity_km_s": event.relative_velocity_km_s,
                "historical_event": event.historical_event,
                "historical_source": event.historical_source,
                "historical_actual_outcome": event.historical_actual_outcome,
            }
            results.append(TelemetryEvent(
                event_id=f"historical-{event.id_slug}-{self.run_id}",
                timestamp=datetime.now(timezone.utc),
                source="historical-replay",
                raw_data=raw,
            ))
        return results
