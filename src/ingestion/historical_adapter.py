"""Replays real, documented historical conjunction records through the
pipeline - proof that this system's deterministic classification and
decision logic would have flagged a real past near-collision correctly,
using the real numbers as they were actually reported at the time.

The documented replay below (the 584m SOCRATES prediction) is
independent of Space-Track and always available - it's what
classify_conjunction_severity actually runs on, unchanged. A real,
SEPARATE cross-check is layered on top when a SpaceTrackClient is
provided: real_repropagate_event() fetches REAL historical TLEs for both
objects (via SpaceTrackClient.fetch_historical_tle_text, live-verified
in ROADMAP_TO_PRODUCT.md Phase 1) and re-runs this project's own real
two-pass propagation search (src/orbital.py - the exact same physics
every live conjunction screen already uses) to independently derive a
closest approach, rather than trusting the documented number alone. This
was structurally impossible before a real Space-Track account existed
(CelesTrak's public gp.php endpoint only ever serves the CURRENT TLE for
a given catalog number, ignoring any historical epoch parameter -
confirmed by querying it directly) - deliberately additive, not a
replacement: the documented numbers remain what the pipeline classifies
on, and a real cross-check either corroborates them independently or
surfaces a real, honest discrepancy worth knowing about, never silently
overriding what SOCRATES actually reported at the time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, TYPE_CHECKING

from skyfield.api import EarthSatellite, load

from ..orbital import find_closest_approach, parse_tle_epoch
from ..schemas import TelemetryEvent
from .base_adapter import DataSourceAdapter

if TYPE_CHECKING:
    from .spacetrack_client import SpaceTrackClient


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


def _parse_bare_tle(text: str, norad_id: str) -> tuple[str, str]:
    """SpaceTrackClient.fetch_historical_tle_text returns Space-Track's
    real `tle` format - bare 2 lines, no name line at all (the same real
    shape that broke fetch_spacetrack_group_text before Phase 1's live
    fix - see tle_source.py). No parse_tle_blocks reuse needed here since
    there's no name-line grouping problem with exactly one object."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) != 2 or not lines[0].startswith("1 ") or not lines[1].startswith("2 "):
        raise ValueError(f"Expected a real 2-line TLE for NORAD {norad_id}, got: {text!r}")
    return lines[0], lines[1]


def real_repropagate_event(event: HistoricalEvent, client: "SpaceTrackClient", lookahead_hours: int = 72) -> dict:
    """Real, independent cross-check: fetches REAL historical element
    sets for both objects (the most recent real state at or before the
    documented closest-approach time) and re-runs this project's own
    real two-pass propagation search on them - the exact same
    orbital.find_closest_approach every live screening call already
    uses, not a special-cased historical path.

    Propagation starts from the EARLIER of the two real TLE epochs (so
    both objects have a valid, already-published state at the start of
    the search window) and searches forward `lookahead_hours` - long
    enough to comfortably cover the gap to the documented event time
    even when the two objects' real epochs differ by a day or more,
    which was true for this project's own default event during live
    testing.

    Raises (does not silently swallow) on a real fetch/parse failure -
    callers decide how to degrade; see HistoricalReplayAdapter.fetch_batch
    for the "attempt it, log and continue on failure" policy used there.
    """
    l1_a, l2_a = _parse_bare_tle(
        client.fetch_historical_tle_text(event.object_a_id, event.time_of_closest_approach), event.object_a_id,
    )
    l1_b, l2_b = _parse_bare_tle(
        client.fetch_historical_tle_text(event.object_b_id, event.time_of_closest_approach), event.object_b_id,
    )

    ts = load.timescale()
    sat_a = EarthSatellite(l1_a, l2_a, event.object_a_name, ts)
    sat_b = EarthSatellite(l1_b, l2_b, event.object_b_name, ts)

    epoch_a = parse_tle_epoch(l1_a)
    epoch_b = parse_tle_epoch(l1_b)
    start_time = min(epoch_a, epoch_b)

    result = find_closest_approach(sat_a, sat_b, ts, start_time, hours=lookahead_hours)
    return {
        "real_repropagated_min_distance_km": result["min_distance_km"],
        "real_repropagated_time_of_closest_approach": result["time_of_closest_approach"].isoformat(),
        "real_repropagated_relative_velocity_km_s": result["relative_velocity_km_s"],
        "real_repropagated_object_a_tle_epoch": epoch_a.isoformat(),
        "real_repropagated_object_b_tle_epoch": epoch_b.isoformat(),
    }


class HistoricalReplayAdapter(DataSourceAdapter):
    """Replays fixed historical conjunction records (default: the real
    2009 Iridium 33/Cosmos 2251 collision - see IRIDIUM_COSMOS_COLLISION)
    as TelemetryEvents, source="historical-replay" so they're never
    mistaken for live data downstream (audit log, dashboard, display all
    surface `source` as-is). Classification always runs on the real
    documented numbers (min_distance_km etc.) - unchanged by whether a
    real cross-check was attempted or what it found.

    run_id follows the same convention as SyntheticCriticalAdapter: included
    in event_id so repeat replays (re-running the demo, clicking a
    dashboard button twice) don't collide on the same id - DecisionLogger
    matches an event_id's FIRST logged occurrence, so a reused id would
    silently update a stale entry instead of the new run's.

    spacetrack_client is optional - when provided, each event also gets a
    real cross-check attempt (see real_repropagate_event) added to
    raw_data under real_repropagated_* keys. A failure there (network,
    unparseable TLE, no historical data that far back) is caught and
    logged, never raised - the documented replay itself must never be
    blocked by an optional enhancement failing.
    """

    def __init__(
        self,
        run_id: str,
        events: Sequence[HistoricalEvent] = DEFAULT_HISTORICAL_EVENTS,
        spacetrack_client: Optional["SpaceTrackClient"] = None,
    ):
        self.run_id = run_id
        self.events = list(events)
        self.spacetrack_client = spacetrack_client

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
            if self.spacetrack_client is not None:
                try:
                    raw.update(real_repropagate_event(event, self.spacetrack_client))
                except Exception as exc:  # noqa: BLE001 - an optional cross-check must never block the replay
                    raw["real_repropagation_error"] = str(exc)
            results.append(TelemetryEvent(
                event_id=f"historical-{event.id_slug}-{self.run_id}",
                timestamp=datetime.now(timezone.utc),
                source="historical-replay",
                raw_data=raw,
            ))
        return results
