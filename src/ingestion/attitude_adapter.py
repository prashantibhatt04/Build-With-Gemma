"""Synthetic spacecraft attitude/pointing-loss fixture.

Unlike conjunction screening (real CelesTrak TLEs) and decay risk (real
orbital elements Skyfield already parses from those same TLEs), there is
no real, publicly-fetchable data source for spacecraft ATTITUDE (pointing
direction, angular/tumble rate). TLEs encode only orbital position and
velocity, never orientation - attitude telemetry is normally proprietary
to each spacecraft's own operator, not published anywhere analogous to
CelesTrak. That's a structural absence, not "real data is rare on
demand" the way a CRITICAL conjunction or a CRITICAL decay reading is -
so this hazard type is necessarily synthetic-only, clearly labeled via
source/event_id, the same way SyntheticCriticalAdapter is for
conjunctions (see its module docstring).

Demonstrates the same deterministic-severity/Gemma-narrates pattern as
every other hazard type - see classify_attitude_severity in pipeline.py.
Deliberately a spread across all four severity bands, not
SyntheticCriticalAdapter's all-CRITICAL design: that design exists
specifically to demo maneuver-budget depletion across repeated CRITICAL
events, which doesn't apply here since attitude loss has no maneuver
machinery (see decide_node's raw_data-shape guard - a re-entry burn or a
collision-avoidance burn doesn't fix a tumbling spacecraft; recovering
attitude control is a real, separate intervention - reaction-wheel
desaturation, thruster-based detumbling, safe-mode recovery procedures -
out of scope for this project).
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..schemas import TelemetryEvent
from .base_adapter import DataSourceAdapter

# (object_name, pointing_error_deg, angular_rate_deg_s, solar_panel_power_pct)
# A realistic spread across NOMINAL/WATCH/WARNING/CRITICAL - a spacecraft
# losing attitude control typically loses solar-panel pointing too, so
# power output is included as a real, correlated (if simplified) signal
# alongside the primary pointing-error threshold - see
# classify_attitude_severity and _attitude_description in pipeline.py.
_FIXTURES = [
    ("SYNTH-SAT-NOMINAL", 0.8, 0.05, 98.0),
    ("SYNTH-SAT-WATCH", 8.0, 0.3, 91.0),
    ("SYNTH-SAT-WARNING", 25.0, 1.2, 68.0),
    ("SYNTH-SAT-CRITICAL", 70.0, 4.5, 22.0),
]


class SyntheticAttitudeAdapter(DataSourceAdapter):
    """Synthetic attitude/pointing-status events - see module docstring
    for why this hazard type has no real data source to draw from
    instead."""

    def __init__(self, run_id: str, id_prefix: str = "attitude-synthetic"):
        self.run_id = run_id
        self.id_prefix = id_prefix

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        events = []
        for i, (name, pointing_error_deg, angular_rate_deg_s, solar_panel_power_pct) in enumerate(_FIXTURES[:limit]):
            raw = {
                "object_id": f"990{i}0",
                "object_name": name,
                "pointing_error_deg": pointing_error_deg,
                "angular_rate_deg_s": angular_rate_deg_s,
                "solar_panel_power_pct": solar_panel_power_pct,
            }
            events.append(TelemetryEvent(
                event_id=f"{self.id_prefix}-{self.run_id}-{i}",
                timestamp=datetime.now(timezone.utc),
                source="synthetic-attitude-fixture",
                raw_data=raw,
            ))
        return events
