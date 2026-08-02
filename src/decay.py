"""Deterministic orbital decay / re-entry risk assessment - a second real
hazard type alongside conjunctions (Phase 14), proving analyze_node/
decide_node generalize beyond collision risk, not conjunction-specific by
construction (see schemas.py's own docstring: "these are intentionally
idea-agnostic"). Uses real orbital elements Skyfield's SGP4 model already
parses (perigee/apogee altitude, BSTAR drag term) from the SAME real TLE
data this project already fetches from CelesTrak - no new data source, no
new credentials, no separate TLE-column parser needed.

Simplified, not a real atmospheric-drag/decay-rate model (a genuine
time-to-reentry prediction needs solar flux and atmospheric density
tables, well beyond this project's scope). Perigee altitude alone is the
signal used here, since "how low is the lowest point of this orbit" is
itself real, well-established, uncontested orbital mechanics - an object
with a perigee below ~200km reliably reenters within days to weeks
regardless of other factors. Not a precise reentry-time predictor, not
flight software - the same spirit as src/maneuver.py's simplified,
clearly-labeled math.
"""
from __future__ import annotations

from skyfield.api import EarthSatellite

# WGS-72 equatorial radius (km) - the reference ellipsoid SGP4 itself
# uses, so this is the right constant to pair with sat.model's altitudes
# (which are already expressed in Earth radii, not the WGS-84 figure used
# for real-world mapping elsewhere in this project's display code).
EARTH_RADIUS_KM = 6378.135


def assess_decay_risk(
    object_id: str, object_name: str, sat: EarthSatellite, tle_epoch_age_hours: float,
) -> dict:
    """Builds a raw_data dict (TelemetryEvent-shaped) for one real tracked
    object's decay risk. Perigee/apogee altitude and the BSTAR drag term
    come directly from Skyfield's own already-parsed SGP4 model - the
    same object used for propagation elsewhere in this project - not a
    second, separate TLE-column parser.

    Single-object shape (object_id/object_name, not object_a_id/
    object_b_id) - decay risk is assessed per-object, not per-pair, and
    this naming difference is exactly what pipeline.analyze_node uses to
    tell a decay finding apart from a conjunction one.
    """
    return {
        "object_id": object_id,
        "object_name": object_name,
        "perigee_altitude_km": sat.model.altp * EARTH_RADIUS_KM,
        "apogee_altitude_km": sat.model.alta * EARTH_RADIUS_KM,
        "bstar": sat.model.bstar,
        "tle_epoch_age_hours": tle_epoch_age_hours,
    }
