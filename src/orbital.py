"""Orbital math: NORAD ID parsing and two-pass closest-approach search.

A flat fixed-step sampling of a trajectory only finds an UPPER BOUND on
the true closest approach between two objects - the real minimum can
fall between samples. find_closest_approach() fixes that with a
two-pass search: a coarse pass over the full lookahead window to find
the approximate minimum, then a fine pass in a narrow window around it
to refine the estimate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield.api import EarthSatellite
from skyfield.timelib import Timescale


# WGS-72 equatorial radius (km) - the reference ellipsoid SGP4 itself
# uses, so this is the right constant to pair with sat.model's altitudes
# (already expressed in Earth radii, not the WGS-84 figure used for
# real-world mapping elsewhere in this project's display code).
EARTH_RADIUS_KM = 6378.135


def perigee_apogee_altitude_km(sat: EarthSatellite) -> tuple[float, float]:
    """Real perigee/apogee altitude (km), straight from Skyfield's own
    already-parsed SGP4 model - originally written for src/decay.py
    (Phase 14), promoted here once src/catalog_screening.py (Phase 3 of
    ROADMAP_TO_PRODUCT.md) became a second real consumer that needed the
    exact same values, matching this project's established "a genuine
    second consumer justifies sharing code, not duplicating it" practice
    (see tle_source.py's own promotion history)."""
    return sat.model.altp * EARTH_RADIUS_KM, sat.model.alta * EARTH_RADIUS_KM


def parse_norad_id(tle_line1: str) -> str:
    """Extract the NORAD catalog number from columns 3-7 of TLE line 1."""
    return tle_line1[2:7].strip()


def parse_tle_epoch(tle_line1: str) -> datetime:
    """Extract the epoch timestamp from TLE line 1: columns 19-20 are a
    2-digit year, columns 21-32 are a fractional day-of-year.

    Standard space-track convention for the 2-digit year: 57-99 -> 1957-1999,
    00-56 -> 2000-2056 (a TLE epoch more than ~70 years old isn't a real
    case this pipeline needs to handle).
    """
    epoch_year_2digit = int(tle_line1[18:20])
    epoch_day = float(tle_line1[20:32])
    year = 1900 + epoch_year_2digit if epoch_year_2digit >= 57 else 2000 + epoch_year_2digit
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epoch_day - 1)


def _closest_in_window(
    sat_a: EarthSatellite,
    sat_b: EarthSatellite,
    ts: Timescale,
    times: list[datetime],
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Return (min_distance_km, min_index, vel_a_at_times, vel_b_at_times)."""
    t = ts.from_datetimes(times)
    geo_a = sat_a.at(t)
    geo_b = sat_b.at(t)

    diff_km = geo_a.position.km - geo_b.position.km
    dist_km = np.sqrt((diff_km ** 2).sum(axis=0))
    min_idx = int(np.argmin(dist_km))

    return float(dist_km[min_idx]), min_idx, geo_a.velocity.km_per_s, geo_b.velocity.km_per_s


def build_coarse_times(start_time: datetime, hours: int, coarse_step_minutes: int) -> list[datetime]:
    """The coarse pass's sample times - a pure function of (start_time,
    hours, coarse_step_minutes) so a caller screening many pairs over the
    same window (see CelesTrakAdapter) can build this once and reuse it for
    every satellite, rather than each pair recomputing an identical list."""
    n_coarse_steps = int(hours * 60 / coarse_step_minutes) + 1
    return [start_time + timedelta(minutes=coarse_step_minutes * i) for i in range(n_coarse_steps)]


def compute_coarse_positions(sat: EarthSatellite, ts: Timescale, coarse_times: list[datetime]) -> np.ndarray:
    """Position (km, shape (3, len(coarse_times))) for one satellite at each
    coarse time. This is the expensive part of screening many pairs - a
    vectorized SGP4 propagation - so a caller screening a whole pool of
    satellites should call this ONCE per satellite (see
    CelesTrakAdapter._rank_conjunctions) and reuse the result across every
    pair that satellite appears in, instead of find_closest_approach's
    per-pair default of recomputing it independently for every pair."""
    t = ts.from_datetimes(coarse_times)
    return sat.at(t).position.km


def coarse_min_distance(positions_a: np.ndarray, positions_b: np.ndarray) -> tuple[float, int]:
    """min distance (km) and its index, from two precomputed coarse position
    arrays (see compute_coarse_positions) - pure numpy, no further
    propagation. Like any coarse-pass result, this is only an UPPER BOUND
    on the true closest approach (see module docstring); ranking many pairs
    by this is a cheap, sound way to pick which ones are worth refining."""
    diff_km = positions_a - positions_b
    dist_km = np.sqrt((diff_km ** 2).sum(axis=0))
    min_idx = int(np.argmin(dist_km))
    return float(dist_km[min_idx]), min_idx


def refine_closest_approach(
    sat_a: EarthSatellite,
    sat_b: EarthSatellite,
    ts: Timescale,
    coarse_min_time: datetime,
    coarse_step_minutes: int = 5,
    fine_step_seconds: int = 10,
) -> dict:
    """Pass 2 of the two-pass search: given an already-known approximate
    closest-approach time (from a coarse pass - see coarse_min_distance),
    refines it into a precise minimum distance, time, and relative
    velocity by sampling a narrow +/- coarse_step_minutes window around it
    at fine_step_seconds."""
    fine_window = timedelta(minutes=coarse_step_minutes)
    fine_start = coarse_min_time - fine_window
    n_fine_steps = int((2 * fine_window).total_seconds() / fine_step_seconds) + 1
    fine_times = [
        fine_start + timedelta(seconds=fine_step_seconds * i) for i in range(n_fine_steps)
    ]

    min_distance_km, fine_min_idx, vel_a, vel_b = _closest_in_window(sat_a, sat_b, ts, fine_times)
    time_of_closest_approach = fine_times[fine_min_idx]

    relative_velocity_vec = vel_a[:, fine_min_idx] - vel_b[:, fine_min_idx]
    relative_velocity_km_s = float(np.linalg.norm(relative_velocity_vec))

    return {
        "min_distance_km": min_distance_km,
        "time_of_closest_approach": time_of_closest_approach,
        "relative_velocity_km_s": relative_velocity_km_s,
    }


def find_closest_approach(
    sat_a: EarthSatellite,
    sat_b: EarthSatellite,
    ts: Timescale,
    start_time: datetime,
    hours: int = 48,
    coarse_step_minutes: int = 5,
    fine_step_seconds: int = 10,
) -> dict:
    """Two-pass search for the closest approach between two satellites.

    Pass 1 (coarse): sample the full lookahead window at
    coarse_step_minutes to find an approximate minimum.
    Pass 2 (fine): sample a +/- coarse_step_minutes window around that
    approximate minimum at fine_step_seconds to refine it.

    Convenience wrapper for a single pair - computes its own coarse
    positions internally, from scratch. Screening many pairs against a
    shared satellite pool (see CelesTrakAdapter._rank_conjunctions) should
    instead call compute_coarse_positions once per satellite and
    refine_closest_approach only for the most promising pairs - repeating
    this function's per-pair coarse propagation does not scale past a few
    dozen objects (O(pairs), not O(satellites)).
    """
    coarse_times = build_coarse_times(start_time, hours, coarse_step_minutes)
    positions_a = compute_coarse_positions(sat_a, ts, coarse_times)
    positions_b = compute_coarse_positions(sat_b, ts, coarse_times)
    _, coarse_min_idx = coarse_min_distance(positions_a, positions_b)
    coarse_min_time = coarse_times[coarse_min_idx]

    return refine_closest_approach(
        sat_a, sat_b, ts, coarse_min_time, coarse_step_minutes, fine_step_seconds,
    )
