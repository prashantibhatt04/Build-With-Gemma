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
    """
    n_coarse_steps = int(hours * 60 / coarse_step_minutes) + 1
    coarse_times = [
        start_time + timedelta(minutes=coarse_step_minutes * i) for i in range(n_coarse_steps)
    ]
    _, coarse_min_idx, _, _ = _closest_in_window(sat_a, sat_b, ts, coarse_times)
    coarse_min_time = coarse_times[coarse_min_idx]

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
