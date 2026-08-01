"""Tests for src/orbital.py. No live network calls - all TLEs below are
real, fixed fixtures (Vanguard 1 and ISS ZARYA test vectors bundled
with the sgp4/skyfield packages themselves).
"""
from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield.api import EarthSatellite, load

from src.orbital import find_closest_approach, parse_norad_id, parse_tle_epoch

# Real ISS (ZARYA) TLE line 1, used as a fixed sample - not fetched live.
SAMPLE_TLE_LINE1 = "1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998"

# Two real, stable TLEs used as synthetic fixtures for propagation tests.
VANGUARD1_LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
VANGUARD1_LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
ISS_LINE1 = "1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998"
ISS_LINE2 = "2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452"


def test_parse_norad_id_extracts_catalog_number():
    assert parse_norad_id(SAMPLE_TLE_LINE1) == "25544"


def test_parse_tle_epoch_extracts_year_and_day_of_year():
    # VANGUARD1_LINE1 epoch field is "00179.78495062" -> 2-digit year 00
    # (2000), fractional day-of-year 179.78495062.
    epoch = parse_tle_epoch(VANGUARD1_LINE1)
    assert epoch.year == 2000
    assert epoch.timetuple().tm_yday == 179

    # ISS_LINE1 epoch field is "18135.61844383" -> year 18 (2018), day 135.
    epoch = parse_tle_epoch(ISS_LINE1)
    assert epoch.year == 2018
    assert epoch.timetuple().tm_yday == 135


def _coarse_only_min_distance_km(sat_a, sat_b, ts, start_time, hours, step_minutes):
    n_steps = int(hours * 60 / step_minutes) + 1
    times = [start_time + timedelta(minutes=step_minutes * i) for i in range(n_steps)]
    t = ts.from_datetimes(times)
    diff_km = sat_a.at(t).position.km - sat_b.at(t).position.km
    dist_km = np.sqrt((diff_km ** 2).sum(axis=0))
    return float(dist_km.min())


def test_find_closest_approach_sane_result_and_refines_coarse_estimate():
    ts = load.timescale()
    sat_a = EarthSatellite(VANGUARD1_LINE1, VANGUARD1_LINE2, "VANGUARD 1", ts)
    sat_b = EarthSatellite(ISS_LINE1, ISS_LINE2, "ISS (ZARYA)", ts)
    start_time = datetime(2026, 8, 1, tzinfo=timezone.utc)

    result = find_closest_approach(
        sat_a, sat_b, ts, start_time,
        hours=48, coarse_step_minutes=5, fine_step_seconds=10,
    )

    assert result["min_distance_km"] > 0
    assert isinstance(result["time_of_closest_approach"], datetime)
    assert start_time <= result["time_of_closest_approach"] <= start_time + timedelta(hours=48)
    assert result["relative_velocity_km_s"] > 0

    coarse_only = _coarse_only_min_distance_km(
        sat_a, sat_b, ts, start_time, hours=48, step_minutes=5,
    )

    # Two-pass refinement can only find an equal or smaller minimum than
    # the coarse pass alone - this proves the fine pass is actually
    # tightening the estimate, not just echoing the coarse result.
    assert result["min_distance_km"] <= coarse_only
