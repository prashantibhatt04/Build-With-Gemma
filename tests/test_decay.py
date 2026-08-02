"""Tests for src/decay.py. Real, fixed TLE fixtures (same ISS ZARYA
fixture test_orbital.py uses) - no network calls, but real SGP4 parsing.
"""
from skyfield.api import EarthSatellite, load

from src.decay import assess_decay_risk

# Real ISS (ZARYA) TLE - same fixture used elsewhere in this test suite.
ISS_LINE1 = "1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998"
ISS_LINE2 = "2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452"


def test_assess_decay_risk_uses_skyfields_own_parsed_orbital_elements():
    ts = load.timescale()
    sat = EarthSatellite(ISS_LINE1, ISS_LINE2, "ISS (ZARYA)", ts)

    result = assess_decay_risk("25544", "ISS (ZARYA)", sat, tle_epoch_age_hours=12.0)

    assert result["object_id"] == "25544"
    assert result["object_name"] == "ISS (ZARYA)"
    # Real ISS TLE at this epoch - a real, sane LEO altitude, not a
    # placeholder or synthetic number.
    assert 300 < result["perigee_altitude_km"] < 500
    assert result["apogee_altitude_km"] >= result["perigee_altitude_km"]
    assert result["bstar"] == sat.model.bstar
    assert result["tle_epoch_age_hours"] == 12.0


def test_assess_decay_risk_perigee_never_exceeds_apogee():
    """Sanity check across a second real fixture (Vanguard 1, a highly
    eccentric orbit) - perigee <= apogee should hold for any real orbit,
    not just the near-circular ISS case."""
    ts = load.timescale()
    l1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    l2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
    sat = EarthSatellite(l1, l2, "VANGUARD 1", ts)

    result = assess_decay_risk("00005", "VANGUARD 1", sat, tle_epoch_age_hours=1.0)

    assert result["perigee_altitude_km"] <= result["apogee_altitude_km"]
    assert result["perigee_altitude_km"] > 0
