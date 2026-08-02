"""Tests for src/live_positions.py. Network calls are mocked, using the
same real, fixed TLE fixtures (Vanguard 1 and ISS ZARYA) other tests in
this suite use - propagation itself is exercised for real, not mocked.
"""
from unittest.mock import MagicMock, patch

from src.live_positions import SatellitePosition, build_live_globe_figure, fetch_live_positions

STATIONS_TLE_TEXT = """ISS (ZARYA)
1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452
VANGUARD 1
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""


def _mock_response(text: str):
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock()
    return response


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_live_positions_computes_real_current_positions(mock_get, tmp_path):
    mock_get.return_value = _mock_response(STATIONS_TLE_TEXT)

    positions = fetch_live_positions(group="stations", cache_dir=tmp_path)

    assert len(positions) == 2
    names = [p.name for p in positions]
    assert "ISS (ZARYA)" in names
    assert "VANGUARD 1" in names
    for p in positions:
        assert isinstance(p, SatellitePosition)
        assert p.norad_id in ("25544", "00005")
        # A real object shouldn't be sitting at Earth's center.
        assert (p.x_km ** 2 + p.y_km ** 2 + p.z_km ** 2) ** 0.5 > 1000
        assert p.altitude_km > 0


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_live_positions_uses_disk_cache(mock_get, tmp_path):
    mock_get.return_value = _mock_response(STATIONS_TLE_TEXT)

    fetch_live_positions(group="stations", cache_dir=tmp_path)
    fetch_live_positions(group="stations", cache_dir=tmp_path)

    assert mock_get.call_count == 1


def test_build_live_globe_figure_has_earth_and_one_marker_per_object():
    positions = [
        SatellitePosition(name="ISS (ZARYA)", norad_id="25544", x_km=6800.0, y_km=0.0, z_km=0.0, altitude_km=420.0),
        SatellitePosition(name="TIANGONG", norad_id="48274", x_km=0.0, y_km=6750.0, z_km=100.0, altitude_km=380.0),
    ]

    fig = build_live_globe_figure(positions)

    # Earth surface + one combined marker trace for every satellite.
    assert len(fig.data) == 2
    marker_trace = next(t for t in fig.data if t.name == "Live position")
    assert list(marker_trace.x) == [6800.0, 0.0]
    assert list(marker_trace.y) == [0.0, 6750.0]
    assert list(marker_trace.z) == [0.0, 100.0]
    assert list(marker_trace.text) == ["ISS (ZARYA)", "TIANGONG"]
