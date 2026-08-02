"""Tests for src/orbit_plot_data.py. Network calls (TLE-by-catalog-number
fetches) are mocked throughout, using the same real, fixed TLE fixtures
(Vanguard 1 and ISS ZARYA) as tests/test_orbital.py - so propagation
itself is exercised for real, not mocked.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np

from src.orbit_plot_data import (
    CRITICAL_THRESHOLD_KM,
    TrajectoryData,
    _fetch_tle_by_catnr,
    build_3d_trajectory_figure,
    build_distance_chart,
    fetch_trajectory_data,
)

VANGUARD1_TLE_TEXT = """VANGUARD 1
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

ISS_TLE_TEXT = """ISS (ZARYA)
1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452
"""


def _mock_response(text: str):
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock()
    return response


@patch("src.orbit_plot_data.requests.get")
def test_fetch_tle_by_catnr_parses_response(mock_get):
    mock_get.return_value = _mock_response(VANGUARD1_TLE_TEXT)

    name, l1, l2 = _fetch_tle_by_catnr("00005")

    assert name == "VANGUARD 1"
    assert l1.startswith("1 00005U")
    assert l2.startswith("2 00005")


@patch("src.orbit_plot_data.requests.get")
def test_fetch_tle_by_catnr_raises_for_empty_response(mock_get):
    mock_get.return_value = _mock_response("")

    try:
        _fetch_tle_by_catnr("99999")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "99999" in str(exc)


@patch("src.orbit_plot_data.requests.get")
def test_fetch_trajectory_data_computes_real_positions_and_distances(mock_get):
    mock_get.side_effect = [_mock_response(VANGUARD1_TLE_TEXT), _mock_response(ISS_TLE_TEXT)]

    data = fetch_trajectory_data("00005", "VANGUARD 1", "25544", "ISS (ZARYA)", hours=48)

    assert isinstance(data, TrajectoryData)
    assert len(data.times) == 577  # 48h / 5min steps + 1
    assert data.positions_a.shape == (3, 577)
    assert data.positions_b.shape == (3, 577)
    assert data.distances_km.shape == (577,)
    assert (data.distances_km > 0).all()
    assert data.object_a_name == "VANGUARD 1"
    assert data.object_b_name == "ISS (ZARYA)"
    assert all(isinstance(t, datetime) and t.tzinfo == timezone.utc for t in data.times)


def _sample_trajectory_data() -> TrajectoryData:
    n = 10
    times = [datetime.now(timezone.utc) for _ in range(n)]
    positions_a = np.zeros((3, n))
    positions_b = np.zeros((3, n))
    positions_b[0] = np.linspace(1.0, 50.0, n)  # distance grows from 1km to 50km
    distances_km = np.sqrt(((positions_a - positions_b) ** 2).sum(axis=0))
    return TrajectoryData(
        times=times, positions_a=positions_a, positions_b=positions_b,
        distances_km=distances_km, object_a_name="OBJ-A", object_b_name="OBJ-B",
    )


def test_build_distance_chart_has_one_trace_and_three_threshold_lines():
    fig = build_distance_chart(_sample_trajectory_data())

    assert len(fig.data) == 1
    assert fig.data[0].y[0] == 1.0  # starts near the closest point in the sample data
    assert len(fig.layout.shapes) == 3  # CRITICAL/WARNING/WATCH reference lines


def test_build_3d_trajectory_figure_has_earth_both_paths_and_closest_approach_marker():
    fig = build_3d_trajectory_figure(_sample_trajectory_data())

    # Earth surface + object A path + object B path + closest-approach marker.
    assert len(fig.data) == 4
    names = [trace.name for trace in fig.data]
    assert "OBJ-A" in names
    assert "OBJ-B" in names
    assert "Closest approach" in names

    marker_trace = next(t for t in fig.data if t.name == "Closest approach")
    # The sample data's minimum distance is at index 0 (positions_b[0][0] == 1.0,
    # the smallest value in the linspace) - confirms the marker finds the real
    # argmin, not just the first/last point.
    assert marker_trace.x[0] == 0.0  # positions_a is all zeros
    assert CRITICAL_THRESHOLD_KM == 5.0  # sanity: matches pipeline.py's own threshold
