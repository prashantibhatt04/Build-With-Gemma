"""Tests for src/ingestion/celestrak_adapter.py. The network call is mocked
throughout - these never hit CelesTrak. The mocked response text is a real,
small TLE sample (3 objects, fixed fixtures reused from test_orbital.py)
so parsing/caching/pairwise logic is exercised for real.
"""
from unittest.mock import MagicMock, patch

from src.ingestion.celestrak_adapter import CelesTrakAdapter

SAMPLE_TLE_TEXT = """VANGUARD 1
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
ISS (ZARYA)
1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452
THIRD TEST SAT
1 04632U 70093B   04031.91070959 -.00000084  00000-0  10000-3 0  9955
2 04632  11.4628 273.1101 1450506 207.6000 143.9350  1.20231981 44145
"""


def _mock_response():
    response = MagicMock()
    response.text = SAMPLE_TLE_TEXT
    response.raise_for_status = MagicMock()
    return response


@patch("src.ingestion.celestrak_adapter.requests.get")
def test_fetch_batch_caches_and_avoids_refetching(mock_get, tmp_path):
    mock_get.return_value = _mock_response()
    adapter = CelesTrakAdapter(
        group="test-group", sample_size=3, lookahead_hours=48, cache_dir=tmp_path,
    )

    adapter.fetch_batch(limit=3)
    assert mock_get.call_count == 1

    adapter.fetch_batch(limit=3)
    assert mock_get.call_count == 1, "second call should reuse the disk cache, not refetch"

    assert (tmp_path / "test-group.txt").exists()


@patch("src.ingestion.celestrak_adapter.requests.get")
def test_fetch_batch_returns_results_sorted_by_distance_ascending(mock_get, tmp_path):
    mock_get.return_value = _mock_response()
    adapter = CelesTrakAdapter(
        group="test-group", sample_size=3, lookahead_hours=48, cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=3)

    # 3 objects -> 3 pairs
    assert len(events) == 3

    distances = [e.raw_data["min_distance_km"] for e in events]
    assert distances == sorted(distances)

    for event in events:
        assert event.source == "celestrak"
        assert event.event_id.startswith("conj-")
        raw = event.raw_data
        assert raw["min_distance_km"] > 0
        assert raw["relative_velocity_km_s"] > 0
        assert isinstance(raw["time_of_closest_approach"], str)
        assert {"object_a_id", "object_a_name", "object_b_id", "object_b_name"} <= raw.keys()
        # SAMPLE_TLE_TEXT's epochs are all real past dates (2000/2004/2018),
        # so age relative to "now" should always be positive and sizeable.
        assert raw["tle_epoch_age_hours"] > 0
