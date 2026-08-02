"""Tests for src/ingestion/decay_adapter.py. Network call mocked
throughout - real, fixed TLE fixtures so parsing/ranking is exercised for
real.
"""
from unittest.mock import MagicMock, patch

from src.ingestion.decay_adapter import DecayRiskAdapter

# Two real, fixed TLE fixtures with genuinely different perigee altitudes,
# reused from test_orbital.py/test_decay.py.
SAMPLE_TLE_TEXT = """VANGUARD 1
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
ISS (ZARYA)
1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452
"""


def _mock_response(text):
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock()
    return response


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_batch_returns_events_ranked_by_ascending_perigee(mock_get, tmp_path):
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
    adapter = DecayRiskAdapter(group="test-group", sample_size=10, cache_dir=tmp_path)

    events = adapter.fetch_batch(limit=10)

    assert len(events) == 2
    perigees = [e.raw_data["perigee_altitude_km"] for e in events]
    assert perigees == sorted(perigees)  # lowest (most at-risk) first

    for event in events:
        assert event.source == "celestrak-decay"
        assert event.event_id.startswith("decay-")
        raw = event.raw_data
        assert "object_a_id" not in raw  # single-object shape, not a pair
        assert raw["perigee_altitude_km"] > 0
        assert raw["apogee_altitude_km"] >= raw["perigee_altitude_km"]
        assert raw["tle_epoch_age_hours"] > 0  # both fixtures are real past dates


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_batch_respects_limit(mock_get, tmp_path):
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
    adapter = DecayRiskAdapter(group="test-group", sample_size=10, cache_dir=tmp_path)

    events = adapter.fetch_batch(limit=1)

    assert len(events) == 1


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_batch_event_ids_are_unique_across_separate_scans(mock_get, tmp_path):
    """Same collision-avoidance reasoning as every other adapter in this
    project: DecisionLogger matches an event_id's FIRST logged occurrence,
    so repeat scans of the same real object need distinct ids."""
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)

    first_scan = DecayRiskAdapter(group="test-group", cache_dir=tmp_path).fetch_batch(limit=2)
    second_scan = DecayRiskAdapter(group="test-group", cache_dir=tmp_path).fetch_batch(limit=2)

    first_ids = {e.event_id for e in first_scan}
    second_ids = {e.event_id for e in second_scan}
    assert first_ids.isdisjoint(second_ids)


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_batch_respects_explicit_run_id(mock_get, tmp_path):
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
    adapter = DecayRiskAdapter(group="test-group", cache_dir=tmp_path, run_id="fixed-id")

    events = adapter.fetch_batch(limit=2)

    assert all(e.event_id.endswith("-fixed-id") for e in events)
