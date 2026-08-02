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

# A second, disjoint group's worth of real TLEs (2 more real fixtures), for
# the multi-group cross-screening tests below.
OTHER_GROUP_TLE_TEXT = """LAGEOS 1
1 08820U 76039A   26100.50000000  .00000001  00000-0  00000-0 0  9990
2 08820 109.8434  50.0000 0044000  90.0000 270.0000  6.38664960123456
LAGEOS 2
1 22195U 92070B   26100.50000000  .00000001  00000-0  00000-0 0  9991
2 22195  52.6435  60.0000 0013000 100.0000 260.0000 6.47294050223456
"""


def _mock_response(text):
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock()
    return response


@patch("src.ingestion.celestrak_adapter.requests.get")
def test_fetch_batch_caches_and_avoids_refetching(mock_get, tmp_path):
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
    adapter = CelesTrakAdapter(
        groups=["test-group"], sample_size_per_group=3, lookahead_hours=48, cache_dir=tmp_path,
    )

    adapter.fetch_batch(limit=3)
    assert mock_get.call_count == 1

    adapter.fetch_batch(limit=3)
    assert mock_get.call_count == 1, "second call should reuse the disk cache, not refetch"

    assert (tmp_path / "test-group.txt").exists()


@patch("src.ingestion.celestrak_adapter.requests.get")
def test_fetch_batch_returns_results_sorted_by_distance_ascending(mock_get, tmp_path):
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
    adapter = CelesTrakAdapter(
        groups=["test-group"], sample_size_per_group=3, lookahead_hours=48, cache_dir=tmp_path,
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
        assert raw["object_a_group"] == "test-group"
        assert raw["object_b_group"] == "test-group"
        # SAMPLE_TLE_TEXT's epochs are all real past dates (2000/2004/2018),
        # so age relative to "now" should always be positive and sizeable.
        assert raw["tle_epoch_age_hours"] > 0


@patch("src.ingestion.celestrak_adapter.requests.get")
def test_fetch_batch_screens_across_multiple_groups(mock_get, tmp_path):
    """Phase 10: the whole point of multiple groups is cross-group
    screening (e.g. real assets vs. real debris) - confirm pairs spanning
    both groups actually show up, each tagged with its real source group,
    not just pairs within a single group."""
    mock_get.side_effect = [
        _mock_response(SAMPLE_TLE_TEXT), _mock_response(OTHER_GROUP_TLE_TEXT),
    ]
    adapter = CelesTrakAdapter(
        groups=["test-group", "other-group"], sample_size_per_group=10, cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=50)

    # 3 objects in test-group + 2 in other-group = 5 total -> C(5,2) = 10 pairs
    assert len(events) == 10
    groups_seen = {
        frozenset([e.raw_data["object_a_group"], e.raw_data["object_b_group"]])
        for e in events
    }
    # At least one pair spans BOTH groups, not just within-group pairs.
    assert frozenset(["test-group", "other-group"]) in groups_seen
    assert (tmp_path / "test-group.txt").exists()
    assert (tmp_path / "other-group.txt").exists()


@patch("src.ingestion.celestrak_adapter.requests.get")
def test_fetch_batch_refine_top_k_bounds_expensive_refinement(mock_get, tmp_path):
    """Phase 10: refine_top_k caps how many pairs get the expensive
    fine-pass refinement, regardless of how many total pairs exist -
    confirmed via last_scan_stats, which is what makes screening a much
    larger real pool than a handful of objects actually feasible (see
    CelesTrakAdapter's class docstring for the measured cost of not doing
    this)."""
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
    adapter = CelesTrakAdapter(
        groups=["test-group"], sample_size_per_group=3, refine_top_k=1, cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=50)

    # 3 objects -> 3 pairs total, but only the single closest-by-coarse-
    # distance pair gets refined and returned.
    assert len(events) == 1
    assert adapter.last_scan_stats == {
        "groups": ["test-group"],
        "total_objects": 3,
        "total_pairs_screened": 3,
        "pairs_refined": 1,
        "cross_group_pairs_refined": 0,  # single group - no cross-group pairs exist
    }


@patch("src.ingestion.celestrak_adapter.requests.get")
def test_fetch_batch_guarantees_minimum_cross_group_refinement(mock_get, tmp_path):
    """A dense same-group cluster could otherwise fill the entire
    refine_top_k ranking (live-verified during development with CelesTrak's
    real cosmos-2251-debris field dominating the closest-by-coarse-distance
    ranking), starving cross-group ("asset vs. debris") pairs entirely.
    min_cross_group_refine guarantees at least that many get refined
    regardless, even with refine_top_k set so tight that the naive top-K
    ranking alone might not include any."""
    mock_get.side_effect = [
        _mock_response(SAMPLE_TLE_TEXT), _mock_response(OTHER_GROUP_TLE_TEXT),
    ]
    adapter = CelesTrakAdapter(
        groups=["test-group", "other-group"], sample_size_per_group=10,
        refine_top_k=1, min_cross_group_refine=1, cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=50)

    assert adapter.last_scan_stats["cross_group_pairs_refined"] >= 1
    assert any(e.raw_data["object_a_group"] != e.raw_data["object_b_group"] for e in events)


@patch("src.ingestion.celestrak_adapter.requests.get")
def test_fetch_batch_excludes_pairs_within_an_excluded_group(mock_get, tmp_path):
    """Live-verified during development: CelesTrak's real 'stations' group
    contains crewed stations AND their currently-docked visiting vehicles,
    which sit at ~0km separation from each other on purpose - not a real
    conjunction risk. exclude_within_group skips same-group pairs entirely
    (default: "stations") so they never crowd out cross-group results,
    while still screening cross-group and other within-group pairs
    normally."""
    mock_get.side_effect = [
        _mock_response(SAMPLE_TLE_TEXT), _mock_response(OTHER_GROUP_TLE_TEXT),
    ]
    adapter = CelesTrakAdapter(
        groups=["test-group", "other-group"], sample_size_per_group=10,
        exclude_within_group=["test-group"], cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=50)

    # 3 in test-group + 2 in other-group = C(5,2) = 10 pairs total, minus
    # C(3,2) = 3 excluded within-test-group pairs = 7 remaining.
    assert len(events) == 7
    assert adapter.last_scan_stats["total_pairs_screened"] == 7
    for e in events:
        pair_groups = {e.raw_data["object_a_group"], e.raw_data["object_b_group"]}
        assert pair_groups != {"test-group"}, "a within-excluded-group pair leaked through"
