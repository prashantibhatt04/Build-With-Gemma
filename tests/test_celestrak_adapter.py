"""Tests for src/ingestion/celestrak_adapter.py. The network call is mocked
throughout - these never hit CelesTrak. The mocked response text is a
fixed TLE sample (5 objects) sharing a common ~400-408km LEO altitude
band, so every pair's altitude ranges genuinely overlap and survives
Phase 3's real apogee/perigee filter (see src/catalog_screening.py) -
each object is a small RAAN/mean-anomaly perturbation of the real ISS
(ZARYA) TLE used elsewhere in this suite, the same way a real debris
cluster from one breakup event shares its parent's orbital regime.
Distinct RAAN/mean anomaly still gives each pair a real, distinct
closest-approach result under propagation, so parsing/caching/pairwise
logic is still exercised for real, just with physically plausible inputs
now that screening actually checks plausibility.
"""
from unittest.mock import MagicMock, patch

from src.ingestion.celestrak_adapter import CelesTrakAdapter

SAMPLE_TLE_TEXT = """TEST SAT A
1 30001U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30001  51.6402 175.0000 0004018  88.8954 100.0000 15.54059185113452
TEST SAT B
1 30002U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30002  51.6402 190.0000 0004018  88.8954 200.0000 15.54059185113452
TEST SAT C
1 30003U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30003  51.6402 160.0000 0004018  88.8954 300.0000 15.54059185113452
"""

# A second, disjoint group's worth of TLEs (2 more), for the multi-group
# cross-screening tests below - same shared altitude band as SAMPLE_TLE_TEXT
# above, so cross-group pairs genuinely overlap too.
OTHER_GROUP_TLE_TEXT = """OTHER SAT D
1 30004U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30004  51.6402 210.0000 0004018  88.8954  50.0000 15.54059185113452
OTHER SAT E
1 30005U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30005  51.6402 145.0000 0004018  88.8954 150.0000 15.54059185113452
"""


def _mock_response(text):
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock()
    return response


@patch("src.ingestion.tle_source.requests.get")
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


@patch("src.ingestion.tle_source.requests.get")
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


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_batch_event_ids_are_unique_across_separate_scans(mock_get, tmp_path):
    """Regression test: the SAME real object pair (identical TLEs -> same
    closest-approach result here) previously produced the IDENTICAL
    event_id on every scan. DecisionLogger.find_entry/mark_reviewed/
    approve_maneuver match an event_id's FIRST logged occurrence, so a
    second scan's entry was indistinguishable from an already-resolved
    earlier one - approving "this scan's" pending entry could silently
    resolve a stale one instead. Each CelesTrakAdapter instance now gets
    its own run_id, matching SyntheticCriticalAdapter/HistoricalReplayAdapter."""
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
    first_scan = CelesTrakAdapter(
        groups=["test-group"], sample_size_per_group=3, cache_dir=tmp_path,
    ).fetch_batch(limit=3)
    second_scan = CelesTrakAdapter(
        groups=["test-group"], sample_size_per_group=3, cache_dir=tmp_path,
    ).fetch_batch(limit=3)

    first_ids = {e.event_id for e in first_scan}
    second_ids = {e.event_id for e in second_scan}
    assert first_ids.isdisjoint(second_ids)


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_batch_respects_explicit_run_id(mock_get, tmp_path):
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
    adapter = CelesTrakAdapter(
        groups=["test-group"], sample_size_per_group=3, cache_dir=tmp_path, run_id="fixed-id",
    )

    events = adapter.fetch_batch(limit=3)

    assert all(e.event_id.endswith("-fixed-id") for e in events)


@patch("src.ingestion.tle_source.requests.get")
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


@patch("src.ingestion.tle_source.requests.get")
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
        "total_possible_pairs": 3,
        "pairs_after_ap_filter": 3,  # all 3 share the same altitude band - none eliminated
        "total_pairs_screened": 3,
        "pairs_refined": 1,
        "cross_group_pairs_refined": 0,  # single group - no cross-group pairs exist
    }


@patch("src.ingestion.tle_source.requests.get")
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


@patch("src.ingestion.tle_source.requests.get")
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


WATCHED_TLE_TEXT = """TEST SAT A
1 30001U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30001  51.6402 175.0000 0004018  88.8954 100.0000 15.54059185113452
"""


def test_watched_norad_ids_are_screened_under_a_dedicated_group_label(tmp_path):
    """Real feature this tests: a customer's own asset(s), given by
    NORAD catalog ID rather than a curated group name, is what actually
    lets this project screen a real satellite that isn't a member of any
    CelesTrak-curated group. "my satellite" isn't a special case for the
    cross-group screening algorithm below - it's fetched differently
    (fetch_watched_ids_fn, injected here) but then treated exactly like
    any other real group (WATCHED_GROUP_LABEL)."""
    def fake_fetch_group(group, cache_dir):
        return OTHER_GROUP_TLE_TEXT

    def fake_fetch_watched(catalog_ids, cache_dir):
        assert list(catalog_ids) == ["30001"]
        return WATCHED_TLE_TEXT

    adapter = CelesTrakAdapter(
        groups=["debris"], watched_norad_ids=["30001"],
        fetch_group_fn=fake_fetch_group, fetch_watched_ids_fn=fake_fetch_watched,
        sample_size_per_group=10, cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=10)

    assert len(events) > 0
    groups_seen = set()
    for e in events:
        groups_seen.add(e.raw_data["object_a_group"])
        groups_seen.add(e.raw_data["object_b_group"])
    assert "my-assets" in groups_seen
    assert "my-assets" in adapter.last_scan_stats["groups"]


WELL_SEPARATED_WATCHED_ASSET_TLE_TEXT = """MY SATELLITE
1 30006U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30006  51.6402 130.0000 0004018  88.8954 400.0000 15.54059185113452
"""


def test_watched_norad_ids_excludes_debris_vs_debris_noise(tmp_path):
    """Real bug this closes, found by a live customer walkthrough: a
    dense debris field's own internal pairs (SAMPLE_TLE_TEXT's 3 objects
    are deliberately close to EACH OTHER - see this file's module
    docstring) are almost always closer than an unrelated, well-
    separated real satellite is to any of them. Without filtering,
    clicking "fetch conjunctions" with a real watch list configured
    could return a result set that's 100% debris-vs-debris noise, never
    once mentioning the customer's own configured asset - the entire
    point of having configured it in the first place."""
    def fake_fetch_group(group, cache_dir):
        return SAMPLE_TLE_TEXT  # 3 objects, close to EACH OTHER

    def fake_fetch_watched(catalog_ids, cache_dir):
        return WELL_SEPARATED_WATCHED_ASSET_TLE_TEXT  # 1 real, separated asset

    adapter = CelesTrakAdapter(
        groups=["debris"], watched_norad_ids=["30006"],
        fetch_group_fn=fake_fetch_group, fetch_watched_ids_fn=fake_fetch_watched,
        sample_size_per_group=10, cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=10)

    assert len(events) > 0
    for e in events:
        assert "my-assets" in (e.raw_data["object_a_group"], e.raw_data["object_b_group"]), \
            "a debris-vs-debris pair leaked through despite a watch list being configured"


def test_fetch_watched_ids_fn_is_never_called_when_watch_list_is_empty(tmp_path):
    """The default, zero-config path (no WATCHED_NORAD_IDS configured)
    must be completely unaffected - no extra real request, no
    "my-assets" group appearing anywhere."""
    watched_fetch_calls = []

    def fake_fetch_group(group, cache_dir):
        return SAMPLE_TLE_TEXT

    def fake_fetch_watched(catalog_ids, cache_dir):
        watched_fetch_calls.append(catalog_ids)
        return ""

    adapter = CelesTrakAdapter(
        groups=["test-group"], sample_size_per_group=10, cache_dir=tmp_path,
        fetch_group_fn=fake_fetch_group, fetch_watched_ids_fn=fake_fetch_watched,
    )

    adapter.fetch_batch(limit=10)

    assert watched_fetch_calls == []
    assert "my-assets" not in adapter.last_scan_stats["groups"]
