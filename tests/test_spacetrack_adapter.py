"""Tests for src/ingestion/spacetrack_adapter.py. The SpaceTrackClient is
faked throughout (never hits space-track.org for real) - the same
overlapping-altitude fixtures test_celestrak_adapter.py uses (small RAAN/
mean-anomaly perturbations of the real ISS TLE, sharing one ~400-408km
LEO band so every pair survives Phase 3's real apogee/perigee filter -
see that module's docstring) are reused here, so the shared screening
algorithm (inherited from CelesTrakAdapter) is exercised against
physically plausible orbital data either way, just with a fake
credential/session layer standing in for a real Space-Track account.
"""
from unittest.mock import MagicMock

import pytest

from src.ingestion.spacetrack_adapter import EnrichedSpaceTrackAdapter, SpaceTrackAdapter

STATIONS_TLE_TEXT = """TEST SAT A
1 30001U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30001  51.6402 175.0000 0004018  88.8954 100.0000 15.54059185113452
"""

DEBRIS_TLE_TEXT = """TEST SAT B
1 30002U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30002  51.6402 190.0000 0004018  88.8954 200.0000 15.54059185113452
TEST SAT C
1 30003U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 30003  51.6402 160.0000 0004018  88.8954 300.0000 15.54059185113452
"""


def _fake_client():
    client = MagicMock()

    def _query_text(path):
        if "ISS" in path:
            return STATIONS_TLE_TEXT
        return DEBRIS_TLE_TEXT

    client.query_text.side_effect = _query_text
    return client


def test_fetch_batch_sources_from_both_group_patterns_and_labels_source(tmp_path):
    client = _fake_client()
    adapter = SpaceTrackAdapter(client=client, cache_dir=tmp_path)

    events = adapter.fetch_batch(limit=10)

    assert len(events) > 0
    for event in events:
        assert event.source == "spacetrack"
        assert event.event_id.startswith("conj-")
    # 1 station object + 2 debris objects, both groups fetched.
    assert client.query_text.call_count == 2


def test_fetch_batch_caches_per_group_and_avoids_requerying(tmp_path):
    client = _fake_client()
    adapter = SpaceTrackAdapter(client=client, cache_dir=tmp_path)

    adapter.fetch_batch(limit=10)
    assert client.query_text.call_count == 2

    adapter.fetch_batch(limit=10)
    assert client.query_text.call_count == 2, "second scan should reuse the disk cache, not re-query"


def test_fetch_batch_respects_custom_group_name_patterns(tmp_path):
    client = _fake_client()
    adapter = SpaceTrackAdapter(
        client=client,
        group_name_patterns={"cosmos-2251-debris": "COSMOS 2251 DEB"},
        exclude_within_group=(),
        cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=10)

    assert client.query_text.call_count == 1
    assert len(events) == 1, "2 debris objects -> exactly 1 pair"
    assert events[0].raw_data["object_a_group"] == "cosmos-2251-debris"
    assert events[0].raw_data["object_b_group"] == "cosmos-2251-debris"


def test_enriched_adapter_attaches_a_real_matching_cdm(tmp_path):
    """cdm_enrichment.py now also requires the CDM's own TCA to land
    close to the freshly-screened event's real (now-relative) TCA - a
    real QA-found bug fix (matching by object pair alone could silently
    apply an unrelated pass's stale Pc). Discovers the real computed TCA
    from a plain, unenriched scan first (same fake TLE data, same cache
    dir - deterministic given identical input), then builds a CDM
    fixture that actually lines up with it, rather than an arbitrary
    fixed date that would only coincidentally match "now"."""
    client = _fake_client()
    plain_adapter = SpaceTrackAdapter(
        client=client, group_name_patterns={"cosmos-2251-debris": "COSMOS 2251 DEB"},
        exclude_within_group=(), cache_dir=tmp_path,
    )
    real_tca = plain_adapter.fetch_batch(limit=10)[0].raw_data["time_of_closest_approach"]

    client.fetch_recent_cdms.return_value = [{
        "PC": "3.5e-04", "TCA": real_tca, "MIN_RNG": "120.0",
        "CREATED": "2026-08-09T15:00:00.000000", "SAT_1_ID": "30002", "SAT_2_ID": "30003",
    }]
    adapter = EnrichedSpaceTrackAdapter(
        client=client, group_name_patterns={"cosmos-2251-debris": "COSMOS 2251 DEB"},
        exclude_within_group=(), cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=10)

    assert len(events) == 1
    assert events[0].raw_data["collision_probability"] == pytest.approx(3.5e-04)


def test_enriched_adapter_leaves_events_unchanged_when_no_cdms_match(tmp_path):
    client = _fake_client()
    client.fetch_recent_cdms.return_value = []
    adapter = EnrichedSpaceTrackAdapter(
        client=client, group_name_patterns={"cosmos-2251-debris": "COSMOS 2251 DEB"},
        exclude_within_group=(), cache_dir=tmp_path,
    )

    events = adapter.fetch_batch(limit=10)

    assert len(events) == 1
    assert "collision_probability" not in events[0].raw_data


def test_enriched_adapter_exposes_the_underlying_scan_stats(tmp_path):
    client = _fake_client()
    client.fetch_recent_cdms.return_value = []
    adapter = EnrichedSpaceTrackAdapter(client=client, cache_dir=tmp_path)

    adapter.fetch_batch(limit=10)

    assert adapter.last_scan_stats is not None
    assert adapter.last_scan_stats["groups"] == ["stations", "cosmos-2251-debris"]
