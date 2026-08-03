"""Tests for src/ingestion/cdm_enrichment.py. SpaceTrackClient is faked -
never hits space-track.org for real.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.ingestion.cdm_enrichment import enrich_conjunction_events_with_pc
from src.schemas import TelemetryEvent

REAL_SHAPED_CDM_ROW = {
    "PC": "3.5e-04",
    "TCA": "2026-08-10T04:12:00.000000",
    "MIN_RNG": "120.0",
    "CREATED": "2026-08-09T15:00:00.000000",
    "SAT_1_ID": "25544",
    "SAT_2_ID": "22675",
}


def _conjunction_event(event_id, object_a_id, object_b_id):
    return TelemetryEvent(
        event_id=event_id,
        timestamp=datetime.now(timezone.utc),
        source="spacetrack",
        raw_data={
            "object_a_id": object_a_id, "object_a_name": "A",
            "object_b_id": object_b_id, "object_b_name": "B",
            "min_distance_km": 250.0,
            "time_of_closest_approach": "2026-08-10T04:12:00+00:00",
            "relative_velocity_km_s": 10.0,
        },
    )


def _fake_client(cdm_rows):
    client = MagicMock()
    client.fetch_recent_cdms.return_value = cdm_rows
    return client


def test_enriches_the_matching_event_regardless_of_pair_order():
    client = _fake_client([REAL_SHAPED_CDM_ROW])
    matching = _conjunction_event("conj-1", "22675", "25544")  # reversed order vs the CDM row
    non_matching = _conjunction_event("conj-2", "1", "2")

    enriched = enrich_conjunction_events_with_pc([matching, non_matching], client)

    assert enriched[0].raw_data["collision_probability"] == 3.5e-04
    assert enriched[0].raw_data["min_distance_km"] == 250.0  # original fields preserved
    assert "collision_probability" not in enriched[1].raw_data


def test_returns_events_unchanged_when_no_cdms_available():
    client = _fake_client([])
    event = _conjunction_event("conj-1", "25544", "22675")

    enriched = enrich_conjunction_events_with_pc([event], client)

    assert enriched == [event]
    assert "collision_probability" not in enriched[0].raw_data


def test_skips_non_conjunction_events_without_erroring():
    client = _fake_client([REAL_SHAPED_CDM_ROW])
    decay_event = TelemetryEvent(
        event_id="decay-1", timestamp=datetime.now(timezone.utc), source="celestrak-decay",
        raw_data={"object_id": "1", "object_name": "X", "perigee_altitude_km": 250.0},
    )

    enriched = enrich_conjunction_events_with_pc([decay_event], client)

    assert enriched == [decay_event]


def test_original_event_objects_are_not_mutated():
    client = _fake_client([REAL_SHAPED_CDM_ROW])
    event = _conjunction_event("conj-1", "25544", "22675")
    original_raw = dict(event.raw_data)

    enrich_conjunction_events_with_pc([event], client)

    assert event.raw_data == original_raw
