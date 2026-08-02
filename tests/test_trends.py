"""Tests for src/trends.py - pure data transforms, no Streamlit involved."""
from datetime import date, datetime, timezone

from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent
from src.trends import (
    build_rationale_source_trend_figure,
    build_severity_trend_figure,
    is_real_live_source,
    rationale_source_counts_by_day,
    recurring_objects,
    severity_counts_by_day,
)


def _entry(
    event_id: str, severity: Severity, timestamp: datetime,
    raw_data: dict, rationale_source: str = "gemma", source: str = "celestrak",
    made_at: datetime | None = None,
) -> DecisionLogEntry:
    telemetry = TelemetryEvent(event_id=event_id, timestamp=timestamp, source=source, raw_data=raw_data)
    finding = AnomalyFinding(event_id=event_id, severity=severity, description="Test.", confidence=0.9)
    decision = Decision(action="continue", rationale="Test rationale.", made_at=made_at or timestamp)
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source=rationale_source, model_used="fake-model", latency_ms=1.0),
    )


def _conjunction_raw(object_a_id, object_a_name, object_b_id, object_b_name):
    return {
        "object_a_id": object_a_id, "object_a_name": object_a_name,
        "object_b_id": object_b_id, "object_b_name": object_b_name,
        "min_distance_km": 50.0,
    }


def _decay_raw(object_id, object_name):
    return {"object_id": object_id, "object_name": object_name, "perigee_altitude_km": 400.0, "apogee_altitude_km": 420.0, "bstar": 0.001}


DAY_1 = datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)
DAY_1_LATER = datetime(2026, 8, 1, 18, 0, tzinfo=timezone.utc)
DAY_2 = datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc)


def test_severity_counts_by_day_buckets_by_real_day_not_exact_timestamp():
    entries = [
        _entry("e1", Severity.WATCH, DAY_1, _conjunction_raw("1", "A", "2", "B")),
        _entry("e2", Severity.CRITICAL, DAY_1_LATER, _conjunction_raw("1", "A", "2", "B")),
        _entry("e3", Severity.WATCH, DAY_2, _conjunction_raw("1", "A", "2", "B")),
    ]

    counts = severity_counts_by_day(entries)

    assert counts[date(2026, 8, 1)] == {"watch": 1, "critical": 1}
    assert counts[date(2026, 8, 2)] == {"watch": 1}


def test_build_severity_trend_figure_has_one_trace_per_severity_band():
    entries = [_entry("e1", Severity.CRITICAL, DAY_1, _conjunction_raw("1", "A", "2", "B"))]

    fig = build_severity_trend_figure(entries)

    names = {trace.name for trace in fig.data}
    assert names == {"NOMINAL", "WATCH", "WARNING", "CRITICAL"}
    critical_trace = next(t for t in fig.data if t.name == "CRITICAL")
    assert list(critical_trace.y) == [1]


def test_recurring_objects_counts_conjunction_pairs_on_both_sides():
    entries = [
        _entry("e1", Severity.WATCH, DAY_1, _conjunction_raw("1", "SAT-A", "2", "SAT-B")),
        _entry("e2", Severity.WATCH, DAY_1, _conjunction_raw("1", "SAT-A", "3", "SAT-C")),
    ]

    result = recurring_objects(entries)

    counts = {row["object_id"]: row["count"] for row in result}
    assert counts["1"] == 2  # SAT-A appears in both events
    assert counts["2"] == 1
    assert counts["3"] == 1


def test_recurring_objects_counts_single_object_hazard_shapes():
    entries = [
        _entry("e1", Severity.WATCH, DAY_1, _decay_raw("33821", "COSMOS 2251 DEB")),
        _entry("e2", Severity.WATCH, DAY_2, _decay_raw("33821", "COSMOS 2251 DEB")),
    ]

    result = recurring_objects(entries)

    assert result == [{"object_id": "33821", "object_name": "COSMOS 2251 DEB", "count": 2, "real": True}]


def test_recurring_objects_groups_by_id_alone_not_by_id_and_name():
    """Regression test: a real object's name string varying slightly
    between two separate fetches (e.g. a CelesTrak catalog rename, or
    stray whitespace) must not split it into two separate rows - the
    same object_id should always count as one recurring object, no
    matter what its name looked like on each individual fetch."""
    entries = [
        _entry("e1", Severity.WATCH, DAY_1, _decay_raw("33821", "COSMOS 2251 DEB")),
        _entry("e2", Severity.WATCH, DAY_2, _decay_raw("33821", "COSMOS 2251 DEB ")),  # trailing space
    ]

    result = recurring_objects(entries)

    assert len(result) == 1
    assert result[0]["object_id"] == "33821"
    assert result[0]["count"] == 2
    # Whichever name was seen LAST is what's displayed - not asserting a
    # specific one here, just that it didn't split into two rows.


def test_recurring_objects_marks_synthetic_objects_as_not_real():
    entries = [
        _entry(
            "e1", Severity.CRITICAL, DAY_1, _conjunction_raw("99000", "SYNTH-A-0", "99010", "SYNTH-B-0"),
            source="synthetic-critical-fixture",
        ),
    ]

    result = recurring_objects(entries)

    assert all(row["real"] is False for row in result)


def test_is_real_live_source_true_only_for_celestrak_sources():
    real_conjunction = _entry("e1", Severity.WATCH, DAY_1, _conjunction_raw("1", "A", "2", "B"), source="celestrak")
    real_decay = _entry("e2", Severity.WATCH, DAY_1, _decay_raw("1", "A"), source="celestrak-decay")
    synthetic = _entry(
        "e3", Severity.CRITICAL, DAY_1, _conjunction_raw("1", "A", "2", "B"), source="synthetic-critical-fixture",
    )
    historical = _entry(
        "e4", Severity.CRITICAL, DAY_1, _conjunction_raw("1", "A", "2", "B"), source="historical-replay",
    )

    assert is_real_live_source(real_conjunction) is True
    assert is_real_live_source(real_decay) is True
    assert is_real_live_source(synthetic) is False
    assert is_real_live_source(historical) is False


def test_day_buckets_by_decision_made_at_not_telemetry_timestamp():
    """Regression test: the entry's telemetry.timestamp is set at
    ingestion, before analyze/decide even run - decision.made_at (set
    right after the Gemma call completes) is much closer to when
    log_node actually writes the entry to disk, and should be what
    Trends buckets by, not the earlier ingestion timestamp."""
    entry = _entry(
        "e1", Severity.WATCH, timestamp=datetime(2026, 8, 1, 23, 59, 58, tzinfo=timezone.utc),
        made_at=datetime(2026, 8, 2, 0, 0, 3, tzinfo=timezone.utc),
        raw_data=_conjunction_raw("1", "A", "2", "B"),
    )

    counts = severity_counts_by_day([entry])

    assert date(2026, 8, 2) in counts
    assert date(2026, 8, 1) not in counts


def test_recurring_objects_skips_entries_with_no_real_object_identity():
    entries = [_entry("e1", Severity.NOMINAL, DAY_1, {"value": 1.0})]

    assert recurring_objects(entries) == []


def test_recurring_objects_respects_top_n_and_orders_most_frequent_first():
    entries = (
        [_entry(f"a{i}", Severity.WATCH, DAY_1, _decay_raw("1", "MOST-FREQUENT")) for i in range(3)]
        + [_entry("b0", Severity.WATCH, DAY_1, _decay_raw("2", "LEAST-FREQUENT"))]
    )

    result = recurring_objects(entries, top_n=1)

    assert len(result) == 1
    assert result[0]["object_id"] == "1"
    assert result[0]["count"] == 3


def test_rationale_source_counts_by_day():
    entries = [
        _entry("e1", Severity.WATCH, DAY_1, _conjunction_raw("1", "A", "2", "B"), rationale_source="gemma"),
        _entry("e2", Severity.WATCH, DAY_1, _conjunction_raw("1", "A", "2", "B"), rationale_source="fallback"),
        _entry("e3", Severity.WATCH, DAY_2, _conjunction_raw("1", "A", "2", "B"), rationale_source="gemma"),
    ]

    counts = rationale_source_counts_by_day(entries)

    assert counts[date(2026, 8, 1)] == {"gemma": 1, "fallback": 1}
    assert counts[date(2026, 8, 2)] == {"gemma": 1}


def test_build_rationale_source_trend_figure_has_two_traces():
    entries = [_entry("e1", Severity.WATCH, DAY_1, _conjunction_raw("1", "A", "2", "B"), rationale_source="fallback")]

    fig = build_rationale_source_trend_figure(entries)

    names = {trace.name for trace in fig.data}
    assert names == {"gemma", "fallback"}
    fallback_trace = next(t for t in fig.data if t.name == "fallback")
    assert list(fallback_trace.y) == [1]
