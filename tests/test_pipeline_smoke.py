"""Smoke test: run the full pipeline against DummyAdapter with a stubbed
GemmaClient, no live network required.
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.gemma_client import CIRCUIT_BREAKER_THRESHOLD, GemmaClient, GemmaClientError
from src.ingestion.base_adapter import DummyAdapter
from src.logging_utils import DecisionLogger
from src.maneuver import DeltaVBudgetTracker
from src.pipeline import (
    _MANEUVER_VETO_SYSTEM,
    _VETO_JSON_SCHEMA,
    _call_gemma_with_provenance,
    _extract_final_answer,
    _parse_veto_json,
    _parse_veto_verdict,
    classify_attitude_severity,
    classify_conjunction_severity,
    classify_decay_severity,
    make_analyze_node,
    make_decide_node,
    make_log_node,
    run_once,
)
from src.schemas import Action, AnomalyFinding, Decision, GemmaProvenance, ManeuverPlan, Severity, TelemetryEvent, VerifiedClearance


def test_extract_final_answer_leaves_clean_single_paragraph_unchanged():
    text = "Recommendation: continue. The predicted minimum distance is well above risk thresholds."
    assert _extract_final_answer(text) == text


def test_extract_final_answer_takes_last_line_from_verbose_reasoning():
    # A trimmed real example captured from the hosted API's actual response
    # style - drafts/checklist reasoning before the final answer, separated
    # by blank lines.
    verbose = (
        "The user wants a single-word reply: \"ok\".\n\n"
        "*   Constraint: Single word.\n"
        "*   Content: \"ok\".\n\n"
        "Wait, let me reconsider the phrasing.\n\n"
        "Recommendation: continue. The projected minimum distance of 50,000 "
        "kilometers between the objects is sufficient to maintain a safe margin."
    )
    assert _extract_final_answer(verbose) == (
        "Recommendation: continue. The projected minimum distance of 50,000 "
        "kilometers between the objects is sufficient to maintain a safe margin."
    )


def test_extract_final_answer_handles_single_newline_label_before_answer():
    # A real example captured from the hosted API: a label and the actual
    # answer on consecutive lines with only a single newline between them
    # (no blank-line paragraph break) - the \n\n-based version of this
    # function missed this case; the current line-based version should not.
    verbose = (
        "Final Polish:\n"
        "    Recommendation: continue. The predicted distance of 50 kilometers "
        "between the objects is sufficient to maintain safety."
    )
    assert _extract_final_answer(verbose) == (
        "Recommendation: continue. The predicted distance of 50 kilometers "
        "between the objects is sufficient to maintain safety."
    )


def test_extract_final_answer_handles_empty_string():
    assert _extract_final_answer("") == ""
    assert _extract_final_answer("   ") == ""


def test_extract_final_answer_skips_short_throwaway_final_line():
    # Real bug observed in production: the model's literal last line was a
    # short filler remark ("Ok, let's go.") instead of the real content,
    # which sat on the line just above it.
    verbose = (
        "The user wants a single-word reply: \"ok\".\n\n"
        "Recommendation: continue. The projected minimum distance of 50,000 "
        "kilometers between the objects is sufficient to maintain a safe margin.\n\n"
        "Ok, let's go."
    )
    assert _extract_final_answer(verbose) == (
        "Recommendation: continue. The projected minimum distance of 50,000 "
        "kilometers between the objects is sufficient to maintain a safe margin."
    )


def test_extract_final_answer_falls_back_to_last_line_if_nothing_substantive():
    # Degenerate case: every line is short - fall back to the literal last
    # line rather than returning nothing.
    assert _extract_final_answer("Ok.\nSure.\nYes.") == "Yes."


@pytest.mark.parametrize(
    "text,expected",
    [
        ("GO - the numbers are safe.", True),
        ("NO-GO - inconsistent numbers.", False),
        ("NO GO, this does not look right.", False),
        ("go", True),
        ("no-go", False),
        ("The verdict is GO.", True),
        ("Ergo, the maneuver should proceed.", None),  # "ergo" isn't a false positive
        ("Going forward, this looks fine.", None),  # "going" isn't a false positive either
        ("Sounds fine to me.", None),  # no verdict token at all
        ("", None),
        # Chain-of-thought before the real verdict - last token found wins,
        # same philosophy as _extract_final_answer.
        ("Let me think... maybe GO? Actually, NO-GO given the risk.", False),
        # Regression test: a clear leading GO must NOT be overridden by a
        # later NEGATED mention of "NO-GO" in the model's own explanation -
        # the first, authoritative token wins over a naive last-match scan.
        ("GO - the numbers check out. This is clearly not a NO-GO situation, so proceed.", True),
        ("NO-GO - this is not a situation where GO makes sense.", False),
    ],
)
def test_parse_veto_verdict(text, expected):
    assert _parse_veto_verdict(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"verdict": "GO", "reason": "Numbers check out."}', (True, "Numbers check out.")),
        ('{"verdict": "NO-GO", "reason": "Inconsistent numbers."}', (False, "Inconsistent numbers.")),
        # Real gemma4:e4b output observed directly via Ollama's structured
        # output (confirmed by curl before relying on this) - pretty-printed
        # with real whitespace/newlines, not a compact one-liner.
        ('{\n"verdict": "GO"\n  ,\n"reason": "Looks safe."\n}', (True, "Looks safe.")),
        ("not json at all", None),  # cross-backend fallback to hosted API, or a malformed response
        ('{"verdict": "MAYBE", "reason": "Unclear."}', None),  # schema violation the model still emitted
        ('{"verdict": "GO"}', None),  # missing required "reason"
        ('{"verdict": "GO", "reason": ""}', None),  # empty reason isn't real content
        ('{"verdict": "GO", "reason": "  "}', None),  # whitespace-only reason isn't real content
        ("[1, 2, 3]", None),  # valid JSON, but not an object
        ("", None),
    ],
)
def test_parse_veto_json(text, expected):
    assert _parse_veto_json(text) == expected


def test_veto_json_schema_matches_expected_shape():
    # Sanity check that the schema constant used to REQUEST structured
    # output actually describes what _parse_veto_json expects to receive -
    # these two must never drift apart independently.
    assert _VETO_JSON_SCHEMA["required"] == ["verdict", "reason"]
    assert _VETO_JSON_SCHEMA["properties"]["verdict"]["enum"] == ["GO", "NO-GO"]


class FakeGemmaClient:
    """Duck-types GemmaClient (generate() + settings.gemma_model/
    gemma_backend) without making any network calls. gemma_backend defaults
    to "ollama" so existing CRITICAL-severity tests (written before the
    human-approval feature existed) keep exercising the autonomous path -
    see test_decide_node_requires_human_approval_for_api_backend for the
    "api" case specifically.

    veto_response controls what a CRITICAL-autonomous maneuver's veto-check
    call (see pipeline._maneuver_veto_check) gets back - distinguished from
    every other call by its distinct system prompt, so it can be varied
    independently of the narration/description stub text below. Defaults
    to an affirming "GO" so existing tests not concerned with Phase 9 keep
    exercising the executed path unchanged."""

    def __init__(self, gemma_backend: str = "ollama", veto_response: str = "GO - the verified numbers look safe."):
        self.settings = SimpleNamespace(gemma_model="fake-model", gemma_backend=gemma_backend)
        self.veto_response = veto_response

    def generate(self, prompt: str, system=None, timeout: int = 60, format=None) -> str:
        if system == _MANEUVER_VETO_SYSTEM:
            return self.veto_response
        return "Stubbed anomaly commentary: nothing to report."


class FailingGemmaClient:
    """Duck-types GemmaClient but always raises, to test fallback paths."""

    def __init__(self, gemma_backend: str = "ollama", gemma_model_api: str = "fake-model-api"):
        self.settings = SimpleNamespace(
            gemma_model="fake-model", gemma_backend=gemma_backend, gemma_model_api=gemma_model_api,
        )

    def generate(self, prompt: str, system=None, timeout: int = 60, format=None) -> str:
        raise GemmaClientError("simulated failure")


def test_call_gemma_with_provenance_records_the_real_prompt_on_success():
    """GemmaProvenance.prompt should carry the exact prompt text that was
    sent - not just a truncated summary, and not just the response - so
    the audit log can reconstruct exactly what Gemma was asked."""
    prompt = "A CRITICAL conjunction was detected between SAT-A and SAT-B."
    text, provenance = _call_gemma_with_provenance(
        FakeGemmaClient(), prompt=prompt, system="test system prompt", fallback_text="fallback",
    )

    assert provenance.prompt == prompt
    assert provenance.source == "gemma"


def test_call_gemma_with_provenance_records_the_real_prompt_even_on_fallback():
    """The prompt was still genuinely sent (and Gemma still failed to
    answer it) even when the deterministic fallback text is what actually
    gets used - the audit trail shouldn't lose the fact that this
    specific question was asked just because nothing answered it."""
    prompt = "A CRITICAL conjunction was detected between SAT-A and SAT-B."
    text, provenance = _call_gemma_with_provenance(
        FailingGemmaClient(), prompt=prompt, system="test system prompt", fallback_text="fallback",
    )

    assert provenance.prompt == prompt
    assert provenance.source == "fallback"
    assert text == "fallback"


def test_client_shared_across_separate_run_once_calls_lets_circuit_breaker_accumulate(tmp_path):
    """Regression test for a QA-found gap: scripts/dashboard.py and
    scripts/run_demo.py originally let every run_once() call default to
    client=None, so build_pipeline() constructed a brand-new GemmaClient
    per call - the circuit breaker (see GemmaClient) could only ever
    accumulate consecutive failures WITHIN one run_once() call's own
    event batch, never across separate calls. Both scripts now hold and
    pass one shared client explicitly - this proves that, once shared,
    consecutive-failure tracking genuinely spans separate run_once()
    invocations, not just events within a single one."""
    settings = Settings(
        gemma_backend="ollama", gemma_model="fake-model", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="fake-model-api", log_dir=str(tmp_path), delta_v_budget_m_s=5.0,
    )
    client = GemmaClient(settings=settings)
    logger = DecisionLogger(settings=settings)

    with patch.object(client, "_generate_ollama", side_effect=GemmaClientError("down")) as mock_ollama:
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            run_once(adapter=DummyAdapter(), client=client, logger=logger, limit=1)
        calls_before = mock_ollama.call_count

        run_once(adapter=DummyAdapter(), client=client, logger=logger, limit=1)

    # The 4th run_once() call's own generate() attempt should have been
    # skipped entirely - the circuit opened after the 3rd call, and that
    # state persisted because the SAME client was reused across all 4
    # separate run_once() invocations.
    assert mock_ollama.call_count == calls_before


def test_pipeline_produces_finding_and_decision_for_every_event(tmp_path):
    settings = Settings(
        gemma_backend="ollama",
        gemma_model="gemma4:e4b",
        ollama_host="http://localhost:11434",
        gemma_api_key="",
        gemma_model_api="gemma-4-26b-a4b-it",
        log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0,
    )
    logger = DecisionLogger(settings=settings)
    adapter = DummyAdapter()

    entries = run_once(adapter=adapter, client=FakeGemmaClient(), logger=logger, limit=3)

    assert len(entries) == 3
    for entry in entries:
        assert entry.finding is not None
        assert entry.finding.severity in Severity
        assert entry.decision is not None
        assert entry.decision.action in Action
        assert entry.finding.event_id == entry.telemetry.event_id
        # DummyAdapter's payload isn't conjunction-shaped, so analyze_node
        # never attempts a Gemma call for the description.
        assert entry.description_provenance is None
        # decide_node always attempts a rationale call.
        assert entry.rationale_provenance.source == "gemma"
        assert entry.rationale_provenance.model_used == "fake-model"
        assert entry.rationale_provenance.latency_ms >= 0

    log_files = list(tmp_path.glob("decisions-*.jsonl"))
    assert len(log_files) == 1
    lines = log_files[0].read_text().strip().splitlines()
    assert len(lines) == 3


def _make_conjunction_event(min_distance_km: float) -> TelemetryEvent:
    """Builds a TelemetryEvent shaped like CelesTrakAdapter's output."""
    raw_data = {
        "object_a_id": "33779",
        "object_a_name": "COSMOS 2251 DEB",
        "object_b_id": "33825",
        "object_b_name": "COSMOS 2251 DEB",
        "min_distance_km": min_distance_km,
        "time_of_closest_approach": "2026-08-02T17:17:40.706542+00:00",
        "relative_velocity_km_s": 13.788,
    }
    return TelemetryEvent(
        event_id="conj-33779-33825",
        timestamp=datetime.now(timezone.utc),
        source="celestrak",
        raw_data=raw_data,
    )


@pytest.mark.parametrize(
    "min_distance_km,expected_severity",
    [
        (4.9, Severity.CRITICAL),
        (5.1, Severity.WARNING),
        (24.9, Severity.WARNING),
        (25.1, Severity.WATCH),
        (99.9, Severity.WATCH),
        (100.1, Severity.NOMINAL),
    ],
)
def test_analyze_node_classifies_conjunction_severity_by_distance(
    min_distance_km, expected_severity,
):
    event = _make_conjunction_event(min_distance_km)
    analyze_node = make_analyze_node(FakeGemmaClient())

    result_state = analyze_node({
        "telemetry": event, "finding": None, "decision": None, "log_path": None,
    })

    finding = result_state["finding"]
    assert finding.severity == expected_severity
    assert finding.confidence == 0.8
    assert finding.description == "Stubbed anomaly commentary: nothing to report."
    assert finding.severity_source == "distance-threshold"


def test_classify_conjunction_severity_respects_custom_thresholds():
    """Real behavior this guards: a real operator's configured
    conjunction_critical_km/warning_km/watch_km (Settings, see
    src/config.py) must actually change where the CRITICAL/WARNING/WATCH
    boundaries fall - not just exist as unused parameters. Uses
    thresholds far from the 5/25/100km defaults so a bug that silently
    ignored the arguments (falling back to the defaults) would fail this,
    not just coincidentally pass."""
    assert classify_conjunction_severity(1.5, critical_km=2.0, warning_km=50.0, watch_km=200.0) == Severity.CRITICAL
    # 10km is WARNING under defaults (>5, <25) but would be NOMINAL/CRITICAL
    # under a much stricter custom critical_km - confirms the real cutoff moved.
    assert classify_conjunction_severity(10.0, critical_km=2.0, warning_km=5.0, watch_km=50.0) == Severity.WATCH
    assert classify_conjunction_severity(300.0, critical_km=2.0, warning_km=50.0, watch_km=200.0) == Severity.NOMINAL


def test_analyze_node_uses_configured_client_settings_thresholds_not_hardcoded_defaults():
    """End-to-end guard: analyze_node must actually read client.settings'
    real configured thresholds (as a real Settings object would carry
    them) rather than the classify_*'s own hardcoded defaults. 10km is
    WARNING under the project's original 5/25/100km defaults - this test
    configures a much stricter conjunction_critical_km=15.0 and confirms
    the SAME 10km event now classifies CRITICAL, proving the configured
    value is what actually drove the decision, not a coincidence."""
    event = _make_conjunction_event(10.0)
    client = FakeGemmaClient()
    client.settings.conjunction_critical_km = 15.0
    client.settings.conjunction_warning_km = 25.0
    client.settings.conjunction_watch_km = 100.0
    analyze_node = make_analyze_node(client)

    result_state = analyze_node({
        "telemetry": event, "finding": None, "decision": None, "log_path": None,
    })

    assert result_state["finding"].severity == Severity.CRITICAL


def test_analyze_node_uses_real_pc_when_a_cdm_was_matched():
    """A real Space-Track CDM match (src/ingestion/cdm_enrichment.py)
    merges collision_probability into raw_data - analyze_node must prefer
    it over the distance threshold, and record which path was used."""
    event = _make_conjunction_event(min_distance_km=250.0)  # would be NOMINAL by distance alone
    event.raw_data["collision_probability"] = 2e-3  # well above the CRITICAL Pc threshold
    analyze_node = make_analyze_node(FakeGemmaClient())

    result_state = analyze_node({
        "telemetry": event, "finding": None, "decision": None, "log_path": None,
    })

    finding = result_state["finding"]
    assert finding.severity == Severity.CRITICAL
    assert finding.severity_source == "probability-of-collision"


def test_analyze_node_severity_source_is_none_for_non_conjunction_hazards():
    decay_event = TelemetryEvent(
        event_id="decay-1", timestamp=datetime.now(timezone.utc), source="celestrak-decay",
        raw_data={
            "object_id": "1", "object_name": "TEST OBJ",
            "perigee_altitude_km": 250.0, "apogee_altitude_km": 400.0, "bstar": 0.0001,
        },
    )
    analyze_node = make_analyze_node(FakeGemmaClient())

    result_state = analyze_node({
        "telemetry": decay_event, "finding": None, "decision": None, "log_path": None,
    })

    assert result_state["finding"].severity_source is None

    provenance = result_state["description_provenance"]
    assert provenance.source == "gemma"
    assert provenance.model_used == "fake-model"
    assert provenance.latency_ms >= 0


@pytest.mark.parametrize(
    "tle_epoch_age_hours,expected_confidence",
    [
        (12.0, 0.9),    # fresh (<= 24h)
        (48.0, 0.8),    # <= 72h
        (100.0, 0.6),   # <= 168h (1 week)
        (300.0, 0.4),   # stale
    ],
)
def test_analyze_node_derives_confidence_from_tle_epoch_age(
    tle_epoch_age_hours, expected_confidence,
):
    event = _make_conjunction_event(10.0)
    event.raw_data["tle_epoch_age_hours"] = tle_epoch_age_hours
    analyze_node = make_analyze_node(FakeGemmaClient())

    result_state = analyze_node({
        "telemetry": event, "finding": None, "decision": None, "log_path": None,
    })

    assert result_state["finding"].confidence == expected_confidence


def _make_decay_event(perigee_altitude_km: float) -> TelemetryEvent:
    """Builds a TelemetryEvent shaped like DecayRiskAdapter's output -
    single-object (object_id/object_name), not a conjunction pair."""
    raw_data = {
        "object_id": "33821",
        "object_name": "COSMOS 2251 DEB",
        "perigee_altitude_km": perigee_altitude_km,
        "apogee_altitude_km": perigee_altitude_km + 50.0,
        "bstar": 0.0008,
        "tle_epoch_age_hours": 12.0,
    }
    return TelemetryEvent(
        event_id="decay-33821",
        timestamp=datetime.now(timezone.utc),
        source="celestrak-decay",
        raw_data=raw_data,
    )


@pytest.mark.parametrize(
    "perigee_altitude_km,expected_severity",
    [
        (199.9, Severity.CRITICAL),
        (200.1, Severity.WARNING),
        (299.9, Severity.WARNING),
        (300.1, Severity.WATCH),
        (499.9, Severity.WATCH),
        (500.1, Severity.NOMINAL),
    ],
)
def test_classify_decay_severity_by_perigee_altitude(perigee_altitude_km, expected_severity):
    assert classify_decay_severity(perigee_altitude_km) == expected_severity


def test_classify_decay_severity_respects_custom_thresholds():
    """Same guard as test_classify_conjunction_severity_respects_custom_thresholds
    - a real operator's decay_*_perigee_km settings must actually move the
    boundary, not just be accepted and ignored."""
    assert classify_decay_severity(150.0, critical_km=100.0, warning_km=180.0, watch_km=250.0) == Severity.WARNING
    assert classify_decay_severity(150.0, critical_km=160.0, warning_km=180.0, watch_km=250.0) == Severity.CRITICAL
    assert classify_decay_severity(600.0, critical_km=100.0, warning_km=180.0, watch_km=250.0) == Severity.NOMINAL


def test_analyze_node_classifies_decay_risk_and_uses_decay_description():
    event = _make_decay_event(150.0)  # CRITICAL range
    analyze_node = make_analyze_node(FakeGemmaClient())

    result_state = analyze_node({
        "telemetry": event, "finding": None, "decision": None, "log_path": None,
    })

    finding = result_state["finding"]
    assert finding.severity == Severity.CRITICAL
    # tle_epoch_age_hours=12.0 in the fixture -> 0.9 via the same
    # compute_confidence signal conjunctions use, not the 0.5 fallback -
    # confirms the decay branch reuses it rather than falling through to
    # the generic placeholder path.
    assert finding.confidence == 0.9
    assert finding.description == "Stubbed anomaly commentary: nothing to report."


def test_decide_node_critical_decay_gets_no_maneuver_machinery():
    """The maneuver/budget/veto/approval machinery is conjunction-specific
    scope (see decide_node's guard) - a CRITICAL decay finding should still
    get a real deterministic action and real Gemma narration, just no
    maneuver_plan, and critically: no KeyError from trying to read
    object_a_id/min_distance_km off decay-shaped raw_data."""
    event = _make_decay_event(150.0)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(FakeGemmaClient())

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.action == Action.ABORT
    assert decision.maneuver_plan is None
    assert decision.verified_clearance is None
    assert decision.maneuver_approval is None
    assert decision.budget_insufficient is False
    assert decision.awaiting_human_approval is False


def test_decide_node_pc_critical_at_large_distance_does_not_crash():
    """Real bug this closes: a Pc-based CRITICAL conjunction (tight
    covariance, not proximity - see pc_severity.py) can have a
    min_distance_km far beyond compute_avoidance_maneuver's own target
    clearance. Before the fix, this made ManeuverPlan's
    magnitude_delta_v > 0 validation raise a pydantic.ValidationError
    straight out of decide_node - this is the exact event shape
    test_analyze_node_uses_real_pc_when_a_cdm_was_matched already proves
    analyze_node produces (250km, Pc=2e-3), just carried one step further
    through decide_node, which is where the crash actually happened."""
    event = _make_conjunction_event(min_distance_km=250.0)
    event.raw_data["collision_probability"] = 2e-3
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(FakeGemmaClient())

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.action == Action.ABORT
    assert decision.maneuver_plan is None
    assert decision.verified_clearance is None
    assert decision.maneuver_approval is None
    assert decision.budget_insufficient is False
    assert decision.awaiting_human_approval is False


def _make_attitude_event(pointing_error_deg: float) -> TelemetryEvent:
    """Builds a TelemetryEvent shaped like SyntheticAttitudeAdapter's
    output - single-object (object_id/object_name), no perigee_altitude_km
    (would be misread as the decay hazard) and no object_a_id (would be
    misread as a conjunction)."""
    raw_data = {
        "object_id": "99010",
        "object_name": "SYNTH-SAT-TEST",
        "pointing_error_deg": pointing_error_deg,
        "angular_rate_deg_s": 1.5,
        "solar_panel_power_pct": 60.0,
    }
    return TelemetryEvent(
        event_id="attitude-99010",
        timestamp=datetime.now(timezone.utc),
        source="synthetic-attitude-fixture",
        raw_data=raw_data,
    )


@pytest.mark.parametrize(
    "pointing_error_deg,expected_severity",
    [
        (4.9, Severity.NOMINAL),
        (5.1, Severity.WATCH),
        (14.9, Severity.WATCH),
        (15.1, Severity.WARNING),
        (44.9, Severity.WARNING),
        (45.1, Severity.CRITICAL),
    ],
)
def test_classify_attitude_severity_by_pointing_error(pointing_error_deg, expected_severity):
    assert classify_attitude_severity(pointing_error_deg) == expected_severity


def test_classify_attitude_severity_respects_custom_thresholds():
    """Same guard as the conjunction/decay threshold tests above - a real
    operator's attitude_*_deg settings must actually move the boundary."""
    assert classify_attitude_severity(8.0, critical_deg=30.0, warning_deg=20.0, watch_deg=10.0) == Severity.NOMINAL
    assert classify_attitude_severity(8.0, critical_deg=30.0, warning_deg=20.0, watch_deg=5.0) == Severity.WATCH
    assert classify_attitude_severity(25.0, critical_deg=30.0, warning_deg=20.0, watch_deg=5.0) == Severity.WARNING
    assert classify_attitude_severity(35.0, critical_deg=30.0, warning_deg=20.0, watch_deg=5.0) == Severity.CRITICAL


def test_analyze_node_classifies_attitude_and_uses_attitude_description():
    event = _make_attitude_event(70.0)  # CRITICAL range
    analyze_node = make_analyze_node(FakeGemmaClient())

    result_state = analyze_node({
        "telemetry": event, "finding": None, "decision": None, "log_path": None,
    })

    finding = result_state["finding"]
    assert finding.severity == Severity.CRITICAL
    # No tle_epoch_age_hours in attitude's raw_data (it's not TLE-derived) -
    # falls to compute_confidence's clearly-labeled placeholder, same as
    # any other non-epoch-bearing shape, not a new special case.
    assert finding.confidence == 0.8
    assert finding.description == "Stubbed anomaly commentary: nothing to report."


def test_decide_node_critical_attitude_gets_no_maneuver_machinery():
    """Same reasoning as test_decide_node_critical_decay_gets_no_maneuver_machinery:
    the maneuver/budget/veto/approval machinery is conjunction-specific
    scope - a CRITICAL attitude finding should still get a real
    deterministic action and real Gemma narration, just no maneuver_plan,
    and critically: no KeyError from trying to read
    object_a_id/min_distance_km off attitude-shaped raw_data."""
    event = _make_attitude_event(70.0)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(FakeGemmaClient())

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.action == Action.ABORT
    assert decision.maneuver_plan is None
    assert decision.verified_clearance is None
    assert decision.maneuver_approval is None
    assert decision.budget_insufficient is False
    assert decision.awaiting_human_approval is False
    assert result_state["veto_provenance"] is None  # no veto check for non-conjunction data


def _make_finding(severity: Severity, event: TelemetryEvent) -> AnomalyFinding:
    return AnomalyFinding(
        event_id=event.event_id,
        severity=severity,
        description="Test finding.",
        confidence=0.8,
    )


@pytest.mark.parametrize(
    "severity,expected_action",
    [
        (Severity.NOMINAL, Action.CONTINUE),
        (Severity.WATCH, Action.CONTINUE),
        (Severity.WARNING, Action.HOLD),
        (Severity.CRITICAL, Action.ABORT),
    ],
)
def test_decide_node_maps_severity_to_action(severity, expected_action):
    event = _make_conjunction_event(10.0)
    finding = _make_finding(severity, event)
    decide_node = make_decide_node(FakeGemmaClient())

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.action == expected_action
    assert isinstance(decision.rationale, str) and len(decision.rationale) > 0

    provenance = result_state["rationale_provenance"]
    assert provenance.source == "gemma"
    assert provenance.model_used == "fake-model"
    assert provenance.latency_ms >= 0


def test_analyze_then_decide_populates_maneuver_for_critical():
    """Full analyze_node -> decide_node path for a synthetic CRITICAL-range
    conjunction: confirms the maneuver plan and its verification actually
    get computed and attached, not just that the action is right."""
    event = _make_conjunction_event(2.5)  # well within CRITICAL (<5km)
    analyze_node = make_analyze_node(FakeGemmaClient())
    decide_node = make_decide_node(FakeGemmaClient())

    analyzed_state = analyze_node({
        "telemetry": event, "finding": None, "decision": None, "log_path": None,
    })
    assert analyzed_state["finding"].severity == Severity.CRITICAL

    result_state = decide_node(analyzed_state)
    decision = result_state["decision"]

    # CRITICAL already maps to Action.ABORT (see ACTION_BY_SEVERITY) -
    # confirming it's unchanged, not inventing a new action type.
    assert decision.action == Action.ABORT

    assert isinstance(decision.maneuver_plan, ManeuverPlan)
    assert decision.maneuver_plan.direction == "radial-outward"
    assert decision.maneuver_plan.magnitude_delta_v > 0

    assert isinstance(decision.verified_clearance, VerifiedClearance)
    assert decision.verified_clearance.cleared is True
    assert decision.verified_clearance.new_min_distance_km > 5.0  # past CRITICAL_THRESHOLD_KM

    # Local backend -> autonomous self-approval, no human in the loop.
    assert decision.awaiting_human_approval is False
    assert decision.maneuver_approval is not None
    assert decision.maneuver_approval.mode == "autonomous"
    assert decision.maneuver_approval.approved is True
    assert decision.maneuver_approval.approved_by is None

    # Phase 9: Gemma's own veto check (a real GO verdict, not hardcoded)
    # ran and affirmed the already-verified-safe maneuver.
    assert "affirmed by Gemma" in decision.maneuver_approval.reason
    veto_provenance = result_state["veto_provenance"]
    assert veto_provenance is not None
    assert veto_provenance.source == "gemma"
    assert veto_provenance.model_used == "fake-model"


def test_decide_node_requires_human_approval_for_api_backend():
    """Same CRITICAL scenario as above, but with the cloud ('api') backend
    configured - the maneuver should be calculated and affordable, but NOT
    auto-executed. It should wait for a human (see DecisionLogger.approve_maneuver)."""
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(FakeGemmaClient(gemma_backend="api"))

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.action == Action.ABORT
    assert decision.maneuver_plan is not None  # visible - just not executed yet
    assert decision.verified_clearance is None  # nothing executed yet
    assert decision.budget_insufficient is False
    assert decision.awaiting_human_approval is True
    assert decision.maneuver_approval is None  # not yet resolved
    # Gemma veto-checking only applies to the autonomous (local) path -
    # the cloud path relies on the human instead, so no veto check runs.
    assert result_state["veto_provenance"] is None
    # FakeGemmaClient ignores the prompt and always returns fixed stub text,
    # so rationale content itself isn't checked here - see the real,
    # non-mocked verification of the actual prompt/rationale content in the
    # QA pass and scripts/run_demo.py.


def test_decide_node_budget_insufficient_takes_priority_over_approval_mode():
    """Regardless of backend, an unaffordable maneuver should never reach
    the approval-gating branch at all - budget is checked first."""
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    tiny_tracker = DeltaVBudgetTracker(starting_budget_m_s=0.0001)
    decide_node = make_decide_node(FakeGemmaClient(gemma_backend="api"), budget_tracker=tiny_tracker)

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.budget_insufficient is True
    assert decision.awaiting_human_approval is False
    assert decision.maneuver_approval is None
    assert decision.verified_clearance is None
    # Budget is checked before any veto check runs.
    assert result_state["veto_provenance"] is None


def test_decide_node_leaves_maneuver_fields_none_for_non_critical():
    event = _make_conjunction_event(50.0)  # WATCH range
    finding = _make_finding(Severity.WATCH, event)
    decide_node = make_decide_node(FakeGemmaClient())

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.maneuver_plan is None
    assert decision.verified_clearance is None


def test_decide_node_executes_maneuver_when_budget_sufficient():
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    tracker = DeltaVBudgetTracker(starting_budget_m_s=5.0)  # plenty
    decide_node = make_decide_node(FakeGemmaClient(), budget_tracker=tracker)

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.budget_insufficient is False
    assert decision.maneuver_plan is not None
    assert isinstance(decision.verified_clearance, VerifiedClearance)
    assert tracker.remaining_m_s < 5.0  # budget was actually spent


def test_decide_node_flags_budget_insufficient_without_executing():
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    tracker = DeltaVBudgetTracker(starting_budget_m_s=0.0001)  # effectively none left
    decide_node = make_decide_node(FakeGemmaClient(), budget_tracker=tracker)

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.budget_insufficient is True
    # Plan is still visible (mission control can see what would've been
    # needed) even though nothing was actually applied.
    assert decision.maneuver_plan is not None
    # Nothing to verify - the maneuver was never executed.
    assert decision.verified_clearance is None
    assert tracker.remaining_m_s == pytest.approx(0.0001)  # untouched


def test_decide_node_falls_back_when_gemma_fails():
    """FailingGemmaClient always raises, so this also proves Phase 9's
    fail-safe: an unreachable Gemma is NOT treated as a veto - the
    already-verified-safe maneuver still executes autonomously even though
    the veto-check call itself failed."""
    event = _make_conjunction_event(3.0)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(FailingGemmaClient())

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.action == Action.ABORT
    assert "fallback" in decision.rationale.lower()

    # Fail-safe: Gemma unreachable during the veto check still results in
    # the maneuver executing (physics-only autonomy), not a veto.
    assert decision.maneuver_approval is not None
    assert decision.maneuver_approval.approved is True
    assert isinstance(decision.verified_clearance, VerifiedClearance)
    veto_provenance = result_state["veto_provenance"]
    assert veto_provenance is not None
    assert veto_provenance.source == "fallback"

    provenance = result_state["rationale_provenance"]
    assert provenance.source == "fallback"
    assert provenance.model_used == "fake-model"
    assert provenance.latency_ms >= 0


def test_fallback_provenance_reports_hosted_model_when_configured_backend_is_api():
    """Regression test: the fallback path used to unconditionally report
    gemma_model (the Ollama tag) regardless of which backend was actually
    configured - so a cloud-only deployment's fallback entries claimed
    "gemma4:e4b" responded, when nothing running that tag was ever
    involved. Should report gemma_model_api instead when GEMMA_BACKEND=api."""
    event = _make_conjunction_event(50.0)  # WATCH - no CRITICAL/veto complexity needed
    finding = _make_finding(Severity.WATCH, event)
    decide_node = make_decide_node(FailingGemmaClient(gemma_backend="api", gemma_model_api="hosted-model-x"))

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    provenance = result_state["rationale_provenance"]
    assert provenance.source == "fallback"
    assert provenance.model_used == "hosted-model-x"


def test_decide_node_gemma_veto_blocks_autonomous_maneuver():
    """Phase 9: on the autonomous (local) path, Gemma gets a real GO/NO-GO
    veto over an already-verified-safe maneuver, standing in for the
    unavailable human. An explicit NO-GO blocks execution entirely, even
    though the deterministic physics check already verified it safe."""
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(
        FakeGemmaClient(veto_response="NO-GO - the numbers look inconsistent.")
    )

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.maneuver_plan is not None  # still visible - just not executed
    assert decision.verified_clearance is None  # NOT executed - Gemma vetoed it
    assert decision.awaiting_human_approval is False  # autonomous path, not cloud
    assert decision.maneuver_approval is not None
    assert decision.maneuver_approval.mode == "autonomous"
    assert decision.maneuver_approval.approved is False
    assert decision.maneuver_approval.approved_by is not None
    assert "gemma" in decision.maneuver_approval.approved_by.lower()
    assert "NO-GO" in decision.maneuver_approval.reason
    # FakeGemmaClient ignores the prompt and always returns fixed stub text
    # for narration, so decision.rationale content isn't checked here - see
    # the real, non-mocked prompt/rationale verification in scripts/run_demo.py.

    veto_provenance = result_state["veto_provenance"]
    assert veto_provenance is not None
    assert veto_provenance.source == "gemma"


def test_decide_node_gemma_veto_blocks_autonomous_maneuver_from_structured_json():
    """Same as test_decide_node_gemma_veto_blocks_autonomous_maneuver, but
    with the real structured JSON response Ollama's schema-constrained
    generation produces (see _VETO_JSON_SCHEMA), not free text - proves
    _maneuver_veto_check prefers the structured path when it's available,
    and that only the "reason" field (not the raw JSON blob) ends up in
    maneuver_approval.reason."""
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(
        FakeGemmaClient(veto_response='{"verdict": "NO-GO", "reason": "Inconsistent numbers."}')
    )

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.verified_clearance is None  # NOT executed - Gemma vetoed it
    assert decision.maneuver_approval.approved is False
    assert "Inconsistent numbers." in decision.maneuver_approval.reason
    assert '{"verdict"' not in decision.maneuver_approval.reason  # the parsed reason, not the raw JSON


def test_decide_node_gemma_veto_affirms_autonomous_maneuver_from_structured_json():
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(
        FakeGemmaClient(veto_response='{"verdict": "GO", "reason": "Clearance margin is safe."}')
    )

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.verified_clearance is not None  # executed
    assert decision.maneuver_approval.approved is True
    assert "Clearance margin is safe." in decision.maneuver_approval.reason


def test_decide_node_gemma_veto_defaults_to_no_go_when_unparseable():
    """Fail-safe: if Gemma responds but its answer has no clear GO/NO-GO
    token, that's treated as a veto (the conservative default), not a free
    pass - distinct from Gemma being unreachable, which IS a free pass
    (see test_decide_node_falls_back_when_gemma_fails)."""
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(FakeGemmaClient(veto_response="Sounds fine to me."))

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.verified_clearance is None
    assert decision.maneuver_approval.approved is False
    assert "parseable" in decision.maneuver_approval.reason.lower()


def _log_state(event, finding):
    decision = Decision(action="abort", rationale="Test rationale.", made_at=datetime.now(timezone.utc))
    return {
        "telemetry": event, "finding": finding, "decision": decision,
        "rationale_provenance": GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
        "description_provenance": None, "veto_provenance": None, "log_path": None,
    }


@patch("src.pipeline.send_critical_alert")
def test_log_node_triggers_alert_for_critical_finding(mock_alert, tmp_path):
    settings = Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0,
    )
    logger = DecisionLogger(settings=settings)
    event = _make_conjunction_event(2.5)
    finding = _make_finding(Severity.CRITICAL, event)
    log_node = make_log_node(logger)

    log_node(_log_state(event, finding))

    mock_alert.assert_called_once()
    logged_entry = mock_alert.call_args.args[0]
    assert logged_entry.finding.severity == Severity.CRITICAL
    assert mock_alert.call_args.args[1] is settings  # logger's own settings, not a new one


def test_log_node_delegates_the_critical_check_to_send_critical_alert_itself(tmp_path):
    """log_node calls send_critical_alert unconditionally for every entry,
    not just CRITICAL ones - the severity gate lives entirely inside
    send_critical_alert (see test_alerting.py), a single source of truth
    for "does this fire" rather than duplicated logic in two places. Uses
    a real (unconfigured) webhook_url, so this also proves a WATCH entry
    through the real log_node -> send_critical_alert path makes no real
    network call and doesn't raise."""
    settings = Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0, alert_webhook_url="",
    )
    logger = DecisionLogger(settings=settings)
    event = _make_conjunction_event(50.0)
    finding = _make_finding(Severity.WATCH, event)
    log_node = make_log_node(logger)

    result_state = log_node(_log_state(event, finding))

    assert result_state["log_path"] is not None


@patch("src.alerting.requests.post")
def test_log_node_suppresses_repeat_alert_for_the_same_still_critical_hazard(mock_post, tmp_path):
    """Real end-to-end guard for the alert-fatigue gap found by a live PM/
    customer review: scripts/scheduler.py's continuous operation
    re-detects the SAME still-unresolved conjunction fresh on every tick
    (a new event_id each time - see CelesTrakAdapter). Without
    log_node actually threading the real logged history into
    send_critical_alert's cooldown check, the same hazard would page an
    operator every tick, indefinitely. Uses a real JSONLDecisionLogStore
    (via tmp_path), not a mock, so this proves the real load_all_entries()
    -> hazard_key() -> cooldown path actually works end to end."""
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    settings = Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0, alert_webhook_url="https://example.com/webhook",
    )
    logger = DecisionLogger(settings=settings)
    log_node = make_log_node(logger)

    # First tick: a fresh CRITICAL detection for object pair 1/2 - the
    # first time this hazard has ever been seen, so it must alert.
    first_event = _make_conjunction_event(2.5)
    first_finding = _make_finding(Severity.CRITICAL, first_event)
    log_node(_log_state(first_event, first_finding))
    assert mock_post.call_count == 1

    # Second tick: the SAME hazard (same object pair), re-detected with a
    # brand-new event_id - a real scheduler tick shape. Must be suppressed.
    second_event = _make_conjunction_event(2.4)
    second_event.event_id = "conj-33779-33825-different-run"
    second_finding = _make_finding(Severity.CRITICAL, second_event)
    log_node(_log_state(second_event, second_finding))

    assert mock_post.call_count == 1  # NOT called again
