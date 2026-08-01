"""Smoke test: run the full pipeline against DummyAdapter with a stubbed
GemmaClient, no live network required.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.gemma_client import GemmaClientError
from src.ingestion.base_adapter import DummyAdapter
from src.logging_utils import DecisionLogger
from src.maneuver import DeltaVBudgetTracker
from src.pipeline import _extract_final_answer, make_analyze_node, make_decide_node, run_once
from src.schemas import Action, AnomalyFinding, ManeuverPlan, Severity, TelemetryEvent, VerifiedClearance


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


class FakeGemmaClient:
    """Duck-types GemmaClient (generate() + settings.gemma_model/
    gemma_backend) without making any network calls. gemma_backend defaults
    to "ollama" so existing CRITICAL-severity tests (written before the
    human-approval feature existed) keep exercising the autonomous path -
    see test_decide_node_requires_human_approval_for_api_backend for the
    "api" case specifically."""

    def __init__(self, gemma_backend: str = "ollama"):
        self.settings = SimpleNamespace(gemma_model="fake-model", gemma_backend=gemma_backend)

    def generate(self, prompt: str, system=None, timeout: int = 60) -> str:
        return "Stubbed anomaly commentary: nothing to report."


class FailingGemmaClient:
    """Duck-types GemmaClient but always raises, to test fallback paths."""

    def __init__(self):
        self.settings = SimpleNamespace(gemma_model="fake-model", gemma_backend="ollama")

    def generate(self, prompt: str, system=None, timeout: int = 60) -> str:
        raise GemmaClientError("simulated failure")


def test_pipeline_produces_finding_and_decision_for_every_event(tmp_path):
    settings = Settings(
        gemma_backend="ollama",
        gemma_model="gemma3:4b",
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
    event = _make_conjunction_event(3.0)
    finding = _make_finding(Severity.CRITICAL, event)
    decide_node = make_decide_node(FailingGemmaClient())

    result_state = decide_node({
        "telemetry": event, "finding": finding, "decision": None, "log_path": None,
    })

    decision = result_state["decision"]
    assert decision.action == Action.ABORT
    assert "fallback" in decision.rationale.lower()

    provenance = result_state["rationale_provenance"]
    assert provenance.source == "fallback"
    assert provenance.model_used == "fake-model"
    assert provenance.latency_ms >= 0
