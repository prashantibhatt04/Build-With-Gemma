"""Smoke/interaction tests for scripts/dashboard.py using Streamlit's
AppTest harness - runs the real script (no mocking of Streamlit itself),
but doesn't assume specific log content (the real log_dir this process
loads may already have real data from other runs) - correctness of the
underlying aggregation/transform logic is covered separately and
thoroughly in tests/test_dashboard_data.py with controlled fixtures. This
file only confirms the UI wiring itself doesn't crash and exposes the
controls it's supposed to.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import settings
from src.logging_utils import DecisionLogger
from src.maneuver import compute_avoidance_maneuver
from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "scripts" / "dashboard.py")

# scripts/dashboard.py no longer uses st.tabs() (see _SECTION_NAV_CSS's
# docstring there for why - a real glide-data-grid zero-width-container
# bug that only a session-state-tracked section, never st.tabs(), can
# fix). The active section is plain st.session_state under this key, so
# any test that asserts on a non-default section's content must set it
# BEFORE the first .run(), the same way every other ambient-config test
# in this file (_operator_tokens, _log_dir, ...) pre-seeds state before
# the script ever executes.
_ACTIVE_SECTION_KEY = "active_section"


def _open_section(at: AppTest, label: str) -> AppTest:
    at.session_state[_ACTIVE_SECTION_KEY] = label
    return at


@contextmanager
def _operator_tokens(tokens: dict[str, str]):
    """`settings` is a module-level singleton, already imported (and
    frozen) long before any individual test runs, so setting the
    OPERATOR_TOKENS env var alone wouldn't reach it - the real fix is
    mutating the one shared instance in place (every module that did
    `from src.config import settings` holds a reference to this same
    object), via object.__setattr__ since Settings is a frozen dataclass.
    Restored afterward so this test can't leak configuration into any
    other test in the same session."""
    original = settings.operator_tokens
    object.__setattr__(settings, "operator_tokens", tokens)
    try:
        yield
    finally:
        object.__setattr__(settings, "operator_tokens", original)


@contextmanager
def _spacetrack_credentials(username: str, password: str):
    """Same object.__setattr__-on-the-shared-singleton pattern as
    _operator_tokens above, and for the same reason - this test's own
    real .env may or may not already have real Space-Track credentials
    configured, so the "unconfigured" test path must force the singleton
    to a known empty state rather than assume the ambient environment."""
    original = (settings.spacetrack_username, settings.spacetrack_password)
    object.__setattr__(settings, "spacetrack_username", username)
    object.__setattr__(settings, "spacetrack_password", password)
    try:
        yield
    finally:
        object.__setattr__(settings, "spacetrack_username", original[0])
        object.__setattr__(settings, "spacetrack_password", original[1])


def test_dashboard_loads_without_exception():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)

    assert not at.exception


def test_dashboard_shows_title_and_metrics():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)

    assert any("Mission Ops Dashboard" in t.value for t in at.title)
    metric_labels = {m.label for m in at.metric}
    assert {
        "Total events", "CRITICAL", "Autonomous executed", "Human-approved executed",
        "Vetoed by Gemma", "Rejected by human", "Awaiting approval", "Blocked by budget",
    } <= metric_labels


def test_dashboard_sidebar_has_expected_controls():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)

    button_labels = {b.label for b in at.sidebar.button}
    assert {"Fetch live CelesTrak conjunctions", "Run synthetic CRITICAL scenario", "Refresh"} <= button_labels
    assert any("Operator name" in ti.label for ti in at.sidebar.text_input)


def test_dashboard_all_decisions_has_severity_and_source_filters(tmp_path):
    """Parity check with scripts/api.py's GET /decisions, which already
    supports severity/source filters - the dashboard's own "All
    decisions" table (the real risk board, per its own caption) had no
    equivalent, and grows unbounded under continuous scheduled operation
    (scripts/scheduler.py).

    Real bug this regression-tests: without an isolated log_dir, this
    test previously depended on whatever real entries happened to
    already be sitting in the ambient default log directory - present on
    a dev machine that had run the app many times, but empty on a fresh
    CI checkout, where `_render_all_decisions_table` takes its "no
    entries yet" early-return branch and never renders the filters at
    all. Seeding one real entry via _log_dir(tmp_path) (the same pattern
    every other data-dependent test in this file already uses) makes the
    filters' presence deterministic instead of environment-dependent."""
    with _log_dir(tmp_path):
        logger = DecisionLogger(settings=settings)
        logger.log(_widget_test_entry("e1", Severity.WATCH))

        at = AppTest.from_file(DASHBOARD_PATH)
        _open_section(at, "Full Log").run(timeout=30)

        multiselect_labels = {m.label for m in at.multiselect}
        assert {"Filter by severity", "Filter by source"} <= multiselect_labels
        assert any("Showing" in c.value and "of" in c.value for c in at.caption)


def test_dashboard_severity_filter_narrows_the_all_decisions_table():
    at = AppTest.from_file(DASHBOARD_PATH)
    _open_section(at, "Full Log").run(timeout=30)

    # "Total events" (an Overview-only metric) isn't rendered while Full
    # Log is the active section - the filters' own presence is the real
    # "is the log empty" signal now (see _render_all_decisions_table's
    # early return, which skips rendering them entirely for zero entries).
    if not any(m.label == "Filter by severity" for m in at.multiselect):
        return  # nothing to narrow in an empty log - covered by the empty-state test instead
    total_before = int(next(c.value for c in at.caption if c.value.startswith("Showing")).split()[3])

    severity_filter = next(m for m in at.multiselect if m.label == "Filter by severity")
    # Narrow to a single severity actually present, so the filtered count
    # is provably <= the total rather than trivially equal to it.
    only_option = severity_filter.options[0]
    severity_filter.set_value([only_option]).run(timeout=30)

    caption_text = next(c.value for c in at.caption if c.value.startswith("Showing"))
    shown, of_total = int(caption_text.split()[1]), int(caption_text.split()[3])
    assert shown <= of_total == total_before


def test_dashboard_review_panel_event_dropdown_is_labeled_not_a_bare_event_id():
    """Real gap this closes: an operator picking an event to inspect/mark
    reviewed previously saw a bare event_id (e.g. a UUID or
    "conj-33765-33818") with no severity or subject - unusable for
    triage without already knowing which id corresponds to what."""
    at = AppTest.from_file(DASHBOARD_PATH)
    _open_section(at, "Full Log").run(timeout=30)

    # "Total events" (an Overview-only metric) isn't rendered while Full
    # Log is the active section - the "Event" dropdown's own presence is
    # the real "is the log empty" signal now (see _render_review_panel's
    # early return, which skips rendering it entirely for zero entries).
    if not any(sb.label == "Event" for sb in at.selectbox):
        return  # nothing to label - covered by the "no logged decisions" caption elsewhere

    event_dropdown = next(sb for sb in at.selectbox if sb.label == "Event")
    assert all("[" in option and "]" in option for option in event_dropdown.options)


def test_dashboard_shows_needs_attention_section():
    """Real gap this closes: CRITICAL decay/attitude findings have no
    maneuver-approval workflow of their own, so without a dedicated
    section they'd be indistinguishable from NOMINAL/WATCH noise in the
    "All decisions" table."""
    at = AppTest.from_file(DASHBOARD_PATH)
    _open_section(at, "Approvals & Attention").run(timeout=30)

    assert any("Needs attention" in h.value for h in at.subheader)


def test_dashboard_shows_unauthenticated_warning_when_operator_tokens_unset():
    with _operator_tokens({}):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

    # Rendered as a neutral system-note markdown box (ui_style.system_note_html),
    # not st.warning - a design-review pass found st.warning's native
    # yellow shared the same hue family as a real WATCH-severity badge,
    # reading as a risk finding when it's really just a config reminder.
    assert any("Unauthenticated" in m.value for m in at.markdown)
    # Content still renders in this mode - free-text operator name, same
    # behavior as before this phase.
    assert len(at.metric) > 0


def test_dashboard_blocks_content_until_a_valid_token_is_entered():
    with _operator_tokens({"alice": "secret123"}):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

        # Login gate only - no metrics, no sidebar controls yet.
        assert len(at.metric) == 0
        assert any("Sign in" == b.label for b in at.button)

        at.text_input[0].set_value("wrong-token")
        at.button[0].click().run(timeout=30)
        assert any("Invalid token" in e.value for e in at.error)
        assert len(at.metric) == 0

        at.text_input[0].set_value("secret123")
        at.button[0].click().run(timeout=30)

    assert not at.exception
    assert any("alice" in s.value for s in at.success)
    assert len(at.metric) > 0


def test_dashboard_hides_spacetrack_button_when_credentials_unconfigured():
    with _spacetrack_credentials("", ""):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

    button_labels = {b.label for b in at.sidebar.button}
    assert not any("Space-Track" in label for label in button_labels)
    assert any("SPACETRACK_USERNAME" in c.value for c in at.sidebar.caption)


@contextmanager
def _watched_norad_ids(ids: tuple[str, ...]):
    """Same object.__setattr__-on-the-shared-singleton pattern as
    _operator_tokens/_spacetrack_credentials above."""
    original = settings.watched_norad_ids
    object.__setattr__(settings, "watched_norad_ids", ids)
    try:
        yield
    finally:
        object.__setattr__(settings, "watched_norad_ids", original)


def test_dashboard_shows_demo_group_notice_when_no_watch_list_configured():
    with _watched_norad_ids(()):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

    # Neutral system-note markdown box, not st.sidebar.info - see
    # ui_style.system_note_html.
    assert any("stations" in m.value and "WATCHED_NORAD_IDS" in m.value for m in at.sidebar.markdown)


def test_dashboard_shows_real_asset_notice_when_watch_list_configured():
    """Real feature this tests: an operator who's actually configured
    their own satellite must see that reflected in the UI, not a demo
    notice that looks identical to the zero-config default - otherwise
    there's no way to tell at a glance whether the product is watching a
    real customer asset or CelesTrak's own placeholder."""
    with _watched_norad_ids(("25544", "48274")):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

    assert any("25544" in s.value and "48274" in s.value for s in at.sidebar.success)


@contextmanager
def _severity_thresholds(**overrides: float):
    """Same object.__setattr__-on-the-shared-singleton pattern as
    _operator_tokens/_spacetrack_credentials/_watched_norad_ids above."""
    originals = {field: getattr(settings, field) for field in overrides}
    for field, value in overrides.items():
        object.__setattr__(settings, field, value)
    try:
        yield
    finally:
        for field, value in originals.items():
            object.__setattr__(settings, field, value)


def test_dashboard_shows_default_severity_thresholds_by_default():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)

    labels = [e.label for e in at.sidebar.expander]
    assert any("Hazard severity thresholds" in label and "(defaults)" in label for label in labels)


def test_dashboard_shows_customized_label_and_real_values_when_thresholds_overridden():
    """Real feature this tests: an operator who configured
    CONJUNCTION_CRITICAL_KM (or any of the other 8 threshold env vars)
    must be able to see that it actually took effect from the dashboard
    itself, without reading source code - the same discoverability gap
    already closed for WATCHED_NORAD_IDS."""
    with _severity_thresholds(conjunction_critical_km=15.0, attitude_warning_deg=20.0):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

        labels = [e.label for e in at.sidebar.expander]
        assert any("Hazard severity thresholds" in label and "(customized)" in label for label in labels)

        expander = next(e for e in at.sidebar.expander if "Hazard severity thresholds" in e.label)
        text = " ".join(m.value for m in expander.markdown)
        assert "15.0km" in text
        assert "20.0°" in text


@contextmanager
def _alert_webhook_url(url: str):
    """Same object.__setattr__-on-the-shared-singleton pattern as
    _operator_tokens/_spacetrack_credentials/_watched_norad_ids above."""
    original = settings.alert_webhook_url
    object.__setattr__(settings, "alert_webhook_url", url)
    try:
        yield
    finally:
        object.__setattr__(settings, "alert_webhook_url", original)


def test_dashboard_shows_unconfigured_alert_notice_when_webhook_unset():
    with _alert_webhook_url(""):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

    # Neutral system-note markdown box, not st.sidebar.info - see
    # ui_style.system_note_html.
    assert any("not configured" in m.value and "ALERT_WEBHOOK_URL" in m.value for m in at.sidebar.markdown)
    assert not any(b.label == "Send test alert" for b in at.sidebar.button)


def test_dashboard_shows_send_test_alert_button_when_webhook_configured():
    """Real gap this closes: an operator configuring ALERT_WEBHOOK_URL
    had no way to verify it's wired correctly before a real CRITICAL
    hazard occurred. The button's mere presence is what's being checked
    here - never clicked by this automated suite (same discipline this
    file's own header comment documents for the CelesTrak/Space-Track
    fetch buttons, to avoid a real network call in tests)."""
    with _alert_webhook_url("https://example.com/webhook"):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

    assert any("configured" in s.value for s in at.sidebar.success)
    assert any(b.label == "Send test alert" for b in at.sidebar.button)


def test_dashboard_shows_spacetrack_button_when_credentials_configured():
    # Real network calls are never triggered in this suite - the button's
    # mere presence is what's being checked here, same discipline this
    # file's own header comment already documents for the CelesTrak
    # button (never clicked by automated tests either).
    with _spacetrack_credentials("test-user", "test-pass"):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

    button_labels = {b.label for b in at.sidebar.button}
    assert any("Space-Track" in label for label in button_labels)


@contextmanager
def _log_dir(tmp_path):
    """Same object.__setattr__-on-the-shared-singleton pattern as
    _operator_tokens/_watched_norad_ids above - points the dashboard's
    real DecisionLogger at an isolated tmp_path so these tests can
    control the exact real log content (needed to actually reproduce a
    real "new entry arrives mid-session" scenario, not just assert on
    whatever's ambiently in the real log)."""
    original = settings.log_dir
    object.__setattr__(settings, "log_dir", str(tmp_path))
    try:
        yield
    finally:
        object.__setattr__(settings, "log_dir", original)


def _widget_test_entry(event_id: str, severity: Severity) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="celestrak",
        raw_data={"object_a_name": "SAT-A", "object_b_name": "SAT-B", "min_distance_km": 50.0},
    )
    finding = AnomalyFinding(event_id=event_id, severity=severity, description="Test.", confidence=0.8)
    decision = Decision(action="continue", rationale="Test rationale.", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def test_dashboard_severity_filter_survives_new_data_arriving_mid_session(tmp_path):
    """Real bug this guards against: without an explicit, stable widget
    key, Streamlit partly derives a multiselect's identity from its
    options list - which grows every time the log grows. An operator's
    active severity filter was silently reverting to empty the moment
    background activity (e.g. a scheduler tick) changed what severities
    were present in the log - no crash, just quietly wrong UI state."""
    with _log_dir(tmp_path):
        logger = DecisionLogger(settings=settings)
        logger.log(_widget_test_entry("e1", Severity.WATCH))
        logger.log(_widget_test_entry("e2", Severity.CRITICAL))

        at = AppTest.from_file(DASHBOARD_PATH)
        _open_section(at, "Full Log").run(timeout=30)

        severity_filter = next(m for m in at.multiselect if m.label == "Filter by severity")
        severity_filter.set_value(["watch"]).run(timeout=30)
        assert severity_filter.value == ["watch"]

        # New data arrives mid-session (e.g. a real scheduler tick, or
        # another operator's action) - the options list changes shape.
        logger.log(_widget_test_entry("e3", Severity.NOMINAL))
        at.run(timeout=30)  # simulates clicking "Refresh"

        severity_filter = next(m for m in at.multiselect if m.label == "Filter by severity")
        assert severity_filter.value == ["watch"]  # real selection survived, not reset to []


def test_dashboard_event_selection_survives_new_data_arriving_mid_session(tmp_path):
    """Same real bug, for the "Inspect / mark reviewed" event dropdown -
    an operator mid-review of a specific, deliberately-selected OLDER
    event must not be silently bounced back to the newest entry just
    because a new one was logged in the background."""
    with _log_dir(tmp_path):
        logger = DecisionLogger(settings=settings)
        logger.log(_widget_test_entry("older-event", Severity.WATCH))

        at = AppTest.from_file(DASHBOARD_PATH)
        _open_section(at, "Full Log").run(timeout=30)

        event_select = next(sb for sb in at.selectbox if sb.label == "Event")
        event_select.set_value("older-event").run(timeout=30)
        assert event_select.value == "older-event"

        logger.log(_widget_test_entry("newer-event", Severity.CRITICAL))  # new entry arrives
        at.run(timeout=30)  # simulates clicking "Refresh"

        event_select = next(sb for sb in at.selectbox if sb.label == "Event")
        assert event_select.value == "older-event"  # real selection survived, not reset to newest


def _pending_approval_entry(event_id: str, tca_iso: str) -> DecisionLogEntry:
    """A full CRITICAL conjunction entry awaiting human approval, with a
    caller-chosen time_of_closest_approach so both an already-passed TCA
    and a still-ahead one can be exercised."""
    plan = compute_avoidance_maneuver(object_a="1", object_b="2", min_distance_km=2.0, relative_velocity_km_s=5.0)
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="celestrak",
        raw_data={
            "object_a_id": "1", "object_a_name": "SAT-A",
            "object_b_id": "2", "object_b_name": "SAT-B",
            "min_distance_km": 2.0, "time_of_closest_approach": tca_iso,
            "relative_velocity_km_s": 5.0,
        },
    )
    finding = AnomalyFinding(event_id=event_id, severity=Severity.CRITICAL, description="Test.", confidence=0.9)
    decision = Decision(
        action="abort", rationale="Test rationale.", made_at=datetime.now(timezone.utc),
        maneuver_plan=plan, awaiting_human_approval=True,
    )
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def test_dashboard_approve_button_disabled_when_tca_already_passed(tmp_path):
    """Real bug this fixes: the warning text alone ("approving this
    maneuver no longer has any effect") didn't stop the Approve button
    itself from rendering as a normal, fully clickable primary button -
    a stale maneuver's Approve control must be genuinely disabled, not
    just annotated."""
    with _log_dir(tmp_path):
        logger = DecisionLogger(settings=settings)
        logger.log(_pending_approval_entry("expired-1", "2020-01-01T00:00:00+00:00"))

        at = AppTest.from_file(DASHBOARD_PATH)
        _open_section(at, "Approvals & Attention").run(timeout=30)

        approve_button = next(b for b in at.button if b.key == "approve_expired-1")
        assert approve_button.disabled is True
        # Reject stays enabled - closing out a stale item is still useful.
        reject_button = next(b for b in at.button if b.key == "reject_expired-1")
        assert reject_button.disabled is False


def test_dashboard_approve_button_enabled_when_tca_still_ahead(tmp_path):
    with _log_dir(tmp_path):
        logger = DecisionLogger(settings=settings)
        future_tca = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        logger.log(_pending_approval_entry("future-1", future_tca))

        at = AppTest.from_file(DASHBOARD_PATH)
        _open_section(at, "Approvals & Attention").run(timeout=30)

        approve_button = next(b for b in at.button if b.key == "approve_future-1")
        assert approve_button.disabled is False


def _attitude_attention_entry(event_id: str) -> DecisionLogEntry:
    """A CRITICAL attitude-hazard entry shaped like needs_attention()
    expects (no maneuver_plan, not yet reviewed) - real event_id, but the
    same object/pointing-error reading every caller in a group shares."""
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="synthetic-attitude-fixture",
        raw_data={
            "object_id": "99030", "object_name": "SYNTH-SAT-CRITICAL",
            "pointing_error_deg": 70.0, "angular_rate_deg_s": 4.5, "solar_panel_power_pct": 22,
        },
    )
    finding = AnomalyFinding(event_id=event_id, severity=Severity.CRITICAL, description="Test.", confidence=0.8)
    decision = Decision(action="abort", rationale="Test rationale.", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def test_dashboard_groups_duplicate_needs_attention_findings(tmp_path):
    """Real bug this fixes: several CRITICAL findings sharing the same
    real object and metric (the same synthetic fixture logged more than
    once) rendered as N nearly-identical cards differing only by
    event_id - they should collapse into one summary card with a count."""
    with _log_dir(tmp_path):
        logger = DecisionLogger(settings=settings)
        for i in range(3):
            logger.log(_attitude_attention_entry(f"attn-{i}"))

        at = AppTest.from_file(DASHBOARD_PATH)
        _open_section(at, "Approvals & Attention").run(timeout=30)

        assert any("3 SYNTH-SAT-CRITICAL events" in m.value for m in at.markdown)
        assert any("pointing error 70" in m.value for m in at.markdown)
        # One bulk "Acknowledge all" action for the group, not 3 separate
        # always-visible individual Acknowledge buttons.
        assert any(b.label.startswith("Acknowledge all") for b in at.button)


def test_dashboard_does_not_group_a_single_needs_attention_finding(tmp_path):
    with _log_dir(tmp_path):
        logger = DecisionLogger(settings=settings)
        logger.log(_attitude_attention_entry("attn-solo"))

        at = AppTest.from_file(DASHBOARD_PATH)
        _open_section(at, "Approvals & Attention").run(timeout=30)

        assert any(b.label == "Acknowledge" and b.key == "acknowledge_attn-solo" for b in at.button)
        assert not any(b.label.startswith("Acknowledge all") for b in at.button)
