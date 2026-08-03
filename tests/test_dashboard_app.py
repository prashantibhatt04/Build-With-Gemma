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
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import settings

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "scripts" / "dashboard.py")


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


def test_dashboard_shows_unauthenticated_warning_when_operator_tokens_unset():
    with _operator_tokens({}):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)

    assert any("Unauthenticated" in w.value for w in at.warning)
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

    assert any("stations" in i.value and "WATCHED_NORAD_IDS" in i.value for i in at.sidebar.info)


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
