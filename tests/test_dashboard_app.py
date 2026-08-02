"""Smoke/interaction tests for scripts/dashboard.py using Streamlit's
AppTest harness - runs the real script (no mocking of Streamlit itself),
but doesn't assume specific log content (the real log_dir this process
loads may already have real data from other runs) - correctness of the
underlying aggregation/transform logic is covered separately and
thoroughly in tests/test_dashboard_data.py with controlled fixtures. This
file only confirms the UI wiring itself doesn't crash and exposes the
controls it's supposed to.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "scripts" / "dashboard.py")


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
