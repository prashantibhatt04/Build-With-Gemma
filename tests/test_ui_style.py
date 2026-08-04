"""Tests for src/ui_style.py - pure functions, no Streamlit/Plotly involved."""
from src.ui_style import (
    CATEGORY_STYLE,
    SEVERITY_BADGE_COLORS,
    SEVERITY_BADGE_TEXT_COLORS,
    classify_object_category,
    severity_badge_html,
)


def test_classify_object_category_recognizes_known_station_names():
    assert classify_object_category("ISS (ZARYA)", group="stations") == "station"
    assert classify_object_category("TIANGONG", group="stations") == "station"


def test_classify_object_category_treats_non_station_names_in_stations_group_as_docked_vehicle():
    assert classify_object_category("PROGRESS MS-25", group="stations") == "docked_vehicle"
    assert classify_object_category("CREW DRAGON ENDEAVOUR", group="stations") == "docked_vehicle"


def test_classify_object_category_reads_debris_from_group_name():
    assert classify_object_category("COSMOS 2251 DEB", group="cosmos-2251-debris") == "debris"


def test_classify_object_category_reads_debris_from_name_token_without_group():
    assert classify_object_category("COSMOS 2251 DEB") == "debris"


def test_classify_object_category_defaults_to_debris_when_unknown():
    # No group info, no station tokens, no debris token - the "we don't
    # actually know" default, per the module docstring.
    assert classify_object_category("SOME-RANDOM-SAT") == "debris"


def test_every_category_has_a_style_entry():
    for category in ("station", "docked_vehicle", "debris"):
        assert category in CATEGORY_STYLE
        assert "color" in CATEGORY_STYLE[category]
        assert "label" in CATEGORY_STYLE[category]


def test_severity_badge_html_uses_the_right_color_and_uppercases():
    html = severity_badge_html("critical")
    assert SEVERITY_BADGE_COLORS["critical"] in html
    assert "CRITICAL" in html


def test_severity_badge_html_handles_unknown_severity_without_crashing():
    html = severity_badge_html("mystery")
    assert "MYSTERY" in html


def test_watch_and_warning_badges_use_dark_text_for_real_contrast():
    """Regression test: white text on the watch/warning/nominal
    backgrounds is under WCAG's 4.5:1 minimum contrast ratio for normal
    text (watch was only 2.94:1) - these three must render with dark
    text, not white."""
    for severity in ("watch", "warning", "nominal"):
        html = severity_badge_html(severity)
        assert SEVERITY_BADGE_TEXT_COLORS[severity] in html
        assert "color:white" not in html.replace("background-color:white", "")


def test_critical_badge_still_uses_white_text():
    # #DC2626 is dark/saturated enough that white text (4.83:1) beats
    # dark text (4.35:1) - the one severity that should stay unchanged.
    html = severity_badge_html("critical")
    assert SEVERITY_BADGE_TEXT_COLORS["critical"] == "white"
    assert "color:white" in html
