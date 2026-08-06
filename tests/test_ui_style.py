"""Tests for src/ui_style.py - pure functions, no Streamlit/Plotly involved."""
import colorsys

from src.ui_style import (
    CATEGORY_STYLE,
    SEVERITY_BADGE_COLORS,
    SEVERITY_BADGE_TEXT_COLORS,
    classify_object_category,
    muted_severity_rgba,
    severity_badge_html,
    system_note_html,
)


def _hue_degrees(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, _l, _s = colorsys.rgb_to_hls(r, g, b)
    return h * 360


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


def test_watch_and_warning_hues_are_widely_separated():
    """Regression test: a design-review pass found the original WATCH/
    WARNING colors (both orange-ish golds, ~20 degrees apart in hue) too
    close to reliably distinguish for red-green color vision deficiency,
    especially as adjacent segments in a stacked bar chart. They must
    stay at least 25 degrees apart, with WATCH the more yellow of the two
    (higher hue) and WARNING the more red-orange (lower hue)."""
    watch_hue = _hue_degrees(SEVERITY_BADGE_COLORS["watch"])
    warning_hue = _hue_degrees(SEVERITY_BADGE_COLORS["warning"])
    assert watch_hue - warning_hue >= 25
    assert watch_hue > warning_hue > _hue_degrees(SEVERITY_BADGE_COLORS["critical"])


def test_system_note_html_uses_a_neutral_tone_not_a_severity_color():
    html = system_note_html("Some config reminder.")
    assert "Some config reminder." in html
    # None of the four severity hex colors should leak into a system note.
    for color in SEVERITY_BADGE_COLORS.values():
        assert color not in html


def test_system_note_html_omits_icon_when_message_already_carries_one():
    html = system_note_html("🔕 Already has an icon.", icon="")
    assert html.count("🔕") == 1


def test_muted_severity_rgba_carries_the_same_hue_as_the_badge():
    rgba = muted_severity_rgba("critical", alpha=0.8)
    assert rgba.startswith("rgba(220, 38, 38")
    assert "0.8" in rgba


def test_muted_severity_rgba_handles_unknown_severity_without_crashing():
    rgba = muted_severity_rgba("mystery")
    assert rgba.startswith("rgba(")
