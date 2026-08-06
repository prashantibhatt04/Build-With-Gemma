"""Shared visual styling for scripts/dashboard.py - the severity badge
palette, and object-category classification/colors for the 3D orbital
plots (src/orbit_plot_data.py's conjunction trajectory plot and
src/live_positions.py's live tracking globe). Pure Python, no Streamlit/
Plotly dependency, so it's directly unit-testable like every other
src/*.py data module - only scripts/dashboard.py and the two plot builders
import from here.

Object-category classification is cosmetic only - it picks a marker
color/shape for a 3D plot, and is never fed back into any real severity
or maneuver decision (those stay strictly deterministic; see
src/pipeline.py). It's a best-effort read of a real object's name/CelesTrak
group, not a new data source.
"""
from __future__ import annotations

from typing import Optional

# Section 3 of the dashboard UI spec: one fixed hex per severity, reused
# everywhere a severity is shown as a short status string - badges, chart
# series, table cells, card left-borders - so CRITICAL always means the
# same red, everywhere. WARNING and WATCH are deliberately pushed apart in
# hue (a ~30 degree gap, vs. ~20 degrees originally) - a design-review
# pass found the original two ("#EA580C"/"#CA8A04", both orange-ish golds)
# too close for anyone with red-green color vision deficiency to reliably
# tell apart at a glance, especially as adjacent segments in the Trends
# stacked bar chart. WARNING now reads as a real red-orange, WATCH as a
# clear yellow - CRITICAL/red, WARNING/red-orange, WATCH/yellow,
# NOMINAL/green are each a distinct hue family, not just a shade apart.
SEVERITY_BADGE_COLORS = {
    "critical": "#DC2626",
    "warning": "#DD450E",
    "watch": "#EBB60A",
    "nominal": "#16A34A",
}
_DEFAULT_BADGE_COLOR = "#6B7280"  # unrecognized severity value - neutral grey, not silently invisible

# Text color per badge background, picked for real WCAG contrast rather
# than assuming white always reads cleanly - white text on any of the
# warning/watch/nominal backgrounds falls well under WCAG AA's 4.5:1
# minimum for normal text (watch's clear yellow is the worst, ~1.9:1).
# Dark text (#111827) clears 4.5:1 against all three; critical (#DC2626)
# is the one background dark red enough that white still reads best there
# (4.83:1 white vs 4.35:1 dark).
_DARK_BADGE_TEXT = "#111827"
SEVERITY_BADGE_TEXT_COLORS = {
    "critical": "white",
    "warning": _DARK_BADGE_TEXT,
    "watch": _DARK_BADGE_TEXT,
    "nominal": _DARK_BADGE_TEXT,
}
_DEFAULT_BADGE_TEXT_COLOR = "white"

# Section 1 of the dashboard UI spec: marker colors/shapes per real
# object category.
STATION_COLOR = "#00F0FF"
DOCKED_VEHICLE_COLOR = "#FBBF24"
DEBRIS_COLOR = "#EF4444"

GRID_LINE_COLOR = "rgba(200, 200, 200, 0.2)"
EARTH_SURFACE_OPACITY = 0.35

# Real, named crewed stations CelesTrak's "stations" group tracks - every
# other object in that same group is a currently-docked visiting vehicle
# (a crew/cargo spacecraft), not a station in its own right.
_KNOWN_STATION_NAME_TOKENS = ("ISS", "ZARYA", "TIANGONG", "TIANHE", "MIR")

CATEGORY_STYLE = {
    "station": {"color": STATION_COLOR, "symbol": "circle", "size": 9, "label": "Crewed Station"},
    "docked_vehicle": {"color": DOCKED_VEHICLE_COLOR, "symbol": "square", "size": 6, "label": "Docked Vehicle"},
    "debris": {"color": DEBRIS_COLOR, "symbol": "circle", "size": 4, "label": "Debris / Unmonitored"},
}


def classify_object_category(name: str, group: Optional[str] = None) -> str:
    """Best-effort category for a tracked object - "station",
    "docked_vehicle", or "debris" (which also covers "unmonitored", per
    the single combined visual bucket the UI spec asks for). "debris" is
    only ever inferred from the object's own CelesTrak group label (e.g.
    "cosmos-2251-debris") or a literal "DEB" token in its name - a bare
    object name alone can't otherwise imply debris. Anything in a real
    "stations"-style group that isn't a named station itself is a docked
    vehicle; everything else (no group info, not a known station name)
    falls back to the debris/unmonitored bucket, since that's the safer
    "we don't actually know" default for a cosmetic marker color."""
    name_upper = name.upper()
    group_lower = (group or "").lower()
    if "debris" in group_lower or "DEB" in name_upper.split():
        return "debris"
    if any(token in name_upper for token in _KNOWN_STATION_NAME_TOKENS):
        return "station"
    if group_lower == "stations":
        return "docked_vehicle"
    return "debris"


# A neutral blue-gray tone for app/config-state notices (auth, alerting,
# monitoring setup) - deliberately a different hue FAMILY from the
# severity palette above (blue-gray vs. red/orange/yellow/green), not
# just a different shade of the same warning color. A design-review pass
# found these config reminders ("OPERATOR_TOKENS is not configured", "..
# webhook alerting is not configured") were sharing the same yellow/amber
# family as a real WATCH-severity finding, reading as a risk finding when
# they're really just "you haven't set this up yet" notes about the app
# itself, not about anything actually being tracked as dangerous.
SYSTEM_NOTE_BG = "rgba(100, 116, 139, 0.12)"
SYSTEM_NOTE_BORDER = "rgba(100, 116, 139, 0.4)"


def system_note_html(message: str, icon: str = "info") -> str:
    """A neutral blue-gray notice box for app/config state, standing in
    for st.warning/st.info for this specific category of message (not a
    replacement for those elsewhere - src/alerting.py-triggered real
    findings, error states, and success confirmations are untouched).
    Text color is left unset so it inherits Streamlit's own theme text
    color - light and dark mode both handled automatically without a
    separate light/dark branch, only the translucent background/border
    carry the tint. Rendered via st.markdown(..., unsafe_allow_html=True).

    `icon` is a bare Material Symbols icon name (e.g. "warning", not
    ":material/warning:") - Streamlit's `:material/icon_name:` shortcode
    only expands when text passes through Streamlit's own markdown
    pipeline, which this raw HTML div bypasses (confirmed live: the
    shortcode rendered as literal text here instead of an icon). The span
    below replicates the exact markup Streamlit itself generates for a
    working icon (inspected from a real rendered st.subheader) - same
    "Material Symbols Rounded" font-family ligature technique, just
    built by hand since the shortcode preprocessor never sees this
    string. Pass icon="" for no icon at all."""
    if icon:
        icon_span = (
            f'<span aria-hidden="true" style="display:inline-block;'
            f'font-family:\'Material Symbols Rounded\';font-weight:400;'
            f'vertical-align:bottom;white-space:nowrap;margin-right:0.35rem;">{icon}</span>'
        )
        body = f"{icon_span}{message}"
    else:
        body = message
    return (
        f'<div style="background-color:{SYSTEM_NOTE_BG};border:1px solid {SYSTEM_NOTE_BORDER};'
        f'border-radius:0.5rem;padding:0.75rem 1rem;font-size:0.925rem;line-height:1.5;">'
        f"{body}</div>"
    )


def muted_severity_rgba(severity: str, alpha: float = 0.8) -> str:
    """The same SEVERITY_BADGE_COLORS hue as an rgba() string instead of
    an opaque hex - used to tint just a KPI number's text color (never a
    filled badge/background) for a risk-indicating count, without
    introducing a second, separately-chosen "risk color" palette. Alpha
    (not a hand-picked lighter hex) is what makes it read as "muted" -
    partial opacity blends toward the page's own background, so the same
    value looks like a soft tint in both light and dark theme without
    needing a separate light/dark branch."""
    color = SEVERITY_BADGE_COLORS.get(severity.lower(), _DEFAULT_BADGE_COLOR)
    r, g, b = (int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def severity_badge_html(severity: str) -> str:
    """An inline-styled HTML pill for a severity value - shared by every
    place scripts/dashboard.py shows a severity as a short status string
    (card headers, table cells), so a CRITICAL badge always means the same
    red everywhere. Text color comes from SEVERITY_BADGE_TEXT_COLORS, not
    a flat white, since white text fails real WCAG contrast on the
    lighter watch/warning/nominal backgrounds. Rendered via
    st.markdown(..., unsafe_allow_html=True)."""
    severity_lower = severity.lower()
    color = SEVERITY_BADGE_COLORS.get(severity_lower, _DEFAULT_BADGE_COLOR)
    text_color = SEVERITY_BADGE_TEXT_COLORS.get(severity_lower, _DEFAULT_BADGE_TEXT_COLOR)
    return (
        f'<span style="background-color:{color};color:{text_color};padding:2px 10px;'
        f"border-radius:999px;font-size:0.75rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.03em;white-space:nowrap;">'
        f"{severity.upper()}</span>"
    )
