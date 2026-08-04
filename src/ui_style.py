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
# series, table cells - so CRITICAL always means the same red, everywhere.
SEVERITY_BADGE_COLORS = {
    "critical": "#DC2626",
    "warning": "#EA580C",
    "watch": "#CA8A04",
    "nominal": "#16A34A",
}
_DEFAULT_BADGE_COLOR = "#6B7280"  # unrecognized severity value - neutral grey, not silently invisible

# Text color per badge background, picked for real WCAG contrast rather
# than assuming white always reads cleanly - white-on-#CA8A04 (watch) is
# only a 2.94:1 contrast ratio (fails WCAG AA's 4.5:1 for normal text),
# and white-on-warning/nominal are both marginal (~3.3-3.6:1). Dark text
# (#111827) clears 4.5:1 against warning/watch/nominal; critical (#DC2626)
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
