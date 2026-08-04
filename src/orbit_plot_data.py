"""Real orbital trajectory + distance-over-time data for visualizing a
specific conjunction, and the Plotly figures built from it.

Only meaningful for source="celestrak" events, which are backed by real
NORAD-catalogued objects - synthetic/historical events have no real TLEs
to propagate (see src/ingestion/historical_adapter.py's module docstring
for why even historical TLEs aren't available at all: CelesTrak's public
API only ever serves the CURRENT TLE for a catalog number, confirmed by
querying it directly).

Re-fetches the two objects' CURRENT TLEs by NORAD catalog number and
re-propagates using the same orbital.py functions the real screening
pipeline uses - no duplicated physics. Because this fetches CURRENT TLEs
rather than the exact TLE epoch that produced an originally-logged
min_distance_km, the propagation window starts from now, not from
whenever the event was first logged - orbital elements update over time,
so this is a live, real recomputation, not a byte-exact replay of a past
log entry.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple, Optional

import numpy as np
import plotly.graph_objects as go
import requests
from skyfield.api import EarthSatellite, load

from .orbital import build_coarse_times, compute_coarse_positions
from .ui_style import CATEGORY_STYLE, EARTH_SURFACE_OPACITY, GRID_LINE_COLOR, SEVERITY_BADGE_COLORS, classify_object_category

EARTH_RADIUS_KM = 6371.0

# Same thresholds classify_conjunction_severity (src/pipeline.py) uses -
# duplicated here (not imported) because pipeline.py's thresholds are
# expressed as branches in a function, not as named constants to import.
CRITICAL_THRESHOLD_KM = 5.0
WARNING_THRESHOLD_KM = 25.0
WATCH_THRESHOLD_KM = 100.0


class TrajectoryData(NamedTuple):
    times: list[datetime]
    positions_a: np.ndarray  # shape (3, n) - km, geocentric
    positions_b: np.ndarray  # shape (3, n)
    distances_km: np.ndarray  # shape (n,)
    object_a_name: str
    object_b_name: str
    object_a_id: str = ""
    object_b_id: str = ""
    # Real CelesTrak/Space-Track group each object came from, when known
    # (see celestrak_adapter.py's object_a_group/object_b_group) - used
    # only to pick a marker/line color (ui_style.classify_object_category),
    # optional and defaulted so existing callers/tests that don't have
    # this real data (e.g. a synthetic sample) keep working unchanged.
    object_a_group: Optional[str] = None
    object_b_group: Optional[str] = None


def _fetch_tle_by_catnr(catnr: str) -> tuple[str, str, str]:
    """Fetches the CURRENT TLE for a single NORAD catalog number. Real
    network call, same endpoint CelesTrakAdapter uses but queried by
    CATNR instead of GROUP - no disk caching here, unlike the adapter's
    per-group cache, since this is a one-off, user-triggered lookup for a
    single object pair rather than a repeated bulk scan."""
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    lines = [line for line in response.text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"No current TLE found for NORAD catalog number {catnr!r}")
    return lines[0].strip(), lines[1].rstrip(), lines[2].rstrip()


def fetch_trajectory_data(
    object_a_id: str, object_a_name: str, object_b_id: str, object_b_name: str,
    hours: int = 48,
    object_a_group: Optional[str] = None, object_b_group: Optional[str] = None,
) -> TrajectoryData:
    """Fetches both objects' CURRENT TLEs and computes their real
    positions/separation over the next `hours` hours starting now, using
    the same coarse-pass propagation (orbital.compute_coarse_positions)
    the real screening pipeline uses. object_a_group/object_b_group are
    optional (the real CelesTrak/Space-Track group each object came from,
    e.g. "stations" or "cosmos-2251-debris") - purely cosmetic, used only
    to color the trajectory by real object category (see ui_style)."""
    ts = load.timescale()
    name_a, l1_a, l2_a = _fetch_tle_by_catnr(object_a_id)
    name_b, l1_b, l2_b = _fetch_tle_by_catnr(object_b_id)
    sat_a = EarthSatellite(l1_a, l2_a, name_a, ts)
    sat_b = EarthSatellite(l1_b, l2_b, name_b, ts)

    start_time = datetime.now(timezone.utc)
    times = build_coarse_times(start_time, hours, coarse_step_minutes=5)
    positions_a = compute_coarse_positions(sat_a, ts, times)
    positions_b = compute_coarse_positions(sat_b, ts, times)
    distances_km = np.sqrt(((positions_a - positions_b) ** 2).sum(axis=0))

    return TrajectoryData(
        times=times, positions_a=positions_a, positions_b=positions_b,
        distances_km=distances_km,
        # Prefer the caller's (logged) names - the live TLE's own name
        # field is only a fallback, since it should always agree anyway.
        object_a_name=object_a_name or name_a, object_b_name=object_b_name or name_b,
        object_a_id=object_a_id, object_b_id=object_b_id,
        object_a_group=object_a_group, object_b_group=object_b_group,
    )


def _distance_status(distance_km: float) -> str:
    """The same deterministic severity band a real min_distance_km would
    classify into (src/pipeline.py's thresholds, duplicated as the module
    constants above) - used only to label the closest-approach hover
    tooltip, never fed back into any real decision."""
    if distance_km < CRITICAL_THRESHOLD_KM:
        return "critical"
    if distance_km < WARNING_THRESHOLD_KM:
        return "warning"
    if distance_km < WATCH_THRESHOLD_KM:
        return "watch"
    return "nominal"


def build_distance_chart(data: TrajectoryData) -> go.Figure:
    """Separation between the two objects over time, with this project's
    severity thresholds drawn as reference lines (colored to match the
    dashboard's own severity badges - see ui_style.SEVERITY_BADGE_COLORS)
    - makes it visually obvious when (and whether) the real curve dips
    into risk territory, not just what the single logged min_distance_km
    number was."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.times, y=data.distances_km, mode="lines", name="Separation (km)",
    ))
    for threshold, label, severity in [
        (CRITICAL_THRESHOLD_KM, "CRITICAL (<5km)", "critical"),
        (WARNING_THRESHOLD_KM, "WARNING (<25km)", "warning"),
        (WATCH_THRESHOLD_KM, "WATCH (<100km)", "watch"),
    ]:
        fig.add_hline(
            y=threshold, line_dash="dot", line_color=SEVERITY_BADGE_COLORS[severity], annotation_text=label,
        )
    fig.update_layout(
        title=f"{data.object_a_name} vs {data.object_b_name} — separation over time",
        xaxis_title="Time (UTC)", yaxis_title="Distance (km, log scale)",
        yaxis_type="log",
    )
    return fig


def build_3d_trajectory_figure(data: TrajectoryData) -> go.Figure:
    """3D plot of both objects' real propagated positions (Earth-centered,
    km) over the same window, with Earth drawn to scale (semi-transparent,
    so a trajectory passing behind the globe stays visible) and the
    closest-approach point marked. Not textured/photorealistic - a plain
    sphere at the real Earth radius, purely as a scale reference. Each
    object's line is colored by its real category (station/docked
    vehicle/debris - see ui_style.classify_object_category), and every
    trace carries a hovertemplate (Object Name, NORAD ID, Altitude,
    Status) instead of a persistent on-plot text label, which would
    overlap along a dense trajectory."""
    min_idx = int(np.argmin(data.distances_km))
    status = _distance_status(float(data.distances_km[min_idx]))

    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    earth_x = EARTH_RADIUS_KM * np.cos(u) * np.sin(v)
    earth_y = EARTH_RADIUS_KM * np.sin(u) * np.sin(v)
    earth_z = EARTH_RADIUS_KM * np.cos(v)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=earth_x, y=earth_y, z=earth_z, showscale=False,
        colorscale=[[0, "rgb(30,60,110)"], [1, "rgb(30,60,110)"]],
        opacity=EARTH_SURFACE_OPACITY, name="Earth", hoverinfo="skip",
    ))

    for positions, name, object_id, group in [
        (data.positions_a, data.object_a_name, data.object_a_id, data.object_a_group),
        (data.positions_b, data.object_b_name, data.object_b_id, data.object_b_group),
    ]:
        style = CATEGORY_STYLE[classify_object_category(name, group)]
        altitude_km = np.linalg.norm(positions, axis=0) - EARTH_RADIUS_KM
        fig.add_trace(go.Scatter3d(
            x=positions[0], y=positions[1], z=positions[2],
            mode="lines", name=name, line=dict(color=style["color"], width=4),
            customdata=np.column_stack([np.full(altitude_km.shape, object_id), altitude_km, np.full(altitude_km.shape, style["label"])]),
            hovertemplate=(
                f"<b>{name}</b><br>NORAD ID: %{{customdata[0]}}<br>"
                "Altitude: %{customdata[1]:.0f} km<br>Status: %{customdata[2]}<extra></extra>"
            ),
        ))
    fig.add_trace(go.Scatter3d(
        x=[data.positions_a[0][min_idx]], y=[data.positions_a[1][min_idx]], z=[data.positions_a[2][min_idx]],
        mode="markers", name="Closest approach", marker=dict(color=SEVERITY_BADGE_COLORS[status], size=8, symbol="diamond"),
        customdata=[[data.object_a_name, data.object_b_name, float(data.distances_km[min_idx]), status.upper()]],
        hovertemplate=(
            "<b>Closest approach</b><br>%{customdata[0]} vs %{customdata[1]}<br>"
            "Distance: %{customdata[2]:.2f} km<br>Status: %{customdata[3]}<extra></extra>"
        ),
    ))
    axis = dict(gridcolor=GRID_LINE_COLOR, title="km")
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis=axis, yaxis=axis, zaxis=axis),
        title=f"{data.object_a_name} vs {data.object_b_name} — real propagated trajectories",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    return fig
