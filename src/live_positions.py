"""Real-time positions of the real crewed space stations (CelesTrak's
"stations" group - ISS, Tiangong, and their currently-docked visiting
vehicles), for a "what's actually up there right now" dashboard view.

Deliberately scoped to this one group, not an attempt at a full
~20,000-object catalog view: it's the exact same "asset actually worth
protecting" set CelesTrakAdapter already treats as the real payload of
this system (see its module docstring) - a small, named, recognizable
set is what makes a live view actually readable, rather than an
unreadable point cloud of debris fragments with no names anyone knows.

Positions are computed in the same Earth-centered (GCRS) frame and units
(km) that orbital.py's compute_coarse_positions already uses, evaluated
at "now" instead of propagated forward - no new physics, no new frame
conversion, just the existing SGP4 propagation at a single instant.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from skyfield.api import EarthSatellite, load

from .ingestion.tle_source import DEFAULT_CACHE_DIR, fetch_tle_group_text, parse_tle_blocks
from .orbital import parse_norad_id
from .ui_style import CATEGORY_STYLE, EARTH_SURFACE_OPACITY, GRID_LINE_COLOR, classify_object_category

EARTH_RADIUS_KM = 6371.0
DEFAULT_GROUP = "stations"


@dataclass
class SatellitePosition:
    name: str
    norad_id: str
    x_km: float
    y_km: float
    z_km: float
    altitude_km: float
    category: str = "debris"


def fetch_live_positions(
    group: str = DEFAULT_GROUP, cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> list[SatellitePosition]:
    """Fetches (or reuses the disk-cached) real current TLEs for `group`
    and computes each object's real current position - a real network
    call and real SGP4 propagation, not staged data."""
    ts = load.timescale()
    text = fetch_tle_group_text(group, Path(cache_dir))
    now = ts.now()

    positions = []
    for name, line1, line2 in parse_tle_blocks(text):
        sat = EarthSatellite(line1, line2, name, ts)
        pos_km = sat.at(now).position.km
        altitude_km = float(np.linalg.norm(pos_km) - EARTH_RADIUS_KM)
        positions.append(SatellitePosition(
            name=name.strip(), norad_id=parse_norad_id(line1),
            x_km=float(pos_km[0]), y_km=float(pos_km[1]), z_km=float(pos_km[2]),
            altitude_km=altitude_km, category=classify_object_category(name, group),
        ))
    return positions


def build_live_globe_figure(positions: list[SatellitePosition]) -> go.Figure:
    """3D Earth (plain sphere at the real Earth radius, same style as
    orbit_plot_data.build_3d_trajectory_figure - not textured, purely a
    scale reference, rendered semi-transparent so a marker on the far side
    of the globe stays visible) with one marker trace per real object
    category present (station/docked vehicle/debris - see
    ui_style.classify_object_category) so Plotly's legend maps each marker
    color/shape to what it actually represents. No persistent on-globe
    text labels (those overlap once more than a couple of objects are
    close together) - identity/altitude/category are hover-only, via
    hovertemplate."""
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

    by_category: dict[str, list[SatellitePosition]] = {}
    for p in positions:
        by_category.setdefault(p.category, []).append(p)

    for category, group_positions in by_category.items():
        style = CATEGORY_STYLE[category]
        fig.add_trace(go.Scatter3d(
            x=[p.x_km for p in group_positions], y=[p.y_km for p in group_positions],
            z=[p.z_km for p in group_positions],
            mode="markers", name=style["label"],
            text=[p.name for p in group_positions],
            customdata=[[p.norad_id, p.altitude_km, style["label"]] for p in group_positions],
            hovertemplate=(
                "<b>%{text}</b><br>NORAD ID: %{customdata[0]}<br>"
                "Altitude: %{customdata[1]:.0f} km<br>Status: %{customdata[2]}<extra></extra>"
            ),
            marker=dict(color=style["color"], size=style["size"], symbol=style["symbol"]),
        ))

    axis = dict(gridcolor=GRID_LINE_COLOR, title="km")
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis=axis, yaxis=axis, zaxis=axis),
        title="Real crewed stations — current positions",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    return fig
