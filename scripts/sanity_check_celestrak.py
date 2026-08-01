#!/usr/bin/env python3
"""THROWAWAY sanity check - not production code.

Fetches real TLE data from CelesTrak, propagates a small sample of
objects over the next 48 hours with Skyfield/SGP4, and finds the
closest pairwise conjunctions by actually sampling the trajectory
(not just checking a single instant).

Just proving the data + math work before building the real adapter.
"""
import itertools
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from skyfield.api import EarthSatellite, load

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle"
SAMPLE_SIZE = 30
LOOKAHEAD_HOURS = 48
STEP_MINUTES = 5


def fetch_tle_text() -> str:
    resp = requests.get(TLE_URL, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_tle_groups(text: str) -> list[tuple[str, str, str]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    groups = []
    for i in range(0, len(lines) - 2, 3):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            groups.append((name.strip(), line1, line2))
    return groups


def main() -> None:
    print(f"Fetching TLE data from: {TLE_URL}")
    text = fetch_tle_text()
    groups = parse_tle_groups(text)
    print(f"Objects loaded from CelesTrak: {len(groups)}")

    if len(groups) < 2:
        print("Not enough objects to compare. Aborting.")
        return

    sample = groups[:SAMPLE_SIZE]
    print(f"Using sample of {len(sample)} objects for pairwise conjunction check")

    ts = load.timescale()
    satellites = [EarthSatellite(l1, l2, name, ts) for name, l1, l2 in sample]
    names = [name for name, _, _ in sample]

    n_steps = int(LOOKAHEAD_HOURS * 60 / STEP_MINUTES) + 1
    start = datetime.now(timezone.utc)
    dt_list = [start + timedelta(minutes=STEP_MINUTES * i) for i in range(n_steps)]
    t = ts.from_datetimes(dt_list)

    print(f"Propagating {len(satellites)} objects across {n_steps} time steps "
          f"({STEP_MINUTES}-min steps over {LOOKAHEAD_HOURS}h)...")

    # positions[k] has shape (3, n_steps) in km, GCRS frame
    positions = np.array([sat.at(t).position.km for sat in satellites])

    results = []
    for i, j in itertools.combinations(range(len(satellites)), 2):
        diff = positions[i] - positions[j]
        dist_km = np.sqrt((diff ** 2).sum(axis=0))
        min_idx = int(np.argmin(dist_km))
        results.append((dist_km[min_idx], dt_list[min_idx], names[i], names[j]))

    results.sort(key=lambda r: r[0])

    print("\nTop 5 closest conjunctions in sample over next "
          f"{LOOKAHEAD_HOURS}h:")
    for rank, (dist, when, name_a, name_b) in enumerate(results[:5], start=1):
        print(f"{rank}. {name_a}  <->  {name_b}")
        print(f"   min distance: {dist:.3f} km   at: {when.isoformat()}")


if __name__ == "__main__":
    main()
