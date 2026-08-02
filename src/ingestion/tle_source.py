"""Shared real CelesTrak TLE fetch/parse/cache logic, used by every
adapter that needs raw TLE data (CelesTrakAdapter for pairwise conjunction
screening, DecayRiskAdapter for per-object decay risk) - a single place
for the fetch-with-disk-caching and TLE-block-parsing logic, instead of
duplicating it per adapter.
"""
from __future__ import annotations

from pathlib import Path
import time

import requests

DEFAULT_CACHE_DIR = Path("data/tle_cache")
CACHE_MAX_AGE_SECONDS = 60 * 60  # 1 hour


def fetch_tle_group_text(group: str, cache_dir: Path) -> str:
    """Fetches (or reuses a disk-cached copy of) the raw TLE text for a
    whole CelesTrak group. Cached for up to an hour - CelesTrak asks users
    to go easy on request volume, and there's no reason to refetch on
    every call during dev/testing. Adapters fetching the SAME group (e.g.
    both CelesTrakAdapter and DecayRiskAdapter defaulting to
    cosmos-2251-debris) share this cache file, not just the fetch logic."""
    cache_path = cache_dir / f"{group}.txt"
    if cache_path.exists():
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds < CACHE_MAX_AGE_SECONDS:
            return cache_path.read_text()

    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def parse_tle_blocks(text: str) -> list[tuple[str, str, str]]:
    """Parses raw multi-object TLE text (name line, then two element
    lines, repeated) into (name, tle_line1, tle_line2) tuples. Silently
    skips anything that doesn't look like a valid 3-line block, rather
    than raising - real CelesTrak responses are well-formed, but this
    stays defensive against a truncated/malformed fetch."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    blocks = []
    for i in range(0, len(lines) - 2, 3):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            blocks.append((name.strip(), line1, line2))
    return blocks
