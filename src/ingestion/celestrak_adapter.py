"""CelesTrak TLE adapter: real orbital data in, conjunction candidates out."""
from __future__ import annotations

import itertools
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from skyfield.api import EarthSatellite, load

from ..orbital import find_closest_approach, parse_norad_id, parse_tle_epoch
from ..schemas import TelemetryEvent
from .base_adapter import DataSourceAdapter

DEFAULT_CACHE_DIR = Path("data/tle_cache")
CACHE_MAX_AGE_SECONDS = 60 * 60  # 1 hour


class CelesTrakAdapter(DataSourceAdapter):
    """Fetches real TLE data from CelesTrak and ranks pairwise conjunctions.

    The raw TLE text is cached to disk for up to an hour (CelesTrak asks
    users to go easy on request volume, and there's no reason to refetch
    on every call during dev/testing). Each fetch_batch() call parses the
    cached/fetched text into EarthSatellite objects, computes every
    pairwise closest approach in the sample via
    orbital.find_closest_approach, and returns the closest ones first.
    """

    def __init__(
        self,
        group: str = "cosmos-2251-debris",
        sample_size: int = 30,
        lookahead_hours: int = 48,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ):
        self.group = group
        self.sample_size = sample_size
        self.lookahead_hours = lookahead_hours
        self.cache_dir = Path(cache_dir)
        self.ts = load.timescale()

    def _cache_path(self) -> Path:
        return self.cache_dir / f"{self.group}.txt"

    def _fetch_tle_text(self) -> str:
        cache_path = self._cache_path()
        if cache_path.exists():
            age_seconds = time.time() - cache_path.stat().st_mtime
            if age_seconds < CACHE_MAX_AGE_SECONDS:
                return cache_path.read_text()

        url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={self.group}&FORMAT=tle"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
        return text

    @staticmethod
    def _parse_tle_groups(text: str) -> list[tuple[str, str, str]]:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        groups = []
        for i in range(0, len(lines) - 2, 3):
            name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
            if line1.startswith("1 ") and line2.startswith("2 "):
                groups.append((name.strip(), line1, line2))
        return groups

    def _rank_conjunctions(self) -> list[dict]:
        text = self._fetch_tle_text()
        sample = self._parse_tle_groups(text)[: self.sample_size]

        satellites = [EarthSatellite(l1, l2, name, self.ts) for name, l1, l2 in sample]
        norad_ids = [parse_norad_id(l1) for _, l1, _ in sample]
        names = [name for name, _, _ in sample]
        epochs = [parse_tle_epoch(l1) for _, l1, _ in sample]

        start_time = datetime.now(timezone.utc)
        conjunctions = []
        for i, j in itertools.combinations(range(len(satellites)), 2):
            result = find_closest_approach(
                satellites[i], satellites[j], self.ts, start_time,
                hours=self.lookahead_hours,
            )
            # How stale the older of the two TLEs is, in hours - a real
            # (if simplified) signal for AnomalyFinding.confidence rather
            # than a flat constant. See pipeline.compute_confidence.
            epoch_age_hours = max(
                (start_time - epochs[i]).total_seconds() / 3600,
                (start_time - epochs[j]).total_seconds() / 3600,
            )
            conjunctions.append({
                "object_a_id": norad_ids[i],
                "object_a_name": names[i],
                "object_b_id": norad_ids[j],
                "object_b_name": names[j],
                "min_distance_km": result["min_distance_km"],
                "time_of_closest_approach": result["time_of_closest_approach"].isoformat(),
                "relative_velocity_km_s": result["relative_velocity_km_s"],
                "tle_epoch_age_hours": epoch_age_hours,
            })

        conjunctions.sort(key=lambda c: c["min_distance_km"])
        return conjunctions

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        conjunctions = self._rank_conjunctions()
        now = datetime.now(timezone.utc)
        return [
            TelemetryEvent(
                event_id=f"conj-{c['object_a_id']}-{c['object_b_id']}",
                timestamp=now,
                source="celestrak",
                raw_data=c,
            )
            for c in conjunctions[:limit]
        ]
