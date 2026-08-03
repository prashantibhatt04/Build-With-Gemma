"""Shared real TLE fetch/parse/cache logic, used by every adapter that
needs raw TLE data (CelesTrakAdapter for pairwise conjunction screening,
DecayRiskAdapter for per-object decay risk, SpaceTrackAdapter as a
credentialed alternative to CelesTrak - see ROADMAP_TO_PRODUCT.md Phase
1) - a single place for the fetch-with-disk-caching and TLE-block-parsing
logic, instead of duplicating it per adapter.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
import time

import requests

if TYPE_CHECKING:
    from .spacetrack_client import SpaceTrackClient

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


def _catalog_ids_cache_key(catalog_ids: Sequence[str]) -> str:
    """A stable key from LIST CONTENT, not order - two watch-list calls
    with the same IDs in a different order must hit the same cache
    entry, not silently miss and refetch."""
    sorted_ids = sorted(str(i) for i in catalog_ids)
    return hashlib.sha256(",".join(sorted_ids).encode("utf-8")).hexdigest()[:10]


def fetch_tle_by_catalog_ids(catalog_ids: Sequence[str], cache_dir: Path) -> str:
    """Fetches (or reuses a disk-cached copy of) real TLE text for a
    specific, caller-provided list of NORAD catalog IDs - a real
    satellite operator's own asset(s), the capability
    fetch_tle_group_text structurally can't provide: CelesTrak's curated
    groups are fixed named lists (e.g. "stations"), with no way to ask
    for "my specific satellite" unless it happens to already be a member
    of one. This is what actually lets this project screen a real
    customer's own object against a real debris field or another group,
    rather than only ever screening CelesTrak's own pre-picked "stations"
    as the stand-in asset.

    CelesTrak's public gp.php has a real per-object query parameter
    (CATNR) but no bulk-by-ID equivalent, so this issues one real
    request per ID - cached as a single combined file keyed by the
    watch list's content (see _catalog_ids_cache_key), not one cache
    file per object, so a repeat call with the same watch list doesn't
    refetch every object individually.

    An unknown, decayed, or otherwise unresolvable ID degrades honestly:
    CelesTrak returns a real "No GP data found" response (not
    TLE-shaped) for those, which is skipped rather than corrupting the
    combined text - the watch list ends up with one fewer object, not a
    crash, matching this project's existing "degrade, don't fail"
    pattern for every other optional/partial data source.
    """
    cache_key = _catalog_ids_cache_key(catalog_ids)
    cache_path = cache_dir / f"catnr-{cache_key}.txt"
    if cache_path.exists():
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds < CACHE_MAX_AGE_SECONDS:
            return cache_path.read_text()

    blocks = []
    for catalog_id in catalog_ids:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={catalog_id}&FORMAT=tle"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        block = response.text.strip()
        if block and "no gp data found" not in block.lower():
            blocks.append(block)

    text = "\n".join(blocks) + ("\n" if blocks else "")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def _run_spacetrack_query_and_cache(query_path: str, cache_path: Path, client: "SpaceTrackClient") -> str:
    """Shared real-query-plus-cache logic behind both
    fetch_spacetrack_group_text and fetch_spacetrack_by_catalog_ids -
    the only difference between the two is how query_path/cache_path get
    built (a name-pattern filter vs. an explicit ID list), not how the
    real request/response/cache/3LE-name-stripping is handled."""
    if cache_path.exists():
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds < CACHE_MAX_AGE_SECONDS:
            return cache_path.read_text()

    raw_text = client.query_text(query_path)
    text = "\n".join(
        line[2:] if line.startswith("0 ") else line
        for line in raw_text.splitlines()
    ) + "\n"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def fetch_spacetrack_by_catalog_ids(
    catalog_ids: Sequence[str], client: "SpaceTrackClient", cache_dir: Path,
) -> str:
    """Space-Track equivalent of fetch_tle_by_catalog_ids above, but a
    single real bulk query - Space-Track's GP class accepts a real
    comma-separated NORAD_CAT_ID list in one request, a genuine
    efficiency advantage over CelesTrak's public gp.php, which has no
    bulk-by-ID equivalent at all. Same DECAY_DATE/EPOCH staleness
    filters as fetch_spacetrack_group_text, for the identical real
    reason documented there - an unfiltered query can return decades-
    stale historical elsets that silently propagate to NaN.
    """
    cache_key = _catalog_ids_cache_key(catalog_ids)
    cache_path = cache_dir / f"spacetrack-catnr-{cache_key}.txt"
    ids_param = ",".join(str(catalog_id) for catalog_id in catalog_ids)
    query_path = (
        f"class/gp/NORAD_CAT_ID/{ids_param}/DECAY_DATE/null-val/"
        f"EPOCH/%3Enow-30/orderby/NORAD_CAT_ID/format/3le"
    )
    return _run_spacetrack_query_and_cache(query_path, cache_path, client)


def fetch_spacetrack_group_text(
    name_pattern: str, client: "SpaceTrackClient", cache_dir: Path, group_slug: str,
) -> str:
    """Space-Track equivalent of fetch_tle_group_text above - same disk-
    caching behavior (1 hour, one file per group), but sourced from a real
    authenticated Space-Track GP-class query instead of CelesTrak's public
    gp.php. CelesTrak has no server-side concept of "groups" beyond its
    own curated named lists - the closest real equivalent on Space-Track
    is a name-pattern filter on the GP class's OBJECT_NAME field, using
    the documented `~~` ("like") predicate operator, e.g. "COSMOS 2251 DEB"
    to match the real debris field CelesTrakAdapter's default group
    already covers.

    Returns the same 3-line-block TLE text format fetch_tle_group_text
    does, so parse_tle_blocks needs no Space-Track-specific branch at
    all - one parser, two real data sources. This requires requesting
    Space-Track's `3le` format specifically, not `tle` - confirmed by a
    real live query against this project's own Space-Track account
    (ROADMAP_TO_PRODUCT.md Phase 1's live verification) that `format=tle`
    returns bare 2-line elements with NO name line at all, unlike
    CelesTrak's own `FORMAT=tle`, which always includes one. Feeding that
    2-line stream through parse_tle_blocks (written and tested against a
    3-line format) silently corrupted results: roughly half of all real
    objects were dropped, and the rest had a neighboring object's orbital
    element line substituted for their name. `3le` is the format that
    actually includes a name line - prefixed with the literal "0 " by the
    real CCSDS/spacetrack 3LE convention (confirmed live: "0 ISS
    (ZARYA)"), which is stripped below so the parsed name matches
    CelesTrak's own unprefixed convention exactly.

    Cached separately from CelesTrak's cache files (a "spacetrack-"
    prefix) - the two sources are never assumed to return identical
    results even for a similarly-named group, so sharing a cache entry
    between them would be wrong. Keyed by group_slug AND a short hash of
    the actual name_pattern (not the raw pattern text itself, which can
    contain characters unsafe in a filename) - group_slug alone isn't a
    safe cache key on its own: two adapter instances using the same slug
    with a DIFFERENT name_pattern (a real possibility - the mapping is
    caller-configurable) would otherwise silently share one cache file,
    serving one pattern's stale TLE data back under the other pattern's
    label with no error or indication anything was wrong.

    Filters to DECAY_DATE/null-val (never decayed) and a recent EPOCH
    (within the last 30 days) - a real bug found live-testing against an
    actual account: unlike CelesTrak's curated, current-only groups,
    Space-Track's raw GP class returns every historical elset ever
    published for a matching name, including objects with decades-stale
    epochs whose orbits have long since decayed. Propagating those
    forward to "now" isn't just inaccurate, it's UNDEFINED - SGP4 breaks
    down and silently returns NaN position for an orbit this far outside
    its valid range, which would otherwise flow straight into
    min_distance_km without any exception being raised. Confirmed live:
    of 160 objects returned by an unfiltered `~~ISS` query, all but one
    produced NaN; the same query with this filter returned 13 objects,
    every one of which propagated cleanly.
    """
    pattern_hash = hashlib.sha256(name_pattern.encode("utf-8")).hexdigest()[:10]
    cache_path = cache_dir / f"spacetrack-{group_slug}-{pattern_hash}.txt"
    query_path = (
        f"class/gp/OBJECT_NAME/~~{name_pattern}/DECAY_DATE/null-val/"
        f"EPOCH/%3Enow-30/orderby/NORAD_CAT_ID/format/3le"
    )
    return _run_spacetrack_query_and_cache(query_path, cache_path, client)


def parse_tle_blocks(text: str) -> list[tuple[str, str, str]]:
    """Parses raw multi-object TLE text (name line, then two element
    lines, repeated) into (name, tle_line1, tle_line2) tuples. Silently
    skips anything that doesn't look like a valid 3-line block, rather
    than raising - real CelesTrak responses are well-formed, but this
    stays defensive against a truncated/malformed fetch.

    Resyncs one line at a time rather than striding in fixed +3 jumps -
    a fixed stride means one missing/corrupted line anywhere in the
    middle of the file desyncs every block after it, silently dropping
    them all rather than just the one block that was actually bad."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    blocks = []
    i = 0
    while i <= len(lines) - 3:
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            blocks.append((name.strip(), line1, line2))
            i += 3
        else:
            i += 1
    return blocks
