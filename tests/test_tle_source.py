"""Tests for src/ingestion/tle_source.py - the shared fetch/parse/cache
logic used by CelesTrakAdapter and DecayRiskAdapter. Network calls mocked
throughout.
"""
from unittest.mock import MagicMock, patch

from src.ingestion.tle_source import (
    fetch_spacetrack_by_catalog_ids,
    fetch_spacetrack_group_text,
    fetch_tle_by_catalog_ids,
    fetch_tle_group_text,
    parse_tle_blocks,
)

SAMPLE_TLE_TEXT = """VANGUARD 1
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
ISS (ZARYA)
1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998
2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452
"""


def _mock_response(text):
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock()
    return response


def test_parse_tle_blocks_extracts_name_and_both_lines():
    blocks = parse_tle_blocks(SAMPLE_TLE_TEXT)

    assert len(blocks) == 2
    assert blocks[0] == (
        "VANGUARD 1",
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
    )
    assert blocks[1][0] == "ISS (ZARYA)"


def test_parse_tle_blocks_skips_malformed_trailing_lines():
    assert parse_tle_blocks(SAMPLE_TLE_TEXT + "TRUNCATED OBJECT\n1 99999U garbage\n") == parse_tle_blocks(SAMPLE_TLE_TEXT)


def test_parse_tle_blocks_handles_empty_text():
    assert parse_tle_blocks("") == []


def test_parse_tle_blocks_recovers_a_valid_block_after_a_corrupted_one():
    """Real bug this closes: the old fixed +3 stride desynced after ANY
    corrupted/missing line and silently dropped every block after it, not
    just the corrupted one - a single network hiccup could zero out a
    whole scan. A genuinely bad block (missing line2) must only cost that
    one object, not the well-formed one right after it."""
    text = (
        "OBJECT A\n"
        "1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998\n"
        # OBJECT A's line2 is missing here - line3 below is really OBJECT B's name.
        "OBJECT B\n"
        "1 33333U 07026A   18135.50000000  .00000010  00000-0  10000-4 0  9991\n"
        "2 33333  98.6000 100.0000 0001000  90.0000 270.0000 14.20000000123456\n"
    )

    blocks = parse_tle_blocks(text)

    assert len(blocks) == 1
    assert blocks[0][0] == "OBJECT B"


def test_parse_tle_blocks_resyncs_across_multiple_corrupted_blocks():
    """Not just recovery from one bad block - a second, later corruption
    must not desync the parser again either."""
    text = (
        "BAD A\n"
        "not a valid line1\n"
        "GOOD A\n"
        "1 11111U 07026A   18135.50000000  .00000010  00000-0  10000-4 0  9991\n"
        "2 11111  98.6000 100.0000 0001000  90.0000 270.0000 14.20000000123456\n"
        "BAD B\n"
        "also not valid\n"
        "GOOD B\n"
        "1 22222U 07026A   18135.50000000  .00000010  00000-0  10000-4 0  9991\n"
        "2 22222  98.6000 100.0000 0001000  90.0000 270.0000 14.20000000123456\n"
    )

    blocks = parse_tle_blocks(text)

    assert [b[0] for b in blocks] == ["GOOD A", "GOOD B"]


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_tle_group_text_caches_and_avoids_refetching(mock_get, tmp_path):
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)

    first = fetch_tle_group_text("test-group", tmp_path)
    assert mock_get.call_count == 1
    assert first == SAMPLE_TLE_TEXT

    second = fetch_tle_group_text("test-group", tmp_path)
    assert mock_get.call_count == 1, "second call should reuse the disk cache, not refetch"
    assert second == SAMPLE_TLE_TEXT

    assert (tmp_path / "test-group.txt").exists()


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_tle_group_text_different_groups_use_different_cache_files(mock_get, tmp_path):
    mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)

    fetch_tle_group_text("group-a", tmp_path)
    fetch_tle_group_text("group-b", tmp_path)

    assert mock_get.call_count == 2
    assert (tmp_path / "group-a.txt").exists()
    assert (tmp_path / "group-b.txt").exists()


ISS_TLE_BLOCK = (
    "ISS (ZARYA)\n"
    "1 25544U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998\n"
    "2 25544  51.6402 181.0633 0004018  88.8954  22.2246 15.54059185113452"
)
VANGUARD_TLE_BLOCK = (
    "VANGUARD 1\n"
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753\n"
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
)


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_tle_by_catalog_ids_issues_one_real_request_per_id(mock_get, tmp_path):
    """Real bug this closes if missing: a real satellite operator's own
    asset almost never happens to be a member of one of CelesTrak's
    curated named groups - CATNR (per-object query) is the only way
    CelesTrak's public API can serve a specific customer's own
    satellite at all."""
    mock_get.side_effect = [_mock_response(ISS_TLE_BLOCK + "\n"), _mock_response(VANGUARD_TLE_BLOCK + "\n")]

    text = fetch_tle_by_catalog_ids(["25544", "5"], tmp_path)

    assert mock_get.call_count == 2
    first_url = mock_get.call_args_list[0].args[0]
    second_url = mock_get.call_args_list[1].args[0]
    assert "CATNR=25544" in first_url
    assert "CATNR=5" in second_url
    blocks = parse_tle_blocks(text)
    assert {b[0] for b in blocks} == {"ISS (ZARYA)", "VANGUARD 1"}


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_tle_by_catalog_ids_caches_by_content_not_order(mock_get, tmp_path):
    mock_get.side_effect = [_mock_response(ISS_TLE_BLOCK + "\n"), _mock_response(VANGUARD_TLE_BLOCK + "\n")]

    fetch_tle_by_catalog_ids(["25544", "5"], tmp_path)
    fetch_tle_by_catalog_ids(["5", "25544"], tmp_path)  # same IDs, different order

    assert mock_get.call_count == 2, "same watch list (regardless of order) must reuse the disk cache"


@patch("src.ingestion.tle_source.requests.get")
def test_fetch_tle_by_catalog_ids_skips_an_unresolvable_id_without_failing(mock_get, tmp_path):
    """A real, honest degrade: an unknown/decayed/private ID must cost
    that one object, not the whole watch list."""
    mock_get.side_effect = [_mock_response("No GP data found\n"), _mock_response(ISS_TLE_BLOCK + "\n")]

    text = fetch_tle_by_catalog_ids(["99999999", "25544"], tmp_path)

    blocks = parse_tle_blocks(text)
    assert len(blocks) == 1
    assert blocks[0][0] == "ISS (ZARYA)"


def test_fetch_spacetrack_by_catalog_ids_issues_one_real_bulk_query(tmp_path):
    """Real advantage over CelesTrak's per-ID-only CATNR: Space-Track's
    GP class genuinely supports a comma-separated NORAD_CAT_ID list in
    ONE real request."""
    client = _fake_spacetrack_client(ISS_TLE_BLOCK.replace("ISS (ZARYA)", "0 ISS (ZARYA)") + "\n")

    text = fetch_spacetrack_by_catalog_ids(["25544", "5"], client, tmp_path)

    assert client.query_text.call_count == 1
    queried_path = client.query_text.call_args.args[0]
    assert "NORAD_CAT_ID/25544,5" in queried_path
    assert "DECAY_DATE/null-val" in queried_path
    assert "EPOCH/%3Enow-30" in queried_path
    assert parse_tle_blocks(text)[0][0] == "ISS (ZARYA)"


def test_fetch_spacetrack_by_catalog_ids_caches_by_content_not_order(tmp_path):
    client = _fake_spacetrack_client(ISS_TLE_BLOCK.replace("ISS (ZARYA)", "0 ISS (ZARYA)") + "\n")

    fetch_spacetrack_by_catalog_ids(["25544", "5"], client, tmp_path)
    fetch_spacetrack_by_catalog_ids(["5", "25544"], client, tmp_path)

    assert client.query_text.call_count == 1, "same watch list (regardless of order) must reuse the disk cache"


def _fake_spacetrack_client(text):
    client = MagicMock()
    client.query_text.return_value = text
    return client


def test_fetch_spacetrack_group_text_queries_with_a_like_filter_and_caches(tmp_path):
    client = _fake_spacetrack_client(SAMPLE_TLE_TEXT)

    first = fetch_spacetrack_group_text("COSMOS 2251 DEB", client, tmp_path, group_slug="cosmos-2251-debris")
    assert parse_tle_blocks(first) == parse_tle_blocks(SAMPLE_TLE_TEXT)
    assert client.query_text.call_count == 1
    queried_path = client.query_text.call_args.args[0]
    assert "OBJECT_NAME/~~COSMOS 2251 DEB" in queried_path
    # Space-Track's `tle` format returns bare 2-line elements with NO name
    # line at all (confirmed live against a real account -
    # ROADMAP_TO_PRODUCT.md Phase 1) - unlike CelesTrak's own FORMAT=tle,
    # which always includes one. `3le` is the format that actually
    # includes a name line, so that's what must be requested here.
    assert "format/3le" in queried_path
    # Regression coverage for the real NaN-propagation bug found live -
    # see fetch_spacetrack_group_text's own docstring.
    assert "DECAY_DATE/null-val" in queried_path
    assert "EPOCH/%3Enow-30" in queried_path

    second = fetch_spacetrack_group_text("COSMOS 2251 DEB", client, tmp_path, group_slug="cosmos-2251-debris")
    assert parse_tle_blocks(second) == parse_tle_blocks(SAMPLE_TLE_TEXT)
    assert client.query_text.call_count == 1, "second call should reuse the disk cache, not re-query"

    # Filename includes a hash of the name_pattern, not just group_slug
    # (see fetch_spacetrack_group_text's own docstring on why) - glob
    # rather than assert an exact literal hash value.
    assert list(tmp_path.glob("spacetrack-cosmos-2251-debris-*.txt"))


def test_fetch_spacetrack_group_text_does_not_collide_across_different_patterns_for_the_same_slug(tmp_path):
    """Real bug this closes: the cache used to be keyed by group_slug
    alone, so two calls sharing a slug but querying a DIFFERENT
    name_pattern (a real possibility - the mapping is caller-configurable)
    would silently serve the first pattern's stale, mislabeled TLE data
    back for the second - no error, no signal anything was wrong."""
    client = _fake_spacetrack_client(SAMPLE_TLE_TEXT)

    fetch_spacetrack_group_text("COSMOS 2251 DEB", client, tmp_path, group_slug="stations")
    fetch_spacetrack_group_text("TIANHE", client, tmp_path, group_slug="stations")

    # Two real, distinct queries were issued - the second pattern was
    # never served from the first pattern's cache entry.
    assert client.query_text.call_count == 2
    assert len(list(tmp_path.glob("spacetrack-stations-*.txt"))) == 2


def test_fetch_spacetrack_group_text_strips_the_real_3le_name_line_prefix(tmp_path):
    """Regression test for a real bug found live-testing against an
    actual Space-Track account: 3LE format prefixes each name line with
    a literal "0 " (confirmed live: "0 ISS (ZARYA)") - stripped here so
    the parsed name matches CelesTrak's own unprefixed convention, not
    leaked into object_a_name/object_b_name downstream."""
    real_shaped_3le = (
        "0 ISS (ZARYA)\r\n"
        "1 25544U 98067A   26214.50635181  .00006342  00000-0  12183-3 0  9997\r\n"
        "2 25544  51.6315  70.8679 0007172   4.7554 355.3502 15.49313226578933\r\n"
    )
    client = _fake_spacetrack_client(real_shaped_3le)

    text = fetch_spacetrack_group_text("ISS", client, tmp_path, group_slug="stations")

    blocks = parse_tle_blocks(text)
    assert len(blocks) == 1
    assert blocks[0][0] == "ISS (ZARYA)", "the leading '0 ' must be stripped, not left in the name"


def test_fetch_spacetrack_group_text_uses_a_cache_file_distinct_from_celestrak(tmp_path):
    st_client = _fake_spacetrack_client(SAMPLE_TLE_TEXT)

    fetch_spacetrack_group_text("ISS", st_client, tmp_path, group_slug="stations")

    with patch("src.ingestion.tle_source.requests.get") as mock_get:
        mock_get.return_value = _mock_response(SAMPLE_TLE_TEXT)
        fetch_tle_group_text("stations", tmp_path)
        assert mock_get.call_count == 1, "CelesTrak's own cache file must be unaffected by the Space-Track fetch above"

    assert list(tmp_path.glob("spacetrack-stations-*.txt"))
    assert (tmp_path / "stations.txt").exists()
