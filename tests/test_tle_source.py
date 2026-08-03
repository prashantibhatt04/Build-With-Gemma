"""Tests for src/ingestion/tle_source.py - the shared fetch/parse/cache
logic used by CelesTrakAdapter and DecayRiskAdapter. Network calls mocked
throughout.
"""
from unittest.mock import MagicMock, patch

from src.ingestion.tle_source import fetch_spacetrack_group_text, fetch_tle_group_text, parse_tle_blocks

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

    assert (tmp_path / "spacetrack-cosmos-2251-debris.txt").exists()


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

    assert (tmp_path / "spacetrack-stations.txt").exists()
    assert (tmp_path / "stations.txt").exists()
