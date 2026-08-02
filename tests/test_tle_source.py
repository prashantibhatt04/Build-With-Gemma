"""Tests for src/ingestion/tle_source.py - the shared fetch/parse/cache
logic used by CelesTrakAdapter and DecayRiskAdapter. Network calls mocked
throughout.
"""
from unittest.mock import MagicMock, patch

from src.ingestion.tle_source import fetch_tle_group_text, parse_tle_blocks

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
