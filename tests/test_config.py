"""Tests for src/config.py's env-var parsing helpers."""
from src.config import _parse_watched_norad_ids


def test_parse_watched_norad_ids_splits_on_commas():
    assert _parse_watched_norad_ids("25544,48274") == ("25544", "48274")


def test_parse_watched_norad_ids_strips_whitespace():
    assert _parse_watched_norad_ids(" 25544 , 48274 ") == ("25544", "48274")


def test_parse_watched_norad_ids_drops_empty_items():
    assert _parse_watched_norad_ids("25544,,48274,") == ("25544", "48274")


def test_parse_watched_norad_ids_empty_string_is_empty_tuple():
    assert _parse_watched_norad_ids("") == ()


def test_parse_watched_norad_ids_single_id():
    assert _parse_watched_norad_ids("25544") == ("25544",)
