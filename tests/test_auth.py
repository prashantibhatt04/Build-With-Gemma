"""Tests for src/auth.py - pure functions, no Streamlit."""
from src.auth import authenticate, parse_operator_tokens


def test_parse_operator_tokens_handles_multiple_entries():
    assert parse_operator_tokens("alice:tok1,bob:tok2") == {"alice": "tok1", "bob": "tok2"}


def test_parse_operator_tokens_handles_whitespace():
    assert parse_operator_tokens(" alice : tok1 , bob:tok2 ") == {"alice": "tok1", "bob": "tok2"}


def test_parse_operator_tokens_returns_empty_dict_for_empty_string():
    assert parse_operator_tokens("") == {}


def test_parse_operator_tokens_skips_malformed_entries():
    assert parse_operator_tokens("alice:tok1,malformed,bob:tok2,:notoken,noname:") == {
        "alice": "tok1", "bob": "tok2",
    }


def test_authenticate_returns_matching_operator_name():
    tokens = {"alice": "tok1", "bob": "tok2"}
    assert authenticate("tok2", tokens) == "bob"


def test_authenticate_returns_none_for_unknown_token():
    assert authenticate("wrong-token", {"alice": "tok1"}) is None


def test_authenticate_returns_none_for_empty_token():
    assert authenticate("", {"alice": "tok1"}) is None


def test_authenticate_returns_none_when_no_tokens_configured():
    assert authenticate("anything", {}) is None
