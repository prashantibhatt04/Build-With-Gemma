"""Tests for src/ingestion/spacetrack_client.py. All real network calls
are mocked (via requests.Session, patched at construction time) - these
never hit space-track.org for real. See that module's docstring for the
honest caveat: these tests prove the client behaves the way we believe
Space-Track's real API does, not that it's been confirmed against a real
account.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.spacetrack_client import SpaceTrackAuthError, SpaceTrackClient


def _mock_session(login_body=None, login_status_ok=True, get_json=None, get_text=None):
    session = MagicMock()

    login_response = MagicMock()
    login_response.raise_for_status = MagicMock()
    login_response.json.return_value = login_body if login_body is not None else {}
    session.post.return_value = login_response

    get_response = MagicMock()
    get_response.raise_for_status = MagicMock()
    if get_json is not None:
        get_response.json.return_value = get_json
    if get_text is not None:
        get_response.text = get_text
    session.get.return_value = get_response

    return session


def test_construction_without_credentials_raises_immediately():
    with pytest.raises(SpaceTrackAuthError):
        SpaceTrackClient(username="", password="")
    with pytest.raises(SpaceTrackAuthError):
        SpaceTrackClient(username="user", password="")


@patch("src.ingestion.spacetrack_client.requests.Session")
def test_query_json_authenticates_then_reuses_the_session(mock_session_cls):
    session = _mock_session(login_body={"foo": "bar"}, get_json=[{"NORAD_CAT_ID": "25544"}])
    mock_session_cls.return_value = session
    client = SpaceTrackClient(username="user", password="pw")

    rows = client.query_json("class/gp/NORAD_CAT_ID/25544/format/json")

    assert rows == [{"NORAD_CAT_ID": "25544"}]
    session.post.assert_called_once()
    assert session.post.call_args.kwargs["data"] == {"identity": "user", "password": "pw"}

    client.query_json("class/gp/NORAD_CAT_ID/25544/format/json")
    assert session.post.call_count == 1, "second query should reuse the authenticated session, not re-login"
    assert session.get.call_count == 2


@patch("src.ingestion.spacetrack_client.requests.Session")
def test_failed_login_raises_spacetrack_auth_error(mock_session_cls):
    session = _mock_session(login_body={"Login": "Failed"})
    mock_session_cls.return_value = session
    client = SpaceTrackClient(username="user", password="wrong")

    with pytest.raises(SpaceTrackAuthError):
        client.query_json("class/gp/format/json")


@patch("src.ingestion.spacetrack_client.requests.Session")
def test_query_text_returns_raw_response_body(mock_session_cls):
    session = _mock_session(login_body={}, get_text="RAW TLE TEXT\n")
    mock_session_cls.return_value = session
    client = SpaceTrackClient(username="user", password="pw")

    text = client.query_text("class/gp/format/tle")

    assert text == "RAW TLE TEXT\n"


@patch("src.ingestion.spacetrack_client.time.sleep")
@patch("src.ingestion.spacetrack_client.requests.Session")
def test_consecutive_queries_are_throttled(mock_session_cls, mock_sleep):
    session = _mock_session(login_body={}, get_text="x")
    mock_session_cls.return_value = session
    client = SpaceTrackClient(username="user", password="pw")

    client.query_text("class/gp/format/tle")
    client.query_text("class/gp/format/tle")

    mock_sleep.assert_called_once()


@patch("src.ingestion.spacetrack_client.requests.Session")
def test_fetch_historical_tle_text_builds_gp_history_query_with_epoch_filter(mock_session_cls):
    session = _mock_session(login_body={}, get_text="HISTORICAL TLE\n")
    mock_session_cls.return_value = session
    client = SpaceTrackClient(username="user", password="pw")

    text = client.fetch_historical_tle_text("24946", "2009-02-10T16:56:00+00:00")

    assert text == "HISTORICAL TLE\n"
    called_url = session.get.call_args.args[0]
    assert "class/gp_history/NORAD_CAT_ID/24946/" in called_url
    assert "EPOCH/%3C2009-02-10T16:56:00+00:00" in called_url
    assert "orderby/EPOCH%20desc" in called_url
    assert "limit/1" in called_url
    assert "format/tle" in called_url


@patch("src.ingestion.spacetrack_client.requests.Session")
def test_fetch_recent_cdms_makes_exactly_one_bounded_query(mock_session_cls):
    # cdm_public - not the full cdm class, which a real live query
    # confirmed this project's account cannot access (see
    # src/pc_severity.py's module docstring).
    session = _mock_session(login_body={}, get_json=[{"PC": "1e-5"}])
    mock_session_cls.return_value = session
    client = SpaceTrackClient(username="user", password="pw")

    rows = client.fetch_recent_cdms(limit=5)

    assert rows == [{"PC": "1e-5"}]
    assert session.get.call_count == 1
    called_url = session.get.call_args.args[0]
    assert "class/cdm_public/" in called_url
    assert "limit/5" in called_url
    assert "format/json" in called_url
