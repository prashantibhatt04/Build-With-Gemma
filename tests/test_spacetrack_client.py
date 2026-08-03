"""Tests for src/ingestion/spacetrack_client.py. All real network calls
are mocked (via requests.Session, patched at construction time) - these
never hit space-track.org for real. See that module's docstring for the
honest caveat: these tests prove the client behaves the way we believe
Space-Track's real API does, not that it's been confirmed against a real
account.
"""
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.spacetrack_client import (
    MAX_REQUESTS_PER_HOUR,
    MIN_SECONDS_BETWEEN_REQUESTS,
    SpaceTrackAuthError,
    SpaceTrackClient,
)


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


@patch("src.ingestion.spacetrack_client.time.sleep")
@patch("src.ingestion.spacetrack_client.requests.Session")
def test_get_reauthenticates_once_on_a_401_and_retries(mock_session_cls, mock_sleep):
    """Real bug this closes: this client is documented to be constructed
    once and reused across every call an adapter makes - a longer-lived
    process WILL eventually hit a real expired/invalidated session, which
    previously had no recovery path at all (a raw HTTPError). Two
    DIFFERENT session mocks stand in for the two real requests.Session()
    instances _authenticate() constructs (the original login, then the
    real re-login after the 401)."""
    first_session = MagicMock()
    first_session.post.return_value = MagicMock(raise_for_status=MagicMock(), json=lambda: {})
    first_session.get.return_value = MagicMock(status_code=401, raise_for_status=MagicMock())

    second_session = MagicMock()
    second_session.post.return_value = MagicMock(raise_for_status=MagicMock(), json=lambda: {})
    success = MagicMock(status_code=200, raise_for_status=MagicMock())
    success.json.return_value = [{"NORAD_CAT_ID": "25544"}]
    second_session.get.return_value = success

    mock_session_cls.side_effect = [first_session, second_session]
    client = SpaceTrackClient(username="user", password="pw")

    result = client.query_json("class/gp/NORAD_CAT_ID/25544/format/json")

    assert result == [{"NORAD_CAT_ID": "25544"}]
    assert mock_session_cls.call_count == 2, "re-authenticated exactly once, not looped"
    assert first_session.get.call_count == 1
    assert second_session.get.call_count == 1


@patch("src.ingestion.spacetrack_client.time.sleep")
@patch("src.ingestion.spacetrack_client.requests.Session")
def test_get_does_not_loop_forever_when_reauthentication_itself_fails(mock_session_cls, mock_sleep):
    """The retry-once design must not mask a genuinely dead credential -
    if the SECOND login also fails, that's a real SpaceTrackAuthError the
    caller needs to see, not a second silent retry attempt."""
    first_session = MagicMock()
    first_session.post.return_value = MagicMock(raise_for_status=MagicMock(), json=lambda: {})
    first_session.get.return_value = MagicMock(status_code=401, raise_for_status=MagicMock())

    second_session = MagicMock()
    second_session.post.return_value = MagicMock(raise_for_status=MagicMock(), json=lambda: {"Login": "Failed"})

    mock_session_cls.side_effect = [first_session, second_session]
    client = SpaceTrackClient(username="user", password="pw")

    with pytest.raises(SpaceTrackAuthError):
        client.query_json("class/gp/NORAD_CAT_ID/25544/format/json")

    assert mock_session_cls.call_count == 2  # tried exactly once to recover, then gave up


@patch("src.ingestion.spacetrack_client.time.monotonic")
@patch("src.ingestion.spacetrack_client.time.sleep")
@patch("src.ingestion.spacetrack_client.requests.Session")
def test_throttle_sleeps_for_the_real_per_hour_cap_once_it_is_reached(mock_session_cls, mock_sleep, mock_monotonic):
    """Real bug this closes: MIN_SECONDS_BETWEEN_REQUESTS alone only
    bounds the per-minute rate - sustained at that cadence a long-running
    process issues ~1200 requests/hour, 4x over Space-Track's own
    documented 300/hour cap, with nothing here ever tracking or
    enforcing it. Simulates MAX_REQUESTS_PER_HOUR requests already made
    "just now" (real time.monotonic() is patched, not real sleeping)."""
    session = _mock_session(login_body={}, get_json=[])
    mock_session_cls.return_value = session
    mock_monotonic.return_value = 100_000.0
    client = SpaceTrackClient(username="user", password="pw")
    client._last_request_time = 100_000.0 - MIN_SECONDS_BETWEEN_REQUESTS  # skip the per-minute sleep
    client._request_times = deque([100_000.0] * MAX_REQUESTS_PER_HOUR)

    client.query_json("class/gp/NORAD_CAT_ID/25544/format/json")

    # A real sleep was requested specifically for the per-hour cap - much
    # larger than the few-second per-minute interval, so this can't be
    # satisfied by MIN_SECONDS_BETWEEN_REQUESTS's own sleep alone.
    assert any(call.args[0] > 60 for call in mock_sleep.call_args_list)


@patch("src.ingestion.spacetrack_client.time.monotonic")
@patch("src.ingestion.spacetrack_client.time.sleep")
@patch("src.ingestion.spacetrack_client.requests.Session")
def test_throttle_prunes_request_times_older_than_an_hour(mock_session_cls, mock_sleep, mock_monotonic):
    """A request from over an hour ago must not still count toward the
    cap - otherwise the per-hour window would never actually roll
    forward and this client would eventually refuse to ever make another
    real request again."""
    session = _mock_session(login_body={}, get_json=[])
    mock_session_cls.return_value = session
    mock_monotonic.return_value = 100_000.0
    client = SpaceTrackClient(username="user", password="pw")
    client._last_request_time = 100_000.0 - MIN_SECONDS_BETWEEN_REQUESTS
    # All MAX_REQUESTS_PER_HOUR "requests" are well over an hour stale.
    client._request_times = deque([100_000.0 - 7200.0] * MAX_REQUESTS_PER_HOUR)

    client.query_json("class/gp/NORAD_CAT_ID/25544/format/json")

    assert not any(call.args[0] > 60 for call in mock_sleep.call_args_list)


@patch("src.ingestion.spacetrack_client.requests.Session")
def test_fetch_historical_tle_text_builds_gp_history_query_with_epoch_filter(mock_session_cls):
    session = _mock_session(login_body={}, get_text="HISTORICAL TLE\n")
    mock_session_cls.return_value = session
    client = SpaceTrackClient(username="user", password="pw")

    text = client.fetch_historical_tle_text("24946", "2009-02-10T16:56:00+00:00")

    assert text == "HISTORICAL TLE\n"
    called_url = session.get.call_args.args[0]
    assert "class/gp_history/NORAD_CAT_ID/24946/" in called_url
    # The strict "<" bound is nudged one second past at_or_before, so an
    # element set whose epoch exactly equals it is genuinely included -
    # matching this method's own "AT OR BEFORE" docstring (Space-Track's
    # predicate operators are strict, with no documented "<=").
    assert "EPOCH/%3C2009-02-10T16:56:01+00:00" in called_url
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
