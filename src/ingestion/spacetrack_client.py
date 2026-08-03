"""Thin real HTTP client for Space-Track.org's REST API - session-based
auth plus rate-limit-aware query helpers. No Space-Track-specific
*business* logic here (which class/fields to query, how to turn a
response into a TelemetryEvent) - that lives in tle_source.py's
fetch_spacetrack_group_text and spacetrack_adapter.py. Kept separate the
same way GemmaClient is separate from pipeline.py's prompt-building.

Space-Track requires a real free account
(https://www.space-track.org/auth/createAccount) - registration is
per-person and can't be automated on anyone's behalf, unlike CelesTrak's
fully public gp.php endpoint. SPACETRACK_USERNAME/SPACETRACK_PASSWORD are
both empty by default (see src/config.py) - constructing this client
without them raises SpaceTrackAuthError immediately rather than making a
doomed request.

Endpoint shapes, class names, and predicate operators below come from
Space-Track's own published API documentation and CDM CSV schema
(fetched and cross-checked directly, not guessed) - but this client has
NOT yet been exercised against a real, live Space-Track session (no
account existed at the time this was written). Every method here is
unit-tested against a fake `requests.Session`, matching this project's
established mocked-HTTP-test convention (see test_celestrak_adapter.py) -
but per this project's own stated discipline (README/PROJECT_OVERVIEW:
"live-verified, not just built"), that proves the code does what it
believes Space-Track's API does, not that Space-Track's API actually
behaves this way. See ROADMAP_TO_PRODUCT.md's Phase 1 progress-log entry
for when real-session verification actually happens.
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import requests

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
BASE_QUERY_URL = "https://www.space-track.org/basicspacedata/query"

# Space-Track's own documented account-level limits: "less than 30
# requests per 1 minute(s) and 300 requests per 1 hour(s)" (confirmed
# directly against their published API documentation). Throttling to one
# request every 3 seconds keeps every real call comfortably under the
# per-minute limit - repeated account-suspension risk from a naive retry
# loop is exactly the kind of real-world failure mode this project's own
# GemmaClient circuit breaker was built to avoid for Gemma calls; the fix
# here is simpler (Space-Track has no equivalent to Gemma's "try anyway,
# might work" fallback backend) so a flat minimum interval is sufficient
# for the per-minute half of the limit. The per-hour half needs its own,
# separate rolling-window check below (MAX_REQUESTS_PER_HOUR) - a flat
# 3-second interval alone permits ~1200 requests/hour if sustained, 4x
# over the documented per-hour cap.
MIN_SECONDS_BETWEEN_REQUESTS = 3.0
MAX_REQUESTS_PER_HOUR = 300
ONE_HOUR_S = 60 * 60


class SpaceTrackAuthError(Exception):
    """Raised when Space-Track login fails, or credentials aren't configured."""


class SpaceTrackClient:
    """One authenticated session against Space-Track.org, meant to be
    constructed once and reused across every call an adapter makes - not
    re-authenticated per request. (This project already learned this
    lesson the hard way for GemmaClient: see PHASE_PROGRESS.md's QA-pass
    fix #1, where a fresh client per call site silently defeated the
    circuit breaker. Applying it here from the start rather than waiting
    to rediscover it.)
    """

    def __init__(self, username: str, password: str):
        if not username or not password:
            raise SpaceTrackAuthError(
                "SPACETRACK_USERNAME/SPACETRACK_PASSWORD are not configured. "
                "Register a free account at "
                "https://www.space-track.org/auth/createAccount and set both "
                "in .env."
            )
        self.username = username
        self.password = password
        self._session: Optional[requests.Session] = None
        self._last_request_time: float = 0.0
        # Timestamps (time.monotonic()) of every real request within the
        # last rolling hour - see _throttle's per-hour check below.
        self._request_times: deque[float] = deque()

    def _authenticate(self) -> requests.Session:
        session = requests.Session()
        response = session.post(
            LOGIN_URL,
            data={"identity": self.username, "password": self.password},
            timeout=30,
        )
        response.raise_for_status()
        # A failed login still returns HTTP 200, with {"Login": "Failed"}
        # in the body - has to be checked explicitly, raise_for_status()
        # alone would not catch bad credentials.
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("Login") == "Failed":
            raise SpaceTrackAuthError(
                "Space-Track login failed - check SPACETRACK_USERNAME/SPACETRACK_PASSWORD."
            )
        return session

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

        # Real per-hour cap - MIN_SECONDS_BETWEEN_REQUESTS above only
        # bounds the per-minute rate. Prunes request timestamps older
        # than an hour, then sleeps until the oldest one ages out if
        # already at MAX_REQUESTS_PER_HOUR, rather than letting a
        # sustained cadence silently run 4x over Space-Track's own
        # documented per-hour limit.
        now = time.monotonic()
        while self._request_times and now - self._request_times[0] > ONE_HOUR_S:
            self._request_times.popleft()
        if len(self._request_times) >= MAX_REQUESTS_PER_HOUR:
            sleep_for = ONE_HOUR_S - (now - self._request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _get(self, path: str) -> requests.Response:
        """path is everything after `.../query/` - e.g.
        "class/gp/NORAD_CAT_ID/25544/format/json". Authenticates on first
        use, throttles every real request thereafter."""
        if self._session is None:
            self._session = self._authenticate()
        self._throttle()
        url = f"{BASE_QUERY_URL}/{path.lstrip('/')}"
        response = self._session.get(url, timeout=30)
        now = time.monotonic()
        self._last_request_time = now
        self._request_times.append(now)

        if response.status_code in (401, 403):
            # A real, if previously unhandled, failure mode: this client
            # is documented to be constructed once and reused across
            # every call an adapter makes (see class docstring), so any
            # longer-lived process eventually hits a real expired/
            # invalidated session. Re-authenticate once and retry the
            # same request - not looped, so a genuinely bad credential
            # still fails loudly via _authenticate's own check rather
            # than retrying forever.
            self._session = self._authenticate()
            self._throttle()
            response = self._session.get(url, timeout=30)
            now = time.monotonic()
            self._last_request_time = now
            self._request_times.append(now)

        response.raise_for_status()
        return response

    def query_json(self, path: str) -> list[dict]:
        """For format/json queries - returns the parsed row list."""
        return self._get(path).json()

    def query_text(self, path: str) -> str:
        """For format/tle (or any other plain-text format) queries -
        returns the raw response body, e.g. directly compatible with
        tle_source.parse_tle_blocks."""
        return self._get(path).text

    def fetch_historical_tle_text(self, norad_id: str, at_or_before: str) -> str:
        """Real historical TLE retrieval - the capability CelesTrak's
        public gp.php endpoint structurally cannot provide (confirmed
        directly: it ignores any EPOCH filter and always serves the
        CURRENT element set - see historical_adapter.py's module
        docstring). Queries the `gp_history` class (Space-Track's
        documented archive of every historical elset, as opposed to `gp`,
        which only ever holds each object's single latest one) for the
        most recent element set AT OR BEFORE `at_or_before` (an ISO 8601
        timestamp) - the physically correct choice for propagating
        FORWARD toward a historical event, not a TLE from after it.

        Returns raw TLE block text (empty string if nothing is found for
        that object/epoch combination - e.g. before the object was first
        tracked). Not yet wired into HistoricalReplayAdapter - this is the
        underlying capability Phase 1 unblocks; using it to let a replay
        genuinely re-propagate instead of reusing the documented
        closest-approach numbers directly is separate follow-up work, so
        this change is independently reviewable.
        """
        # Space-Track's predicate operators are strict ("<"/">", no
        # documented "<="), so a literal "%3C{at_or_before}" excludes an
        # element set whose epoch exactly equals at_or_before - a real
        # mismatch against this method's own "AT OR BEFORE" docstring.
        # Nudging the bound one second later makes the strict "<" query
        # genuinely inclusive of that exact instant, without pulling in
        # anything meaningfully after it.
        try:
            query_bound = (datetime.fromisoformat(at_or_before) + timedelta(seconds=1)).isoformat()
        except ValueError:
            query_bound = at_or_before
        path = (
            f"class/gp_history/NORAD_CAT_ID/{norad_id}/"
            f"EPOCH/%3C{query_bound}/orderby/EPOCH%20desc/limit/1/format/tle"
        )
        return self.query_text(path)

    def fetch_recent_cdms(self, limit: int = 10) -> list[dict]:
        """Real Conjunction Data Messages, from Space-Track's `cdm_public`
        class specifically - see src/pc_severity.py's module docstring
        for the real finding this corrected: the full `cdm` class
        (higher-detail, includes covariance) genuinely requires
        registered owner/operator privileges this project doesn't have -
        confirmed live by a real `{"error":"Your Class Does Not Exist "}`
        400 response - but `cdm_public` (lower-detail: SAT_1_ID/SAT_2_ID/
        PC/TCA/MIN_RNG, no covariance) is real, current, and accessible
        to any registered account. Confirmed live with a real recent row
        (two real objects, a real PC of 0.0016).

        Deliberately ONE bounded query, not one per object pair - even
        though `cdm_public`'s own rate limit isn't separately documented
        (only the full `cdm` class's tighter limit - 3/day, 1/hour per
        event - is), querying once and matching locally against
        already-screened conjunctions (see src/ingestion/cdm_enrichment.py)
        is the conservative, correct design regardless of the exact
        number.
        """
        path = f"class/cdm_public/orderby/CREATED%20desc/limit/{limit}/format/json"
        return self.query_json(path)
