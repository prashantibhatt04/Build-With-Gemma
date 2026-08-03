"""Real operator authentication for the dashboard - ROADMAP_TO_PRODUCT.md
Phase 5 ("operational trust").

Before this, scripts/dashboard.py's "Operator name" was a free-text
field - anyone with the page open could type "alice" and approve or
reject a real maneuver, or mark a decision reviewed, as her. That's fine
for a local demo, but it's exactly the "open localhost page" gap real
operational trust requires closing: `approved_by`/`reviewed_by` should
identify who ACTUALLY did something, not what they happened to type.

OPERATOR_TOKENS (src/config.py) is a real, if simple, fix: a
comma-separated list of `name:token` pairs configured out-of-band (an
environment variable, not something a dashboard visitor can see or set).
Entering a valid token authenticates as the operator it maps to - no
password hashing/session/expiry machinery, deliberately: this project's
existing trust model (a single shared audit log, a small number of real
operators) doesn't call for building a general-purpose auth system, just
for closing the specific "anyone can claim to be anyone" gap. Empty (the
default) leaves the dashboard exactly as it was - unauthenticated, with a
visible warning rather than a silent gap (see scripts/dashboard.py).
"""
from __future__ import annotations

import hmac
from typing import Optional


def parse_operator_tokens(raw: str) -> dict[str, str]:
    """Parses "alice:tok1,bob:tok2" into {"alice": "tok1", "bob": "tok2"}.
    Empty/malformed entries are skipped, not raised on - a typo in one
    operator's config shouldn't take down every other operator's access,
    and this runs at dashboard startup where a raised exception would be
    a confusing way to surface a config mistake."""
    tokens: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name, _, token = entry.partition(":")
        name, token = name.strip(), token.strip()
        if name and token:
            tokens[name] = token
    return tokens


def authenticate(token: str, operator_tokens: dict[str, str]) -> Optional[str]:
    """Returns the operator name whose token matches, or None. Ties break
    by iteration order (first match wins) - only matters if two operators
    were ever misconfigured with the same token, which is itself a
    real configuration error worth someone noticing, not silently
    resolving one particular way.

    Uses hmac.compare_digest rather than == - a real, if narrow, hardening:
    a naive string comparison returns as soon as the first mismatched
    character is found, so how long it takes can leak how many leading
    characters of a guess were correct. compare_digest runs in
    length-dependent-only constant time instead. Cheap to do correctly
    from the start rather than a gap worth flagging later."""
    if not token:
        return None
    for name, expected_token in operator_tokens.items():
        if hmac.compare_digest(token, expected_token):
            return name
    return None
