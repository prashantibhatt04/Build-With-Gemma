"""Real in-memory rate limiting for scripts/api.py - a post-Phase-6
improvement (ROADMAP_TO_PRODUCT.md).

A real token-bucket limiter, not a placeholder: each client gets
`capacity` tokens, refilled continuously at a steady rate, one token
consumed per request - the standard algorithm real APIs use, not a naive
fixed-window counter (which lets a client burst 2x its real limit right
at a window boundary).

Deliberately in-memory and per-process: this project's API runs as one
FastAPI process (see docker-compose.yml's `api` service - no horizontal
scaling configured), so a real distributed store (Redis, etc.) would be
real infrastructure this project doesn't otherwise need or run. Honest
limit of this choice, not hidden: if this API is ever scaled to multiple
processes or instances, this limiter's state does NOT synchronize across
them - each process enforces its own independent budget, which
effectively multiplies the real configured limit by instance count. A
real distributed limiter would be the correct fix at that point, not
something to fake here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    def __init__(self, capacity: int, refill_per_second: float, now: Callable[[], float] = time.monotonic):
        if capacity <= 0:
            raise ValueError("capacity must be positive - use None/disabled at the caller level for 'no limit'")
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._now = now
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, client_id: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). retry_after_seconds is
        0.0 when allowed=True, and the real real time until at least one
        token would be available otherwise - suitable for a Retry-After
        response header."""
        now = self._now()
        bucket = self._buckets.get(client_id)
        if bucket is None:
            bucket = _Bucket(tokens=float(self.capacity), last_refill=now)
            self._buckets[client_id] = bucket

        elapsed = max(0.0, now - bucket.last_refill)
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.last_refill = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0.0

        retry_after = (1.0 - bucket.tokens) / self.refill_per_second
        return False, retry_after
