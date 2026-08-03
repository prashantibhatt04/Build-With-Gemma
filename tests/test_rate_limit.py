"""Tests for src/rate_limit.py - pure logic, a fake clock for determinism
(no real time.sleep needed to prove refill behavior)."""
import pytest

from src.rate_limit import PRUNE_INTERVAL_CALLS, RateLimiter


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_allows_requests_up_to_capacity():
    clock = _FakeClock()
    limiter = RateLimiter(capacity=3, refill_per_second=1.0, now=clock)

    results = [limiter.allow("client-a")[0] for _ in range(3)]

    assert results == [True, True, True]


def test_rejects_once_capacity_is_exhausted():
    clock = _FakeClock()
    limiter = RateLimiter(capacity=2, refill_per_second=1.0, now=clock)
    limiter.allow("client-a")
    limiter.allow("client-a")

    allowed, retry_after = limiter.allow("client-a")

    assert allowed is False
    assert retry_after > 0


def test_refills_over_real_elapsed_time():
    clock = _FakeClock()
    limiter = RateLimiter(capacity=1, refill_per_second=1.0, now=clock)
    limiter.allow("client-a")
    assert limiter.allow("client-a")[0] is False

    clock.advance(1.0)  # exactly one token's worth of refill

    assert limiter.allow("client-a")[0] is True


def test_partial_refill_is_not_enough_for_a_full_token():
    clock = _FakeClock()
    limiter = RateLimiter(capacity=1, refill_per_second=1.0, now=clock)
    limiter.allow("client-a")

    clock.advance(0.5)

    assert limiter.allow("client-a")[0] is False


def test_different_clients_have_independent_buckets():
    clock = _FakeClock()
    limiter = RateLimiter(capacity=1, refill_per_second=1.0, now=clock)
    limiter.allow("client-a")

    assert limiter.allow("client-a")[0] is False
    assert limiter.allow("client-b")[0] is True


def test_retry_after_reflects_real_remaining_wait_time():
    clock = _FakeClock()
    limiter = RateLimiter(capacity=1, refill_per_second=2.0, now=clock)  # 2 tokens/sec
    limiter.allow("client-a")

    _, retry_after = limiter.allow("client-a")

    assert retry_after == pytest.approx(0.5)  # need 1 full token at 2/sec


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        RateLimiter(capacity=0, refill_per_second=1.0)


def test_prunes_a_fully_refilled_idle_bucket_after_enough_calls():
    """Real bug this closes: _buckets grew without bound for the life of
    the process - every distinct client_id (a raw source IP for any
    unauthenticated caller) permanently added an entry, even long after
    that client stopped sending real requests. One client makes a single
    real request (leaving its bucket not-full), then goes idle long
    enough to fully refill while PRUNE_INTERVAL_CALLS worth of OTHER
    traffic crosses the sweep threshold - its now-idle bucket must be
    evicted."""
    clock = _FakeClock()
    limiter = RateLimiter(capacity=5, refill_per_second=1.0, now=clock)

    limiter.allow("idle-client")
    clock.advance(100.0)  # plenty of time for idle-client's bucket to fully refill

    for i in range(PRUNE_INTERVAL_CALLS):
        limiter.allow(f"other-client-{i}")

    assert "idle-client" not in limiter._buckets


def test_does_not_prune_a_bucket_with_real_pending_consumption_state():
    """A bucket that ISN'T fully refilled holds real state (how close
    this client is to its own limit) that pruning must never discard -
    only idle, fully-refilled buckets are safe to evict."""
    clock = _FakeClock()
    limiter = RateLimiter(capacity=5, refill_per_second=1.0, now=clock)

    limiter.allow("active-client")  # tokens now 4/5 - not full, must survive any sweep

    for i in range(PRUNE_INTERVAL_CALLS):
        limiter.allow(f"other-client-{i}")

    assert "active-client" in limiter._buckets
