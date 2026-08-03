"""Tests for src/rate_limit.py - pure logic, a fake clock for determinism
(no real time.sleep needed to prove refill behavior)."""
import pytest

from src.rate_limit import RateLimiter


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
