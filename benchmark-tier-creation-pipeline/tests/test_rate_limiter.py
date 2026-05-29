"""
Tests for the async rate limiter.

Covers: semaphore concurrency, token bucket RPM limiting,
        acquire() context manager, backoff on retryable errors.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_pipeline.rate_limiter import (
    RateLimiter,
    RateLimiterConfig,
    TokenBucket,
    _extract_status,
)


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_acquire_with_full_bucket(self):
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        # Should return immediately
        start = time.monotonic()
        await bucket.acquire(1.0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # should be near-instant

    @pytest.mark.asyncio
    async def test_bucket_limits_fast_requests(self):
        # Very slow rate: 2 tokens/sec, capacity 2
        # First two should be instant, third should wait ~0.5s
        bucket = TokenBucket(rate=2.0, capacity=2.0)
        t0 = time.monotonic()
        await bucket.acquire(1.0)
        await bucket.acquire(1.0)
        # Bucket is now empty; next acquire should block ~0.5s
        await bucket.acquire(1.0)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.4  # must have waited

    @pytest.mark.asyncio
    async def test_bucket_does_not_exceed_capacity(self):
        """Tokens should not accumulate above capacity."""
        bucket = TokenBucket(rate=100.0, capacity=5.0)
        await asyncio.sleep(0.1)  # let bucket potentially over-refill
        # Should be able to acquire at most 5 tokens immediately
        bucket._refill()
        assert bucket._tokens <= 5.0


# ---------------------------------------------------------------------------
# RateLimiter acquire()
# ---------------------------------------------------------------------------

class TestRateLimiterAcquire:
    @pytest.mark.asyncio
    async def test_acquire_as_context_manager(self):
        rl = RateLimiter(RateLimiterConfig(max_concurrent_requests=3, requests_per_minute=60))
        async with rl.acquire():
            pass  # should not raise

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """With max_concurrent_requests=2, only 2 coroutines should run simultaneously."""
        rl = RateLimiter(RateLimiterConfig(max_concurrent_requests=2, requests_per_minute=120))
        in_flight = []
        max_in_flight = 0

        async def task():
            nonlocal max_in_flight
            async with rl.acquire():
                in_flight.append(1)
                max_in_flight = max(max_in_flight, len(in_flight))
                await asyncio.sleep(0.05)
                in_flight.pop()

        await asyncio.gather(*(task() for _ in range(5)))
        assert max_in_flight <= 2

    @pytest.mark.asyncio
    async def test_multiple_acquires_complete(self):
        rl = RateLimiter(RateLimiterConfig(max_concurrent_requests=5, requests_per_minute=300))
        results = []

        async def task(i):
            async with rl.acquire():
                results.append(i)

        await asyncio.gather(*(task(i) for i in range(10)))
        assert sorted(results) == list(range(10))


# ---------------------------------------------------------------------------
# execute_with_backoff
# ---------------------------------------------------------------------------

class TestExecuteWithBackoff:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        rl = RateLimiter(RateLimiterConfig(max_concurrent_requests=5, requests_per_minute=60))
        call_count = 0

        async def coro():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await rl.execute_with_backoff(coro)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        rl = RateLimiter(
            RateLimiterConfig(
                max_concurrent_requests=5,
                requests_per_minute=300,
                initial_backoff_s=0.01,
                max_backoff_s=0.05,
                max_retries=3,
            )
        )
        call_count = 0

        class Mock429Error(Exception):
            status_code = 429

        async def coro():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Mock429Error("Rate limited")
            return "ok"

        result = await rl.execute_with_backoff(coro)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        rl = RateLimiter(
            RateLimiterConfig(
                max_concurrent_requests=5,
                requests_per_minute=300,
                initial_backoff_s=0.01,
                max_backoff_s=0.05,
                max_retries=2,
            )
        )

        class Mock429Error(Exception):
            status_code = 429

        async def coro():
            raise Mock429Error("Always 429")

        with pytest.raises(Mock429Error):
            await rl.execute_with_backoff(coro)

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(self):
        rl = RateLimiter(RateLimiterConfig(max_retries=3, initial_backoff_s=0.01))
        call_count = 0

        async def coro():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await rl.execute_with_backoff(coro)
        assert call_count == 1  # No retries for non-429 errors


# ---------------------------------------------------------------------------
# _extract_status helper
# ---------------------------------------------------------------------------

class TestExtractStatus:
    def test_extract_from_status_code_attr(self):
        class Exc(Exception):
            status_code = 429
        assert _extract_status(Exc()) == 429

    def test_extract_529_from_attr(self):
        class Exc(Exception):
            status_code = 529
        assert _extract_status(Exc()) == 529

    def test_extract_from_string(self):
        exc = Exception("HTTP 429 Too Many Requests")
        assert _extract_status(exc) == 429

    def test_returns_none_for_unknown(self):
        assert _extract_status(ValueError("generic")) is None
