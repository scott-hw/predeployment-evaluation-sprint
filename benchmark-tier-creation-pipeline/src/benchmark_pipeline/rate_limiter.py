"""
Async rate limiter for Anthropic API calls.

Features:
- Semaphore for max_concurrent_requests
- Token bucket for RPM (requests_per_minute)
- Exponential backoff with full jitter on 429 / 529
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterConfig:
    requests_per_minute: int = 50
    max_concurrent_requests: int = 5
    # Backoff config
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    backoff_multiplier: float = 2.0
    max_retries: int = 6


class TokenBucket:
    """Thread-safe token bucket for rate limiting."""

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate          # tokens added per second
        self._capacity = capacity  # max tokens
        self._tokens = capacity    # start full
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until `tokens` tokens are available."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # How long until we have enough?
                deficit = tokens - self._tokens
                wait = deficit / self._rate
                await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


class RateLimiter:
    """
    Combines a semaphore (concurrency) with a token bucket (RPM).

    Usage:
        limiter = RateLimiter(config)
        async with limiter.acquire():
            response = await api_call(...)
    """

    def __init__(self, config: RateLimiterConfig | None = None) -> None:
        self.config = config or RateLimiterConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        rate_per_second = self.config.requests_per_minute / 60.0
        self._bucket = TokenBucket(
            rate=rate_per_second,
            capacity=float(self.config.max_concurrent_requests),
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Async context manager that enforces both concurrency and RPM limits."""
        await self._bucket.acquire(1.0)
        async with self._semaphore:
            yield

    async def execute_with_backoff(self, coro_factory, retryable_status_codes=(429, 529)):
        """
        Execute an async coroutine with exponential backoff + full jitter.

        `coro_factory` is a zero-argument callable that returns a coroutine.
        Re-raises the last exception after max_retries.
        """
        cfg = self.config
        attempt = 0
        backoff = cfg.initial_backoff_s

        while True:
            try:
                async with self.acquire():
                    return await coro_factory()
            except Exception as exc:
                status = _extract_status(exc)
                if status not in retryable_status_codes or attempt >= cfg.max_retries:
                    raise

                attempt += 1
                jitter = random.uniform(0, backoff)
                wait = min(backoff + jitter, cfg.max_backoff_s)
                logger.warning(
                    "HTTP %s — retry %d/%d in %.1fs",
                    status,
                    attempt,
                    cfg.max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * cfg.backoff_multiplier, cfg.max_backoff_s)


def _extract_status(exc: Exception) -> int | None:
    """Try to pull an HTTP status code from an anthropic API exception."""
    # anthropic SDK raises anthropic.APIStatusError with .status_code
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    # Fallback: check the string representation
    msg = str(exc)
    for code in (429, 529):
        if str(code) in msg:
            return code
    return None
