"""Token bucket rate limiter for LLM API calls.

Handles both proactive capacity tracking (rolling 60s window) and
reactive retry-with-backoff when the API returns a rate limit error.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0
_MAX_RETRIES = 10


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit error via status code or message."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    return "rate limit" in str(exc).lower()


T = TypeVar("T")


@dataclass
class _UsageEntry:
    """A single recorded API call within the rolling window."""

    timestamp: float
    tokens: int


class RateLimitTicket:
    """Returned by rate_limiter.request() — call report_usage() when done."""

    def __init__(self, limiter: TokenBucketRateLimiter, estimated_tokens: int) -> None:
        self._limiter = limiter
        self._estimated = estimated_tokens

    def report_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Report actual token usage, correcting the initial estimate."""
        actual = input_tokens + output_tokens
        delta = actual - self._estimated
        if delta != 0:
            self._limiter._adjust_tokens(delta)


class TokenBucketRateLimiter:
    """Async rate limiter that tracks a rolling 60-second window of usage.

    Two layers of protection:
    1. **Proactive**: ``call()`` waits until the rolling window has
       enough capacity before streaming.
    2. **Reactive**: ``call_with_retry()`` wraps ``call()`` with
       retry-with-backoff when the API returns a rate limit error.

    Usage::

        async for chunk in rate_limiter.call_with_retry(
            "agent_name", estimated_tokens,
            lambda: llm.generate_with_tools_streaming(system, messages, tools),
        ):
            process(chunk)
    """

    def __init__(
        self,
        max_tokens_per_minute: int = 200_000,
        max_requests_per_minute: int = 500,
    ) -> None:
        self.max_tpm = max_tokens_per_minute
        self.max_rpm = max_requests_per_minute
        self._entries: deque[_UsageEntry] = deque()
        self._request_timestamps: deque[float] = deque()
        self._tokens_in_window: int = 0
        self._requests_in_window: int = 0

    def _prune_window(self) -> None:
        """Remove entries older than the rolling window."""
        cutoff = time.monotonic() - _WINDOW_SECONDS
        while self._entries and self._entries[0].timestamp < cutoff:
            entry = self._entries.popleft()
            self._tokens_in_window -= entry.tokens
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
            self._requests_in_window -= 1

    def _adjust_tokens(self, delta: int) -> None:
        """Adjust token count when actual usage differs from estimate."""
        self._tokens_in_window += delta

    async def _wait_for_capacity(self, tokens: int) -> None:
        """Sleep until there's enough capacity in the rolling window."""
        while True:
            self._prune_window()
            tokens_ok = (self._tokens_in_window + tokens) <= self.max_tpm
            requests_ok = self._requests_in_window < self.max_rpm
            if tokens_ok and requests_ok:
                return

            # Calculate how long to wait for enough capacity to free up
            now = time.monotonic()
            wait = 1.0  # default
            if not tokens_ok and self._entries:
                needed = (self._tokens_in_window + tokens) - self.max_tpm
                freed = 0
                for entry in self._entries:
                    freed += entry.tokens
                    if freed >= needed:
                        wait = max(0.1, entry.timestamp + _WINDOW_SECONDS - now)
                        break
            if not requests_ok and self._request_timestamps:
                req_wait = max(
                    0.1,
                    self._request_timestamps[0] + _WINDOW_SECONDS - now,
                )
                wait = max(wait, req_wait)

            logger.warning(
                "Rate limiter: %d/%d TPM, %d/%d RPM — waiting %.1fs",
                self._tokens_in_window,
                self.max_tpm,
                self._requests_in_window,
                self.max_rpm,
                wait,
            )
            await asyncio.sleep(wait)

    def _reserve_capacity(self, estimated_tokens: int) -> RateLimitTicket:
        """Reserve capacity in the window and return a ticket."""
        now = time.monotonic()
        self._entries.append(_UsageEntry(timestamp=now, tokens=estimated_tokens))
        self._tokens_in_window += estimated_tokens
        self._request_timestamps.append(now)
        self._requests_in_window += 1
        return RateLimitTicket(self, estimated_tokens)

    def _on_rate_limit_error(self) -> None:
        """Bump token count to force a cooldown on the next request."""
        self._tokens_in_window = self.max_tpm
        logger.warning("Rate limit error received — forcing cooldown")

    async def call(
        self,
        caller_name: str,
        estimated_tokens: int,
        stream_factory: Callable[[], AsyncGenerator[T, None]],
    ) -> AsyncGenerator[T, None]:
        """Acquire capacity and stream chunks. No retry — raises on error."""
        await self._wait_for_capacity(estimated_tokens)
        ticket = self._reserve_capacity(estimated_tokens)
        last_usage: dict[str, Any] | None = None
        async for chunk in stream_factory():
            usage = getattr(chunk, "usage_raw", None)
            if usage:
                last_usage = usage
            yield chunk

        # Correct estimate with actual usage
        if last_usage:
            ticket.report_usage(
                last_usage.get("input_tokens", 0),
                last_usage.get("output_tokens", 0),
            )

    async def call_with_retry(
        self,
        caller_name: str,
        estimated_tokens: int,
        stream_factory: Callable[[], AsyncGenerator[T, None]],
    ) -> AsyncGenerator[T, None]:
        """Wrap ``call()`` with retry on rate-limit errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                async for chunk in self.call(caller_name, estimated_tokens, stream_factory):
                    yield chunk
                return  # success
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < _MAX_RETRIES - 1:
                    self._on_rate_limit_error()
                    delay = 2.0 * (2**attempt)
                    logger.warning(
                        "Rate limit hit for %s (attempt %d/%d), retrying in %.1fs",
                        caller_name,
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise


class NullRateLimiter:
    """No-op rate limiter — no proactive throttling but still retries on 429."""

    async def call(
        self,
        caller_name: str,
        estimated_tokens: int,
        stream_factory: Callable[[], AsyncGenerator[T, None]],
    ) -> AsyncGenerator[T, None]:
        """Pass-through — no throttling. Raises on error."""
        async for chunk in stream_factory():
            yield chunk

    async def call_with_retry(
        self,
        caller_name: str,
        estimated_tokens: int,
        stream_factory: Callable[[], AsyncGenerator[T, None]],
    ) -> AsyncGenerator[T, None]:
        """Wrap ``call()`` with retry on rate-limit errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                async for chunk in self.call(caller_name, estimated_tokens, stream_factory):
                    yield chunk
                return  # success
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < _MAX_RETRIES - 1:
                    delay = 2.0 * (2**attempt)
                    logger.warning(
                        "Rate limit hit for %s (attempt %d/%d), retrying in %.1fs",
                        caller_name,
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    def _adjust_tokens(self, delta: int) -> None:
        pass
