"""
Retry wrapper for the mock Workday/ServiceNow API calls.

Built on `tenacity` rather than hand-rolled retry loops: this is a place
where using a library is teaching the concept ("real external API calls
need retries with backoff"), not hiding it -- the retry POLICY (3 attempts,
exponential backoff, which exceptions count as retryable) is stated plainly
right here in one function.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("agentic_rag.agents.retry")

T = TypeVar("T")

RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)


def with_retry(fn: Callable[..., T], *args, max_attempts: int = 3, **kwargs) -> T:
    """Calls fn(*args, **kwargs) with up to `max_attempts` tries, using
    exponential backoff (0.5s, 1s, 2s, ...) between attempts, only for
    connection/timeout-style errors -- a 404 or a validation error is not
    retried, since retrying won't fix those."""

    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def _call() -> T:
        return fn(*args, **kwargs)

    try:
        return _call()
    except RETRYABLE_EXCEPTIONS as exc:
        logger.error("call to %s failed after %d attempts: %s", getattr(fn, "__name__", fn), max_attempts, exc)
        raise
