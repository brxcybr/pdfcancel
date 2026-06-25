"""Shared retry helper for Mistral API calls.

Retries transient failures (connection errors, timeouts, HTTP 429/5xx)
with exponential backoff and jitter. No external dependencies.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

from rich.console import Console

console = Console()

T = TypeVar("T")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Exception class-name fragments that indicate transient transport problems
# (covers httpx.ConnectError / ConnectTimeout / ReadTimeout etc. without
# importing httpx directly).
_RETRYABLE_NAME_FRAGMENTS = ("Timeout", "Connect", "Connection")


def _status_code(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception."""
    for source in (exc, getattr(exc, "response", None), getattr(exc, "raw_response", None)):
        if source is None:
            continue
        code: Any = getattr(source, "status_code", None)
        if code is not None:
            try:
                return int(code)
            except (TypeError, ValueError):
                continue
    return None


def is_retryable(exc: Exception) -> bool:
    """True if the exception looks like a transient connection/HTTP error."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    status = _status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS or status >= 500
    name = type(exc).__name__
    return any(frag in name for frag in _RETRYABLE_NAME_FRAGMENTS)


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 2.0,
    jitter: float = 1.0,
    description: str = "Mistral API call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn(), retrying transient failures with exponential backoff.

    Delays: base_delay * 2^attempt + uniform(0, jitter).
    Non-retryable exceptions and the final failure propagate unchanged.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not is_retryable(exc):
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
            console.print(
                f"  [yellow]Retrying {description} after error "
                f"({type(exc).__name__}: {exc}) — attempt "
                f"{attempt + 2}/{attempts} in {delay:.1f}s[/yellow]"
            )
            sleep(delay)
    raise last_exc  # pragma: no cover — unreachable
