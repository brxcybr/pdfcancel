"""Tests for the shared retry helper (retry.py)."""

from __future__ import annotations

import pytest

from pdfcancel.retry import is_retryable, with_retry


class FakeHttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_retries_connection_error_then_succeeds():
    attempts = []
    sleeps = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("network down")
        return "ok"

    result = with_retry(flaky, attempts=3, sleep=sleeps.append)
    assert result == "ok"
    assert len(attempts) == 3
    # Exponential backoff: 2*2^0 and 2*2^1, plus jitter in [0, 1)
    assert len(sleeps) == 2
    assert 2.0 <= sleeps[0] < 3.0
    assert 4.0 <= sleeps[1] < 5.0


def test_retries_http_429_and_5xx():
    for status in (429, 500, 502, 503, 504):
        calls = []

        def flaky(status=status):
            calls.append(1)
            if len(calls) == 1:
                raise FakeHttpError(status)
            return "ok"

        assert with_retry(flaky, sleep=lambda _d: None) == "ok"
        assert len(calls) == 2


def test_non_retryable_error_raises_immediately():
    calls = []

    def bad():
        calls.append(1)
        raise FakeHttpError(401)

    with pytest.raises(FakeHttpError):
        with_retry(bad, sleep=lambda _d: None)
    assert len(calls) == 1


def test_value_error_is_not_retried():
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        with_retry(bad, sleep=lambda _d: None)
    assert len(calls) == 1


def test_gives_up_after_max_attempts():
    calls = []

    def always_fails():
        calls.append(1)
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError):
        with_retry(always_fails, attempts=3, sleep=lambda _d: None)
    assert len(calls) == 3


def test_is_retryable_classification():
    assert is_retryable(ConnectionError())
    assert is_retryable(TimeoutError())
    assert is_retryable(FakeHttpError(429))
    assert is_retryable(FakeHttpError(503))
    assert not is_retryable(FakeHttpError(404))
    assert not is_retryable(ValueError("nope"))

    # Transport errors detected by class name (httpx-style)
    class ConnectTimeout(Exception):
        pass

    assert is_retryable(ConnectTimeout())

    # Status code on a nested response object
    class SDKError(Exception):
        def __init__(self):
            class _Resp:
                status_code = 502
            self.response = _Resp()

    assert is_retryable(SDKError())
