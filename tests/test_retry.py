"""Tests for retry classification and backoff."""

from __future__ import annotations

import pytest
from lancedb_ray._retry import (
    RetryPolicy,
    call_with_retry,
    is_commit_conflict,
    is_transient,
)


class TestClassification:
    @pytest.mark.parametrize(
        "message",
        [
            "connection timed out",
            "Connection reset by peer",
            "503 Service Unavailable",
            "429 Too Many Requests",
            "Bad Gateway",
            "temporarily unavailable",
        ],
    )
    def test_transient_errors_are_recognised(self, message: str) -> None:
        assert is_transient(RuntimeError(message))

    @pytest.mark.parametrize(
        "message",
        [
            "schema mismatch: column 'id' not found",
            "invalid filter expression",
            "permission denied",
        ],
    )
    def test_permanent_errors_are_not_retried(self, message: str) -> None:
        assert not is_transient(RuntimeError(message))

    def test_classification_is_case_insensitive(self) -> None:
        assert is_transient(RuntimeError("CONNECTION RESET"))

    @pytest.mark.parametrize(
        "message",
        [
            "Commit conflict detected",
            "concurrent write to dataset",
            "version already exists",
        ],
    )
    def test_commit_conflicts_are_recognised(self, message: str) -> None:
        assert is_commit_conflict(RuntimeError(message))

    def test_commit_conflict_is_distinct_from_transient(self) -> None:
        error = RuntimeError("commit conflict detected")
        assert is_commit_conflict(error)
        assert not is_transient(error)


class TestCallWithRetry:
    def test_returns_immediately_on_success(self) -> None:
        calls: list[int] = []

        def fn() -> str:
            calls.append(1)
            return "value"

        assert call_with_retry(fn, RetryPolicy(), description="op") == "value"
        assert len(calls) == 1

    def test_retries_until_success(self) -> None:
        attempts: list[int] = []
        slept: list[float] = []

        def fn() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise TimeoutError("connection timed out")
            return "ok"

        result = call_with_retry(
            fn,
            RetryPolicy(max_attempts=5, initial_backoff_s=0.01),
            description="op",
            sleep=slept.append,
        )

        assert result == "ok"
        assert len(attempts) == 3
        assert len(slept) == 2

    def test_raises_after_exhausting_attempts(self) -> None:
        attempts: list[int] = []

        def fn() -> None:
            attempts.append(1)
            raise TimeoutError("connection timed out")

        with pytest.raises(TimeoutError):
            call_with_retry(
                fn,
                RetryPolicy(max_attempts=3, initial_backoff_s=0.0),
                description="op",
                sleep=lambda _: None,
            )

        assert len(attempts) == 3

    def test_does_not_retry_a_permanent_error(self) -> None:
        attempts: list[int] = []

        def fn() -> None:
            attempts.append(1)
            raise ValueError("schema mismatch")

        with pytest.raises(ValueError, match="schema mismatch"):
            call_with_retry(
                fn,
                RetryPolicy(max_attempts=5),
                description="op",
                sleep=lambda _: None,
            )

        # Failing fast on a permanent error is the point: retrying a schema
        # mismatch five times just wastes time and hides the real cause.
        assert len(attempts) == 1

    def test_max_attempts_of_one_disables_retrying(self) -> None:
        attempts: list[int] = []

        def fn() -> None:
            attempts.append(1)
            raise TimeoutError("connection timed out")

        with pytest.raises(TimeoutError):
            call_with_retry(
                fn, RetryPolicy(max_attempts=1), description="op", sleep=lambda _: None
            )
        assert len(attempts) == 1

    def test_custom_predicate_is_honoured(self) -> None:
        attempts: list[int] = []

        def fn() -> None:
            attempts.append(1)
            raise ValueError("retry me")

        with pytest.raises(ValueError):
            call_with_retry(
                fn,
                RetryPolicy(
                    max_attempts=3, initial_backoff_s=0.0, predicate=lambda _: True
                ),
                description="op",
                sleep=lambda _: None,
            )
        assert len(attempts) == 3


class TestRetryPolicy:
    def test_rejects_zero_attempts(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RetryPolicy(max_attempts=0)

    def test_backoff_grows_and_is_capped(self) -> None:
        policy = RetryPolicy(initial_backoff_s=1.0, max_backoff_s=4.0)
        # Full jitter means each value is a sample from [0, cap], so assert the
        # bound rather than an exact value.
        for attempt in range(1, 8):
            assert 0.0 <= policy.backoff_for(attempt) <= 4.0

    def test_backoff_respects_the_uncapped_schedule_early(self) -> None:
        policy = RetryPolicy(initial_backoff_s=1.0, max_backoff_s=1000.0)
        assert 0.0 <= policy.backoff_for(1) <= 1.0
        assert 0.0 <= policy.backoff_for(2) <= 2.0
        assert 0.0 <= policy.backoff_for(3) <= 4.0
