# SPDX-License-Identifier: Apache-2.0
"""Retry helpers for transient LanceDB failures.

Remote (Cloud/Enterprise) calls go over HTTP and fail transiently; local writes
contend on the dataset commit lock. Both are worth retrying with backoff, and
neither should be retried forever.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ["RetryPolicy", "call_with_retry", "is_commit_conflict", "is_transient"]

# Substrings identifying errors that are worth another attempt. Matched against
# the lowercased ``str()`` of the exception because the LanceDB Python SDK
# surfaces server-side failures as generic exception types with descriptive
# messages rather than a dedicated exception hierarchy.
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "broken pipe",
    "temporarily unavailable",
    "service unavailable",
    "too many requests",
    "internal server error",
    "bad gateway",
    "gateway timeout",
)


def _status_pattern(*codes: str) -> re.Pattern[str]:
    """Match an HTTP status code as a whole number, not as digits anywhere.

    A bare substring test reads ``429`` out of "dimension 1429" and ``503`` out
    of "row 8503", classifying a deterministic schema error as retryable. Word
    boundaries keep a code from matching inside a longer number, which is how
    row counts, dimensions and byte sizes reach these messages.
    """
    return re.compile(rf"\b(?:{'|'.join(codes)})\b")


_TRANSIENT_STATUS = _status_pattern("429", "502", "503", "504")

_COMMIT_CONFLICT_MARKERS = (
    "commit conflict",
    "concurrent",
    "version already exists",
    "retryable commit",
    "commit was rejected",
)


def is_transient(error: BaseException) -> bool:
    """Return whether ``error`` looks like a retryable transient failure."""
    message = str(error).lower()
    if any(marker in message for marker in _TRANSIENT_MARKERS):
        return True
    return _TRANSIENT_STATUS.search(message) is not None


#: Failures that mean the request never reached the service, or was rejected
#: without being applied. Re-sending these cannot duplicate anything.
_NOT_APPLIED_MARKERS = (
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "too many requests",
    "name or service not known",
    "nodename nor servname",
    "failed to resolve",
)

_NOT_APPLIED_STATUS = _status_pattern("429", "503")


def is_definitely_not_applied(error: BaseException) -> bool:
    """Whether ``error`` proves the write never took effect.

    A read timeout or a dropped connection is ambiguous: the service may have
    committed and only the response was lost. Re-sending a non-idempotent
    append in that case silently duplicates rows, so appends retry only on
    this narrower class.
    """
    message = str(error).lower()
    if any(marker in message for marker in _NOT_APPLIED_MARKERS):
        return True
    return _NOT_APPLIED_STATUS.search(message) is not None


#: Arrow-rs refuses to import a buffer whose pointer is not aligned for its
#: scalar type. Ray hands out zero-copy views into its object store, and a
#: view can land unaligned for a type that needs more than 8 bytes, such as
#: decimal128. The import fails before anything is written.
_ALIGNMENT_MARKERS = ("is not aligned with the specified scalar type",)


def is_arrow_alignment_error(error: BaseException) -> bool:
    """Whether ``error`` is arrow-rs rejecting an unaligned FFI buffer.

    Recoverable by copying the batch into freshly allocated memory, and safe
    to retry: the import fails before any data is committed.
    """
    message = str(error)
    return any(marker in message for marker in _ALIGNMENT_MARKERS)


def is_commit_conflict(error: BaseException) -> bool:
    """Return whether ``error`` looks like a losing race on a dataset commit."""
    message = str(error).lower()
    return any(marker in message for marker in _COMMIT_CONFLICT_MARKERS)


class RetryPolicy:
    """Exponential backoff with full jitter.

    Args:
        max_attempts: Total attempts including the first. ``1`` disables retrying.
        initial_backoff_s: Backoff before the second attempt.
        max_backoff_s: Ceiling on the backoff between attempts.
        predicate: Returns whether a given exception should be retried.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        initial_backoff_s: float = 0.5,
        max_backoff_s: float = 32.0,
        predicate: Callable[[BaseException], bool] = is_transient,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self.max_attempts = max_attempts
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self.predicate = predicate

    def backoff_for(self, attempt: int) -> float:
        """Return the sleep, in seconds, before attempt number ``attempt`` (1-based)."""
        uncapped = self.initial_backoff_s * (2 ** (attempt - 1))
        return random.uniform(0.0, min(uncapped, self.max_backoff_s))


def call_with_retry[T](
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    description: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn``, retrying while ``policy.predicate`` accepts the raised error.

    Args:
        fn: Zero-argument callable to invoke.
        policy: Attempt count, backoff schedule and retry predicate.
        description: Human-readable operation name used in log messages.
        sleep: Injectable sleep, so tests do not spend real time backing off.

    Returns:
        Whatever ``fn`` returns on its first successful attempt.

    Raises:
        BaseException: The final error, once attempts are exhausted or the
            predicate rejects it.
    """
    last_error: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - re-raised below
            last_error = error
            if attempt == policy.max_attempts or not policy.predicate(error):
                raise
            backoff = policy.backoff_for(attempt)
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.2fs.",
                description,
                attempt,
                policy.max_attempts,
                error,
                backoff,
            )
            sleep(backoff)

    # Unreachable: the loop either returns or raises.
    raise AssertionError(f"retry loop exited without result: {last_error}")
