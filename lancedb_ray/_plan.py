# SPDX-License-Identifier: Apache-2.0
"""Pure planning helpers for distributed LanceDB reads and writes.

Nothing in this module touches Ray, the network, or the filesystem. Keeping the
planning arithmetic free of I/O is what makes it cheap to test exhaustively --
the interesting edge cases (empty tables, parallelism exceeding row counts,
uneven remainders) are all covered by ``tests/test_plan.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

__all__ = [
    "OffsetRange",
    "chunk_offsets",
    "plan_offset_shards",
    "split_arrow_table",
]


class OffsetRange(NamedTuple):
    """A half-open ``[start, end)`` range of row offsets within a table."""

    start: int
    end: int

    @property
    def num_rows(self) -> int:
        return self.end - self.start


def plan_offset_shards(num_rows: int, parallelism: int) -> list[OffsetRange]:
    """Split ``[0, num_rows)`` into at most ``parallelism`` contiguous shards.

    Rows are distributed as evenly as possible: the first ``num_rows %
    parallelism`` shards receive one extra row. Empty shards are never
    produced, so the result is empty when ``num_rows`` is zero and has
    ``num_rows`` single-row entries when ``parallelism`` exceeds ``num_rows``.

    Args:
        num_rows: Total number of rows in the table. Must not be negative.
        parallelism: Desired number of shards. Must be positive.

    Returns:
        Contiguous, non-overlapping ranges covering ``[0, num_rows)`` in order.
    """
    if num_rows < 0:
        raise ValueError(f"num_rows must not be negative, got {num_rows}")
    if parallelism <= 0:
        raise ValueError(f"parallelism must be positive, got {parallelism}")

    if num_rows == 0:
        return []

    num_shards = min(parallelism, num_rows)
    base, remainder = divmod(num_rows, num_shards)

    shards: list[OffsetRange] = []
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < remainder else 0)
        shards.append(OffsetRange(start, start + size))
        start += size

    # The loop above is exact arithmetic; this guards against future edits.
    assert start == num_rows, f"shards covered {start} of {num_rows} rows"
    return shards


def chunk_offsets(offsets: OffsetRange, batch_size: int) -> Iterator[list[int]]:
    """Materialise an offset range as batches of explicit row offsets.

    ``Table.take_offsets`` requires an explicit list of offsets rather than a
    range. For a large shard that list can be enormous, so it is generated
    lazily inside the worker -- only the two integers of the ``OffsetRange``
    ever cross the wire -- and yielded in ``batch_size`` chunks so no single
    request carries the whole shard.

    Args:
        offsets: The half-open range to expand.
        batch_size: Maximum number of offsets per yielded chunk. Must be positive.

    Yields:
        Lists of row offsets, each of length ``batch_size`` except possibly the last.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    for start in range(offsets.start, offsets.end, batch_size):
        end = min(start + batch_size, offsets.end)
        yield list(range(start, end))


def split_arrow_table(num_rows: int, max_rows: int) -> list[OffsetRange]:
    """Split ``num_rows`` into consecutive slices of at most ``max_rows``.

    Used by the write path to break an accumulated batch into request-sized
    pieces before handing them to LanceDB.
    """
    if num_rows < 0:
        raise ValueError(f"num_rows must not be negative, got {num_rows}")
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")

    return [
        OffsetRange(start, min(start + max_rows, num_rows))
        for start in range(0, num_rows, max_rows)
    ]
