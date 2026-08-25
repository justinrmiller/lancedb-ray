"""Tests for the pure planning arithmetic."""

from __future__ import annotations

import pytest
from lancedb_ray._plan import (
    OffsetRange,
    chunk_offsets,
    plan_offset_shards,
    split_arrow_table,
)


class TestPlanOffsetShards:
    def test_even_split(self) -> None:
        assert plan_offset_shards(100, 4) == [
            OffsetRange(0, 25),
            OffsetRange(25, 50),
            OffsetRange(50, 75),
            OffsetRange(75, 100),
        ]

    def test_uneven_split_puts_remainder_first(self) -> None:
        shards = plan_offset_shards(10, 3)
        assert shards == [OffsetRange(0, 4), OffsetRange(4, 7), OffsetRange(7, 10)]

    def test_empty_table_yields_no_shards(self) -> None:
        assert plan_offset_shards(0, 8) == []

    def test_parallelism_exceeding_rows_yields_one_row_each(self) -> None:
        shards = plan_offset_shards(3, 16)
        assert shards == [OffsetRange(0, 1), OffsetRange(1, 2), OffsetRange(2, 3)]

    def test_single_row(self) -> None:
        assert plan_offset_shards(1, 1) == [OffsetRange(0, 1)]

    @pytest.mark.parametrize("num_rows", [1, 2, 7, 99, 1000, 65_537])
    @pytest.mark.parametrize("parallelism", [1, 2, 3, 8, 64])
    def test_shards_exactly_cover_the_row_space(
        self, num_rows: int, parallelism: int
    ) -> None:
        shards = plan_offset_shards(num_rows, parallelism)

        assert sum(s.num_rows for s in shards) == num_rows
        assert all(s.num_rows > 0 for s in shards)
        assert shards[0].start == 0
        assert shards[-1].end == num_rows
        # Contiguous and non-overlapping.
        for previous, current in zip(shards, shards[1:], strict=False):
            assert previous.end == current.start

    def test_shard_sizes_differ_by_at_most_one(self) -> None:
        sizes = [s.num_rows for s in plan_offset_shards(1000, 7)]
        assert max(sizes) - min(sizes) <= 1

    def test_rejects_negative_rows(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            plan_offset_shards(-1, 4)

    @pytest.mark.parametrize("parallelism", [0, -3])
    def test_rejects_non_positive_parallelism(self, parallelism: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            plan_offset_shards(10, parallelism)


class TestChunkOffsets:
    def test_chunks_are_explicit_offsets(self) -> None:
        chunks = list(chunk_offsets(OffsetRange(0, 5), 2))
        assert chunks == [[0, 1], [2, 3], [4]]

    def test_offsets_are_absolute_not_relative(self) -> None:
        chunks = list(chunk_offsets(OffsetRange(100, 105), 2))
        assert chunks == [[100, 101], [102, 103], [104]]

    def test_exact_multiple_has_no_short_chunk(self) -> None:
        chunks = list(chunk_offsets(OffsetRange(0, 6), 3))
        assert chunks == [[0, 1, 2], [3, 4, 5]]

    def test_batch_larger_than_range_yields_one_chunk(self) -> None:
        assert list(chunk_offsets(OffsetRange(3, 5), 100)) == [[3, 4]]

    def test_empty_range_yields_nothing(self) -> None:
        assert list(chunk_offsets(OffsetRange(7, 7), 10)) == []

    def test_chunks_reconstruct_the_range(self) -> None:
        offsets = OffsetRange(13, 977)
        flattened = [o for chunk in chunk_offsets(offsets, 64) for o in chunk]
        assert flattened == list(range(13, 977))

    def test_is_lazy(self) -> None:
        # A huge range must not be materialised just to start iterating; this
        # is what keeps shard bounds cheap to ship to workers.
        iterator = chunk_offsets(OffsetRange(0, 10**9), 10)
        assert next(iterator) == list(range(10))

    def test_rejects_non_positive_batch_size(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            list(chunk_offsets(OffsetRange(0, 5), 0))


class TestSplitArrowTable:
    def test_splits_into_capped_pieces(self) -> None:
        assert split_arrow_table(10, 4) == [
            OffsetRange(0, 4),
            OffsetRange(4, 8),
            OffsetRange(8, 10),
        ]

    def test_no_split_needed(self) -> None:
        assert split_arrow_table(3, 10) == [OffsetRange(0, 3)]

    def test_zero_rows(self) -> None:
        assert split_arrow_table(0, 10) == []

    def test_pieces_cover_all_rows(self) -> None:
        pieces = split_arrow_table(1000, 128)
        assert sum(p.num_rows for p in pieces) == 1000
        assert all(p.num_rows <= 128 for p in pieces)

    def test_rejects_non_positive_max_rows(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            split_arrow_table(10, 0)

    def test_rejects_negative_rows(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            split_arrow_table(-1, 10)


def test_offset_range_num_rows() -> None:
    assert OffsetRange(5, 12).num_rows == 7
    assert OffsetRange(0, 0).num_rows == 0
