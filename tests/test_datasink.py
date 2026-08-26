"""Unit tests for datasink batching, retry and error policy.

These drive the datasink directly rather than through Ray so that failure
injection is deterministic and no backoff time is actually spent.
"""

from __future__ import annotations

from typing import Any

import lancedb
import pyarrow as pa
import pytest
from lancedb_ray._retry import RetryPolicy
from lancedb_ray.connection import LanceDBConnectionSpec
from lancedb_ray.datasink import LanceDBDatasink, WriteStats

from _fakes import FlakyRemoteTable
from conftest import make_table


@pytest.fixture
def spec(remote_uri: str, remote_kwargs: dict[str, Any]) -> LanceDBConnectionSpec:
    return LanceDBConnectionSpec.create(remote_uri, **remote_kwargs)


@pytest.fixture
def seeded_spec(
    seeded_remote: tuple[str, pa.Table], remote_kwargs: dict[str, Any]
) -> LanceDBConnectionSpec:
    uri, _ = seeded_remote
    return LanceDBConnectionSpec.create(uri, **remote_kwargs)


def no_sleep_policy(**kwargs: Any) -> RetryPolicy:
    return RetryPolicy(initial_backoff_s=0.0, max_backoff_s=0.0, **kwargs)


def write_arrow(
    sink: LanceDBDatasink, handle: Any, arrow: pa.Table, stats: WriteStats
) -> None:
    """Drive one transaction directly, bypassing Ray's block plumbing."""
    sink._write_once(handle, arrow.schema, arrow.to_batches(), arrow.num_rows, stats)


class TestValidation:
    def test_rejects_bad_mode(self, spec: LanceDBConnectionSpec) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            LanceDBDatasink(spec, "items", mode="sideways")  # type: ignore[arg-type]

    def test_rejects_upsert_without_key(self, spec: LanceDBConnectionSpec) -> None:
        with pytest.raises(ValueError, match="requires 'on'"):
            LanceDBDatasink(spec, "items", mode="upsert")

    def test_rejects_key_without_upsert(self, spec: LanceDBConnectionSpec) -> None:
        with pytest.raises(ValueError, match="only meaningful"):
            LanceDBDatasink(spec, "items", mode="append", on="id")

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_bad_rows_per_transaction(
        self, spec: LanceDBConnectionSpec, value: int
    ) -> None:
        with pytest.raises(ValueError, match="rows_per_transaction must be positive"):
            LanceDBDatasink(spec, "items", rows_per_transaction=value)

    def test_rejects_bad_max_rows(self, spec: LanceDBConnectionSpec) -> None:
        with pytest.raises(ValueError, match="max_rows_per_request must be positive"):
            LanceDBDatasink(spec, "items", max_rows_per_request=0)

    def test_rejects_bad_error_policy(self, spec: LanceDBConnectionSpec) -> None:
        with pytest.raises(ValueError, match="on_batch_error must be"):
            LanceDBDatasink(spec, "items", on_batch_error="explode")  # type: ignore[arg-type]

    def test_string_key_is_normalised_to_a_list(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(spec, "items", mode="upsert", on="id")
        assert sink._on == ["id"]

    def test_name_includes_the_table(self, spec: LanceDBConnectionSpec) -> None:
        assert LanceDBDatasink(spec, "items").get_name() == "LanceDB(items)"

    def test_supports_distributed_writes(self, spec: LanceDBConnectionSpec) -> None:
        assert LanceDBDatasink(spec, "items").supports_distributed_writes

    def test_rows_per_transaction_is_exposed_to_ray(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        assert (
            LanceDBDatasink(spec, "items", rows_per_transaction=99).min_rows_per_write
            == 99
        )


class TestTableCreation:
    def test_create_mode_makes_the_table(self, spec: LanceDBConnectionSpec) -> None:
        sink = LanceDBDatasink(spec, "fresh", mode="create")
        sink.on_write_start(make_table(1).schema)
        assert (
            "fresh" in lancedb.connect(spec.uri, **spec.connect_kwargs()).table_names()
        )

    def test_create_mode_refuses_an_existing_table(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items", mode="create")
        with pytest.raises(ValueError, match="already exists"):
            sink.on_write_start(make_table(1).schema)

    def test_overwrite_empties_the_table(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items", mode="overwrite")
        sink.on_write_start(make_table(1).schema)
        db = lancedb.connect(seeded_spec.uri, **seeded_spec.connect_kwargs())
        assert db.open_table("items").count_rows() == 0

    def test_append_leaves_existing_rows_alone(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items", mode="append")
        sink.on_write_start(make_table(1).schema)
        db = lancedb.connect(seeded_spec.uri, **seeded_spec.connect_kwargs())
        assert db.open_table("items").count_rows() == 100

    def test_create_without_a_schema_is_a_clear_error(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(spec, "fresh", mode="create")
        with pytest.raises(ValueError, match="needs a schema"):
            sink.on_write_start(None)

    def test_append_to_a_missing_table_without_a_schema_errors(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(spec, "missing", mode="append")
        with pytest.raises(ValueError, match="no schema is available"):
            sink.on_write_start(None)

    def test_explicit_schema_overrides_rays(self, spec: LanceDBConnectionSpec) -> None:
        explicit = pa.schema([pa.field("only", pa.int64())])
        sink = LanceDBDatasink(spec, "fresh", mode="create", schema=explicit)
        sink.on_write_start(make_table(1).schema)
        db = lancedb.connect(spec.uri, **spec.connect_kwargs())
        assert db.open_table("fresh").schema.names == ["only"]


class TestBatching:
    def test_a_task_writes_one_transaction_regardless_of_block_count(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        """Ten incoming blocks must not become ten transactions.

        Each transaction is a new table version and at least one fragment, so
        per-block writes are what leave a table with thousands of tiny
        fragments and invite rate limiting on Cloud/Enterprise.
        """
        sink = LanceDBDatasink(seeded_spec, "items")
        blocks = [make_table(10, start=i * 10) for i in range(10)]

        stats = sink.write(iter(blocks), ctx=None)  # type: ignore[arg-type]

        assert stats.num_rows == 100
        assert stats.num_batches == 1

    def test_max_rows_per_request_trades_memory_for_transactions(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items", max_rows_per_request=25)
        stats = sink.write(iter([make_table(100)]), ctx=None)  # type: ignore[arg-type]

        assert stats.num_rows == 100
        assert stats.num_batches == 4

    def test_a_small_task_still_writes(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items")
        stats = sink.write(iter([make_table(7)]), ctx=None)  # type: ignore[arg-type]
        assert stats.num_rows == 7
        assert stats.num_batches == 1

    def test_rows_per_transaction_is_what_ray_bundles(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items", rows_per_transaction=4096)
        assert sink.min_rows_per_write == 4096

    def test_empty_blocks_are_skipped(self, seeded_spec: LanceDBConnectionSpec) -> None:
        sink = LanceDBDatasink(seeded_spec, "items")
        stats = sink.write(iter([make_table(0), make_table(0)]), ctx=None)  # type: ignore[arg-type]
        assert stats.num_rows == 0
        assert stats.num_batches == 0

    def test_no_blocks_at_all(self, seeded_spec: LanceDBConnectionSpec) -> None:
        sink = LanceDBDatasink(seeded_spec, "items")
        assert sink.write(iter([]), ctx=None).num_rows == 0  # type: ignore[arg-type]

    def test_transform_runs_before_writing(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        seen: list[int] = []

        def record(batch: pa.Table) -> pa.Table:
            seen.append(batch.num_rows)
            return batch

        sink = LanceDBDatasink(seeded_spec, "items", transform_fn=record)
        sink.write(iter([make_table(5), make_table(5)]), ctx=None)  # type: ignore[arg-type]
        assert seen == [5, 5]

    def test_transform_dropping_all_rows_writes_nothing(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(
            seeded_spec, "items", transform_fn=lambda b: b.slice(0, 0)
        )
        assert sink.write(iter([make_table(10)]), ctx=None).num_rows == 0  # type: ignore[arg-type]


class TestRetryAndErrorPolicy:
    def _flaky(self, spec: LanceDBConnectionSpec, failures: int) -> FlakyRemoteTable:
        inner = lancedb.connect(spec.uri, **spec.connect_kwargs())._inner.open_table(  # type: ignore[attr-defined]
            "items"
        )
        return FlakyRemoteTable(inner, "items", failures=failures)

    def test_transient_failures_are_retried(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        table = self._flaky(seeded_spec, failures=2)
        sink = LanceDBDatasink(seeded_spec, "items", retry_policy=no_sleep_policy())
        stats = WriteStats()

        write_arrow(sink, table, make_table(5), stats)

        assert stats.num_rows == 5
        assert table.attempts == 3, "expected two failures then a success"

    def test_persistent_failure_raises_by_default(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        table = self._flaky(seeded_spec, failures=99)
        sink = LanceDBDatasink(
            seeded_spec, "items", retry_policy=no_sleep_policy(max_attempts=3)
        )

        # Raising is the deliberate default: silently dropping batches would
        # let a write job report success on partial data.
        with pytest.raises(TimeoutError):
            write_arrow(sink, table, make_table(5), WriteStats())

    def test_skip_policy_drops_the_batch_and_records_it(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        table = self._flaky(seeded_spec, failures=99)
        sink = LanceDBDatasink(
            seeded_spec,
            "items",
            on_batch_error="skip",
            retry_policy=no_sleep_policy(max_attempts=2),
        )
        stats = WriteStats()

        write_arrow(sink, table, make_table(5), stats)

        assert stats.num_rows == 0
        assert stats.num_skipped_rows == 5

    def test_skip_policy_continues_with_later_chunks(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        """With a memory ceiling set, a failed chunk must not sink the rest."""
        table = self._flaky(seeded_spec, failures=1)
        sink = LanceDBDatasink(
            seeded_spec,
            "items",
            on_batch_error="skip",
            max_rows_per_request=5,
            retry_policy=no_sleep_policy(max_attempts=1),
        )
        stats = WriteStats()
        arrow = make_table(10)

        sink._write_in_chunks(table, arrow.schema, arrow.to_batches(), stats)

        # First 5-row transaction fails and is skipped; the second still lands.
        assert stats.num_skipped_rows == 5
        assert stats.num_rows == 5

    def test_permanent_errors_are_not_retried(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        inner = lancedb.connect(
            seeded_spec.uri, **seeded_spec.connect_kwargs()
        )._inner.open_table(  # type: ignore[attr-defined]
            "items"
        )
        table = FlakyRemoteTable(
            inner, "items", failures=99, error=ValueError("schema mismatch")
        )
        sink = LanceDBDatasink(
            seeded_spec, "items", retry_policy=no_sleep_policy(max_attempts=5)
        )

        with pytest.raises(ValueError, match="schema mismatch"):
            write_arrow(sink, table, make_table(5), WriteStats())
        assert table.attempts == 1


class TestWriteStats:
    def test_repr_is_informative(self) -> None:
        assert "num_rows=3" in repr(WriteStats(num_rows=3, num_batches=1))

    def test_completion_logs_totals(
        self, seeded_spec: LanceDBConnectionSpec, caplog: pytest.LogCaptureFixture
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items")
        result = type("R", (), {"write_returns": [WriteStats(10, 1), WriteStats(5, 1)]})
        with caplog.at_level("INFO", logger="lancedb_ray.datasink"):
            sink.on_write_complete(result)  # type: ignore[arg-type]
        assert "15 rows" in caplog.text

    def test_completion_warns_about_dropped_rows(
        self, seeded_spec: LanceDBConnectionSpec, caplog: pytest.LogCaptureFixture
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items")
        result = type(
            "R", (), {"write_returns": [WriteStats(10, 1, num_skipped_rows=4)]}
        )
        with caplog.at_level("WARNING", logger="lancedb_ray.datasink"):
            sink.on_write_complete(result)  # type: ignore[arg-type]
        assert "dropped" in caplog.text


class TestUpsertBuilderOptions:
    def _sink(self, spec: LanceDBConnectionSpec, **kwargs: Any) -> LanceDBDatasink:
        return LanceDBDatasink(spec, "items", mode="upsert", on="id", **kwargs)

    def test_delete_unmatched_rows_entirely(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        db = lancedb.connect(seeded_spec.uri, **seeded_spec.connect_kwargs())
        table = db.open_table("items")

        sink = self._sink(seeded_spec, when_not_matched_by_source_delete=True)
        write_arrow(sink, table, make_table(10), WriteStats())

        # Rows 10..99 had no match in the source and were removed.
        assert table.count_rows() == 10

    def test_delete_unmatched_rows_matching_a_condition(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        db = lancedb.connect(seeded_spec.uri, **seeded_spec.connect_kwargs())
        table = db.open_table("items")

        sink = self._sink(seeded_spec, when_not_matched_by_source_delete="id >= 50")
        write_arrow(sink, table, make_table(10), WriteStats())

        # Only unmatched rows satisfying the predicate are deleted.
        assert table.count_rows() == 50

    def test_delete_is_off_by_default(self, seeded_spec: LanceDBConnectionSpec) -> None:
        db = lancedb.connect(seeded_spec.uri, **seeded_spec.connect_kwargs())
        table = db.open_table("items")

        write_arrow(self._sink(seeded_spec), table, make_table(10), WriteStats())
        assert table.count_rows() == 100


def test_retry_predicate_covers_transient_and_commit_conflicts() -> None:
    from lancedb_ray.datasink import _retry_predicate

    assert _retry_predicate(TimeoutError("connection timed out"))
    assert _retry_predicate(RuntimeError("commit conflict detected"))
    assert not _retry_predicate(ValueError("schema mismatch"))


class RecordingTable:
    """Captures the kwargs a write actually passes to LanceDB."""

    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.readers: list[Any] = []

    def add(self, reader: Any, **kwargs: Any) -> None:
        self.readers.append(reader)
        self.add_calls.append(kwargs)


class TestWriteParallelism:
    """``write_parallelism`` controls how many parts the client uploads
    concurrently inside one transaction, so it has to reach ``add``."""

    def test_is_forwarded_when_set(self, seeded_spec: LanceDBConnectionSpec) -> None:
        table = RecordingTable()
        sink = LanceDBDatasink(seeded_spec, "items", write_parallelism=4)

        write_arrow(sink, table, make_table(5), WriteStats())

        assert table.add_calls == [{"write_parallelism": 4}]

    def test_is_omitted_when_unset(self, seeded_spec: LanceDBConnectionSpec) -> None:
        table = RecordingTable()
        sink = LanceDBDatasink(seeded_spec, "items")

        write_arrow(sink, table, make_table(5), WriteStats())

        # Absent rather than None, so LanceDB applies its own default.
        assert table.add_calls == [{}]

    def test_rows_stream_rather_than_materialise(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        table = RecordingTable()
        sink = LanceDBDatasink(seeded_spec, "items")

        write_arrow(sink, table, make_table(7), WriteStats())

        # A RecordBatchReader, not a materialised Table: that is what lets the
        # client stream the rows as multiple parts under one upload.
        assert isinstance(table.readers[0], pa.RecordBatchReader)
        assert table.readers[0].read_all().num_rows == 7


class TestSchemaAlignmentAcrossBlocks:
    """A task streams every block through one reader, which carries one schema.

    Ray does not promise blocks share a column *order* -- ``from_arrow`` over
    differently-ordered tables produces exactly that. Feeding a mis-ordered
    block to a reader declared with another schema reads its buffers against
    the wrong fields, surfacing as an opaque Arrow C data interface error that
    mentions buffers rather than columns.
    """

    def test_reordered_columns_are_realigned(
        self, seeded_spec: LanceDBConnectionSpec
    ) -> None:
        sink = LanceDBDatasink(seeded_spec, "items")
        first = make_table(5)
        reordered = first.select(list(reversed(first.schema.names)))
        assert reordered.schema.names != first.schema.names

        stats = sink.write(iter([first, reordered]), ctx=None)  # type: ignore[arg-type]

        assert stats.num_rows == 10
        assert stats.num_batches == 1

    def test_a_missing_column_is_reported_clearly(self) -> None:
        from lancedb_ray.datasink import _align_to_schema

        full = make_table(3)
        dropped = full.drop_columns(["label"])
        with pytest.raises(ValueError, match="missing column"):
            _align_to_schema(dropped, full.schema)

    def test_matching_order_is_passed_through_untouched(self) -> None:
        from lancedb_ray.datasink import _align_to_schema

        table = make_table(3)
        assert _align_to_schema(table, table.schema) is table
