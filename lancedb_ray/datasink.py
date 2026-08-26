# SPDX-License-Identifier: Apache-2.0
"""Ray Data sink for LanceDB tables.

This sink drives writes through LanceDB's public table API (``add`` and
``merge_insert``), which is the only option for Cloud/Enterprise and the right
option locally whenever row-level upsert semantics are needed.

Local append-style writes take a different and much faster route -- workers
write Lance fragments in parallel and the driver commits them as a single
atomic transaction. That path lives in :func:`lancedb_ray.io.write_lancedb`,
which delegates to ``lance-ray``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import pyarrow as pa
from ray.data.block import Block, BlockAccessor
from ray.data.datasource import Datasink
from ray.data.datasource.datasink import WriteResult

from ._plan import split_arrow_table
from ._retry import RetryPolicy, call_with_retry, is_commit_conflict, is_transient
from .connection import LanceDBConnectionSpec, connect, list_table_names

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import TaskContext

logger = logging.getLogger(__name__)

__all__ = ["LanceDBDatasink", "WriteMode", "WriteStats", "validate_write_args"]

WriteMode = Literal["create", "append", "overwrite", "upsert"]
OnBatchError = Literal["raise", "skip"]

TransformFn = Callable[[pa.Table], pa.Table]

WRITE_MODES = ("create", "append", "overwrite", "upsert")


def validate_write_args(
    mode: str, on: Optional[Union[str, list[str]]]
) -> Optional[list[str]]:
    """Validate mode/key combinations and normalise ``on`` to a list.

    Shared by the datasink and by the fragment write path, which never builds a
    datasink and so needs the same checks applied independently.
    """
    if mode not in WRITE_MODES:
        raise ValueError(f"mode must be one of {WRITE_MODES}, got {mode!r}")
    if mode == "upsert" and not on:
        raise ValueError("mode='upsert' requires 'on' (the key column(s) to match)")
    if on is not None and mode != "upsert":
        raise ValueError(f"'on' is only meaningful for mode='upsert', not {mode!r}")
    if on is None:
        return None
    return [on] if isinstance(on, str) else list(on)


class WriteStats:
    """Per-task tally of what a write actually accomplished."""

    def __init__(
        self, num_rows: int = 0, num_batches: int = 0, num_skipped_rows: int = 0
    ) -> None:
        self.num_rows = num_rows
        self.num_batches = num_batches
        self.num_skipped_rows = num_skipped_rows

    def __repr__(self) -> str:
        return (
            f"WriteStats(num_rows={self.num_rows}, num_batches={self.num_batches}, "
            f"num_skipped_rows={self.num_skipped_rows})"
        )


def _retry_predicate(error: BaseException) -> bool:
    """Retry both remote flakiness and lost races on the local commit lock."""
    return is_transient(error) or is_commit_conflict(error)


class LanceDBDatasink(Datasink[WriteStats]):
    """Write a Ray :class:`~ray.data.Dataset` into a LanceDB table.

    Args:
        spec: How workers should connect to the database.
        table_name: Destination table.
        mode: ``create`` (must not exist), ``append``, ``overwrite`` (replace
            existing data), or ``upsert`` (merge on ``on``).
        on: Key column(s) to match on. Required when ``mode="upsert"``.
        schema: Arrow schema for table creation. Defaults to the schema Ray
            reports for the first input bundle.
        min_rows_per_write: Rows accumulated before a request is issued.
        max_rows_per_request: Ceiling on rows in a single request; larger
            accumulations are split.
        transform_fn: Optional per-batch transform applied before writing,
            for enrichment such as computing embeddings.
        when_matched_update_all: For upserts, update rows that matched.
        when_not_matched_insert_all: For upserts, insert rows that did not match.
        when_not_matched_by_source_delete: For upserts, delete target rows with
            no match in the source. ``True`` deletes all such rows; a string
            restricts the deletion to rows satisfying that predicate.
        on_batch_error: ``raise`` (default) fails the job; ``skip`` logs and
            drops the batch. The default is deliberate -- silently dropping
            batches lets a write report success while having lost data.
        retry_policy: Retry behaviour for failed batches.
    """

    def __init__(
        self,
        spec: LanceDBConnectionSpec,
        table_name: str,
        *,
        mode: WriteMode = "append",
        on: Optional[Union[str, list[str]]] = None,
        schema: Optional[pa.Schema] = None,
        rows_per_transaction: int = 256 * 1024,
        max_rows_per_request: Optional[int] = None,
        write_parallelism: Optional[int] = None,
        transform_fn: Optional[TransformFn] = None,
        when_matched_update_all: bool = True,
        when_not_matched_insert_all: bool = True,
        when_not_matched_by_source_delete: Union[bool, str] = False,
        on_batch_error: OnBatchError = "raise",
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        normalized_on = validate_write_args(mode, on)
        if rows_per_transaction <= 0:
            raise ValueError(
                f"rows_per_transaction must be positive, got {rows_per_transaction}"
            )
        if max_rows_per_request is not None and max_rows_per_request <= 0:
            raise ValueError(
                f"max_rows_per_request must be positive, got {max_rows_per_request}"
            )
        if on_batch_error not in ("raise", "skip"):
            raise ValueError(
                f"on_batch_error must be 'raise' or 'skip', got {on_batch_error!r}"
            )

        self._spec = spec
        self._table_name = table_name
        self._mode: WriteMode = mode
        self._on = normalized_on
        self._schema = schema
        self._rows_per_transaction = rows_per_transaction
        self._max_rows_per_request = max_rows_per_request
        self._write_parallelism = write_parallelism
        self._transform_fn = transform_fn
        self._when_matched_update_all = when_matched_update_all
        self._when_not_matched_insert_all = when_not_matched_insert_all
        self._when_not_matched_by_source_delete = when_not_matched_by_source_delete
        self._on_batch_error: OnBatchError = on_batch_error
        self._retry_policy = retry_policy or RetryPolicy(predicate=_retry_predicate)

    def get_name(self) -> str:
        return f"LanceDB({self._table_name})"

    @property
    def supports_distributed_writes(self) -> bool:
        return True

    @property
    def min_rows_per_write(self) -> Optional[int]:
        """Rows Ray should bundle into each write task.

        This is what sets the transaction size: a task issues one LanceDB write
        for everything it receives, so bundling more rows per task means fewer,
        larger transactions and fewer fragments.
        """
        return self._rows_per_transaction

    def on_write_start(self, schema: Optional[pa.Schema] = None) -> None:
        """Create or reset the destination table, once, on the driver.

        Doing this here rather than in the write tasks is what keeps workers
        from racing to create the same table.
        """
        db = connect(self._spec)
        effective_schema = self._schema or schema

        existing = set(list_table_names(db))
        if self._mode == "create" and self._table_name in existing:
            raise ValueError(
                f"Table {self._table_name!r} already exists. Use mode='append', "
                "mode='overwrite' or mode='upsert'."
            )

        if self._mode in ("create", "overwrite"):
            if effective_schema is None:
                raise ValueError(
                    f"mode={self._mode!r} needs a schema to create the table, but "
                    "none was supplied and Ray reported none for the input data. "
                    "Pass schema=... explicitly."
                )
            db.create_table(
                self._table_name,
                schema=effective_schema,
                mode="overwrite" if self._mode == "overwrite" else "create",
            )
        elif self._table_name not in existing:
            # append/upsert into a table that does not exist yet: create it so
            # the write tasks have a target, rather than failing N times.
            if effective_schema is None:
                raise ValueError(
                    f"Table {self._table_name!r} does not exist and no schema is "
                    "available to create it. Pass schema=... explicitly."
                )
            db.create_table(
                self._table_name, schema=effective_schema, mode="create", exist_ok=True
            )

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> WriteStats:
        """Write one task's blocks as a **single** LanceDB transaction.

        Each transaction produces a new table version and at least one fragment,
        so issuing one per incoming batch would leave a table with thousands of
        tiny fragments and, against Cloud/Enterprise, invite rate limiting. The
        rows are instead handed to LanceDB as one ``RecordBatchReader``: the
        client streams that to the service as multiple parts under a single
        upload, committing once.

        Blocks are retained as Arrow batches rather than concatenated so the
        reader can be rebuilt for a retry (a reader is single-use) without
        paying for a copy. Peak memory is therefore one task's worth of rows,
        which is what ``rows_per_transaction`` controls.
        """
        table = connect(self._spec).open_table(self._table_name)
        stats = WriteStats()

        schema: Optional[pa.Schema] = None
        batches: list[pa.RecordBatch] = []
        num_rows = 0

        for block in blocks:
            arrow_block = BlockAccessor.for_block(block).to_arrow()
            if self._transform_fn is not None:
                arrow_block = self._transform_fn(arrow_block)
            if arrow_block.num_rows == 0:
                continue
            if schema is None:
                schema = arrow_block.schema
            batches.extend(arrow_block.to_batches())
            num_rows += arrow_block.num_rows

        if not batches or schema is None:
            return stats

        if self._max_rows_per_request is not None:
            # Opt-in: trade extra transactions for a lower memory ceiling.
            self._write_in_chunks(table, schema, batches, stats)
        else:
            self._write_once(table, schema, batches, num_rows, stats)

        return stats

    def _write_once(
        self,
        table: Any,
        schema: pa.Schema,
        batches: list[pa.RecordBatch],
        num_rows: int,
        stats: WriteStats,
    ) -> None:
        """Issue exactly one write for the whole task."""
        try:
            call_with_retry(
                # Rebuild the reader per attempt: a RecordBatchReader is
                # exhausted once consumed, so a retry needs a fresh one.
                lambda: self._write_reader(
                    table, pa.RecordBatchReader.from_batches(schema, iter(batches))
                ),
                self._retry_policy,
                description=f"write to {self._table_name}",
            )
        except Exception as error:  # noqa: BLE001 - policy decides
            if self._on_batch_error == "raise":
                raise
            logger.error(
                "Dropping %d rows destined for %s after retries: %s",
                num_rows,
                self._table_name,
                error,
            )
            stats.num_skipped_rows += num_rows
            return

        stats.num_rows += num_rows
        stats.num_batches += 1

    def _write_in_chunks(
        self,
        table: Any,
        schema: pa.Schema,
        batches: list[pa.RecordBatch],
        stats: WriteStats,
    ) -> None:
        """Write in bounded pieces when a memory ceiling was requested."""
        assert self._max_rows_per_request is not None
        limit = self._max_rows_per_request

        chunk: list[pa.RecordBatch] = []
        chunk_rows = 0

        def flush() -> None:
            nonlocal chunk, chunk_rows
            if not chunk:
                return
            self._write_once(table, schema, chunk, chunk_rows, stats)
            chunk, chunk_rows = [], 0

        for batch in batches:
            for piece in split_arrow_table(batch.num_rows, limit):
                sliced = batch.slice(piece.start, piece.num_rows)
                chunk.append(sliced)
                chunk_rows += sliced.num_rows
                if chunk_rows >= limit:
                    flush()
        flush()

    def _write_reader(self, table: Any, reader: pa.RecordBatchReader) -> None:
        """Issue a single ``add`` or ``merge_insert`` for a stream of rows."""
        if self._mode != "upsert":
            add_kwargs: dict[str, Any] = {}
            if self._write_parallelism is not None:
                # Controls how many parts the client uploads concurrently
                # within this one transaction.
                add_kwargs["write_parallelism"] = self._write_parallelism
            table.add(reader, **add_kwargs)
            return

        assert self._on is not None  # guaranteed by __init__ validation
        builder = table.merge_insert(self._on)
        if self._when_matched_update_all:
            builder = builder.when_matched_update_all()
        if self._when_not_matched_insert_all:
            builder = builder.when_not_matched_insert_all()
        delete_cond = self._when_not_matched_by_source_delete
        if delete_cond is not False:
            builder = builder.when_not_matched_by_source_delete(
                None if delete_cond is True else delete_cond
            )
        builder.execute(reader)

    def on_write_complete(self, write_result: WriteResult[WriteStats]) -> None:
        total_rows = sum(s.num_rows for s in write_result.write_returns)
        total_skipped = sum(s.num_skipped_rows for s in write_result.write_returns)
        if total_skipped:
            logger.warning(
                "Wrote %d rows to %s; %d rows were dropped because "
                "on_batch_error='skip'.",
                total_rows,
                self._table_name,
                total_skipped,
            )
        else:
            logger.info("Wrote %d rows to %s.", total_rows, self._table_name)
