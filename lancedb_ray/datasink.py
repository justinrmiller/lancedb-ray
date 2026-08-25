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
        min_rows_per_write: int = 1024,
        max_rows_per_request: int = 65_536,
        transform_fn: Optional[TransformFn] = None,
        when_matched_update_all: bool = True,
        when_not_matched_insert_all: bool = True,
        when_not_matched_by_source_delete: Union[bool, str] = False,
        on_batch_error: OnBatchError = "raise",
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        normalized_on = validate_write_args(mode, on)
        if min_rows_per_write <= 0:
            raise ValueError(
                f"min_rows_per_write must be positive, got {min_rows_per_write}"
            )
        if max_rows_per_request <= 0:
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
        self._min_rows_per_write = min_rows_per_write
        self._max_rows_per_request = max_rows_per_request
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
        return self._min_rows_per_write

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
        """Write one task's blocks, accumulating them into sized requests."""
        table = connect(self._spec).open_table(self._table_name)
        stats = WriteStats()

        pending: list[pa.Table] = []
        pending_rows = 0

        for block in blocks:
            arrow_block = BlockAccessor.for_block(block).to_arrow()
            if self._transform_fn is not None:
                arrow_block = self._transform_fn(arrow_block)
            if arrow_block.num_rows == 0:
                continue

            pending.append(arrow_block)
            pending_rows += arrow_block.num_rows

            if pending_rows >= self._min_rows_per_write:
                self._flush(table, pending, stats)
                pending, pending_rows = [], 0

        if pending:
            self._flush(table, pending, stats)

        return stats

    def _flush(self, table: Any, pending: list[pa.Table], stats: WriteStats) -> None:
        """Concatenate pending blocks and write them in request-sized pieces."""
        combined = (
            pending[0]
            if len(pending) == 1
            else pa.concat_tables(pending, promote_options="default")
        )

        for piece in split_arrow_table(combined.num_rows, self._max_rows_per_request):
            batch = combined.slice(piece.start, piece.num_rows)
            try:
                call_with_retry(
                    lambda batch=batch: self._write_batch(table, batch),  # type: ignore[misc]
                    self._retry_policy,
                    description=f"write to {self._table_name}",
                )
            except Exception as error:  # noqa: BLE001 - policy decides
                if self._on_batch_error == "raise":
                    raise
                logger.error(
                    "Dropping a batch of %d rows destined for %s after retries: %s",
                    batch.num_rows,
                    self._table_name,
                    error,
                )
                stats.num_skipped_rows += batch.num_rows
                continue

            stats.num_rows += batch.num_rows
            stats.num_batches += 1

    def _write_batch(self, table: Any, batch: pa.Table) -> None:
        """Issue a single ``add`` or ``merge_insert`` for one batch."""
        if self._mode != "upsert":
            table.add(batch)
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
        builder.execute(batch)

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
