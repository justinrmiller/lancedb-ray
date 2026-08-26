# SPDX-License-Identifier: Apache-2.0
"""Ray Data source for LanceDB Cloud / Enterprise tables.

Local (OSS) tables are read through ``lance-ray``'s fragment-parallel reader --
see :func:`lancedb_ray.io.read_lancedb`. This module handles the remote case,
where no fragment access exists and parallelism has to be built out of the
query API instead.

The remote strategy is: pin a table version on the driver so every shard sees
the same snapshot, size the scan with ``count_rows``, then split the row space
across read tasks. Without a filter each shard fetches its rows positionally
via ``take_offsets``; with a filter, offsets cannot carry the predicate, so
shards page through server-side ``offset``/``limit`` instead.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import pyarrow as pa
from ray.data.block import BlockMetadata
from ray.data.datasource import Datasource
from ray.data.datasource.datasource import ReadTask

from ._plan import OffsetRange, chunk_offsets, plan_offset_shards
from ._retry import RetryPolicy, call_with_retry
from .connection import LanceDBConnectionSpec, open_table

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["LanceDBDatasource", "RemoteReadStrategy"]

RemoteReadStrategy = Literal["auto", "offsets", "pagination", "single"]

#: Rows sampled to estimate the in-memory size of a remote table.
_SIZE_SAMPLE_ROWS = 32


def _build_block_metadata(
    num_rows: Optional[int], schema: Optional[pa.Schema]
) -> BlockMetadata:
    """Construct ``BlockMetadata`` across Ray versions.

    Ray 2.48 removed the ``schema`` parameter; probe for it rather than pinning
    a Ray version. This mirrors the shim in ``lance_ray/datasource.py``.
    """
    if "schema" in inspect.signature(BlockMetadata.__init__).parameters:
        return BlockMetadata(
            num_rows=num_rows,
            schema=schema,  # type: ignore[call-arg]
            input_files=None,
            size_bytes=None,
            exec_stats=None,
        )
    return BlockMetadata(
        num_rows=num_rows,
        input_files=None,
        size_bytes=None,
        exec_stats=None,
    )


def _apply_columns(query: Any, columns: Optional[list[str]]) -> Any:
    """Apply a column projection to a query builder, if one was requested."""
    if columns:
        return query.select(columns)
    return query


def _empty_table(schema: Optional[pa.Schema]) -> pa.Table:
    """An empty block that still carries the schema, when one is known.

    A read task must yield at least one block so Ray can infer a schema for the
    resulting Dataset, even when the task matched no rows.
    """
    return pa.Table.from_pylist([], schema=schema) if schema else pa.table({})


def _read_offsets(
    spec: LanceDBConnectionSpec,
    table_name: str,
    version: Optional[Union[int, str]],
    offsets: OffsetRange,
    columns: Optional[list[str]],
    batch_size: int,
    retry_policy: RetryPolicy,
    schema: Optional[pa.Schema],
) -> Iterator[pa.Table]:
    """Read one shard positionally via ``take_offsets``.

    Runs inside a Ray worker. Only the shard bounds were shipped here; the
    explicit offset lists ``take_offsets`` requires are generated locally.
    """
    table = open_table(spec, table_name, version=version)

    produced = False
    for chunk in chunk_offsets(offsets, batch_size):
        result = call_with_retry(
            lambda chunk=chunk: _apply_columns(  # type: ignore[misc]
                table.take_offsets(chunk), columns
            ).to_arrow(),
            retry_policy,
            description=f"take_offsets on {table_name}",
        )
        if result.num_rows:
            produced = True
            # Yield per chunk rather than accumulating: a large shard would
            # otherwise hold every batch in worker memory at once.
            yield result

    if not produced:
        yield _empty_table(schema)


def _read_pagination(
    spec: LanceDBConnectionSpec,
    table_name: str,
    version: Optional[Union[int, str]],
    offsets: OffsetRange,
    columns: Optional[list[str]],
    filter_: Optional[str],
    batch_size: int,
    retry_policy: RetryPolicy,
    schema: Optional[pa.Schema],
) -> Iterator[pa.Table]:
    """Read one shard by paging server-side with ``offset``/``limit``.

    Used when a filter is present, because row offsets are positional over the
    whole table and cannot carry a predicate.
    """
    table = open_table(spec, table_name, version=version)

    produced = False
    for start in range(offsets.start, offsets.end, batch_size):
        limit = min(batch_size, offsets.end - start)

        def fetch(start: int = start, limit: int = limit) -> pa.Table:
            query = table.search(None)
            if filter_:
                query = query.where(filter_)
            return _apply_columns(query.offset(start).limit(limit), columns).to_arrow()

        result = call_with_retry(
            fetch, retry_policy, description=f"paged read of {table_name}"
        )
        if result.num_rows:
            produced = True
            yield result
        if result.num_rows < limit:
            # Short page: the filtered result set ended early. Later pages
            # would be empty, so stop rather than issuing pointless requests.
            break

    if not produced:
        yield _empty_table(schema)


def _read_single(
    spec: LanceDBConnectionSpec,
    table_name: str,
    version: Optional[Union[int, str]],
    columns: Optional[list[str]],
    filter_: Optional[str],
    batch_size: int,
    retry_policy: RetryPolicy,
    schema: Optional[pa.Schema],
) -> Iterator[pa.Table]:
    """Stream the whole table through a single task.

    The escape hatch for cases where sharding is undesirable -- notably a
    highly selective filter, where paging costs more than a single scan.
    """
    table = open_table(spec, table_name, version=version)

    def fetch() -> Any:
        query = table.search(None)
        if filter_:
            query = query.where(filter_)
        return _apply_columns(query, columns).to_batches(batch_size)

    batches = call_with_retry(
        fetch, retry_policy, description=f"streaming read of {table_name}"
    )

    empty = True
    for batch in batches:
        empty = False
        yield pa.Table.from_batches([batch])
    if empty:
        yield _empty_table(schema)


class LanceDBDatasource(Datasource):
    """Ray :class:`~ray.data.Datasource` for a LanceDB Cloud/Enterprise table.

    Args:
        spec: How workers should connect to the database.
        table_name: Name of the table to read.
        columns: Optional column projection.
        filter: Optional SQL predicate, evaluated server-side.
        version: Table version to pin. Defaults to the version current when the
            datasource is constructed, which keeps shards consistent even if
            another process writes to the table mid-read.
        strategy: Read strategy; see :data:`RemoteReadStrategy`.
        batch_size: Rows fetched per request by a read task. Larger values mean
            fewer round trips for the same total payload, since the number of
            offsets sent is fixed by the row count either way.
        retry_policy: Retry behaviour for transient remote failures.
    """

    def __init__(
        self,
        spec: LanceDBConnectionSpec,
        table_name: str,
        *,
        columns: Optional[list[str]] = None,
        filter: Optional[str] = None,
        version: Optional[Union[int, str]] = None,
        strategy: RemoteReadStrategy = "auto",
        batch_size: int = 50_000,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        valid_strategies = ("auto", "offsets", "pagination", "single")
        if strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}, got {strategy!r}"
            )

        self._spec = spec
        self._table_name = table_name
        self._columns = list(columns) if columns else None
        self._filter = filter
        self._batch_size = batch_size
        self._retry_policy = retry_policy or RetryPolicy()

        table = open_table(spec, table_name)
        # Pin the version on the driver so every shard reads one snapshot.
        self._version: Union[int, str] = (
            version if version is not None else table.version
        )
        if version is not None:
            table.checkout(version)

        self._schema: Optional[pa.Schema] = getattr(table, "schema", None)
        self._num_rows: int = table.count_rows(filter) if filter else table.count_rows()
        self._strategy: RemoteReadStrategy = self._resolve_strategy(strategy)

    def _resolve_strategy(self, strategy: RemoteReadStrategy) -> RemoteReadStrategy:
        if strategy != "auto":
            return strategy
        # ``take_offsets`` is positional and cannot apply a predicate, so a
        # filtered read has to go through server-side pagination.
        return "pagination" if self._filter else "offsets"

    @property
    def num_rows(self) -> int:
        """Number of rows this read will produce."""
        return self._num_rows

    @property
    def strategy(self) -> RemoteReadStrategy:
        """The resolved read strategy."""
        return self._strategy

    @property
    def version(self) -> Union[int, str]:
        """The pinned table version."""
        return self._version

    def get_name(self) -> str:
        return f"LanceDB({self._table_name})"

    def estimate_inmemory_data_size(self) -> Optional[int]:
        """Estimate the decoded size by sampling a few rows.

        Remote tables expose no size metadata, so sample a small batch and
        extrapolate by row count. Used only for Ray's scheduling heuristics.
        """
        if self._num_rows == 0:
            return 0
        try:
            table = open_table(self._spec, self._table_name, version=self._version)
            sample_size = min(_SIZE_SAMPLE_ROWS, self._num_rows)
            sample = _apply_columns(
                table.take_offsets(list(range(sample_size))), self._columns
            ).to_arrow()
        except Exception as error:  # noqa: BLE001 - estimation must never fail a read
            logger.debug("Could not estimate size of %s: %s", self._table_name, error)
            return None

        if sample.num_rows == 0:
            return 0
        bytes_per_row = sample.nbytes / sample.num_rows
        return int(bytes_per_row * self._num_rows)

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[ReadTask]:
        """Build read tasks.

        Ray passes ``per_task_row_limit`` when a downstream ``limit`` can be
        pushed into the read; extra positional/keyword arguments are absorbed so
        that newer Ray versions adding parameters do not break the integration.
        """
        if self._num_rows == 0:
            return []

        if self._strategy == "single":
            read_fn = partial(
                _read_single,
                self._spec,
                self._table_name,
                self._version,
                self._columns,
                self._filter,
                self._batch_size,
                self._retry_policy,
                self._schema,
            )
            rows = (
                self._num_rows
                if per_task_row_limit is None
                else min(self._num_rows, per_task_row_limit)
            )
            return [ReadTask(read_fn, _build_block_metadata(rows, self._schema))]

        shards = plan_offset_shards(self._num_rows, parallelism)
        if per_task_row_limit is not None:
            shards = [
                OffsetRange(s.start, min(s.end, s.start + per_task_row_limit))
                for s in shards
            ]
        read_tasks: list[ReadTask] = []
        for shard in shards:
            if self._strategy == "offsets":
                # ``partial`` binds this iteration's shard eagerly; a closure
                # over the loop variable would capture only the final value.
                read_fn = partial(
                    _read_offsets,
                    self._spec,
                    self._table_name,
                    self._version,
                    shard,
                    self._columns,
                    self._batch_size,
                    self._retry_policy,
                    self._schema,
                )
            else:
                read_fn = partial(
                    _read_pagination,
                    self._spec,
                    self._table_name,
                    self._version,
                    shard,
                    self._columns,
                    self._filter,
                    self._batch_size,
                    self._retry_policy,
                    self._schema,
                )
            read_tasks.append(
                ReadTask(read_fn, _build_block_metadata(shard.num_rows, self._schema))
            )
        return read_tasks
