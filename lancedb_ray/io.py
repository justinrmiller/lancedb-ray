# SPDX-License-Identifier: Apache-2.0
"""Public read and write entry points for LanceDB tables.

These two functions hide the fact that LanceDB is really two systems behind one
API. Local (OSS) tables are backed by a Lance dataset and support fragment-level
parallelism; Cloud/Enterprise tables are an opaque service reachable only
through queries. ``read_lancedb`` and ``write_lancedb`` pick the right strategy
and expose the same surface either way.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal, Optional, Union, cast

import pyarrow as pa
import ray
from ray.data import Dataset

from ._retry import RetryPolicy
from .connection import (
    LanceDBConnectionSpec,
    open_table,
    table_exists,
    table_uri,
    to_lance,
)
from .datasink import (
    LanceDBDatasink,
    OnBatchError,
    TransformFn,
    WriteMode,
    validate_write_args,
)
from .datasource import LanceDBDatasource, RemoteReadStrategy

logger = logging.getLogger(__name__)

__all__ = ["read_lancedb", "write_lancedb"]

LocalWriteStrategy = Literal["auto", "fragment", "api"]


def _build_spec(
    uri: str,
    api_key: Optional[str],
    region: str,
    host_override: Optional[str],
    storage_options: Optional[Mapping[str, Any]],
    client_config: Optional[Mapping[str, Any]],
    namespace_client_impl: Optional[str],
    namespace_client_properties: Optional[Mapping[str, Any]],
) -> LanceDBConnectionSpec:
    return LanceDBConnectionSpec.create(
        uri,
        api_key=api_key,
        region=region,
        host_override=host_override,
        storage_options=storage_options,
        client_config=client_config,
        namespace_client_impl=namespace_client_impl,
        namespace_client_properties=namespace_client_properties,
    )


def read_lancedb(
    table: str,
    *,
    uri: str,
    api_key: Optional[str] = None,
    region: str = "us-east-1",
    host_override: Optional[str] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
    client_config: Optional[Mapping[str, Any]] = None,
    namespace_client_impl: Optional[str] = None,
    namespace_client_properties: Optional[Mapping[str, Any]] = None,
    columns: Optional[list[str]] = None,
    filter: Optional[str] = None,
    version: Optional[Union[int, str]] = None,
    remote_read_strategy: RemoteReadStrategy = "auto",
    batch_size: int = 10_000,
    ray_remote_args: Optional[dict[str, Any]] = None,
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
) -> Dataset:
    """Read a LanceDB table into a Ray :class:`~ray.data.Dataset`.

    The read is always against a single pinned table version, so concurrent
    writers cannot tear the result across shards.

    For local tables this reads Lance fragments in parallel (via ``lance-ray``),
    pushing ``columns`` and ``filter`` down into the scan. For Cloud/Enterprise
    tables it shards the row space across read tasks: positionally when there is
    no filter, and by server-side pagination when there is.

    Examples:
        Local::

            ds = read_lancedb("my_table", uri="/data/lancedb")

        LanceDB Enterprise (API key from ``LANCEDB_API_KEY``)::

            ds = read_lancedb(
                "my_table",
                uri="db://my-database",
                region="us-east-1",
                columns=["id", "vector"],
            )

    Args:
        table: Name of the table to read.
        uri: Database URI. ``db://...`` selects Cloud/Enterprise.
        api_key: Cloud/Enterprise key. Defaults to ``LANCEDB_API_KEY``, which is
            preferable because it keeps the secret out of Ray task definitions.
        region: Cloud region.
        host_override: Alternate endpoint for Enterprise deployments.
        storage_options: Object-store options for local/OSS URIs.
        client_config: HTTP client configuration for remote connections.
        namespace_client_impl: Lance Namespace implementation to resolve through.
        namespace_client_properties: Properties for the namespace implementation.
        columns: Optional column projection.
        filter: Optional SQL predicate, evaluated server-side.
        version: Table version to pin. Defaults to the current version.
        remote_read_strategy: Remote sharding strategy -- ``auto`` (default),
            ``offsets``, ``pagination`` or ``single``. Ignored for local tables.
        batch_size: Rows per request issued by each remote read task.
        ray_remote_args: Resource arguments for the read tasks.
        concurrency: Maximum number of concurrent read tasks.
        override_num_blocks: Override the number of output blocks.

    Returns:
        A Ray Dataset over the table's contents.
    """
    spec = _build_spec(
        uri,
        api_key,
        region,
        host_override,
        storage_options,
        client_config,
        namespace_client_impl,
        namespace_client_properties,
    )

    if not spec.is_remote:
        return _read_local(
            spec,
            table,
            columns=columns,
            filter=filter,
            version=version,
            ray_remote_args=ray_remote_args,
            concurrency=concurrency,
            override_num_blocks=override_num_blocks,
        )

    datasource = LanceDBDatasource(
        spec,
        table,
        columns=columns,
        filter=filter,
        version=version,
        strategy=remote_read_strategy,
        batch_size=batch_size,
    )
    return cast(
        "Dataset",
        ray.data.read_datasource(
            datasource,
            ray_remote_args=ray_remote_args,
            concurrency=concurrency,
            override_num_blocks=override_num_blocks,
        ),
    )


def _read_local(
    spec: LanceDBConnectionSpec,
    table: str,
    *,
    columns: Optional[list[str]],
    filter: Optional[str],
    version: Optional[Union[int, str]],
    ray_remote_args: Optional[dict[str, Any]],
    concurrency: Optional[int],
    override_num_blocks: Optional[int],
) -> Dataset:
    """Read a local table via ``lance-ray``'s fragment-parallel reader.

    Delegating here rather than reimplementing gets us fragment splitting,
    predicate and projection pushdown, and IO retry that is already exercised by
    the lance-ray test suite.
    """
    import lance_ray

    dataset = to_lance(open_table(spec, table, version=version))
    # Pin explicitly: without a version the workers would each resolve "latest"
    # independently and could disagree.
    pinned = version if version is not None else dataset.version

    return lance_ray.read_lance(
        uri=dataset.uri,
        columns=columns,
        filter=filter,
        storage_options=dict(spec.storage_options) if spec.storage_options else None,
        dataset_options={"version": pinned},
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )


def write_lancedb(
    ds: Dataset,
    table: str,
    *,
    uri: str,
    api_key: Optional[str] = None,
    region: str = "us-east-1",
    host_override: Optional[str] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
    client_config: Optional[Mapping[str, Any]] = None,
    namespace_client_impl: Optional[str] = None,
    namespace_client_properties: Optional[Mapping[str, Any]] = None,
    mode: WriteMode = "append",
    on: Optional[Union[str, list[str]]] = None,
    schema: Optional[pa.Schema] = None,
    transform_fn: Optional[TransformFn] = None,
    min_rows_per_write: int = 1024,
    max_rows_per_request: int = 65_536,
    when_matched_update_all: bool = True,
    when_not_matched_insert_all: bool = True,
    when_not_matched_by_source_delete: Union[bool, str] = False,
    on_batch_error: OnBatchError = "raise",
    local_write_strategy: LocalWriteStrategy = "auto",
    max_rows_per_file: int = 1024 * 1024,
    min_rows_per_file: int = 1024,
    retry_policy: Optional[RetryPolicy] = None,
    ray_remote_args: Optional[dict[str, Any]] = None,
    concurrency: Optional[int] = None,
) -> None:
    """Write a Ray :class:`~ray.data.Dataset` into a LanceDB table.

    Local append-style writes take the fragment path by default: workers write
    Lance fragments in parallel and the driver commits them as a **single atomic
    transaction**, so the table advances by exactly one version no matter how
    many tasks participated. Upserts and all Cloud/Enterprise writes go through
    LanceDB's table API instead.

    Examples:
        Distributed local append::

            write_lancedb(ds, "my_table", uri="/data/lancedb", mode="append")

        Upsert into LanceDB Enterprise::

            write_lancedb(
                ds, "my_table", uri="db://my-database", mode="upsert", on="id"
            )

    Args:
        ds: The dataset to write.
        table: Destination table name.
        uri: Database URI. ``db://...`` selects Cloud/Enterprise.
        api_key: Cloud/Enterprise key; defaults to ``LANCEDB_API_KEY``.
        region: Cloud region.
        host_override: Alternate endpoint for Enterprise deployments.
        storage_options: Object-store options for local/OSS URIs.
        client_config: HTTP client configuration for remote connections.
        namespace_client_impl: Lance Namespace implementation to resolve through.
        namespace_client_properties: Properties for the namespace implementation.
        mode: ``create``, ``append``, ``overwrite`` or ``upsert``.
        on: Key column(s) to match on. Required when ``mode="upsert"``.
        schema: Schema used when creating the table. Defaults to the dataset's.
        transform_fn: Optional per-batch transform applied before writing.
            Only supported on the API write path; requesting it for a local
            append switches that write to the API path.
        min_rows_per_write: Rows accumulated per request on the API path.
        max_rows_per_request: Ceiling on rows in a single request.
        when_matched_update_all: For upserts, update rows that matched.
        when_not_matched_insert_all: For upserts, insert rows that did not match.
        when_not_matched_by_source_delete: For upserts, delete unmatched target
            rows -- ``True`` for all, or a predicate string to restrict them.
        on_batch_error: ``raise`` (default) or ``skip``. ``skip`` drops batches
            that fail after retries, which trades data loss for job completion.
        local_write_strategy: ``auto`` (default) uses the fragment path where it
            applies; ``fragment`` forces it; ``api`` forces the table API.
        max_rows_per_file: Fragment sizing for the fragment path.
        min_rows_per_file: Fragment sizing for the fragment path.
        retry_policy: Retry behaviour for failed batches on the API path.
        ray_remote_args: Resource arguments for the write tasks.
        concurrency: Maximum number of concurrent write tasks. Local upserts
            default to 4 because concurrent merge-insert contends on the
            dataset commit lock.
    """
    spec = _build_spec(
        uri,
        api_key,
        region,
        host_override,
        storage_options,
        client_config,
        namespace_client_impl,
        namespace_client_properties,
    )

    if local_write_strategy not in ("auto", "fragment", "api"):
        raise ValueError(
            "local_write_strategy must be 'auto', 'fragment' or 'api', got "
            f"{local_write_strategy!r}"
        )

    # Validate here rather than relying on the datasink: the fragment path does
    # not construct a datasink at all, so it would otherwise accept a bad mode
    # and fail much later with a confusing storage-level error.
    validate_write_args(mode, on)

    use_fragment_path = _should_use_fragment_path(
        spec, mode, local_write_strategy, transform_fn
    )

    if use_fragment_path:
        _write_local_fragments(
            ds,
            spec,
            table,
            mode=mode,
            schema=schema,
            max_rows_per_file=max_rows_per_file,
            min_rows_per_file=min_rows_per_file,
            ray_remote_args=ray_remote_args,
            concurrency=concurrency,
        )
        return

    if concurrency is None and not spec.is_remote and mode == "upsert":
        # Concurrent merge-insert against one local dataset contends on the
        # commit lock; every loser retries. Cap it rather than let the job
        # thrash. Remote upserts have no such limit -- the service serialises.
        concurrency = 4

    datasink = LanceDBDatasink(
        spec,
        table,
        mode=mode,
        on=on,
        # When schema is None, Ray hands the datasink the input schema in
        # on_write_start, so there is no need to materialise it here.
        schema=schema,
        min_rows_per_write=min_rows_per_write,
        max_rows_per_request=max_rows_per_request,
        transform_fn=transform_fn,
        when_matched_update_all=when_matched_update_all,
        when_not_matched_insert_all=when_not_matched_insert_all,
        when_not_matched_by_source_delete=when_not_matched_by_source_delete,
        on_batch_error=on_batch_error,
        retry_policy=retry_policy,
    )
    ds.write_datasink(
        datasink, ray_remote_args=ray_remote_args, concurrency=concurrency
    )


def _should_use_fragment_path(
    spec: LanceDBConnectionSpec,
    mode: WriteMode,
    local_write_strategy: LocalWriteStrategy,
    transform_fn: Optional[TransformFn],
) -> bool:
    """Decide whether a write can take the distributed fragment path.

    The fragment path writes raw Lance fragments and commits them in one
    transaction. That is only meaningful for local tables doing whole-row
    appends -- it cannot express upsert semantics, and it bypasses the
    per-batch hook that ``transform_fn`` relies on.
    """
    if local_write_strategy == "api":
        return False

    blockers: list[str] = []
    if spec.is_remote:
        blockers.append(
            "the database is LanceDB Cloud/Enterprise, which exposes no "
            "fragment-level API"
        )
    if mode == "upsert":
        blockers.append("upsert requires row matching, which fragment writes cannot do")
    if transform_fn is not None:
        blockers.append("transform_fn is only applied on the table API path")

    if not blockers:
        return True

    if local_write_strategy == "fragment":
        raise ValueError(
            "local_write_strategy='fragment' was requested but is not possible: "
            + "; ".join(blockers)
            + "."
        )
    logger.debug("Using the table API write path because %s.", "; ".join(blockers))
    return False


def _write_local_fragments(
    ds: Dataset,
    spec: LanceDBConnectionSpec,
    table: str,
    *,
    mode: WriteMode,
    schema: Optional[pa.Schema],
    max_rows_per_file: int,
    min_rows_per_file: int,
    ray_remote_args: Optional[dict[str, Any]],
    concurrency: Optional[int],
) -> None:
    """Write fragments in parallel and commit them as one transaction.

    Delegates to ``lance-ray``, which writes each block as a Lance fragment on a
    worker and commits the resulting fragment metadata once on the driver.
    """
    import lance_ray

    dataset_uri = table_uri(spec, table)
    exists = table_exists(spec, table)

    if mode == "create" and exists:
        raise ValueError(
            f"Table {table!r} already exists. Use mode='append', mode='overwrite' "
            "or mode='upsert'."
        )
    if mode == "append" and not exists:
        # lance-ray cannot append to a dataset that is not there yet.
        lance_mode: Literal["create", "append", "overwrite"] = "create"
    elif mode == "overwrite":
        lance_mode = "overwrite"
    elif mode == "create":
        lance_mode = "create"
    else:
        lance_mode = "append"

    lance_ray.write_lance(
        ds,
        uri=dataset_uri,
        schema=schema,
        mode=lance_mode,
        max_rows_per_file=max_rows_per_file,
        min_rows_per_file=min_rows_per_file,
        storage_options=dict(spec.storage_options) if spec.storage_options else None,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
    )

    # Make sure the database can now resolve the table we just materialised.
    if not table_exists(spec, table):
        raise RuntimeError(
            f"Wrote a Lance dataset to {dataset_uri!r} but database {spec.uri!r} "
            f"does not list a table named {table!r}."
        )
