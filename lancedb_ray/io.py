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
    connect,
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

#: Block count used to hash-partition an upsert when nothing better is known.
_DEFAULT_UPSERT_PARTITIONS = 16


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
    batch_size: int = 50_000,
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
    partition_on_keys: bool = True,
    schema: Optional[pa.Schema] = None,
    transform_fn: Optional[TransformFn] = None,
    rows_per_transaction: int = 256 * 1024,
    max_rows_per_request: Optional[int] = None,
    write_parallelism: Optional[int] = None,
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

    Delivery semantics:
        Appends are at-least-once, not exactly-once. If the service commits a
        write and the response is then lost, no client can tell that apart from
        a write that never landed, and re-sending duplicates every row in the
        batch. The default retry policy narrows this considerably -- appends
        retry only when the failure proves nothing was applied -- but it cannot
        remove it, and neither can any layer above LanceDB without an
        idempotency key.

        For exactly-once, use ``mode="upsert"`` with ``on=`` naming a key.
        Merge-insert is idempotent: replaying it converges on the same table.

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
        partition_on_keys: For upserts, hash-partition the input on ``on``
            first so all rows for a key land in one write task. This costs a
            shuffle. It changes nothing when the source's keys are already
            unique -- a unique key can only occupy one block -- but when the
            source repeats a key it turns silent cross-task duplication into
            LanceDB's explicit ambiguous-merge error. Disable only when you
            know the keys are unique and want to skip the shuffle.
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
        retry_policy: Retry behaviour for a failed write. The default depends
            on ``mode``, because ``add`` is not idempotent and ``merge_insert``
            is: appends retry only on failures that prove nothing was applied
            (a refused connection, a rate-limit rejection), while upserts also
            retry ambiguous ones such as a read timeout. See the note on
            delivery semantics below.
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
    normalized_on = validate_write_args(mode, on)

    if mode == "upsert" and partition_on_keys and _can_race(concurrency):
        # Two tasks each holding one row for the same key will each find the
        # key absent and each insert it -- silently duplicating it, since
        # neither source is internally ambiguous. Hash-partitioning on the key
        # columns puts every row for a key in one task, where LanceDB rejects
        # the ambiguity outright instead. A loud error beats corrupt data.
        ds = _hash_partition(ds, normalized_on, concurrency)

    use_fragment_path = _should_use_fragment_path(spec, mode, local_write_strategy)

    if use_fragment_path:
        if transform_fn is not None:
            # Run the transform as its own Ray stage rather than inside the
            # sink, so the write itself still takes the single-commit path.
            ds = ds.map_batches(transform_fn, batch_format="pyarrow")
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
        # Hash partitioning already rules out same-key races; this cap is purely
        # about commit-lock contention on a single local dataset. Remote upserts
        # have no such limit -- the service serialises for us.
        concurrency = 4

    datasink = LanceDBDatasink(
        spec,
        table,
        mode=mode,
        on=on,
        # When schema is None, Ray hands the datasink the input schema in
        # on_write_start, so there is no need to materialise it here.
        schema=schema,
        rows_per_transaction=rows_per_transaction,
        max_rows_per_request=max_rows_per_request,
        write_parallelism=write_parallelism,
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


def _can_race(concurrency: Optional[int]) -> bool:
    """Whether more than one write task can run at once.

    A single-task write cannot race with itself, so the shuffle that guards
    against concurrent merge-inserts is pure cost there.
    """
    return concurrency is None or concurrency > 1


def _hash_partition(
    ds: Dataset, keys: Optional[list[str]], concurrency: Optional[int]
) -> Dataset:
    """Co-locate every row sharing a key in the same block.

    Without this, two write tasks holding the same key can both observe it as
    absent and both insert it. Repartitioning by the merge keys makes that
    impossible by construction rather than relying on the backend to detect
    the conflict.
    """
    if not keys:
        return ds

    num_blocks = concurrency if concurrency and concurrency > 0 else None
    if num_blocks is None:
        try:
            num_blocks = ds.num_blocks()
        except Exception:  # noqa: BLE001 - lazy datasets cannot report this
            # num_blocks() raises unless the dataset is materialised, and
            # materialising just to count is not worth it. Ray will still
            # hash-partition correctly with a nominal block count.
            num_blocks = _DEFAULT_UPSERT_PARTITIONS
    if not num_blocks or num_blocks < 1:
        num_blocks = 1
    return ds.repartition(num_blocks, keys=keys)


def _should_use_fragment_path(
    spec: LanceDBConnectionSpec,
    mode: WriteMode,
    local_write_strategy: LocalWriteStrategy,
) -> bool:
    """Decide whether a write can take the distributed fragment path.

    The fragment path writes raw Lance fragments and commits them in one
    transaction. That is only meaningful for local tables doing whole-row
    appends -- it cannot express upsert semantics.
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

    # One catalog listing, reused: listing is paginated and costs a round trip
    # per page against Cloud/Enterprise.
    exists = table_exists(spec, table)
    dataset_uri = table_uri(spec, table, exists=exists)

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

    # Confirm the database can now resolve what we materialised. Opening the
    # table directly is one call; re-listing the whole catalog is not.
    try:
        connect(spec).open_table(table)
        return
    except Exception as error:  # noqa: BLE001 - one recoverable case below
        resolve_error = error

    # A dataset with no rows produces no fragments, and therefore no manifest
    # for the database to open. Asking to create a table from an empty input is
    # a reasonable thing to do -- a job whose filter matched nothing still wants
    # its table -- so materialise it from the schema instead of failing.
    if schema is not None:
        connect(spec).create_table(table, schema=schema, mode="overwrite")
        return

    raise RuntimeError(
        f"Wrote a Lance dataset to {dataset_uri!r} but database {spec.uri!r} "
        f"cannot open a table named {table!r}. If the input was empty, pass "
        "schema=... so the table can be created from it."
    ) from resolve_error
