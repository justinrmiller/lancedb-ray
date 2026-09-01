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
from enum import Enum
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
from .datasource import (
    DEFAULT_BATCH_SIZE,
    LanceDBDatasource,
    RemoteReadStrategy,
    validate_columns,
)

logger = logging.getLogger(__name__)

__all__ = ["read_lancedb", "write_lancedb"]

LocalWriteStrategy = Literal["auto", "fragment", "api"]

#: Block count used to hash-partition an upsert when nothing better is known.
_DEFAULT_UPSERT_PARTITIONS = 16


class _DatasetState(Enum):
    """What we could establish about a dataset URI after a write."""

    #: Nothing is there -- the write produced no fragments, so the input was empty.
    ABSENT = "absent"
    #: A dataset opened; it holds the rows that were written.
    PRESENT = "present"
    #: Storage could not answer. Treated as PRESENT would be, never as ABSENT.
    UNKNOWN = "unknown"


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


def _validate_concurrency(concurrency: Optional[int]) -> None:
    """Reject a concurrency Ray cannot act on.

    Passed through unchecked, a zero or negative value surfaces from inside
    Ray as "`size` must be >= 1" -- naming an internal argument rather than the
    one the caller actually set.
    """
    if concurrency is not None and concurrency < 1:
        raise ValueError(
            f"concurrency must be at least 1, got {concurrency}. Pass "
            "concurrency=None (the default) to let Ray decide."
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
    batch_size: Optional[int] = None,
    scanner_options: Optional[Mapping[str, Any]] = None,
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
        batch_size: Rows fetched per read request. On a remote read this sizes
            each task's requests; on a local read it sizes the Lance scanner's
            batches, and takes precedence over a ``batch_size`` named in
            ``scanner_options``. Defaults to the backend's own sizing.
        scanner_options: Extra options for the Lance scanner on **local** reads
            -- ``batch_size``, ``use_scalar_index``, ``late_materialization``,
            ``with_row_id`` and friends. Without this the local scan is only
            tunable through ``columns`` and ``filter``, which leaves the scan
            knobs that matter for a wide table unreachable. ``columns`` and
            ``filter`` are set from their own arguments and win over anything
            named here. Rejected for ``db://`` URIs, which have no scanner.
        ray_remote_args: Resource arguments for the read tasks.
        concurrency: Maximum number of concurrent read tasks.
        override_num_blocks: Override the number of output blocks.

    Returns:
        A Ray Dataset over the table's contents.
    """
    _validate_concurrency(concurrency)
    if batch_size is not None and batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    # Checked before the local/remote split: the two paths hand ``columns`` to
    # different engines, which disagree about what an empty list means (one
    # reads every column, the other reads none). Neither is what was asked for.
    validate_columns(columns)
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
            batch_size=batch_size,
            scanner_options=scanner_options,
            ray_remote_args=ray_remote_args,
            concurrency=concurrency,
            override_num_blocks=override_num_blocks,
        )

    if scanner_options:
        # Silently dropping these would look like a tuning that did nothing.
        raise ValueError(
            "scanner_options applies to the Lance scanner, which only exists "
            f"for local tables; {uri!r} is Cloud/Enterprise. Use batch_size and "
            "remote_read_strategy to tune a remote read."
        )

    datasource = LanceDBDatasource(
        spec,
        table,
        columns=columns,
        filter=filter,
        version=version,
        strategy=remote_read_strategy,
        batch_size=batch_size if batch_size is not None else DEFAULT_BATCH_SIZE,
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
    batch_size: Optional[int],
    scanner_options: Optional[Mapping[str, Any]],
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

    # ``batch_size`` is a scanner option locally. Folding it in here is what
    # makes the argument mean the same thing on both backends -- it used to be
    # accepted and then silently dropped, which is the "tuning that did
    # nothing" that ``scanner_options`` is rejected for remotely.
    scan_opts = dict(scanner_options) if scanner_options else {}
    if batch_size is not None:
        scan_opts["batch_size"] = batch_size

    return lance_ray.read_lance(
        uri=dataset.uri,
        columns=columns,
        filter=filter,
        storage_options=dict(spec.storage_options) if spec.storage_options else None,
        dataset_options={"version": pinned},
        scanner_options=scan_opts or None,
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
    max_bytes_per_request: Optional[int] = None,
    write_parallelism: Optional[int] = None,
    when_matched_update_all: bool = True,
    when_not_matched_insert_all: bool = True,
    when_not_matched_by_source_delete: Union[bool, str] = False,
    on_batch_error: OnBatchError = "raise",
    local_write_strategy: LocalWriteStrategy = "auto",
    max_rows_per_file: int = 1024 * 1024,
    min_rows_per_file: int = 1024,
    data_storage_version: Optional[str] = None,
    enable_stable_row_ids: bool = False,
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
            On the API path the sink applies it per batch; on the fragment path
            it runs as its own Ray stage beforehand, so the write itself still
            commits once. Note it runs *after* ``partition_on_keys`` has
            shuffled, so a transform that rewrites a key column undoes that
            co-location.
        rows_per_transaction: Rows Ray bundles into each write task, which is
            what sets the transaction size and the task's peak memory.
        max_rows_per_request: Ceiling on rows in a single request; larger
            accumulations are split across several. Setting either per-request
            ceiling means a task no longer writes in one transaction, so a
            failure partway can leave earlier chunks committed.
        max_bytes_per_request: Ceiling on the size of a single request's
            payload, on the API path. Rows are a poor proxy for size once a
            schema is wide -- 256K rows of a 1536-dimension float32 embedding is
            1.5GB before anything else in the row is counted -- so this bounds
            what one transaction hands to LanceDB in the unit that matters.
            Combines with ``max_rows_per_request``; whichever is met first
            closes the request. Note this does **not** lower the task's peak
            memory: a task materialises all of its rows before either ceiling
            applies, so ``rows_per_transaction`` is the knob for that.
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
        data_storage_version: Lance file format version for newly written
            fragments (``"stable"``, ``"2.1"``, ...). Fragment path only.
        enable_stable_row_ids: Give rows IDs that survive compaction, on the
            fragment path. Off by default because it costs an index, but it is
            fixed when the dataset is created -- a table written without it
            cannot be switched later without a rewrite -- so it is worth
            deciding at creation rather than discovering afterwards.
        write_parallelism: How many parts the client uploads concurrently
            within one transaction, on the API path.
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
    _validate_concurrency(concurrency)
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
            data_storage_version=data_storage_version,
            enable_stable_row_ids=enable_stable_row_ids,
            ray_remote_args=ray_remote_args,
            concurrency=concurrency,
        )
        return

    _reject_fragment_only_options(
        data_storage_version=data_storage_version,
        enable_stable_row_ids=enable_stable_row_ids,
    )

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
        max_bytes_per_request=max_bytes_per_request,
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


def _reject_fragment_only_options(
    *, data_storage_version: Optional[str], enable_stable_row_ids: bool
) -> None:
    """Refuse fragment-path options on a write that will not take that path.

    Both are properties of the Lance files a fragment write produces, so the
    table API has nowhere to put them. Accepting and ignoring them would be
    worse than refusing: ``enable_stable_row_ids`` is fixed at dataset creation,
    so a caller who thinks they enabled it and did not finds out only when they
    need the IDs and the dataset has to be rewritten.
    """
    unusable = []
    if data_storage_version is not None:
        unusable.append("data_storage_version")
    if enable_stable_row_ids:
        unusable.append("enable_stable_row_ids")
    if unusable:
        raise ValueError(
            f"{', '.join(unusable)} only applies to the local fragment write "
            "path, which this write does not take (Cloud/Enterprise, "
            "mode='upsert' or local_write_strategy='api')."
        )


def _can_race(concurrency: Optional[int]) -> bool:
    """Whether more than one write task can run at once.

    A single-task write cannot race with itself, so the shuffle that guards
    against concurrent merge-inserts is pure cost there.
    """
    return concurrency is None or concurrency > 1


#: Ray shuffle strategies that implement key-based repartitioning.
_HASH_SHUFFLE_STRATEGIES = ("hash_shuffle", "hash_shuffle_v2", "gpu_shuffle")


def _require_hash_shuffle() -> None:
    """Fail early when Ray cannot hash-partition on a key.

    ``repartition(keys=...)`` is implemented only for the hash-based shuffle
    strategies. Under a sort-based one Ray raises at plan time, naming an
    internal config rather than the upsert that needed it -- and the guarantee
    the shuffle exists to provide would be gone either way. The default is
    hash-based, so this only fires where a cluster has changed it.
    """
    try:
        from ray.data.context import DataContext

        strategy = DataContext.get_current().shuffle_strategy
    except Exception:  # pragma: no cover - depends on the installed Ray
        return
    if str(getattr(strategy, "value", strategy)) in _HASH_SHUFFLE_STRATEGIES:
        return
    raise ValueError(
        "mode='upsert' with partition_on_keys=True needs a hash-based shuffle "
        f"to co-locate each key, but DataContext.shuffle_strategy is {strategy!r}. "
        "Set it to ShuffleStrategy.HASH_SHUFFLE, or pass partition_on_keys=False "
        "if the source's keys are already unique -- with repeated keys and more "
        "than one write task, rows for the same key duplicate silently."
    )


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

    _require_hash_shuffle()

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
    data_storage_version: Optional[str],
    enable_stable_row_ids: bool,
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

    # An input with no rows produces no fragments, and lance-ray then commits
    # nothing at all -- leaving the dataset exactly as it was, at the same
    # version. That is correct for an append and wrong for an overwrite, so the
    # version is recorded here to tell the two apart afterwards.
    version_before = _dataset_version(dataset_uri, spec) if exists else None

    lance_ray.write_lance(
        ds,
        uri=dataset_uri,
        schema=schema,
        mode=lance_mode,
        max_rows_per_file=max_rows_per_file,
        min_rows_per_file=min_rows_per_file,
        data_storage_version=data_storage_version,
        enable_stable_row_ids=enable_stable_row_ids,
        storage_options=dict(spec.storage_options) if spec.storage_options else None,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
    )

    if mode == "overwrite" and exists:
        _finish_empty_overwrite(spec, table, dataset_uri, schema, version_before)

    # Confirm the database can now resolve what we materialised. Opening the
    # table directly is one call; re-listing the whole catalog is not.
    try:
        connect(spec).open_table(table)
        return
    except Exception as error:  # noqa: BLE001 - one recoverable case below
        resolve_error = error

    # Exactly one failure here is recoverable: an input with no rows produces
    # no fragments and so no dataset at all, leaving nothing for the database
    # to open. Every other cause -- a transient catalog error, a permission
    # fault, a race -- means the rows *are* on disk, and recovering by writing
    # an empty dataset over them would destroy a completed write and report
    # success. So establish which case this is before assuming.
    if _dataset_state(dataset_uri, spec) is not _DatasetState.ABSENT:
        raise RuntimeError(
            f"Wrote a Lance dataset to {dataset_uri!r} but database "
            f"{spec.uri!r} cannot open a table named {table!r}. The data is "
            "there -- this is a resolution failure, not an empty write, so it "
            "is left untouched rather than replaced with an empty table."
        ) from resolve_error

    # An absent dataset is only benign when the input genuinely had no rows.
    # If it had rows, they went somewhere this write cannot account for, and
    # putting an empty table at that URI would report success over the loss.
    # Establish the count before assuming; "cannot tell" is not "empty".
    num_input_rows = _input_row_count(ds)
    if num_input_rows != 0:
        raise RuntimeError(
            f"The write produced no Lance dataset at {dataset_uri!r} and database "
            f"{spec.uri!r} cannot open a table named {table!r}, but the input was "
            + (
                "not empty"
                if num_input_rows is None
                else f"{num_input_rows} rows"
            )
            + ". Refusing to replace it with an empty table."
        ) from resolve_error

    # A job whose filter matched nothing still wants its table, so materialise
    # it from the schema rather than failing. Ray knows the input's schema even
    # when the input has no rows, and the table API path already creates the
    # table from it -- so the two paths agreed on everything except this, where
    # the default local path was the one that failed.
    effective_schema = schema if schema is not None else _arrow_schema(ds)
    if effective_schema is None:
        raise RuntimeError(
            f"Wrote a Lance dataset to {dataset_uri!r} but database {spec.uri!r} "
            f"cannot open a table named {table!r}. The input was empty and Ray "
            "reported no schema for it, so pass schema=... to create the table."
        ) from resolve_error

    _create_empty_dataset(
        dataset_uri,
        effective_schema,
        spec,
        data_storage_version=data_storage_version,
        enable_stable_row_ids=enable_stable_row_ids,
    )
    # Writing the files is not the same as the database being able to resolve
    # them. Confirm rather than return an unverified success.
    try:
        connect(spec).open_table(table)
    except Exception as error:
        raise RuntimeError(
            f"Created an empty Lance dataset at {dataset_uri!r} but database "
            f"{spec.uri!r} still cannot open a table named {table!r}."
        ) from error


def _arrow_schema(ds: Dataset) -> Optional[pa.Schema]:
    """The dataset's own Arrow schema, or ``None`` if Ray cannot report one.

    Only called on the recovery path, where the write produced no fragments and
    the input was therefore empty -- so ``fetch_if_missing`` re-executes an
    empty plan rather than real work. ``Dataset.schema`` returns Ray's wrapper,
    which carries the Arrow schema underneath.
    """
    try:
        reported = ds.schema(fetch_if_missing=True)
    except Exception as error:  # noqa: BLE001 - a missing schema is not fatal
        logger.debug("Could not resolve the dataset schema: %s", error)
        return None
    base = getattr(reported, "base_schema", reported)
    return base if isinstance(base, pa.Schema) else None


def _input_row_count(ds: Dataset) -> Optional[int]:
    """How many rows the input held, or ``None`` if Ray would not say.

    Only called on the recovery path, where the write produced no dataset. For
    the case that path exists to serve -- an input with no rows -- counting
    executes an empty plan. For any other cause it costs a re-execution on the
    way to raising, which is the right trade against silently standing an empty
    table where a completed write should be.
    """
    try:
        return int(ds.count())
    except Exception as error:  # noqa: BLE001 - cannot tell; must not assume
        logger.debug("Could not count the input dataset: %s", error)
        return None


def _dataset_state(dataset_uri: str, spec: LanceDBConnectionSpec) -> _DatasetState:
    """Establish whether a Lance dataset was materialised at ``dataset_uri``.

    The answer decides whether it is safe to overwrite that location with an
    empty table, so the uncertain case must never be reported as ``ABSENT``. A
    missing dataset raises ``ValueError`` from ``lance``; anything else -- a
    permission fault, an object store that would not answer -- tells us nothing
    about whether data is there, and guessing wrong destroys a completed write.
    Failing a genuinely empty write with a clear error is the cheaper mistake.
    """
    import lance

    try:
        lance.dataset(
            dataset_uri,
            storage_options=dict(spec.storage_options)
            if spec.storage_options
            else None,
        )
    except ValueError:
        return _DatasetState.ABSENT
    except Exception as error:  # noqa: BLE001 - cannot tell; must not assume
        logger.debug("Could not determine the state of %s: %s", dataset_uri, error)
        return _DatasetState.UNKNOWN
    return _DatasetState.PRESENT


def _dataset_version(dataset_uri: str, spec: LanceDBConnectionSpec) -> Optional[int]:
    """The current version of the Lance dataset at ``dataset_uri``, if readable.

    ``None`` means the question could not be answered -- the dataset is absent,
    or storage would not say. Callers must treat that as "cannot tell" rather
    than as any particular version, because the comparison it feeds decides
    whether to replace a table's contents.
    """
    import lance

    try:
        dataset = lance.dataset(
            dataset_uri,
            storage_options=(
                dict(spec.storage_options) if spec.storage_options else None
            ),
        )
        return int(dataset.version)
    except Exception as error:  # noqa: BLE001 - absent or unreadable, same answer
        logger.debug("Could not read the version of %s: %s", dataset_uri, error)
        return None


def _finish_empty_overwrite(
    spec: LanceDBConnectionSpec,
    table: str,
    dataset_uri: str,
    schema: Optional[pa.Schema],
    version_before: Optional[int],
) -> None:
    """Make an overwrite whose input was empty actually empty the table.

    ``write_lance`` commits nothing for a zero-row input, so an overwrite that
    matched no rows leaves every previous row in place and reports success. A
    nightly job whose filter came back empty would then publish yesterday's data
    as today's, which is worse than either failing or emptying the table. The
    Cloud/Enterprise path already replaces the table up front, so this also
    stops the two backends disagreeing about what ``mode="overwrite"`` means.

    An unchanged version is the signal that nothing was committed. Both probes
    have to succeed for that comparison to mean anything: if either could not
    read the version, the table is left alone, because the cost of being wrong
    here is deleting rows the caller wanted to keep.
    """
    version_after = _dataset_version(dataset_uri, spec)
    if version_before is None or version_after is None:
        logger.warning(
            "Could not confirm whether the overwrite of %s committed anything, "
            "so its previous contents were left in place. If the input was "
            "empty, the table still holds the rows it had before.",
            table,
        )
        return
    if version_after != version_before:
        return

    # Nothing was committed, so the input had no rows. Reuse the existing
    # schema when the caller did not name one -- an empty overwrite is a
    # request to keep the table and drop its rows, not to reshape it.
    effective_schema = schema
    if effective_schema is None:
        import lance

        effective_schema = lance.dataset(
            dataset_uri,
            storage_options=(
                dict(spec.storage_options) if spec.storage_options else None
            ),
        ).schema

    logger.info(
        "Overwrite of %s had no rows to write; replacing it with an empty "
        "table rather than leaving its previous contents in place.",
        table,
    )
    # ``create_table`` is safe here in a way it is not for an empty *create*:
    # overwriting an existing dataset makes a new version of it, so the
    # manifest's ``enable_stable_row_ids`` and file format version carry over
    # (verified). A create has no manifest to inherit from, which is why that
    # path writes through lance directly instead.
    connect(spec).create_table(table, schema=effective_schema, mode="overwrite")


def _create_empty_dataset(
    dataset_uri: str,
    schema: pa.Schema,
    spec: LanceDBConnectionSpec,
    *,
    data_storage_version: Optional[str],
    enable_stable_row_ids: bool,
) -> None:
    """Materialise a zero-row Lance dataset the database can open.

    Written through ``lance`` rather than ``create_table`` for the sake of the
    options. ``enable_stable_row_ids`` is fixed when a dataset is created and
    ``create_table`` has no parameter for it, so routing the empty case through
    the database would quietly produce a table that can never have stable row
    IDs -- the exact outcome the option exists to let a caller avoid, and one
    they would discover only when a later job needed the IDs. Writing Lance
    files at this URI and letting the database resolve them afterwards is also
    what the non-empty path already does.
    """
    import lance

    lance.write_dataset(
        pa.Table.from_pylist([], schema=schema),
        dataset_uri,
        mode="overwrite",
        # lance types this as a Literal over the versions it knows today.
        # Mirroring that set here would reject a version a newer lance accepts,
        # so pass the string through and let lance reject a bad one -- which it
        # does, by name.
        data_storage_version=cast("Any", data_storage_version),
        enable_stable_row_ids=enable_stable_row_ids,
        storage_options=dict(spec.storage_options) if spec.storage_options else None,
    )
