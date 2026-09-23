# SPDX-License-Identifier: Apache-2.0
"""Write scenarios: fan-out, atomicity, and the modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import counters as counters_mod
from .. import datagen
from ..harness import BenchRun

if TYPE_CHECKING:
    from lancedb_ray.io import LocalWriteStrategy
from . import ALL_TIERS, register
from ._common import (
    TABLE,
    collect_counters,
    expected_bytes,
    read_back,
    seed,
    verify_roundtrip,
)


@register(
    "write_create",
    group="write",
    description="Create a table from a Ray Dataset -- the headline write path",
    backends=("local", "fake", "s3"),
)
def write_create(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)

    with run.case("write_create", backend=backend, dataset=dataset) as case:
        case.set_volume(rows, expected_bytes(dataset, rows))
        source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

        def run_write(db_dir: str) -> str:
            from lancedb_ray import write_lancedb

            uri = case.uri(db_dir)
            write_lancedb(
                source,
                TABLE,
                uri=uri,
                mode="create",
                max_rows_per_file=max(1, rows // run.blocks),
                **case.connect_kwargs,
            )
            return uri

        outcome = case.measure(run_write, fresh=True)
        uri = outcome.value

        found = collect_counters(case, uri)
        if found is not None:
            case.checks.equals("rows landed", found.rows, rows)
            if backend == "local":
                # The whole point of the fragment path: many tasks, one commit.
                case.checks.equals("single atomic commit", found.versions, 1)
                case.checks.at_least(
                    "write fanned out", found.fragments, 2, detail="fragments produced"
                )
            else:
                # The remote path is the table API: a create, then one
                # transaction per write task. Recorded, not asserted against the
                # fragment path's shape.
                case.note(
                    f"table-API path: {found.versions} versions, "
                    f"{found.fragments} fragments"
                )
        verify_roundtrip(case, uri, dataset, rows)


@register(
    "write_append",
    group="write",
    description="Append into an existing table -- one new version per write",
    backends=("local", "fake", "s3"),
)
def write_append(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)
    half = rows // 2

    with run.case("write_append", backend=backend, dataset=dataset) as case:
        case.set_volume(half, expected_bytes(dataset, half))
        second = datagen.build_dataset(dataset, half, blocks=run.blocks, start=half)

        before: dict[str, int] = {}

        def build(db_dir: str) -> str:
            uri = seed(case, db_dir, dataset, half, blocks=run.blocks)
            found = counters_mod.dataset_counters(uri, TABLE)
            before["versions"] = found.versions if found else 0
            return uri

        def run_write(db_dir: str) -> str:
            from lancedb_ray import write_lancedb

            uri = case.uri(db_dir)
            write_lancedb(second, TABLE, uri=uri, mode="append", **case.connect_kwargs)
            return uri

        outcome = case.measure(run_write, fresh=True, setup=build)
        uri = outcome.value

        found = collect_counters(case, uri)
        if found is not None:
            delta = found.versions - before.get("versions", 0)
            case.counter("versions_added", delta)
            case.checks.equals("rows landed", found.rows, rows)
            if backend == "local":
                # However many tasks wrote fragments, the table advances once.
                case.checks.equals("append added exactly one version", delta, 1)
            else:
                case.checks.at_least("append advanced the table", delta, 1)
        verify_roundtrip(case, uri, dataset, rows)


@register(
    "write_path_comparison",
    group="write",
    description="Fragment path vs table-API path for the same local append",
    backends=("local",),
    tiers=ALL_TIERS,
)
def write_path_comparison(run: BenchRun, backend: str) -> None:
    """The comparison the README makes: one transaction per task, not per batch.

    Both variants write identical data. The interesting output is not which is
    faster but how many versions and fragments each leaves behind, because that
    is what a later reader pays for.
    """
    dataset = "vector"
    rows = run.rows(dataset)
    source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

    strategies: tuple[LocalWriteStrategy, ...] = ("fragment", "api")
    for strategy in strategies:
        with run.case(
            f"write_path.{strategy}",
            backend=backend,
            dataset=dataset,
            params={"local_write_strategy": strategy},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))

            def run_write(db_dir: str, strategy: LocalWriteStrategy = strategy) -> str:
                from lancedb_ray import write_lancedb

                uri = case.uri(db_dir)
                write_lancedb(
                    source,
                    TABLE,
                    uri=uri,
                    mode="create",
                    local_write_strategy=strategy,
                )
                return uri

            outcome = case.measure(run_write, fresh=True)
            uri = outcome.value
            found = collect_counters(case, uri)
            if found is not None:
                case.checks.equals("rows landed", found.rows, rows)
                if strategy == "fragment":
                    case.checks.equals("single atomic commit", found.versions, 1)
                    case.checks.at_least("write fanned out", found.fragments, 2)
                else:
                    # Not a defect -- the API path is a transaction per task by
                    # design. Recorded so the cost of choosing it is visible.
                    case.note(
                        f"API path left {found.versions} versions and "
                        f"{found.fragments} fragments"
                    )
            verify_roundtrip(case, uri, dataset, rows)


@register(
    "write_overwrite",
    group="write",
    description="Overwrite replaces rather than appends, including from empty",
    backends=("local", "fake", "s3"),
)
def write_overwrite(run: BenchRun, backend: str) -> None:
    dataset = "narrow"
    rows = run.rows(dataset)
    replacement_rows = max(1, rows // 4)

    with run.case("write_overwrite", backend=backend, dataset=dataset) as case:
        case.set_volume(replacement_rows, expected_bytes(dataset, replacement_rows))
        replacement = datagen.build_dataset(
            dataset, replacement_rows, blocks=run.blocks, start=1_000_000
        )

        def build(db_dir: str) -> str:
            return seed(case, db_dir, dataset, rows, blocks=run.blocks)

        def run_write(db_dir: str) -> str:
            from lancedb_ray import write_lancedb

            uri = case.uri(db_dir)
            write_lancedb(
                replacement, TABLE, uri=uri, mode="overwrite", **case.connect_kwargs
            )
            return uri

        outcome = case.measure(run_write, fresh=True, setup=build)
        uri = outcome.value

        found = collect_counters(case, uri)
        if found is not None:
            case.checks.equals(
                "overwrite replaced the rows", found.rows, replacement_rows
            )
        verify_roundtrip(
            case, uri, dataset, replacement_rows, id_start=1_000_000, check_schema=False
        )


@register(
    "write_overwrite_empty",
    group="write",
    description="An empty overwrite must empty the table, not silently keep it",
    tiers=ALL_TIERS,
    backends=("local", "fake", "s3"),
)
def write_overwrite_empty(run: BenchRun, backend: str) -> None:
    """Regression guard for a bug where an empty overwrite kept the old rows.

    Cheap and non-negotiable, so it runs in every tier. Timing is meaningless
    here; the assertion is the point.
    """
    dataset = "narrow"
    rows = min(50_000, run.rows(dataset))

    with run.case("write_overwrite_empty", backend=backend, dataset=dataset) as case:
        case.set_volume(0)
        import ray

        empty = ray.data.from_arrow(datagen.get_spec(dataset).schema.empty_table())

        def build(db_dir: str) -> str:
            # Stable row IDs are a fragment-path option; the table API rejects
            # them outright, so they are only part of this check locally.
            extra: dict[str, Any] = (
                {"enable_stable_row_ids": True} if backend == "local" else {}
            )
            return seed(case, db_dir, dataset, rows, blocks=run.blocks, **extra)

        def run_write(db_dir: str) -> str:
            from lancedb_ray import write_lancedb

            uri = case.uri(db_dir)
            write_lancedb(
                empty,
                TABLE,
                uri=uri,
                mode="overwrite",
                schema=datagen.get_spec(dataset).schema,
                **case.connect_kwargs,
            )
            return uri

        outcome = case.measure(run_write, fresh=True, setup=build, warmup=0, repeat=1)
        uri = outcome.value

        found = collect_counters(case, uri)
        if found is not None:
            case.checks.equals("empty overwrite emptied the table", found.rows, 0)

        dataset_handle = counters_mod.open_lance(uri, TABLE)
        if dataset_handle is not None:
            # Stable row IDs are fixed at creation, so an overwrite that
            # recreated the dataset would silently drop them.
            manifest = getattr(dataset_handle, "_ds", None)
            stable = None
            for attr in ("uses_stable_row_ids", "_uses_stable_row_ids"):
                if hasattr(dataset_handle, attr):
                    stable = bool(getattr(dataset_handle, attr))
                    break
            if stable is None and manifest is not None:
                stable = None
            if stable is not None:
                case.checks.that(
                    "stable row ids survived the empty overwrite",
                    stable,
                    expected=True,
                    actual=stable,
                )
            else:
                case.note("stable row-id flag not exposed by this Lance version")

        ds = read_back(case, uri)
        case.checks.equals("read back is empty", int(ds.count()), 0)
