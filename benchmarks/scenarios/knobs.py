# SPDX-License-Identifier: Apache-2.0
"""Tuning-knob sweeps.

Each of these exists because the README claims the knob does something. The
sweep measures what it costs and asserts that it actually did what it says --
a ceiling that does not bound anything is a documentation bug.
"""

from __future__ import annotations

from typing import Any

from .. import counters as counters_mod
from .. import datagen
from ..harness import BenchRun
from . import ALL_TIERS, register
from ._common import (
    TABLE,
    collect_counters,
    expected_bytes,
    read_back,
    seed,
    verify_roundtrip,
)


def _api_kwargs(backend: str) -> dict[str, Any]:
    """Force the table-API path, which is where the request ceilings apply.

    A local append would otherwise take the fragment path, where there are no
    requests to bound.
    """
    return {} if backend == "fake" else {"local_write_strategy": "api"}


@register(
    "knob_rows_per_transaction",
    group="knobs",
    description="Transaction size: the knob that sets a write task's peak memory",
    tiers=ALL_TIERS,
)
def knob_rows_per_transaction(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)
    sizes = (
        (64 * 1024, 256 * 1024, 1024 * 1024)
        if run.tier.full_sweeps
        else (64 * 1024, 256 * 1024)
    )

    for size in sizes:
        with run.case(
            f"knob_rows_per_transaction.{size}",
            backend=backend,
            dataset=dataset,
            params={"rows_per_transaction": size},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))
            source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

            def run_write(db_dir: str, size: int = size, source: Any = source) -> str:
                from lancedb_ray import write_lancedb

                uri = case.uri(db_dir)
                write_lancedb(
                    source,
                    TABLE,
                    uri=uri,
                    mode="create",
                    rows_per_transaction=size,
                    **_api_kwargs(backend),
                    **case.connect_kwargs,
                )
                return uri

            outcome = case.measure(run_write, fresh=True)
            uri = outcome.value

            probe = case.probe("add")
            case.add_counters(probe, prefix="add_")
            collect_counters(case, uri)
            if probe["max_rows"]:
                # Ray bundles whole blocks, so a transaction is the smallest
                # number of blocks that reaches the target -- it overshoots by
                # up to one block and never by more. This is the invariant worth
                # gating on; a plain ceiling would be asserting something the
                # knob does not promise.
                block_rows = -(-rows // run.blocks)
                case.counter("block_rows", block_rows)
                case.counter("overshoot_rows", probe["max_rows"] - size)
                case.checks.at_most(
                    "transaction overshoots by at most one block",
                    probe["max_rows"],
                    size + block_rows,
                    detail=f"blocks are {block_rows} rows",
                )
            verify_roundtrip(case, uri, dataset, rows, sample=64)


@register(
    "knob_max_bytes_per_request",
    group="knobs",
    description="The byte ceiling that bounds a wide schema's request payload",
    tiers=ALL_TIERS,
)
def knob_max_bytes_per_request(run: BenchRun, backend: str) -> None:
    """Rows are a poor proxy for size once a schema is wide.

    Measured on 1536-dimension embeddings, where 256K rows is 1.5GB before
    anything else in the row is counted.
    """
    dataset = "wide_vector"
    rows = run.rows(dataset)
    ceilings = (
        (8 * 1024**2, 32 * 1024**2, 128 * 1024**2)
        if run.tier.full_sweeps
        else (8 * 1024**2, 32 * 1024**2)
    )

    for ceiling in ceilings:
        with run.case(
            f"knob_max_bytes_per_request.{ceiling // 1024**2}MB",
            backend=backend,
            dataset=dataset,
            params={"max_bytes_per_request": ceiling},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))
            source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

            def run_write(
                db_dir: str, ceiling: int = ceiling, source: Any = source
            ) -> str:
                from lancedb_ray import write_lancedb

                uri = case.uri(db_dir)
                write_lancedb(
                    source,
                    TABLE,
                    uri=uri,
                    mode="create",
                    max_bytes_per_request=ceiling,
                    **_api_kwargs(backend),
                    **case.connect_kwargs,
                )
                return uri

            outcome = case.measure(run_write, fresh=True)
            uri = outcome.value

            probe = case.probe("add")
            case.add_counters(probe, prefix="add_")
            collect_counters(case, uri)
            if probe["count"]:
                # Arrow's in-memory size is not exactly the encoded payload, so
                # a small overshoot is expected; an unbounded one is the bug.
                case.checks.at_most(
                    "requests stayed within the byte ceiling",
                    probe["max_bytes"],
                    ceiling * 1.5,
                    detail=f"largest request was {probe['max_bytes'] / 1024**2:.1f}MB",
                )
                case.checks.at_least(
                    "the ceiling actually split the write", probe["count"], 1
                )
            verify_roundtrip(case, uri, dataset, rows, sample=32)


@register(
    "knob_max_rows_per_request",
    group="knobs",
    description="The row ceiling splits one task into several transactions",
    tiers=ALL_TIERS,
)
def knob_max_rows_per_request(run: BenchRun, backend: str) -> None:
    dataset = "narrow"
    rows = run.rows(dataset)
    ceiling = max(1_000, rows // 8)

    with run.case(
        "knob_max_rows_per_request",
        backend=backend,
        dataset=dataset,
        params={"max_rows_per_request": ceiling},
    ) as case:
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
                max_rows_per_request=ceiling,
                **_api_kwargs(backend),
                **case.connect_kwargs,
            )
            return uri

        outcome = case.measure(run_write, fresh=True)
        uri = outcome.value

        probe = case.probe("add")
        case.add_counters(probe, prefix="add_")
        if probe["max_rows"]:
            case.checks.at_most(
                "no request exceeded the row ceiling", probe["max_rows"], ceiling
            )
        verify_roundtrip(case, uri, dataset, rows, sample=64)


@register(
    "knob_write_concurrency",
    group="knobs",
    description="How write throughput scales with concurrent tasks",
    backends=("local",),
    tiers=ALL_TIERS,
)
def knob_write_concurrency(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)

    for concurrency in (1, 2, 4, 8):
        with run.case(
            f"knob_write_concurrency.{concurrency}",
            backend=backend,
            dataset=dataset,
            params={"concurrency": concurrency},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))
            source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

            def run_write(
                db_dir: str, concurrency: int = concurrency, source: Any = source
            ) -> str:
                from lancedb_ray import write_lancedb

                uri = case.uri(db_dir)
                write_lancedb(
                    source, TABLE, uri=uri, mode="create", concurrency=concurrency
                )
                return uri

            outcome = case.measure(run_write, fresh=True)
            found = collect_counters(case, outcome.value)
            if found is not None:
                # However many tasks ran, the table still advances once.
                case.checks.equals("single atomic commit", found.versions, 1)
                case.checks.equals("rows landed", found.rows, rows)


@register(
    "knob_file_layout",
    group="knobs",
    description="max_rows_per_file sets the fragment count, and the read pays for it",
    backends=("local",),
    tiers=ALL_TIERS,
)
def knob_file_layout(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)
    per_task = max(1, rows // run.blocks)
    divisors = (1, 4, 16) if run.tier.full_sweeps else (1, 8)

    for divisor in divisors:
        max_rows_per_file = max(1, per_task // divisor)
        with run.case(
            f"knob_file_layout.{divisor}x",
            backend=backend,
            dataset=dataset,
            params={"max_rows_per_file": max_rows_per_file},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))
            source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

            def run_write(
                db_dir: str, cap: int = max_rows_per_file, source: Any = source
            ) -> str:
                from lancedb_ray import write_lancedb

                uri = case.uri(db_dir)
                write_lancedb(
                    source,
                    TABLE,
                    uri=uri,
                    mode="create",
                    max_rows_per_file=cap,
                    # The default floor is 1024, which is above the ceiling at
                    # small tiers; the two have to move together.
                    min_rows_per_file=min(1024, cap),
                )
                return uri

            outcome = case.measure(run_write, fresh=True)
            uri = outcome.value
            found = collect_counters(case, uri)
            if found is not None:
                case.checks.equals("single atomic commit", found.versions, 1)
                case.checks.at_most(
                    "no fragment exceeded max_rows_per_file",
                    max(found.fragment_rows or [0]),
                    max_rows_per_file,
                )
                case.checks.equals("rows landed", found.rows, rows)

            # The layout only matters because of what a later read costs.
            def read_it(db_dir: str, uri: str = uri) -> Any:
                return read_back(case, uri).materialize()

            read_timer = run.case(
                f"knob_file_layout.{divisor}x.read",
                backend=backend,
                dataset=dataset,
                params={"max_rows_per_file": max_rows_per_file},
            )
            with read_timer as read_case:
                read_case.set_volume(rows, expected_bytes(dataset, rows))
                read_outcome = read_case.measure(read_it, fresh=False)
                read_case.checks.equals(
                    "rows read", int(read_outcome.value.count()), rows
                )
                read_case.counter("read_blocks", int(read_outcome.value.num_blocks()))
                if found is not None:
                    read_case.counter("fragments", found.fragments)


@register(
    "knob_storage_version",
    group="knobs",
    description="Lance file format version: write cost, size, and read cost",
    backends=("local",),
    tiers=ALL_TIERS,
)
def knob_storage_version(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)

    for version in ("stable", "2.1"):
        with run.case(
            f"knob_storage_version.{version}",
            backend=backend,
            dataset=dataset,
            params={"data_storage_version": version},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))
            source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

            def run_write(
                db_dir: str, version: str = version, source: Any = source
            ) -> str:
                from lancedb_ray import write_lancedb

                uri = case.uri(db_dir)
                write_lancedb(
                    source, TABLE, uri=uri, mode="create", data_storage_version=version
                )
                return uri

            try:
                outcome = case.measure(run_write, fresh=True)
            except Exception as exc:
                case.skip(f"data_storage_version={version!r} unsupported: {exc}")
                continue

            collect_counters(case, outcome.value)
            verify_roundtrip(case, outcome.value, dataset, rows, sample=64)


@register(
    "knob_stable_row_ids",
    group="knobs",
    description="What stable row IDs cost to write and to store",
    backends=("local",),
    tiers=ALL_TIERS,
)
def knob_stable_row_ids(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)

    for enabled in (False, True):
        with run.case(
            f"knob_stable_row_ids.{str(enabled).lower()}",
            backend=backend,
            dataset=dataset,
            params={"enable_stable_row_ids": enabled},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))
            source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

            def run_write(
                db_dir: str, enabled: bool = enabled, source: Any = source
            ) -> str:
                from lancedb_ray import write_lancedb

                uri = case.uri(db_dir)
                write_lancedb(
                    source,
                    TABLE,
                    uri=uri,
                    mode="create",
                    enable_stable_row_ids=enabled,
                )
                return uri

            outcome = case.measure(run_write, fresh=True)
            found = collect_counters(case, outcome.value)
            if found is not None:
                case.checks.equals("rows landed", found.rows, rows)
                case.checks.equals("single atomic commit", found.versions, 1)
            verify_roundtrip(case, outcome.value, dataset, rows, sample=64)


@register(
    "knob_scanner_options",
    group="knobs",
    description="Lance scanner options on a local read",
    backends=("local",),
    tiers=ALL_TIERS,
)
def knob_scanner_options(run: BenchRun, backend: str) -> None:
    dataset = "wide_scalar"
    rows = run.rows(dataset)
    variants: dict[str, dict[str, Any]] = {
        "default": {},
        "batch_8192": {"batch_size": 8192},
        "batch_65536": {"batch_size": 65536},
        "late_materialization": {"late_materialization": True},
    }
    if not run.tier.full_sweeps:
        variants.pop("late_materialization")

    for name, options in variants.items():
        with run.case(
            f"knob_scanner_options.{name}",
            backend=backend,
            dataset=dataset,
            params={"scanner_options": options},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))

            def build(db_dir: str) -> str:
                return seed(case, db_dir, dataset, rows, blocks=run.blocks)

            def run_read(db_dir: str, options: dict[str, Any] = options) -> Any:
                return read_back(
                    case, case.uri(db_dir), scanner_options=options or None
                ).materialize()

            outcome = case.measure(run_read, fresh=False, setup=build)
            uri = case.uri(outcome.work)
            case.checks.equals("rows read", int(outcome.value.count()), rows)

            scan = counters_mod.analyze_scan(
                uri, TABLE, scanner_options=dict(options) or None
            )
            if scan:
                case.add_counters(scan.as_dict())
