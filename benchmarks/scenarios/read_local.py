# SPDX-License-Identifier: Apache-2.0
"""Read scenarios: full scans, pushdown, type fidelity, and version pinning."""

from __future__ import annotations

from typing import Any

from .. import counters as counters_mod
from .. import datagen
from ..checks import (
    compare_tables,
    id_stats,
    sample_ids,
    verify_id_space,
    verify_schema,
)
from ..harness import BenchRun
from . import register
from ._common import TABLE, expected_bytes, read_back, seed, verify_roundtrip


def _base_schema(ds: Any) -> Any:
    schema = ds.schema()
    return getattr(schema, "base_schema", schema)


@register(
    "read_full",
    group="read",
    description="Full parallel scan of a table",
    backends=("local", "fake", "s3"),
)
def read_full(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)

    with run.case("read_full", backend=backend, dataset=dataset) as case:
        case.set_volume(rows, expected_bytes(dataset, rows))

        def build(db_dir: str) -> str:
            return seed(case, db_dir, dataset, rows, blocks=run.blocks)

        def run_read(db_dir: str) -> Any:
            return read_back(case, case.uri(db_dir)).materialize()

        outcome = case.measure(run_read, fresh=False, setup=build)
        materialized = outcome.value
        uri = case.uri(outcome.work)

        case.counter("read_blocks", int(materialized.num_blocks()))
        case.checks.equals("rows read", int(materialized.count()), rows)
        case.checks.at_least(
            "read fanned out", int(materialized.num_blocks()), 2, detail="output blocks"
        )
        verify_roundtrip(case, uri, dataset, rows)


@register(
    "read_projection",
    group="read",
    description="Column projection reaches the scan instead of being applied after",
)
def read_projection(run: BenchRun, backend: str) -> None:
    dataset = "wide_scalar"
    rows = run.rows(dataset)
    projected = ["id", "c00"]

    with run.case(
        "read_projection",
        backend=backend,
        dataset=dataset,
        params={"columns": projected},
    ) as case:
        case.set_volume(rows, expected_bytes(dataset, rows))

        def build(db_dir: str) -> str:
            return seed(case, db_dir, dataset, rows, blocks=run.blocks)

        def run_read(db_dir: str) -> Any:
            return read_back(case, case.uri(db_dir), columns=projected).materialize()

        outcome = case.measure(run_read, fresh=False, setup=build)
        materialized = outcome.value
        uri = case.uri(outcome.work)

        case.checks.equals("rows read", int(materialized.count()), rows)
        case.checks.equals(
            "only the projected columns came back",
            sorted(_base_schema(materialized).names),
            sorted(projected),
        )

        if backend != "local":
            # The scan counters below come from the Lance dataset the fake wraps,
            # not from the remote query path, so they say nothing about how a
            # real endpoint applies a projection.
            case.note(
                "scan counters describe the underlying Lance dataset, not the remote path"
            )

        full = counters_mod.analyze_scan(uri, TABLE)
        narrow = counters_mod.analyze_scan(uri, TABLE, columns=projected)
        if full and narrow and full.bytes_read:
            ratio = narrow.bytes_read / full.bytes_read
            case.counter("scan_bytes_full", full.bytes_read)
            case.counter("scan_bytes_projected", narrow.bytes_read)
            case.counter("scan_bytes_ratio", round(ratio, 4))
            # 2 of 41 columns. Anything close to 1.0 means the projection is
            # being applied after the read rather than pushed into it.
            case.checks.at_most(
                "projection pushed into the scan",
                ratio,
                0.25,
                detail="bytes read ratio",
            )

        ids = sample_ids(rows, count=128)
        from ..checks import read_rows

        got = read_rows(
            uri, TABLE, ids, connect_kwargs=case.connect_kwargs, columns=projected
        )
        case.checks.add(
            compare_tables(got, datagen.expected_rows(dataset, ids).select(projected))
        )


@register(
    "read_filter",
    group="read",
    description="Predicate pushdown returns exactly the selected rows",
    backends=("local", "fake", "s3"),
)
def read_filter(run: BenchRun, backend: str) -> None:
    dataset = "narrow"
    rows = run.rows(dataset)
    selectivities = (0.001, 0.1, 0.9) if run.tier.full_sweeps else (0.001, 0.5)

    db_holder: dict[str, str] = {}

    def build(db_dir: str) -> str:
        uri = seed(case, db_dir, dataset, rows, blocks=run.blocks)
        db_holder["uri"] = uri
        return uri

    for selectivity in selectivities:
        limit = max(1, int(rows * selectivity))
        predicate = f"id < {limit}"
        with run.case(
            f"read_filter.{selectivity:g}",
            backend=backend,
            dataset=dataset,
            params={"filter": predicate, "selectivity": selectivity},
        ) as case:
            case.set_volume(limit, expected_bytes(dataset, limit))

            def run_read(db_dir: str, predicate: str = predicate) -> Any:
                return read_back(case, case.uri(db_dir), filter=predicate).materialize()

            outcome = case.measure(run_read, fresh=False, setup=build)
            materialized = outcome.value
            uri = case.uri(outcome.work)

            case.checks.equals("filtered row count", int(materialized.count()), limit)
            stats = id_stats(materialized.select_columns(["id"]))
            verify_id_space(
                case.checks, stats, num_rows=limit, name="filtered id space"
            )

            scan = counters_mod.analyze_scan(uri, TABLE, filter=predicate)
            if scan:
                case.add_counters(scan.as_dict())
                case.checks.equals(
                    "scan returned the selected rows", int(scan.output_rows), limit
                )


@register(
    "read_fidelity",
    group="read",
    description="Types, nulls, unicode, inf/NaN and nested values survive the round trip",
    backends=("local", "fake", "s3"),
)
def read_fidelity(run: BenchRun, backend: str) -> None:
    dataset = "fidelity"
    rows = run.rows(dataset)

    with run.case("read_fidelity", backend=backend, dataset=dataset) as case:
        case.set_volume(rows, expected_bytes(dataset, rows))

        def build(db_dir: str) -> str:
            return seed(case, db_dir, dataset, rows, blocks=min(run.blocks, 4))

        def run_read(db_dir: str) -> Any:
            return read_back(case, case.uri(db_dir)).materialize()

        outcome = case.measure(run_read, fresh=False, setup=build)
        materialized = outcome.value
        uri = case.uri(outcome.work)

        case.checks.equals("rows read", int(materialized.count()), rows)
        verify_schema(case.checks, _base_schema(materialized), datagen.FIDELITY_SCHEMA)

        # Small enough to compare in full rather than by sample: this is the one
        # dataset where every value is chosen to break something.
        from ..checks import read_rows

        ids = list(range(rows))
        got = read_rows(uri, TABLE, ids, connect_kwargs=case.connect_kwargs)
        case.checks.add(compare_tables(got, datagen.expected_rows(dataset, ids)))


@register(
    "read_version_pinning",
    group="read",
    description="A writer landing mid-read cannot tear the result",
    backends=("local", "fake", "s3"),
)
def read_version_pinning(run: BenchRun, backend: str) -> None:
    """The guarantee that every read pins a version before planning shards.

    A read planned against N rows must still return N rows even though another
    write doubled the table in between. Without pinning, some shards would see
    the new rows and the result would be neither the old table nor the new one.
    """
    dataset = "narrow"
    rows = min(200_000, run.rows(dataset))

    with run.case("read_version_pinning", backend=backend, dataset=dataset) as case:
        case.set_volume(rows, expected_bytes(dataset, rows))
        extra = datagen.build_dataset(dataset, rows, blocks=run.blocks, start=rows)

        def build(db_dir: str) -> str:
            return seed(case, db_dir, dataset, rows, blocks=run.blocks)

        def run_read(db_dir: str) -> Any:
            from lancedb_ray import write_lancedb

            uri = case.uri(db_dir)
            # Planned here, against the current version.
            pinned = read_back(case, uri)
            # The table doubles before a single block has been read.
            write_lancedb(extra, TABLE, uri=uri, mode="append", **case.connect_kwargs)
            return pinned.materialize()

        outcome = case.measure(run_read, fresh=True, setup=build, warmup=0, repeat=1)
        materialized = outcome.value
        uri = case.uri(outcome.work)

        case.checks.equals(
            "read saw only the pinned version", int(materialized.count()), rows
        )
        stats = id_stats(materialized.select_columns(["id"]))
        verify_id_space(case.checks, stats, num_rows=rows, name="pinned id space")

        after = read_back(case, uri).materialize()
        case.checks.equals(
            "a later read does see the append", int(after.count()), rows * 2
        )
