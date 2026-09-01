# SPDX-License-Identifier: Apache-2.0
"""Upsert scenarios: the shuffle, the duplicate-key trap, and idempotency."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from .. import datagen
from ..checks import id_stats, read_rows, verify_id_space
from ..harness import BenchRun
from . import register
from ._common import TABLE, collect_counters, expected_bytes, read_back, seed

#: Marks a row as having come from the upsert rather than the original write,
#: so "did the update actually apply" is answerable rather than assumed.
_UPDATED = "updated-"


def _updated_labels(ds: Any) -> Any:
    """Rewrite ``label`` so updated rows are distinguishable from original ones."""

    def relabel(batch: pa.Table) -> pa.Table:
        ids = batch.column("id").to_pylist()
        return batch.set_column(
            batch.schema.get_field_index("label"),
            "label",
            pa.array([f"{_UPDATED}{i}" for i in ids], pa.string()),
        )

    return ds.map_batches(relabel, batch_format="pyarrow").materialize()


@register(
    "upsert_merge",
    group="upsert",
    description="Merge-insert with and without the key-partitioning shuffle",
    backends=("local", "fake", "s3"),
)
def upsert_merge(run: BenchRun, backend: str) -> None:
    dataset = "vector"
    rows = run.rows(dataset)
    # Half the upsert overlaps the existing table and half is new, so both arms
    # of merge-insert are exercised by the same write.
    overlap = rows // 2
    upsert_rows = rows
    final_rows = rows + overlap

    for partition in (True, False):
        with run.case(
            f"upsert_merge.partition_{str(partition).lower()}",
            backend=backend,
            dataset=dataset,
            params={"partition_on_keys": partition},
        ) as case:
            case.set_volume(upsert_rows, expected_bytes(dataset, upsert_rows))
            source = _updated_labels(
                datagen.build_dataset(
                    dataset, upsert_rows, blocks=run.blocks, start=overlap
                )
            )

            def build(db_dir: str) -> str:
                return seed(case, db_dir, dataset, rows, blocks=run.blocks)

            def run_upsert(
                db_dir: str, partition: bool = partition, source: Any = source
            ) -> str:
                from lancedb_ray import write_lancedb

                uri = case.uri(db_dir)
                write_lancedb(
                    source,
                    TABLE,
                    uri=uri,
                    mode="upsert",
                    on="id",
                    partition_on_keys=partition,
                    **case.connect_kwargs,
                )
                return uri

            outcome = case.measure(run_upsert, fresh=True, setup=build)
            uri = outcome.value

            collect_counters(case, uri)
            ds = read_back(case, uri)
            stats = id_stats(ds.select_columns(["id"]))
            case.add_counters(stats.as_dict())
            # A merge-insert must converge on the union of the key spaces, not
            # the sum of the row counts.
            verify_id_space(
                case.checks, stats, num_rows=final_rows, name="merged id space"
            )

            probe = case.probe("merge_insert")
            case.add_counters(probe, prefix="merge_")

            # An id below the overlap was never in the upsert; one above it was.
            untouched = read_rows(uri, TABLE, [0], connect_kwargs=case.connect_kwargs)
            updated = read_rows(
                uri, TABLE, [final_rows - 1], connect_kwargs=case.connect_kwargs
            )
            if untouched.num_rows and updated.num_rows:
                case.checks.that(
                    "untouched row kept its original value",
                    not untouched.column("label")[0].as_py().startswith(_UPDATED),
                    expected="original label",
                    actual=untouched.column("label")[0].as_py(),
                )
                case.checks.that(
                    "upserted row carries the new value",
                    updated.column("label")[0].as_py().startswith(_UPDATED),
                    expected=f"{_UPDATED}*",
                    actual=updated.column("label")[0].as_py(),
                )


@register(
    "upsert_duplicate_keys",
    group="upsert",
    description="A repeated key is rejected, not silently written twice",
    backends=("local", "fake", "s3"),
    tiers=("smoke", "ci", "local", "full"),
)
def upsert_duplicate_keys(run: BenchRun, backend: str) -> None:
    """The trap the key-partitioning shuffle exists to close.

    Two tasks each holding one row for the same key would each find it absent
    and each insert it. Partitioned, the rows land in one task and LanceDB
    rejects the ambiguous merge -- a loud error instead of a duplicated key.
    """
    dataset = "narrow"
    rows = min(20_000, run.rows(dataset))

    with run.case("upsert_duplicate_keys", backend=backend, dataset=dataset) as case:
        case.set_volume(rows)
        import ray

        base = datagen.build_table(dataset, rows)
        # Every id appears exactly twice, spread across blocks.
        duplicated = pa.concat_tables([base, base])
        source = ray.data.from_arrow(duplicated).repartition(run.blocks).materialize()

        def build(db_dir: str) -> str:
            return seed(case, db_dir, dataset, rows, blocks=run.blocks)

        def run_upsert(db_dir: str) -> dict[str, Any]:
            from lancedb_ray import write_lancedb

            uri = case.uri(db_dir)
            try:
                write_lancedb(
                    source,
                    TABLE,
                    uri=uri,
                    mode="upsert",
                    on="id",
                    partition_on_keys=True,
                    **case.connect_kwargs,
                )
            except Exception as exc:  # the documented, wanted outcome
                return {"uri": uri, "raised": type(exc).__name__}
            return {"uri": uri, "raised": ""}

        outcome = case.measure(run_upsert, fresh=True, setup=build, warmup=0, repeat=1)
        result = outcome.value
        uri = result["uri"]

        case.counter("raised", result["raised"] or "none")
        if result["raised"]:
            case.checks.that(
                "ambiguous merge was rejected",
                True,
                expected="an error",
                actual=result["raised"],
            )
        else:
            # It did not raise, so the only acceptable outcome is that the table
            # is still keyed -- no id may appear twice.
            ds = read_back(case, uri)
            stats = id_stats(ds.select_columns(["id"]))
            case.add_counters(stats.as_dict())
            case.checks.equals(
                "no key was duplicated", stats.count, rows, detail="rows after upsert"
            )


@register(
    "upsert_idempotency",
    group="upsert",
    description="Replaying the same upsert converges on the same table",
    backends=("local", "fake", "s3"),
    tiers=("smoke", "ci", "local", "full"),
)
def upsert_idempotency(run: BenchRun, backend: str) -> None:
    """The basis of the exactly-once claim: merge-insert is idempotent.

    The second application is timed; what matters is that the table after it is
    indistinguishable from the table after the first.
    """
    dataset = "narrow"
    rows = min(100_000, run.rows(dataset))

    with run.case("upsert_idempotency", backend=backend, dataset=dataset) as case:
        case.set_volume(rows, expected_bytes(dataset, rows))
        source = _updated_labels(
            datagen.build_dataset(dataset, rows, blocks=run.blocks, start=rows // 2)
        )
        first_state: dict[str, Any] = {}

        def build(db_dir: str) -> str:
            from lancedb_ray import write_lancedb

            uri = seed(case, db_dir, dataset, rows, blocks=run.blocks)
            write_lancedb(
                source, TABLE, uri=uri, mode="upsert", on="id", **case.connect_kwargs
            )
            ds = read_back(case, uri)
            first_state["stats"] = id_stats(ds.select_columns(["id"]))
            return uri

        def replay(db_dir: str) -> str:
            from lancedb_ray import write_lancedb

            uri = case.uri(db_dir)
            write_lancedb(
                source, TABLE, uri=uri, mode="upsert", on="id", **case.connect_kwargs
            )
            return uri

        outcome = case.measure(replay, fresh=True, setup=build, warmup=0, repeat=1)
        uri = outcome.value

        ds = read_back(case, uri)
        after = id_stats(ds.select_columns(["id"]))
        before = first_state["stats"]
        case.add_counters(after.as_dict())
        case.checks.equals(
            "replay left the row count unchanged", after.count, before.count
        )
        case.checks.equals(
            "replay left the id sum unchanged", after.total, before.total
        )

        expected_rows = rows + rows // 2
        verify_id_space(
            case.checks, after, num_rows=expected_rows, name="replayed id space"
        )
