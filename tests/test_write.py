"""Write tests, run against both the local and Cloud/Enterprise backends."""

from __future__ import annotations

from typing import Any

import lance
import pyarrow as pa
import pytest
import ray
from lancedb_ray import write_lancedb

from conftest import Backend, make_table


def version_count(db_dir: str, name: str = "items") -> int:
    """Number of committed versions of a table's backing Lance dataset."""
    versions = lance.dataset(f"{db_dir}/{name}.lance").versions()  # type: ignore[no-untyped-call]
    return len(versions)


def dataset_of(table: pa.Table, blocks: int = 1) -> ray.data.Dataset:
    ds = ray.data.from_arrow(table)
    return ds.repartition(blocks) if blocks > 1 else ds


class TestWriteModes:
    def test_create(self, backend: Backend) -> None:
        write_lancedb(
            dataset_of(make_table(300)),
            "items",
            uri=backend.uri,
            mode="create",
            **backend.kwargs,
        )
        assert backend.count("items") == 300

    def test_create_refuses_an_existing_table(self, backend: Backend) -> None:
        backend.create("items", make_table(10))
        with pytest.raises(ValueError, match="already exists"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=backend.uri,
                mode="create",
                **backend.kwargs,
            )

    def test_append(self, backend: Backend) -> None:
        backend.create("items", make_table(100))
        write_lancedb(
            dataset_of(make_table(50, start=100)),
            "items",
            uri=backend.uri,
            mode="append",
            **backend.kwargs,
        )
        assert backend.count("items") == 150

    def test_append_creates_a_missing_table(self, backend: Backend) -> None:
        write_lancedb(
            dataset_of(make_table(25)),
            "items",
            uri=backend.uri,
            mode="append",
            **backend.kwargs,
        )
        assert backend.count("items") == 25

    def test_overwrite_replaces_all_rows(self, backend: Backend) -> None:
        backend.create("items", make_table(500))
        write_lancedb(
            dataset_of(make_table(20)),
            "items",
            uri=backend.uri,
            mode="overwrite",
            **backend.kwargs,
        )
        assert backend.count("items") == 20

    def test_rejects_an_unknown_mode(self, backend: Backend) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            write_lancedb(
                dataset_of(make_table(5)),
                "items",
                uri=backend.uri,
                mode="bogus",  # type: ignore[arg-type]
                **backend.kwargs,
            )

    def test_writing_an_empty_dataset(self, backend: Backend) -> None:
        backend.create("items", make_table(10))
        write_lancedb(
            dataset_of(make_table(0)),
            "items",
            uri=backend.uri,
            mode="append",
            **backend.kwargs,
        )
        assert backend.count("items") == 10


class TestRoundTrip:
    def test_content_survives_a_round_trip(self, backend: Backend) -> None:
        source = make_table(200)
        write_lancedb(
            dataset_of(source, blocks=4),
            "items",
            uri=backend.uri,
            mode="create",
            **backend.kwargs,
        )

        result = backend.rows("items")
        assert result.num_rows == 200
        assert sorted(result.column("id").to_pylist()) == list(range(200))
        got = dict(
            zip(
                result.column("id").to_pylist(),
                result.column("label").to_pylist(),
                strict=False,
            )
        )
        want = dict(
            zip(
                source.column("id").to_pylist(),
                source.column("label").to_pylist(),
                strict=False,
            )
        )
        assert got == want

    def test_vector_values_are_preserved(self, backend: Backend) -> None:
        source = make_table(50)
        write_lancedb(
            dataset_of(source),
            "items",
            uri=backend.uri,
            mode="create",
            **backend.kwargs,
        )
        result = backend.rows("items").sort_by("id")
        # Flatten: pytest.approx does not handle nested sequences.
        got = [v for row in result.column("vector").to_pylist() for v in row]
        want = [v for row in source.column("vector").to_pylist() for v in row]
        assert got == pytest.approx(want)


class TestDistributedLocalWrite:
    """The local fragment path is the reason this library is worth using."""

    def test_parallel_write_commits_exactly_one_version(self, db_dir: str) -> None:
        write_lancedb(
            dataset_of(make_table(8000), blocks=8),
            "items",
            uri=db_dir,
            mode="create",
            max_rows_per_file=2000,
            min_rows_per_file=500,
        )

        dataset = lance.dataset(f"{db_dir}/items.lance")
        # Eight workers wrote concurrently, but the table advanced by a single
        # version: the fragments were committed in one transaction.
        assert version_count(db_dir) == 1
        assert dataset.count_rows() == 8000

    def test_parallel_write_produces_multiple_fragments(self, db_dir: str) -> None:
        write_lancedb(
            dataset_of(make_table(8000), blocks=8),
            "items",
            uri=db_dir,
            mode="create",
            max_rows_per_file=2000,
            min_rows_per_file=500,
        )
        dataset = lance.dataset(f"{db_dir}/items.lance")
        # Multiple fragments prove the write actually fanned out.
        assert len(dataset.get_fragments()) > 1

    def test_append_adds_exactly_one_version(self, db_dir: str) -> None:
        write_lancedb(dataset_of(make_table(1000)), "items", uri=db_dir, mode="create")
        before = version_count(db_dir)

        write_lancedb(
            dataset_of(make_table(1000, start=1000), blocks=4),
            "items",
            uri=db_dir,
            mode="append",
        )
        after = version_count(db_dir)
        assert after == before + 1

    def test_written_table_is_visible_to_lancedb(self, db_dir: str) -> None:
        import lancedb

        write_lancedb(dataset_of(make_table(100)), "items", uri=db_dir, mode="create")
        assert "items" in lancedb.connect(db_dir).table_names()


class TestWriteStrategySelection:
    def test_forcing_the_api_path_still_writes_correctly(self, db_dir: str) -> None:
        write_lancedb(
            dataset_of(make_table(100)),
            "items",
            uri=db_dir,
            mode="create",
            local_write_strategy="api",
        )
        assert lance.dataset(f"{db_dir}/items.lance").count_rows() == 100

    def test_forcing_the_fragment_path_on_remote_is_refused(
        self, remote_uri: str, remote_kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="Cloud/Enterprise"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=remote_uri,
                mode="create",
                local_write_strategy="fragment",
                **remote_kwargs,
            )

    def test_forcing_the_fragment_path_for_upsert_is_refused(self, db_dir: str) -> None:
        with pytest.raises(ValueError, match="upsert requires row matching"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=db_dir,
                mode="upsert",
                on="id",
                local_write_strategy="fragment",
            )

    def test_a_transform_still_gets_the_single_commit_path(self, db_dir: str) -> None:
        """transform_fn must not force a write onto the slower API path.

        It runs as its own Ray stage, so the write itself is still one atomic
        commit rather than a transaction per batch.
        """

        def shout(batch: pa.Table) -> pa.Table:
            labels = [str(v).upper() for v in batch.column("label").to_pylist()]
            return batch.set_column(
                batch.schema.get_field_index("label"), "label", pa.array(labels)
            )

        write_lancedb(
            dataset_of(make_table(2000), blocks=4),
            "items",
            uri=db_dir,
            mode="create",
            transform_fn=shout,
            local_write_strategy="fragment",
        )

        assert version_count(db_dir) == 1
        table = lance.dataset(f"{db_dir}/items.lance").to_table()
        assert all(v.startswith("ROW-") for v in table.column("label").to_pylist())

    def test_rejects_an_unknown_strategy(self, db_dir: str) -> None:
        with pytest.raises(ValueError, match="local_write_strategy must be"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=db_dir,
                mode="create",
                local_write_strategy="nope",  # type: ignore[arg-type]
            )


class TestTransformFn:
    def test_transform_is_applied_to_written_rows(self, backend: Backend) -> None:
        def shout(batch: pa.Table) -> pa.Table:
            labels = [str(v).upper() for v in batch.column("label").to_pylist()]
            return batch.set_column(
                batch.schema.get_field_index("label"), "label", pa.array(labels)
            )

        write_lancedb(
            dataset_of(make_table(40)),
            "items",
            uri=backend.uri,
            mode="create",
            transform_fn=shout,
            **backend.kwargs,
        )

        labels = backend.rows("items").column("label").to_pylist()
        assert all(label.startswith("ROW-") for label in labels)

    def test_transform_can_add_a_column(self, backend: Backend) -> None:
        schema = make_table(1).schema.append(pa.field("extra", pa.int64()))

        def add_extra(batch: pa.Table) -> pa.Table:
            return batch.append_column(
                "extra", pa.array([7] * batch.num_rows, pa.int64())
            )

        write_lancedb(
            dataset_of(make_table(30)),
            "items",
            uri=backend.uri,
            mode="create",
            schema=schema,
            transform_fn=add_extra,
            **backend.kwargs,
        )

        assert set(backend.rows("items").column("extra").to_pylist()) == {7}


class TestTransactionEfficiency:
    """One transaction per write task, not one per batch.

    Every LanceDB transaction is a new table version and at least one fragment.
    Writing per batch leaves a table full of tiny fragments and, against
    Cloud/Enterprise, burns request quota -- so the sink hands each task's rows
    to LanceDB as a single streamed write.
    """

    def test_api_path_writes_one_version_per_task(self, db_dir: str) -> None:
        write_lancedb(dataset_of(make_table(1)), "items", uri=db_dir, mode="create")
        before = version_count(db_dir)

        # 20 blocks through the API path; Ray bundles them into few tasks.
        write_lancedb(
            dataset_of(make_table(20_000, start=1), blocks=20),
            "items",
            uri=db_dir,
            mode="append",
            local_write_strategy="api",
            concurrency=2,
        )

        added = version_count(db_dir) - before
        # Bounded by task count, not the 20 input blocks.
        assert added <= 2, f"{added} versions for a 2-task write"

    def test_small_max_rows_per_request_reintroduces_transactions(
        self, db_dir: str
    ) -> None:
        """The memory-ceiling knob costs transactions; that's the trade-off."""
        write_lancedb(dataset_of(make_table(1)), "items", uri=db_dir, mode="create")
        before = version_count(db_dir)

        write_lancedb(
            dataset_of(make_table(1000, start=1)),
            "items",
            uri=db_dir,
            mode="append",
            local_write_strategy="api",
            max_rows_per_request=100,
            concurrency=1,
        )

        assert version_count(db_dir) - before >= 10

    def test_fragment_count_stays_low_for_a_bulk_append(self, db_dir: str) -> None:
        write_lancedb(
            dataset_of(make_table(20_000), blocks=20),
            "items",
            uri=db_dir,
            mode="create",
            local_write_strategy="api",
            concurrency=2,
        )
        fragments = len(lance.dataset(f"{db_dir}/items.lance").get_fragments())
        assert fragments <= 4, f"{fragments} fragments for a 2-task write"


class TestUpsertHashPartitioning:
    """Parallel merge-inserts must not silently duplicate a key.

    Two tasks each holding one row for key K will each find K absent and each
    insert it. Neither source is internally ambiguous, so LanceDB accepts both
    and the key ends up twice. Hash-partitioning on the key columns puts every
    row for K in one task, where LanceDB rejects the ambiguity instead.
    """

    def test_unique_keys_across_many_tasks_produce_no_duplicates(
        self, backend: Backend
    ) -> None:
        import collections

        backend.create("items", make_table(200))

        write_lancedb(
            ray.data.from_arrow(make_table(400)).repartition(8),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            concurrency=4,
            **backend.kwargs,
        )

        ids = backend.rows("items").column("id").to_pylist()
        duplicates = {k: c for k, c in collections.Counter(ids).items() if c > 1}
        assert not duplicates, f"duplicate keys after parallel upsert: {duplicates}"
        assert len(ids) == 400

    def test_repeated_source_keys_raise_instead_of_duplicating(
        self, backend: Backend
    ) -> None:
        """A key repeated in the source is an error, not a silent duplicate."""
        backend.create("items", make_table(50))
        repeated = pa.concat_tables([make_table(25) for _ in range(4)])

        with pytest.raises(Exception, match="[Aa]mbiguous"):
            write_lancedb(
                ray.data.from_arrow(repeated).repartition(8),
                "items",
                uri=backend.uri,
                mode="upsert",
                on="id",
                concurrency=4,
                **backend.kwargs,
            )

    def test_partitioning_can_be_disabled(self, backend: Backend) -> None:
        """Opting out skips the shuffle; correctness is then the caller's."""
        backend.create("items", make_table(100))

        write_lancedb(
            ray.data.from_arrow(make_table(50)),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            partition_on_keys=False,
            concurrency=1,
            **backend.kwargs,
        )
        assert backend.count("items") == 100


class TestUpsertShuffleAvoidance:
    def test_a_single_task_upsert_skips_the_shuffle(self, db_dir: str) -> None:
        """One write task cannot race with itself, so the guard is pure cost."""
        import lancedb_ray.io as io_mod

        calls: list[object] = []
        original = io_mod._hash_partition

        def spy(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return original(*args, **kwargs)

        io_mod._hash_partition = spy  # type: ignore[assignment]
        try:
            backend_uri = db_dir
            write_lancedb(
                dataset_of(make_table(50)), "items", uri=backend_uri, mode="create"
            )
            write_lancedb(
                dataset_of(make_table(20)),
                "items",
                uri=backend_uri,
                mode="upsert",
                on="id",
                concurrency=1,
            )
            assert not calls, "single-task upsert should not shuffle"
        finally:
            io_mod._hash_partition = original  # type: ignore[assignment]

    def test_a_parallel_upsert_still_shuffles(self, db_dir: str) -> None:
        import lancedb_ray.io as io_mod

        calls: list[object] = []
        original = io_mod._hash_partition

        def spy(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return original(*args, **kwargs)

        io_mod._hash_partition = spy  # type: ignore[assignment]
        try:
            write_lancedb(
                dataset_of(make_table(50)), "items", uri=db_dir, mode="create"
            )
            write_lancedb(
                dataset_of(make_table(20)),
                "items",
                uri=db_dir,
                mode="upsert",
                on="id",
                concurrency=4,
            )
            assert calls, "a parallel upsert must keep the duplicate guard"
        finally:
            io_mod._hash_partition = original  # type: ignore[assignment]


class TestHashPartitionEdges:
    """``_hash_partition`` guards upserts; its degenerate inputs still matter."""

    def test_no_keys_passes_the_dataset_through_untouched(self) -> None:
        from lancedb_ray.io import _hash_partition

        ds = dataset_of(make_table(10))
        assert _hash_partition(ds, None, 4) is ds
        assert _hash_partition(ds, [], 4) is ds

    @pytest.mark.parametrize("concurrency", [0, -1])
    def test_non_positive_concurrency_falls_back_to_a_valid_block_count(
        self, concurrency: int
    ) -> None:
        """Ray rejects a repartition into zero blocks, so it must be clamped."""
        from lancedb_ray.io import _hash_partition

        partitioned = _hash_partition(dataset_of(make_table(20)), ["id"], concurrency)
        assert partitioned.materialize().count() == 20

    def test_a_lazy_dataset_still_partitions(self) -> None:
        """num_blocks() raises on a lazy dataset; the fallback must cope."""
        from lancedb_ray.io import _hash_partition

        lazy = dataset_of(make_table(30)).map_batches(lambda b: b)
        partitioned = _hash_partition(lazy, ["id"], None)
        assert partitioned.materialize().count() == 30


class TestFragmentWriteVerification:
    def test_a_dataset_the_database_cannot_open_is_reported(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fragment write the catalog cannot resolve must not pass silently.

        Rows on disk that LanceDB cannot open are worse than an outright
        failure, because nothing signals that anything went wrong. Simulated
        by letting the fragment write report success while writing nothing.
        """
        import lance_ray

        monkeypatch.setattr(lance_ray, "write_lance", lambda *args, **kwargs: None)

        with pytest.raises(RuntimeError, match="cannot open a table"):
            write_lancedb(
                dataset_of(make_table(20)), "items", uri=db_dir, mode="create"
            )
