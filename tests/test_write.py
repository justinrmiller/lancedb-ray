"""Write tests, run against both the local and Cloud/Enterprise backends."""

from __future__ import annotations

import json
import logging
from typing import Any

import lance
import pyarrow as pa
import pytest
import ray
from lancedb_ray import read_lancedb, write_lancedb
from ray.exceptions import RayTaskError

from conftest import Backend, make_table


def version_count(db_dir: str, name: str = "items") -> int:
    """Number of committed versions of a table's backing Lance dataset."""
    versions = lance.dataset(f"{db_dir}/{name}.lance").versions()  # type: ignore[no-untyped-call]
    return len(versions)


def file_versions(db_dir: str, name: str = "items") -> list[tuple[int, int]]:
    """The Lance file format version of each data file in a table."""
    ds = lance.dataset(f"{db_dir}/{name}.lance")  # type: ignore[no-untyped-call]
    versions = []
    for fragment in ds.get_fragments():
        # to_json() returns a dict on current lance and a JSON string on older
        # ones; accept either rather than pinning the test to one.
        raw = fragment.metadata.to_json()
        meta = raw if isinstance(raw, dict) else json.loads(raw)
        for data_file in meta["files"]:
            versions.append(
                (data_file["file_major_version"], data_file["file_minor_version"])
            )
    return versions


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


class TestHashShuffleRequirement:
    def test_a_sort_shuffle_strategy_is_refused_with_a_usable_message(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Key-based repartitioning only exists under a hash shuffle.

        Ray otherwise raises at plan time naming an internal config, and the
        co-location the shuffle exists to guarantee would be gone regardless.
        """
        from ray.data.context import DataContext, ShuffleStrategy

        Backend("local", db_dir, {}).create("items", make_table(5))
        monkeypatch.setattr(
            DataContext.get_current(),
            "shuffle_strategy",
            ShuffleStrategy.SORT_SHUFFLE_PULL_BASED,
        )
        with pytest.raises(ValueError, match="needs a hash-based shuffle"):
            write_lancedb(
                dataset_of(make_table(5)), "items", uri=db_dir, mode="upsert", on="id"
            )

    def test_the_escape_hatch_still_works_under_a_sort_shuffle(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import lancedb
        from ray.data.context import DataContext, ShuffleStrategy

        Backend("local", db_dir, {}).create("items", make_table(5))
        monkeypatch.setattr(
            DataContext.get_current(),
            "shuffle_strategy",
            ShuffleStrategy.SORT_SHUFFLE_PULL_BASED,
        )
        write_lancedb(
            dataset_of(make_table(5, start=3)),
            "items",
            uri=db_dir,
            mode="upsert",
            on="id",
            partition_on_keys=False,
        )
        assert lancedb.connect(db_dir).open_table("items").count_rows() == 8

    def test_the_default_strategy_is_accepted(self, db_dir: str) -> None:
        import lancedb

        Backend("local", db_dir, {}).create("items", make_table(5))
        write_lancedb(
            dataset_of(make_table(5, start=3)),
            "items",
            uri=db_dir,
            mode="upsert",
            on="id",
        )
        assert lancedb.connect(db_dir).open_table("items").count_rows() == 8


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


class TestUpsertPartitionCount:
    """The shuffle's fan-out must follow the transaction budget, not the input.

    A hash shuffle costs one fragment per (source block x output partition).
    Sizing the output from ``ds.num_blocks()`` made that product quadratic, so a
    dataset split finely for write throughput paid for it again on every upsert:
    at a fixed 8M rows, 1,024 fragments took ~4 minutes and 65,536 had not
    finished after 20.
    """

    def test_output_follows_rows_per_transaction_not_block_count(self) -> None:
        from lancedb_ray.io import _upsert_partition_count

        ds = dataset_of(make_table(1000), blocks=50).materialize()
        # 1000 rows at 100 per transaction wants 10 partitions, whatever the
        # source's own block count happens to be.
        assert _upsert_partition_count(ds, 4, 100) == 10

    def test_block_count_does_not_change_the_answer(self) -> None:
        """The regression: fan-out used to track the source's blocking."""
        from lancedb_ray.io import _upsert_partition_count

        coarse = dataset_of(make_table(1000), blocks=2).materialize()
        fine = dataset_of(make_table(1000), blocks=50).materialize()
        assert _upsert_partition_count(coarse, 4, 100) == _upsert_partition_count(
            fine, 4, 100
        )

    def test_a_coarse_source_is_split_finer_than_its_input(self) -> None:
        """A few huge blocks must still yield transaction-sized partitions."""
        from lancedb_ray.io import _upsert_partition_count

        ds = dataset_of(make_table(1000), blocks=2).materialize()
        assert _upsert_partition_count(ds, 1, 10) == 100

    def test_never_fewer_partitions_than_writers(self) -> None:
        """Fewer partitions than concurrent writers leaves writers idle."""
        from lancedb_ray.io import _upsert_partition_count

        ds = dataset_of(make_table(100), blocks=8).materialize()
        assert _upsert_partition_count(ds, 8, 10_000) == 8

    @pytest.mark.parametrize("concurrency", [None, 0, -1])
    def test_absent_concurrency_still_yields_a_valid_count(
        self, concurrency: object
    ) -> None:
        from lancedb_ray.io import _upsert_partition_count

        ds = dataset_of(make_table(100), blocks=4).materialize()
        assert _upsert_partition_count(ds, concurrency, 10_000) >= 1  # type: ignore[arg-type]

    def test_a_lazy_plan_is_never_counted(self) -> None:
        """Counting a lazy plan runs the source, which the repartition repeats."""
        from lancedb_ray.io import _DEFAULT_UPSERT_PARTITIONS, _upsert_partition_count

        counted = []

        class Lazy:
            def num_blocks(self) -> int:
                raise RuntimeError("plan not executed")

            def count(self) -> int:
                counted.append(1)  # pragma: no cover - must not be reached
                return 10_000_000

        got = _upsert_partition_count(Lazy(), 1, 256 * 1024)  # type: ignore[arg-type]
        assert got == _DEFAULT_UPSERT_PARTITIONS
        assert not counted, "count() must stay behind the num_blocks gate"

    def test_large_fan_out_is_warned_about(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from lancedb_ray.io import _warn_if_shuffle_is_large

        with caplog.at_level(logging.WARNING, logger="lancedb_ray.io"):
            _warn_if_shuffle_is_large(256, 256)
        assert "shuffle" in caplog.text
        assert "rows_per_transaction" in caplog.text

    def test_a_modest_fan_out_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        from lancedb_ray.io import _warn_if_shuffle_is_large

        with caplog.at_level(logging.WARNING, logger="lancedb_ray.io"):
            _warn_if_shuffle_is_large(16, 16)
        assert not caplog.text


class TestHashPartitionEdges:
    """``_hash_partition`` guards upserts; its degenerate inputs still matter."""

    def test_no_keys_passes_the_dataset_through_untouched(self) -> None:
        from lancedb_ray.io import _hash_partition

        ds = dataset_of(make_table(10))
        assert _hash_partition(ds, None, 4, 256 * 1024) is ds
        assert _hash_partition(ds, [], 4, 256 * 1024) is ds

    @pytest.mark.parametrize("concurrency", [0, -1])
    def test_non_positive_concurrency_falls_back_to_a_valid_block_count(
        self, concurrency: int
    ) -> None:
        """Ray rejects a repartition into zero blocks, so it must be clamped."""
        from lancedb_ray.io import _hash_partition

        partitioned = _hash_partition(
            dataset_of(make_table(20)), ["id"], concurrency, 256 * 1024
        )
        assert partitioned.materialize().count() == 20

    def test_a_lazy_dataset_still_partitions(self) -> None:
        """num_blocks() raises on a lazy dataset; the fallback must cope."""
        from lancedb_ray.io import _hash_partition

        lazy = dataset_of(make_table(30)).map_batches(lambda b: b)
        partitioned = _hash_partition(lazy, ["id"], None, 256 * 1024)
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


class TestEmptyOverwrite:
    """An overwrite that matched no rows must still empty the table.

    ``write_lance`` commits nothing for a zero-row input, so without help the
    local path leaves every previous row in place and reports success -- while
    Cloud/Enterprise, which replaces the table up front, empties it. The two
    backends have to agree, and they have to agree on the answer that does not
    silently republish stale data.
    """

    def test_an_empty_overwrite_empties_the_table(self, backend: Backend) -> None:
        backend.create("items", make_table(100))
        empty = make_table(0)

        write_lancedb(
            dataset_of(empty),
            "items",
            uri=backend.uri,
            mode="overwrite",
            schema=empty.schema,
            **backend.kwargs,
        )

        assert backend.count("items") == 0

    def test_an_empty_overwrite_without_a_schema_keeps_the_shape(
        self, db_dir: str
    ) -> None:
        """Dropping the rows is not a request to reshape the table."""
        import lancedb

        original = make_table(100)
        lancedb.connect(db_dir).create_table("items", data=original)

        write_lancedb(dataset_of(make_table(0)), "items", uri=db_dir, mode="overwrite")

        table = lancedb.connect(db_dir).open_table("items")
        assert table.count_rows() == 0
        assert set(table.schema.names) == set(original.schema.names)

    def test_an_empty_append_is_still_a_no_op(self, backend: Backend) -> None:
        """Only overwrite means "replace the contents"; append must not."""
        backend.create("items", make_table(10))

        write_lancedb(
            dataset_of(make_table(0)),
            "items",
            uri=backend.uri,
            mode="append",
            **backend.kwargs,
        )

        assert backend.count("items") == 10

    def test_a_non_empty_overwrite_still_replaces(self, backend: Backend) -> None:
        backend.create("items", make_table(500))

        write_lancedb(
            dataset_of(make_table(20)),
            "items",
            uri=backend.uri,
            mode="overwrite",
            **backend.kwargs,
        )

        assert backend.count("items") == 20

    def test_an_unreadable_version_leaves_the_table_alone(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uncertainty must not license deleting rows.

        The empty-overwrite recovery replaces a table's contents, so it may only
        run when both version probes actually answered. If either did not, the
        old rows stay and the caller is warned.
        """
        import lancedb
        import lancedb_ray.io as io_module

        lancedb.connect(db_dir).create_table("items", data=make_table(100))
        monkeypatch.setattr(io_module, "_dataset_version", lambda *a, **k: None)

        write_lancedb(
            dataset_of(make_table(0)),
            "items",
            uri=db_dir,
            mode="overwrite",
            schema=make_table(0).schema,
        )

        assert lancedb.connect(db_dir).open_table("items").count_rows() == 100


class TestWriteVerification:
    """The post-write check must never "recover" by destroying a good write.

    An empty input leaves no dataset at all, which is what makes the empty case
    safe to rebuild from the schema. Any other reason the database cannot
    resolve the table means the rows are on disk, and overwriting them with an
    empty table would lose a completed write while reporting success.
    """

    @staticmethod
    def _break_open_table(monkeypatch: pytest.MonkeyPatch) -> None:
        import lancedb_ray.io as io_module
        from lancedb_ray.connection import connect as real_connect

        class Wrapper:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def open_table(self, name: str, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("transient catalog hiccup")

        monkeypatch.setattr(
            io_module, "connect", lambda spec: Wrapper(real_connect(spec))
        )

    def test_a_failed_verification_keeps_the_rows(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arrow = make_table(300)
        self._break_open_table(monkeypatch)

        with pytest.raises(RuntimeError, match="resolution failure"):
            write_lancedb(
                dataset_of(arrow),
                "items",
                uri=db_dir,
                mode="create",
                schema=arrow.schema,
            )

        monkeypatch.undo()
        # The write completed; only the check after it failed. Losing the rows
        # here would be a silent, total data loss for the table.
        assert lance.dataset(f"{db_dir}/items.lance").count_rows() == 300  # type: ignore[no-untyped-call]

    def test_a_failed_verification_fails_loudly(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returning success would let a job believe a table it cannot see."""
        arrow = make_table(50)
        self._break_open_table(monkeypatch)

        with pytest.raises(RuntimeError, match="cannot open a table"):
            write_lancedb(
                dataset_of(arrow),
                "items",
                uri=db_dir,
                mode="create",
                schema=arrow.schema,
            )


class TestDatasetState:
    """The probe that decides whether overwriting a location is safe."""

    def test_absent_when_nothing_was_written(self, db_dir: str) -> None:
        from lancedb_ray.connection import LanceDBConnectionSpec
        from lancedb_ray.io import _dataset_state, _DatasetState

        spec = LanceDBConnectionSpec.create(db_dir)
        assert _dataset_state(f"{db_dir}/nothing.lance", spec) is _DatasetState.ABSENT

    def test_present_when_a_dataset_is_there(self, db_dir: str) -> None:
        from lancedb_ray.connection import LanceDBConnectionSpec
        from lancedb_ray.io import _dataset_state, _DatasetState

        write_lancedb(dataset_of(make_table(10)), "items", uri=db_dir, mode="create")
        spec = LanceDBConnectionSpec.create(db_dir)
        assert _dataset_state(f"{db_dir}/items.lance", spec) is _DatasetState.PRESENT

    def test_an_unreadable_location_is_unknown_not_absent(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uncertainty must never be reported as absent.

        ABSENT is what licenses an overwrite, so a storage layer that will not
        answer has to fall on the side that preserves data.
        """
        import lance
        from lancedb_ray.connection import LanceDBConnectionSpec
        from lancedb_ray.io import _dataset_state, _DatasetState

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise PermissionError("storage said no")

        monkeypatch.setattr(lance, "dataset", refuse)
        spec = LanceDBConnectionSpec.create(db_dir)
        assert _dataset_state(f"{db_dir}/items.lance", spec) is _DatasetState.UNKNOWN


class TestEmptyCreate:
    """A create whose input turns out to be empty still owes you a table.

    A dataset with no rows produces no fragments and therefore no manifest, so
    the fragment path leaves nothing the database can open. A job whose filter
    matched nothing still wants its table to exist.
    """

    def test_empty_dataset_creates_an_empty_table(self, db_dir: str) -> None:
        import lancedb

        schema = make_table(0).schema
        write_lancedb(
            dataset_of(make_table(0)), "items", uri=db_dir, mode="create", schema=schema
        )

        table = lancedb.connect(db_dir).open_table("items")
        assert table.count_rows() == 0
        assert set(table.schema.names) == set(schema.names)

    def test_the_empty_table_accepts_a_later_append(self, db_dir: str) -> None:
        import lancedb

        write_lancedb(
            dataset_of(make_table(0)),
            "items",
            uri=db_dir,
            mode="create",
            schema=make_table(0).schema,
        )
        write_lancedb(dataset_of(make_table(25)), "items", uri=db_dir, mode="append")

        assert lancedb.connect(db_dir).open_table("items").count_rows() == 25

    def test_empty_create_without_a_schema_uses_the_dataset_schema(
        self, db_dir: str
    ) -> None:
        """The default local path must not need a schema Ray already knows.

        The table API path creates the table from the input's schema, so the
        fragment path failing here made the same call succeed or fail purely on
        which strategy it happened to take.
        """
        import lancedb

        source = make_table(0)
        write_lancedb(dataset_of(source), "items", uri=db_dir, mode="create")

        table = lancedb.connect(db_dir).open_table("items")
        assert table.count_rows() == 0
        assert set(table.schema.names) == set(source.schema.names)

    def test_both_write_paths_agree_on_an_empty_create(self, db_dir: str) -> None:
        import lancedb

        for strategy, name in (("fragment", "frag"), ("api", "api")):
            write_lancedb(
                dataset_of(make_table(0)),
                name,
                uri=db_dir,
                mode="create",
                local_write_strategy=strategy,  # type: ignore[arg-type]
            )
        db = lancedb.connect(db_dir)
        assert db.open_table("frag").count_rows() == 0
        assert set(db.open_table("frag").schema.names) == set(
            db.open_table("api").schema.names
        )

    def test_without_any_schema_the_failure_is_explained(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import lance_ray
        from lancedb_ray import io as io_module

        # An empty input that produces nothing resolvable *and* a dataset Ray
        # cannot report a schema for: the error should say what to pass.
        monkeypatch.setattr(lance_ray, "write_lance", lambda *a, **k: None)
        monkeypatch.setattr(io_module, "_arrow_schema", lambda ds: None)
        with pytest.raises(RuntimeError, match="schema="):
            write_lancedb(dataset_of(make_table(0)), "items", uri=db_dir, mode="create")

    @pytest.mark.parametrize("schema_arg", [None, "explicit"])
    def test_rows_that_vanished_are_never_papered_over(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch, schema_arg: Any
    ) -> None:
        """An absent dataset is only benign when the input had no rows.

        Standing an empty table where a completed write should be reports
        success over the loss. Passing ``schema=`` used to be enough to reach
        that path with a non-empty input.
        """
        import lance_ray

        monkeypatch.setattr(lance_ray, "write_lance", lambda *a, **k: None)
        schema = make_table(0).schema if schema_arg else None
        with pytest.raises(RuntimeError, match="Refusing to replace it"):
            write_lancedb(
                dataset_of(make_table(20)),
                "items",
                uri=db_dir,
                mode="create",
                schema=schema,
            )


class TestRecoveryProbes:
    """The recovery path's probes must answer "cannot tell", never guess.

    Both feed the decision to stand an empty table at a URI, so a probe that
    reported a confident wrong answer would do it over a completed write.
    """

    def test_an_unreadable_schema_reports_cannot_tell(self) -> None:
        from lancedb_ray.io import _arrow_schema

        class Broken:
            def schema(self, fetch_if_missing: bool = False) -> Any:
                raise RuntimeError("plan could not be executed")

        assert _arrow_schema(Broken()) is None  # type: ignore[arg-type]

    def test_a_non_arrow_schema_reports_cannot_tell(self) -> None:
        from lancedb_ray.io import _arrow_schema

        class Simple:
            def schema(self, fetch_if_missing: bool = False) -> Any:
                return "not a schema"

        assert _arrow_schema(Simple()) is None  # type: ignore[arg-type]

    def test_an_uncountable_input_reports_cannot_tell(self) -> None:
        from lancedb_ray.io import _input_row_count

        class Broken:
            def count(self) -> int:
                raise RuntimeError("plan could not be executed")

        assert _input_row_count(Broken()) is None  # type: ignore[arg-type]

    def test_an_uncountable_input_is_never_papered_over(
        self, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import lance_ray
        from lancedb_ray import io as io_module

        monkeypatch.setattr(lance_ray, "write_lance", lambda *a, **k: None)
        monkeypatch.setattr(io_module, "_input_row_count", lambda ds: None)
        with pytest.raises(RuntimeError, match="input was not empty"):
            write_lancedb(
                dataset_of(make_table(0)),
                "items",
                uri=db_dir,
                mode="create",
                schema=make_table(0).schema,
            )


class TestFailedWriteAtomicity:
    """A distributed write that fails partway must not half-land.

    The fragment path is the reason the README can claim atomicity: workers
    write fragments independently and the driver commits once at the end, so a
    task failing before that commit leaves the table exactly as it was. This is
    the property that makes a failed job safe to simply re-run.
    """

    def _boom_after(self, threshold: int) -> Any:
        def transform(batch: pa.Table) -> pa.Table:
            if batch.num_rows and int(batch.column("id")[0].as_py()) >= threshold:
                raise RuntimeError("deliberate failure in one write task")
            return batch

        return transform

    def test_fragment_path_failure_commits_nothing(self, db_dir: str) -> None:
        write_lancedb(dataset_of(make_table(100)), "items", uri=db_dir, mode="create")
        rows_before = (
            version_count(db_dir),
            lance.dataset(f"{db_dir}/items.lance").count_rows(),
        )

        # Fail inside a Ray stage rather than transform_fn, which would divert
        # the write to the API path.
        failing = dataset_of(make_table(400, start=1000), blocks=4).map_batches(
            self._boom_after(1300), batch_format="pyarrow"
        )
        with pytest.raises(RayTaskError, match="deliberate failure"):
            write_lancedb(failing, "items", uri=db_dir, mode="append")

        rows_after = (
            version_count(db_dir),
            lance.dataset(f"{db_dir}/items.lance").count_rows(),
        )
        assert rows_after == rows_before, "a failed write must not commit anything"

    def test_the_table_is_still_usable_afterwards(self, db_dir: str) -> None:
        write_lancedb(dataset_of(make_table(100)), "items", uri=db_dir, mode="create")

        failing = dataset_of(make_table(400, start=1000), blocks=4).map_batches(
            self._boom_after(1300), batch_format="pyarrow"
        )
        with pytest.raises(RayTaskError, match="deliberate failure"):
            write_lancedb(failing, "items", uri=db_dir, mode="append")

        # Re-running is safe precisely because nothing landed.
        write_lancedb(
            dataset_of(make_table(400, start=1000)), "items", uri=db_dir, mode="append"
        )
        assert lance.dataset(f"{db_dir}/items.lance").count_rows() == 500


class TestConcurrencyValidation:
    """A concurrency Ray cannot act on should be named, not translated.

    Passed through unchecked it surfaces from inside Ray as "`size` must be
    >= 1", which names an internal argument rather than the one that was set.
    """

    @pytest.mark.parametrize("value", [0, -1, -5])
    def test_write_rejects_non_positive_concurrency(
        self, db_dir: str, value: int
    ) -> None:
        with pytest.raises(ValueError, match="concurrency must be at least 1"):
            write_lancedb(
                dataset_of(make_table(5)),
                "items",
                uri=db_dir,
                mode="create",
                concurrency=value,
            )

    @pytest.mark.parametrize("value", [0, -2])
    def test_read_rejects_non_positive_concurrency(
        self, seeded_local: tuple[str, pa.Table], value: int
    ) -> None:
        db_dir, _ = seeded_local
        with pytest.raises(ValueError, match="concurrency must be at least 1"):
            read_lancedb("items", uri=db_dir, concurrency=value)

    def test_none_and_positive_values_are_accepted(
        self, seeded_local: tuple[str, pa.Table]
    ) -> None:
        db_dir, _ = seeded_local
        assert read_lancedb("items", uri=db_dir, concurrency=None).count() == 100
        assert read_lancedb("items", uri=db_dir, concurrency=2).count() == 100


class TestFragmentWriteOptions:
    """Options that describe the Lance files a fragment write produces.

    They have no analogue on the table API path, so a write that does not take
    the fragment path has to refuse them rather than drop them silently.
    """

    def test_stable_row_ids_reach_the_dataset(self, db_dir: str) -> None:
        write_lancedb(
            dataset_of(make_table(100), blocks=4),
            "items",
            uri=db_dir,
            mode="create",
            enable_stable_row_ids=True,
        )

        ds = lance.dataset(f"{db_dir}/items.lance")  # type: ignore[no-untyped-call]
        fragments = ds.get_fragments()
        # A fragment carries row-ID metadata only when stable IDs are on;
        # without them this is None and a row's address moves under compaction.
        assert fragments and all(f.metadata.row_id_meta is not None for f in fragments)

    def test_stable_row_ids_are_off_by_default(self, db_dir: str) -> None:
        """The default has to be observably off, or the flag proves nothing."""
        write_lancedb(
            dataset_of(make_table(100), blocks=4), "items", uri=db_dir, mode="create"
        )

        ds = lance.dataset(f"{db_dir}/items.lance")  # type: ignore[no-untyped-call]
        fragments = ds.get_fragments()
        assert fragments and all(f.metadata.row_id_meta is None for f in fragments)

    def test_data_storage_version_reaches_the_dataset(self, db_dir: str) -> None:
        write_lancedb(
            dataset_of(make_table(50)),
            "items",
            uri=db_dir,
            mode="create",
            data_storage_version="2.1",
        )

        assert read_lancedb("items", uri=db_dir).count() == 50
        # Assert the version reached the files. A row count would pass whether
        # or not the option was wired through, which is how a dropped
        # pass-through stays invisible.
        assert file_versions(db_dir) == [(2, 1)]

    def test_a_different_storage_version_is_distinguishable(self, db_dir: str) -> None:
        """The previous test only means something if the value can differ."""
        write_lancedb(
            dataset_of(make_table(50)),
            "items",
            uri=db_dir,
            mode="create",
            data_storage_version="2.0",
        )
        assert file_versions(db_dir) == [(2, 0)]

    def test_default_write_leaves_both_alone(self, db_dir: str) -> None:
        write_lancedb(dataset_of(make_table(20)), "items", uri=db_dir, mode="create")
        assert read_lancedb("items", uri=db_dir).count() == 20

    def test_an_empty_input_still_honours_stable_row_ids(self, db_dir: str) -> None:
        """The empty-input fallback must not quietly drop these.

        An empty input produces no fragments, so the table is materialised from
        the schema instead. Creating it through the database would lose
        ``enable_stable_row_ids`` -- and because stable IDs are fixed at
        creation, every later append to that table would silently lack them
        with no way back short of a rewrite.
        """
        empty = make_table(0)
        write_lancedb(
            dataset_of(empty),
            "items",
            uri=db_dir,
            mode="create",
            schema=empty.schema,
            enable_stable_row_ids=True,
        )
        write_lancedb(dataset_of(make_table(50)), "items", uri=db_dir, mode="append")

        ds = lance.dataset(f"{db_dir}/items.lance")  # type: ignore[no-untyped-call]
        fragments = ds.get_fragments()
        assert fragments and all(f.metadata.row_id_meta is not None for f in fragments)
        assert read_lancedb("items", uri=db_dir).count() == 50

    def test_an_empty_overwrite_keeps_stable_row_ids(self, db_dir: str) -> None:
        """An empty overwrite drops the rows without dropping the setting.

        Two behaviours meet here: emptying a table whose overwrite matched
        nothing, and keeping ``enable_stable_row_ids`` across it. Overwriting an
        existing dataset makes a new version, so the manifest's setting carries
        over on its own -- this pins that end to end rather than proving any
        particular implementation of the recovery.
        """
        write_lancedb(
            dataset_of(make_table(100)),
            "items",
            uri=db_dir,
            mode="create",
            enable_stable_row_ids=True,
        )
        empty = make_table(0)
        write_lancedb(
            dataset_of(empty),
            "items",
            uri=db_dir,
            mode="overwrite",
            schema=empty.schema,
            enable_stable_row_ids=True,
        )
        write_lancedb(dataset_of(make_table(30)), "items", uri=db_dir, mode="append")

        ds = lance.dataset(f"{db_dir}/items.lance")  # type: ignore[no-untyped-call]
        fragments = ds.get_fragments()
        assert ds.count_rows() == 30, "the empty overwrite should have dropped the rows"
        assert fragments and all(f.metadata.row_id_meta is not None for f in fragments)

    def test_an_empty_input_still_honours_the_storage_version(
        self, db_dir: str
    ) -> None:
        empty = make_table(0)
        write_lancedb(
            dataset_of(empty),
            "items",
            uri=db_dir,
            mode="create",
            schema=empty.schema,
            data_storage_version="2.0",
        )
        write_lancedb(dataset_of(make_table(10)), "items", uri=db_dir, mode="append")

        assert file_versions(db_dir) == [(2, 0)]

    def test_refused_against_cloud_enterprise(
        self, remote_uri: str, remote_kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="enable_stable_row_ids only applies"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=remote_uri,
                mode="create",
                enable_stable_row_ids=True,
                **remote_kwargs,
            )

    def test_refused_for_an_upsert(self, db_dir: str) -> None:
        write_lancedb(dataset_of(make_table(10)), "items", uri=db_dir, mode="create")
        with pytest.raises(ValueError, match="data_storage_version only applies"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=db_dir,
                mode="upsert",
                on="id",
                data_storage_version="stable",
            )

    def test_refused_when_the_api_path_is_forced(self, db_dir: str) -> None:
        with pytest.raises(ValueError, match="only applies to the local fragment"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=db_dir,
                mode="create",
                local_write_strategy="api",
                enable_stable_row_ids=True,
            )

    def test_both_are_named_when_both_are_unusable(self, db_dir: str) -> None:
        with pytest.raises(ValueError, match="data_storage_version, enable_stable"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=db_dir,
                mode="create",
                local_write_strategy="api",
                data_storage_version="stable",
                enable_stable_row_ids=True,
            )


class TestMaxBytesPerRequest:
    def test_a_byte_ceiling_splits_a_task_into_transactions(self, db_dir: str) -> None:
        """The byte ceiling has to reach the sink, the same way rows do."""
        arrow = make_table(400)
        write_lancedb(
            dataset_of(arrow),
            "items",
            uri=db_dir,
            mode="create",
            local_write_strategy="api",
            max_bytes_per_request=arrow.nbytes // 8,
        )

        assert read_lancedb("items", uri=db_dir).count() == 400
        # Creation plus more than one append, because the ceiling split the task.
        assert version_count(db_dir) > 2
