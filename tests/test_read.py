"""Read tests, run against both the local and Cloud/Enterprise backends."""

from __future__ import annotations

import pyarrow as pa
import pytest
from lancedb_ray import read_lancedb

from conftest import Backend, make_table, sorted_ids


class TestBasicReads:
    def test_reads_every_row(self, backend: Backend) -> None:
        backend.create("items", make_table(250))
        ds = read_lancedb("items", uri=backend.uri, **backend.kwargs)
        assert ds.count() == 250
        assert sorted_ids(ds) == list(range(250))

    def test_preserves_schema(self, backend: Backend) -> None:
        backend.create("items", make_table(10))
        ds = read_lancedb("items", uri=backend.uri, **backend.kwargs)
        assert set(ds.schema().names) == {"id", "vector", "label"}

    def test_reads_an_empty_table(self, backend: Backend) -> None:
        backend.create("items", make_table(0))
        ds = read_lancedb("items", uri=backend.uri, **backend.kwargs)
        assert ds.count() == 0

    def test_reads_a_single_row(self, backend: Backend) -> None:
        backend.create("items", make_table(1))
        assert read_lancedb("items", uri=backend.uri, **backend.kwargs).count() == 1

    def test_vector_column_round_trips(self, backend: Backend) -> None:
        backend.create("items", make_table(20))
        rows = read_lancedb("items", uri=backend.uri, **backend.kwargs).take_all()
        assert all(len(row["vector"]) == 8 for row in rows)

    def test_content_matches_exactly(self, backend: Backend) -> None:
        source = make_table(64)
        backend.create("items", source)
        rows = read_lancedb("items", uri=backend.uri, **backend.kwargs).take_all()
        by_id = {int(row["id"]): row["label"] for row in rows}
        expected = dict(
            zip(
                source.column("id").to_pylist(),
                source.column("label").to_pylist(),
                strict=False,
            )
        )
        assert by_id == expected


class TestProjectionAndFilter:
    def test_column_projection(self, backend: Backend) -> None:
        backend.create("items", make_table(50))
        ds = read_lancedb(
            "items", uri=backend.uri, columns=["id", "label"], **backend.kwargs
        )
        assert set(ds.schema().names) == {"id", "label"}
        assert ds.count() == 50

    def test_single_column_projection(self, backend: Backend) -> None:
        backend.create("items", make_table(30))
        ds = read_lancedb("items", uri=backend.uri, columns=["id"], **backend.kwargs)
        assert ds.schema().names == ["id"]

    def test_filter(self, backend: Backend) -> None:
        backend.create("items", make_table(200))
        ds = read_lancedb("items", uri=backend.uri, filter="id < 40", **backend.kwargs)
        assert ds.count() == 40
        assert sorted_ids(ds) == list(range(40))

    def test_filter_matching_nothing(self, backend: Backend) -> None:
        backend.create("items", make_table(50))
        ds = read_lancedb(
            "items", uri=backend.uri, filter="id > 100000", **backend.kwargs
        )
        assert ds.count() == 0

    def test_filter_matching_everything(self, backend: Backend) -> None:
        backend.create("items", make_table(75))
        ds = read_lancedb("items", uri=backend.uri, filter="id >= 0", **backend.kwargs)
        assert ds.count() == 75

    def test_filter_and_projection_together(self, backend: Backend) -> None:
        backend.create("items", make_table(120))
        ds = read_lancedb(
            "items",
            uri=backend.uri,
            columns=["id"],
            filter="id >= 100",
            **backend.kwargs,
        )
        assert ds.schema().names == ["id"]
        assert sorted_ids(ds) == list(range(100, 120))

    def test_non_contiguous_filter(self, backend: Backend) -> None:
        backend.create("items", make_table(100))
        ds = read_lancedb(
            "items", uri=backend.uri, filter="id % 10 = 0", **backend.kwargs
        )
        assert sorted_ids(ds) == list(range(0, 100, 10))


class TestVersionPinning:
    def test_reads_a_pinned_older_version(self, backend: Backend) -> None:
        backend.create("items", make_table(100))
        backend.open("items").add(make_table(50, start=100))

        assert backend.count("items") == 150
        pinned = read_lancedb("items", uri=backend.uri, version=1, **backend.kwargs)
        assert pinned.count() == 100

    def test_default_read_sees_the_latest_version(self, backend: Backend) -> None:
        backend.create("items", make_table(100))
        backend.open("items").add(make_table(50, start=100))
        ds = read_lancedb("items", uri=backend.uri, **backend.kwargs)
        assert ds.count() == 150

    def test_a_concurrent_write_does_not_tear_the_read(self, backend: Backend) -> None:
        """A write landing mid-read must not change what the read returns.

        The datasource pins a version on the driver before planning shards, so
        rows appended afterwards are invisible to every shard. Without pinning,
        shards planned against 100 rows could read a 200-row table and return
        duplicated or shifted data.
        """
        backend.create("items", make_table(100))

        ds = read_lancedb("items", uri=backend.uri, **backend.kwargs)
        # Land a write after planning but before the dataset is consumed.
        backend.open("items").add(make_table(100, start=100))

        assert ds.count() == 100
        assert sorted_ids(ds) == list(range(100))


class TestParallelism:
    def test_override_num_blocks(self, backend: Backend) -> None:
        backend.create("items", make_table(400))
        ds = read_lancedb(
            "items", uri=backend.uri, override_num_blocks=4, **backend.kwargs
        ).materialize()
        assert ds.count() == 400
        assert ds.num_blocks() <= 4

    def test_read_is_actually_sharded(self, backend: Backend) -> None:
        """More than one block means the read really did fan out."""
        backend.create("items", make_table(2000))
        ds = read_lancedb(
            "items", uri=backend.uri, override_num_blocks=4, **backend.kwargs
        ).materialize()
        assert ds.num_blocks() > 1
        assert sorted_ids(ds) == list(range(2000))

    def test_shards_do_not_duplicate_or_drop_rows(self, backend: Backend) -> None:
        backend.create("items", make_table(1111))
        ds = read_lancedb(
            "items", uri=backend.uri, override_num_blocks=7, **backend.kwargs
        )
        ids = [int(row["id"]) for row in ds.take_all()]
        assert len(ids) == len(set(ids)) == 1111


class TestErrors:
    def test_missing_table(self, backend: Backend) -> None:
        backend.create("items", make_table(5))
        # The two backends surface different exception types for a missing
        # table, so match on the name rather than pinning a class.
        with pytest.raises(Exception, match="nope"):
            read_lancedb("nope", uri=backend.uri, **backend.kwargs)

    def test_rejects_bad_batch_size(self, seeded_remote: tuple[str, pa.Table]) -> None:
        uri, _ = seeded_remote
        with pytest.raises(ValueError, match="batch_size must be positive"):
            read_lancedb("items", uri=uri, api_key="k", batch_size=0)

    def test_rejects_unknown_strategy(
        self, seeded_remote: tuple[str, pa.Table]
    ) -> None:
        uri, _ = seeded_remote
        with pytest.raises(ValueError, match="strategy must be one of"):
            read_lancedb("items", uri=uri, api_key="k", remote_read_strategy="bogus")  # type: ignore[arg-type]
