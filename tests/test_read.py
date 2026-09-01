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

    def test_empty_column_list_is_rejected_on_both_backends(
        self, backend: Backend
    ) -> None:
        # The two backends used to disagree: remote silently read every column
        # and local produced a schema-less dataset. Neither is a projection.
        backend.create("items", make_table(10))
        with pytest.raises(ValueError, match=r"columns=\[\] selects no columns"):
            read_lancedb("items", uri=backend.uri, columns=[], **backend.kwargs)

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


class TestAfterDeletions:
    """Offsets are positional over the *current* version.

    Deleting rows shifts every later row's offset, so a shard plan built from a
    stale row count would read past the end or skip rows. These pin down that
    planning and reading agree on one post-deletion snapshot.
    """

    def test_offsets_read_is_correct_after_deletions(self, backend: Backend) -> None:
        backend.create("items", make_table(200))
        backend.open("items").delete("id % 2 = 0")

        ds = read_lancedb("items", uri=backend.uri, **backend.kwargs)
        assert ds.count() == 100
        assert sorted_ids(ds) == list(range(1, 200, 2))

    def test_sharded_read_is_correct_after_deletions(self, backend: Backend) -> None:
        backend.create("items", make_table(500))
        backend.open("items").delete("id < 200")

        ds = read_lancedb(
            "items", uri=backend.uri, override_num_blocks=6, **backend.kwargs
        )
        ids = [int(row["id"]) for row in ds.take_all()]
        assert len(ids) == len(set(ids)) == 300
        assert sorted(ids) == list(range(200, 500))

    def test_filtered_read_is_correct_after_deletions(self, backend: Backend) -> None:
        backend.create("items", make_table(300))
        backend.open("items").delete("id >= 100")

        ds = read_lancedb("items", uri=backend.uri, filter="id >= 50", **backend.kwargs)
        assert sorted_ids(ds) == list(range(50, 100))

    def test_reading_a_fully_emptied_table(self, backend: Backend) -> None:
        backend.create("items", make_table(50))
        backend.open("items").delete("id >= 0")

        ds = read_lancedb("items", uri=backend.uri, **backend.kwargs)
        assert ds.count() == 0


class TestScannerOptions:
    """Tuning that reaches the Lance scanner on a local read.

    Without this the only levers on a local scan are ``columns`` and
    ``filter``; everything the scanner itself understands is unreachable.
    """

    def test_scan_batch_size_does_not_change_the_result(self, db_dir: str) -> None:
        import lancedb

        lancedb.connect(db_dir).create_table("items", data=make_table(300))

        ds = read_lancedb("items", uri=db_dir, scanner_options={"batch_size": 64})

        assert ds.count() == 300
        assert sorted_ids(ds) == list(range(300))

    def test_with_row_id_adds_the_metadata_column(self, db_dir: str) -> None:
        """A scanner option that is visible in the output proves it arrived."""
        import lancedb

        lancedb.connect(db_dir).create_table("items", data=make_table(20))

        ds = read_lancedb("items", uri=db_dir, scanner_options={"with_row_id": True})

        assert "_rowid" in ds.schema().names

    def test_combines_with_projection_and_filter(self, db_dir: str) -> None:
        import lancedb

        lancedb.connect(db_dir).create_table("items", data=make_table(100))

        ds = read_lancedb(
            "items",
            uri=db_dir,
            columns=["id"],
            filter="id < 10",
            scanner_options={"batch_size": 8},
        )

        assert ds.count() == 10
        assert set(ds.schema().names) == {"id"}

    def test_rejected_for_a_remote_uri(
        self, seeded_remote: tuple[str, pa.Table], remote_kwargs: dict[str, object]
    ) -> None:
        uri, _ = seeded_remote
        with pytest.raises(ValueError, match="scanner_options applies to the Lance"):
            read_lancedb(
                "items",
                uri=uri,
                scanner_options={"batch_size": 64},
                **remote_kwargs,  # type: ignore[arg-type]
            )

    def test_an_empty_mapping_is_not_a_rejection(
        self, seeded_remote: tuple[str, pa.Table], remote_kwargs: dict[str, object]
    ) -> None:
        """Passing nothing must behave like passing nothing."""
        uri, _ = seeded_remote
        ds = read_lancedb(
            "items",
            uri=uri,
            scanner_options={},
            **remote_kwargs,  # type: ignore[arg-type]
        )
        assert ds.count() == 100
