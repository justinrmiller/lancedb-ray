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

    def test_forcing_the_fragment_path_with_a_transform_is_refused(
        self, db_dir: str
    ) -> None:
        with pytest.raises(ValueError, match="transform_fn"):
            write_lancedb(
                dataset_of(make_table(10)),
                "items",
                uri=db_dir,
                mode="create",
                transform_fn=lambda t: t,
                local_write_strategy="fragment",
            )

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
