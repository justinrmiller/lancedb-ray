"""Upsert (merge-insert) tests across both backends."""

from __future__ import annotations

import pyarrow as pa
import pytest
import ray
from lancedb_ray import write_lancedb

from conftest import VECTOR_DIM, Backend, make_table


def rows_with(ids: list[int], label: str) -> pa.Table:
    """Build rows carrying a recognisable label for the given ids."""
    import numpy as np

    rng = np.random.default_rng(seed=0)
    vectors = rng.random((len(ids), VECTOR_DIM), dtype=np.float32)
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), pa.float32()), VECTOR_DIM
            ),
            "label": pa.array([label] * len(ids)),
        }
    )


def labels_by_id(backend: Backend, name: str = "items") -> dict[int, str]:
    table = backend.rows(name)
    return dict(
        zip(
            table.column("id").to_pylist(),
            table.column("label").to_pylist(),
            strict=False,
        )
    )


class TestUpsert:
    def test_updates_matched_rows_without_adding_any(self, backend: Backend) -> None:
        backend.create("items", make_table(100))

        write_lancedb(
            ray.data.from_arrow(rows_with([1, 2, 3], "UPDATED")),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            **backend.kwargs,
        )

        assert backend.count("items") == 100
        labels = labels_by_id(backend)
        assert labels[1] == labels[2] == labels[3] == "UPDATED"
        assert labels[4] == "row-4"

    def test_inserts_unmatched_rows(self, backend: Backend) -> None:
        backend.create("items", make_table(50))

        write_lancedb(
            ray.data.from_arrow(rows_with([100, 101], "NEW")),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            **backend.kwargs,
        )

        assert backend.count("items") == 52
        assert labels_by_id(backend)[100] == "NEW"

    def test_mixed_update_and_insert(self, backend: Backend) -> None:
        backend.create("items", make_table(20))

        write_lancedb(
            ray.data.from_arrow(rows_with([0, 1, 50, 51], "MIXED")),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            **backend.kwargs,
        )

        assert backend.count("items") == 22
        labels = labels_by_id(backend)
        assert labels[0] == labels[50] == "MIXED"

    def test_is_idempotent(self, backend: Backend) -> None:
        """Re-running the same upsert must not change the table."""
        backend.create("items", make_table(30))
        update = rows_with([1, 2, 3], "ONCE")

        for _ in range(3):
            write_lancedb(
                ray.data.from_arrow(update),
                "items",
                uri=backend.uri,
                mode="upsert",
                on="id",
                **backend.kwargs,
            )

        assert backend.count("items") == 30
        assert labels_by_id(backend)[1] == "ONCE"

    def test_upsert_into_a_missing_table_creates_it(self, backend: Backend) -> None:
        write_lancedb(
            ray.data.from_arrow(make_table(10)),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            **backend.kwargs,
        )
        assert backend.count("items") == 10

    def test_disabling_insert_only_updates(self, backend: Backend) -> None:
        backend.create("items", make_table(10))

        write_lancedb(
            ray.data.from_arrow(rows_with([1, 999], "UPDATE-ONLY")),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            when_not_matched_insert_all=False,
            **backend.kwargs,
        )

        assert backend.count("items") == 10
        assert labels_by_id(backend)[1] == "UPDATE-ONLY"

    def test_disabling_update_only_inserts(self, backend: Backend) -> None:
        backend.create("items", make_table(10))

        write_lancedb(
            ray.data.from_arrow(rows_with([1, 999], "INSERT-ONLY")),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            when_matched_update_all=False,
            **backend.kwargs,
        )

        labels = labels_by_id(backend)
        assert labels[1] == "row-1", "matched row should have been left alone"
        assert labels[999] == "INSERT-ONLY"

    def test_multi_column_key(self, backend: Backend) -> None:
        backend.create("items", make_table(20))
        write_lancedb(
            ray.data.from_arrow(rows_with([1], "MULTI")),
            "items",
            uri=backend.uri,
            mode="upsert",
            on=["id", "label"],
            **backend.kwargs,
        )
        # ("1", "MULTI") does not match ("1", "row-1"), so it is inserted.
        assert backend.count("items") == 21

    def test_upsert_across_multiple_write_tasks(self, backend: Backend) -> None:
        backend.create("items", make_table(200))
        update = rows_with(list(range(0, 100)), "BULK")

        write_lancedb(
            ray.data.from_arrow(update).repartition(4),
            "items",
            uri=backend.uri,
            mode="upsert",
            on="id",
            **backend.kwargs,
        )

        assert backend.count("items") == 200
        labels = labels_by_id(backend)
        assert all(labels[i] == "BULK" for i in range(100))
        assert labels[150] == "row-150"


class TestUpsertValidation:
    def test_upsert_requires_a_key(self, backend: Backend) -> None:
        with pytest.raises(ValueError, match="requires 'on'"):
            write_lancedb(
                ray.data.from_arrow(make_table(5)),
                "items",
                uri=backend.uri,
                mode="upsert",
                **backend.kwargs,
            )

    def test_key_is_rejected_for_non_upsert_modes(self, backend: Backend) -> None:
        with pytest.raises(ValueError, match="only meaningful for mode='upsert'"):
            write_lancedb(
                ray.data.from_arrow(make_table(5)),
                "items",
                uri=backend.uri,
                mode="append",
                on="id",
                **backend.kwargs,
            )
