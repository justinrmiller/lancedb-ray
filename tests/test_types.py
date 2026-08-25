"""Schema and data-type coverage for the round trip.

Vector databases carry awkward column types -- fixed-size list embeddings,
nulls, nested structs, large binary blobs -- and these are where a naive
Arrow round trip tends to break.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
import ray
from lancedb_ray import read_lancedb, write_lancedb

from conftest import Backend


def round_trip(backend: Backend, table: pa.Table) -> pa.Table:
    """Write a table through Ray and read it back."""
    write_lancedb(
        ray.data.from_arrow(table),
        "typed",
        uri=backend.uri,
        mode="create",
        schema=table.schema,
        **backend.kwargs,
    )
    return backend.rows("typed")


class TestScalarTypes:
    def test_integers_and_floats(self, backend: Backend) -> None:
        source = pa.table(
            {
                "i8": pa.array([1, 2, 3], pa.int8()),
                "i64": pa.array([2**40, -(2**40), 0], pa.int64()),
                "f32": pa.array([1.5, 2.5, 3.5], pa.float32()),
                "f64": pa.array([1e300, -1e300, 0.0], pa.float64()),
            }
        )
        result = round_trip(backend, source).sort_by("i64")
        assert result.num_rows == 3
        assert set(result.column("i8").to_pylist()) == {1, 2, 3}

    def test_booleans_and_strings(self, backend: Backend) -> None:
        source = pa.table(
            {
                "id": pa.array([1, 2, 3], pa.int64()),
                "flag": pa.array([True, False, True]),
                "text": pa.array(["a", "", "unicode: ✓ 日本語"]),
            }
        )
        result = round_trip(backend, source).sort_by("id")
        assert result.column("flag").to_pylist() == [True, False, True]
        assert "unicode: ✓ 日本語" in result.column("text").to_pylist()

    def test_timestamps(self, backend: Backend) -> None:
        import datetime as dt

        source = pa.table(
            {
                "id": pa.array([1, 2], pa.int64()),
                "at": pa.array(
                    [dt.datetime(2026, 1, 1), dt.datetime(2026, 8, 25)],
                    pa.timestamp("us"),
                ),
            }
        )
        result = round_trip(backend, source).sort_by("id")
        assert result.column("at").to_pylist()[0].year == 2026


class TestNulls:
    def test_nulls_survive_the_round_trip(self, backend: Backend) -> None:
        source = pa.table(
            {
                "id": pa.array([1, 2, 3], pa.int64()),
                "maybe_int": pa.array([1, None, 3], pa.int64()),
                "maybe_text": pa.array(["x", None, None]),
            }
        )
        result = round_trip(backend, source).sort_by("id")
        assert result.column("maybe_int").to_pylist() == [1, None, 3]
        assert result.column("maybe_text").to_pylist() == ["x", None, None]

    def test_an_entirely_null_column(self, backend: Backend) -> None:
        source = pa.table(
            {
                "id": pa.array([1, 2], pa.int64()),
                "empty": pa.array([None, None], pa.int64()),
            }
        )
        result = round_trip(backend, source)
        assert result.column("empty").to_pylist() == [None, None]

    def test_nan_is_preserved_distinctly_from_null(self, backend: Backend) -> None:
        import math

        source = pa.table(
            {
                "id": pa.array([1, 2, 3], pa.int64()),
                "value": pa.array([float("nan"), None, 1.0], pa.float64()),
            }
        )
        result = round_trip(backend, source).sort_by("id")
        values = result.column("value").to_pylist()
        assert math.isnan(values[0])
        assert values[1] is None
        assert values[2] == 1.0


class TestVectorColumns:
    @pytest.mark.parametrize("dim", [1, 8, 128, 1536])
    def test_fixed_size_list_embeddings(self, backend: Backend, dim: int) -> None:
        import numpy as np

        rng = np.random.default_rng(seed=dim)
        rows = 4
        values = rng.random(rows * dim, dtype=np.float32)
        source = pa.table(
            {
                "id": pa.array(range(rows), pa.int64()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(values, pa.float32()), dim
                ),
            }
        )
        result = round_trip(backend, source).sort_by("id")
        assert all(len(v) == dim for v in result.column("vector").to_pylist())

    def test_float16_vectors(self, backend: Backend) -> None:
        import numpy as np

        dim = 4
        values = np.array([0.5, 1.5, 2.5, 3.5] * 2, dtype=np.float16)
        source = pa.table(
            {
                "id": pa.array([1, 2], pa.int64()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(values, pa.float16()), dim
                ),
            }
        )
        result = round_trip(backend, source)
        assert result.num_rows == 2

    def test_variable_length_lists(self, backend: Backend) -> None:
        source = pa.table(
            {
                "id": pa.array([1, 2, 3], pa.int64()),
                "tags": pa.array([["a", "b"], [], ["c"]], pa.list_(pa.string())),
            }
        )
        result = round_trip(backend, source).sort_by("id")
        assert result.column("tags").to_pylist() == [["a", "b"], [], ["c"]]


class TestNestedAndBinary:
    def test_struct_columns(self, backend: Backend) -> None:
        struct_type = pa.struct([("lat", pa.float64()), ("lon", pa.float64())])
        source = pa.table(
            {
                "id": pa.array([1, 2], pa.int64()),
                "point": pa.array(
                    [{"lat": 1.0, "lon": 2.0}, {"lat": 3.0, "lon": 4.0}], struct_type
                ),
            }
        )
        result = round_trip(backend, source).sort_by("id")
        assert result.column("point").to_pylist()[0]["lat"] == 1.0

    def test_binary_columns(self, backend: Backend) -> None:
        source = pa.table(
            {
                "id": pa.array([1, 2], pa.int64()),
                "payload": pa.array([b"\x00\x01\x02", b""], pa.binary()),
            }
        )
        result = round_trip(backend, source).sort_by("id")
        assert result.column("payload").to_pylist() == [b"\x00\x01\x02", b""]

    def test_large_binary_values(self, backend: Backend) -> None:
        blob = b"x" * (256 * 1024)
        source = pa.table(
            {
                "id": pa.array([1], pa.int64()),
                "payload": pa.array([blob], pa.large_binary()),
            }
        )
        result = round_trip(backend, source)
        assert len(result.column("payload").to_pylist()[0]) == len(blob)


class TestWideSchemas:
    def test_many_columns(self, backend: Backend) -> None:
        columns = {f"c{i}": pa.array([i, i + 1], pa.int64()) for i in range(100)}
        result = round_trip(backend, pa.table(columns))
        assert len(result.schema.names) == 100
        assert result.num_rows == 2

    def test_projection_of_a_wide_schema(self, backend: Backend) -> None:
        columns = {f"c{i}": pa.array([i, i + 1], pa.int64()) for i in range(50)}
        source = pa.table(columns)
        write_lancedb(
            ray.data.from_arrow(source),
            "wide",
            uri=backend.uri,
            mode="create",
            schema=source.schema,
            **backend.kwargs,
        )
        ds = read_lancedb(
            "wide", uri=backend.uri, columns=["c0", "c49"], **backend.kwargs
        )
        assert set(ds.schema().names) == {"c0", "c49"}
