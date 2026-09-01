# SPDX-License-Identifier: Apache-2.0
"""Deterministic datasets for the benchmark suite.

Every column is a pure function of the row's ``id``. That is the property the
correctness checks are built on: to verify a row that came back from LanceDB we
recompute what it should have been from its id alone, so verification never
needs the source table in memory and stays exact at any scale.

Datasets are built as Ray Datasets generated on the workers rather than on the
driver, so a multi-GB tier does not have to fit in the driver process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import ray
from ray.data import Dataset

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "build_dataset",
    "build_table",
    "expected_rows",
    "get_spec",
]

#: Rows generated per worker batch. Bounds peak memory during generation: a
#: 1536-dimension float32 batch of this size is ~400MB, which is why it is not
#: larger.
_GEN_BATCH_ROWS = 65_536

_EPOCH_US = 1_700_000_000_000_000

# Multipliers for the id-derived pseudo-random values. Arbitrary odd constants;
# the point is only that consecutive ids produce unrelated-looking values while
# staying exactly reproducible from the id.
_MIX_ROW = np.uint64(2654435761)
_MIX_COL = np.uint64(40503)
_MOD = np.uint64(1000003)


def _mixed(ids: np.ndarray, col: int) -> np.ndarray:
    """A reproducible value in ``[0, 1)`` derived from ``(id, col)``."""
    u = ids.astype(np.uint64)
    return ((u * _MIX_ROW + np.uint64(col) * _MIX_COL) % _MOD).astype(
        np.float64
    ) / float(_MOD)


def _vectors(ids: np.ndarray, dim: int) -> np.ndarray:
    """A ``(len(ids), dim)`` float32 matrix derived from the ids."""
    u = ids.astype(np.uint64)[:, None]
    cols = np.arange(dim, dtype=np.uint64)[None, :]
    raw = (u * _MIX_ROW + cols * _MIX_COL) % _MOD
    return (raw.astype(np.float32) / np.float32(float(_MOD))).astype(np.float32)


def _labels(ids: np.ndarray) -> pa.Array:
    return pa.array([f"item-{int(i) % 1000:04d}" for i in ids], pa.string())


# -- schemas -----------------------------------------------------------------


def _narrow_batch(ids: np.ndarray) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "label": _labels(ids),
            "ts": pa.array(
                _EPOCH_US + ids.astype(np.int64) * 1_000_000,
                pa.timestamp("us"),
            ),
        }
    )


def _vector_batch(ids: np.ndarray, dim: int) -> pa.Table:
    mat = _vectors(ids, dim)
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(mat.reshape(-1), pa.float32()), dim
            ),
            "label": _labels(ids),
        }
    )


def _wide_scalar_batch(ids: np.ndarray, num_cols: int) -> pa.Table:
    data: dict[str, Any] = {"id": pa.array(ids, pa.int64())}
    for c in range(num_cols):
        data[f"c{c:02d}"] = pa.array(_mixed(ids, c + 1), pa.float64())
    return pa.table(data)


#: The fidelity dataset's schema. Written out rather than inferred so a silent
#: type change on the round trip is a diff against something explicit.
FIDELITY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("maybe_int", pa.int32(), nullable=True),
        pa.field("unicode", pa.string(), nullable=True),
        pa.field("tiny", pa.int8(), nullable=True),
        pa.field("big_float", pa.float64(), nullable=True),
        pa.field("when", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("vector", pa.list_(pa.float32(), 4), nullable=True),
        pa.field("blob", pa.large_binary(), nullable=True),
        pa.field(
            "nested",
            pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.string())]),
            nullable=True,
        ),
    ]
)

#: Strings chosen to break anything that assumes ASCII or that a string is a
#: sequence of code points: combining marks, an emoji with a ZWJ sequence, RTL
#: text, and an embedded NUL.
_UNICODE_SAMPLES = [
    "plain",
    "café́",
    "👩‍💻 family",
    "مرحبا بالعالم",
    "nul\x00inside",
    "🇯🇵🇺🇸",
    "\U0001d539\U0001d556\U0001d55d\U0001d55d\U0001d560",
    "",
]


def _fidelity_batch(ids: np.ndarray) -> pa.Table:
    n = len(ids)
    idx = ids.astype(np.int64)
    floats: list[Optional[float]] = []
    for i in idx:
        pick = int(i) % 6
        floats.append(
            [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.5e308][pick]
        )
    return pa.table(
        {
            "id": pa.array(idx, pa.int64()),
            # Every fourth row null, so a null-dropping bug shows up as a count.
            "maybe_int": pa.array(
                [None if int(i) % 4 == 0 else int(i) % 2147483647 for i in idx],
                pa.int32(),
            ),
            "unicode": pa.array(
                [_UNICODE_SAMPLES[int(i) % len(_UNICODE_SAMPLES)] for i in idx],
                pa.string(),
            ),
            "tiny": pa.array([(int(i) % 256) - 128 for i in idx], pa.int8()),
            "big_float": pa.array(floats, pa.float64()),
            "when": pa.array(
                [_EPOCH_US + int(i) * 3_600_000_000 for i in idx],
                pa.timestamp("us", tz="UTC"),
            ),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(_vectors(ids, 4).reshape(-1), pa.float32()), 4
            ),
            "blob": pa.array(
                [bytes([int(i) % 256]) * (int(i) % 17) for i in idx], pa.large_binary()
            ),
            "nested": pa.array(
                [{"a": int(i) * 3, "b": f"n{int(i) % 7}"} for i in idx],
                pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.string())]),
            ),
        },
        schema=FIDELITY_SCHEMA,
    ).slice(0, n)


@dataclass(frozen=True)
class DatasetSpec:
    """A named dataset shape, and how to build a batch of it."""

    name: str
    #: Builds the rows for the given ids. Must be a pure function of the ids.
    batch_fn: Callable[[np.ndarray], pa.Table]
    #: Approximate uncompressed bytes per row, for sizing and MB/s.
    bytes_per_row: int
    description: str

    def batch(self, ids: np.ndarray) -> pa.Table:
        return self.batch_fn(ids)

    @property
    def schema(self) -> pa.Schema:
        return self.batch(np.arange(1, dtype=np.int64)).schema


DATASETS: dict[str, DatasetSpec] = {
    "narrow": DatasetSpec(
        "narrow",
        _narrow_batch,
        bytes_per_row=32,
        description="id/label/ts -- per-row and per-task overhead",
    ),
    "vector": DatasetSpec(
        "vector",
        lambda ids: _vector_batch(ids, 128),
        bytes_per_row=8 + 128 * 4 + 13,
        description="128-dim embedding -- the realistic shape",
    ),
    "wide_vector": DatasetSpec(
        "wide_vector",
        lambda ids: _vector_batch(ids, 1536),
        bytes_per_row=8 + 1536 * 4 + 13,
        description="1536-dim embedding -- what the byte ceilings exist for",
    ),
    "wide_scalar": DatasetSpec(
        "wide_scalar",
        lambda ids: _wide_scalar_batch(ids, 40),
        bytes_per_row=8 + 40 * 8,
        description="41 scalar columns -- projection pushdown",
    ),
    "fidelity": DatasetSpec(
        "fidelity",
        _fidelity_batch,
        bytes_per_row=120,
        description="nulls, unicode, inf/nan, tz, struct, binary -- type fidelity",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(DATASETS)}") from None


def build_table(name: str, num_rows: int, *, start: int = 0) -> pa.Table:
    """Build the dataset directly as an Arrow table, on the driver.

    Only for small data -- the fidelity dataset and check fixtures. Anything
    large should go through :func:`build_dataset`.
    """
    spec = get_spec(name)
    if num_rows == 0:
        return spec.schema.empty_table()
    chunks = [
        spec.batch(
            np.arange(s, min(s + _GEN_BATCH_ROWS, start + num_rows), dtype=np.int64)
        )
        for s in range(start, start + num_rows, _GEN_BATCH_ROWS)
    ]
    return pa.concat_tables(chunks)


def expected_rows(name: str, ids: list[int]) -> pa.Table:
    """Recompute what the given ids *should* contain.

    This is what makes verification exact without holding the source: a row read
    back from LanceDB is compared against its id's recomputed value.
    """
    return get_spec(name).batch(np.asarray(ids, dtype=np.int64))


def build_dataset(
    name: str,
    num_rows: int,
    *,
    blocks: int,
    start: int = 0,
    materialize: bool = True,
) -> Dataset:
    """Build a Ray Dataset of ``num_rows`` rows, generated on the workers.

    Materialised by default so generation is not counted inside a timed write.
    """
    spec = get_spec(name)

    def to_rows(batch: pa.Table) -> pa.Table:
        # Shift the ids *before* deriving the other columns. Every column is a
        # function of its own row's id, so shifting afterwards would hand row
        # ``start + i`` the values belonging to row ``i``.
        ids = batch.column("id").to_numpy(zero_copy_only=False).astype(np.int64) + start
        return spec.batch(ids)

    ds = ray.data.range(num_rows, override_num_blocks=blocks).map_batches(
        to_rows, batch_format="pyarrow"
    )
    return ds.materialize() if materialize else ds
