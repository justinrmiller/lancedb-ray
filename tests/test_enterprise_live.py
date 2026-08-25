"""Live tests against a real LanceDB Cloud / Enterprise endpoint.

Skipped unless both ``LANCEDB_URI`` (a ``db://`` URI) and ``LANCEDB_API_KEY``
are set. These deliberately mirror the assertions the fake-backed suite makes,
so running them validates that the fake in ``tests/_fakes.py`` still matches the
real service.

Run with::

    LANCEDB_URI=db://your-db LANCEDB_API_KEY=... pytest -m enterprise
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest
import ray
from lancedb_ray import read_lancedb, write_lancedb

from conftest import make_table

LIVE_URI = os.environ.get("LANCEDB_URI", "")
LIVE_KEY = os.environ.get("LANCEDB_API_KEY", "")
LIVE_REGION = os.environ.get("LANCEDB_REGION", "us-east-1")
LIVE_HOST_OVERRIDE = os.environ.get("LANCEDB_HOST_OVERRIDE")

pytestmark = [
    pytest.mark.enterprise,
    pytest.mark.skipif(
        not (LIVE_URI.startswith("db://") and LIVE_KEY),
        reason="set LANCEDB_URI (db://...) and LANCEDB_API_KEY to run live tests",
    ),
]


def live_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": LIVE_KEY, "region": LIVE_REGION}
    if LIVE_HOST_OVERRIDE:
        kwargs["host_override"] = LIVE_HOST_OVERRIDE
    return kwargs


@pytest.fixture
def live_table() -> Iterator[str]:
    """A uniquely named table on the live service, dropped afterwards."""
    import lancedb

    name = f"lancedb_ray_test_{uuid.uuid4().hex[:12]}"
    db = lancedb.connect(LIVE_URI, **live_kwargs())
    try:
        yield name
    finally:
        # Cleanup must not mask a test failure.
        with contextlib.suppress(Exception):
            db.drop_table(name)


class TestLiveRoundTrip:
    def test_write_then_read(self, live_table: str) -> None:
        source = make_table(500)
        write_lancedb(
            ray.data.from_arrow(source).repartition(4),
            live_table,
            uri=LIVE_URI,
            mode="create",
            **live_kwargs(),
        )

        ds = read_lancedb(live_table, uri=LIVE_URI, **live_kwargs())
        assert ds.count() == 500
        assert sorted(int(r["id"]) for r in ds.take_all()) == list(range(500))

    def test_projection_and_filter(self, live_table: str) -> None:
        write_lancedb(
            ray.data.from_arrow(make_table(200)),
            live_table,
            uri=LIVE_URI,
            mode="create",
            **live_kwargs(),
        )

        ds = read_lancedb(
            live_table,
            uri=LIVE_URI,
            columns=["id"],
            filter="id < 25",
            **live_kwargs(),
        )
        assert ds.schema().names == ["id"]
        assert ds.count() == 25

    @pytest.mark.parametrize("strategy", ["offsets", "pagination", "single"])
    def test_every_read_strategy_agrees(self, live_table: str, strategy: str) -> None:
        write_lancedb(
            ray.data.from_arrow(make_table(300)),
            live_table,
            uri=LIVE_URI,
            mode="create",
            **live_kwargs(),
        )

        ds = read_lancedb(
            live_table,
            uri=LIVE_URI,
            remote_read_strategy=strategy,  # type: ignore[arg-type]
            **live_kwargs(),
        )
        assert sorted(int(r["id"]) for r in ds.take_all()) == list(range(300))

    def test_upsert(self, live_table: str) -> None:
        write_lancedb(
            ray.data.from_arrow(make_table(100)),
            live_table,
            uri=LIVE_URI,
            mode="create",
            **live_kwargs(),
        )

        update = make_table(10).set_column(
            make_table(10).schema.get_field_index("label"),
            "label",
            pa.array(["UPDATED"] * 10),
        )
        write_lancedb(
            ray.data.from_arrow(update),
            live_table,
            uri=LIVE_URI,
            mode="upsert",
            on="id",
            **live_kwargs(),
        )

        ds = read_lancedb(live_table, uri=LIVE_URI, **live_kwargs())
        rows = {int(r["id"]): r["label"] for r in ds.take_all()}
        assert len(rows) == 100
        assert rows[0] == "UPDATED"
        assert rows[50] == "row-50"

    def test_version_pinning(self, live_table: str) -> None:
        write_lancedb(
            ray.data.from_arrow(make_table(100)),
            live_table,
            uri=LIVE_URI,
            mode="create",
            **live_kwargs(),
        )
        pinned_version = read_lancedb(live_table, uri=LIVE_URI, **live_kwargs()).count()

        write_lancedb(
            ray.data.from_arrow(make_table(50, start=100)),
            live_table,
            uri=LIVE_URI,
            mode="append",
            **live_kwargs(),
        )

        assert pinned_version == 100
        assert read_lancedb(live_table, uri=LIVE_URI, **live_kwargs()).count() == 150


class TestLiveCapabilities:
    def test_remote_tables_really_do_refuse_fragment_access(
        self, live_table: str
    ) -> None:
        """Confirms the assumption the whole two-backend design rests on."""
        import lancedb
        from lancedb_ray.connection import to_lance

        db = lancedb.connect(LIVE_URI, **live_kwargs())
        db.create_table(live_table, make_table(5))
        with pytest.raises(TypeError, match="Cloud/Enterprise"):
            to_lance(db.open_table(live_table))
