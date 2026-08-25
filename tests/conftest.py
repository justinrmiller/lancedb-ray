"""Shared pytest fixtures for lancedb-ray tests."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import ray

# Ensure the ``tests`` package directory is importable from Ray workers, which
# don't inherit pytest's sys.path.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import lancedb_ray  # noqa: E402
from lancedb_ray.connection import clear_connection_cache  # noqa: E402

from _fakes import fake_uri, install_fake_remote  # noqa: E402
from _ray_test_support import (  # noqa: E402
    patch_memory_profiler,
    patch_psutil_for_containers,
)

VECTOR_DIM = 8


@pytest.fixture(scope="session", autouse=True)
def ray_context() -> Iterator[None]:
    """Initialize Ray once per pytest session.

    Ported from lance-ray, whose test suite already worked through the
    container/PID-namespace failure modes patched below. Set
    ``RAY_TEST_ADDRESS`` to run against an existing cluster instead.
    """
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    patch_psutil_for_containers()

    if ray.is_initialized():
        ray.shutdown()

    patch_memory_profiler()
    install_fake_remote()

    runtime_env = {
        "worker_process_setup_hook": "_ray_test_support.setup_worker",
        "env_vars": {
            "PYTHONPATH": _TESTS_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        },
    }

    address = os.environ.get("RAY_TEST_ADDRESS")
    if address:
        ray.init(address=address, ignore_reinit_error=True, runtime_env=runtime_env)
        yield
    else:
        # Must live under a short root: Ray builds AF_UNIX socket paths
        # beneath it and macOS caps those at 103 bytes, which the default
        # /var/folders/... temp directory blows past on its own.
        temp_dir = tempfile.mkdtemp(prefix="ldbr_", dir="/tmp")
        ray.init(
            num_cpus=4,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            _temp_dir=temp_dir,
            runtime_env=runtime_env,
        )
        try:
            yield
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(autouse=True)
def _reset_connection_cache() -> Iterator[None]:
    """Keep cached connections from leaking between tests.

    The cache is keyed on the connection spec and specs repeat across tests
    that use different temp directories, so stale handles would otherwise
    point at deleted databases.
    """
    clear_connection_cache()
    yield
    clear_connection_cache()


@pytest.fixture
def db_dir(tmp_path: Any) -> str:
    """A local LanceDB database directory."""
    path = os.path.join(str(tmp_path), "lancedb")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture
def remote_uri(db_dir: str) -> str:
    """A ``db://`` URI served by the fake Cloud/Enterprise backend."""
    return fake_uri(db_dir)


@pytest.fixture
def remote_kwargs() -> dict[str, Any]:
    """Connection kwargs required for the fake remote backend."""
    return {"api_key": "fake-api-key"}


def make_table(num_rows: int = 100, *, start: int = 0, label: str = "row") -> pa.Table:
    """Build a deterministic Arrow table with id, vector and label columns."""
    rng = np.random.default_rng(seed=start + num_rows)
    vectors = rng.random((num_rows, VECTOR_DIM), dtype=np.float32)
    return pa.table(
        {
            "id": pa.array(range(start, start + num_rows), pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), pa.float32()), VECTOR_DIM
            ),
            "label": pa.array([f"{label}-{i}" for i in range(start, start + num_rows)]),
        }
    )


@pytest.fixture
def sample_table() -> pa.Table:
    """A 100-row Arrow table."""
    return make_table(100)


@pytest.fixture
def seeded_local(db_dir: str, sample_table: pa.Table) -> tuple[str, pa.Table]:
    """A local database with table ``items`` already populated."""
    import lancedb

    db = lancedb.connect(db_dir)
    db.create_table("items", sample_table)
    return db_dir, sample_table


@pytest.fixture
def seeded_remote(
    remote_uri: str, remote_kwargs: dict[str, Any], sample_table: pa.Table
) -> tuple[str, pa.Table]:
    """A fake remote database with table ``items`` already populated."""
    import lancedb

    db = lancedb.connect(remote_uri, **remote_kwargs)
    db.create_table("items", sample_table)
    return remote_uri, sample_table


def sorted_ids(dataset: Any) -> list[int]:
    """Collect and sort the ``id`` column of a Ray Dataset."""
    return sorted(int(row["id"]) for row in dataset.take_all())


__all__ = [
    "Backend",
    "VECTOR_DIM",
    "lancedb_ray",
    "make_table",
    "sorted_ids",
]


class Backend:
    """A database under test, plus the kwargs needed to reach it.

    Lets one test body run against both the local/OSS backend and the
    Cloud/Enterprise backend, which is the point: the public API is supposed to
    behave identically even though the strategies underneath differ completely.
    """

    def __init__(self, kind: str, uri: str, kwargs: dict[str, Any]) -> None:
        self.kind = kind
        self.uri = uri
        self.kwargs = kwargs

    @property
    def is_remote(self) -> bool:
        return self.kind == "remote"

    def connect(self) -> Any:
        import lancedb

        return lancedb.connect(self.uri, **self.kwargs)

    def create(self, name: str, data: pa.Table) -> Any:
        return self.connect().create_table(name, data)

    def open(self, name: str) -> Any:
        return self.connect().open_table(name)

    def count(self, name: str) -> int:
        return int(self.open(name).count_rows())

    def rows(self, name: str) -> pa.Table:
        """Read a whole table back, using an API both backends support."""
        return self.open(name).search(None).limit(None).to_arrow()

    def __repr__(self) -> str:
        return f"Backend({self.kind})"


@pytest.fixture(params=["local", "remote"])
def backend(request: pytest.FixtureRequest, db_dir: str) -> Backend:
    """Parametrized over the local and fake Cloud/Enterprise backends."""
    if request.param == "local":
        return Backend("local", db_dir, {})
    return Backend("remote", fake_uri(db_dir), {"api_key": "fake-api-key"})
