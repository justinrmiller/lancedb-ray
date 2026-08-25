"""A fake LanceDB Cloud/Enterprise backend.

The remote LanceDB backend differs from the local one in ways that matter to
this library: it has no fragment access, ``to_arrow``/``to_lance`` raise, and
``optimize``/``compact_files`` quietly do nothing. Those differences are exactly
what the ``db://`` code paths are written around, so they need to be exercised.

Rather than mock the transport, this wraps a *real* local LanceDB database and
narrows it to the remote API surface. Every query genuinely executes, so the
sharding and batching logic is tested against real results, while the
capability restrictions match the documented behaviour of ``RemoteTable``.

Kept as a top-level module so Ray worker processes can import it via
``runtime_env.worker_process_setup_hook`` -- that is what lets ``db://`` tests
run through real distributed execution instead of only on the driver.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import lancedb

#: ``db://`` URIs with this prefix are served by the fake. The remainder of the
#: URI is the local directory that actually holds the data.
FAKE_REMOTE_PREFIX = "db://fake"

_REAL_CONNECT = lancedb.connect


def fake_uri(path: str) -> str:
    """Build a ``db://`` URI that the fake backend resolves to ``path``."""
    return f"{FAKE_REMOTE_PREFIX}{path}"


def _local_path(uri: str) -> str:
    return uri[len(FAKE_REMOTE_PREFIX) :]


class FakeRemoteTable:
    """A table restricted to the LanceDB Cloud/Enterprise API surface."""

    def __init__(self, inner: Any, name: str) -> None:
        self._inner = inner
        self._name = name

    # -- supported remotely ------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return int(self._inner.version)

    @property
    def schema(self) -> Any:
        return self._inner.schema

    def checkout(self, version: Union[int, str]) -> None:
        self._inner.checkout(version)

    def checkout_latest(self) -> None:
        self._inner.checkout_latest()

    def list_versions(self) -> Any:
        return self._inner.list_versions()

    def count_rows(self, filter: Optional[str] = None) -> int:
        return int(self._inner.count_rows(filter))

    def take_offsets(self, offsets: list[int]) -> Any:
        return self._inner.take_offsets(offsets)

    def take_row_ids(self, row_ids: list[int]) -> Any:
        return self._inner.take_row_ids(row_ids)

    def search(self, query: Any = None, **kwargs: Any) -> Any:
        return self._inner.search(query, **kwargs)

    def add(self, data: Any, **kwargs: Any) -> Any:
        return self._inner.add(data, **kwargs)

    def merge_insert(self, on: Any) -> Any:
        return self._inner.merge_insert(on)

    def delete(self, where: str) -> Any:
        return self._inner.delete(where)

    def create_index(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.create_index(*args, **kwargs)

    # -- unsupported remotely ---------------------------------------------

    def to_arrow(self) -> Any:
        raise NotImplementedError("to_arrow() is not yet supported on LanceDB cloud")

    def to_pandas(self) -> Any:
        raise NotImplementedError("to_pandas() is not yet supported on LanceDB cloud")

    def to_lance(self) -> Any:
        raise NotImplementedError("to_lance() is not supported on LanceDB cloud")

    @property
    def uri(self) -> str:
        raise NotImplementedError("uri is not supported on LanceDB cloud")

    # -- silent no-ops remotely -------------------------------------------

    def optimize(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("optimize() is a no-op on LanceDB Cloud", stacklevel=2)

    def compact_files(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("compact_files() is a no-op on LanceDB Cloud", stacklevel=2)

    def cleanup_old_versions(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "cleanup_old_versions() is a no-op on LanceDB Cloud", stacklevel=2
        )


class FakeRemoteDBConnection:
    """A connection restricted to the LanceDB Cloud/Enterprise API surface."""

    def __init__(self, uri: str, api_key: Optional[str], **kwargs: Any) -> None:
        if not api_key:
            raise ValueError("api_key is required to connect to LanceDB cloud")
        self._uri = uri
        self.api_key = api_key
        self.connect_kwargs = kwargs
        self._inner = _REAL_CONNECT(_local_path(uri))

    @property
    def uri(self) -> str:
        return self._uri

    def open_table(self, name: str, **kwargs: Any) -> FakeRemoteTable:
        return FakeRemoteTable(self._inner.open_table(name, **kwargs), name)

    def create_table(self, name: str, *args: Any, **kwargs: Any) -> FakeRemoteTable:
        return FakeRemoteTable(self._inner.create_table(name, *args, **kwargs), name)

    def drop_table(self, name: str, **kwargs: Any) -> Any:
        return self._inner.drop_table(name, **kwargs)

    def table_names(self) -> list[str]:
        return list(self._inner.table_names())

    def list_tables(self, **kwargs: Any) -> Any:
        return self._inner.list_tables(**kwargs)


class FlakyRemoteTable(FakeRemoteTable):
    """A fake table that fails a fixed number of times before succeeding.

    Used to prove the retry path actually retries, and that a permanently
    failing batch is surfaced (or skipped) according to ``on_batch_error``.
    """

    def __init__(
        self,
        inner: Any,
        name: str,
        *,
        failures: int,
        error: Optional[BaseException] = None,
        methods: tuple[str, ...] = ("add",),
    ) -> None:
        super().__init__(inner, name)
        self.remaining_failures = failures
        self.attempts = 0
        self._error = error or TimeoutError("connection timed out")
        self._methods = methods

    def _maybe_fail(self, method: str) -> None:
        if method not in self._methods:
            return
        self.attempts += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise self._error

    def add(self, data: Any, **kwargs: Any) -> Any:
        self._maybe_fail("add")
        return super().add(data, **kwargs)

    def take_offsets(self, offsets: list[int]) -> Any:
        self._maybe_fail("take_offsets")
        return super().take_offsets(offsets)

    def merge_insert(self, on: Any) -> Any:
        self._maybe_fail("merge_insert")
        return super().merge_insert(on)


def _fake_connect(uri: Any = None, **kwargs: Any) -> Any:
    """Stand-in for :func:`lancedb.connect` that serves fake ``db://`` URIs."""
    if isinstance(uri, str) and uri.startswith(FAKE_REMOTE_PREFIX):
        api_key = kwargs.pop("api_key", None)
        kwargs.pop("region", None)
        kwargs.pop("host_override", None)
        return FakeRemoteDBConnection(uri, api_key, **kwargs)
    return _REAL_CONNECT(uri, **kwargs)


def install_fake_remote() -> None:
    """Route fake ``db://`` URIs to the fake backend, process-wide.

    Called on the driver by the test fixtures and on every Ray worker by
    ``_ray_test_support.setup_worker``.
    """
    lancedb.connect = _fake_connect


def uninstall_fake_remote() -> None:
    """Restore the real :func:`lancedb.connect`."""
    lancedb.connect = _REAL_CONNECT
