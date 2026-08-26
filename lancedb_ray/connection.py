# SPDX-License-Identifier: Apache-2.0
"""Connection handling for LanceDB across local and Cloud/Enterprise backends.

A live ``lancedb.DBConnection`` holds sockets and thread pools and cannot be
pickled into a Ray worker. Instead we ship :class:`LanceDBConnectionSpec` -- a
small frozen, hashable description of *how* to connect -- and let each worker
build its own client. The client is cached per worker process rather than per
task, so a job with thousands of tasks still opens one connection per process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union, cast

if TYPE_CHECKING:
    import lance
    import lancedb

__all__ = [
    "LanceDBConnectionSpec",
    "REMOTE_URI_PREFIX",
    "connect",
    "list_table_names",
    "open_table",
    "table_dataset_uri",
    "table_exists",
    "table_uri",
]

#: URI scheme identifying a LanceDB Cloud / Enterprise database.
REMOTE_URI_PREFIX = "db://"

#: Environment variable consulted when no explicit API key is supplied.
API_KEY_ENV_VAR = "LANCEDB_API_KEY"

_Frozen = tuple[tuple[str, Any], ...]


#: Tags marking a frozen container so it can be rebuilt. The NUL prefix keeps
#: them from colliding with a real configuration value.
_DICT_TAG = "\x00lancedb_ray.dict"
_LIST_TAG = "\x00lancedb_ray.list"


def _freeze_value(value: Any) -> Any:
    """Recursively convert a value into something hashable.

    Freezing only the top level is not enough: LanceDB's ``client_config`` is
    documented as nested (``retry_config``, ``timeout_config``, ``tls_config``),
    so a realistic connection would leave a bare dict inside the spec and fail
    to hash with a message that names neither the option nor the cause.
    """
    if isinstance(value, Mapping):
        return (
            _DICT_TAG,
            tuple(sorted((str(k), _freeze_value(v)) for k, v in value.items())),
        )
    if isinstance(value, (list, tuple)):
        return (_LIST_TAG, tuple(_freeze_value(item) for item in value))
    return value


def _thaw_value(value: Any) -> Any:
    """Invert :func:`_freeze_value`."""
    if isinstance(value, tuple) and len(value) == 2:
        tag, payload = value
        if tag == _DICT_TAG:
            return {key: _thaw_value(item) for key, item in payload}
        if tag == _LIST_TAG:
            return [_thaw_value(item) for item in payload]
    return value


def _freeze(mapping: Optional[Mapping[str, Any]]) -> Optional[_Frozen]:
    """Convert a mapping into a hashable, order-independent tuple of pairs."""
    if mapping is None:
        return None
    return tuple(sorted((str(k), _freeze_value(v)) for k, v in mapping.items()))


def _thaw(frozen: Optional[_Frozen]) -> Optional[dict[str, Any]]:
    """Rebuild a mutable dict from :func:`_freeze` output."""
    if frozen is None:
        return None
    return {key: _thaw_value(value) for key, value in frozen}


@dataclass(frozen=True)
class LanceDBConnectionSpec:
    """A hashable, picklable description of a LanceDB connection.

    Mapping arguments are stored frozen so the spec can key an ``lru_cache``.
    Construct via :meth:`create`, which accepts ordinary dicts.

    Attributes:
        uri: Database URI. A ``db://`` prefix selects LanceDB Cloud/Enterprise;
            anything else (local path, ``s3://``, ``gs://``, ...) is local/OSS.
        api_key: Cloud/Enterprise API key. When omitted, workers fall back to
            the ``LANCEDB_API_KEY`` environment variable. Prefer the env var:
            it keeps the secret out of the object store and out of task specs.
        region: Cloud region.
        host_override: Alternate endpoint, as used by LanceDB Enterprise
            deployments that are not on the public Cloud endpoint.
    """

    uri: str
    api_key: Optional[str] = None
    region: str = "us-east-1"
    host_override: Optional[str] = None
    _storage_options: Optional[_Frozen] = field(default=None, repr=False)
    _client_config: Optional[_Frozen] = field(default=None, repr=False)
    _namespace_client_properties: Optional[_Frozen] = field(default=None, repr=False)
    namespace_client_impl: Optional[str] = None

    @classmethod
    def create(
        cls,
        uri: str,
        *,
        api_key: Optional[str] = None,
        region: str = "us-east-1",
        host_override: Optional[str] = None,
        storage_options: Optional[Mapping[str, Any]] = None,
        client_config: Optional[Mapping[str, Any]] = None,
        namespace_client_impl: Optional[str] = None,
        namespace_client_properties: Optional[Mapping[str, Any]] = None,
    ) -> LanceDBConnectionSpec:
        """Build a spec from ordinary keyword arguments."""
        if not uri:
            raise ValueError("uri must be a non-empty string")
        return cls(
            uri=uri,
            api_key=api_key,
            region=region,
            host_override=host_override,
            _storage_options=_freeze(storage_options),
            _client_config=_freeze(client_config),
            _namespace_client_properties=_freeze(namespace_client_properties),
            namespace_client_impl=namespace_client_impl,
        )

    @property
    def is_remote(self) -> bool:
        """Whether this spec points at LanceDB Cloud / Enterprise.

        Matched case-insensitively and ignoring surrounding whitespace: getting
        this wrong silently routes the write down the local fragment path,
        which then fails while trying to treat ``DB://name`` as a filesystem
        path rather than saying the URI was not understood.
        """
        return self.uri.strip().lower().startswith(REMOTE_URI_PREFIX)

    @property
    def storage_options(self) -> Optional[dict[str, Any]]:
        return _thaw(self._storage_options)

    @property
    def client_config(self) -> Optional[dict[str, Any]]:
        return _thaw(self._client_config)

    @property
    def namespace_client_properties(self) -> Optional[dict[str, Any]]:
        return _thaw(self._namespace_client_properties)

    def resolve_api_key(self) -> Optional[str]:
        """Return the explicit API key, else the one in the environment."""
        return self.api_key or os.environ.get(API_KEY_ENV_VAR)

    def connect_kwargs(self) -> dict[str, Any]:
        """Build the keyword arguments for :func:`lancedb.connect`."""
        kwargs: dict[str, Any] = {}
        if self.is_remote:
            api_key = self.resolve_api_key()
            if not api_key:
                raise ValueError(
                    f"Connecting to {self.uri!r} requires an API key. Pass "
                    f"api_key=... or set the {API_KEY_ENV_VAR} environment "
                    "variable (which must also be set on Ray workers)."
                )
            kwargs["api_key"] = api_key
            kwargs["region"] = self.region
            if self.host_override is not None:
                kwargs["host_override"] = self.host_override
        if self._storage_options is not None:
            kwargs["storage_options"] = self.storage_options
        if self._client_config is not None:
            kwargs["client_config"] = self.client_config
        if self.namespace_client_impl is not None:
            kwargs["namespace_client_impl"] = self.namespace_client_impl
        if self._namespace_client_properties is not None:
            kwargs["namespace_client_properties"] = self.namespace_client_properties
        return kwargs

    def __repr__(self) -> str:
        # Never let a secret reach a log line or a Ray task name.
        redacted = "***" if self.api_key else None
        return (
            f"LanceDBConnectionSpec(uri={self.uri!r}, api_key={redacted!r}, "
            f"region={self.region!r}, host_override={self.host_override!r})"
        )


@cache
def _connect_cached(spec: LanceDBConnectionSpec) -> lancedb.DBConnection:
    import lancedb

    return lancedb.connect(spec.uri, **spec.connect_kwargs())


def connect(spec: LanceDBConnectionSpec) -> lancedb.DBConnection:
    """Return a connection for ``spec``, reusing one per process.

    Ray workers call this on every task; the cache makes all but the first call
    in a given process free.
    """
    return _connect_cached(spec)


def clear_connection_cache() -> None:
    """Drop all cached connections and table handles. Primarily a test hook."""
    _connect_cached.cache_clear()
    _open_pinned_table.cache_clear()


@lru_cache(maxsize=32)
def _open_pinned_table(
    spec: LanceDBConnectionSpec, name: str, version: Union[int, str]
) -> lancedb.table.Table:
    table = connect(spec).open_table(name)
    table.checkout(version)
    return table


def open_table(
    spec: LanceDBConnectionSpec,
    name: str,
    *,
    version: Optional[Union[int, str]] = None,
) -> lancedb.table.Table:
    """Open ``name``, optionally pinned to ``version``.

    Pinning matters for distributed reads: without it, shards issued at
    different wall-clock times could observe different table versions and the
    resulting Ray Dataset would be torn.

    A pinned handle is cached per worker process, keyed on the version, because
    opening a table is a round trip against Cloud/Enterprise and a read job
    issues one per task. Caching is safe only because the version is fixed --
    the handle can never drift. An unpinned open is deliberately *not* cached:
    ``checkout`` mutates the handle in place, and a writer holding a stale
    "latest" would be operating on the wrong version.
    """
    if version is not None:
        return _open_pinned_table(spec, name, version)
    return connect(spec).open_table(name)


def table_dataset_uri(table: lancedb.table.Table) -> str:
    """Return the URI of the Lance dataset backing a local table.

    Raises:
        TypeError: If ``table`` is a remote (Cloud/Enterprise) table, which has
            no client-accessible dataset.
    """
    return str(to_lance(table).uri)


def to_lance(table: lancedb.table.Table) -> lance.LanceDataset:
    """Return the ``LanceDataset`` backing a local table.

    Remote tables raise: ``to_lance`` is only defined on ``LanceTable``, and the
    Cloud/Enterprise implementation deliberately does not expose the underlying
    storage.
    """
    getter = getattr(table, "to_lance", None)
    if getter is None:
        raise TypeError(
            f"{type(table).__name__} does not expose an underlying Lance "
            "dataset. Fragment-level access is only available for local/OSS "
            "LanceDB tables, not LanceDB Cloud/Enterprise."
        )
    try:
        return cast("lance.LanceDataset", getter())
    except NotImplementedError as error:
        raise TypeError(
            f"{type(table).__name__}.to_lance() is not supported on this "
            "backend. Fragment-level access is only available for local/OSS "
            "LanceDB tables, not LanceDB Cloud/Enterprise."
        ) from error


def list_table_names(spec_or_db: Union[LanceDBConnectionSpec, Any]) -> list[str]:
    """List table names, tolerating the two listing APIs LanceDB exposes.

    ``list_tables()`` is the current API but is paginated and unimplemented for
    some local connection types; ``table_names()`` is deprecated but universally
    available. Try the former, fall back to the latter.
    """
    db = (
        connect(spec_or_db)
        if isinstance(spec_or_db, LanceDBConnectionSpec)
        else spec_or_db
    )

    lister = getattr(db, "list_tables", None)
    if lister is not None:
        try:
            names: list[str] = []
            page_token = None
            while True:
                response = lister(page_token=page_token) if page_token else lister()
                page = list(getattr(response, "tables", response) or [])
                names.extend(page)
                page_token = getattr(response, "page_token", None)
                if not page_token or not page:
                    return names
        except (NotImplementedError, TypeError, AttributeError):
            pass

    return list(db.table_names())


def table_exists(spec_or_db: Union[LanceDBConnectionSpec, Any], name: str) -> bool:
    """Whether ``name`` exists, without relying on ``DBConnection.table_exists``.

    ``table_exists`` raises ``NotImplementedError`` on directory-backed local
    connections, so go through the listing instead.
    """
    return name in list_table_names(spec_or_db)


def table_uri(
    spec: LanceDBConnectionSpec, name: str, *, exists: Optional[bool] = None
) -> str:
    """Return the Lance dataset URI backing local table ``name``.

    Works whether or not the table exists yet, so the distributed write path can
    create a dataset at the location LanceDB will later resolve.

    Args:
        exists: Pass a known existence result to skip the catalog listing.
            Listing is paginated and costs a round trip per page against
            Cloud/Enterprise, so callers that already checked should say so.
    """
    if spec.is_remote:
        raise TypeError(
            "LanceDB Cloud/Enterprise tables have no client-accessible dataset URI."
        )
    if exists if exists is not None else table_exists(spec, name):
        # ``uri`` exists on LanceTable; the base Table type does not declare it.
        return str(connect(spec).open_table(name).uri)  # type: ignore[attr-defined]
    return f"{spec.uri.rstrip('/')}/{name}.lance"
