"""Tests for connection specs, caching and backend detection."""

from __future__ import annotations

import os
from typing import Any

import pyarrow as pa
import pytest
from lancedb_ray.connection import (
    API_KEY_ENV_VAR,
    LanceDBConnectionSpec,
    clear_connection_cache,
    connect,
    list_table_names,
    open_table,
    table_exists,
    table_uri,
    to_lance,
)


class TestBackendDetection:
    @pytest.mark.parametrize(
        "uri", ["db://my-db", "db://fake/tmp/x", "db://prod-database"]
    )
    def test_db_scheme_is_remote(self, uri: str) -> None:
        assert LanceDBConnectionSpec.create(uri).is_remote

    @pytest.mark.parametrize(
        "uri", ["/data/lancedb", "s3://bucket/db", "gs://bucket/db", "./relative"]
    )
    def test_everything_else_is_local(self, uri: str) -> None:
        assert not LanceDBConnectionSpec.create(uri).is_remote

    def test_empty_uri_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LanceDBConnectionSpec.create("")


class TestSpecHashing:
    def test_specs_with_equal_mappings_are_equal(self) -> None:
        a = LanceDBConnectionSpec.create("/db", storage_options={"region": "us-west-2"})
        b = LanceDBConnectionSpec.create("/db", storage_options={"region": "us-west-2"})
        assert a == b
        assert hash(a) == hash(b)

    def test_mapping_order_does_not_affect_identity(self) -> None:
        a = LanceDBConnectionSpec.create("/db", storage_options={"x": "1", "y": "2"})
        b = LanceDBConnectionSpec.create("/db", storage_options={"y": "2", "x": "1"})
        assert a == b
        assert hash(a) == hash(b)

    def test_spec_is_hashable_so_it_can_key_the_connection_cache(self) -> None:
        spec = LanceDBConnectionSpec.create(
            "/db",
            storage_options={"a": "b"},
            client_config={"timeout": 30},
            namespace_client_properties={"root": "/x"},
        )
        assert {spec: "value"}[spec] == "value"

    def test_differing_uris_are_distinct(self) -> None:
        assert LanceDBConnectionSpec.create("/a") != LanceDBConnectionSpec.create("/b")

    def test_mappings_round_trip(self) -> None:
        spec = LanceDBConnectionSpec.create(
            "/db",
            storage_options={"region": "us-west-2"},
            client_config={"timeout": 30},
            namespace_client_properties={"root": "/tables"},
        )
        assert spec.storage_options == {"region": "us-west-2"}
        assert spec.client_config == {"timeout": 30}
        assert spec.namespace_client_properties == {"root": "/tables"}

    def test_absent_mappings_stay_none(self) -> None:
        spec = LanceDBConnectionSpec.create("/db")
        assert spec.storage_options is None
        assert spec.client_config is None
        assert spec.namespace_client_properties is None

    def test_spec_is_picklable(self) -> None:
        import pickle

        spec = LanceDBConnectionSpec.create("/db", storage_options={"a": "b"})
        assert pickle.loads(pickle.dumps(spec)) == spec


class TestApiKeyHandling:
    def test_explicit_key_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_KEY_ENV_VAR, "from-env")
        spec = LanceDBConnectionSpec.create("db://x", api_key="explicit")
        assert spec.resolve_api_key() == "explicit"

    def test_falls_back_to_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_KEY_ENV_VAR, "from-env")
        assert LanceDBConnectionSpec.create("db://x").resolve_api_key() == "from-env"

    def test_missing_key_for_remote_raises_a_useful_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        spec = LanceDBConnectionSpec.create("db://x")
        with pytest.raises(ValueError, match=API_KEY_ENV_VAR):
            spec.connect_kwargs()

    def test_local_connections_need_no_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        assert LanceDBConnectionSpec.create("/db").connect_kwargs() == {}

    def test_repr_redacts_the_key(self) -> None:
        spec = LanceDBConnectionSpec.create("db://x", api_key="super-secret")
        rendered = repr(spec)
        # The repr reaches logs and Ray task names, so the key must not be in it.
        assert "super-secret" not in rendered
        assert "***" in rendered

    def test_repr_of_keyless_spec_shows_none(self) -> None:
        assert "None" in repr(LanceDBConnectionSpec.create("/db"))


class TestConnectKwargs:
    def test_remote_kwargs_include_region_and_key(self) -> None:
        spec = LanceDBConnectionSpec.create(
            "db://x", api_key="k", region="eu-west-1", host_override="https://host"
        )
        kwargs = spec.connect_kwargs()
        assert kwargs["api_key"] == "k"
        assert kwargs["region"] == "eu-west-1"
        assert kwargs["host_override"] == "https://host"

    def test_host_override_is_omitted_when_unset(self) -> None:
        spec = LanceDBConnectionSpec.create("db://x", api_key="k")
        assert "host_override" not in spec.connect_kwargs()

    def test_local_kwargs_carry_storage_options(self) -> None:
        spec = LanceDBConnectionSpec.create(
            "s3://bucket/db", storage_options={"region": "us-west-2"}
        )
        assert spec.connect_kwargs() == {"storage_options": {"region": "us-west-2"}}

    def test_namespace_kwargs_are_passed_through(self) -> None:
        spec = LanceDBConnectionSpec.create(
            "/db",
            namespace_client_impl="dir",
            namespace_client_properties={"root": "/tables"},
        )
        kwargs = spec.connect_kwargs()
        assert kwargs["namespace_client_impl"] == "dir"
        assert kwargs["namespace_client_properties"] == {"root": "/tables"}


class TestConnectionCaching:
    def test_same_spec_reuses_one_connection(self, db_dir: str) -> None:
        spec = LanceDBConnectionSpec.create(db_dir)
        assert connect(spec) is connect(spec)

    def test_different_specs_get_different_connections(self, tmp_path: Any) -> None:
        a = LanceDBConnectionSpec.create(str(tmp_path / "a"))
        b = LanceDBConnectionSpec.create(str(tmp_path / "b"))
        os.makedirs(a.uri, exist_ok=True)
        os.makedirs(b.uri, exist_ok=True)
        assert connect(a) is not connect(b)

    def test_clearing_the_cache_forces_a_reconnect(self, db_dir: str) -> None:
        spec = LanceDBConnectionSpec.create(db_dir)
        first = connect(spec)
        clear_connection_cache()
        assert connect(spec) is not first


class TestTableHelpers:
    def test_listing_and_existence(self, seeded_local: tuple[str, pa.Table]) -> None:
        db_dir, _ = seeded_local
        spec = LanceDBConnectionSpec.create(db_dir)
        assert list_table_names(spec) == ["items"]
        assert table_exists(spec, "items")
        assert not table_exists(spec, "missing")

    def test_listing_works_on_the_remote_backend(
        self, seeded_remote: tuple[str, pa.Table], remote_kwargs: dict[str, Any]
    ) -> None:
        uri, _ = seeded_remote
        spec = LanceDBConnectionSpec.create(uri, **remote_kwargs)
        assert list_table_names(spec) == ["items"]

    def test_table_uri_of_existing_table(
        self, seeded_local: tuple[str, pa.Table]
    ) -> None:
        db_dir, _ = seeded_local
        spec = LanceDBConnectionSpec.create(db_dir)
        assert table_uri(spec, "items").endswith("items.lance")

    def test_table_uri_of_a_table_that_does_not_exist_yet(self, db_dir: str) -> None:
        # The write path needs a target location before the table exists.
        spec = LanceDBConnectionSpec.create(db_dir)
        assert table_uri(spec, "future") == f"{db_dir}/future.lance"

    def test_table_uri_is_refused_for_remote(self) -> None:
        spec = LanceDBConnectionSpec.create("db://x", api_key="k")
        with pytest.raises(TypeError, match="no client-accessible dataset"):
            table_uri(spec, "items")

    def test_open_table_pins_a_version(
        self, seeded_local: tuple[str, pa.Table]
    ) -> None:
        db_dir, table = seeded_local
        spec = LanceDBConnectionSpec.create(db_dir)
        connect(spec).open_table("items").add(table)

        pinned = open_table(spec, "items", version=1)
        assert pinned.count_rows() == 100
        assert open_table(spec, "items").count_rows() == 200


class TestToLance:
    def test_local_table_exposes_its_dataset(
        self, seeded_local: tuple[str, pa.Table]
    ) -> None:
        db_dir, _ = seeded_local
        spec = LanceDBConnectionSpec.create(db_dir)
        dataset = to_lance(open_table(spec, "items"))
        assert dataset.count_rows() == 100

    def test_remote_table_is_refused_with_an_explanatory_error(
        self, seeded_remote: tuple[str, pa.Table], remote_kwargs: dict[str, Any]
    ) -> None:
        uri, _ = seeded_remote
        spec = LanceDBConnectionSpec.create(uri, **remote_kwargs)
        # This is the capability gap the whole two-backend design exists for.
        with pytest.raises(TypeError, match="Cloud/Enterprise"):
            to_lance(open_table(spec, "items"))


class TestListingFallback:
    def test_falls_back_to_table_names_when_listing_is_unsupported(
        self, seeded_local: tuple[str, pa.Table]
    ) -> None:
        """Some local connections raise NotImplementedError from list_tables.

        The helper must fall back to the deprecated ``table_names`` rather than
        propagate that, which is why it exists at all.
        """
        db_dir, _ = seeded_local
        spec = LanceDBConnectionSpec.create(db_dir)
        db = connect(spec)

        class Unsupported:
            def list_tables(self, **kwargs: Any) -> Any:
                raise NotImplementedError("Namespace operations are not supported")

            def table_names(self) -> list[str]:
                return list(db.table_names())

        assert list_table_names(Unsupported()) == ["items"]

    def test_paginated_listing_is_followed(self) -> None:
        class Paged:
            def __init__(self) -> None:
                self.calls = 0

            def list_tables(self, page_token: Any = None, **kwargs: Any) -> Any:
                self.calls += 1
                if page_token is None:
                    return type("R", (), {"tables": ["a"], "page_token": "next"})()
                return type("R", (), {"tables": ["b"], "page_token": None})()

        paged = Paged()
        assert list_table_names(paged) == ["a", "b"]
        assert paged.calls == 2


def test_table_dataset_uri_returns_the_backing_dataset(
    seeded_local: tuple[str, pa.Table],
) -> None:
    from lancedb_ray.connection import table_dataset_uri

    db_dir, _ = seeded_local
    spec = LanceDBConnectionSpec.create(db_dir)
    assert table_dataset_uri(open_table(spec, "items")).endswith("items.lance")
