"""Tests for remote read-task planning and strategy selection."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from lancedb_ray.connection import LanceDBConnectionSpec
from lancedb_ray.datasource import LanceDBDatasource

from conftest import make_table


@pytest.fixture
def spec(
    seeded_remote: tuple[str, pa.Table], remote_kwargs: dict[str, Any]
) -> LanceDBConnectionSpec:
    uri, _ = seeded_remote
    return LanceDBConnectionSpec.create(uri, **remote_kwargs)


class TestStrategySelection:
    def test_auto_uses_offsets_without_a_filter(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        assert LanceDBDatasource(spec, "items").strategy == "offsets"

    def test_auto_uses_pagination_with_a_filter(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        # take_offsets is positional and cannot carry a predicate, so a
        # filtered read has to page server-side instead.
        source = LanceDBDatasource(spec, "items", filter="id > 5")
        assert source.strategy == "pagination"

    @pytest.mark.parametrize("strategy", ["offsets", "pagination", "single"])
    def test_explicit_strategy_is_respected(
        self, spec: LanceDBConnectionSpec, strategy: str
    ) -> None:
        source = LanceDBDatasource(spec, "items", strategy=strategy)  # type: ignore[arg-type]
        assert source.strategy == strategy

    def test_unknown_strategy_is_rejected(self, spec: LanceDBConnectionSpec) -> None:
        with pytest.raises(ValueError, match="strategy must be one of"):
            LanceDBDatasource(spec, "items", strategy="teleport")  # type: ignore[arg-type]

    def test_non_positive_batch_size_is_rejected(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            LanceDBDatasource(spec, "items", batch_size=0)


class TestVersionPinning:
    def test_pins_the_current_version_by_default(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        source = LanceDBDatasource(spec, "items")
        assert source.version == 1

    def test_pinned_version_is_captured_before_later_writes(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        source = LanceDBDatasource(spec, "items")

        import lancedb

        lancedb.connect(spec.uri, **spec.connect_kwargs()).open_table("items").add(
            make_table(10, start=1000)
        )

        # The datasource captured version 1 at construction time; a later write
        # cannot retroactively change what it will read.
        assert source.version == 1
        assert source.num_rows == 100

    def test_explicit_version_is_used(self, spec: LanceDBConnectionSpec) -> None:
        assert LanceDBDatasource(spec, "items", version=1).version == 1


class TestReadTaskPlanning:
    def test_one_task_per_requested_shard(self, spec: LanceDBConnectionSpec) -> None:
        tasks = LanceDBDatasource(spec, "items").get_read_tasks(4)
        assert len(tasks) == 4

    def test_task_row_counts_sum_to_the_table(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        tasks = LanceDBDatasource(spec, "items").get_read_tasks(7)
        assert sum(t.metadata.num_rows or 0 for t in tasks) == 100

    def test_empty_table_produces_no_tasks(
        self, remote_uri: str, remote_kwargs: dict[str, Any]
    ) -> None:
        import lancedb

        lancedb.connect(remote_uri, **remote_kwargs).create_table(
            "empty", make_table(0)
        )
        spec = LanceDBConnectionSpec.create(remote_uri, **remote_kwargs)
        assert LanceDBDatasource(spec, "empty").get_read_tasks(4) == []

    def test_single_strategy_produces_one_task(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        tasks = LanceDBDatasource(spec, "items", strategy="single").get_read_tasks(8)
        assert len(tasks) == 1

    def test_parallelism_above_row_count_is_clamped(
        self, remote_uri: str, remote_kwargs: dict[str, Any]
    ) -> None:
        import lancedb

        lancedb.connect(remote_uri, **remote_kwargs).create_table("tiny", make_table(3))
        spec = LanceDBConnectionSpec.create(remote_uri, **remote_kwargs)
        assert len(LanceDBDatasource(spec, "tiny").get_read_tasks(64)) == 3

    def test_per_task_row_limit_caps_each_shard(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        tasks = LanceDBDatasource(spec, "items").get_read_tasks(4, per_task_row_limit=5)
        assert all((t.metadata.num_rows or 0) <= 5 for t in tasks)

    def test_extra_arguments_are_tolerated(self, spec: LanceDBConnectionSpec) -> None:
        # Newer Ray releases add parameters here; absorbing them keeps the
        # integration from breaking on upgrade.
        tasks = LanceDBDatasource(spec, "items").get_read_tasks(
            2, None, "positional", future_option=True
        )
        assert len(tasks) == 2

    def test_tasks_are_executable_and_cover_the_table(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        tasks = LanceDBDatasource(spec, "items").get_read_tasks(4)
        ids: list[int] = []
        for task in tasks:
            for block in task():
                ids.extend(block.column("id").to_pylist())
        assert sorted(ids) == list(range(100))

    def test_pagination_tasks_cover_the_filtered_rows(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        source = LanceDBDatasource(spec, "items", filter="id >= 60")
        ids: list[int] = []
        for task in source.get_read_tasks(3):
            for block in task():
                ids.extend(block.column("id").to_pylist())
        assert sorted(ids) == list(range(60, 100))

    def test_projection_is_applied_in_tasks(self, spec: LanceDBConnectionSpec) -> None:
        source = LanceDBDatasource(spec, "items", columns=["id"])
        blocks = [block for task in source.get_read_tasks(2) for block in task()]
        assert all(block.column_names == ["id"] for block in blocks)


class TestSizeEstimation:
    def test_estimate_is_positive_for_a_populated_table(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        estimate = LanceDBDatasource(spec, "items").estimate_inmemory_data_size()
        assert estimate is not None and estimate > 0

    def test_estimate_scales_with_row_count(
        self, remote_uri: str, remote_kwargs: dict[str, Any]
    ) -> None:
        import lancedb

        db = lancedb.connect(remote_uri, **remote_kwargs)
        db.create_table("small", make_table(50))
        db.create_table("large", make_table(500))
        spec = LanceDBConnectionSpec.create(remote_uri, **remote_kwargs)

        small = LanceDBDatasource(spec, "small").estimate_inmemory_data_size() or 0
        large = LanceDBDatasource(spec, "large").estimate_inmemory_data_size() or 0
        assert large > small

    def test_empty_table_estimates_zero(
        self, remote_uri: str, remote_kwargs: dict[str, Any]
    ) -> None:
        import lancedb

        lancedb.connect(remote_uri, **remote_kwargs).create_table(
            "empty", make_table(0)
        )
        spec = LanceDBConnectionSpec.create(remote_uri, **remote_kwargs)
        assert LanceDBDatasource(spec, "empty").estimate_inmemory_data_size() == 0

    def test_estimation_failure_is_not_fatal(
        self, spec: LanceDBConnectionSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = LanceDBDatasource(spec, "items")

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("sampling exploded")

        monkeypatch.setattr("lancedb_ray.datasource.open_table", boom)
        # Estimation is only a scheduling hint; failing it must not fail a read.
        assert source.estimate_inmemory_data_size() is None


def test_datasource_name_includes_the_table(spec: LanceDBConnectionSpec) -> None:
    assert LanceDBDatasource(spec, "items").get_name() == "LanceDB(items)"


class TestSingleStrategyExecution:
    def test_single_task_streams_the_whole_table(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        source = LanceDBDatasource(spec, "items", strategy="single", batch_size=32)
        (task,) = source.get_read_tasks(8)

        ids: list[int] = []
        for block in task():
            ids.extend(block.column("id").to_pylist())
        assert sorted(ids) == list(range(100))

    def test_single_task_applies_filter_and_projection(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        source = LanceDBDatasource(
            spec, "items", strategy="single", filter="id < 10", columns=["id"]
        )
        (task,) = source.get_read_tasks(4)
        blocks = list(task())
        ids = [i for block in blocks for i in block.column("id").to_pylist()]
        assert sorted(ids) == list(range(10))
        assert all(block.column_names == ["id"] for block in blocks)

    def test_single_task_on_an_empty_result_yields_an_empty_block(
        self, seeded_remote: tuple[str, pa.Table], remote_kwargs: dict[str, Any]
    ) -> None:
        uri, _ = seeded_remote
        spec = LanceDBConnectionSpec.create(uri, **remote_kwargs)
        source = LanceDBDatasource(
            spec, "items", strategy="single", filter="id > 100000"
        )
        # num_rows is zero so no tasks are planned at all.
        assert source.get_read_tasks(4) == []

    def test_single_task_respects_per_task_row_limit(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        source = LanceDBDatasource(spec, "items", strategy="single")
        (task,) = source.get_read_tasks(1, per_task_row_limit=10)
        assert (task.metadata.num_rows or 0) == 10


class TestPaginationEdgeCases:
    def test_pagination_stops_early_on_a_short_page(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        """A page shorter than the limit means the result set ended.

        Continuing to request further pages would only produce empty responses,
        so the shard stops rather than burning round trips.
        """
        source = LanceDBDatasource(
            spec, "items", strategy="pagination", filter="id < 7", batch_size=3
        )
        ids: list[int] = []
        for task in source.get_read_tasks(1):
            for block in task():
                ids.extend(block.column("id").to_pylist())
        assert sorted(ids) == list(range(7))

    def test_pagination_without_a_filter_still_works(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        source = LanceDBDatasource(spec, "items", strategy="pagination", batch_size=10)
        ids: list[int] = []
        for task in source.get_read_tasks(4):
            for block in task():
                ids.extend(block.column("id").to_pylist())
        assert sorted(ids) == list(range(100))

    def test_offsets_strategy_chunks_requests(
        self, spec: LanceDBConnectionSpec
    ) -> None:
        # A batch_size below the shard size forces multiple take_offsets calls
        # per shard, which must still concatenate into the full shard.
        source = LanceDBDatasource(spec, "items", strategy="offsets", batch_size=7)
        ids: list[int] = []
        for task in source.get_read_tasks(2):
            for block in task():
                ids.extend(block.column("id").to_pylist())
        assert sorted(ids) == list(range(100))
