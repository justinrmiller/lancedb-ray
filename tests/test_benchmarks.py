# SPDX-License-Identifier: Apache-2.0
"""Tests for the benchmark suite's pure logic.

The suite is tooling, exercised by being run rather than by the unit tests, and
it stays out of the coverage gate for that reason. These two pieces are the
exception: both are pure functions whose failure mode is silence, and both were
caught the expensive way -- one by a six-hour run at the ``xl`` tier that
reported a projection ratio of 3.5e7, the other by a tier that selected zero
scenarios and exited before running anything.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path

import pytest
from benchmarks.counters import parse_plan_metrics
from benchmarks.harness import (
    TIERS,
    BenchRun,
    CaseTimeout,
    RunConfig,
    case_deadline,
    sweep_stale_run_roots,
)
from benchmarks.scenarios import ALL_TIERS, select


class TestParsePlanMetrics:
    """Lance renders plan metrics with a *count* suffix, never a byte unit.

    39,870 bytes prints as ``39.87 K``, so ``B`` means billion. Reading it as
    "bytes" silently divides the value by a billion, which is what turned a
    0.035 projection ratio into 3.5e7.
    """

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            # Observed verbatim in analyze_plan output across five magnitudes.
            ("rows_scanned=100", 100.0),
            ("bytes_read=39.87 K", 39_870.0),
            ("bytes_read=6.40 M", 6_400_000.0),
            ("bytes_read=125.3 M", 125_300_000.0),
            ("bytes_read=1.25 B", 1_250_000_000.0),
            ("bytes_read=12.59 B", 12_590_000_000.0),
            ("rows_scanned=4.00 M", 4_000_000.0),
            ("rows_scanned=400.0 K", 400_000.0),
        ],
    )
    def test_magnitude_suffixes(self, token: str, expected: float) -> None:
        key = token.split("=")[0]
        assert parse_plan_metrics(token)[key] == pytest.approx(expected)

    def test_billion_is_not_read_as_a_byte_unit(self) -> None:
        """The regression: ``B`` must multiply, not be swallowed as a unit."""
        assert parse_plan_metrics("bytes_read=12.59 B")["bytes_read"] > 1e9

    def test_a_byte_unit_is_still_tolerated(self) -> None:
        """Kept working in case a release switches to a byte formatter."""
        assert parse_plan_metrics("bytes_read=1.25 GB")["bytes_read"] == pytest.approx(
            1.25e9
        )

    def test_takes_the_largest_value_across_plan_nodes(self) -> None:
        plan = "LanceRead: bytes_read=1.25 B\n  Parent: bytes_read=6.40 M"
        assert parse_plan_metrics(plan)["bytes_read"] == pytest.approx(1.25e9)

    def test_ignores_metrics_that_are_not_counters(self) -> None:
        assert "elapsed" not in parse_plan_metrics("elapsed=60.037ms, iops=336")

    def test_trailing_comma_does_not_absorb_digits(self) -> None:
        assert parse_plan_metrics("iops=336, requests=336,") == {
            "iops": 336.0,
            "requests": 336.0,
        }

    def test_no_metrics_is_empty_not_an_error(self) -> None:
        assert parse_plan_metrics("") == {}


class TestTierRegistry:
    """A tier absent from ``ALL_TIERS`` selects nothing and the run exits.

    ``select`` filters on ``Scenario.tiers``, so a tier added to ``TIERS``
    without being registered here prints "no scenarios selected" -- a failure
    that looks like a CLI typo rather than a missing declaration.
    """

    def test_every_tier_is_registered(self) -> None:
        assert set(TIERS) <= set(ALL_TIERS), (
            f"tiers in TIERS but not ALL_TIERS: {sorted(set(TIERS) - set(ALL_TIERS))}"
        )

    def test_all_tiers_names_a_real_tier(self) -> None:
        assert set(ALL_TIERS) <= set(TIERS), (
            f"tiers in ALL_TIERS but not TIERS: {sorted(set(ALL_TIERS) - set(TIERS))}"
        )

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_each_tier_selects_scenarios(self, tier: str) -> None:
        assert select(None, tier, ("local", "fake")), f"tier {tier!r} selected nothing"

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_each_tier_sizes_every_dataset(self, tier: str) -> None:
        from benchmarks.datagen import DATASETS

        assert set(TIERS[tier].rows) == set(DATASETS)


class TestCaseDeadline:
    """A stuck call must cost one case, not the whole run.

    An ``xl`` run spent 91 minutes inside a single ``merge_insert`` before an
    external signal ended it, taking every scenario after it. pytest carries
    ``--timeout=300`` for exactly this; the benchmark had no equivalent.
    """

    def test_overrun_raises(self) -> None:
        with pytest.raises(CaseTimeout, match="budget"), case_deadline(0.05):
            time.sleep(5)

    def test_work_inside_the_budget_is_untouched(self) -> None:
        with case_deadline(30):
            result = sum(range(1000))
        assert result == 499_500

    def test_timer_is_cleared_on_the_way_out(self) -> None:
        """A leftover timer would fire during an unrelated later case."""
        with case_deadline(0.5):
            pass
        time.sleep(1.0)  # would raise here if the itimer were still armed

    def test_previous_handler_is_restored(self) -> None:
        before = signal.getsignal(signal.SIGALRM)
        with case_deadline(30):
            pass
        assert signal.getsignal(signal.SIGALRM) is before

    @pytest.mark.parametrize("disabled", [None, 0, 0.0, -1])
    def test_disabled_budget_is_a_noop(self, disabled: float | None) -> None:
        with case_deadline(disabled):
            pass

    def test_nested_exception_still_clears_the_timer(self) -> None:
        with contextlib.suppress(ValueError), case_deadline(30):
            raise ValueError("boom")
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0


class TestSweepStaleRunRoots:
    """Cleanup that only works in-process is not cleanup.

    No handler catches SIGKILL, and a process dying in native code can abort
    before Python regains control -- which left 39GB behind after one run. The
    sweep is what makes the next run self-healing.
    """

    def test_removes_an_aged_root(self, tmp_path: Path) -> None:
        stale = tmp_path / "ldbrbench_old"
        stale.mkdir()
        (stale / "data").write_text("x")
        os.utime(stale, (time.time() - 7 * 3600, time.time() - 7 * 3600))

        assert sweep_stale_run_roots(str(tmp_path), "ldbrbench_") == 1
        assert not stale.exists()

    def test_leaves_a_fresh_root_alone(self, tmp_path: Path) -> None:
        """A concurrent run's directory must survive."""
        fresh = tmp_path / "ldbrbench_running"
        fresh.mkdir()

        assert sweep_stale_run_roots(str(tmp_path), "ldbrbench_") == 0
        assert fresh.exists()

    def test_ignores_unrelated_directories(self, tmp_path: Path) -> None:
        other = tmp_path / "something_else"
        other.mkdir()
        os.utime(other, (time.time() - 7 * 3600, time.time() - 7 * 3600))

        assert sweep_stale_run_roots(str(tmp_path), "ldbrbench_") == 0
        assert other.exists()

    def test_missing_base_is_not_an_error(self, tmp_path: Path) -> None:
        assert sweep_stale_run_roots(str(tmp_path / "nope"), "ldbrbench_") == 0


class TestTierBudgets:
    def test_every_tier_sets_a_case_timeout(self) -> None:
        missing = [n for n, t in TIERS.items() if not t.case_timeout_s]
        assert not missing, f"tiers with no per-case budget: {missing}"

    def test_budgets_grow_with_tier_size(self) -> None:
        """A budget that does not scale with the data is a false failure."""
        ordered = ["smoke", "ci", "local", "full", "xl"]
        budgets = [TIERS[n].case_timeout_s for n in ordered if n in TIERS]
        assert budgets == sorted(budgets)


class TestOverrides:
    """``--blocks`` and ``--case-timeout`` exist to isolate one variable.

    Block count turned out to matter more than data volume for the upsert
    shuffle -- at a fixed 8M rows, 32 blocks took 155s, 64 took 530s and 256
    did not finish -- and there was no way to vary it without editing a tier.
    """

    def test_blocks_defaults_to_the_tier(self) -> None:
        run = BenchRun(RunConfig(tier="smoke"))
        try:
            assert run.blocks == TIERS["smoke"].blocks
        finally:
            run.cleanup()

    def test_blocks_override_wins(self) -> None:
        run = BenchRun(RunConfig(tier="smoke", blocks=97))
        try:
            assert run.blocks == 97
        finally:
            run.cleanup()

    def test_case_timeout_defaults_to_the_tier(self) -> None:
        run = BenchRun(RunConfig(tier="smoke"))
        try:
            assert run.case_timeout_s == TIERS["smoke"].case_timeout_s
        finally:
            run.cleanup()

    @pytest.mark.parametrize("override", [0, 60.0])
    def test_case_timeout_override_wins_including_zero(self, override: float) -> None:
        """Zero must disable the budget, not fall back to the tier's."""
        run = BenchRun(RunConfig(tier="smoke", case_timeout_s=override))
        try:
            assert run.case_timeout_s == override
        finally:
            run.cleanup()


class TestCaseTimeoutIsNotSwallowed:
    """The budget must survive the library's own error handling.

    ``lancedb_ray._retry.call_with_retry`` catches ``Exception`` and retries.
    An alarm landing inside it would have been read as a transient failure and
    the overrunning call retried, so the timeout has to pass through the way
    ``KeyboardInterrupt`` does.
    """

    def test_is_not_an_exception_subclass(self) -> None:
        assert issubclass(CaseTimeout, BaseException)
        assert not issubclass(CaseTimeout, Exception)

    def test_a_broad_handler_does_not_catch_it(self) -> None:
        caught = False
        with contextlib.suppress(CaseTimeout):
            try:
                with case_deadline(0.05):
                    time.sleep(5)
            except Exception:  # noqa: BLE001 - the point of the test
                caught = True
        assert not caught, "an `except Exception` swallowed the budget"

    def test_the_runner_names_it_so_a_timeout_is_recorded(self) -> None:
        """A BaseException the runner did not name would end the whole run."""
        import inspect

        from benchmarks import __main__ as entry

        src = inspect.getsource(entry.main)
        assert "CaseTimeout" in src, "the runner must catch CaseTimeout explicitly"
