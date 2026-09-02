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

import pytest
from benchmarks.counters import parse_plan_metrics
from benchmarks.harness import TIERS
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
