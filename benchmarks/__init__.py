# SPDX-License-Identifier: Apache-2.0
"""Benchmark suite for lancedb-ray.

Run with ``python -m benchmarks`` or ``make benchmark``. See ``PLAN.md`` for the
design and ``README.md`` for usage.

The suite measures performance *and* asserts correctness in the same pass: a
scenario that produces a fast but wrong result fails the run. Timings are noisy
on shared hardware, so the hard gates are deterministic counters -- fragments,
versions, rows scanned, request sizes -- and timings are only compared against a
baseline with a wide tolerance.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
