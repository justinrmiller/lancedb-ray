# SPDX-License-Identifier: Apache-2.0
"""Ray worker setup for benchmark runs.

Referenced by name from ``runtime_env.worker_process_setup_hook``, so it has to
be importable inside a worker process from the repository root.

Reuses the test suite's fake Cloud/Enterprise backend rather than carrying a
second copy: the fake is the one that ``tests/test_enterprise_live.py``
validates against a real service, and a benchmark measuring a *different* fake
would be measuring fiction.
"""

from __future__ import annotations

import os
import sys

__all__ = ["setup_worker"]


def _ensure_tests_importable() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(root, "tests")
    for path in (root, tests_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def setup_worker() -> None:
    _ensure_tests_importable()

    try:
        from _ray_test_support import (
            patch_memory_profiler,
            patch_psutil_for_containers,
        )

        patch_memory_profiler()
        patch_psutil_for_containers()
    except ImportError:  # pragma: no cover - only if tests/ is absent
        pass

    if os.environ.get("LANCEDB_RAY_BENCH_FAKE_REMOTE") == "1":
        from _fakes import install_fake_remote

        install_fake_remote()

    from benchmarks.probe import install_probe

    install_probe()
