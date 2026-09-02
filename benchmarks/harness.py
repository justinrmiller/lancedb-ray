# SPDX-License-Identifier: Apache-2.0
"""Run lifecycle for the benchmark suite.

Owns the things every scenario needs and none of them should re-invent: tier
sizing, the timed repeat loop, resource sampling, environment capture, and the
cleanup contract -- a run must leave nothing behind even when it is killed.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Optional, TypeVar

from .checks import CheckList
from .probe import PROBE_DIR_ENV, clear_events, read_events, summarize

__all__ = [
    "BenchRun",
    "Case",
    "RunConfig",
    "TIERS",
    "Tier",
    "capture_environment",
]

T = TypeVar("T")

#: Directories created by a run, removed on exit however the process ends.
_TO_CLEAN: set[str] = set()
_cleanup_installed = False


def _clean_all() -> None:
    for path in list(_TO_CLEAN):
        shutil.rmtree(path, ignore_errors=True)
        _TO_CLEAN.discard(path)


def _install_cleanup() -> None:
    """Make cleanup survive Ctrl-C and a CI cancellation, not just a clean exit."""
    global _cleanup_installed
    if _cleanup_installed:
        return
    atexit.register(_clean_all)

    def handler(signum: int, frame: Optional[FrameType]) -> None:
        _clean_all()
        # Restore and re-raise so the exit status still reflects the signal.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, handler)
    _cleanup_installed = True


def register_for_cleanup(path: str) -> str:
    _install_cleanup()
    _TO_CLEAN.add(path)
    return path


# -- tiers -------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    """How big a run is. Only sizes change -- every check runs in every tier."""

    name: str
    rows: dict[str, int]
    blocks: int
    repeat: int
    warmup: int
    #: Whether knob sweeps run their full range or a reduced one.
    full_sweeps: bool
    #: Refuse to start if the run root has less free space than this.
    min_free_bytes: int
    description: str

    def rows_for(self, dataset: str) -> int:
        return self.rows[dataset]


_GB = 1024**3

TIERS: dict[str, Tier] = {
    "smoke": Tier(
        name="smoke",
        rows={
            "narrow": 20_000,
            "vector": 10_000,
            "wide_vector": 1_000,
            "wide_scalar": 10_000,
            "fidelity": 2_000,
        },
        blocks=4,
        repeat=1,
        warmup=0,
        full_sweeps=False,
        min_free_bytes=1 * _GB,
        description="seconds; proves the suite itself works",
    ),
    "ci": Tier(
        name="ci",
        rows={
            "narrow": 250_000,
            "vector": 150_000,
            "wide_vector": 12_000,
            "wide_scalar": 150_000,
            "fidelity": 4_000,
        },
        blocks=8,
        repeat=2,
        warmup=1,
        full_sweeps=False,
        min_free_bytes=3 * _GB,
        description="sized for a 4-vCPU / 16GB hosted runner",
    ),
    "local": Tier(
        name="local",
        rows={
            "narrow": 2_000_000,
            "vector": 1_000_000,
            "wide_vector": 60_000,
            "wide_scalar": 800_000,
            "fidelity": 20_000,
        },
        blocks=16,
        repeat=3,
        warmup=1,
        full_sweeps=True,
        min_free_bytes=8 * _GB,
        description="the default for a development machine",
    ),
    "full": Tier(
        name="full",
        rows={
            "narrow": 20_000_000,
            "vector": 8_000_000,
            "wide_vector": 400_000,
            "wide_scalar": 5_000_000,
            "fidelity": 50_000,
        },
        blocks=32,
        repeat=3,
        warmup=1,
        full_sweeps=True,
        min_free_bytes=60 * _GB,
        description="opt-in; multi-GB and minutes per scenario",
    ),
    # Narrow scales 20x but the vector datasets stop at 8x, which is not a
    # preference: macOS caps Ray's object store at 2GB, so read_full's
    # materialize() spills roughly a second copy of the table to disk, and
    # upsert_merge holds one and a half. At 20x vector that peaks past a 128GB
    # volume; at 8x it peaks near 68GB.
    "xl": Tier(
        name="xl",
        rows={
            "narrow": 400_000_000,
            "vector": 64_000_000,
            "wide_vector": 3_200_000,
            "wide_scalar": 40_000_000,
            "fidelity": 1_000_000,
        },
        # Up from full's 32. Blocks are what Ray schedules, and at 32 a vector
        # block would be ~1GB -- eight of them in flight is four times the whole
        # object store. At 256 every dataset lands in a 50-133MB block.
        blocks=256,
        # This tier exists for scale, not for timing stability. Every check and
        # every gated counter still runs; one timed iteration rather than four
        # is what makes the run finish in hours instead of days.
        repeat=1,
        warmup=0,
        full_sweeps=True,
        min_free_bytes=80 * _GB,
        description="opt-in; 400M narrow rows and hours per run",
    ),
}


@dataclass
class RunConfig:
    """Everything the CLI can set."""

    tier: str = "local"
    scenarios: Optional[list[str]] = None
    backends: tuple[str, ...] = ("local", "fake")
    repeat: Optional[int] = None
    warmup: Optional[int] = None
    num_cpus: Optional[int] = None
    object_store_bytes: Optional[int] = None
    run_root: Optional[str] = None
    out_dir: Optional[str] = None
    baseline: Optional[str] = None
    regression_factor: float = 2.5
    fail_fast: bool = False
    keep_artifacts: bool = False
    enterprise_uri: Optional[str] = None
    s3_uri: Optional[str] = None
    junit: Optional[str] = None
    quiet: bool = False


# -- environment -------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def capture_environment() -> dict[str, Any]:
    """Everything needed to decide whether two runs are comparable."""
    import psutil

    versions: dict[str, str] = {}
    for name in ("ray", "lancedb", "lance", "pyarrow", "numpy", "lance_ray"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "?"))
        except Exception:
            versions[name] = "missing"

    return {
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "total_ram_bytes": int(psutil.virtual_memory().total),
        "ci": bool(os.environ.get("CI")),
        "versions": versions,
    }


class _ResourceSampler:
    """Samples RSS across the driver and the Ray worker processes.

    Ray workers are children of the raylet rather than of this process, so they
    are found by command line. Sampling is best-effort: a missed sample makes the
    peak an underestimate, which is preferable to a crash in a benchmark.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss = 0

    def _sample(self) -> int:
        import psutil

        total = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                info = proc.info
                cmdline = info.get("cmdline") or []
                joined = " ".join(cmdline)
                is_ray = "ray::" in joined or "ray/_private/workers" in joined
                if proc.pid != os.getpid() and not is_ray:
                    continue
                mem = info.get("memory_info")
                if mem is not None:
                    total += int(mem.rss)
            except Exception:
                continue
        return total

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak_rss = max(self.peak_rss, self._sample())

    def __enter__(self) -> _ResourceSampler:
        self.peak_rss = self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


# -- results -----------------------------------------------------------------


@dataclass
class CaseResult:
    """One measured, checked case."""

    scenario: str
    name: str
    backend: str
    dataset: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    timings: list[float] = field(default_factory=list)
    counters: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    rows: int = 0
    bytes_processed: int = 0
    peak_rss_bytes: int = 0
    notes: list[str] = field(default_factory=list)
    skipped: str = ""
    error: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}[{self.backend}]"

    @property
    def median(self) -> Optional[float]:
        return statistics.median(self.timings) if self.timings else None

    @property
    def failed_checks(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]

    @property
    def ok(self) -> bool:
        return bool(self.skipped) or (not self.error and not self.failed_checks)

    def as_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        median = self.median
        data["median_s"] = median
        data["min_s"] = min(self.timings) if self.timings else None
        data["max_s"] = max(self.timings) if self.timings else None
        data["stdev_s"] = (
            statistics.stdev(self.timings) if len(self.timings) > 1 else 0.0
        )
        if median and self.rows:
            data["rows_per_s"] = self.rows / median
        if median and self.bytes_processed:
            data["mb_per_s"] = self.bytes_processed / median / 1e6
        return data


@dataclass
class Outcome:
    """What a measured call produced, kept alive until the case ends."""

    value: Any
    work: str
    fixture: Any
    timings: list[float]


class Case:
    """One measured, checked unit of work."""

    def __init__(
        self,
        run: BenchRun,
        name: str,
        *,
        scenario: str,
        backend: str,
        dataset: str = "",
        params: Optional[dict[str, Any]] = None,
        rows: int = 0,
        bytes_processed: int = 0,
    ) -> None:
        self.run = run
        self.checks = CheckList()
        self.result = CaseResult(
            scenario=scenario,
            name=name,
            backend=backend,
            dataset=dataset,
            params=dict(params or {}),
            rows=rows,
            bytes_processed=bytes_processed,
        )
        self._dirs: list[str] = []
        self._root = ""

    # -- workspace ---------------------------------------------------------

    def workspace(self) -> str:
        """A fresh database directory, removed when the case ends."""
        path = tempfile.mkdtemp(prefix="case_", dir=self._root)
        db_dir = os.path.join(path, "lancedb")
        os.makedirs(db_dir, exist_ok=True)
        self._dirs.append(path)
        return db_dir

    def uri(self, db_dir: str) -> str:
        """The URI for this case's backend over a database directory.

        The opt-in targets ignore the directory entirely: an Enterprise endpoint
        and an object-store bucket are addresses, not paths.
        """
        backend = self.result.backend
        if backend == "fake":
            return f"db://fake{db_dir}"
        if backend == "enterprise":
            uri = self.run.config.enterprise_uri or os.environ.get("LANCEDB_URI", "")
            if not uri:
                raise RuntimeError("enterprise backend needs LANCEDB_URI")
            return uri
        if backend == "s3":
            base = self.run.config.s3_uri or os.environ.get("BENCH_S3_URI", "")
            if not base:
                raise RuntimeError("s3 backend needs BENCH_S3_URI")
            # A bucket has no directories, but it does have key prefixes, so the
            # workspace directory this iteration would have used locally becomes
            # the prefix. That gives every iteration the same isolation on object
            # storage that a fresh temp directory gives locally.
            workspace_id = os.path.basename(os.path.dirname(db_dir)) if db_dir else ""
            return f"{base.rstrip('/')}/{workspace_id}" if workspace_id else base
        return db_dir

    @property
    def connect_kwargs(self) -> dict[str, Any]:
        backend = self.result.backend
        if backend == "fake":
            return {"api_key": "fake-api-key"}
        if backend == "enterprise":
            # The key stays in the environment rather than in an argument, so it
            # does not end up in a Ray task definition or a result file.
            kwargs: dict[str, Any] = {}
            region = os.environ.get("LANCEDB_REGION")
            if region:
                kwargs["region"] = region
            host = os.environ.get("LANCEDB_HOST_OVERRIDE")
            if host:
                kwargs["host_override"] = host
            return kwargs
        if backend == "s3":
            return {"storage_options": self.run.storage_options()}
        return {}

    @property
    def table(self) -> str:
        """Table name for this case.

        A real Enterprise endpoint is a flat, shared namespace, so the name has
        to be unique. Everything else is isolated by directory or key prefix
        already, and a stable name keeps the counter helpers simple.
        """
        if self.result.backend == "enterprise":
            return f"bench_{self.run.table_suffix}"
        return "bench"

    @property
    def storage_options(self) -> Optional[dict[str, str]]:
        """Object-store options needed to open this case's data directly."""
        return self.run.storage_options() if self.result.backend == "s3" else None

    # -- measurement -------------------------------------------------------

    def measure(
        self,
        fn: Callable[[str], T],
        *,
        fresh: bool = True,
        setup: Optional[Callable[[str], Any]] = None,
        warmup: Optional[int] = None,
        repeat: Optional[int] = None,
    ) -> Outcome:
        """Time ``fn`` over warmup + repeat iterations.

        ``fresh=True`` gives every iteration a new database directory -- the
        right shape for a write, where re-using a directory would measure an
        append into a growing table instead of the write under test.
        ``fresh=False`` builds once via ``setup`` and times ``fn`` against it,
        which is the read shape.
        """
        warmup = self.run.warmup if warmup is None else warmup
        repeat = self.run.repeat if repeat is None else repeat

        timings: list[float] = []
        value: Any = None
        fixture: Any = None
        work = ""

        with _ResourceSampler() as sampler:
            if not fresh:
                work = self.workspace()
                fixture = setup(work) if setup else None

            for index in range(warmup + repeat):
                timed = index >= warmup
                if fresh:
                    if work and (timed or index > 0):
                        # Drop the previous iteration's data as we go, so peak
                        # disk stays at one iteration rather than all of them.
                        self._drop_last_dir()
                    work = self.workspace()
                    fixture = setup(work) if setup else None

                self.run.clear_probe()
                start = time.perf_counter()
                value = fn(work)
                elapsed = time.perf_counter() - start
                if timed:
                    timings.append(elapsed)

        self.result.timings = timings
        self.result.peak_rss_bytes = sampler.peak_rss
        return Outcome(value=value, work=work, fixture=fixture, timings=timings)

    def _drop_last_dir(self) -> None:
        if len(self._dirs) > 1:
            shutil.rmtree(self._dirs.pop(0), ignore_errors=True)

    # -- recording ---------------------------------------------------------

    def counter(self, name: str, value: Any) -> None:
        self.result.counters[name] = value

    def add_counters(self, values: dict[str, Any], prefix: str = "") -> None:
        for key, value in values.items():
            self.result.counters[f"{prefix}{key}"] = value

    def note(self, message: str) -> None:
        self.result.notes.append(message)

    def set_volume(self, rows: int, bytes_processed: int = 0) -> None:
        self.result.rows = rows
        self.result.bytes_processed = bytes_processed

    def skip(self, reason: str) -> None:
        self.result.skipped = reason

    def probe(self, event: Optional[str] = None) -> dict[str, Any]:
        """Counters for the LanceDB calls this case actually issued."""
        return summarize(self.run.probe_events(), event)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Case:
        self._root = tempfile.mkdtemp(prefix="bench_", dir=self.run.run_root)
        self.run.clear_probe()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None and not isinstance(exc, KeyboardInterrupt):
            self.result.error = f"{type(exc).__name__}: {exc}"
        self.result.checks = [
            {
                "name": c.name,
                "passed": c.passed,
                "expected": _jsonable(c.expected),
                "actual": _jsonable(c.actual),
                "detail": c.detail,
            }
            for c in self.checks.results
        ]
        if not self.run.config.keep_artifacts:
            shutil.rmtree(self._root, ignore_errors=True)
        self.run.record(self.result)
        return self.result.error != "" and not self.run.config.fail_fast


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _apply_driver_patches(*, fake_remote: bool) -> None:
    """Give the driver the same environment the workers get."""
    try:
        from _ray_test_support import (
            patch_memory_profiler,
            patch_psutil_for_containers,
        )

        patch_psutil_for_containers()
        patch_memory_profiler()
    except ImportError:
        pass

    if fake_remote:
        from _fakes import install_fake_remote

        install_fake_remote()

    from .probe import install_probe

    install_probe()


class BenchRun:
    """One invocation of the suite: Ray, the run root, and the results."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.tier = TIERS[config.tier]
        self.repeat = config.repeat or self.tier.repeat
        self.warmup = self.tier.warmup if config.warmup is None else config.warmup
        self.results: list[CaseResult] = []
        self.environment = capture_environment()
        self.started = time.time()
        self.ray_init_seconds = 0.0
        self._current_scenario = ""
        #: Distinguishes this run's tables on a shared endpoint, and is what the
        #: stale-table sweep matches on.
        self.table_suffix = f"{int(self.started)}_{os.getpid()}"

        root = config.run_root or os.environ.get("BENCH_RUN_ROOT")
        if root:
            os.makedirs(root, exist_ok=True)
            self.run_root = tempfile.mkdtemp(prefix="run_", dir=root)
        else:
            # Ray builds AF_UNIX socket paths under its temp dir and macOS caps
            # those at 103 bytes, which the default /var/folders/... root blows
            # past on its own.
            base = "/tmp" if sys.platform == "darwin" else None
            self.run_root = tempfile.mkdtemp(prefix="ldbrbench_", dir=base)
        register_for_cleanup(self.run_root)

        self.probe_dir = os.path.join(self.run_root, "probe")
        os.makedirs(self.probe_dir, exist_ok=True)
        os.environ[PROBE_DIR_ENV] = self.probe_dir

    # -- sizing ------------------------------------------------------------

    def rows(self, dataset: str) -> int:
        return self.tier.rows_for(dataset)

    @property
    def blocks(self) -> int:
        return self.tier.blocks

    def storage_options(self) -> dict[str, str]:
        """Object-store options for the S3 target, read from the environment.

        Matches the set ``examples/object_storage/verify_s3.py`` already proves
        works against the emulator: ``allow_http`` because it speaks plain HTTP,
        and path-style addressing because nothing resolves
        ``<bucket>.localhost``. Both are dropped automatically against real S3,
        where no endpoint override is set.
        """
        options: dict[str, str] = {}
        for env, key in (
            ("BENCH_S3_ENDPOINT", "aws_endpoint"),
            ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
            ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
            ("AWS_REGION", "aws_region"),
        ):
            value = os.environ.get(env)
            if value:
                options[key] = value
        if options.get("aws_endpoint"):
            options.setdefault("allow_http", "true")
            options.setdefault("aws_virtual_hosted_style_request", "false")
        return options

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.run_root).free

    def has_room(self, needed: int) -> bool:
        return self.free_bytes() >= needed

    def disk_utilization(self) -> float:
        usage = shutil.disk_usage(self.run_root)
        return usage.used / usage.total if usage.total else 0.0

    def _spill_threshold(self) -> Optional[float]:
        """Raise Ray's spill guard when the disk is large but proportionally full.

        Ray refuses to spill once the filesystem is above 95% *utilised*, which
        on a big disk can mean refusing while tens of GB are still free. The tier
        guard has already checked free bytes, so on such a disk the percentage is
        the wrong question -- but the answer still has to leave the tier's
        headroom intact, so the replacement threshold is derived from that.
        """
        used = self.disk_utilization()
        if used < 0.95:
            return None
        usage = shutil.disk_usage(self.run_root)
        keep_free = 1.0 - (self.tier.min_free_bytes / usage.total)
        threshold = min(0.99, max(used + 0.005, keep_free))
        return threshold if threshold > used else None

    # -- probe -------------------------------------------------------------

    def clear_probe(self) -> None:
        clear_events(self.probe_dir)

    def probe_events(self) -> list[dict[str, Any]]:
        return read_events(self.probe_dir)

    # -- ray ---------------------------------------------------------------

    def start_ray(self, *, fake_remote: bool) -> None:
        """Initialise Ray once for the whole run.

        Pinned deliberately rather than left to autodetection: an object store
        sized from ``/dev/shm`` varies between a laptop and a hosted runner, and
        a benchmark whose parallelism moves with the machine is not comparable
        across runs.
        """
        import ray

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tests_dir = os.path.join(repo_root, "tests")
        for path in (repo_root, tests_dir):
            if path not in sys.path:
                sys.path.insert(0, path)

        # The driver runs library code too, so it needs the same patches and
        # backends the workers get.
        _apply_driver_patches(fake_remote=fake_remote)

        if fake_remote:
            os.environ["LANCEDB_RAY_BENCH_FAKE_REMOTE"] = "1"
        os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

        num_cpus = self.config.num_cpus or self._default_cpus()
        object_store = self.config.object_store_bytes or self._default_object_store()

        runtime_env = {
            "worker_process_setup_hook": "benchmarks._bench_worker.setup_worker",
            "env_vars": {
                "PYTHONPATH": os.pathsep.join(
                    [repo_root, tests_dir, os.environ.get("PYTHONPATH", "")]
                ).strip(os.pathsep),
                "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
                PROBE_DIR_ENV: self.probe_dir,
                "LANCEDB_RAY_BENCH_FAKE_REMOTE": "1" if fake_remote else "0",
            },
        }

        if ray.is_initialized():
            ray.shutdown()

        ray_temp = os.path.join(self.run_root, "ray")
        os.makedirs(ray_temp, exist_ok=True)

        system_config: dict[str, Any] = {}
        threshold = self._spill_threshold()
        if threshold is not None:
            system_config["local_fs_capacity_threshold"] = threshold
            self.environment["spill_threshold"] = threshold

        start = time.perf_counter()
        ray.init(
            num_cpus=num_cpus,
            object_store_memory=object_store,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            _temp_dir=ray_temp,
            runtime_env=runtime_env,
            _system_config=system_config or None,
        )
        self.ray_init_seconds = time.perf_counter() - start
        self.environment["ray_num_cpus"] = num_cpus
        self.environment["ray_object_store_bytes"] = object_store

    def _default_cpus(self) -> int:
        available = os.cpu_count() or 4
        # Leave a core for the driver and the OS; a fully saturated machine
        # measures scheduler contention rather than the library.
        return max(2, min(available - 1, 8))

    def _default_object_store(self) -> int:
        import psutil

        total = int(psutil.virtual_memory().total)
        # Ray's default reads /dev/shm, which is 64MB in many containers and
        # half of RAM on a bare runner -- neither is reproducible. macOS refuses
        # anything above 2GB outright, so that is the ceiling everywhere; using
        # the same number on both keeps a laptop run comparable to a CI one.
        ceiling = 2 * _GB
        return int(min(ceiling, max(512 * 1024**2, total * 0.25)))

    def stop_ray(self) -> None:
        import ray

        if ray.is_initialized():
            ray.shutdown()

    def record(self, result: CaseResult) -> None:
        self.results.append(result)

    # -- cases -------------------------------------------------------------

    def case(
        self,
        name: str,
        *,
        backend: str,
        dataset: str = "",
        params: Optional[dict[str, Any]] = None,
    ) -> Case:
        return Case(
            self,
            name,
            scenario=self._current_scenario,
            backend=backend,
            dataset=dataset,
            params=params,
        )

    @contextlib.contextmanager
    def scenario(self, name: str) -> Iterator[None]:
        previous = self._current_scenario
        self._current_scenario = name
        try:
            yield
        finally:
            self._current_scenario = previous

    # -- output ------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "tier": self.tier.name,
            "repeat": self.repeat,
            "warmup": self.warmup,
            "started": self.started,
            "duration_s": time.time() - self.started,
            "ray_init_s": self.ray_init_seconds,
            "environment": self.environment,
            "results": [r.as_dict() for r in self.results],
        }

    def write_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2, sort_keys=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.run_root, ignore_errors=True)
        _TO_CLEAN.discard(self.run_root)
        os.environ.pop(PROBE_DIR_ENV, None)
