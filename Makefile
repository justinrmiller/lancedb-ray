.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  build       Install the project with dev dependencies"
	@echo "  test        Run pytest with coverage"
	@echo "  test-fast   Run pytest without coverage"
	@echo "  enterprise  Run live LanceDB Enterprise tests (needs LANCEDB_URI/LANCEDB_API_KEY)"
	@echo "  lint        Run ruff check, ruff format --check and mypy"
	@echo "  fix         Auto-fix lint issues and format code"
	@echo "  example     Run the end-to-end quickstart example"
	@echo "  example-mcap  Generate sample MCAP logs and index them into LanceDB"
	@echo "  benchmark   Run the benchmark suite (TIER=smoke|ci|local|full)"
	@echo "  benchmark-compare   Run and diff against the committed baseline"
	@echo "  benchmark-baseline  Rewrite the baseline for this tier (deliberate)"
	@echo "  benchmark-enterprise Run the opt-in live Enterprise target"
	@echo "  benchmark-s3         Run the opt-in object-storage target"
	@echo "  benchmark-clean      Remove stray benchmark run directories"
	@echo "  clean       Remove build artifacts and caches"

.PHONY: build
build:
	uv pip install -e ".[dev]"

.PHONY: test
test:
	uv run pytest

.PHONY: test-fast
test-fast:
	uv run pytest -q --no-cov

.PHONY: enterprise
enterprise:
	uv run pytest -m enterprise -v

.PHONY: lint
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

.PHONY: fix
fix:
	uv run ruff check --fix --unsafe-fixes .
	uv run ruff format .

.PHONY: example
example:
	uv run python examples/quickstart/quickstart.py

# Generates synthetic recordings under a temp directory, then indexes them.
.PHONY: example-mcap
example-mcap:
	uv run python examples/mcap_ingest/make_sample_mcap.py --out $${MCAP_LOGS:-/tmp/ldbr_mcap_logs}
	uv run python examples/mcap_ingest/ingest_mcap.py \
		--logs $${MCAP_LOGS:-/tmp/ldbr_mcap_logs} \
		--uri $${MCAP_DB:-/tmp/ldbr_mcap_db} \
		--mode overwrite

# Tier sizes the run; every correctness check runs in every tier. ARGS is
# passed through, e.g. ARGS="--scenario read --repeat 5".
TIER ?= local
ARGS ?=
BASELINE = benchmarks/baselines/ci-ubuntu-latest.json

.PHONY: benchmark
benchmark:
	uv run python -m benchmarks --tier $(TIER) $(ARGS)

.PHONY: benchmark-compare
benchmark-compare:
	uv run python -m benchmarks --tier $(TIER) --compare $(BASELINE) $(ARGS)

.PHONY: benchmark-baseline
benchmark-baseline:
	uv run python -m benchmarks --tier $(TIER) --write-baseline $(BASELINE) $(ARGS)

# Creates uniquely named tables and drops them, including any left by a run
# that was killed. Needs LANCEDB_URI and LANCEDB_API_KEY.
.PHONY: benchmark-enterprise
benchmark-enterprise:
	uv run python -m benchmarks --tier $(TIER) --backend enterprise $(ARGS)

# Needs BENCH_S3_URI (and BENCH_S3_ENDPOINT for an emulator); see
# examples/object_storage for a compose file that provides one.
.PHONY: benchmark-s3
benchmark-s3:
	uv run python -m benchmarks --tier $(TIER) --backend s3 $(ARGS)

.PHONY: benchmark-clean
benchmark-clean:
	rm -rf benchmarks/results
	rm -rf /tmp/ldbrbench_* $${BENCH_RUN_ROOT:-/nonexistent}/run_*
	@echo "removed benchmark run directories and results"

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -f .coverage .coverage.* coverage.xml
	find . -type d -name htmlcov -exec rm -rf {} +
	rm -rf benchmarks/results
	rm -rf /tmp/ldbrbench_*
	rm -rf /tmp/ldbr_mcap_logs /tmp/ldbr_mcap_db
