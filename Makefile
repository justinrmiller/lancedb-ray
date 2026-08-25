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
	uv run python examples/quickstart.py

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -f .coverage .coverage.* coverage.xml
	find . -type d -name htmlcov -exec rm -rf {} +
