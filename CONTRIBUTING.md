# Contributing

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Workflow

```bash
make test    # pytest with branch coverage (gate: 95%)
make lint    # ruff check + format check + mypy (strict)
make fix     # auto-fix and format
```

All three must pass before a PR merges. CI runs the suite on Python 3.12.

## Testing against both backends

Most read/write tests use the `backend` fixture, which is parametrized over the local
backend and a fake LanceDB Cloud/Enterprise backend. The fake (`tests/_fakes.py`) wraps a
real local database but narrows it to the remote API surface, including its restrictions
(`to_arrow`/`to_lance` raise, `optimize` is a no-op). Every query genuinely executes, so
the sharding and batching logic is tested against real results.

The fake is installed on Ray workers too, via `runtime_env.worker_process_setup_hook`, so
`db://` tests exercise real distributed execution rather than driver-only code paths.

When you add a code path that differs between backends, add the test to the parametrized
suite rather than writing two versions of it.

## Live Enterprise tests

`tests/test_enterprise_live.py` mirrors the fake-backed assertions against a real service,
which is how the fake gets validated. They skip unless credentials are present:

```bash
LANCEDB_URI=db://your-db LANCEDB_API_KEY=... pytest -m enterprise
```

These create uniquely-named tables and drop them afterwards.

## Guidelines

- New public arguments need a docstring entry and a test.
- Prefer adding to `_plan.py` for anything that can be expressed as pure arithmetic —
  it is far cheaper to test exhaustively there than through Ray.
- Keep secrets out of `LanceDBConnectionSpec` reprs and log lines.
