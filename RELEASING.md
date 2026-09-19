# Releasing

Versions are derived from commit messages. Nobody edits a version number by hand.

## How a release happens

1. Commits land on `main` using [Conventional Commits](https://www.conventionalcommits.org/).
2. [release-please](https://github.com/googleapis/release-please) keeps a PR open
   titled `chore(main): release X.Y.Z`. It bumps `version` in `pyproject.toml`,
   rewrites `CHANGELOG.md`, and recomputes `X.Y.Z` on every push.
3. Merging that PR *is* the release. The `Release` workflow tags the merge commit
   `vX.Y.Z`, cuts the GitHub release, re-runs the full CI suite against the tag,
   builds the sdist and wheel, and uploads them to PyPI.
4. If nothing has landed since the last tag, there is no PR and nothing is
   published.

## What each commit type does to the version

| Commit | Bump | In the changelog |
| --- | --- | --- |
| `fix: …` | patch (`0.3.1` → `0.3.2`) | Bug Fixes |
| `feat: …` | minor (`0.3.1` → `0.4.0`) | Features |
| `perf:`, `refactor:`, `docs:`, `deps:` | patch | own section |
| `feat!: …`, or a `BREAKING CHANGE:` footer | minor while below 1.0.0, major after | ⚠ Breaking Changes |
| `test:`, `style:`, `ci:`, `build:`, `chore:` | patch | hidden |

Note the last row: release-please bumps a patch for *any* conventional commit it
can parse, and `hidden` only keeps it out of the changelog body. A week of
`chore:` and `ci:` commits will still open a release PR — just an empty-looking
one. Close it, or merge it and spend the patch number; it reopens on the next
push either way. Only a commit release-please cannot parse as conventional is
ignored outright.

Below 1.0.0 a breaking change bumps the minor version (`bump-minor-pre-major`), so
the jump to 1.0.0 stays a deliberate act rather than a side effect of the first
`feat!:`. To make it, merge an empty commit carrying a `Release-As` footer:

```bash
git commit --allow-empty -m "chore: release 1.0.0" -m "Release-As: 1.0.0"
```

The same trick forces any other version, e.g. to skip a number.

## Squash-merge PRs

release-please reads the commit that lands on `main`. With squash merges that is
the PR title, so the title is what has to be conventional — one reviewable message
per change instead of whatever the branch's intermediate commits happened to say.
Set *Settings → General → Pull Requests → Allow squash merging* and default the
commit message to the PR title.

## Where the version lives at runtime

`pyproject.toml` is the only place the number is written. `lancedb_ray.__version__`
reads it back from the installed distribution metadata, so the two cannot drift.
An uninstalled source checkout reports `0.0.0.dev0`.

## One-time setup

These are account and repository settings; the workflows cannot create them.

1. **Let Actions open PRs.** *Settings → Actions → General → Workflow permissions*
   → tick *Allow GitHub Actions to create and approve pull requests*. Without it
   release-please fails with a 403 when it tries to open the release PR.
2. **Create the `pypi` environment.** *Settings → Environments → New environment*,
   named `pypi`. Adding yourself as a required reviewer puts a human approval in
   front of every upload; the release still tags and builds without it.
3. **Register the trusted publisher on PyPI.** The project does not exist yet, so
   add a *pending* publisher at
   <https://pypi.org/manage/account/publishing/>:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `lancedb-ray` |
   | Owner | `justinrmiller` |
   | Repository name | `lancedb-ray` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   This is [Trusted Publishing](https://docs.pypi.org/trusted-publishers/): the
   workflow exchanges a short-lived OIDC token for an upload token per run, so
   there is no API token in repository secrets to leak or rotate.

Once those three are done, the next push to `main` opens the first release PR.

## Trying it without touching PyPI

`uv build` and `uvx twine check --strict dist/*` run on every PR (the `package` CI
job), which is where packaging mistakes surface. For an end-to-end rehearsal,
register a second pending publisher on <https://test.pypi.org> with the same
fields, and add `repository-url: https://test.pypi.org/legacy/` to the publish
step on a scratch branch.

## When a release goes wrong

A version number on PyPI is spent permanently — deleting a release does not free
it. Fix forward with a `fix:` commit and let the next release PR cut the next
patch. Yanking (`pip` will not resolve to it, but it stays downloadable by exact
pin) is done from the PyPI project page, not from here.
