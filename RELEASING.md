# Releasing

Versions are derived from commit messages. Nobody edits a version number by hand.

- [How a release happens](#how-a-release-happens)
- [What each commit type does to the version](#what-each-commit-type-does-to-the-version)
- [First-time setup](#first-time-setup)
- [Cutting the first release](#cutting-the-first-release)
- [Rehearsing against TestPyPI](#rehearsing-against-testpypi)
- [When something goes wrong](#when-something-goes-wrong)

## How a release happens

1. Commits land on `main` using [Conventional Commits](https://www.conventionalcommits.org/).
2. [release-please](https://github.com/googleapis/release-please) keeps a PR open
   titled `chore(main): release X.Y.Z`. It bumps `version` in `pyproject.toml`,
   updates `.release-please-manifest.json`, and rewrites `CHANGELOG.md`. It
   recomputes `X.Y.Z` on every push to `main`.
3. Merging that PR *is* the release. `.github/workflows/release.yml` then tags the
   merge commit `vX.Y.Z`, cuts the GitHub release, re-runs the full CI suite
   against the tag, checks the built version against the tag, uploads the sdist
   and wheel to PyPI, and attaches them to the GitHub release.
4. If nothing has landed since the last tag, there is no PR and nothing is
   published.

Two consequences of that design are worth knowing before you rely on it.

**The tag is created before the suite re-runs.** release-please tags and cuts the
GitHub release in the same step that reports `release_created`, so a failure in
`verify` or `publish` leaves `vX.Y.Z` and a GitHub release in place with nothing on
PyPI. That is recoverable — see [When something goes
wrong](#when-something-goes-wrong) — and it is the reason the gate exists at all: a
PyPI upload is the part that cannot be undone.

**The release PR itself gets no CI run.** GitHub deliberately does not trigger
workflows for events caused by `GITHUB_TOKEN`, so the `pull_request` triggers in
`ci.yml` never fire on release-please's PR. This is tolerable because that PR
changes only version metadata — every code change in it already passed CI on its
own PR — and because `verify` re-runs the whole suite against the tagged commit
before anything is published. If you want CI on the release PR anyway, give the
`release-please` job a PAT or GitHub App token instead of `GITHUB_TOKEN`.

## What each commit type does to the version

| Commit | Bump | In the changelog |
| --- | --- | --- |
| `fix: …` | patch (`0.3.1` → `0.3.2`) | Bug Fixes |
| `feat: …` | minor (`0.3.1` → `0.4.0`) | Features |
| `perf:`, `refactor:`, `docs:`, `deps:`, `revert:` | patch | own section |
| `feat!: …`, or a `BREAKING CHANGE:` footer | minor while below 1.0.0, major after | ⚠ BREAKING CHANGES |
| `test:`, `style:`, `ci:`, `build:`, `chore:` | patch | hidden |

Note the last row: release-please bumps a patch for *any* conventional commit it
can parse, and `hidden` only keeps it out of the changelog body. A week of
`chore:` and `ci:` commits will still open a release PR — just an empty-looking
one. Close it, or merge it and spend the patch number; it reopens on the next
push either way. Only a commit release-please cannot parse as conventional is
ignored outright.

Below 1.0.0 a breaking change bumps the minor version (`bump-minor-pre-major` in
`.release-please-config.json`), so the jump to 1.0.0 stays a deliberate act rather
than a side effect of the first `feat!:`. To make it, merge an empty commit
carrying a `Release-As` footer:

```bash
git commit --allow-empty -m "chore: release 1.0.0" -m "Release-As: 1.0.0"
```

The same trick forces any other version, e.g. to skip a number.

### Squash-merge PRs

release-please reads the commit that lands on `main`. With squash merges that is
the PR title, so the title is what has to be conventional — one reviewable message
per change instead of whatever the branch's intermediate commits happened to say.

### Where the version lives at runtime

`pyproject.toml` is the only place the number is written. `lancedb_ray.__version__`
reads it back from the installed distribution metadata, so the two cannot drift.
Between releases `main` carries the placeholder `0.0.0`; an uninstalled source
checkout reports `0.0.0.dev0`.

## First-time setup

Four things have to be clicked by a human. The workflows cannot create any of
them, and until all four exist the release either never starts or never uploads.

### 1. Let Actions open pull requests

*Settings → Actions → General → Workflow permissions*

- Tick **Allow GitHub Actions to create and approve pull requests**.

Without it, the `release-please` job fails with a 403 the first time it tries to
open the release PR.

You do **not** need to switch the radio above it to *Read and write permissions*.
`release.yml` requests `contents`, `pull-requests` and `issues` write per job, which
works with the safer read-only default.

### 2. Turn on squash merging

*Settings → General → Pull Requests*

- Tick **Allow squash merging**.
- Set **Default commit message** to **Pull request title**.
- Optionally untick *Allow merge commits* so the choice cannot be made by accident.

### 3. Create the `pypi` environment

*Settings → Environments → New environment*, named exactly `pypi` (the name is
part of what PyPI verifies in step 4).

Two optional hardening settings, both recommended:

- **Required reviewers** — add yourself. Every upload then waits for a human click
  in the Actions UI. The tag, the GitHub release and the test run all still happen
  automatically; only the PyPI upload waits.
- **Deployment branches and tags** — *Selected branches and tags*, add `main`. The
  publish job is reachable only from a push to `main`, so this closes off any
  future workflow from borrowing the environment's identity.

### 4. Register the trusted publisher on PyPI

This is [Trusted Publishing](https://docs.pypi.org/trusted-publishers/): the
workflow exchanges a short-lived OIDC token for an upload token on each run. There
is no API token to store in repository secrets, so there is none to leak or rotate.

The project does not exist on PyPI yet, so it is registered as a *pending*
publisher, which becomes a normal one the moment the first upload creates the
project.

1. Sign in at <https://pypi.org> (a PyPI account requires 2FA; set that up first
   if you have not).
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Under **Add a new pending publisher**, choose the **GitHub** tab and fill in:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `lancedb-ray` |
   | Owner | `justinrmiller` |
   | Repository name | `lancedb-ray` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. Click **Add**.

Every field is matched exactly against the OIDC claims at upload time. `release.yml`
is the file name only, not a path and not the workflow's display name. If you
renamed the environment in step 3, that name has to match here too.

## Cutting the first release

With the four settings in place, merge this branch to `main`. Then:

1. The `Release` workflow runs and opens **`chore(main): release 0.1.0`**. It is
   `0.1.0` and not `0.0.1` because `pyproject.toml` and
   `.release-please-manifest.json` start at the placeholder `0.0.0`, and the
   history contains `feat:` commits.
2. Read the generated `CHANGELOG.md` in that PR. This first one covers the whole
   history, so it is the one worth editing by hand if anything reads badly — push
   edits straight to the release PR's own branch.
3. Merge it. Watch the `Release` run: `release-please` → `verify` (the full CI
   suite against the tag) → `publish`.
4. If you set required reviewers on the `pypi` environment, approve the deployment
   when Actions prompts.
5. `pip install lancedb-ray` now resolves.

After that, releasing is merging the PR that release-please keeps open.

## Rehearsing against TestPyPI

Packaging mistakes surface on every pull request already: the `package` job in
`ci.yml` builds the real sdist and wheel, runs `twine check --strict`, installs the
wheel and asserts `py.typed` shipped. Locally that is `make dist`.

For a full end-to-end rehearsal including an upload, register a second pending
publisher at <https://test.pypi.org/manage/account/publishing/> with the same
fields, and on a scratch branch add a repository URL to the publish step:

```yaml
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
```

TestPyPI burns version numbers the same way the real index does, so use throwaway
ones via `Release-As`.

## When something goes wrong

**`verify` or `publish` failed, so the tag exists but PyPI has nothing.** No
version number has been spent: spending one takes a successful upload. What to do
depends on where the fault is, because re-running a job re-tests and rebuilds the
*same tagged commit* — it never picks up anything pushed to `main` since.

- *The fault is outside the tagged code* — a lost runner, a network or PyPI outage,
  a benchmark tripping on a noisy runner, or a trusted-publisher field that does
  not match (the most likely failure on the very first release). Fix the outside
  cause if there is one, then *Re-run failed jobs* from the Actions run page.
- *The fault is in the tagged code* — `verify` found a real bug. Re-running cannot
  help. Leave the tag and the GitHub release where they are (release-please reads
  them to work out what has shipped; deleting them confuses its next run), edit
  the release notes to say the version was never published, and fix forward with
  a `fix:` commit. The next release PR cuts the next patch. PyPI does not need
  version numbers to be contiguous, so the skipped one costs nothing.

**The version guard failed** (`release vX.Y.Z is version 'X.Y.Z', but the build
produced …`). `pyproject.toml` on the tagged commit does not hold the version
release-please says it released — almost always a merge that clobbered its edit.
Nothing was uploaded. This is a fault in the tagged code, so treat it as above:
do not re-run, fix forward. The next release PR rewrites the `version` line
whatever it currently says.

**A release PR you did not want.** Close it. It reopens on the next push to `main`
with the same accumulated changelog, so closing costs nothing.

**Something wrong was published.** A version number on PyPI is spent permanently;
deleting a release does not free it. Fix forward with a `fix:` commit and let the
next release PR cut the next patch. If the bad version is actively harmful, *yank*
it from the PyPI project page: `pip` will stop resolving to it, but anyone with an
exact pin can still install it. Yanking is done on PyPI, not from this repository.

**Uploads suddenly fail with an OIDC or "not a valid publisher" error.** Something
in the four fields of step 4 no longer matches: the repository was renamed or
transferred, the workflow file was renamed, or the `pypi` environment was renamed
or deleted. Update the publisher on PyPI to match.
