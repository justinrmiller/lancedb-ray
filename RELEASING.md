# Releasing

Versions are derived from commit messages. Nobody edits a version number by hand.

- [How a release happens](#how-a-release-happens)
- [What each commit type does to the version](#what-each-commit-type-does-to-the-version)
- [Cutting a release](#cutting-a-release)
- [First-time setup](#first-time-setup)
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

This turns into a hard block if you protect `main` with **required status
checks**: a check that never runs never reports, so the release PR sits at
*Expected — Waiting for status to be reported* and cannot be merged. `main` is
unprotected today. If you protect it later, do one of: leave the CI checks
unrequired, allow administrators to bypass the rule, or switch release-please
to a GitHub App token so its PRs get CI like any other.

## What each commit type does to the version

| Commit | Bump | In the changelog |
| --- | --- | --- |
| `fix: …` | patch (`0.3.1` → `0.3.2`) | Bug Fixes |
| `feat: …` | minor (`0.3.1` → `0.4.0`) | Features |
| `perf:`, `refactor:`, `docs:`, `deps:`, `revert:` | patch | own section |
| `feat!: …`, or a `BREAKING CHANGE:` footer | minor while below 1.0.0, major after | ⚠ BREAKING CHANGES |
| `test:`, `style:`, `ci:`, `build:`, `chore:` | none on their own | hidden |

Note the last row. release-please skips a release whose notes would be empty, so
the hidden types never open a release PR by themselves: a week of `chore:` and
`ci:` commits leaves no PR behind, and they ship with the next visible change. A
`Release-As` footer (below) is the exception — it always makes the notes, whatever
the commit's type. A commit release-please cannot parse as conventional is ignored
outright.

Below 1.0.0 a breaking change bumps the minor version (`bump-minor-pre-major` in
`.release-please-config.json`), so the jump to 1.0.0 stays a deliberate act rather
than a side effect of the first `feat!:`. To make it, merge an empty commit
carrying a `Release-As` footer:

```bash
git commit --allow-empty -m "chore: release 1.0.0" -m "Release-As: 1.0.0"
```

The same trick forces any other version, e.g. to skip a number.

### The type sets the bump, not the scope

A scope does not soften a type. `feat(examples): …` cuts a minor release of the
library and `fix(benchmarks): …` a patch, although neither directory ships in the
sdist or the wheel — PyPI users get a new version number with nothing new in it.
Pick the type by what reaches a `pip install`:

| The change is to | Use |
| --- | --- |
| `lancedb_ray/`, or published metadata (dependencies, extras, `README.md` as the PyPI page) | `feat:`, `fix:`, `perf:`, `refactor:`, `deps:` |
| user-facing docs or `examples/` | `docs:` — a patch, listed under Documentation |
| `tests/`, `benchmarks/`, CI, tooling, maintainer docs such as this file | `test:`, `ci:`, `build:`, `chore:` — hidden, no release of their own |

release-please's `exclude-paths` option looks like a way to enforce this
mechanically. It is deliberately not used: it drops every commit whose files all
sit under an excluded directory, and an empty commit has no files, so it would
drop the `Release-As` commit above too. It also cannot see a PR that touches an
example *and* `README.md`, which is most of them.

### Squash-merge PRs

release-please reads the commit that lands on `main`. With squash merges that is
the PR title, so the title is what has to be conventional — one reviewable message
per change instead of whatever the branch's intermediate commits happened to say.

A merge commit undoes this: release-please then reads every commit on the branch,
so one `feat(benchmarks):` among a `fix(io):` PR's commits turns a patch release
into a minor one. Read the title in GitHub's squash box before confirming — it is
the changelog line.

**A merged title was wrong.** Edit the *merged* PR's description and add

```
BEGIN_COMMIT_OVERRIDE
fix(io): the message release-please should have read
END_COMMIT_OVERRIDE
```

then re-run the most recent `Release` run from the Actions tab (or push anything
to `main`). release-please uses the override in place of the squash commit's
message and rewrites the open release PR to match. This only works for squash
merges. Do it before merging the release PR; afterwards the version is spent.

### Where the version lives at runtime

`pyproject.toml` is the only place the number is written. `lancedb_ray.__version__`
reads it back from the installed distribution metadata, so the two cannot drift.
`main` carries the version of the latest release — only a release PR moves it — and
an uninstalled source checkout reports `0.0.0.dev0`.

## Cutting a release

0.1.0 was the first, on 2026-09-23. Every release since works the same way:

1. **Squash-merge** changes into `main` with a conventional title whose type matches
   what ships ([above](#the-type-sets-the-bump-not-the-scope)).
2. The `Release` workflow opens or updates **`chore(main): release X.Y.Z`** within a
   minute or two. Check the version it chose and read its `CHANGELOG.md` diff. A
   version higher than the changes warrant means a title was wrong — fix it with an
   override (above), not by editing the number. Push wording edits to the release
   PR's own branch.
3. Batch as much as you like first: the PR recomputes on every push to `main`, and
   nothing ships until it is merged.
4. Merge it (squash). Watch the `Release` run: `release-please` → `verify` (the full
   CI suite against the tag) → `publish`. If the `pypi` environment has required
   reviewers, approve the deployment when Actions prompts.
5. Check <https://pypi.org/project/lancedb-ray/> shows the new version.

If anything after step 4 fails, see [When something goes
wrong](#when-something-goes-wrong).

## First-time setup

Four things have to be clicked by a human. The workflows cannot create any of
them, and until all four exist the release either never starts or never uploads.
All four were done for 0.1.0; they are recorded here for when one has to be redone,
e.g. after the repository or workflow is renamed.

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

The project did not exist on PyPI before 0.1.0, so it was registered as a
*pending* publisher, which the first upload turned into a normal one. Now that the
project exists, a publisher is managed on the project itself, not the account-wide
pending form:

1. Sign in at <https://pypi.org> (a PyPI account requires 2FA; set that up first
   if you have not).
2. Go to <https://pypi.org/manage/project/lancedb-ray/settings/publishing/>.
3. Under **Add a new publisher**, choose the **GitHub** tab and fill in the fields
   below (the project form has no project-name field; the project is implied):

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `lancedb-ray` |
   | Owner | `justinrmiller` |
   | Repository name | `lancedb-ray` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. Click **Add**, then remove the publisher it replaces.

Every field is matched exactly against the OIDC claims at upload time. `release.yml`
is the file name only, not a path and not the workflow's display name. If you
renamed the environment in step 3, that name has to match here too.

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
  not match (see the last entry below). Fix the outside
  cause if there is one, then *Re-run failed jobs* from the Actions run page.
  Re-running `publish` is safe wherever it stopped: attaching to the GitHub
  release overwrites, and the PyPI step skips files an earlier attempt already
  uploaded.
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

**A release PR you did not want** — typically one opened by a lone `docs:` commit
you would rather ship with the next fix. You can simply leave it open: it keeps
accumulating, and nothing ships until it is merged. To get it out of the way,
close it *and* add the label `autorelease: snooze`. release-please
then leaves it closed until a commit changes the release notes, at which point it
reopens the same PR with the new notes and drops the label. A plain close without the label achieves
nothing: the next push to `main` opens a fresh PR with the same contents.

**Something wrong was published.** A version number on PyPI is spent permanently;
deleting a release does not free it. Fix forward with a `fix:` commit and let the
next release PR cut the next patch. If the bad version is actively harmful, *yank*
it from the PyPI project page: `pip` will stop resolving to it, but anyone with an
exact pin can still install it. Yanking is done on PyPI, not from this repository.

**Uploads suddenly fail with an OIDC or "not a valid publisher" error.** Something
in the four fields of step 4 no longer matches: the repository was renamed or
transferred, the workflow file was renamed, or the `pypi` environment was renamed
or deleted. Update the publisher on PyPI to match.
