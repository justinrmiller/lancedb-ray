# Baselines

`ci-ubuntu-latest.json` is the timing baseline the CI job diffs against. It is
**committed deliberately**, never written automatically: a baseline that updates
itself ratchets away exactly the regression it is supposed to catch.

To regenerate it, take the `benchmark-results` artifact from a green run of the
`benchmark` job on `main`, and commit its JSON here under this name. Do that
only when a change to the timings is understood and intended -- a dependency
bump, a deliberate rewrite -- and say which in the commit message.

Until the file exists the CI job still runs; it prints `no baseline ... skipping
comparison` and gates on correctness and counters alone.

Timings from a different machine are not comparable. A baseline produced on a
laptop would make the CI job fail or pass for reasons that have nothing to do
with the code, so this file must come from a hosted runner.
