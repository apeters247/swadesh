# pytest plugin and assertions: score a user's own systems, inside their test suite

Status: additive tooling (shortlist item s4), written 2026-11-27. No score changes; every helper calls
`corpus_score` and nothing else.

## Why it exists

A team that uses swadesh wants two things in its own CI: "this system must not drop below 30 BLEU",
and "record what every system scored so the numbers show up in the job log". Both are one line of
glue, and everyone writes it slightly differently. This ships the two pieces and the entry point that
makes the second one automatic.

## What ships

**`swadesh.testing` - assertions, no pytest import.** `assert_score_at_least(metric, hyps, refs,
minimum, name=None)`, `assert_score_at_most(...)`, and the metric-specific shorthands
`assert_bleu_at_least`, `assert_chrf_at_least`, `assert_ter_at_most`. Each returns the score object on
success and raises `AssertionError` on failure with the metric's own formatted line (so the failure
shows the number, the signature and the seed) plus the bound that was missed. TER's direction differs
from BLEU's, which is why the "at most" form exists separately.

**`swadesh.pytest_plugin` - the pytest plugin.** Registered through the `pytest11` entry point, so
`pip install swadesh` is enough for pytest to load it; also loadable explicitly with
`-p swadesh.pytest_plugin` when the entry point is not (yet) installed.

* A `swadesh_record` fixture: `swadesh_record("sys-a", BLEU(), hyps, refs)` scores, records
  `{name, metric, score, signature}` and returns the score. Function-scoped; the records accumulate
  for the whole session.
* `--swadesh-report PATH` writes the accumulated records as JSON at the end of the session.
* A terminal summary lists every recorded score whether or not a report path was given.

## What it is not

* Not a way to change a score: the fixture and the assertions call the same `corpus_score` the library
  does, so a recorded number is the number.
* Not a replacement for the differential suite. It asserts user systems against user data; it does not
  compare against sacreBLEU.

## CI

`.github/workflows/ci.yml` already runs the Rust tests, clippy and the differential suite on
Linux/macOS/Windows across Python 3.10-3.14 and both numpy lines. A `pytest plugin` step is added to
the differential job so the plugin and the assertions are exercised on every push.

## Invariants (each asserted by a test)

1. `assert_score_at_least` raises `AssertionError` whose message contains the metric name and the
   score line, and returns the score object when the bound is met.
2. In a real pytest run the `swadesh_record` fixture populates the terminal summary and, with
   `--swadesh-report PATH`, writes a JSON list whose entry is `{name, metric, score, signature}`.
3. The plugin is importable and loads without touching any score path: a test session that never asks
   for the fixture is unchanged.
