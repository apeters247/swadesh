# Phase 5 status: making python/swadesh standalone

The goal is that `patch()` is the only place the Python package touches sacreBLEU. Five modules
imported it. This records where each one stands and what is genuinely left.

## Done

### `significance.py`

`Result`, `_compute_p_value`, `estimate_ci` and `PairedTest` are written for swadesh, from
`docs/spec/significance.md`. The p-value and interval are verified equal to the oracle on random
inputs; `Result`'s `repr` and its `None` defaults are matched exactly. The two remaining imports are
`install()`/`uninstall()` — which is the `patch()` adapter and must touch sacreBLEU to redirect it —
and the metric-class checks inside the native scoring helpers.

### Provenance rule restructured

`tests/test_provenance.py` now separates the allowlist into `PERMANENT` and `PENDING`:

* **`PERMANENT`** — `__init__.py` (the `patch()` adapter) and `verify.py` (the comparison tool). These
  will never stop importing sacreBLEU, because interoperating with it *is* their job.
* **`PENDING`** — `metrics.py`, `significance.py`, `cli.py`. Each leaves as it is rewritten.

A new test fails if `PENDING` ever grows, and if a module appears in both sets.

## Pending, with the actual obstacle named

### `metrics.py` — the largest

It subclasses sacreBLEU's `BLEU`, `CHRF` and `TER` to inherit four things: `_cache_references` (36
lines), `_aggregate_and_compute` (8 lines, per metric), `_extract_reference_info` (26 lines), and the
`BLEUScore` / `CHRFScore` / `TERScore` value objects whose formatting the CLI prints. swadesh already
overrides `corpus_score` and `sentence_score`, so the inheritance is almost entirely for those.

Standalone needs its own score objects with `format()`, `__str__` and the signature plumbing, plus
the aggregate arithmetic for three metrics. That is the day of work the plan anticipated, and it is
the next slice.

### `cli.py` — blocked on a packaging decision, not on code

It is 19 lines: `swadesh.patch()`, then delegate to `sacrebleu.sacrebleu.main`. The delegation exists
because sacreBLEU's CLI *is* the contract, and `patch()` makes every metric on the path native.

The obvious standalone replacement is to call the **Rust CLI**, which already exists
(`crates/swadesh-cli`, binary `swadesh`, byte-verified against sacreBLEU). Two obstacles:

1. `maturin` builds only the extension module (`module-name = "swadesh._swadesh"`); a wheel does not
   carry the CLI binary, so `python -m swadesh.cli` could not find it.
2. The parity tests run `python -m swadesh.cli` and compare stdout byte for byte with the sacreBLEU
   binary across 50+ flag combinations. Switching the implementation behind that entry point is a
   behavioural change that must be re-verified end to end, not a rename.

So `cli.py` needs either a maturin binary target added to the wheel, or an in-process binding to the
CLI logic. Both are packaging work; neither is a licence problem once done, because the CLI is
already clean-room.

### `verify.py` — permanent

It compares swadesh against sacreBLEU **on the user's own data**. That comparison is the tool's
entire purpose, so it will never be standalone and is listed under `PERMANENT`.

## What this means for licensing

The Apache-2.0 notice covers `metrics.py` and `cli.py` today, plus `__init__.py` and `verify.py` for
good. It will cover `metrics.py` and `cli.py` until those two slices land. `significance.py` is
clean-room apart from its adapter seam.