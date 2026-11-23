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

### `metrics.py` — only the subclass scaffolding remains

Everything the metric classes *do* is swadesh's own: the score objects with their `format()` and
`__str__`, `sum_of_lists`, `_aggregate_and_compute`, `_score_from_row`, `_segment_ref_cache`,
`_sentence_score`, and all six fallback statistics methods — TER's two, and now BLEU's and chrF's
`_extract_reference_info` and `_compute_segment_statistics`.

What is still inherited is the scaffolding a subclass cannot avoid: the constructors and their
argument validation, the tokenizer plumbing, the constants, the signature printing, and
`_compute_score_from_stats` (whose identity the bootstrap fast path recognises). So `metrics.py`
stays in the provenance `PENDING` set until that scaffolding is rewritten or the classes stop being
subclasses.

### `cli.py` — measured, and staying a documented adapter

It is 19 lines: `swadesh.patch()`, then delegate to `sacrebleu.sacrebleu.main`. The delegation exists
because sacreBLEU's CLI *is* the contract, and `patch()` makes every metric on the path native. The
question was whether to replace it with the Rust CLI. Measured on troy-1, 2026-11-22:

| route | 77-case CLI parity |
|---|---|
| the adapter, today | **77/77** |
| Rust CLI only, binary or `_swadesh.run_cli` | 38/77 |
| Rust, else the adapter on nonzero exit | 73/77 |
| Rust, else the adapter for help/version too | 77/77 |

The Rust CLI is also not correct inside the flags it claims: `tests/test_cli.py` (448 cases) gives
**392 passed, 56 failed**, 50 of them TER. The 39 failures at the 77-case gate are help/version text,
`-ci`/`-par`/`-pbs`, test sets, `--detail`, `--citation`, and argparse error wording. Bringing it to
parity is 10.5–17.5 days, dominated by the test-set registry (3–7 d) and TER/chrF/BLEU correctness
(2–4 d). The hybrid route buys no wall-clock time in the Python package, because `import swadesh`
already dominates process startup, and a wheel cannot carry the binary anyway (`maturin` builds only
the extension module).

**Decision: keep `cli.py` as the documented adapter.** The Rust CLI stays the standalone binary. The
adapter's behaviour is the contract and is 77/77 by construction.

### `verify.py` — permanent

It compares swadesh against sacreBLEU **on the user's own data**. That comparison is the tool's
entire purpose, so it will never be standalone and is listed under `PERMANENT`.

## What this means for licensing

The Apache-2.0 notice covers `metrics.py` and `cli.py` today, plus `__init__.py` and `verify.py` for
good. It will cover `metrics.py` and `cli.py` until those two slices land. `significance.py` is
clean-room apart from its adapter seam.