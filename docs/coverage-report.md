# swadesh coverage report (testing criterion 3)

Target (TESTING.md, criterion 3): at least 95% line coverage for `crates/swadesh-core` (llvm-cov)
and `python/swadesh` (coverage.py), with every uncovered line listed and justified.

Branch `test/coverage`, based on master `861a11d`. Measured on the VPS (x86-64) with
Rust (stable, `llvm-tools`), CPython 3.14.3, sacrebleu 2.6.0, numpy 2.5.5, mcp (optional extra).

## Summary

| Scope | Before (master tests) | After (this branch) | Target |
|---|---|---|---|
| `crates/swadesh-core/src` (lines, union of all instrumented objects) | 92.09% (146 of 1846 missed) | **95.07%** (91 of 1846 missed) | >= 95% |
| `python/swadesh` (statements, coverage.py) | 74% (159 of 605 missed) | **99.7%** (2 of 605 missed) | >= 95% |

The Rust figure counts every line, including the dead code listed below. Excluding the 82 lines of
dead private tokenizer code, the figure is 99.5% (9 of 1764). Nothing was excluded from the
reported number.

## Method

Manual instrumentation only (no cargo-llvm-cov wrapper, no `show-env`, no `RUSTC_WRAPPER`).

* Rust objects: `swadesh_core` unit tests, `tests/nprng.rs`, `tests/coverage_gaps.rs`,
  `tests/resample_serial.rs`, the `swadesh` CLI binary, and the Python extension `_swadesh.abi3.so`
  (its swadesh-core code is part of the count).
* Python: `coverage run` over the test files, with the extension instrumented.
* Per-object line counts come from `llvm-cov export --format=lcov`. A line is covered if any
  object counts it (union, max count per line). `llvm-cov report` with several `--object`
  arguments loses counts for functions whose mapping differs between objects, so it was not used.

## Per-file results (swadesh-core/src)

| File | Before (covered/lines) | After (covered/lines) |
|---|---|---|
| lib.rs | 32/33 | 33/33 |
| ngram.rs | 72/81 | 81/81 |
| nprng.rs | 273/281 | 279/281 |
| pairwise.rs | 70/71 | 71/71 |
| par.rs | 21/21 | 21/21 |
| resample.rs | 68/78 | 78/78 |
| score.rs | 220/230 | 230/230 |
| stats.rs | 149/152 | 152/152 |
| table.rs | 80/87 | 87/87 |
| ter.rs | 437/445 | 438/445 |
| tokenize.rs | 227/316 | 234/316 |
| unicode.rs | 51/51 | 51/51 |
| **total** | **1700/1846 (92.09%)** | **1755/1846 (95.07%)** |

## Per-file results (python/swadesh, coverage.py)

| File | Stmts | Before missed (cover) | After missed (cover) |
|---|---|---|---|
| `__init__.py` | 66 | 3 (95%) | 0 (100%) |
| `cli.py` | 10 | 10 (0%) | 0 (100%) |
| `mcp_server.py` | 61 | 61 (0%) | 0 (100%) |
| `metrics.py` | 228 | 17 (93%) | 2 (99%) |
| `rewards.py` | 29 | 4 (86%) | 0 (100%) |
| `significance.py` | 154 | 7 (95%) | 0 (100%) |
| `verify.py` | 57 | 57 (0%) | 0 (100%) |
| **total** | 605 | 159 (74%) | 2 (99.7%) |

## Uncovered lines after this branch, with reasons

### Python (2 lines)

* `metrics.py:103`, `_SacreMetric._native_ok` returns `True`. BLEU and CHRF override it, and no
  call path reaches the base method (TER has its own `_extract_corpus_statistics`). Dead code.
* `metrics.py:132`, `_py_statistics` raises when there are no references and the cache is empty.
  Every caller goes through `_resolve_refs`, which raises the same error first. Unreachable.

### swadesh-core (91 lines, all listed)

Dead code (no caller in the workspace; justified, recommended for removal by the lead):

* `tokenize.rs` lines 54-72 (`sub2`), 76-87 (`sub1`), 91-107 (`join_split_chars`), 110-113
  (`is_13a_class1`), 116-118 (`is_ascii_digit`), 220-235 (`sub2b`), 238-264 (`tok_re_chars`).
  They are private and only referenced by `tok_re_chars`, which has no callers. 82 lines.
  `preprocess_bleu` and `tokens` are `pub` and are now tested.

Unreachable by construction (justified by reading the code, not by a proof):

* `ter.rs:160-161`, the trailing `'s$` rule. sacreBLEU pads each sentence with spaces
  (`tokenizers/tokenizer_ter.py` line 37, `sent = f" {sent} "`), and the punctuation step keeps a
  trailing space, so `'s$` cannot match. The port mirrors the same sequence.
* `ter.rs:243`, `wrap_whole_string_in` formats a string only when every character lies in a CJK
  range. It is only called on the padded string, which always contains a space, so the condition
  is false.
* `ter.rs:452` and `ter.rs:581`, the `panic!` arms for unknown TER op codes. Op codes come from a
  fixed internal set.
* `nprng.rs:219`, `bounded_u64` with `rng == u64::MAX`. `Generator::integers` cannot produce that
  range (it asserts `low < high`). `next_below(0)` would, but no caller passes 0 (`resample`
  returns early for `n == 0`). Note: `next_below(0)` returns a full-range value instead of failing;
  this is a latent API hazard, not a reachable path.
* `nprng.rs:388`, `round_ratio` returns `f64::INFINITY` when the rounded exact value is at least
  2^1024. The mean of finite doubles is at most `f64::MAX`, so finite inputs cannot reach it.

Open (not proven unreachable, not reached by any input tried):

* `ter.rs:463`, the cache-hit arm (`Some(&c) => c`) in `_add_cache`. Not reached by 40 random
  pairs over a 5-word vocabulary, 60 unbalanced-length pairs, or the hand-written multi-reference
  and shift cases.
* `ter.rs:630`, the `break` when the shift offset runs past the alignment. Not reached by the same
  inputs. It may be guarded by an invariant (alignment length equals reference length); that was
  not verified.

## Tests added

### `tests/test_coverage_gaps.py` (Python, differential against sacrebleu 2.6.0)

* Patch and unpatch behaviour, paired tests while patched, bootstrap with overridden score
  arithmetic (`_BLEUOverride`, `_ChrFOverride`).
* Python-tokenizer path (`_py_statistics`, `_prepare`) for 13a and intl, including references that
  contain `None` and the tokenized-period warning.
* Bootstrap resampling for native kernels in many configurations (BLEU orders 1-6, chrF char orders
  1-7, chrF++ with word order 2, which covers resample widths 3, 4, 6, 9, 12 and wide 21-bit
  n-grams), plus `CHRF(whitespace=True).pairwise`, and TER edge cases (possessive at the end,
  very short hypothesis against long reference, all-CJK sentence, repeated prefixes).
* Paired significance with the numpy-2 kernel disabled, with chrF arithmetic overridden, and with
  zero-order chrF; out-of-range fancy indices (`_resampled_sums`); the native CHRF kernel declining
  malformed rows.
* CLI (`swadesh.cli`, `verify`) output compared with sacreBLEU's CLI using `capfd` (sacreBLEU's
  `main()` reopens fd 0/1, so `capsys` does not work). Fixture: a fresh stdin per invocation.
* MCP: argument validators called directly, `mbr_select`, `sentence_scores` with unknown metric.
* `test_mcp_errors_for_bad_arguments`: `xfail(strict=True)`, see the bugs section.
* `test_chrf_bootstrap_matches_reference`: asserts the same `TypeError` as sacreBLEU (see below).

### `crates/swadesh-core/tests/coverage_gaps.rs` (27 tests)

* n-gram representation (`Wide` runs, `total`, `is_empty`, `max_merge` over packed and wide parts).
* Score arithmetic: `py_sum` naive mode and integer items, `SumMode::for_python`, `PyNum::as_f64`,
  `Smooth::from_name`, BLEU with only a zero precision (score 0.0).
* RNG: `mean_f64` on an empty slice, `bit_generator`, a large-range `integers` draw that exercises
  the Lemire rejection loop, with numpy 2.5.5 fixtures for seed 12345 (ranges 2^63-1 and 2^62+1).
* `bootstrap_sums_f32_seeded` with zero draws or zero samples; `Interner::clear`; `CountTable` default
  and generation wraparound (4.3e9 resets, about 20 s in release); `corpus_stats` with a wrong-width
  row; `Tokenizer::from_name`; `preprocess_bleu`; `tokens`.
* TER: normalized possessive, normalized CJK with asian support (sacreBLEU: 1 edit, length 5.0),
  shifted blocks, multiple references, a short hypothesis against a 60-word reference, repeated
  prefixes, and 100 generated pairs (random seeds 20261119 and 7331). All expectations come from
  sacreBLEU 2.6.0 `TER().sentence_score`.

### `crates/swadesh-core/tests/resample_serial.rs` (1 test)

Runs in its own process with `RAYON_NUM_THREADS=1`, so the single-threaded bootstrap path is taken.
It checks the output against an explicit index stream.

## Bugs and findings

1. **Real defect, not a sacreBLEU mismatch: MCP bad arguments.** `mcp_server.py` raises plain
   `ValueError` for bad input (unknown metric, empty hypotheses, mismatched references). The MCP
   framework turns that into `Error executing tool X` and hides the message, so agents never see
   the cause. Repro:
   `python -c "import asyncio; from swadesh.mcp_server import server; asyncio.run(server.call_tool('sentence_scores', {'hypotheses': ['a'], 'references': [['a']], 'metric': 'nope'}))"`
   raises `UnexpectedToolError: Error executing tool sentence_scores`, where a direct call to
   `sentence_scores(...)` raises `ValueError: unknown metric 'nope'`. Covered by a strict xfail,
   `tests/test_coverage_gaps.py::test_mcp_errors_for_bad_arguments`. Fix: raise `mcp`'s `ToolError`
   for input validation.

2. **Master test failures in `tests/test_parity.py` (not fixed here).** Nine CLI parity cases fail
   on master: `test_cli_informational_flags[--help|-h|--version|-V]` and
   `test_cli_scoring_flags[-s bogus|-tok bogus|-m nope|-w x|-par -pbs]`. Cause: `_norm` masks
   `sacrebleu`/`swadesh` followed by `:` or space, but the Python entry point prints
   `python -m swadesh.cli`, which is not masked. Scores are identical; only the program name
   differs, and `docs/parity-report.md` already documents that difference as intentional. This is
   the parity track's file, so it is reported and not changed.

3. **Upstream incompatibility, parity-equal.** sacreBLEU 2.6.0 raises `TypeError: don't know how to
   coerce float32 and float` in `estimate_ci` for `CHRF().corpus_score(H, R, n_bootstrap=100)` with
   `H` containing an empty hypothesis, on CPython 3.14.3. swadesh raises the identical exception
   through the same code path. The draft test expected success and was corrected to assert the same
   exception. The 4-segment input was tested; the minimal trigger was not isolated.

4. No new score mismatches were found in any path added here (all new differential tests pass).

## Commands

Environment: `export PATH="$HOME/.cargo/bin:$PATH"`. Every build and test ran under
`systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`, with `CARGO_BUILD_JOBS=2`.

```sh
# one-time
rustup component add llvm-tools
uv venv .venv
uv pip install -p .venv/bin/python sacrebleu==2.6.0 maturin numpy pytest hypothesis coverage "mcp>=2"

# Rust (instrumented)
mkdir -p cov
RUSTFLAGS="-C instrument-coverage" LLVM_PROFILE_FILE="$PWD/cov/rust-%p-%m.profraw" \
  CARGO_TARGET_DIR=target-cov CARGO_BUILD_JOBS=2 cargo test -p swadesh-core --release
RUSTFLAGS="-C instrument-coverage" LLVM_PROFILE_FILE="$PWD/cov/rust-%p-%m.profraw" \
  CARGO_TARGET_DIR=target-cov CARGO_BUILD_JOBS=2 cargo test -p swadesh-core --release \
  --test coverage_gaps --test resample_serial

# CLI (instrumented)
RUSTFLAGS="-C instrument-coverage" LLVM_PROFILE_FILE="$PWD/cov/rustcli-%p-%m.profraw" \
  CARGO_TARGET_DIR=target-cov CARGO_BUILD_JOBS=2 cargo build --release -p swadesh-cli
SWADESH_BIN=$PWD/target-cov/release/swadesh LLVM_PROFILE_FILE="$PWD/cov/cli-%p-%m.profraw" \
  .venv/bin/python -m pytest tests/test_cli.py -q

# Python extension (instrumented, no wrapper)
RUSTFLAGS="-C instrument-coverage" CARGO_TARGET_DIR=target-cov CARGO_BUILD_JOBS=2 \
  VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release

# Python coverage
LLVM_PROFILE_FILE="$PWD/cov/py-%p-%m.profraw" .venv/bin/python -m coverage run \
  --source=python/swadesh -m pytest tests/test_differential.py tests/test_ter.py tests/test_parity.py
LLVM_PROFILE_FILE="$PWD/cov/py-%p-%m.profraw" SWADESH_BIN=$PWD/target-cov/release/swadesh \
  .venv/bin/python -m coverage run -a --source=python/swadesh \
  -m pytest tests/test_coverage_gaps.py tests/test_mcp.py
.venv/bin/python -m coverage report -m

# Merge and per-object line counts (union)
BIN=$(echo ~/.rustup/toolchains/*/lib/rustlib/*/bin)
$BIN/llvm-profdata merge -sparse cov/*.profraw -o cov/all.profdata
for obj in <test binaries> python/swadesh/_swadesh.abi3.so; do
  $BIN/llvm-cov export --format=lcov --instr-profile=cov/all.profdata "$obj"
done   # take the max DA count per line per SF file, count the lines with count 0 as missed
```

The union script is about 30 lines: it parses `SF:` and `DA:` records per object, keeps the maximum
count per line for each `crates/swadesh-core/src/*.rs` file, and reports the lines still at zero.
The before/after numbers were produced with the same script, using `cov/before.profdata` (the
master suite plus CLI runs) and `cov/all.profdata`.

Cleanup: `cargo clean` (in `CARGO_TARGET_DIR=target-cov` and in the worktree), delete `cov/*.profraw`.
