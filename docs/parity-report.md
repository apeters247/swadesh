# swadesh parity report (criterion 5: API and CLI parity with sacreBLEU 2.6.0)

Track: `parity`, branch `test/parity`. Reference: sacreBLEU 2.6.0 (`/var/www/llmeval/.venv`
sources, same version in the track venv). Python 3.11.14, numpy 2.4.6, rustc 1.100.0, Linux x86-64 (VPS).

Tests: `tests/test_parity.py`. Every test runs the same call on sacreBLEU and on swadesh and asserts
the same outcome: the same value (scores, every formatted string, every attribute, signatures) or the
same exception type and message. Stdout and exit codes are compared for the CLI.

## Surface checked

1. **Constructors** `BLEU`, `CHRF`, `TER`: every argument, positional and keyword, including the
   invalid values (unknown `smooth_method`, `tokenize`, `max_ngram_order` 0/-1/2.0/"3"/None,
   `char_order` 0/-1/2.0, `word_order` -2, `beta` 0/2.5/"2", unknown keyword). Constructor
   signatures and annotations, class constants, `trg_lang` tokenizer selection and its warnings,
   instance attribute names.
2. **`corpus_score`**: tuples, generators, numpy arrays (of str and object), `str` in place of a list,
   empty hypotheses, empty references, empty stream `[[]]`, mismatched lengths (hypotheses longer or
   shorter than references, streams of different length), `None` references in the first and later
   streams, all-`None` segments, `None` hypotheses, int and bytes items. `references=None` with and
   without a cache, `references=[]`/`[[]]`/`[[None, None]]` as a cache (truthiness), cached then
   explicit references, `num_refs` in the signature, `n_bootstrap` 0/1/2/30/40.
3. **`sentence_score`**: `None` in any position, all `None`, empty refs, int/list hypothesis, `str`/
   tuple/generator/numpy refs, empty hypothesis or reference, cache plus `None` refs. The
   effective-order warning is logged exactly once per call (checked against the captured log).
4. **`Score`**: `score`, `counts`, `totals`, `precisions`, `bp`, `sys_len`, `ref_len`, `char_order`,
   `word_order`, `beta`, `num_edits`, `ref_length`, `name`, `_mean`, `_ci`, `_verbose`;
   `format()` with `width` (0, 4), `score_only`, `signature`, `is_json` (with and without signature);
   `str()`, `repr()`. `Signature`: `str`, `format(short=...)`, `update`, `info`.
5. **Compat functions**: `corpus_bleu`, `raw_corpus_bleu`, `sentence_bleu`, `corpus_chrf`,
   `sentence_chrf`, `corpus_ter`, `sentence_ter` with keyword and positional arguments, invalid
   tokenizer, empty input, `None` references; signatures; public names.
6. **`significance`**: `PairedTest` (`ar` and `bs`, 1, 2 and 3 systems, `n_jobs=1` and `n_jobs=2`,
   cached references, `n_ar_confidence`, `n_samples=0` defaults, bad `test_type`, mismatched system
   length), `estimate_ci`, `Result`, `_bootstrap_resample` (valid widths and malformed widths).
7. **Pickling and copying**: pickle protocols 0 to 5 with and without a reference cache, `copy.copy`,
   `copy.deepcopy`, pickled `Score` and `Signature`, bootstrapped scores; a `fork` multiprocessing pool
   scoring with swadesh.
8. **`patch()` / `unpatch()`**: every rebound name (`sacrebleu.BLEU/CHRF/TER`,
   `sacrebleu.metrics.*`, `sacrebleu.metrics.METRICS`, `sacrebleu.compat`, `significance` functions),
   restoration on `unpatch`, idempotent `patch` and `unpatch`, `unpatch` without `patch`,
   `isinstance` in both directions, HF-`evaluate`-style calls (`sacrebleu.corpus_bleu`,
   `sacrebleu.corpus_chrf`, `sacrebleu.sentence_bleu`, `sacrebleu.significance.PairedTest`).
9. **CLI** (`python -m swadesh.cli` vs `sacrebleu`): every flag listed in `sacrebleu --help`
   (`--citation/--cite`, `--list`, `-t`, `-l`, `-ol/--origlang`, `--subset`, `--echo` with
   `src`, `ref`, `docid`, `ref src`, `all`, `-i`, `-nr`, `-e`, `-m`, `-sl`, `-s`, `-sv`, `-tok`
   (including `ja-mecab`), `-lc`, `--force`, `-cc`, `-cw`, `--chrf-*`, `--ter-*`, `-ci`, `-cin`,
   `-par`, `-pbs`, `-parn`, `-pbsn`, `-j`, `-q`, `-sh`, `-b`, `-w`, `-d`, `-nc`, `-f json|text|latex`,
   `--short`, `-h`, `-V`), plus invalid values, mutually exclusive flags, a missing file, invalid
   UTF-8, stdin input, multiple reference files, and `-t wmt16 -l en-de` from the cached dataset
   (offline). 77 CLI cases, all green.

Not covered: `--download` (needs network; it fetches a set and quits). `spm`, `flores101/200`,
`spBLEU-1K` tokenizers (they download SentencePiece models). Tokenizers that need extra packages
(`ja-mecab` and `ko-mecab`) are compared as they behave in this environment.

## Results

| Suite | Result (final run) |
|---|---|
| `tests/test_parity.py` + `tests/test_mcp.py` (repaired 2026-11-21) | 497 passed, 0 xfailed (the Rust defects below are fixed or guarded; see "Rust kernels") |
| `tests/test_mcp.py` | 3 passed (needs the `mcp` extra installed; before that it skipped silently) |
| `tests/test_cli.py`, `test_coverage_gaps.py`, `test_fuzz_unicode.py`, `test_ter.py`, `test_regressions.py` (re-run 2026-11-21) | 574 passed |
| of which CLI (`-k cli`) | 77 passed |
| `tests/test_differential.py` | 57 passed, 1 skipped |
| `tests/test_fuzz_unicode.py` | 4 passed |
| `tests/test_ter.py` | 53 passed |

Before the fixes, the API tests had 61 failures and the CLI tests had 9 (the first run of this track).
The 1 skip in `test_differential.py` is an existing skip in that file, not one added by this track.

## Bugs found and fixed (Python layer)

All fixes are in `python/swadesh/metrics.py`, `python/swadesh/significance.py` and
`python/swadesh/cli.py`. Each one is covered by a test in `tests/test_parity.py` that failed before.

1. **Explicit constructor signatures.** `BLEU`, `CHRF` and `TER` took `*args, references=None, **kwargs`,
   so sacreBLEU's 9th positional argument (`references` in `BLEU(...)`) raised `TypeError`. The
   constructors now spell out sacreBLEU's parameters, so names, order, defaults and unknown-keyword
   errors match.
2. **`references=[]` as a cache.** `_RefCache` was always truthy. sacreBLEU's empty cache is falsy, so
   `corpus_score(h, None)` must raise `RuntimeError`. The truth value now matches.
3. **Pickling with protocols 0 and 1.** `_RefCache` used `__slots__` without `__getstate__`. The
   class no longer uses slots.
4. **Inputs the native kernels do not cover now go through sacreBLEU's own loop.** The Rust path is
   used only for valid input: integer parameters within range (`max_ngram_order >= 1`, `char_order >= 1`,
   `word_order >= 0`), str items, and segments that have at least one reference. Everything else runs
   sacreBLEU's own `_compute_segment_statistics` loop with its own reference cache, so the result
   or the exception is sacreBLEU's, including the `AttributeError`/`TypeError` messages for
   non-str items, `IndexError` for `char_order=0`, and `TypeError` for segments with all-`None`
   references. A native `TypeError`/`AttributeError`/`OverflowError` also falls back. This adds one
   check per call; the timing on wmt16 ref (3000 lines, best of 15) is unchanged within noise.
5. **Single warning per call.** `BLEU.sentence_score` without a native tokenizer warned twice.
6. **Bootstrap with mismatched system lengths.** `_resampled_sums` passed `len(stats)` as the kernel's
   `n` (indices per resample), so `PairedTest(bs)` with systems of different length raised a
   reshape `ValueError` instead of sacreBLEU's result or error. It now passes `idxs.shape[1]`, and
   out-of-range, negative or empty indices go to numpy, which is sacreBLEU's own expression.
7. **Mismatched shapes in the AR shuffle.** `_ar_shuffle` now returns `None` on mismatched shapes, so
   numpy raises sacreBLEU's `ValueError: matmul ...`. Before, the Rust kernel panicked, and inside a
   worker that deadlocked the pool.
8. **Malformed bootstrap rows.** `_native_bleu_scores` returns `None` unless the row width is
   `2 + 2*max_ngram_order`, so sacreBLEU's scorer decides (the Rust scorer panics on such rows).
9. **CLI program name.** See intentional differences.

## Rust kernels (status: fixed, the six tests below now pass as plain tests)

This section used to list six open Rust defects pinned by strict `xfail` tests. All six tests now
pass without `xfail`; the kernels validate their inputs and the fork hazard is handled by a
process-pid thread pool (`crates/swadesh-core/src/par.rs`), which builds a fresh rayon pool after
a `fork` instead of reusing the inherited one.

1. **Fork deadlock.** `test_forked_child_after_parent_parallel_kernel_does_not_hang`,
   `test_paired_test_parallel_workers_after_native_use_do_not_hang`, `test_fork_pool_corpus_scores`.
   `PairedTest(n_jobs=2)` with a warm parent is confirmed identical to sacreBLEU on real model
   output as well (see `~/eval-compare/TEST-STATUS-2026-11-21.md`).
2. `ar_sums_i64` on mismatched shapes raises `ValueError`:
   `test_mismatched_ar_kernel_raises_valueerror_not_panic`.
3. `bleu_scores_np2_f32` on narrow rows raises `ValueError`:
   `test_bleu_np2_kernel_malformed_rows_raise_valueerror_not_panic`.
4. `bootstrap_sums_f32` on an out-of-range index raises:
   `test_bootstrap_kernel_out_of_range_index_raises_not_panics`.
5. `bootstrap_sums_f32_seeded` on a width mismatch raises:
   `test_seeded_bootstrap_kernel_width_mismatch_raises_not_panics`.

## Intentional differences (documented, not asserted equal)

1. **Class identity and repr.** `swadesh.BLEU` is a subclass of sacreBLEU's class, so
   `isinstance` works both ways, but `type(x).__module__` is `swadesh.metrics` and `repr` shows
   `<swadesh.metrics.BLEU object ...>`. After `patch()`, `sacrebleu.BLEU is swadesh.BLEU`.
   Objects created before `patch()` are plain sacreBLEU objects (and are not `swadesh` instances).
2. **Extra API.** `sentence_scores`, `pairwise`, `mbr`, `swadesh.tokenize`, `swadesh.verify` and the
   MCP server are swadesh-only additions. They are not compared against sacreBLEU.
3. **Annotations on `corpus_score` and `sentence_score`.** The parameter names, kinds and defaults
   match. The annotations on those two methods are absent in swadesh (`hypotheses` vs `hypotheses:
   Sequence[str]`). The constructor annotations match exactly, and the test checks them.
4. **CLI program name.** `swadesh` replaces `sacrebleu` in the usage line, in argparse errors and
   in `--version` (`swadesh 2.6.0`). argparse wraps usage text at the name's length, so the
   usage text is compared with whitespace collapsed. Scores, tables, echoed data and exit codes are
   compared byte for byte.
5. **Uncaught exceptions.** The traceback frames differ (`cli.py` / `swadesh` entry point vs
   `sacrebleu`). The exit code and the final exception line are compared, and they match.
6. **Private attributes.** `_ref_cache` holds a `swadesh.metrics._RefCache` rather than sacreBLEU's list.
   The attribute names are the same. Private helpers (`_cache_references`, `_extract_corpus_statistics`) return
   native types on the fast path (numpy arrays) and sacreBLEU's own types on the fallback path.
   The public results are identical.

## Known remaining differences

None observed in the public API or CLI surface covered above. The fork deadlock that used to be
listed here is fixed (see "Rust kernels").
