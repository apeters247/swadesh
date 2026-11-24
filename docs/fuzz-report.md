# Fuzz report (criterion 4, `fuzz` track)

> **Status (2026-11-24): historical.** This report records the 2026-11-19 testing campaign. It has
> not been re-run since the 2026-11-22 clean-room rewrite of the tokenizers and the segment
> statistics, so the figures below describe the tree as it was then. `TESTING.md` carries the
> re-measured state and the list of known divergences; `docs/lab-notebook.md` records what was
> measured and when.

Machine: GX10 (aarch64, 20 cores, CPU only). Build: `cargo +nightly fuzz build`
(cargo-fuzz 0.13.2, libFuzzer via libfuzzer-sys 0.4.13, AddressSanitizer, release
profile with overflow checks). Harnesses live in `fuzz/fuzz_targets/`.

## Targets and campaign

Each target ran 2100 s (35 min) with `-timeout=10 -rss_limit_mb=4096`, queued through
`xargs -P 10` (at most 10 concurrent jobs). This concurrency and RSS limit were set before
a later GX10 memory rule (at most 4 jobs, 2048 MB RSS, stop launching below 12 GB available).
That rule arrived after the campaign had finished; no fuzz job was run after it.
The pre-fix crash inputs were replayed afterwards with `-rss_limit_mb=2048`.

| Target | What it exercises | Executions | Findings |
|---|---|---|---|
| `tok_13a` | `Tokenizer::T13a`: `apply` + `tokenize_unjoined` on lossy UTF-8 bytes | 16,352,644 | none |
| `tok_intl` | `Tokenizer::Intl` (same harness) | 16,751,904 | none |
| `tok_zh` | `Tokenizer::Zh` (same harness) | 29,600,802 | none |
| `tok_char` | `Tokenizer::Char` (same harness) | 86,201,155 | none |
| `tok_none` | `Tokenizer::None` (same harness) | 204,773,215 | none |
| `bleu_segment` | all 5 tokenizers, `max_order` 0..=8, 1..=4 refs; invariants `correct <= total`, lengths | 2,950,092 | none |
| `chrf_segment` | `char_order` 0..=10 (wide keys above 6), `word_order` 0..=6 (wide above 4), arbitrary `beta`, 1..=4 refs; invariants on `[hyp, ref, match]` | 2,794,960 | beta overflow (fixed) |
| `ter_segment` | all 4 flags, inputs capped at 256 bytes, 0..=4 refs; asserts time < 5 s and finite outputs | 388,464 | none; no libFuzzer timeout |
| `pairwise` | `chrf_matrix` and `bleu_matrix`, cells cross-checked against per-pair `chrf_segment`/`bleu_segment` | 2,494,344 | beta overflow (fixed, same site) |
| `resample` | `bootstrap_sums_f32` vs naive sequential f32 sum, `ar_sums_i64` vs naive, `bootstrap_sums_f32_seeded` determinism; widths 0..=25, indices in range | 1,697,929 | none |
| `cli_argv` | `swadesh_cli::run_argv` with an option vocabulary from the real option table; every path token and suffix is an in-memory file (no real filesystem access) | 1,763,353 | non-ASCII argument panic (fixed) |

Run time: 2101 s per target, 11 targets, all exit status 0 (no crash, no timeout,
no sanitizer report) in the final campaign. The TER target ran at about 185 exec/s
and never triggered the 10-second per-input timeout or the 5-second time assertion.

## Findings and fixes

1. **chrF beta overflow** (`crates/swadesh-core/src/score.rs`, `chrf_f_score`).
   `beta * beta` in i64 panics under overflow checks for |beta| > 3.0e9. Reached by
   `chrf_segment`, `chrf_matrix` and the CLI `--chrf-beta`. Fix: compute the square in
   i128 before the `as f64` cast. This is identical to the old result for every input
   that did not overflow. Regression: `crates/swadesh-core/tests/fuzz_regressions.rs`
   (`chrf_huge_beta_does_not_overflow`, `chrf_small_beta_unchanged`).
   Minimized input: none possible; libFuzzer reported a 14-byte arbitrary-encoded
   input that cannot shrink further. The failing value is any beta with |beta| > 3.0e9.

2. **Non-ASCII short option panic** (`crates/swadesh-cli/src/args.rs`, `find_opt`).
   `tok.split_at(2)` panics when byte 2 is inside a multi-byte character, so an
   argument such as `-é` or `-Ւ` aborted the process instead of failing with exit
   code 2. Fix: `tok.get(..2)` and `tok.get(2..)`. Regression:
   `crates/swadesh-cli/tests/fuzz_regressions.rs` (`non_ascii_short_option_is_an_error_not_a_panic`),
   with minimized inputs `-é` and `-Ւ`.

Both fixes were verified: the pre-fix artifacts exit 0 against the fixed binaries,
the regression tests pass, and the existing core and CLI test suites pass.

## Observed, not fixed (outside the crash scope)

- `find_opt` takes a 2-byte head for attached short values, so `-sv0.5` resolves to
  `-s` with the value `v0.5` instead of `-sv` with `0.5`. This is a CLI parity question
  (criterion 5), not a crash, and the change would alter outputs, so it is reported
  here for the `parity` track. Read from the code, not reproduced on GX10.

## Python differential tests

Run on GX10 against the fixed tree, after `pip install .` (swadesh 0.1.0, sacrebleu 2.6.0):
`pytest tests/test_differential.py tests/test_ter.py` gave **108 passed, 3 skipped, 0 failed**
(904 s) at the time. Re-measured on 2026-11-24 the same two files have failing cases; see
`TESTING.md`. The skipped tests were not individually examined in this run.
