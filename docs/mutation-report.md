# Mutation report: swadesh-core (criterion 2)

**Status: incomplete. The 90% target is not reached, and no survivor has been analyzed or killed yet.**
The run was interrupted by a memory exhaustion on troy-1 (shared with other tracks), and the coordinator
then stopped all work there. Results below are only what was captured before that.

## Scores

| Measure | Value |
|---|---|
| Mutants generated (`cargo mutants --list -p swadesh-core`) | 1477 |
| Outcomes captured | 1279 |
| caught | 1037 |
| timeout (counted as caught, per the track rules) | 19 |
| missed | 173 |
| unviable (excluded from the score) | 50 |
| not run (no outcome before interruption) | 198 |

- **Score on the captured set:** (1037 + 19) / (1037 + 19 + 173) = 1056 / 1229 = **85.9%**.
- **Bounds on the full crate:** the 198 untested mutants could be viable or unviable, so the full score is
  unknown. If all 198 were viable and caught, the best case is (1056 + 198) / (1229 + 198) = 1254 / 1427 = **87.9%**.
  Reaching 90% of 1427 needs at least 1285 caught, so at least about 31 of the 173 missed mutants must be
  killed (or justified as equivalent) even in the best case.
- **Before/after:** no "after" yet. The "before" figure is 85.9% on the captured set. No tests were added,
  so there is no improvement to report.

## Survivors and justifications

Not done. The full `missed.txt` (173 entries) lives on troy-1 at `~/swadesh-mutants/mutants.out/missed.txt`.
Each entry needs a kill test or a written equivalence argument. Survivors seen in the progress output
(examples only, not analyzed):

- `lib.rs:41` `>=` -> `<` in `corpus_stats`
- `ngram.rs:28` `AnyRuns::is_empty` -> `true` / `false`
- `ngram.rs:41`, `ngram.rs:105` match guard `*last == k` -> `false` in `rle` / `max_merge_t`
- `ngram.rs:58` `|` -> `^` in `count`
- `ngram.rs:77-78` `+=` -> `*=` in `clipped_matches_t`
- `ngram.rs:119` `!` deleted in `max_merge`
- `pairwise.rs:30`, `pairwise.rs:60` `>` -> `>=` in `chrf_profile` / `chrf_matrix`
- `score.rs:231` `+` -> `*` in `chrf_f_score`; `score.rs:260` `>` -> `>=` in `compute_bleu_np2_f32`
- `ter.rs:656` `-` deleted in `shift`; `ter.rs:598` `shift` replaced by constants (these were TIMEOUTs, counted caught)
- `ter.rs:692` `<=` -> `>` in `translation_edit_rate`
- `tokenize.rs:112` `is_13a_class1` -> `false` / `true`

No real bug (a behavior that differs from sacreBLEU) has been identified from the survivors. That is not a
finding of absence, because none has been analyzed.

## Tests added

None.

## Command lines (as run)

```
# On troy-1 (interrupted), with the cargo wrapper in this repo:
CARGO=$HOME/swadesh-mutants/tools/mutants/cargo-wrap.sh TMPDIR=$HOME/cm-tmp \
  cargo-mutants mutants -p swadesh-core -j 8 --minimum-test-timeout 300 --no-shuffle
```

`tools/mutants/cargo-wrap.sh` runs `cargo test` (Rust tests) and then, if those pass, `maturin develop --release`
into a free slot venv (`$MUTANTS_SLOTS/venv.N`, 8 slots, file-locked) and
`python -m pytest -q -x tests/test_differential.py tests/test_ter.py -k "not long"`. It sets
`CARGO_PROFILE_RELEASE_LTO=off` and `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16` to cut build time, which is
applied to baseline and mutants alike. The `.cargo/mutants.toml` in the repo sets `examine_globs` to
`crates/swadesh-core/src/**/*.rs` and the `-p swadesh-core --release` test arguments.

The unmutated baseline passed in 3 s build + 70 s test.

## Remaining work

1. Stop: do not run further mutation jobs on troy-1 (coordinator instruction).
2. Analyze the 173 missed mutants from `missed.txt` (troy-1): kill each with a Rust unit test in its module or a
   Python test in `tests/test_mutation_kills.py`, or document it as equivalent.
3. Run the 198 untested mutants on a lower-memory machine (Mac mini at `-j 2`, under 8 GB) and add their outcomes.
4. Re-score against 90%.
