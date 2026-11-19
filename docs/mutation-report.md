# Mutation report: swadesh-core (criterion 2)

**Status: 94.6% caught on the captured mutant set (target 90%), 99.8% with equivalent mutants excluded.
The full-crate score is not yet measured: 198 of the 1,477 mutants have no outcome (see "Not yet measured").**

## Scores

Baseline is the run in `docs/mutants-run1/` (troy-1, interrupted). Viable = caught + timeout + missed.

| Measure | Before | After |
|---|---|---|
| caught (incl. timeouts, which count as caught) | 1,056 | 1,163 |
| missed | 173 | 66 (64 equivalent, 2 unresolved) |
| unviable (excluded) | 50 | 50 |
| not run | 198 | 198 |
| **Score on captured viable set** (equivalents counted as misses) | 85.9% | **94.6%** (1,163 / 1,229) |
| Score, equivalent mutants excluded from the denominator | - | **99.8%** (1,163 / 1,165) |

Full crate (1,477 mutants): the 198 unrun mutants bound the score between
1,163 / 1,427 = 81.5% (all survive) and 1,361 / 1,427 = 95.4% (all caught).
Reaching the criterion on the full crate requires running those 198 (see "Not yet measured").

## How it was computed

1. The 173 survivors in `docs/mutants-run1/missed.txt` were grouped by file and classified: killed by a new test,
   equivalent (with a reason), or unresolved.
2. Each file's survivors were re-run with cargo-mutants, one file at a time, `-j 1`, under
   `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`, with `CARGO_PROFILE_RELEASE_LTO=off`
   and `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16` (same as the wrapper). The `-f` flag did not filter in this
   cargo-mutants version (27.1.0), so `-F` regexes on the mutant names were used instead.
3. **Verification scope:** kills were checked with `cargo test --release -p swadesh-core` only. The old wrapper also ran
   `tests/test_differential.py` and `tests/test_ter.py`, so mutants it caught before stay caught. The new kills are a lower
   bound, because the Python subset would only add kills. `tests/test_mutation_kills.py` is not part of the wrapper.
4. `resample.rs`, `ter.rs` and `nprng.rs` changed line numbers (or gained code) since the old run. Survivors there were
   re-located by description, and the whole file's mutants were re-checked where needed.

## Tests added

All oracle values come from sacreBLEU 2.6.0, numpy 2.5.5, CPython 3.12 and 3.14 (`tests/gen_mutation_fixtures.py`
writes `crates/swadesh-core/tests/fixtures/mut_*.txt`; floats are stored as IEEE bit patterns).

- **ter.rs:** sacreBLEU TER on 1,504 generated cases (shifts with distance > 50, repeated words, permutations, length
  ratios above 50, edge placements, multiple and empty references), in `sacrebleu_tests`.
- **score.rs:** `SumMode::for_python` table; `PyNum::as_f64`; `py_sum` against CPython 3.12 (Neumaier), 3.14 (NeumaierInts)
  and a naive reduction, 3,008 lists including overflow cases; `compute_bleu` against sacreBLEU
  `_compute_score_from_stats` on 1,500 stat rows (all smoothing modes, `effective_order`, `bp` edge cases, inconsistent
  zero-total rows); `chrf_f_score` on 1,500 rows (zero hyp/ref counts, eps smoothing).
- **stats.rs:** `Interner::clear`; `closest_ref_len` (no references, ties); `remove_punctuation` against sacreBLEU;
  `chrf_segment` against sacreBLEU segment statistics, including 60 cases where two references tie on F-score (first wins).
- **pairwise.rs:** every `chrf_matrix` cell bit-equal to `CHRF.sentence_score` on 400 cases (short references included).
- **ngram.rs:** repeated-n-gram merging, wide-key counting, clipped matches against a naive min-sum, `max_merge` (packed
  and wide, duplicate keys), `total`, `is_empty`.
- **table.rs:** `fits` bit budget; `packed_windows` against naive keys; `CountTable` against a HashMap model with
  resets and growth (add, set_max, take, get); generation wraparound (stale generation-0 entries must not leak).
- **resample.rs:** every specialized width (2..24) and generic widths against a naive sequential f32 sum; empty draws;
  seeded sums against `Generator::choice` indices, including a batched case (17 x 2^20 indices);
  paired-sum selection against a naive mask sum.
- **nprng.rs:** Lemire 64-bit rejection (`integers(-2**62, 2**62+1)`); 32-bit Lemire at the exact threshold and below it,
  using crafted PCG64 states checked against numpy; `next_below` against numpy; `statistics_mean` against CPython on
  wide-exponent values, similar-magnitude values, exact halfway ties (ties-to-even and sticky), and values near `f64::MAX`.
- **tests/test_mutation_kills.py** (14 tests, pass against the built `swadesh` extension): TER with shifts and extreme
  ratios, BLEU orders 5 and 6 (multi-reference wide keys), chrF `char_order=7`/`word_order=5`, smoothing edge cases,
  BP when both lengths are zero, chrF first-best ties.

Rust library unit tests: 6 before, 38 now (+32); integration tests unchanged at 11. All pass.
No mismatch with sacreBLEU was found in any fixture, so no real behavior difference was reported.

## Survivors: killed, equivalent, unresolved

Killed: **107** (ter 43, score 17, nprng 24, resample 12, ngram 5, stats 3, table 2, pairwise 1).
Timeouts in the verification runs (infinite loops in the mutant) count as caught.

### Equivalent mutants (64)

Each mutant either leaves the output unchanged on every input, or changes only performance, or is in code that is
unreachable. Release-mode semantics are assumed where noted (tests run with `--release`).

- **lib.rs:41:10** `>=`->`<` in `corpus_stats`: chooses serial vs parallel; both produce identical output.
- **pairwise.rs:30:23** `>`->`>=` in `chrf_profile`: the word profile is built with an empty range when `word_order == 0`; no output.
- **stats.rs:47:45** `<`->`<=` in `closest_ref_len`: when `r == closest_len` the assignment writes the same value.
- **stats.rs:62:32** `<`->`<=` in `bleu_segment`: pushes one unused empty string.
- **stats.rs:179:29** `||`->`&&` in `order_stats`: when either side has no n-grams the match count is 0 in both paths.
- **stats.rs:207:70** `>`->`>=` in `chrf_segment` (`word_order > 0`): the word ids are computed but no word loop runs; no output.
- **ngram.rs:58:33** `|`->`^` in `count`: the shifted key has zero low bits, so `|` and `^` coincide.
- **ngram.rs:77:19, 78:19** `i += 1` / `j += 1` -> `*=` in `clipped_matches_t`: keys are strictly increasing, so a stalled
  pointer is advanced by the next comparison; same counts and termination.
- **table.rs:27:5** `mix` -> `0` / `1`, **28:19** `>>`->`<<`, **29:9** `^`->`|` / `&`: only the hash changes; the table
  is correct for any hash (performance only).
- **table.rs:50:18** `>`->`>=`, **50:14** and **51:27** `*`->`+` in `reset`: the table is reallocated or sized to n+2,
  which is still above the distinct-key bound (performance only).
- **table.rs:71:20** `+`->`-` in `find`: probing reverses consistently in insert and lookup and visits all slots.
- **table.rs:106:21** `>`->`>=` in `set_max`: writes an equal value.
- **table.rs:133:26** `|`->`^` in `packed_windows`: the shifted key has zero low bits.
- **score.rs:156:25** `>`->`>=` in `compute_bleu`: inside `sys < ref`, `sys == 0` gives `exp(-inf) = 0` either way.
- **score.rs:194:37** `/`->`*` in `compute_bleu` (Int-sum branch): the mean is at most -1e10, so `exp` is 0.0 either way.
- **score.rs:231:17** `+`->`*` in `chrf_f_score`: when one average is zero the score is zero in both forms.
- **score.rs:260:25** and **301:37** (same two reasons) in `compute_bleu_np2_f32`.
- **nprng.rs:107:40** `|`->`^`: the two halves occupy disjoint bits.
- **nprng.rs:181:13** and **197:13** `<`->`<=` in the Lemire helpers: when `left == excl`, the threshold is below `excl`, so the rejection loop does not run.
- **nprng.rs:278:33** `%`->`+` in `integers_bool`: in release builds the shift amount is masked to 5 bits, so `buf >> (i + 32)` equals `buf >> (i % 32)`.
- **nprng.rs:369:12, 369:32** (four mutants) in `round_ratio`: the shift `k = max(56 + den_bits - a_bits, 1076)`; 1076 already leaves at least two bits beyond the target precision for every finite result, so changing the first term never changes the rounded value.
- **nprng.rs:386:23, 386:29** in `round_ratio`: the overflow test `top > 1023` never fires for finite inputs (results are <= f64::MAX), so these change only a dead branch.
- **nprng.rs:415:27** `/`->`*` in `statistics_mean` (non-finite branch): the total is +-inf or NaN, so `inf / n == inf * n`.
- **nprng.rs:427:19** `|`->`^`: bit 52 is clear in the fraction.
- **resample.rs:29..37** (nine mutants) delete a specialized arm of `accumulate`: the generic loop performs the same sequential f32 sums per column.
- **resample.rs:99:37** (three mutants, `>` -> `==`/`<`/`>=`) in `bootstrap_sums_f32_seeded`: the draw stream and the sums are the same whether or not the parallel batch path runs. (The batch-size mutants at 103:34 and 104:30 are new code, not in the old 173, and were also equivalent.)
- **ter.rs:160:28** (two mutants) `-` -> `+` / `/` in `normalize_general_and_western`: unreachable. After the 13a wrap the string always ends in a space, so `ends_with("'s")` is never true.
- **ter.rs:242:8** (`!` deleted) and **242:25** (`&&` -> `||`) in `wrap_whole_string_in`: the wrap only adds spaces around an all-kana string, and output is whitespace-tokenised.
- **ter.rs:389:35** `>`->`>=` in `BeamEdit::eval`: with `n_h == 0` no rows are computed.
- **ter.rs:459:28** (three mutants) and **469:41** (two mutants) in the prefix cache: the cache holds rows identical to recomputed ones (same reference, same `n_h`), so its size limit is unobservable.
- **ter.rs:489:10** `>`->`>=` in `py_slice`: an empty slice either way.
- **ter.rs:499:15** `<`->`<=` in `perform_shift`: with `target == start` both branches return the identity.
- **ter.rs:626:32** `&&`->`||` and **626:50** `<`->`<=` in `shift`: `pos` is always in `[-1, n_ref - 1]`, so the bounds check never fails.
- **ter.rs:649:33** `>`->`>=` in `shift`: equal keys imply the same start, length and index, hence the same shifted words.

### Unresolved (2)

- **ter.rs:655:34** and **ter.rs:656:30** (`delete -` on `neg_start_h` / `neg_idx` in `shift`): these change the tie-break
  between shifts with equal score and length. They were not killed by exhaustive search over all binary pairs with
  length <= 9 (1,045,506 pairs), by 200,000 tie-heavy random pairs, or by 40,000 mixed random pairs. The analysis
  suggests the mutant can only differ when two candidates from overlapping windows tie, but this is not proven.
  They stay listed as survivors until a constructed case is found or an argument is written.

## Equivalence notes and corrections

- `score.rs:88:31` in `py_sum` was first thought equivalent; it is killed, because the mutant loses the parentheses
  (`a || b && c`), so a Neumaier overflow NaN appears.
- `ngram.rs:28:9` (`AnyRuns::is_empty` -> `false`) was first thought equivalent; it is killed by a direct test of `is_empty`.

## Not yet measured

- 198 mutants have no outcome in the old run (`cargo mutants --list -p swadesh-core` gives 1,477; 1,229 were run).
  The full-crate score needs these run with the new tests, on a machine with memory headroom.
- The Python subset of the wrapper was not re-run for the new kills (see scope above).
- `tests/test_mutation_kills.py` is not in the wrapper's pytest list; adding it would require a slot venv per mutant.

## Command lines (as run)

```
# per file (Rust-only verification), inside the memory cap, -j 1
CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  cargo mutants -F 'src/ter\.rs:...' -j 1 --minimum-test-timeout 300 --no-shuffle --output <dir>
# fixtures
.venv/bin/python tests/gen_mutation_fixtures.py --ter-cases 1500
```

Verification scratch files (probe crates, mutant copies) were kept outside the repository and removed after use.
