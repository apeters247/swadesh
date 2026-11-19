# swadesh testing criteria (v2, 2026-11-19)

Publishing is out of scope for this phase. The goal is to make swadesh as correct and robust as
possible, measured against the criteria below. Every criterion has an owner track, and
every failure found becomes a regression test before it is fixed.

| # | Criterion | Target | Track |
|---|---|---|---|
| 1 | Differential exactness (BLEU/chrF/chrF++/TER, every option) | 0 mismatches over ≥ 1M fuzzed cases per metric, plus all WMT/IWSLT data in the sacreBLEU index | `diff` |
| 2 | Mutation score of `swadesh-core` (cargo-mutants) | ≥ 90% of viable mutants caught; every surviving mutant is explained or killed by a new test | `mutants` |
| 3 | Line coverage | ≥ 95% for `swadesh-core` (llvm-cov) and `python/swadesh` (coverage.py); uncovered lines listed and justified | `coverage` |
| 4 | Crash safety | 0 panics, hangs or sanitizer reports from cargo-fuzz (tokenizers, stats, TER, CLI argv/inputs), ≥ 30 min per target | `fuzz` |
| 5 | API and CLI parity | every public sacreBLEU 2.6.0 metric method, compat function, `Score`/`Signature` attribute and CLI flag behaves identically, including exception types and messages, or the difference is documented as intentional | `parity` |
| 6 | Robustness and performance | no super-linear blow-ups (very long lines, 10M segments, empty and huge corpora, many refs); peak RSS ≤ sacreBLEU's; no performance regressions vs `bench/results/final4-*` | `stress` |
| 7 | Platforms | the full suite passes on Linux x86-64 (VPS, troy-1), Linux ARM64 (GX10) and macOS ARM64 (Mac mini) | lead |

## Rules for all tracks
- Exact equality only. Never loosen, skip or xfail a test to get green; a real bug gets an
  `xfail(strict=True)` regression test plus a report, and the lead fixes it (or the track fixes it
  with a clear commit).
- Work in a git worktree `/var/www/llmeval-wt/<track>` on branch `test/<track>`; the lead merges.
- Shared production VPS: at most 3 parallel jobs (`CARGO_BUILD_JOBS=2`); disk is ~94% full, so keep
  temp files small and `cargo clean` when done. Don't touch GPUs on other machines.

## Status (2026-11-19, end of the testing phase)

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Differential exactness | **Met.** 1,000,000 fuzzed cases each for BLEU (×5 tokenizers), chrF, chrF++ and TER; 9.25M sentence comparisons on WMT20–24/IWSLT data in 22 languages; 0 score mismatches. One state difference (`num_refs` after batch `sentence_scores`) was found and fixed. Long-sentence TER (20–200 words) got 1,800 cases, because sacreBLEU itself needs ~1 s per case. | `docs/diff-report.md`, `docs/torture-report.md` |
| 2 | Mutation score | **Met on the measured set.** 94.6% of 1,229 viable mutants caught, with Python differential tests in the loop; 99.8% excluding 64 documented equivalent mutants. A later Rust-only run exposed gaps in the float32 bootstrap scorers added today; those are now covered by 8,000 bit-exact sacreBLEU fixtures. | `docs/mutation-report.md`, `crates/swadesh-core/tests/np2_scoring.rs` |
| 3 | Line coverage | **Met.** swadesh-core 95.07%, python/swadesh 99.7%; every remaining line is justified. | `docs/coverage-report.md` |
| 4 | Crash safety | **Met.** About 336M fuzz executions over 11 targets, 35 min each, under ASan. Two crash bugs found and fixed (a chrF `beta²` overflow and a CLI non-ASCII option panic). Native kernels now raise `ValueError` on malformed input instead of panicking. | `docs/fuzz-report.md`, `fuzz/` |
| 5 | API and CLI parity | **Met.** 494 parity tests, plus 448 byte-identical CLI tests. Fixed: a fork deadlock in multiprocessing (per-process thread pool), constructor positional arguments, pickling, and error messages. Intentional differences are documented. | `docs/parity-report.md`, `tests/test_parity.py` |
| 6 | Robustness and performance | **Met.** Linear scaling to 10M segments; sacreBLEU crashes at 10M (BLEU) and on 200 × 1,000-word TER sentences, swadesh doesn't. Bootstrap memory was fixed (1M × 1000: 717 MB vs sacreBLEU's 45.6 GB). Known limit: TER on 1,000-word sentences uses ~160 MB per sentence per thread. | `docs/stress-report.md`, `bench/results/compare-macmini-m4-v2.md` |
| 7 | Platforms | **Met.** Linux x86-64 (VPS, troy-1), Linux ARM64 (GX10) and macOS ARM64 (Mac mini), each on Python 3.10–3.14 (including free-threaded 3.14t) with numpy 1.26/2.x. | `bench/results/`, compat matrix in git history |
