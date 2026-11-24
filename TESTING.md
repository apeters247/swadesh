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

## Status (2026-11-24, re-measured)

The 2026-11-19 version of this table said all seven criteria were **Met**. That predates the
2026-11-22 clean-room rewrite of the tokenizers and the statistics, and re-measuring on troy-1 shows
several no longer hold. This table is the current state. Where a figure could not be re-measured it
says so instead of repeating the old one. The suites are not derandomized, so the failing counts move
a little between runs; an exact zero is what would be seed-independent.

| # | Criterion | 2026-11-24 | Evidence |
|---|---|---|---|
| 1 | Differential exactness | **Not met, but nearly** (see `## Known divergence` at the end of this file). The seven-file selection reports **18 failing cases**, all in the TER shift search: `test_ter.py` 16, `test_deep.py` 2. The other five files are green: `test_frozen_oracle.py` 17 passed, `test_parity.py` 494 passed, `test_differential.py` 79 passed, `test_fuzz_unicode.py` 4 passed, `test_coverage_gaps.py` 68 passed (selection total 705 passed, 2 skipped). | re-run 2026-11-24, Python 3.11.14, troy-1 |
| 2 | Mutation score | **Unverified.** `cargo-mutants` is not installed on troy-1; 94.6% is the 2026-11-19 run and has not been repeated. | `docs/mutation-report.md` |
| 3 | Line coverage | **Unverified.** Neither `coverage` nor `cargo-llvm-cov` is installed on troy-1; 95.07% / 99.7% are the 2026-11-19 figures. | `docs/coverage-report.md` |
| 4 | Crash safety | **Partly verified.** Both crash-fix regression tests pass (`swadesh-core` 2, `swadesh-cli` 1). The execution campaign has not been re-run, and its total is quoted inconsistently in this repo: the 2026-11-19 row said 336M while `docs/fuzz-report.md` sums to 365.8M. | `docs/fuzz-report.md`, `fuzz/` |
| 5 | API and CLI parity | **Partly met, CLI unverified on troy-1.** `tests/test_parity.py` **494 passed** (including the `-k cli` cases). `tests/test_cli.py` collects **448** and reports **13 failed, 107 passed, 328 errors**: every error is a setup error, because the CLI fixtures read `/var/www/llmeval/bench/data`, which exists on the VPS and not on troy-1. The earlier 43/405 figure came from a tree where that data was present and is not reproducible here. | re-run 2026-11-24, troy-1 |
| 6 | Robustness and performance | **Unverified here.** The stress campaign has not been re-run. The README's speed-ups do still hold against `bench/results/final4-troy1-100k.json`, which records `exact: true` on every row. | `docs/stress-report.md`, `bench/results/` |
| 7 | Platforms | **Unverified except troy-1**, where the differential suite currently fails (row 1). No other platform has been re-run since the clean-room rewrite. | `bench/results/` |

**Divergences (2026-11-24).** Closed and pushed this session, each with a dated
`docs/lab-notebook.md` entry: the `intl`, `13a` and `zh` tokenizer rules (measured from black-box
probes), chrF's whitespace characters and reference selection, TER's normalized hyphen and
`.`/`,`-run rules and its `no_punct` marker identity, BLEU `add-k`, and the `PairedTest` repr (an
`f"{v}"` on a `np.float32` widens to double where the oracle uses `str`).

**Still open: the TER shift search**, 18 failing cases. Six one-axis variants of its candidate
generation have been measured on a 4,000-sequence battery; all are worse than the committed search
(20/4000), so the oracle differs structurally. The battery, the minimal counterexample and the
traced acceptance sequence are in the notebook. The CLI figure above is unverified on troy-1 for the
reason in row 5.

## Known divergence: TER shift search on long unrelated references

**Status 2026-11-24.** 18 of the 706 selected tests fail, all in
`test_ter.py::test_ter_long_sentences_shifts` (16) and `test_deep.py::test_ter_long_shifts` (2), and all
from one cause: the TER shift search accepts a marginal move that the oracle rejects. It has not been
fixed, and it is recorded here rather than hidden.

**Where it shows, and where it does not** (measured against sacreBLEU 2.6.0, default options):

| input regime | disagreement |
|---|---|
| random word sequences, 1-45 tokens, general battery | **1 / 4,000** |
| references built from block moves (realistic MT-style), 20-60 tokens | **0 / 80** |
| random *unrelated* references, 20-60 tokens | **3 / 80**, all at 51-59 tokens |

So the divergence needs a long segment whose reference is unrelated to the hypothesis — in real
evaluation data the reference is a translation of the same sentence, which is the `0/80` row. No
corpus in `tests/corpus/` or `tests/oracle/` reproduces it.

**What it is, exactly.** Replaying the failing case (57 vs 56 tokens, ours 42 edits, oracle 43) shows
the two searches agree from the third shift onward; the whole difference is one edit unit created in
the first two.

**Reproduction.**
`~/venv311/bin/python -m pytest "tests/test_ter.py::test_ter_long_sentences_shifts" -q`
(the test is Hypothesis-driven, so which draw fails moves between runs), and the two batteries in the
lab notebook for 2026-11-24 (independent vs perturbed references).

**What has been ruled out**, each measured on the batteries above and each worse than the committed
search: the tie-break key (6 variants, 27-34/80), the alignment tie-break order (only
`substitute, insert, delete` works — that fix landed and is why the general battery is 1/4,000 rather
than 20/4,000), the candidate set (aligned-pairs only, 224/400), the candidate collection order
(ref-major, 7/600), the evaluation budget (250-10,000; 1,000 is optimal), the block-length cap (10,
not 50), the move-distance cap (50), the gain metric (banded vs full matrix, identical) and the gain
source (alignment error counts, 43/80).

**Next step, if anyone takes it.** The remaining route is structural rather than a knob: re-derive the
candidate generation and gain accounting of the reference's shift search as a two-stage routine from
black-box probes. That is a multi-day job with no guarantee, and it is the reason this is documented
instead of patched.
