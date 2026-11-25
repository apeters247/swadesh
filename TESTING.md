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
| 1 | Differential exactness | **Met** (2026-11-25). The seven-file selection reports **0 failing cases** (723 passed / 2 skipped), and the wider local selection — the same seven plus the dependency-licence, version, metric-base, TER-tokenizer, regression and None-reference files — reports **937 passed, 2 skipped, 0 failed** in 9m13s, `test_ter.py` and `test_deep.py` included. The 18 failures at the goal start were closed by two measured fixes: the TER candidate-budget semantics and the XML-entity unescape order. Separately, the library was exercised with sacreBLEU blocked at `sys.meta_path`: every metric, bootstrap, cached references, `mbr` and `PairedTest` run, and `patch()`/the compat wrappers/the mecab tokenizers fail with a message instead of a traceback. | re-run 2026-11-25, Python 3.11.14, troy-1 |
| 2 | Mutation score | **Unverified.** `cargo-mutants` is not installed on troy-1; 94.6% is the 2026-11-19 run and has not been repeated. | `docs/mutation-report.md` |
| 3 | Line coverage | **Unverified.** Neither `coverage` nor `cargo-llvm-cov` is installed on troy-1; 95.07% / 99.7% are the 2026-11-19 figures. | `docs/coverage-report.md` |
| 4 | Crash safety | **Partly verified.** Both crash-fix regression tests pass (`swadesh-core` 2, `swadesh-cli` 1). The execution campaign has not been re-run, and its total is quoted inconsistently in this repo: the 2026-11-19 row said 336M while `docs/fuzz-report.md` sums to 365.8M. | `docs/fuzz-report.md`, `fuzz/` |
| 5 | API and CLI parity | **Partly met, CLI unverified on troy-1.** `tests/test_parity.py` **494 passed** (including the `-k cli` cases). `tests/test_cli.py` collects **448** and reports **13 failed, 107 passed, 328 errors**: every error is a setup error, because the CLI fixtures read `/var/www/llmeval/bench/data`, which exists on the VPS and not on troy-1. The earlier 43/405 figure came from a tree where that data was present and is not reproducible here. | re-run 2026-11-24, troy-1 |
| 6 | Robustness and performance | **Unverified here.** The stress campaign has not been re-run. The README's speed-ups do still hold against `bench/results/final4-troy1-100k.json`, which records `exact: true` on every row. | `docs/stress-report.md`, `bench/results/` |
| 7 | Platforms | **Unverified except troy-1**, where the differential selection is now green (row 1). No other platform has been re-run since the clean-room rewrite. | `bench/results/` |

**Divergences (2026-11-24).** Closed and pushed this session, each with a dated
`docs/lab-notebook.md` entry: the `intl`, `13a` and `zh` tokenizer rules (measured from black-box
probes), chrF's whitespace characters and reference selection, TER's normalized hyphen and
`.`/`,`-run rules and its `no_punct` marker identity, BLEU `add-k`, and the `PairedTest` repr (an
`f"{v}"` on a `np.float32` widens to double where the oracle uses `str`).

**Closed this session: the TER shift search**, the last divergence (18 failing cases). Two causes,
both measured. (1) The 1,000-candidate budget is tested by the oracle once per matching block and
*before* the reduction, so it overshoots by up to one block and refuses a beneficial shift found on
the exhausting iteration; testing it per candidate and applying first lands on exactly 1,000 and
takes the extra shift. (2) Once that was fixed, the normalized TER tokenizer's XML-entity pass turned
out to run `&amp;` first, collapsing `&amp;quot;` to a quote where the oracle leaves `& quot ;`; the
measured order is `&quot;`, `&amp;`, `&lt;`, `&gt;`. Both fixed, both with a spec entry
(`docs/spec/ter.md` §4.2.2 and §3.2). The batteries go to 0/4,000, 0/80 and 0/600, and the
seven-file selection to 0 failing. The CLI figure above is unverified on troy-1 for the reason in
row 5.

## Resolved: TER shift search on long unrelated references (2026-11-24)

**Closed 2026-11-24.** Two causes, both measured, both recorded in `docs/spec/ter.md` (§4.2.2 and
§3.2). Most of the eighteen were not the search at all but the candidate-budget semantics: the oracle
tests the 1,000-candidate cap once per matching block and before the reduction, so it overshoots by up
to one block and refuses the shift found on the exhausting iteration. The last one found (seed 52526)
was a separate bug in the normalized tokenizer: its XML-entity pass ran `&amp;` first, collapsing
`&amp;quot;` to a quote where the oracle leaves `& quot ;`. With both fixed the seven-file selection
reports **0 failing cases** and the batteries report 0/4,000 (general), 0/80 (unrelated-long) and
0/600 (shift-heavy). What follows is the pre-fix record, kept for the history.

**Status (before the fix).** 18 of the 706 selected tests failed, in
`test_ter.py::test_ter_long_sentences_shifts` (16) and `test_deep.py` (2), and the write-up assumed one
cause: the TER shift search accepts a marginal move that the oracle rejects. That was right for most
of them; one was the tokenizer entity order. It has not been
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
the first two. Measured directly on that case (2026-11-24): the plain distance is 49 and the oracle
reaches 43, and of our 161 first-iteration candidates, three would let the oracle reach 41, 65 would
reach 42, 62 leave it at 43 — and the oracle's own move is one of the last group, a net-zero move.
Neither maximising the distance gain, nor the match-count gain, nor first-improvement reproduces that
choice, which is why the remaining route is structural rather than a knob.

**The oracle's first move is not the maximum of any objective measured.** Cross-referencing every one
of the 157 distinct first-iteration candidates on the 57-token case against the oracle's own
continuation (`1 + oracle_total(shifted)`) rules out the two obvious objectives at once: the four
candidates that maximise the edit-distance gain lead to totals 41, 41, 42, 42 - never the oracle's own
43 - and the four that maximise the match-count gain are *the same four*. The 62 candidates that leave
the oracle at 43 are almost all gain-1 moves. So the oracle's first move neither most reduces the edit
distance (the published definition) nor most increases the number of matches, and the oracle's own
result is unreachable by any maximum-gain move in the list.

**Reproduction.**
`~/venv311/bin/python -m pytest "tests/test_ter.py::test_ter_long_sentences_shifts" -q`
(the test is Hypothesis-driven, so which draw fails moves between runs), and the two batteries in the
lab notebook for 2026-11-24 (independent vs perturbed references).

**What has been ruled out**, each measured on the batteries above and each worse than the committed
search: the tie-break key (6 shapes plus first-with-equal-gain-and-length, 27-34/80), the alignment
tie-break order (only `substitute, insert, delete` works — that fix landed and is why the general
battery is 1/4,000 rather than 20/4,000), the candidate set (aligned pairs only, 224/400),
the candidate collection order (ref-major, 7/600), the evaluation budget (250-10,000; 1,000 is
optimal), the block-length cap (10, not 50), the move-distance cap (25-200; 50 is optimal), the gain
metric (banded vs full matrix, identical), the gain source (alignment error counts and match counts,
43/80 and no better), the alignment orientation (reference-as-rows, 48/80), the error-filter position
(at the block's start or boundary, 12-14/80), the full-matrix alignment for candidates (identical)
and first-improvement hill climbing (71/80). Added this session, also all worse: the candidate set
restricted to maximal matching blocks (4/80 and 7/80, against 3/80 and 0/80 committed), the source
block required to be all-errors (28/80, 40/80), the destination block required to be all-errors
(22/80, 36/80), the move-distance cap removed (identical - it never binds on these lengths), the
minimum accepted gain raised to 2 (22/80, 11/80) and to 3 (71/80, 46/80). Every mechanism measured is
worse than the committed search, and on all three observed failures the committed search is strictly
better than the oracle (42<43, 39<40, 9<11), never worse - so the oracle is not a restricted version
of the published search either.

**Against the published algorithm.** `docs/spec/ter.md` §4.2.1 quotes the description this
implementation follows (Snover et al. 2006, Algorithm 1 and its three constraints). The oracle departs
from that description on these long unrelated-reference segments: its first move leaves its total
unchanged where cheaper moves exist in the same candidate list. The implementation follows the
published algorithm, so the divergence is the oracle's, not a known constant that was mis-set.

**External review (2026-11-24).** The sacreBLEU issue tracker was searched from a host with working
network access. Every issue touching TER is about something this implementation already matches:
`asian_support` gating (`normalized` is what enables tokenization, #229), score scaling to 0-100
(#208), empty references and empty segments (#228, #232, #161), `normalized` docstrings (#230, #235)
and zh tokenization (#127). **None describes the shift search**, so the departure is undocumented
publicly and cannot be resolved by reading discussion. Two further sources were read this session and
add nothing: open issue #314 (kana-boundary regexes in Asian TER normalisation, still unmerged - the
implementation returns exactly the oracle's value for its example, i.e. it matches the pre-fix 2.6.0
behaviour), and the TERp technical description (Snover et al. 2009, TER-Plus), whose section 3.2
restates the same four constraints and the same "most reduces the WER" selection rule with no
tie-break or enumeration detail.

**Next step, if anyone takes it.** The remaining route is structural rather than a knob: re-derive the
candidate generation and gain accounting of the reference's shift search as a two-stage routine from
black-box probes. That is a multi-day job with no guarantee, and it is the reason this is documented
instead of patched.
