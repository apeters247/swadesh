# How swadesh stays bit-exact with sacreBLEU

This document explains where swadesh's exactness comes from, which sacreBLEU
behaviours it reproduces on purpose, and how the claim is tested. Read the
[user guide](guide.md) first for the API.

"Exact" here means: for the same inputs, on the same CPython and NumPy, swadesh
returns the same `float` (compared with `==`), the same integer statistics, the
same formatted string, and the same signature as sacreBLEU 2.6.0. It does not
mean the same results across interpreters. sacreBLEU itself changes its last bits
between some CPython versions (see [CPython sum() semantics](#34-cpython-sum-semantics)).

## 1. Where the boundary is

The fastest part of sacreBLEU is the per-segment work: tokenizing, extracting
n-grams and matching them against each reference. swadesh moves that work to
Rust. The rest stays in sacreBLEU's Python.

| Stage | Where it runs | Why |
|---|---|---|
| Tokenization for `13a`, `intl`, `zh`, `char`, `none` | Rust (`tokenize.rs`) | Per-character work, the hot path |
| Tokenization for `ja-mecab`, `ko-mecab`, `spm`, `flores*`, `spBLEU-1K` | sacreBLEU's Python, then native matching | Only the five tokenizers in the row above are in Rust |
| Lowercasing | Rust for most text; CPython's `str.lower()` for the rest (`lib.rs`, `text`) | See [section 3.7](#37-strlower-delegation-for-unstable-blocks) |
| n-gram extraction and matching; BLEU and chrF sufficient statistics | Rust (`stats.rs`, `ngram.rs`, `lib.rs`) | Per-segment work |
| Corpus score: `compute_bleu`, chrF F-score, `BLEUScore`/`CHRFScore` objects, signatures, formatting | sacreBLEU's Python | Runs once per corpus. Keeps `sum()` and float formatting identical by construction |
| Sentence scores in batch (`sentence_scores`, `pairwise`, `mbr`, rewards) | Rust port of `compute_bleu` and the chrF F-score; BLEU's log-mean uses an emulated `sum()` | Called once per segment, so the Python path would dominate |
| Single `sentence_score` | Native statistics, then sacreBLEU's Python for the score | Same arithmetic as corpus scoring |
| Bootstrap resample sums | Rust (`resample.rs`) over the same `float32` values, without building sacreBLEU's sampled tensor | The tensor is the memory cost |
| Approximate randomization sums | Rust (`ar_sums_i64`), exact integer arithmetic | Integer statistics make this exact |
| Random index draws | NumPy `default_rng`, called as sacreBLEU calls it | See [section 3.6](#36-random-draws-still-come-from-numpy) |
| Confidence interval, p-value, `Result` | sacreBLEU's Python | Runs once per test |

The consequence is that the score arithmetic you see in a corpus result is
sacreBLEU's own code. swadesh only has to produce the same integer statistics.
For the batch paths, the score arithmetic is a port, and it is tested for `==`
against sacreBLEU's `sentence_score`.

The Python classes subclass sacreBLEU's `BLEU` and `CHRF` (`python/swadesh/metrics.py`).
`_extract_corpus_statistics` is the method swapped for the native version. Its
output is the matrix that sacreBLEU's `_aggregate_and_compute` already consumes.

## 2. Testing

### 2.1 Differential property tests

`tests/test_differential.py` runs sacreBLEU and swadesh on the same inputs with
[hypothesis](https://hypothesis.readthedocs.io/). Each hypothesis test runs 300
examples (`max_examples=300`, no deadline). The compared values are:

- integer sufficient statistics, compared as lists of `int` (`==`);
- corpus scores, compared with `==`, along with `str()` and the statistics
  (`counts`, `totals`, `precisions`, `bp`, `sys_len`, `ref_len`);
- signatures (`str(get_signature())`);
- every sentence score, from `sentence_scores`, `pairwise`, and single
  `sentence_score` calls;
- bootstrap results: `str()`, and the private `_mean` and `_ci` fields, compared
  with `==` and matching type;
- paired-test output: `str()` of the signatures and `repr()` of every `Result`;
- `mbr`, reward functions (`chrf_reward`, `bleu_reward`, with conversational
  input), and the emulated `sum()` against the builtin `sum()` on 20,000 random
  lists.

Corpora have one to six segments of up to 25 symbols each. Parameters vary over:
tokenizer (`13a`, `intl`, `zh`, `char`, `none`), lowercasing, smoothing method and
value, maximum n-gram order, effective order, chrF orders and beta, whitespace,
epsilon smoothing, and the number of references. Some references are `None`,
which exercises sacreBLEU's missing-reference handling.

The alphabet is chosen to hit each special case in the tokenizers:

- ASCII punctuation that 13a splits on, and characters from its digit and
  punctuation classes;
- the HTML entities `&amp;`, `&quot;`, `&lt;`, `&gt;`; the markers `<skipped>`
  and `-\n`; newlines and tabs;
- Unicode whitespace: U+3000, U+001C, U+0085, and others. Zero-width space
  U+200B is included as a non-space;
- CJK ideographs, fullwidth forms, CJK radicals (U+2F81 and U+2FA1, for example),
  and characters in the range the `zh` quirk covers (see section 3.1);
- Greek final sigma and other case-mapping edge cases: `Σ σ ς`, `İ`, `ı`, `ß`,
  `ẞ`, `ǅ`;
- combining marks, Arabic-Indic digits, superscripts, Roman numerals;
- astral characters (U+1F600, U+20001, U+2A6D6, U+1D7CE).

Other tests cover the `references=` cache, the `corpus_bleu` and `sentence_bleu`
compatibility functions, bootstrap over several sizes (1, 2, 37, 1500
segments), a bootstrap case above float32's exact integer range (section 3.5),
both paired tests with three systems, and pairwise matrices.

### 2.2 The CLI against sacreBLEU's CLI

`tests/test_cli.py` runs the standalone binary and sacreBLEU's CLI on the same
WMT22 files (from `bench/data`, fetched by `bench/get_data.sh`). It compares
stdout byte for byte and the exit code. It skips when the binary is not built.

### 2.3 Real data

`bench/bench.py` runs both implementations on WMT22 references (en-de by default),
with perturbed hypotheses, and it checks each timed result with `==`. The
results are in `bench/results/SUMMARY.md`. Every row there is `exact: yes`.

`python -m swadesh.verify` runs the same comparison on any reference and
hypothesis files.

### 2.4 Interpreter and platform matrix

The CI workflow (`.github/workflows/ci.yml`) runs:

- `cargo test -p swadesh-core` and `cargo clippy -D warnings` on Linux;
- the differential suite on Python 3.10, 3.11, 3.12, 3.13 and 3.14, with
  NumPy 2.x, on `ubuntu-24.04`, `macos-15` and `windows-2025`;
- the differential suite on Python 3.10, 3.11 and 3.12 with NumPy 1.26 on
  `ubuntu-24.04`;
- `tools/check_sum_semantics.py` in the NumPy 2 jobs.

So the NumPy 1 matrix is smaller than the full product: it does not include 3.13
or 3.14. The release workflow builds wheels for Linux x86_64 and aarch64 (manylinux and
musllinux), macOS Intel and Apple silicon, and Windows x64.

The benchmark runs are the cross-machine evidence: an x86_64 Linux host
(troy-1, AMD Ryzen 9 7900X), an aarch64 Linux host (GX10, Cortex-X925 and
Cortex-A725), and an arm64 macOS host (Mac mini, Apple M4). Each runs the
exactness check against sacreBLEU 2.6.0 on the same machine.

### 2.5 Rust-side tests

`crates/swadesh-core/tests/nprng.rs` checks the Rust port of NumPy's PCG64,
`SeedSequence` and `choice` against fixtures in
`crates/swadesh-core/tests/fixtures/`. `tests/gen_rng_fixtures.py` writes those
fixtures from the installed NumPy. The port is not used by the Python package
yet (section 3.6).

## 3. The sacreBLEU behaviours that are reproduced

### 3.1 The `zh` tokenizer's two-character ranges

sacreBLEU's `TokenizerZh` keeps a table of Unicode ranges as Python strings. Two
of its entries are written as `u'\u20000'` and `u'\u2f800'`. Those are not single
code points. Python reads `\u2000` as one character, so `u'\u20000'` is the
two-character string `"\u2000" + "0"`. The same holds for `u'\u2f800'`.

The membership test compares a one-character string against each range with
`<=` and `>=`. A one-character string is ordered before any longer string with the
same prefix, so the two-character bounds act as range endpoints just outside
the codepoints they name:

- the lower bound `"\u2000" + "0"` admits U+2001 and above, and excludes U+2000;
- the upper bound `"\u2a6d" + "6"` admits U+2A6D and below, and excludes U+2A6E.

The result is an effective range of U+2001 to U+2A6D. Astral code points are
outside it, because U+20000 is numerically greater than U+2A6D. The same
reasoning gives U+2F81 to U+2FA1 for the second entry, although that entry adds
nothing, because the Kangxi radicals range U+2F00 to U+2FDF already covers it.

`tokenize.rs` writes the effective ranges directly, as `(0x2001, 0x2A6D)` and
`(0x2F81, 0x2FA1)`, and documents the reason. Excluding the code points the other
ranges already cover, the quirk adds 2,221 code points. These include the em dash
U+2014 and the mathematical operator U+2200. The zh tokenizer therefore puts
spaces around them, as in `a—b` to `a — b`.

We checked the Rust table against sacreBLEU's `TokenizerZh._is_chinese_char` for
every Unicode scalar value (1,112,064 values, surrogates excluded). There were 0
mismatches.

### 3.2 Unicode tables generated from the installed `regex` module

The tokenizers depend on three character classes, plus Python's whitespace:

- `\p{N}`, `\p{P}` and `\p{S}`, as the `regex` module defines them. sacreBLEU's
  `intl` tokenizer uses these classes.
- The set of code points that `str.split()` treats as whitespace.

`tools/gen_tables.py` produces `crates/swadesh-core/src/tables.rs` by testing
every code point with the installed `regex` module and CPython. It writes the
ranges as sorted, inclusive `(u32, u32)` pairs. The header of the checked-in
file records the generator's versions: CPython 3.11.14 and `regex` 2026.11.19.
Regenerate the tables with `python tools/gen_tables.py > crates/swadesh-core/src/tables.rs`
when you change either version.

The tables are checked in, not generated at build time. The differential tests
cover only the code points in their alphabet and in the WMT22 data. A change in
Unicode data between Python versions could therefore change a class for a code
point that no test covers. That is a known limit (section 4).

### 3.3 CPython `str.split()` whitespace

`' '.join(line.split())` and `str.rstrip()` split on a fixed set of code points.
`unicode.rs` reproduces it:

- ASCII: `\t`, `\n`, `\x0b`, `\x0c`, `\r`, `\x1c` to `\x1f`, and space;
- non-ASCII: the non-ASCII entries of the `PY_WHITESPACE` table, which holds the
  code points where `c.isspace()` is true and `('a' + c + 'b').split()` has two
  parts. They are U+0085, U+00A0, U+1680, U+2000 to U+200A,
  U+2028, U+2029, U+202F, U+205F and U+3000. Zero-width space U+200B is not in it.

`py_split` and `py_strip` use these predicates. The tokenizers rely on them,
and `join_split_str` builds the final token string from them.

### 3.4 CPython `sum()` semantics

sacreBLEU 2.6.0 computes the BLEU geometric mean with
`sum([my_log(p) for p in precisions[:eff_order]])`. `my_log` returns the int
`-9999999999` for a zero precision and a float otherwise. The list is short and
mixes ints and floats, so how CPython's builtin `sum()` accumulates it matters.

CPython changed that algorithm for floats in 3.12, and again for ints in the
float phase in 3.14. `score.rs` emulates the three behaviours:

| `SumMode` | CPython | Float items | Int items after the first float |
|---|---|---|---|
| `Naive` | up to 3.11 | `f += x` | `f += x` |
| `Neumaier` | 3.12, 3.13 | Neumaier compensated sum; compensation folded in at the end | added naively |
| `NeumaierInts` | 3.14 and later | compensated, as for floats | compensated |

`SumMode::for_python` picks the mode from `sys.version_info` when a binding is
called. Versions after 3.14 use `NeumaierInts`, and the code marks that as
unverified. The `_sum_mode()` diagnostic reports the selected mode. On the CPython
3.11 used for the checked-in tables it returns `naive`.

Where this matters:

- Corpus scores are computed by sacreBLEU's own `sum()` on the host interpreter,
  so they follow it automatically.
- The Rust sentence paths (`sentence_scores`, `pairwise`, `mbr`, rewards) use the
  emulated `sum()`. They match sacreBLEU's `sentence_score` on the same interpreter.
- The standalone binary has no interpreter. Its `--python-version X.Y` option selects
  the mode, and its default is 3.12.

[python-version-drift.md](python-version-drift.md) measures the effect. On 13,455 perturbed WMT22
segments, 2,132 sentence BLEU scores (15.8%) differ between CPython 3.11 and 3.12,
by at most 9 ulps (8.5e-14). Two of seven corpus scores differ by about 4e-14 and
7e-14. The 3.12, 3.13 and 3.14 runs in that document showed no differences from
each other on that data. The document says 3.14 drift is still possible, because
`tools/check_sum_semantics.py` detects a difference on random inputs for the
3.14 model. It was not observed on the WMT22 data.

`tools/check_sum_semantics.py` compares the three emulations with the builtin
`sum()` on random lists, using the bit pattern of each float. A new CPython
version should run it before its `SumMode` is trusted.

### 3.5 NumPy `float32` sums in bootstrap

sacreBLEU's bootstrap builds `stats_f32[idxs]`, a tensor of shape
`n_samples x n_segments x width` in `float32`, and reduces it with `.sum(0)`. The
sums depend on the order of the additions in `float32`, so swadesh does the same
additions in the same order.

`resample.rs` (`bootstrap_sums_f32`) adds each selected row into a zeroed
accumulator in index order. The summation runs sequentially over the segments for
each output column. The code's comment says that NumPy starts from the first
element instead of `0.0`, and that this gives the same value for the non-negative
statistics that occur here. The sequential order was checked against NumPy
empirically, not taken from NumPy's documentation.

Once a sum passes 2^24, `float32` can no longer represent every integer, so the
summation order decides the last bits. `test_bootstrap_float32_rounding_regime` checks a corpus
that is large enough to cause this.

The paired approximate-randomization test uses `int64` sums (`ar_sums_i64`). Integer
addition is exact, so the result does not depend on the order.

### 3.6 Random draws still come from NumPy

Index sampling (`rng.choice`) and the approximate-randomization mask
(`rng.integers`) are called from Python, through `np.random.default_rng` with the same
arguments as sacreBLEU. The seed comes from `SACREBLEU_SEED`, as in sacreBLEU.

`nprng.rs` contains a Rust port of NumPy's generator. It is tested against NumPy's
output, but the Python bindings do not call it yet. Porting the draws would remove
the NumPy dependency from the resampling path.

### 3.7 `str.lower()` delegation for unstable blocks

Lowercasing is applied before tokenization when `lowercase=True`. `lib.rs` (`text`)
handles each string as follows:

1. If the string is ASCII, lowercase its ASCII letters. That matches `str.lower()`.
2. If every character is in the "stable" set, call Rust's `str::to_lowercase`.
   `lower_is_stable` lists the code points with no context-dependent or
   version-dependent case mapping: all of U+0000 to U+052F except U+03A3, plus
   the blocks U+1E00 to U+1EFF, U+2000 to U+206F, U+3000 to U+9FFF, U+AC00 to
   U+D7A3, and U+FF00 to U+FFEF. The code comment says those mappings are the
   same in Unicode 13 and later, which every CPython that swadesh supports includes.
   That is the author's claim. The fuzzing alphabet covers part of it, and this
   document does not prove it.
3. Otherwise, call the string's `lower()` method with CPython. That is exact by
   construction.

U+03A3 (Greek capital sigma) is excluded from the stable set. Its lowercase form
depends on context: the final-sigma rule gives `ς` at the end of a word and `σ`
elsewhere. Rust and CPython could disagree on that, so the string goes to CPython.

chrF's `lowercase` option goes through the same `text` function, because
`chrf_stats` takes the same flag.

## 4. Limits

- Exactness holds for one interpreter and one NumPy at a time. The drift
  document shows that sacreBLEU's own corpus scores change in the last bits
  between CPython 3.11 and 3.12.
- The Unicode tables and the `lower_is_stable` set are fixed when they are
  written. Tests catch a difference only if an input contains a code point whose
  class changed. Regenerate the tables when the Python or `regex` version changes
  and rerun the suite.
- The differential alphabet is hand-picked. It is broad, but it is not exhaustive.
  The WMT22 and perturbation data are real text, but not every language.
- Bootstrap and significance draws come from NumPy (section 3.6), so a future
  NumPy change to `default_rng` would change swadesh's results and sacreBLEU's
  together.
- TER is sacreBLEU's Python code. Python-tokenized BLEU tokenizers (`ja-mecab`,
  `ko-mecab`, `spm`, `flores101`, `flores200`, `spBLEU-1K`) run sacreBLEU's
  tokenizer in Python, and the examples did not exercise them.
- The standalone binary's `sum()` mode is set by a flag, not by the interpreter.
