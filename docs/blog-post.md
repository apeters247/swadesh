# Rewriting sacreBLEU in Rust without changing a single bit

*sacreBLEU is the standard way to report BLEU and chrF in machine translation and LLM
evaluation. swadesh is a re-implementation in Rust that returns exactly the same numbers,
down to the last bit of every float, and is 11–65× faster on one core and up to 379× faster on a
24-thread desktop. This post covers what "exactly the same" turned out to require.*

---

## Why bother?

BLEU and chrF are cheap metrics, but people now run them at a scale nobody planned for.
RL fine-tuning computes a sentence-level reward for every sample in every batch. MBR decoding
scores every candidate against every other candidate (n² sentence scores per input).
Data-filtering pipelines score millions of sentence pairs. Significance testing resamples a
corpus thousands of times. In all of these sacreBLEU, which is pure Python, becomes the bottleneck.

A faster implementation is only useful if nobody has to wonder whether its numbers are
comparable. sacreBLEU's whole reason to exist is reproducibility: Matt Post wrote it because
BLEU scores were not comparable across papers. A fast BLEU that differs by 0.01 now and then
would break that. So we set the bar at bit-identical: every `float` compares `==`, every
printed signature is byte-identical, and the CLI produces identical output.

## The architecture that makes exactness tractable

The key decision was choosing what *not* to rewrite.

BLEU, chrF and TER all split into two phases:

1. **Sufficient statistics per segment**: tokenize, extract n-grams, count clipped matches.
   This is where all the time goes, and it is all integer arithmetic.
2. **Score from summed statistics**: a handful of floating point operations, once per corpus.

swadesh moves phase 1 to Rust and leaves phase 2 to sacreBLEU's own Python code: `swadesh.BLEU`
is a *subclass* of `sacrebleu.BLEU` that overrides only `_extract_corpus_statistics`. Integers
are exact, so if our counts match, the corpus score is sacreBLEU's score by construction, along with
its `BLEUScore` object, verbose string and signature.

Floating point emulation is only needed on the new fast paths that compute many scores
natively: batch sentence scores, pairwise matrices for MBR, and bootstrap resampling.

## Quirk 1: the tokenizer bug you must keep

sacreBLEU's Chinese tokenizer decides what counts as a Chinese character using a table of ranges.
Two of the entries look like this:

```python
(u'\u20000', u'\u2a6d6'),  # (UTF16) CJK Unified Ideographs Extension B, release 3.1
(u'\u2f800', u'\u2fa1d'),  # (UTF16) CJK Compatibility Supplement, release 3.1
```

Python's `\u` escape takes exactly four hex digits, so `u'\u20000'` is the two-character string
`'\u2000' + '0'`. The check is `start <= char <= end` on strings, so for a single
character `c` it becomes `c > '\u2000' and c <= '\u2a6d'`. The tokenizer therefore treats **U+2001–U+2A6D**
as Chinese characters: general punctuation, currency symbols, arrows, mathematical operators,
box drawing. Every `€` or `→` in a `zh`-tokenized hypothesis is split out as if it were a Han character.

Is that a bug? Arguably yes. But every published `tok:zh` BLEU score includes its effect, so a
"fixed" implementation would produce different numbers. swadesh reproduces it on purpose, with
a comment explaining why.

## Quirk 2: sacreBLEU's scores depend on your Python version

BLEU's final step is a geometric mean of n-gram precisions:

```python
score = bp * math.exp(sum([my_log(p) for p in precisions[:eff_order]]) / eff_order)
```

That's the builtin `sum()`. **CPython 3.12 changed `sum()` of floats** to use Neumaier
compensated summation (more accurate, and different in the last bits). **CPython 3.14 changed it again**: the
compensated accumulator now also absorbs `int` items, which matters here because
`my_log(0.0)` returns the *int* `-9999999999`.

So the same sacreBLEU version on the same data can give different floats on different Pythons. We
measured it on 13,455 WMT22 segments: between Python 3.11 and 3.12, **15.8% of sentence-level
BLEU scores differ** (by up to 9 ulps, about 1e-13), and 2 of 7 corpus scores differ. At two decimals
nobody will notice. It still means "bit-reproducible across environments" was never quite true,
and an exact re-implementation has to detect the running interpreter and emulate the matching `sum()`.
Details: [python-version-drift.md](python-version-drift.md).

## Quirk 3: bootstrap resampling is a float32 computation

sacreBLEU's confidence intervals and paired significance tests resample the corpus with numpy:

```python
rng = np.random.default_rng(seed)
idxs = rng.choice(len(stats), size=(n_samples, len(stats)), replace=True)
stats_np = np.array(stats, dtype='float32')
scores = [metric._compute_score_from_stats(_s.sum(0)) for _s in stats_np[idxs]]
```

This is the slow part of `--confidence` and `--paired-bs`. `stats_np[idxs]` materializes an
`n_samples × n_segments × n_stats` float32 tensor: 1000 × 5000 × 10 is 200 MB, just to sum it.

There are two exactness traps:

- The statistics are **float32**. Above 2²⁴ (16.7M), a float32 sum is no longer exact, and a
  chrF character count over a large corpus can exceed that. The result then depends on the
  *order* of additions. We checked empirically that numpy reduces axis 0 of each slice by plain
  sequential row-by-row addition (not pairwise summation, which numpy only uses along the
  contiguous axis), and reproduce that order.
- The per-sample score arithmetic mixes Python floats with numpy float32 scalars, and numpy's
  promotion rules for exactly that case changed between numpy 1.x and 2.x (NEP 50). Emulating
  both rule sets in Rust would be fragile.

swadesh keeps numpy for the random draws, so the indices are identical by construction, and
replaces the tensor materialization with a parallel Rust kernel that sums the selected rows
in the same float32 order. Each resampled row then goes through sacreBLEU's own
`_compute_score_from_stats`, so the per-sample arithmetic, including whichever promotion
rules the installed numpy applies, is still sacreBLEU's and numpy's own code.
The paired approximate randomization test works the same way with exact int64 sums. p-values and
confidence intervals match exactly.

## Quirk 4: "lowercase" and "whitespace" are Python's definitions

`-lc` means Python's `str.lower()`, which follows the Unicode version of the interpreter
(Python 3.11 is on Unicode 14, Rust's standard library is newer). `str.split()` splits
on Python's whitespace set, which includes `\x1c`–`\x1f` and `\x85` but not U+200B.
The `intl` tokenizer's `\p{P}`, `\p{S}` and `\p{N}` come from the third-party `regex` module's Unicode
tables.

swadesh generates its whitespace and category tables from the installed Python and `regex`
module. For lowercasing it uses Rust only for Unicode blocks whose case mappings have
been stable across all relevant Unicode versions, and calls CPython's own `str.lower()` for
everything else (and for `Σ`, whose lowercase depends on context).

## How it's tested

- **Differential property-based fuzzing** with hypothesis, using an adversarial alphabet built
  from everything the tokenizers special-case: Unicode spaces, the `zh` boundary characters, `&amp;`, `<skipped>`,
  `-\n`, Turkish `İ`, final sigma, astral-plane CJK, combining marks. Every test compares
  statistics, scores, `str(score)` and signatures with `==`.
- **Tokenizers compared as strings**: 3,000 fuzzed lines per tokenizer plus the full WMT22 files.
- **Real data**: WMT22 en-de, de-en, en-zh, zh-en, en-ja, en-ru and cs-en.
- **Environment matrix**: CPython 3.10–3.14 × numpy 1.26 / 2.x; x86-64 Linux, ARM64 Linux (GB10)
  and Apple M4.
- **The standalone Rust CLI** is compared byte-for-byte against the `sacrebleu` CLI on 290 flag
  combinations, including CRLF, BOM, invalid UTF-8 and multiple references.

And `python -m swadesh.verify -r ref.txt -i hyp.txt` runs the same comparison on *your* data.

## Where the speed comes from

Most of it is just not being Python: no per-n-gram tuple allocations, no `Counter` objects.

- **Exact n-gram keys, no hashing of strings.** Tokens are interned to integer ids per segment,
  and an n-gram of up to 4 word ids (or up to 6 Unicode code points at 21 bits each) is packed
  losslessly into a `u128`. There are no hash collisions to worry about, because the key *is* the n-gram.
- **Clipped matching in one pass.** Instead of building two `Counter`s and taking the
  elementwise minimum, we count reference n-grams in a small open-addressing table and let each
  hypothesis n-gram "take" one unit of count. The table is cleared in O(1) with generation stamps.
- **The `13a` tokenizer on bytes.** Its regexes only test ASCII classes, so all four passes
  can run on UTF-8 bytes and jump between the few bytes that can match (`.`, `,`, `-`) with
  `memchr`, copying the spans in between.
- **Parallelism.** Segments are independent; rayon spreads them across cores with no
  effect on results, since the statistics are integers.

We also evaluated C and hand-written SIMD/assembly for the hot kernels; see
[language-choice.md](language-choice.md) for the measurements.

## Try it

```bash
pip install swadesh
python -c "import swadesh as sacrebleu; print(sacrebleu.corpus_bleu(['the cat sat'], [['the cat sat down']]))"
```

Or, without touching your code: `import swadesh; swadesh.patch()`.

The name comes from Morris Swadesh, the linguist and anthropologist whose core-vocabulary lists
let researchers measure how much two languages share. That is what BLEU does too.
