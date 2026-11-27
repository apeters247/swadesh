---
title: "A faster scorer I couldn't trust: bit-exact sacreBLEU in Rust"
thumbnail: docs/img/benchmark.svg
authors:
- name: Andrew Peters
---

*swadesh is a drop-in replacement for sacreBLEU, written in Rust. It returns exactly the same
numbers — every float compares `==`, the signatures and CLI output are byte-identical — and it is
much faster. This post is about the Naxi translation model I'm training, the measurement standards
around it, and why "faster scorer" only counts if it is also the same scorer.*

Code: [github.com/apeters247/swadesh](https://github.com/apeters247/swadesh) (public, AGPL-3.0 for
everyone, free for commercial use below USD 5M revenue).

---

**tl;dr**

- sacreBLEU's numbers are what people trust, so a faster scorer is worth having only if it produces
  *identical* numbers. A scorer that differs in the last float bit isn't 100× faster; it's a new
  metric.
- On real model output — `qwen3.8-27b` and `llama3.3:70b`, 150 English→German segments each, 12
  configurations apiece — swadesh and sacreBLEU agree exactly: **24/24 corpus scores bit-identical,
  0 of 300 sentence scores differing**, and the same holds for five more cached model outputs.
  **10 more language pairs** (Gujarati, Kazakh, Lithuanian, Arabic both directions, Japanese, Korean,
  German→French, Turkish) are byte-identical too.
- Speedups on 20k WMT22 segments on a 24-core desktop: BLEU 140×, chrF 316×, chrF++ 280×, TER 116×.
  On a single core: 11–36×.
- The design is to move only the *integer* part (tokenization and n-gram statistics) into Rust and
  leave sacreBLEU's own floating-point score arithmetic in place, so corpus scores are identical *by
  construction* rather than by luck.
- It now also has things sacreBLEU doesn't: per-segment TER explanations, a streaming accumulator,
  a paired difference with a power estimate, thread control, a `pytest` plugin, and a HuggingFace
  `evaluate` module.
- Why I'm telling you: if the numbers are bit-identical, swapping the implementation cannot make
  anybody's published benchmark incomparable, and it can only make them wait less.

---

## Why this exists: the Naxi project

I'm training a Naxi ↔ Chinese/English translation model for [naxiai.com](https://naxiai.com). Naxi is
written in Dongba glyphs and in a syllabic romanisation. The model I serve (`nxkd1-k-q5`, a 4.8B
Q5_K_M GGUF, fine-tuned on ~1.15 TB of data on the GPU box in my rack) currently scores
**chrF++ 45.78 / NTQS 69.07** on a leak-free 446-row promotion gate
([Apeters247/naxi-mt-benchmark](https://huggingface.co/Apeters247/naxi-mt-benchmark)).

Naxi is why the measurement rules are severe:

- **Two metrics, always both, same rows.** chrF++ (sacreBLEU, `word_order=2`) says whether the right
  *sentence* was translated; NTQS (my scorer: tone /30, semantic /35, dialect /20, completeness /15)
  says whether the output is well-formed Naxi at all. They fail differently on purpose. One without
  the other is not a result I can quote.
- **Significance by paired bootstrap, clustered by reference, 3000 resamples.** An unclustered
  interval is roughly half as wide and overstates confidence. Everything is reported as an interval,
  not a point score.
- **Exactness, because the history has to stay comparable.** Every number in the ledger — earlier
  arms, published baselines, past decisions — was produced by sacreBLEU 2.6.0. A scorer that agrees
  to seven digits is not the same instrument. I don't accept a new scorer number until it is
  byte-identical to the old one on the same rows, same prompts, same decode, same scorer version.

So the workflow was: change a checkpoint → score every candidate on chrF++ and NTQS over the same
rows → run the paired test → read the interval → usually find the move is inside the noise band. The
noise band is ~1.4 chrF++ across ranks 4–22; an "improvement" of 0.4 is not news. And the scoring
step, in Python, was quietly eating my afternoon.

## The obvious fix, and why it isn't trivial

Install a faster BLEU/chrF implementation. The 100× is easy to get; getting it *without changing a
single number* is the hard part, and here it isn't optional. Every past metric in my logs came from
sacreBLEU 2.6.0. If my scorer returns `45.78173283426314` where sacreBLEU returns
`45.78173283426307`, then every paper and baseline I've measured becomes incomparable, at the last
bit, indefinitely.

That is the bar: statistics in integers (where they are exact), and score arithmetic that is
sacreBLEU's own, or an exact emulation of it where computing natively is much faster.

## What I *didn't* rewrite — and why that's the trick

BLEU, chrF and TER split into two phases:

1. **Per-segment sufficient statistics** — tokenize, extract n-grams, count clipped matches. Integer
   arithmetic, and where all the time goes.
2. **Score from summed statistics** — a handful of floating-point operations, once per corpus.

swadesh moves phase 1 to Rust and leaves phase 2 to sacreBLEU: `swadesh.BLEU` subclasses
`sacrebleu.BLEU` and overrides only `_extract_corpus_statistics`. Integers are exact, so if the counts
match, the corpus score *is* sacreBLEU's score by construction — the same `BLEUScore` object, the same
verbose string, the same signature. Floating-point emulation is only needed on the paths that compute
thousands of scores natively: batch sentence scores, MBR pairwise matrices, bootstrap resampling.

(The Python metric layer has since been ported as well, so the library API no longer needs sacreBLEU
installed at all. What still touches it — `patch()`, the CLI's test-set downloads, `verify.py` and
the mecab/sentencepiece tokenizers — does so on use, and `NOTICE` names each one.)

## Quirks that had to be reproduced exactly

The exactness is mostly archaeology. Four that cost real time:

1. **The `zh` tokenizer counts arrows, math and currency as Chinese.** sacreBLEU's `zh` range table
   contains the 2-character string `u'\u20000'`, which Python reads as `'\u2000' + '0'`; string
   comparison therefore treats U+2001–U+2A6D as Chinese punctuation. We reproduce the bug, and say so.
2. **CPython changed `sum()` over floats in 3.12, and again in 3.14.** BLEU's geometric mean goes
   through `sum()`, so sacreBLEU's own scores differ between interpreters: on WMT22, 2,132 of 13,455
   sentence-level BLEU scores (15.8%) differ between 3.11 and 3.12, by at most 9 ulps. swadesh detects
   the interpreter and matches it. The drift table is in `docs/python-version-drift.md`.
3. **Bootstrap arithmetic is float32 numpy.** Materialising `stats_f32[idxs]` is where sacreBLEU's
   memory goes. We keep numpy for the random draws (so the indices are identical by construction) and
   sum the selected rows natively in numpy's float32 order; the per-row arithmetic is sacreBLEU's own
   `_compute_score_from_stats`.
4. **"lowercase" means Python's `str.lower()`** — tied to the interpreter's Unicode version — and
   `str.split()` splits on Python's whitespace set (which includes `\x1c`–`\x1f` and `\x85`, but not
   U+200B). Rust's standard library is not the same; we generate tables from the installed Python and
   call back into CPython for case mappings that have moved across Unicode versions.

The most interesting bug was in my own bookkeeping, not the maths. `num_refs` in the signature is the
*per-segment* count of non-`None` references, not `len(references)`. I "optimised" it to the latter,
which is only equal when no reference stream contains `None`. On a warm-cache path it made
`corpus_bleu(hypotheses, None)` raise `TypeError` where sacreBLEU returns a score, and printed
`num_refs=5` where sacreBLEU printed `1`. The parity suite caught it, and it now has its own
regression test with `None` in the first, middle and only position, across BLEU, chrF and TER.

That is the shape of this work: the maths is exact long before the bookkeeping is. The same thing
happened again, later, in the streaming accumulator: the point score was right while the retained
bootstrap rows were silently wrong, because the running total aliased the first stored row. A
point-score test would have passed. The bootstrap comparison is what caught it.

## Does it agree with sacreBLEU on real model output?

That is the question that decides whether any of this is publishable, and only real output answers it.
On a workstation with an RTX PRO 6000 I generated translations with small models served by Ollama,
then scored the same hypotheses with sacreBLEU 2.6.0 and with swadesh, comparing every corpus score
*and every sentence score* exactly.

Data: WMT24 English→German news, temperature 0, greedy decode.

| model | segments | BLEU | chrF | chrF++ | TER | corpus `==` | sentence diffs |
|---|---|---|---|---|---|---|---|
| `qwen2.5vl:7b` | 600 | 22.256488 | 52.564888 | 49.482473 | 68.130071 | all | **0 / 600** |
| `granite4:small-h` | 150 | 25.871423 | 58.480785 | 54.969857 | 62.555391 | all | **0 / 150** |
| `devstral:24b` | 150 | 26.958969 | 59.130336 | 55.705941 | 61.275234 | all | **0 / 150** |
| `qwen3.8-abliterated` | 150 | 26.407621 | 59.285038 | 55.861321 | 62.161497 | all | **0 / 150** |
| `nxkd1-k-q5` (my Naxi model, English→Naxi vs. the English source) | 250 | 0.101294 | 11.884049 | 9.851019 | 111.973392 | all | **0 / 250** |

Across BLEU (default / `intl` / floor smoothing), chrF (default / `beta=3`), chrF++ and TER (default /
normalized), that is 24 scored configurations for the three 150-segment models alone: 24 identical
corpus scores and zero differing sentence scores in every row. The Naxi row matters most to me — it is
my worst-quality output, and if a scorer were going to disagree anywhere, that is where it would.

They also rank identically. Of the three small models on the same 150 segments, devstral and qwen3.8
sit 0.03 chrF apart — inside the ~1.4 chrF noise band from the Naxi work. The honest reading is
"these two are tied", and both scorers say so.

**One trap worth passing on.** `qwen3.8` is a reasoning model. With a small `num_predict` it returns
an empty `response` and puts everything into `thinking`, so my harness recorded 150 empty hypotheses.
Both scorers then returned a very tidy `0.00`. Identical, and completely meaningless. If you evaluate
a thinking model, assert your hypotheses are non-empty before you believe anything downstream.

## Speed, on the shapes that actually hurt

All numbers below are from the same machine, same data, back to back, with the result compared exactly
each time. Sources: `bench/bench.py`, `bench/stress/`, and an extra harness written for this post.

Corpus scores, WMT22 en→de, Ryzen 9 7900X (24 threads), sacreBLEU 2.6.0 on CPython 3.11.14, idle
machine:

| case | sacreBLEU | swadesh | speedup | exact |
|---|---|---|---|---|
| BLEU corpus, 20,000 | 1.168 s | 0.0083 s | **140×** | yes |
| chrF corpus, 20,000 | 3.458 s | 0.0109 s | **316×** | yes |
| chrF++ corpus, 20,000 | 4.006 s | 0.0143 s | **280×** | yes |
| TER corpus, 10,000 | 6.406 s | 0.0552 s | **116×** | yes |
| BLEU corpus, 100,000 | 5.816 s | 0.0497 s | **117×** | yes |
| chrF corpus, 100,000 | 18.559 s | 0.0591 s | **314×** | yes |
| chrF++ corpus, 100,000 | 20.802 s | 0.0735 s | **283×** | yes |

Pinned to one core (`taskset -c 0`) the same 20,000-segment cases are **11–36×**. The rest is
parallelism, and segments are independent, so the thread count never changes a result — a property
worth having, which is why the thread count is now settable (`swadesh.n_jobs(k)`) rather than fixed.

Now the workloads that hurt inside a fine-tuning loop. The first two are what the Naxi promotion gate
does every time: batch the sentence scores, then run a paired test.

| workload | sacreBLEU | swadesh | speedup | exact |
|---|---|---|---|---|
| BLEU `sentence_scores` batch, 20,000 (per-sample reward shape) | 0.126 s | 0.0093 s | **13.5×** | yes |
| chrF `sentence_scores` batch, 20,000 | 0.283 s | 0.0123 s | **23.1×** | yes |
| MBR chrF++, 10 candidates × 10 inputs | 0.346 s | 0.0086 s | **40.4×** | yes |
| BLEU + 1000-resample bootstrap CI, 5,000 segments | 0.195 s | 0.0184 s | **10.6×** | yes |
| naive loop of 2,000 `sentence_score` calls | 0.062 s | 0.0162 s | **3.8×** | yes |
| `PairedTest` (bs), 2,000 segments, 2 metrics | 0.666 s | 0.790 s | 0.8× | yes |
| `PairedTest` (ar), 2,000 segments, 2 metrics | 0.618 s | 0.750 s | 0.8× | yes |
| CLI `BLEU` corpus, 20,000 lines, process start to output | 0.057 s | 0.064 s | 0.9× | yes |
| CLI `chrF++` corpus, 20,000 lines | 0.062 s | 0.065 s | 1.0× | yes |

Three rows are honest negatives, and I left them in:

- **`PairedTest` is not faster at this size.** On 2,000 segments and 500 resamples, sacreBLEU's
  numpy implementation is already good; mine is exact but comes out level. It may win on much larger
  resample or segment counts, where sacreBLEU's tensor materialisation dominates — I haven't measured
  that. Do not quote it as a win.
- **The CLI is at parity, not ahead.** Start to finish on 20,000 lines it is 0.9–1.0×: both tools pay
  the interpreter's import graph, and at that size scoring is a few tens of milliseconds either way.
  If you script one call per file, use the library.
- **chrF at a million segments has no "other side" to compare.** See below.

Where it *is* dramatic is inside the API. Sentence-level TER is the funny one: sacreBLEU takes about
**2.6 seconds per sentence** on that data, so a 150-segment normalized-TER comparison is several
minutes on one side and under a second on the other. That single number is the difference between
"I re-scored every arm tonight" and "I'll re-score the day I have to defend it".

At a million segments it stops being a timing question:

| | sacreBLEU | swadesh | exact |
|---|---|---|---|
| BLEU, 1,000,000 segments | 205 s, **9.0 GB** peak | 2.5 s, **0.97 GB** | identical score, `64.60594655357025` |
| chrF, 1,000,000 segments | **`MemoryError`; no score** (hit a 19.9 GB address-space cap) | 1.04 s, 1.06 GB | — |

To be precise about what I measured: I ran sacreBLEU under a 19.9 GB address-space cap and it raised
`MemoryError`; it may well succeed on a machine with considerably more RAM, and I haven't tested that.
There is no score on the other side either way, so I'm not calling it a speedup.

## What's in it now that sacreBLEU doesn't have

All of these are additive, and all of them are bit-identical to doing the same thing with sacreBLEU in
a loop:

```python
import swadesh
from swadesh import BLEU, CHRF, TER, compare, evaluate

# Per-segment TER, broken into its parts: the accepted shifts, the final edit distance,
# the alignment counts, which reference won. Same search corpus_score runs.
TER().explain(hyps, [refs])[0]
# -> {"ter": 16.67, "edits": 1, "shifts": 1, "distance": 0, "counts": {...}, "accepted_shifts": [...]}

# Stream a corpus that doesn't fit in memory: one statistics row of state, score() any time.
acc = swadesh.Accumulator(BLEU())
for hyp, ref in zip(hyps, refs):
    acc.add(hyp, [ref])
acc.score()

# The signed gap between two systems, with its interval, and how many segments a target gap needs.
d = swadesh.paired_difference(base, new, [refs], BLEU())
d.delta, d.ci_low, d.ci_high, d.p_value      # same p-value as PairedTest, plus the size of the gap
d.required_n(effect=1.0)                     # segments needed to call a 1.0-BLEU gap

# Several metrics in one call; every system scored, paired and ranked in another.
evaluate(hyps, [refs])                       # {"BLEU": ..., "chrF2": ..., "TER": ...}
print(compare({"base": base, "new": new}, [refs]))

with swadesh.strict():                       # refuse the silent truncation
    BLEU().corpus_score(hyps, [refs])
with swadesh.seed(7):                        # the interval's own signature says which seed
    BLEU().corpus_score(hyps, [refs], n_bootstrap=1000)
with swadesh.n_jobs(4):                      # cap the threads; the score is identical at any count
    BLEU().corpus_score(hyps, [refs])
```

For a test suite, there is `swadesh.testing.assert_bleu_at_least(hyps, refs, 30)` plus a `pytest`
plugin (`pytest --swadesh-report scores.json` records every score a test computes). For the
HuggingFace ecosystem there is an `evaluate` module, `pip install "swadesh[hf]"`, then
`evaluate.load(".../swadesh/evaluate_module.py")` — note that loading *by name* would need a Hub
publication, which I haven't done, so the path form is the one that works.

I declined one thing on purpose. sacreBLEU's `ja-mecab`, `ko-mecab` and `spm` tokenizers are served
by delegating to sacreBLEU, not reimplemented in Rust: their segmentation depends on a *dictionary*
version I don't control, so a port would be exact today and silently wrong after any dictionary bump.
The measurement and the reasoning are in `docs/tokenizer-dependency-decision.md`. If you need those
tokenizers, `pip install "swadesh[sacrebleu]"` and the numbers are identical.

## How it's tested

I don't think the speedup is the interesting claim; the exactness is, so here is what stands behind it.

- **Parity suite: 494 tests**, comparing every public metric method, compat function, `Score` /
  `Signature` attribute, exception type and message, CLI flag and output, including pickling,
  `patch()`/`unpatch()` and forked multiprocessing.
- **Differential fuzzing** against sacreBLEU over an adversarial Unicode alphabet: Unicode
  whitespace, CJK, combining marks, `&`/`</`/`-\n`, Turkish `İ`, final sigma, astral characters —
  79 differential cases plus the long-sentence TER shift suite.
- **The real-model evidence above**, re-verified on the current build.
- **10 extra language pairs** byte-identical: Gujarati, Kazakh, Lithuanian, Arabic in both directions,
  Japanese and Korean, German→French, Turkish (`docs/multilingual-parity.md`).
- **Rust side**: 13 test suites, 140 tests, all green — the tokenizers, the statistics, the shift
  search and the score arithmetic, against their own recorded expectations.
- **My own data, on your machine**: `python -m swadesh.verify -r ref.txt -i hyp.txt` runs the same
  comparison on yours and prints either "identical" or the first mismatch.

What is *not* verified by me: the GitHub Actions matrix is committed (Linux/macOS/Windows across
Python 3.10–3.14 and both numpy lines) but I have not watched it run on GitHub's runners, so I can't
claim it. Every number in this post comes from x86-64 Linux machines I own.

## Try it

```bash
pip install "git+https://github.com/apeters247/swadesh"
python -c "import swadesh as sacrebleu; print(sacrebleu.corpus_bleu(['the cat sat'], [['the cat sat down']]))"
```

If you already have code that calls `sacrebleu`, `swadesh.patch()` is the whole change:

```python
import swadesh
swadesh.patch()      # sacrebleu.* now runs on swadesh, numbers unchanged
```

If you write new code, `sentence_scores`, `pairwise` (MBR matrices), `explain`, the streaming
accumulator and the native bootstrap are there too. The library needs only NumPy.

The name comes from Morris Swadesh, the linguist whose core-vocabulary lists let researchers measure
how much two languages share. BLEU and chrF do the same for a translation and its reference, which
felt like the right name for a tool whose job is to be exactly comparable.

swadesh is dual-licensed: **AGPL-3.0-or-later** for everyone, and a commercial licence for the cases
the AGPL doesn't suit — free of charge below USD 5 million annual revenue, negotiated above
(`LICENSE.commercial`, apeters247@gmail.com).

*More from me: the Naxi project at [naxiai.com](https://naxiai.com), its benchmark at
[huggingface.co/Apeters247/naxi-mt-benchmark](https://huggingface.co/Apeters247/naxi-mt-benchmark),
and everything else under [huggingface.co/Apeters247](https://huggingface.co/Apeters247).*

*Thanks for reading. If you re-run this on your own data, `swadesh.verify` prints either "identical"
or the first mismatch — that report is more useful to me than a star on the repo.*
