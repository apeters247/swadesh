---
title: "I wanted my Naxi translator to be scored faster, then found I couldn't trust the scores"
thumbnail: docs/img/benchmark.svg
authors:
- name: Andrew Peters
---

*swadesh is a drop-in replacement for sacreBLEU, written in Rust. It returns exactly the same
numbers — every float compares `==`, the signatures and CLI output are byte-identical — and it is
much faster. This post is about the Naxi translation model I'm training, the measurement standards
around it, and why "faster scorer" only counts if it is also the same scorer.*

The code is [github.com/apeters247/swadesh](https://github.com/apeters247/swadesh).

---

tl;dr

- sacreBLEU's numbers are the thing people trust, so a faster scorer is useful only if it produces
  *identical* numbers.
- On real model output — qwen3.8-27b and llama3.3:70b, 150 English→German segments each, 12
  configurations apiece, re-verified on this build — swadesh and sacreBLEU agree exactly on every
  metric, corpus and sentence level: **24/24 corpus scores bit-exact, 0 of 300 sentence scores
  differing**, and the same for five more cached model outputs in the earlier pass.
- Typical speedups on 20k WMT22 segments on a 24-core desktop: BLEU 140×, chrF 316×, chrF++ 280×,
  TER 116×. On a single core: 11–36×.
- The Naxi project is why the speed matters: it is scored on chrF++ *plus* NTQS on the same rows,
  with paired bootstrap CIs clustered by reference — 3000 resamples, every time a checkpoint
  is considered.
- Why I'm telling you all this: if a metric is bit-identical and faster, anyone who re-runs a
  published benchmark gets the same number in less wall-clock time. That is a property of the
  scorer, not of the project, and it's the reason to hand it to other people.

---

## The Naxi project, in one paragraph

I'm training a Naxi ↔ Chinese/English translation model for [naxiai.com](https://naxiai.com). Naxi
is written in Dongba glyphs and in a syllabic romanisation; the model I serve (`nxkd1-k-q5`, a 4.8B
Q5_K_M GGUF, coming from fine-tuning on ~1.15 TB of data on the GPU box in my rack) currently scores
**chrF++ 45.78 / NTQS 69.07** on a leak-free 446-row promotion gate
([Apeters247/naxi-mt-benchmark](https://huggingface.co/Apeters247/naxi-mt-benchmark)).

Naxi is why the measurement rules are severe:

- **Two metrics, always both, same rows.** chrF++ (sacreBLEU, `word_order=2`) says whether the right
  *sentence* was translated; NTQS (my scorer: tone /30, semantic /35, dialect /20, completeness /15)
  says whether the output is well-formed Naxi at all. They fail differently on purpose. One without
  the other is not a result I can quote.
- **Significance by paired bootstrap, clustered by reference, 3000 resamples.** An unclustered CI is
  roughly half as wide and overstates confidence. Everything is reported as a CI, not a point score.
- **Exactness, because history has to stay comparable.** The gate's numbers exist to be compared
  against earlier arms, published baselines and a ledger of past decisions, and all of those were
  produced by sacreBLEU 2.6.0. A scorer that agrees to seven digits is not the same instrument. That
  is why I don't accept a new scorer number until it is byte-identical to the old one on the same
  rows, same prompts, same decode, same scorer version.

So the workflow was: change a checkpoint → score every candidate on chrF++ and NTQS over the same
rows → run the paired test → read the CI → usually find the move is inside the noise band. The
"noise band" is ~1.4 chrF++ across ranks 4–22; an "improvement" of 0.4 is not news. And the
scoring step, in Python, was quietly eating my afternoon.

## The obvious fix, and why it wasn't

Install a faster BLEU/chrF implementation. The 100× speedup is easy to get. Getting it *without
changing a single number* is the hard part, and for this project it isn't optional: every past
metric number in my logs was produced by sacreBLEU 2.6.0. If my scorer returns
45.78173283426314 where sacreBLEU returns 45.78173283426307, the whole history becomes
uncomparable to every paper and every baseline I've measured, at the last bit, indefinitely.

A faster scorer that differs by a float ulp isn't 100× faster. It's a new metric.

That's the bar I set for swadesh: statistics in integers (where exact), and score arithmetic that
is sacreBLEU's own Python code, or an exact emulation of it where it is faster to do so natively.

## The design that makes exactness possible

The key decision was what *not* to rewrite. BLEU, chrF and TER split into two phases:

1. **Per-segment sufficient statistics**: tokenize, extract n-grams, count clipped matches. Integer
   arithmetic, and where all the time goes.
2. **Score from summed statistics**: a handful of floating-point operations, once per corpus.

swadesh moves phase 1 to Rust and leaves phase 2 to sacreBLEU: `swadesh.BLEU` subclasses
`sacrebleu.BLEU` and overrides only `_extract_corpus_statistics`. Integers are exact, so if the
counts match, the corpus score is sacreBLEU's score *by construction*, with sacreBLEU's `BLEUScore`
object, verbose string and signature. Floating-point emulation is only needed on paths that compute
many scores natively: batch sentence scores, pairwise matrices, bootstrap resampling.

Four of the quirks that turned out to need exact reproduction:

1. **The Chinese tokenizer counts arrows, math and currency as Chinese.** sacreBLEU's `zh`
   tokenizer range table has the 2-character string `u'\u20000'` in it, so the range
   U+2001–U+2A6D (arrows, math, currency) matches. We reproduce the bug, and say so in the docs.
2. **CPython changed `sum()` over floats in 3.12**, and again in 3.14. So sacreBLEU's own scores differ
   between Python versions: on WMT22, 2,132 of 13,455 sentence-level BLEU scores (15.8%) differ
   between 3.11 and 3.12, by at most 9 ulps. We detect the interpreter and match it. My own check on
   the same 2,037-segment data: corpus BLEU is the same float (`0x1.507f36d3491a3p+6`) on 3.11, 3.12
   and 3.14, and 0 sentence scores differ. The drift table is in `docs/python-version-drift.md`.
3. **Bootstrap CI arithmetic is float32 numpy.** Materialising `stats_f32[idxs]` is where sacreBLEU's
   memory goes. We keep numpy for the random draws (so the indices are identical by construction)
   and sum the selected rows natively in the same float32 order; the per-row arithmetic is
   sacreBLEU's own `_compute_score_from_stats`.
4. **"lowercase" means Python's `str.lower()`**, tied to the interpreter's Unicode version, and
   `str.split()` splits on Python's whitespace set (which includes `\x1c`–`\x1f` and `\x85`, but not
   U+200B). We generate our tables from the installed Python and call back into CPython for case
   mappings that have moved across Unicode versions. Rust's std library was not good enough here.

The most interesting bug I hit was in my own bookkeeping, not the maths. `num_refs` in the score
signature is the *per-segment* count of non-None references, not `len(references)`. I "optimised"
it to the latter, and that is only equal when no reference stream contains `None`. On a warm cache
path it made `corpus_bleu(hypotheses, None)` raise `TypeError` where sacreBLEU returns a score, and
`num_refs` printed 5 where sacreBLEU printed 1. The parity suite caught it; it now has its own
regression test with `None` in the first, middle and only position, across BLEU, CHRF and TER. The
fix is a C-level scan for `None` on the common path and sacreBLEU's exact loop when there is one.
That is the shape of this work: the maths is exact long before the bookkeeping is.

## Does it agree with the incumbent on real model output?

That is the question that decides whether any of this is publishable, and only real output answers
it. On a single RTX PRO 6000 workstation (TROY-1), I generated translations with small models served
by Ollama, then scored the same hypotheses with sacreBLEU 2.6.0 and swadesh, comparing every corpus
score and *every sentence score* exactly.

Data: WMT24 English→German news, temperature 0, greedy decode.

| model | segments | BLEU | chrF | chrF++ | TER | corpus `==` | sentence diffs |
|---|---|---|---|---|---|---|---|
| `qwen2.5vl:7b` | 600 | 22.256488 | 52.564888 | 49.482473 | 68.130071 | all | **0 / 600** |
| `granite4:small-h` | 150 | 25.871423 | 58.480785 | 54.969857 | 62.555391 | all | **0 / 150** |
| `devstral:24b` | 150 | 26.958969 | 59.130336 | 55.705941 | 61.275234 | all | **0 / 150** |
| `qwen3.8-abliterated` | 150 | 26.407621 | 59.285038 | 55.861321 | 62.161497 | all | **0 / 150** |
| `nxkd1-k-q5` (my Naxi model, English→Naxi, scored vs. the English source) | 250 | 0.101294 | 11.884049 | 9.851019 | 111.973392 | all | **0 / 250** |

Across BLEU (default / intl / floor smoothing), chrF (default / beta=3), chrF++ and TER
(default / normalized) that is 24 scored configurations for the three 150-segment models alone,
24 identical corpus scores, and zero differing sentence scores in every row. The Naxi row matters
most to me: it is my worst-quality output, and if a scorer were going to disagree anywhere, this is
where it would.

They also rank identically. Of the three small models on the same 150 segments, devstral and
qwen3.8 sit 0.03 chrF apart — inside the ~1.4 chrF noise band from the Naxi work. So the honest
reading is "these two are tied", and both scorers say so with the same numbers.

One trap I want you to avoid: qwen3.8 is a reasoning model. With a small `num_predict` it returns
an empty `response` and puts everything into `thinking`, so the harness recorded 150 empty
hypotheses. Both scorers then returned a very tidy 0.00. Identical, and completely meaningless. If
you run an eval pipeline against a thinking model, assert that hypotheses are non-empty before you
believe any of it.

## The speed, measured on the shapes that caused the pain

All numbers below are from the same machine, same data, back to back, with the result compared
exactly each time. Sources: `bench/bench.py`, `bench/stress/`, and the extra harness I added for
this post.

Corpus scores, WMT22 en-de, 24-core Ryzen 9 7900X, sacreBLEU 2.6.0 on CPython 3.11.14, idle machine
(`bench/results/speed-2026-11-26-*.json`, 2026-11-26):

| case | sacreBLEU | swadesh | speedup | exact |
|---|---|---|---|---|
| BLEU corpus, 20,000 | 1.168 s | 0.0083 s | **140×** | yes |
| chrF corpus, 20,000 | 3.458 s | 0.0109 s | **316×** | yes |
| chrF++ corpus, 20,000 | 4.006 s | 0.0143 s | **280×** | yes |
| TER corpus, 10,000 | 6.406 s | 0.0552 s | **116×** | yes |
| BLEU corpus, 100,000 | 5.816 s | 0.0497 s | **117×** | yes |
| chrF corpus, 100,000 | 18.559 s | 0.0591 s | **314×** | yes |
| chrF++ corpus, 100,000 | 20.802 s | 0.0735 s | **283×** | yes |

Pinned to one core (`taskset -c 0`) the same 20,000-segment cases are **11–36×**; the rest is
parallelism, and segments are independent so it does not change a result.

Now the workloads that actually hurt in a fine-tuning loop. The first two are what the Naxi
promotion gate does: batch the sentence scores, then a paired test.

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

Same machine, back to back (`bench/results/extra-2026-11-26.json`); every row compared its result
exactly.

Three rows are honest negatives, and I left them in:

- **`PairedTest` is not faster at this size.** sacreBLEU's Python/numpy implementation is already
  good at 2,000 segments and 500 resamples; my version is exact but comes out level. It may still
  win when the resamples or the segment count are much larger, where sacreBLEU's tensor
  materialisation dominates, but I have not measured that here. Do not quote it as a win.
- **The CLI is at parity, not ahead.** Start-to-finish on 20,000 lines it is 0.9–1.0×: both tools
  pay the interpreter's import graph, and at that size the scoring is a few tens of milliseconds
  either way. It used to be 0.6–0.7×, because `python -m swadesh.cli` was an adapter that patched
  sacreBLEU and ran *its* parser; the console script now runs the compiled code in process, which
  removed that overhead. A longer run is dominated by scoring; a batch of short invocations is
  interpreter startup on both sides. If you are scripting one call per file, use the library.

Where it *is* dramatic is the inside-the-API workloads. Sentence-level TER is the funny one:
sacreBLEU takes about **2.6 seconds per sentence** on that data, so a 150-segment normalized-TER
comparison takes several minutes on one side and under a second on the other. That single number is
the difference between "I re-scored every arm tonight" and "I will re-score the day I have to
defend it".

At corpus scale it stops being a timing question. 1,000,000 segments (WMT22 en-de perturbed):

| | sacreBLEU | swadesh | exact |
|---|---|---|---|
| BLEU | 205 s, **9.0 GB** peak | 2.5 s, **0.97 GB** | identical score, `64.60594655357025` |
| chrF | **`MemoryError`; no score** (hit a 19.9 GB address-space cap) | 1.04 s, 1.06 GB | — |

At a million segments sacreBLEU's chrF runs out of memory where swadesh uses about 1 GB. To be
precise about what I measured: I ran sacreBLEU under a 19.9 GB address-space cap and it raised
`MemoryError`; it may well succeed on a machine with considerably more RAM, and I haven't tested
that. There is no score on the other side either way, so I am not calling this a speedup.

## What this means for anyone re-running a benchmark

The reason I think this is worth handing out is a narrow, boring property: **if the numbers are
bit-identical, then swapping the implementation cannot make anybody's published result
incomparable, and it can only make them wait less.**

For me that is literal. My promotion gate is 446 rows, chrF++ and NTQS, paired bootstrap clustered by
reference, 3000 resamples. Every candidate checkpoint goes through it. Between the sentence scores
for chrF++ and the resampling for the CI, the scorer is the slowest part of a decision I make
several times a day — and the decision is usually "this move is inside the noise band, keep going".
A scorer that returns the same values faster buys me that time back without adding a single new
rule to the ledger.

For everyone else the same argument holds with different numbers. If you score a reward per sample
in RL fine-tuning, or you do MBR decoding (n² sentence scores per input), or you filter millions of
sentence pairs, you are running the same Python loop. Bit-exactness is what makes it safe to swap.

## How it's tested

I don't think the speedup is the interesting claim; the exactness is, so here is what stands behind it.

- **Parity suite, 494 tests** (plus 3 MCP and 6 regression tests), comparing every public metric
  method, compat function, `Score`/`Signature` attribute, exception type and message, CLI flag and
  output, including pickling, `patch()`/`unpatch()` and forked multiprocessing.
- **Differential fuzzing** against sacreBLEU over an adversarial Unicode alphabet; the latest deep
  run was 20,000 cases per metric across all five BLEU tokenizers, chrF, chrF++ and TER, plus the
  long-sentence TER shifts.
- **Env matrix**: CPython 3.11, 3.12 and 3.14; numpy 2 on the VPS and the GPU box. The full parity
  suite passes on all three interpreters.
- **Real model output**, as above: qwen3.8-27b and llama3.3:70b re-verified on this build (12
  configurations each, 24/24 corpus scores bit-exact, 0/300 sentence scores differing), and five
  cached model outputs compared in the earlier pass.
- **My own data**: `python -m swadesh.verify -r ref.txt -i hyp.txt` runs the same comparison on
  yours and prints either "identical" or the first mismatch.

What is not yet verified: the GitHub Actions matrix has not run (the repo is private until the
release), and macOS/ARM runners are unexercised. The claims above are from x86-64 machines I own.

## Try it

```bash
pip install swadesh                      # once the vX.Y.Z release is on PyPI
python -c "import swadesh; swadesh.patch()"     # makes existing `import sacrebleu` fast
python -c "import swadesh as sacrebleu; print(sacrebleu.corpus_bleu(['the cat sat'], [['the cat sat down']]))"
```

If you already have code that calls `sacrebleu`, `swadesh.patch()` is the whole change. If you write
new code, `sentence_scores`, `pairwise` (MBR matrices) and the native bootstrap are there too.

The name comes from Morris Swadesh, the linguist whose core-vocabulary lists let researchers measure
how much two languages share. BLEU and chrF do the same for a translation and its reference, which
felt like the right name for a tool whose job is to be exactly comparable.

swadesh is AGPL-3.0-or-later, free for everyone; the commercial licence for the cases the AGPL
does not suit is free of charge for organisations under USD 5 million annual revenue, and
negotiated above that (`LICENSE.commercial`, apeters247@gmail.com).

*More from me: the Naxi project at [naxiai.com](https://naxiai.com), its benchmark at
[huggingface.co/Apeters247/naxi-mt-benchmark](https://huggingface.co/Apeters247/naxi-mt-benchmark),
and everything else under [huggingface.co/Apeters247](https://huggingface.co/Apeters247).*

*Thanks for reading. If you re-run this on your own data, `swadesh.verify` prints either "identical"
or the first mismatch — that report is more useful to me than a star on the repo.*