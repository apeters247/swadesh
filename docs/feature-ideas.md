# Feature ideas beyond sacreBLEU

These are proposals, not commitments and not part of parity. Each one is grounded in something the
tree already has, so the effort estimates are about wiring, not research. Ordered by value per day.

## 1. `TER.sentence_scores` and `TER.pairwise` (hours)

`sentence_scores` and `pairwise` exist for `BLEU` and `CHRF` but not for `TER`, so the three metrics
are not interchangeable in a multi-metric harness. The `ter_corpus_stats` kernel and the `pairwise`
machinery are already there; this is a wrapper plus tests. (Found while hunting divergences: our own
API, not a parity gap.)

## 2. One-pass multi-metric `evaluate()` (1 day)

`swadesh.evaluate(hyps, refs, metrics=[BLEU, chrF, TER])` returning a dict of score objects. Today
each metric tokenizes the corpus separately; sharing one tokenized stream across BLEU and chrF (both
`13a`/`intl` users) removes two of the three tokenization passes. On the 1,000-segment corpus that
is the difference between 11 + 19 + 8 ms and roughly one pass plus the statistics.

## 3. `explain()` on TER (1 day)

TER already computes the alignment and the accepted shifts inside `ter_clean.rs`. Exposing them —
`(edits, shifts_taken, aligned_pairs)` per segment — gives a per-segment answer to "why did this
score badly", which is what users actually want when a system underperforms. sacreBLEU offers only
the number.

## 4. Streaming accumulator (1-2 days)

`Accumulator.add(hypothesis, references)` with `.score()` at any point. The statistics are additive
`i64`/`f64` rows (`corpus_stats` already takes them as a slice), so this is a thin state object.
The stress report records sacreBLEU needing 40-46 GB at 1M segments and OOM-ing on chrF at 19.9 GB;
swadesh completed that size in 1.04 s, and streaming would make the peak independent of corpus size.

## 5. `n_jobs` on every corpus call (1 day)

`PairedTest(n_jobs=)` already exists and the crate has a `parallel` feature (rayon). Extending the
same knob to `corpus_score` / `sentence_scores` turns the 24-core machine into roughly a 20x corpus
scorer, far beyond the single-thread speed-up already measured.

## 6. `compare(systems, refs)` (hours to 1 day)

One call returning per-system scores, paired-bootstrap p-values, confidence intervals and a ranking
table. Every piece exists (`PairedTest`, `estimate_ci`, the native resamplers); today a user has to
assemble it, and the assembly is exactly where sacreBLEU's APIs are most awkward.

## 7. `strict=True` input validation (hours)

sacreBLEU silently truncates when a reference stream is shorter than the hypothesis list (observed
while building the hunt harness: a 1-element stream against 6 hypotheses produced a score rather
than an error). A strict mode that raises with the offending index prevents publishing a number
computed from the wrong segments.

## 8. Seeds in every signature (hours)

BLEU and chrF signatures carry `seed`; every bootstrapped path should, so a reported number is
reproducible from its own output line. The RNG (PCG64-compatible) and the seed plumbing already
exist.

## Shipped

| | feature | landed |
|---|---|---|
| 1 | `TER.sentence_scores` and `TER.pairwise` | 2026-11-27 |

## Not proposed

* Anything that changes a score. Parity is the product; the value here is that these features sit on
  top of an implementation that is provably identical to the reference on the measured corpora.
* A Python fallback for the native kernels. It would be slower and would exist only to duplicate
  tested code.
