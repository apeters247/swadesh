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

**Measured 2026-11-27: the premise is false, and the ergonomic half shipped instead.** The stream
cannot be shared. chrF is character-based - it never calls BLEU's tokenizer, and handing it BLEU's
tokenized text would change its score, which is the one thing this package must not do. On 20,000
WMT22 en-de segments (troy-1, `~/venv311`, one process, warm), 5 systems with BLEU + chrF++ each:

    fresh metric, references passed each call    123.6 ms
    one metric each, references cached once      124.4 ms     (-1%, the wrong direction)

Reference preparation is about 1% of the work; the native statistics are the whole cost. So idea 2's
speed claim is withdrawn and `swadesh.evaluate()` ships as the convenience it actually is: one call,
one dict keyed by each metric's own name, identical numbers.

## 3. `explain()` on TER (1 day)

TER already computes the alignment and the accepted shifts inside `ter_clean.rs`. Exposing them —
`(edits, shifts_taken, aligned_pairs)` per segment — gives a per-segment answer to "why did this
score badly", which is what users actually want when a system underperforms. sacreBLEU offers only
the number.

**Shipped 2026-11-27.** `TER.explain(hyps, refs)` returns one dict per segment: `ter` (identical to
`sentence_scores`), `edits`, `shifts`, `distance`, `ref_len` (the mean length), `best_ref`/
`best_ref_len`, `hyp_len`, `n_refs`, `counts` (match/sub/ins/del of the final alignment) and
`accepted_shifts`. One Rust pass over the same search, no second implementation, no Python fallback;
the record and its invariants are `docs/spec/ter-explain.md`.

## 4. Streaming accumulator (1-2 days)

`Accumulator.add(hypothesis, references)` with `.score()` at any point. The statistics are additive
`i64`/`f64` rows (`corpus_stats` already takes them as a slice), so this is a thin state object.
The stress report records sacreBLEU needing 40-46 GB at 1M segments and OOM-ing on chrF at 19.9 GB;
swadesh completed that size in 1.04 s, and streaming would make the peak independent of corpus size.

**Shipped 2026-11-27.** `swadesh.Accumulator(metric)` keeps exactly one statistics row: `add(hyp,
refs)` per segment and `score()` at any point. It rests on `_aggregate_and_compute`'s contract (a
corpus score is the element-wise sum of the per-segment rows in arrival order), so a streamed score
is bit-identical to `corpus_score`. Bootstrapped intervals resample segments and so need the rows:
`keep_rows=True` keeps them, the default holds the sum and `score(n_bootstrap>1)` raises instead of
returning an interval it cannot compute. Record and invariants: `docs/spec/accumulator.md`.

## 5. `n_jobs` on every corpus call (1 day)

`PairedTest(n_jobs=)` already exists and the crate has a `parallel` feature (rayon). Extending the
same knob to `corpus_score` / `sentence_scores` turns the 24-core machine into roughly a 20x corpus
scorer, far beyond the single-thread speed-up already measured.

**Shipped 2026-11-27, premise corrected by measurement.** The corpus kernels were *already* parallel
(`swadesh-core`'s default `parallel` feature). On troy-1, 40,000 segments x 25 words, `taskset`
capping the CPUs: TER ~9x and BLEU ~6x at 16 CPUs, **identical scores at every count**. So the knob
does not turn parallelism on; it chooses the thread count, which already defaults to the machine.
`swadesh.n_jobs(k)` (context manager) / `set_n_jobs(k)` / `get_n_jobs()` size that pool.
`PairedTest(n_jobs=)` is untouched: it spawns processes, this sizes threads inside one process. Spec:
`docs/spec/n_jobs.md`.

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

## Shortlist

Five smaller additions proposed alongside the eight above.

| | item | status |
|---|---|---|
| s1 | `evaluate.load("swadesh")` compatibility (HuggingFace `evaluate`) | pending |
| s2 | Paired CI on the *difference* + power analysis (`swadesh.paired_difference`) | shipped 2026-11-27 |
| s3 | Native `ja-mecab` / `sentencepiece` tokenizers | pending (dependency decision) |
| s4 | A pytest plugin + a GitHub Action | shipped 2026-11-27 (Action already existed; the plugin is new) |
| s5 | Streaming accumulator | shipped as item 4 |

## Shipped

| | feature | landed |
|---|---|---|
| 1 | `TER.sentence_scores` and `TER.pairwise` | 2026-11-27 |
| 7 | `strict=True` input validation (`swadesh.strict()`, `SWADESH_STRICT`) | 2026-11-27 |
| 8 | Seeds in every signature (`swadesh.seed()`, `set_seed()`) | 2026-11-27 |
| 6 | `swadesh.compare(systems, refs)` | 2026-11-27 |
| 2 | `swadesh.evaluate(hyps, refs, metrics=[...])` | 2026-11-27 |
| 3 | `TER.explain(hyps, refs)` | 2026-11-27 |
| 4 | `swadesh.Accumulator(metric)` (streaming) | 2026-11-27 |
| 5 | `swadesh.n_jobs(k)` (worker-thread count) | 2026-11-27 |

## Not proposed

* Anything that changes a score. Parity is the product; the value here is that these features sit on
  top of an implementation that is provably identical to the reference on the measured corpora.
* A Python fallback for the native kernels. It would be slower and would exist only to duplicate
  tested code.
