# Streaming accumulator: constant-memory corpus scoring

Status: additive API (idea 4 of `docs/feature-ideas.md`), written 2026-11-27 from the existing
aggregation contract in `python/swadesh/metrics.py`. No score changes; nothing here is oracle-facing.

## Why it exists

`corpus_score` materializes one statistics row per segment (`n × width`) and sums them at the end.
At 1M segments that is the peak-RSS driver the stress report records (sacreBLEU 8.8-19.9 GB; swadesh
~1 GB, but still O(n)). A corpus scored as it arrives needs only the running sum: O(width).

## The contract it must not break

`metrics._aggregate_and_compute` sums a **list of rows** with `sum_of_lists`, which iterates the rows
in order and does `total[i] += row[i]`, seeded from the first row's element type so integer rows stay
integer. Therefore a running sum that adds each row in arrival order is **exactly** the batch sum:
same operations, same order, same type. The accumulator is that running sum and nothing else.

## API

```
from swadesh import Accumulator
acc = Accumulator(BLEU())
for hyp, refs in stream:
    acc.add(hyp, refs)              # one segment; refs is that segment's list of references
print(acc.score())                  # the corpus score over everything added so far
acc.add_corpus(hyps, refs)          # or add a whole batch; still O(width)
acc.n_segments                      # how many segments have been added
```

* `Accumulator(metric, keep_rows=False)` - `metric` is a metric instance (BLEU/CHRF/TER).
* `.add(hypothesis, references=None)` - one segment; `references` is a **per-segment** list (the
  `sentence_score` shape), or omitted when the metric was built with `references=`.
* `.add_corpus(hypotheses, references)` - the corpus shape, for convenience and batching.
* `.score(n_bootstrap=1)` - the metric's own score object for the running sum, via the metric's
  `_score_from_row`, so it is bit-identical to `corpus_score` over the concatenated input.
* `.n_segments` - the count added.

## The one honest limit: bootstrapped confidence intervals

`score(n_bootstrap>1)` resamples **segments**, so it needs the per-segment rows, not the sum
(`significance._bootstrap_resample` reads `len(stats)` and resamples rows). Streamed scoring therefore
cannot produce a bootstrap CI without retaining the rows. The accumulator says so instead of guessing:

* `keep_rows=False` (default): `score(n_bootstrap>1)` raises `ValueError` naming the setting.
* `keep_rows=True`: rows are retained and `score(n_bootstrap>1)` behaves exactly like `corpus_score`.

Constant memory is the point, so the default is `keep_rows=False`; a bootstrapped CI is the reason to
pay for the rows.

## Invariants (each asserted by a test)

1. `Accumulator(m)` fed every segment equals `m.corpus_score(hyps, refs)`, for BLEU, chrF and TER.
2. Incremental `add` calls equal one `add_corpus` call, and equal a different segmentation of the
   same stream (the sum is order-preserving but segment-boundary independent).
3. `keep_rows=False` holds exactly one row of state, independent of `n_segments` (peak memory does not
   grow with the corpus).
4. `score(n_bootstrap>1)` raises under the default and equals `corpus_score(..., n_bootstrap)` under
   `keep_rows=True`, seed included.
5. `swadesh.strict()` still raises on a length mismatch inside `add_corpus`.
6. An empty accumulator takes the metric's own empty-input path: `score()` on it raises the same
   `IndexError` `corpus_score([], [[]])` raises (sacreBLEU raises there too). "" is not a corpus.
