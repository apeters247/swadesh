# metrics: the Python metric classes, standalone

Phase 5, the largest module. Written before the implementation.

## What `metrics.py` uses from sacreBLEU today

swadesh's `BLEU`, `CHRF` and `TER` subclass sacreBLEU's classes, which supplies five things.
`corpus_score`, `sentence_score` and the segment statistics are already swadesh's own; the
inheritance buys:

| inherited | lines | what it does |
|---|---|---|
| `_cache_references` | 36 | turns reference streams into per-segment cached info, and sets `num_refs` |
| `_extract_reference_info` | 26 / 22 / 14 | per-metric reference preparation (lowercasing, tokenising) |
| `_aggregate_and_compute` | 7 each | turns the summed statistics row into a score object |
| `Score` | 93 | the score value object: fields, `format`, `__str__`, `estimate_ci`, signature plumbing |
| `BLEUScore` / `CHRFScore` / `TERScore` | 30 / 18 / 12 | per-metric `format` and `__str__` |

About 236 lines in total, which matches the plan's estimate of a day.

## Reference caching

For each segment, gather that segment's references across the streams, **drop the `None` entries**,
count how many remain, preprocess each with the metric's segment preprocessor, and store the
per-metric reference info. The set of counts across segments becomes `num_refs`: a single value when
every segment has the same count, and `-1` when they vary. That `-1` is what the signature prints as
`nrefs:2` or a varying marker, and it is the rule that earlier made `None` handling look
inconsistent — `None` is removed *before* counting, so a two-stream corpus where one stream is all
`None` for a segment has one reference there, not two.

## Reference info, per metric

* **BLEU** — lowercase unless case-sensitive, then tokenise and split. Kept per reference so the
  clipping can take the maximum over references.
* **chrF** — character sequence (whitespace kept or removed per the option) and, when `word_order > 0`,
  the word-id sequence after punctuation splitting.
* **TER** — the TER tokenisation of the reference.

## Aggregate

Each metric turns the summed statistics row into a score object:

* **BLEU** — geometric mean of the precisions with the brevity penalty, using the smoothing that was
  requested; `Score` carries `counts`, `totals`, `precisions`, `bp`, `sys_len`, `ref_len`.
* **chrF** — the f-score over the per-order precision/recall, with `beta` and effective-order
  smoothing as configured.
* **TER** — `edits / reference length`, as a percentage.

The float arithmetic must follow the running interpreter's summation semantics, which
`SumMode::for_python` already provides in Rust; the Python aggregate must use the same order.

## The score object

`Score` holds `name`, `score`, `_mean`, `_ci`, `_verbose`. `format(width, score_only, signature)`
produces the three output shapes the CLI prints — text with the signature, bare score, and the
verbose breakdown — and `__str__` is the text form with a default width. `estimate_ci(scores)` sets
`_mean` and `_ci` from the bootstrap samples. The signature is a list of `(long, short, value)`
triples rendered as `long:value|…`.

Because the CLI's byte output depends on all of this, the parity tests for text, JSON and LaTeX
formats are the verification, not a unit test of the formatting alone.

## Plan

1. Write `_cache_references`, `_extract_reference_info` × 3, `_aggregate_and_compute` × 3, `Score`
   and the three `*Score` classes in `python/swadesh/metrics.py`.
2. Flip the three classes to inherit from a local base rather than sacreBLEU's.
3. Remove `metrics.py` from `PENDING` in the provenance test.
4. Verify against the frozen oracle and the full parity suite, on 3.11/3.12/3.14.

## Risk

The formatting is where the risk is: 500+ parity assertions compare the exact strings, including
JSON key order and whitespace. I expect to iterate on the format methods against the oracle's output
rather than get them right first time, which is the same pattern as the TER and intl work.