# Python-version drift in sacreBLEU 2.6.0 BLEU scores

## Summary

For identical inputs, sacreBLEU 2.6.0 BLEU scores differ between CPython 3.11 and
CPython 3.12. The cause is a change in the builtin `sum()` over floats, which
sacreBLEU uses inside `compute_bleu`. In our measurement on 13,455 perturbed
WMT22 segments, 2,132 sentence-level BLEU scores (15.8%) differ, by at most 9 ulps
(8.5e-14 absolute). Two of seven corpus-level BLEU scores also differ, by about
4e-14 and 7e-14 absolute. Between 3.12, 3.13 and 3.14 we observed no differences
on this data.

This is a reproducibility issue, not a correctness bug in sacreBLEU's formula.
Anyone who compares BLEU numbers across machines or Python versions should pin the
interpreter, or report the Python version alongside the score.

## Mechanism

`sacrebleu/metrics/bleu.py` computes the geometric mean with
`sum([my_log(p) for p in precisions[:eff_order]])`, where `my_log` returns
`-9999999999` (an int) for zero precision and `math.log(p)` (a float) otherwise.
This is a builtin `sum()` over a short list of mixed float and int values.

CPython's `builtin_sum_impl` (`Python/bltinmodule.c`) changed as follows:

| CPython | Float items | Int items in the float phase |
|---|---|---|
| <= 3.11 | naive `f += x` | naive |
| 3.12, 3.13 | Neumaier compensated sum | naive, compensation folded in at the end |
| >= 3.14 | compensated `(hi, lo)` accumulator | compensated, same as floats |

We confirmed these rules empirically, not only from the source. A pure-Python
transcription of each algorithm matched builtin `sum()` on random mixed lists with
0 mismatches out of 200,000 on the matching interpreter (naive on 3.10, Neumaier on
3.12 and 3.13, compensated ints on 3.14). The algorithm for one version group
mismatched on the interpreters of the other groups: naive mismatched 24,404 times on
3.12, Neumaier mismatched 24,404 times on 3.10, and compensated ints mismatched
16,115 times on 3.12 and 3.13.

## Measurement

Script: `tools/py_version_drift.py` (`score` then `compare`).

- Data: `wmt22.{cs-en,de-en,en-de,en-ja,en-ru,en-zh,zh-en}.ref`, 13,455 reference
  lines in total.
- Hypotheses: each reference with deterministic word drops (p = 0.15) and adjacent
  swaps (p = 0.10), seeded (`SEED = 20221122`). Only `random.random()` is used,
  so the hypotheses are identical on every interpreter.
- Scores: sentence-level `BLEU().sentence_score` for every segment, and
  `BLEU().corpus_score` per language pair, using sacreBLEU 2.6.0 defaults (13a, exp
  smoothing). Each score is recorded as `float.hex()`, so the comparison is exact.
- Interpreters: CPython 3.10.19, 3.11.14, 3.12.12, 3.13.12 and 3.14.3 (from uv),
  each in a fresh venv with `sacrebleu==2.6.0`.

Results (`compare`, exact comparison):

| Pair of interpreters | Sentence scores differing | Max ulps | Max abs. diff | Corpus scores differing |
|---|---|---|---|---|
| 3.10 vs 3.11 | 0 / 13,455 | 0 | 0 | 0 / 7 |
| 3.11 vs 3.12 | 2,132 / 13,455 (15.8%) | 9 | 8.5e-14 | 2 / 7 |
| 3.12 vs 3.13 | 0 / 13,455 | 0 | 0 | 0 / 7 |
| 3.13 vs 3.14 | 0 / 13,455 | 0 | 0 | 0 / 7 |
| 3.10 vs 3.14 | 2,132 / 13,455 | 9 | 8.5e-14 | 2 / 7 |

Corpus scores that differ (3.11 vs 3.12): en-ru 53.741225862632255 vs
53.741225862632300 (abs. diff 4.3e-14), and en-zh 69.41499584367152 vs
69.41499584367145 (abs. diff 7.1e-14). The other five pairs are identical to the
last bit.

## Scope and limits

- One sacreBLEU version (2.6.0), one tokenizer (13a), one smoothing setting, and one
  perturbation scheme. We did not vary these, so the percentages describe this
  setup only. Other data or settings will differ in rate and magnitude.
- The 3.14 change did not alter any score on this data. The sum-semantics check
  (`tools/check_sum_semantics.py`) does show a difference for 3.14 on random inputs,
  so 3.14 drift is possible. It was not observed here.
- The corpus-level differences are small (about 1e-15 relative), but they are real
  differences in the last bits of a reported score. They will matter if a score is
  compared for exact equality, or is rounded at a boundary.
- `swadesh` matches sacreBLEU on each interpreter. Its corpus scores are computed by
  sacreBLEU's own Python code on the host interpreter, so they follow the same
  interpreter-dependent behaviour. Its `sentence_scores` reproduce the same values by
  emulating `sum()` per interpreter (`SumMode`).

## Reproduce

```
python tools/py_version_drift.py score --out run-py311.json   # on CPython 3.11, with sacrebleu==2.6.0
python tools/py_version_drift.py score --out run-py312.json   # on CPython 3.12
python tools/py_version_drift.py compare run-py311.json run-py312.json
```

Set `SWADESH_WMT22_DIR` to the directory containing the `wmt22.*.ref` files if it is
not `bench/data/`.
