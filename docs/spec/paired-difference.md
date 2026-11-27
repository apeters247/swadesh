# Paired difference: the signed gap, its interval, and a power estimate

Status: additive API (shortlist item, alongside `docs/feature-ideas.md`), written 2026-11-27 from the
existing paired-bootstrap machinery in `python/swadesh/significance.py`. No score changes; the
p-value is the one `PairedTest` already reports.

## Why it exists

`PairedTest` answers "is system B different from A?" with a p-value, and it has the *distribution* of
per-resample differences in hand - it uses `|scores_B - scores_A|` to build that distribution and then
throws the sign away. The sign is the thing a user wants: **how much** better, and **how sure**.
`paired_difference()` keeps the signed distribution, so it reports the gap and its interval from the
same draws that produced the p-value. It then uses the spread of that distribution for the one number
everyone asks next: "how many segments would I need to see this reliably?"

## What ships

`swadesh.paired_difference(baseline, system, references, metric=None, n_samples=1000)` returns a
`Difference` for one metric (default `BLEU()`):

| field / method | meaning |
|---|---|
| `delta` | `system_score - baseline_score`, signed, in the metric's units |
| `ci_low`, `ci_high` | the 95% interval of the signed difference, same order-statistic rule as `estimate_ci` |
| `p_value` | **identical** to `PairedTest`'s for the same systems, metric, seed and `n_samples` |
| `se` | the bootstrap standard error of the difference (the resample SD of the signed difference) |
| `n_segments` | how many segments the test saw |
| `significant(alpha=0.05)` | `p_value < alpha` **and** the 95% interval excludes zero (see below) |
| `detectable(power=0.8, alpha=0.05)` | the smallest gap detectable at `n_segments` for that power |
| `required_n(effect, power=0.8, alpha=0.05)` | segments needed to detect `effect` in the paired test |
| `to_dict()` | JSON-able fields |

## The p-value is not re-derived

The draws and the p-value are `_paired_bs_test`'s exactly: the same resample indices (native when the
seed is an integer), the same `_paired_sums`/`_f32_scores`, the same
`_compute_p_value(|d| - mean(|d|), |delta|)`. `paired_difference` differs only in *keeping* the signed
`scores_sys - scores_bl`. A test pins the p-value against `PairedTest` on the same seed, so the two
cannot drift.

## The power numbers, and their assumptions

Standard normal approximation for a two-sided test (textbook formulae; no new dependency,
`statistics.NormalDist` supplies the quantiles):

    z = NormalDist().inv_cdf(1 - alpha/2) + NormalDist().inv_cdf(power)
    detectable(power)   = z * se                        # the gap at the edge of detection
    required_n(effect)  = ceil(n_segments * (z * se / effect) ** 2)

`se` is the bootstrap SE of the difference at the observed `n_segments`, and it is taken as the
estimate of `sd_segment / sqrt(n_segments)`, i.e. `sd_segment` is assumed to be well estimated by the
sample and the segments are assumed independent and identically distributed. That is what makes the
`sqrt(n)` scaling usable. The result is an **estimate under those assumptions**, not a promise: heavy
segment dependence (e.g. document-level correlation) inflates the true requirement, and the function
documentation says so. `required_n` is reported in whole segments and is never below 1.

## The floor p-value, and why `significant` also reads the interval

`_compute_p_value` is `(c + 1) / (n_samples + 1)` - both `+1`s are sacreBLEU's finite-resample
correction, and swadesh's `PairedTest` keeps them for parity - so the smallest possible p-value is
`1 / (n_samples + 1)`. **Measured 2026-11-27:** a system against itself with `n_samples=50` reports
`p = 1/51 = 0.0196`, below `alpha = 0.05`. The p-value alone would call a zero difference significant
whenever `n_samples < 20 / alpha`, i.e. for any default-sized run.

A paired *test* must reproduce that (it is the oracle's arithmetic). A new convenience need not
inherit it, so `Difference.significant` requires **both** the p-value and the interval: the identical
case has the interval `[0, 0]`, which excludes nothing, so it is not significant. Where the difference
is real the two always agree; the interval is only ever the tie-breaker at the floor.

## Invariants (each asserted by a test)

1. `delta == system.corpus_score(...).score - baseline.corpus_score(...).score`, exactly.
2. `p_value` equals `PairedTest(...)` run on the same systems, metric, seed and `n_samples`.
3. `ci_low <= delta <= ci_high` whenever the interval brackets the estimate (`ci_low < ci_high`).
4. Swapping baseline and system negates `delta` and leaves `p_value` unchanged (the test is two-sided
   in `|d|`).
5. `required_n(detectable(power), power) == n_segments` (the two are inverses at the same `se`).
6. `required_n` is non-increasing in `effect` and returns an int `>= 1`; a larger gap needs fewer
   segments.
7. A system against itself gives `delta == 0`, the floor p-value `1 / (n_samples + 1)`, an interval
   `[0, 0]`, and is reported as **not** significant (the interval excludes nothing).
