# significance: standalone paired significance testing

Phase 5, first module. Written before the implementation.

## What has to change

`python/swadesh/significance.py` reimplements the bootstrap and approximate-randomisation resampling
natively, but still imports four names from `sacrebleu.significance` and subclasses its
`PairedTest`. The goal is that `patch()` is the only place swadesh touches sacreBLEU. This module is
the smallest of the five to make standalone: three small symbols and one base class.

## The three symbols

### p-value

Given the per-resample score differences and the real difference between the two systems:

    c = count of resamples whose difference exceeds the real difference
    p = (c + 1) / (n + 1)

Strict `>` matters: a system compared with itself must not be attributed to chance. The `+1` in
both places keeps the probability of falsely rejecting the null hypothesis no greater than the
chosen level, which is the standard correction for a finite number of shufflings.

The real difference is compared against resamples of the *same* statistic; the two systems are
swapped in half the resamples, which is why the comparison is one-sided.

### confidence interval

Sort the resampled scores, take the element at `n // 40` and at `n - (n // 40) - 1`, and report
`mean(scores)` with half-width `0.5 * (upper - lower)`. `n // 40` is the 2.5th percentile boundary
for a two-sided 95% interval, expressed as an index rather than a quantile interpolation.

### the result record

A small value object carrying the p-value, the mean and the confidence interval, with the same
`__repr__` shape, because the CLI prints it.

**Measured 2026-11-24: each field is interpolated with `str`, not with an f-string.** The
distinction only shows on a `np.float32`, whose two renderings differ in NumPy 2: for
`v = np.float32(20.834093)`, `str(v)` is `20.834093` but `f"{v}"` is `20.83409309387207` (the f-string
goes through `__format__`, which widens to double). The oracle prints `mean=20.834093`, so the
record's fields must be formatted with `str`. For a `np.float64` the two agree (`str(np.float64(2.5))`
= `f"{np.float64(2.5)}"` = `2.5`), which is why the earlier repr change looked complete.

## The base class

`PairedTest` takes named systems (each a list of hypothesis segments), a list of reference streams,
a set of metrics and a test type (`bs` or `ar`), then for each metric:

1. scores every system on the real corpus,
2. resamples segment indices `n_samples` times — with replacement for `bs`, as a stratified shuffle
   for `ar` — swapping the systems in each resample with probability one half,
3. computes the per-resample difference, then the p-value and confidence interval above,
4. returns a signature per metric and a score per system.

swadesh's version already implements steps 2 and 3 with native kernels; what it needs from
sacreBLEU is the scaffolding around them.

## Plan

1. Port the three symbols into `python/swadesh/significance.py`, written from the definitions above
   rather than transcribed, and keep the existing native resampling.
2. Make `PairedTest` a plain class in this module.
3. Remove the `sacrebleu.significance` import and drop `significance.py` from the provenance
   allowlist.
4. Verify against `tests/test_parity.py`, which already compares significance output with the
   oracle byte for byte, plus the frozen fixtures.

## Scope note

This is one of five modules. `metrics.py`, `cli.py`, `verify.py` and `__init__.py` follow; `metrics.py`
is the largest because it owns the score objects and the aggregate step.