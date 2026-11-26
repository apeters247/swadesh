"""The Rust `estimate_ci` must agree with the arithmetic sacreBLEU uses.

`sacrebleu.metrics.base.Score.estimate_ci` takes `statistics.mean` of the resample scores and
half the 1/40..39/40 order-statistic spread. CPython's `statistics.mean` sums floats exactly, so
any floating-point sum can differ from it by an ulp; what has to match is the printed value,
which is the mean and the interval at `width` decimals. The exact interval is checked too.
"""

import random
import statistics

import numpy as np
import pytest

from swadesh import _swadesh


def reference(scores):
    s = sorted(scores)
    n = len(s)
    li = n // 40
    ui = n - li - 1
    return statistics.mean(scores), 0.5 * (s[ui] - s[li])


@pytest.mark.parametrize("n", [40, 50, 100, 1000, 10000])
def test_matches_cpython_mean_and_interval(n):
    rng = random.Random(n)
    for kind in range(4):
        if kind == 0:
            a = [rng.uniform(0.0, 100.0) for _ in range(n)]
        elif kind == 1:
            a = [float(rng.randint(0, 5)) for _ in range(n)]
        elif kind == 2:
            a = [3.5] * n
        else:
            a = [rng.uniform(0.0, 100.0) for _ in range(n)]
            a[rng.randrange(n)] = rng.uniform(0.0, 1e-3)
        want_mean, want_ci = reference(a)
        got_mean, got_ci = _swadesh.estimate_ci(list(a))
        assert f"{got_mean:.2f}" == f"{want_mean:.2f}", (kind, n, got_mean, want_mean)
        assert got_ci == want_ci, (kind, n, got_ci, want_ci)


def test_small_and_single_element_samples():
    for a in ([7.5], [1.0, 2.0], [3.0, 1.0, 2.0]):
        want_mean, want_ci = reference(a)
        got_mean, got_ci = _swadesh.estimate_ci(list(a))
        assert got_mean == want_mean
        assert got_ci == want_ci
