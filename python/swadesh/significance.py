"""Fast, bit-identical versions of sacreBLEU's bootstrap resampling and paired
significance tests (``sacrebleu.significance``).

The random draws use numpy exactly as sacreBLEU does, so they are identical.
What changes is the expensive part: sacreBLEU materializes
``stats_f32[idxs]``, an ``n_samples x n_segments x width`` float32 tensor
(hundreds of MB for a 5k-segment corpus), and reduces it with ``.sum(0)``.
We compute the same row sums natively in the same float32 summation order,
then hand each row to the metric's own ``_compute_score_from_stats``, so the
per-sample score arithmetic is still sacreBLEU's own code.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sacrebleu.significance as _sig
from sacrebleu.significance import Result, _compute_p_value, estimate_ci

from . import _swadesh

sacrelogger = logging.getLogger("sacrebleu")


def _seed_from_env() -> Tuple[str, Optional[int]]:
    seed = os.environ.get("SACREBLEU_SEED", "12345")
    return seed, (None if seed.lower() == "none" else int(seed))


def _resampled_sums(stats, idxs: np.ndarray) -> np.ndarray:
    """``np.array(stats, dtype='float32')[idxs].sum(1)``, computed natively."""
    s32 = np.ascontiguousarray(np.array(stats, dtype="float32"))
    n, width = s32.shape
    ix = np.ascontiguousarray(idxs, dtype=np.int64)
    buf = _swadesh.bootstrap_sums_f32(s32.tobytes(), width, ix.tobytes(), n)
    return np.frombuffer(buf, dtype=np.float32).reshape(ix.shape[0], width).copy()


def _bootstrap_resample(stats, metric, n_samples: int = 1000):
    """Drop-in for ``sacrebleu.significance._bootstrap_resample``."""
    seed, _seed = _seed_from_env()
    n = len(stats)
    if _seed is not None and 0 <= _seed < 2**128 and n > 0:
        # Native draw: exact emulation of numpy's default_rng(seed).choice(...)
        s32 = np.ascontiguousarray(np.array(stats, dtype="float32"))
        buf = _swadesh.bootstrap_sums_f32_seeded(s32.tobytes(), s32.shape[1], n, n_samples, _seed)
        sums = np.frombuffer(buf, dtype=np.float32).reshape(n_samples, s32.shape[1]).copy()
    else:
        rng = np.random.default_rng(_seed)
        idxs = rng.choice(n, size=(n_samples, n), replace=True)
        sums = _resampled_sums(stats, idxs)
    scores = [metric._compute_score_from_stats(_s) for _s in sums]
    return str(seed).lower(), scores


def _ar_shuffle(bl_stats, sys_stats, pos_sel):
    """``pos_sel @ bl + neg_sel @ sys`` and its mirror, exact for int stats.
    Returns None when the stats are not integral (falls back to numpy)."""
    bl = np.asarray(bl_stats)
    sy = np.asarray(sys_stats)
    if bl.dtype.kind not in "iu" or sy.dtype.kind not in "iu":
        return None
    bl = np.ascontiguousarray(bl, dtype=np.int64)
    sy = np.ascontiguousarray(sy, dtype=np.int64)
    sel = np.ascontiguousarray(pos_sel, dtype=np.uint8)
    n, width = bl.shape
    a, b = _swadesh.ar_sums_i64(bl.tobytes(), sy.tobytes(), width, sel.tobytes(), n)
    shape = (sel.shape[0], width)
    return (np.frombuffer(a, dtype=np.int64).reshape(shape),
            np.frombuffer(b, dtype=np.int64).reshape(shape))


def _paired_ar_test(baseline_info, sys_name, hypotheses, references, metrics,
                    n_samples=10000, n_ar_confidence=-1, seed=None):
    """Drop-in for ``sacrebleu.significance._paired_ar_test``."""
    rng = np.random.default_rng(seed)
    pos_sel = rng.integers(2, size=(n_samples, len(hypotheses)), dtype=bool)
    neg_sel = ~pos_sel
    if n_ar_confidence > 0:
        bs_idxs = rng.choice(len(hypotheses), size=(n_ar_confidence, len(hypotheses)), replace=True)

    results = {}
    for name, metric in metrics.items():
        bl_stats, bl_result = baseline_info[name]
        sacrelogger.info(f'Computing {name} for {sys_name!r} and extracting sufficient statistics')
        sys_stats = metric._extract_corpus_statistics(hypotheses, references)
        sys_score = metric._aggregate_and_compute(sys_stats)
        diff = abs(bl_result.score - sys_score.score)
        sacrelogger.info(f' > Performing approximate randomization test (# trials: {n_samples})')
        fast = _ar_shuffle(bl_stats, sys_stats, pos_sel)
        if fast is None:
            shuf_a = pos_sel @ bl_stats + neg_sel @ sys_stats
            shuf_b = neg_sel @ bl_stats + pos_sel @ sys_stats
        else:
            shuf_a, shuf_b = fast
        scores_a = np.array([metric._aggregate_and_compute(x).score for x in shuf_a[:, None]])
        scores_b = np.array([metric._aggregate_and_compute(x).score for x in shuf_b[:, None]])
        p = _compute_p_value(np.abs(np.array(scores_a) - np.array(scores_b)), diff)
        res = Result(sys_score.score, p)
        if n_ar_confidence > 0:
            sacrelogger.info(f' > Performing bootstrap resampling for confidence interval (# resamples: {n_ar_confidence})')
            sums = _resampled_sums(sys_stats, bs_idxs)
            sys_scores = np.array([metric._compute_score_from_stats(_s).score for _s in sums])
            res.mean, res.ci = estimate_ci(sys_scores)
        results[name] = res
    return sys_name, results


def _paired_bs_test(baseline_info, sys_name, hypotheses, references, metrics,
                    n_samples=1000, n_ar_confidence=-1, seed=None):
    """Drop-in for ``sacrebleu.significance._paired_bs_test``."""
    rng = np.random.default_rng(seed)
    results = {}
    idxs = rng.choice(len(hypotheses), size=(n_samples, len(hypotheses)), replace=True)
    for name, metric in metrics.items():
        bl_stats, bl_result = baseline_info[name]
        sacrelogger.info(f'Computing {name} for {sys_name!r} and extracting sufficient statistics')
        sys_stats = metric._extract_corpus_statistics(hypotheses, references)
        sys_score = metric._aggregate_and_compute(sys_stats)
        diff = abs(bl_result.score - sys_score.score)
        sacrelogger.info(f' > Performing paired bootstrap resampling test (# resamples: {n_samples})')
        scores_bl = np.array([metric._compute_score_from_stats(_s).score for _s in _resampled_sums(bl_stats, idxs)])
        scores_sys = np.array([metric._compute_score_from_stats(_s).score for _s in _resampled_sums(sys_stats, idxs)])
        sys_mean, sys_ci = estimate_ci(scores_sys)
        sample_diffs = np.abs(scores_sys - scores_bl)
        stats = sample_diffs - sample_diffs.mean()
        p = _compute_p_value(stats, diff)
        results[name] = Result(sys_score.score, p, sys_mean, sys_ci)
    return sys_name, results


_FAST = {
    "_bootstrap_resample": _bootstrap_resample,
    "_paired_ar_test": _paired_ar_test,
    "_paired_bs_test": _paired_bs_test,
}


def install():
    """Replace sacreBLEU's significance functions globally; returns the originals."""
    old = {k: getattr(_sig, k) for k in _FAST}
    for k, v in _FAST.items():
        setattr(_sig, k, v)
    return old


def uninstall(old):
    for k, v in old.items():
        setattr(_sig, k, v)


@contextlib.contextmanager
def _installed():
    if getattr(_sig, "_bootstrap_resample") is _bootstrap_resample:
        yield
        return
    old = install()
    try:
        yield
    finally:
        uninstall(old)


class PairedTest(_sig.PairedTest):
    """Drop-in ``sacrebleu.significance.PairedTest`` using the fast kernels.
    Results are identical to sacreBLEU's."""

    def __init__(self, *args, **kwargs):
        with _installed():
            super().__init__(*args, **kwargs)
            self._fn = _paired_ar_test if self.test_type == "ar" else _paired_bs_test

    def __call__(self):
        with _installed():
            return super().__call__()
