"""Paired significance testing, written for swadesh.

The bootstrap and approximate-randomisation resampling are native (the kernels live in
`_swadesh`); `Result`, the p-value, the confidence interval and the `PairedTest` manager are defined
in this module from `docs/spec/significance.md`, written before the code.

One sacreBLEU import remains and will not go away: `install()`/`uninstall()` is the adapter that
`patch()` calls to redirect sacreBLEU users to these functions, so it must touch sacreBLEU to
redirect it. The native scoring helpers used to import `sacrebleu.metrics` classes as well; they now
classify a metric by the attributes the kernels read (`_metric_kind`), so swadesh's own metric
classes qualify exactly as sacreBLEU's do.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
import platform
import sys

import numpy as np
import sacrebleu.significance as _sig

from . import _swadesh

sacrelogger = logging.getLogger("sacrebleu")


def _seed_from_env() -> Tuple[str, Optional[int]]:
    seed = os.environ.get("SACREBLEU_SEED", "12345")
    return seed, (None if seed.lower() == "none" else int(seed))


def _resampled_sums(stats, idxs: np.ndarray) -> np.ndarray:
    """``np.array(stats, dtype='float32')[idxs].sum(1)``, computed natively."""
    s32 = np.ascontiguousarray(np.array(stats, dtype="float32"))
    ix = np.ascontiguousarray(idxs, dtype=np.int64)
    if (s32.ndim != 2 or ix.ndim != 2 or ix.shape[1] == 0
            or (ix.size and (ix.min() < 0 or ix.max() >= s32.shape[0]))):
        # Empty, negative or out-of-range indices (e.g. systems of different length):
        # numpy itself decides, so the error or the wrap-around is exactly sacreBLEU's.
        return s32[idxs].sum(1)
    width = s32.shape[1]
    # The kernel's `n` is the number of indices per resample (ix.shape[1]), not len(stats).
    buf = _swadesh.bootstrap_sums_f32(s32.tobytes(), width, ix.tobytes(), ix.shape[1])
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
    native = _native_bleu_scores(metric, sums)
    if native is not None:
        return str(seed).lower(), native
    scores = [metric._compute_score_from_stats(_s) for _s in sums]
    return str(seed).lower(), scores


class _Score:
    """Minimal stand-in for a bootstrap `Score`: callers only read `.score`."""

    __slots__ = ("score",)

    def __init__(self, score):
        self.score = score


_NUMPY2 = int(np.__version__.split(".")[0]) >= 2


def _metric_kind(metric):
    """Which native kernel can score this metric: "bleu", "chrf", or None.

    Duck-typed on the attributes the kernels read rather than on class identity, so that swadesh's
    own classes qualify exactly as sacREBLEU's do — the kernels only need these fields, and
    requiring a particular class would exclude the very implementation this project is moving to.
    """
    if hasattr(metric, "max_ngram_order") and hasattr(metric, "smooth_method"):
        return "bleu"
    if hasattr(metric, "char_order") and hasattr(metric, "word_order") and hasattr(metric, "beta"):
        return "chrf"
    return None


def _native_bleu_scores(metric, sums: np.ndarray):
    """Score float32 bootstrap rows natively for plain BLEU or chrF under NumPy >= 2,
    where the float32/Python-scalar promotion rules (NEP 50) are fixed and
    emulated exactly in Rust. Returns None to fall back to sacreBLEU's code."""
    if not _NUMPY2:
        return None
    buf = np.ascontiguousarray(sums, dtype=np.float32).tobytes()
    kind = _metric_kind(metric)
    if kind == "bleu":
        if metric.max_ngram_order < 1 or sums.ndim != 2 or sums.shape[1] != 2 + 2 * metric.max_ngram_order:
            # Rows that are not BLEU statistics of this order: sacreBLEU's own scorer decides.
            return None
        sv = metric.smooth_value
        vals = _swadesh.bleu_scores_np2_f32(
            buf, sums.shape[1], metric.max_ngram_order,
            metric.smooth_method, None if sv is None else float(sv), bool(metric.effective_order))
        return [_Score(v) for v in vals]
    if kind == "chrf":
        if metric.order < 1 or sums.ndim != 2 or sums.shape[1] != 3 * metric.order:
            return None
        vals, is32 = _swadesh.chrf_scores_np2_f32(buf, sums.shape[1], metric.order, int(metric.beta),
                                                  bool(metric.eps_smoothing))
        f32 = np.float32
        return [_Score(f32(v) if b else v) for v, b in zip(vals, is32)]
    return None


def _f32_scores(metric, sums):
    """Scores of float32 resample rows: native when exact, else sacreBLEU's code."""
    native = _native_bleu_scores(metric, sums)
    if native is not None:
        return np.array([x.score for x in native])
    return np.array([metric._compute_score_from_stats(_s).score for _s in sums])


def _i64_scores(metric, rows):
    """Scores of exact int64 statistic rows (paired AR test), natively when the
    metric is plain BLEU/chrF; else via the metric's own ``_aggregate_and_compute``."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    width = rows.shape[1] if rows.ndim == 2 else -1
    kind = _metric_kind(metric)
    if kind == "bleu" and metric.max_ngram_order >= 1 and width == 2 + 2 * metric.max_ngram_order:
        sv = metric.smooth_value
        return np.array(_swadesh.bleu_scores_i64(rows.tobytes(), width, metric.max_ngram_order, metric.smooth_method,
                                                 None if sv is None else float(sv), bool(metric.effective_order)))
    if kind == "chrf" and metric.order >= 1 and width == 3 * metric.order:
        return np.array(_swadesh.chrf_scores_i64(rows.tobytes(), width, metric.order, int(metric.beta),
                                                 bool(metric.eps_smoothing)))
    return np.array([metric._aggregate_and_compute(x).score for x in rows[:, None]])


def _paired_sums(stats, idxs, seed, n_samples, n):
    """Resampled float32 row sums for the paired bootstrap test. With an integer
    seed and stats as long as the system output, the indices
    ``default_rng(seed).choice(n, (n_samples, n))`` are drawn natively."""
    if idxs is None:
        s32 = np.ascontiguousarray(np.array(stats, dtype="float32"))
        if s32.ndim == 2 and s32.shape[0] == n and s32.shape[1] > 0:
            buf = _swadesh.bootstrap_sums_f32_seeded(s32.tobytes(), s32.shape[1], n, n_samples, seed)
            return np.frombuffer(buf, dtype=np.float32).reshape(n_samples, s32.shape[1]).copy()
        idxs = np.random.default_rng(seed).choice(n, size=(n_samples, n), replace=True)
    return _resampled_sums(stats, idxs)


def _ar_shuffle(bl_stats, sys_stats, pos_sel):
    """``pos_sel @ bl + neg_sel @ sys`` and its mirror, exact for int stats.
    Returns None when the stats are not integral (falls back to numpy)."""
    bl = np.asarray(bl_stats)
    sy = np.asarray(sys_stats)
    if bl.dtype.kind not in "iu" or sy.dtype.kind not in "iu":
        return None
    sel = np.ascontiguousarray(pos_sel, dtype=np.uint8)
    if bl.ndim != 2 or bl.shape != sy.shape or sel.ndim != 2 or sel.shape[1] != bl.shape[0]:
        # Mismatched shapes: numpy's matmul below raises sacreBLEU's own error.
        return None
    bl = np.ascontiguousarray(bl, dtype=np.int64)
    sy = np.ascontiguousarray(sy, dtype=np.int64)
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
        if fast is None:
            scores_a = np.array([metric._aggregate_and_compute(x).score for x in shuf_a[:, None]])
            scores_b = np.array([metric._aggregate_and_compute(x).score for x in shuf_b[:, None]])
        else:
            scores_a, scores_b = _i64_scores(metric, shuf_a), _i64_scores(metric, shuf_b)
        p = _compute_p_value(np.abs(np.array(scores_a) - np.array(scores_b)), diff)
        res = Result(sys_score.score, p)
        if n_ar_confidence > 0:
            sacrelogger.info(f' > Performing bootstrap resampling for confidence interval (# resamples: {n_ar_confidence})')
            sums = _resampled_sums(sys_stats, bs_idxs)
            sys_scores = _f32_scores(metric, sums)
            res.mean, res.ci = estimate_ci(sys_scores)
        results[name] = res
    return sys_name, results


def _paired_bs_test(baseline_info, sys_name, hypotheses, references, metrics,
                    n_samples=1000, n_ar_confidence=-1, seed=None):
    """Drop-in for ``sacrebleu.significance._paired_bs_test``."""
    rng = np.random.default_rng(seed)
    results = {}
    native_seed = isinstance(seed, int) and 0 <= seed < 2**128 and len(hypotheses) > 0
    # With an integer seed the indices are re-drawn natively (identical stream);
    # numpy draws them only when needed.
    idxs = None if native_seed else rng.choice(len(hypotheses), size=(n_samples, len(hypotheses)), replace=True)
    for name, metric in metrics.items():
        bl_stats, bl_result = baseline_info[name]
        sacrelogger.info(f'Computing {name} for {sys_name!r} and extracting sufficient statistics')
        sys_stats = metric._extract_corpus_statistics(hypotheses, references)
        sys_score = metric._aggregate_and_compute(sys_stats)
        diff = abs(bl_result.score - sys_score.score)
        sacrelogger.info(f' > Performing paired bootstrap resampling test (# resamples: {n_samples})')
        scores_bl = _f32_scores(metric, _paired_sums(bl_stats, idxs, seed, n_samples, len(hypotheses)))
        scores_sys = _f32_scores(metric, _paired_sums(sys_stats, idxs, seed, n_samples, len(hypotheses)))
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


class Result:
    """One system's score for one metric, with the paired test's p-value and interval.

    Every field except the score is None until a test fills it in — the baseline row of a paired
    test carries no p-value — and they are printed as they stand, so the ``repr`` is the field
    dump the CLI shows.
    """

    def __init__(self, score: float, p_value=None, mean=None, ci=None):
        self.score = score
        self.p_value = p_value
        self.mean = mean
        self.ci = ci

    def __repr__(self) -> str:
        # `str`, not an f-string and not `repr`: the oracle interpolates each field with `str`, so a
        # numpy scalar prints as its bare number. `f"{v}"` is *not* the same for a `np.float32` --
        # `__format__` widens it to double, printing `20.83409309387207` where the oracle prints
        # `20.834093` (measured 2026-11-24; docs/spec/significance.md).
        attrs = ",".join(f"{k}={v!s}" for k, v in vars(self).items())
        return attrs


def _compute_p_value(stats: np.ndarray, real_difference: float) -> float:
    """One-sided p-value from the resampled differences.

    A resample counts against the null hypothesis only when its difference **exceeds** the real one,
    strictly: a system compared with itself has a real difference of zero, and a zero must not be
    attributed to chance. Both `+1`s correct for having only a finite number of resamples, which
    keeps the probability of a false rejection at or below the chosen level.
    """
    c = np.sum(stats > real_difference).item()
    return (c + 1) / (len(stats) + 1)


def estimate_ci(scores: np.ndarray) -> Tuple[float, float]:
    """Mean and 95% confidence-interval half-width of the resampled scores.

    The bounds are the `n // 40` and `n - n // 40 - 1` order statistics, i.e. the 2.5th and 97.5th
    percentiles by index. Taking order statistics rather than an interpolated quantile is what makes
    the interval reproducible from the same draws.
    """
    scores = np.sort(scores)
    n = len(scores)
    lower_idx = n // 40
    upper_idx = n - lower_idx - 1
    lower, upper = scores[lower_idx], scores[upper_idx]
    return (scores.mean(), 0.5 * (upper - lower))


IS_WINDOWS = platform.system() == "Windows"


class _PairedTestBase:
    """Runs a paired bootstrap (`bs`) or approximate-randomisation (`ar`) test between systems.

    The first named system is the baseline. For each other system and each metric, the segment
    statistics are resampled `n_samples` times with the two systems swapped in each resample, and the
    resulting difference distribution gives the p-value and the baseline's confidence interval.
    """

    _DEFAULT_SAMPLES = {"ar": 10000, "bs": 1000}

    def __init__(
        self,
        named_systems: List[Tuple[str, Sequence[str]]],
        metrics: Mapping[str, Any],
        references: Optional[Sequence[Sequence[str]]],
        test_type: str = "ar",
        n_samples: int = 0,
        n_ar_confidence: int = -1,
        n_jobs: int = 1,
        resampler=None,
        ar_test=None,
        bs_test=None,
    ):
        if test_type not in ("ar", "bs"):
            raise AssertionError(f"Unknown test type {test_type!r}")
        self.test_type = test_type
        self._fn = ar_test if test_type == "ar" else bs_test
        if self._fn is None:
            raise ValueError("the paired test needs the matching resampler")

        seed = os.environ.get("SACREBLEU_SEED", "12345")
        self._seed = None if seed.lower() == "none" else int(seed)
        self.n_jobs = n_jobs
        self.references = references
        self.named_systems = named_systems

        self.n_ar_confidence = n_ar_confidence if n_ar_confidence != 0 else self._DEFAULT_SAMPLES["bs"]
        self.n_samples = n_samples if n_samples > 0 else self._DEFAULT_SAMPLES[self.test_type]
        self.n_systems = len(named_systems) - 1

        if IS_WINDOWS:
            self.n_jobs = 1
        elif self.n_jobs == 0:
            n_max_jobs = (mp.cpu_count() or 1) // 2
            self.n_jobs = 1 if n_max_jobs == 0 else min(n_max_jobs, self.n_systems)

        # Pre-compute the baseline: its statistics, its score, and its interval.
        self._signatures: Dict[str, Any] = {}
        self._baseline_info: Dict[str, Tuple[Any, Result]] = {}
        self.metrics: Dict[str, Any] = {}

        bl_name, bl_hyps = self.named_systems[0]
        for name, metric in metrics.items():
            bl_stats = metric._extract_corpus_statistics(bl_hyps, self.references)
            bl_score = metric._aggregate_and_compute(bl_stats)

            confidence_n = self.n_samples if self.test_type == "bs" else self.n_ar_confidence
            bl_mean, bl_ci = None, None
            if confidence_n > 0:
                _, bl_scores = resampler(bl_stats, metric, confidence_n)
                bl_mean, bl_ci = estimate_ci(np.array([x.score for x in bl_scores]))

            result = Result(bl_score.score, mean=bl_mean, ci=bl_ci)
            self._baseline_info[bl_score.name] = (bl_stats, result)
            self.metrics[bl_score.name] = metric

            sig = metric.get_signature()
            sig.update("seed", str(self._seed).lower())
            sig.update(self.test_type, self.n_samples)
            if self.n_ar_confidence > 0:
                sig.update("bs", self.n_ar_confidence)
            self._signatures[bl_score.name] = sig

    def __call__(self) -> Tuple[Dict[str, Any], Dict[str, List[Union[str, Result]]]]:
        """Run the test on every non-baseline system, in order."""
        tasks = []
        scores: Dict[str, List[Union[str, Result]]] = {"System": [ns[0] for ns in self.named_systems]}

        for metric, (_, result) in self._baseline_info.items():
            scores[metric] = [result]

        for name, hyps in self.named_systems[1:]:
            tasks.append(
                (
                    self._baseline_info,
                    name,
                    hyps,
                    self.references,
                    self.metrics,
                    self.n_samples,
                    self.n_ar_confidence,
                    self._seed if self._seed else None,
                )
            )

        if self.n_jobs == 1 or sys.platform == "win32":
            results = [self._fn(*args) for args in tasks]
        else:
            with mp.get_context("fork").Pool(self.n_jobs) as pool:
                jobs = [pool.apply_async(self._fn, args) for args in tasks]
                results = [j.get() for j in jobs]

        for _sys_name, sys_results in results:
            for metric, r in sys_results.items():
                scores[metric].append(r)

        return self._signatures, scores

class PairedTest(_PairedTestBase):
    """Paired significance test using the native resamplers.

    Same constructor and the same returned shape as sacreBLEU's ``PairedTest``; the arithmetic and
    the manager are defined in this module (see ``docs/spec/significance.md``).
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("resampler", _bootstrap_resample)
        kwargs.setdefault("ar_test", _paired_ar_test)
        kwargs.setdefault("bs_test", _paired_bs_test)
        super().__init__(*args, **kwargs)
