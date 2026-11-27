"""Streaming corpus statistics: score a corpus as it arrives, in O(width) memory.

``Accumulator(metric)`` keeps a running sum of the metric's per-segment statistics rows, so the peak
memory is one row wide rather than ``n_segments`` rows. The contract it rests on is
``metrics._aggregate_and_compute``: a corpus score is the element-wise sum of the per-segment rows in
arrival order, with the first row's element type kept (see ``sum_of_lists``). Adding the rows in the
same order is exactly that sum, so a streamed score is bit-identical to ``corpus_score``.

Bootstrapped confidence intervals resample *segments* (``significance._bootstrap_resample`` reads the
row count), so they need the rows, not the sum. ``keep_rows=True`` retains them for that; the default
keeps only the running sum, which is what makes the memory constant, and ``score(n_bootstrap>1)``
raises rather than silently returning an interval it cannot compute.
"""
from __future__ import annotations

from typing import Any, List, Optional


def _as_rows(stats) -> List[List[Any]]:
    """A statistics object as a list of row lists (an ``n x width`` array or an already-list row)."""
    tolist = getattr(stats, "tolist", None)
    if tolist is not None:
        stats = tolist()
    if stats and isinstance(stats[0], list):
        return stats
    if stats:
        return [stats]
    return []


class Accumulator:
    """Add segments, read the corpus score at any point, without holding the corpus in memory.

    ``metric`` is a metric instance (``BLEU()``, ``CHRF()``, ``TER()``). ``add(hypothesis,
    references)`` takes one segment's references; ``add_corpus(hypotheses, references)`` takes the
    corpus shape. ``score()`` returns the metric's own score object for everything added so far.
    """

    def __init__(self, metric, keep_rows: bool = False):
        self.metric = metric
        self.keep_rows = bool(keep_rows)
        self.n_segments = 0
        self._row: Optional[List[Any]] = None
        self._rows: Optional[List[List[Any]]] = [] if keep_rows else None

    def _add_row(self, row) -> None:
        row = list(row)
        if self._row is None:
            # a separate copy: the running row must not alias the row kept for the bootstrap
            self._row = list(row)
        else:
            for i, v in enumerate(row):
                self._row[i] += v
        self.n_segments += 1
        if self._rows is not None:
            self._rows.append(list(row))

    def add(self, hypothesis: str, references) -> "Accumulator":
        """Add one segment; ``references`` is that segment's list of reference strings."""
        stats = self.metric._extract_corpus_statistics(
            [hypothesis], [[r] for r in references],
        )
        for row in _as_rows(stats):
            self._add_row(row)
        return self

    def add_corpus(self, hypotheses, references) -> "Accumulator":
        """Add a whole batch (the corpus shape); the batch itself is materialized, ``add`` is not."""
        stats = self.metric._extract_corpus_statistics(hypotheses, references)
        for row in _as_rows(stats):
            self._add_row(row)
        return self

    def reset(self) -> "Accumulator":
        """Forget everything added; the metric is reused as is."""
        self.n_segments = 0
        self._row = None
        if self._rows is not None:
            self._rows = []
        return self

    def score(self, n_bootstrap: int = 1):
        """The metric's score object over everything added so far, identical to ``corpus_score``.

        ``n_bootstrap > 1`` adds the bootstrap confidence interval, which needs the per-segment rows:
        it raises unless the accumulator was built with ``keep_rows=True``.
        """
        if self.n_segments == 0:
            return self.metric.corpus_score([], [[]])
        score = self.metric._score_from_row(self._row)
        if n_bootstrap > 1:
            if self._rows is None:
                raise ValueError(
                    "a bootstrapped confidence interval resamples segments, so it needs the "
                    "per-segment rows: build the accumulator with keep_rows=True. The default keeps "
                    "only the running sum, which is what keeps the memory constant."
                )
            import numpy as np

            from .significance import _bootstrap_resample

            stats = np.array(self._rows)
            self.metric.n_bootstrap = n_bootstrap
            self.metric.seed, bootstrap_scores = _bootstrap_resample(stats, self.metric, n_bootstrap)
            score.estimate_ci(bootstrap_scores)
        return score
