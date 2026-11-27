"""Assertion helpers for a hypothesis/reference test suite.

    from swadesh.testing import assert_bleu_at_least
    def test_system(): assert_bleu_at_least(hyps, refs, 30.0)

Each helper scores with ``corpus_score`` and raises ``AssertionError`` with the metric's own
formatted line when the bound is missed, so the failure shows the number, its signature and the
seed. Nothing here changes a score.
"""
from __future__ import annotations

from typing import Any, Optional


def _fail(name: str, score: Any, bound: str, value: float) -> "AssertionError":
    return AssertionError(f"{name}: score {score.score:.4f} misses the {bound} of {value:.4f}\n  {score}")


def assert_score_at_least(metric, hypotheses, references, minimum: float,
                          name: Optional[str] = None) -> Any:
    """Assert the corpus score is at least ``minimum``; returns the score object."""
    score = metric.corpus_score(hypotheses, references)
    if not score.score >= minimum:
        raise _fail(name or score.name, score, "minimum", minimum)
    return score


def assert_score_at_most(metric, hypotheses, references, maximum: float,
                         name: Optional[str] = None) -> Any:
    """Assert the corpus score is at most ``maximum`` (TER counts errors); returns the score object."""
    score = metric.corpus_score(hypotheses, references)
    if not score.score <= maximum:
        raise _fail(name or score.name, score, "maximum", maximum)
    return score


def assert_bleu_at_least(hypotheses, references, minimum: float, **metric_kwargs) -> Any:
    """``assert_score_at_least(BLEU(**metric_kwargs), ...)``."""
    from .metrics import BLEU

    return assert_score_at_least(BLEU(**metric_kwargs), hypotheses, references, minimum, "BLEU")


def assert_chrf_at_least(hypotheses, references, minimum: float, **metric_kwargs) -> Any:
    """``assert_score_at_least(CHRF(**metric_kwargs), ...)``."""
    from .metrics import CHRF

    return assert_score_at_least(CHRF(**metric_kwargs), hypotheses, references, minimum, "chrF")


def assert_ter_at_most(hypotheses, references, maximum: float, **metric_kwargs) -> Any:
    """``assert_score_at_most(TER(**metric_kwargs), ...)`` - TER counts errors, so lower is better."""
    from .metrics import TER

    return assert_score_at_most(TER(**metric_kwargs), hypotheses, references, maximum, "TER")
