"""`swadesh.evaluate`: one call, one dict of score objects.

    from swadesh import evaluate
    scores = evaluate(hyps, [refs])                 # {"BLEU": ..., "chrF2": ..., "TER": ...}
    scores["BLEU"].score

The ergonomic half of `docs/feature-ideas.md`'s idea 2, and the reason the *speed* half is not here:
measured on this tree, 20,000 WMT22 en-de segments,

    fresh metric, references passed each call    123.6 ms   (5 systems x BLEU + chrF++)
    one metric each, references cached once      124.4 ms

- a 1% difference, in the wrong direction - because there is no shared tokenization to remove. chrF is
character-based: it never calls BLEU's tokenizer, so feeding it BLEU's tokenized text would change its
score, which is the one thing this package must not do. Each metric's reference preparation is a small
fraction of its work; the native statistics are the cost.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

from .metrics import BLEU, CHRF, TER


def _as_metrics(metrics):
    """`[BLEU, CHRF(), TER]` -> an ordered `{name: metric}`, in the order given."""
    items = metrics.items() if isinstance(metrics, dict) else [(None, m) for m in metrics]
    out: "OrderedDict[str, object]" = OrderedDict()
    for name, entry in items:
        metric = entry() if isinstance(entry, type) else entry
        if name is None:
            # A metric knows its own name only after it has been evaluated; use the class for now and
            # replace it with the score's name below.
            name = type(metric).__name__
        out[str(name)] = metric
    return out


def evaluate(
    hypotheses: Sequence[str],
    references: Optional[Sequence[Sequence[str]]] = None,
    metrics: Union[Sequence[object], Dict[str, object], None] = None,
    n_bootstrap: int = 1,
) -> Dict[str, object]:
    """Score `hypotheses` against `references` with several metrics, in one call.

    `metrics` is a list of classes or instances (`[BLEU, chrF, TER]`), or a mapping of names to
    metrics; it defaults to BLEU, chrF2 and TER. The result is an ordered dict keyed by each
    metric's own name (`BLEU`, `chrF2`, `TER`), so `scores["BLEU"].score` and `.format()` work
    exactly as they do on a single `corpus_score` call.

    Every metric sees the same hypotheses and the same references, and the length check happens once:
    under `swadesh.strict()` a mismatch raises before any metric is computed. The numbers are
    identical to calling each metric's `corpus_score` in turn - this is a convenience, not a second
    implementation, and the module docstring records why it is not faster.
    """
    metrics = _as_metrics(metrics if metrics is not None else [BLEU, CHRF, TER])
    scores: "OrderedDict[str, object]" = OrderedDict()
    for name, metric in metrics.items():
        score = metric.corpus_score(hypotheses, references, n_bootstrap=n_bootstrap)
        # The metric's own name, which is what the CLI prints and what the signature starts with.
        scores[str(getattr(score, "name", name))] = score
    return scores
