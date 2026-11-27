"""The HuggingFace ``evaluate`` bridge: ``compute`` without any HuggingFace import.

    from swadesh.hf_metric import compute
    compute(predictions, references, metric="bleu")

``references`` is HuggingFace's layout (a list of references per prediction); it is transposed into
swadesh's corpus layout (one stream per reference slot). The numbers come from the same
``corpus_score`` the library exposes, and the returned dict carries the metric's signature.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

_BY_NAME = {
    "bleu": ("BLEU", {}),
    "chrf": ("CHRF", {}),
    "chrf2": ("CHRF", {}),
    "chrf++": ("CHRF", {"word_order": 2}),
    "chrf2++": ("CHRF", {"word_order": 2}),
    "ter": ("TER", {}),
}


def _metric(name: str):
    key = str(name).strip().lower()
    if key not in _BY_NAME:
        raise ValueError(
            "unknown metric %r; expected one of %s" % (name, ", ".join(sorted(set(_BY_NAME))))
        )
    from . import metrics as _m

    cls_name, kwargs = _BY_NAME[key]
    return getattr(_m, cls_name)(**kwargs)


def compute(predictions: Sequence[str], references: Sequence[Sequence[str]],
            metric: str = "bleu") -> Dict[str, Any]:
    """Score ``predictions`` against ``references`` with one metric, HuggingFace's call shape.

    ``references`` is a list with one entry per prediction (a list of that prediction's references).
    Returns ``{name: score, "signature": ...}`` where ``name`` is the metric's own name.
    """
    predictions = list(predictions)
    references = [list(refs) for refs in references]
    if len(references) != len(predictions):
        raise ValueError(
            "references has %d entries for %d predictions" % (len(references), len(predictions))
        )
    widths = {len(refs) for refs in references}
    if len(widths) > 1:
        raise ValueError(
            "swadesh needs a rectangular reference set (every prediction with the same number of "
            "references); got counts %s" % sorted(widths)
        )
    streams: List[List[str]] = [list(slot) for slot in zip(*references)] if references else []

    m = _metric(metric)
    score = m.corpus_score(predictions, streams)
    return {score.name: score.score, "signature": str(m.get_signature())}
