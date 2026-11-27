"""The HuggingFace ``evaluate`` module file for swadesh.

HuggingFace loads a *module file*, so point ``evaluate.load`` at this path::

    import evaluate
    module = evaluate.load("path/to/swadesh/evaluate_module.py")
    module.compute(predictions=["the cat"], references=[["the cat"]], metric="bleu")

``evaluate.load("swadesh")`` by *name* resolves on the Hub, and this project publishes nothing there,
so the path form is the shipping one. Every number comes from ``swadesh.hf_metric.compute``, i.e. from
swadesh's own ``corpus_score``.
"""
import datasets
import evaluate

from swadesh.hf_metric import compute as _compute

_DESCRIPTION = (
    "Bit-exact sacreBLEU-compatible BLEU, chrF, chrF++ and TER over a Rust core, with the corpus "
    "and interval statistics swadesh exposes."
)
_KWARGS_DESCRIPTION = """
Args:
    predictions: list of translated sentences (one per segment).
    references: list of lists of reference sentences, one list per prediction.
    metric: one of "bleu", "chrf", "chrf++", "ter" (default "bleu"), case-insensitive.
Returns:
    the metric's own name (e.g. "BLEU", "chrF2++", "TER") mapped to its score, and "signature",
    the metric's signature line (references, case, tokenizer, smooth, version).
Example:
    module.compute(predictions=["the cat"], references=[["the cat"]], metric="bleu")
    -> {"BLEU": 100.00000000000004, "signature": "nrefs:1|case:mixed|eff:no|tok:13a|..."}
"""
_CITATION = """\
@misc{swadesh,
  title  = {swadesh: a bit-exact sacreBLEU-compatible metric library},
  author = {Andrew Peters},
  year   = {2026},
  url    = {https://github.com/apeters247/swadesh},
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Swadesh(evaluate.EvaluationModule):
    """swadesh's BLEU / chrF / chrF++ / TER behind the ``evaluate`` interface."""

    def _info(self):
        return evaluate.EvaluationModuleInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Value("string", id="sequence"),
                "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
            }),
            reference_urls=["https://github.com/apeters247/swadesh"],
        )

    def _compute(self, predictions, references, metric="bleu"):
        return _compute(predictions, references, metric=metric)
