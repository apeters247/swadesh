import os

import pytest
import sacrebleu

import swadesh
from swadesh.hf_metric import compute

HYPS = ["the cat sat on the mat", "a dog barked at the moon", "prices rose by 3.2%"]
REFS_HF = [["the cat is on the mat"], ["a dog barked at the moon"], ["prices increased 3.2%"]]
REFS_STREAMS = [["the cat is on the mat", "a dog barked at the moon", "prices increased 3.2%"]]


def test_compute_is_corpus_score_in_the_huggingface_shape():
    """Each metric goes through the same corpus_score, keyed by the metric's own name."""
    for name, metric in (("bleu", swadesh.BLEU()), ("chrf", swadesh.CHRF()),
                         ("chrF++", swadesh.CHRF(word_order=2)), ("ter", swadesh.TER())):
        got = compute(HYPS, REFS_HF, metric=name)
        expected = metric.corpus_score(HYPS, REFS_STREAMS)
        assert list(got) == [expected.name, "signature"]
        assert got[expected.name] == expected.score
        assert got["signature"] == str(metric.get_signature())

    # And the number is sacreBLEU's.
    assert compute(HYPS, REFS_HF, "bleu")["BLEU"] == \
        sacrebleu.BLEU().corpus_score(HYPS, REFS_STREAMS).score

    # The HuggingFace layout transposes into swadesh's streams; two references each.
    refs_two = [["ref one a", "ref one b"], ["ref two a", "ref two b"], ["ref three a", "ref three b"]]
    streams_two = [["ref one a", "ref two a", "ref three a"], ["ref one b", "ref two b", "ref three b"]]
    assert compute(HYPS, refs_two, "bleu")["BLEU"] == \
        swadesh.BLEU().corpus_score(HYPS, streams_two).score


def test_compute_rejects_ragged_references_and_unknown_metrics():
    with pytest.raises(ValueError, match="rectangular"):
        compute(HYPS, [["a"], ["b", "c"], ["d"]], "bleu")
    with pytest.raises(ValueError, match="references has"):
        compute(HYPS, [["a"]], "bleu")
    with pytest.raises(ValueError, match="unknown metric"):
        compute(HYPS, REFS_HF, "rouge")


def test_evaluate_loads_the_module_by_path():
    """The HuggingFace integration, skipped unless `evaluate` is installed (the `hf` extra)."""
    evaluate = pytest.importorskip("evaluate")
    path = os.path.join(os.path.dirname(swadesh.__file__), "evaluate_module.py")
    module = evaluate.load(path)
    got = module.compute(predictions=HYPS, references=REFS_HF, metric="bleu")
    assert got["BLEU"] == compute(HYPS, REFS_HF, "bleu")["BLEU"]
    assert str(module)  # an EvaluationModuleInfo renders
