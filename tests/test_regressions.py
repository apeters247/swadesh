"""Regression tests for differential mismatches found by the deep run.

Each test is ``xfail(strict=True)`` and pins one open mismatch with its exact,
minimal inputs (see docs/diff-report.md). When the core is fixed the test passes,
strict xfail then reports it as a failure, and the marker must be removed.
"""
import pytest
import sacrebleu

import swadesh


def test_sentence_scores_signature_state_matches_sacrebleu():
    hyps = ["a b", "c d"]
    refs = [["a b", "c d"], [None, "c d"]]
    ours, theirs = swadesh.BLEU(), sacrebleu.BLEU()
    ours.corpus_score(hyps, refs)
    theirs.corpus_score(hyps, refs)
    ours_scores = ours.sentence_scores(hyps, refs)
    theirs_scores = [theirs.sentence_score(h, [r[i] for r in refs]).score for i, h in enumerate(hyps)]
    assert ours_scores == theirs_scores
    assert str(ours.get_signature()) == str(theirs.get_signature())


def test_patch_reaches_the_call_shapes_dependents_use():
    """lm-eval, lighteval and COMET reach sacreBLEU through module attributes; the patch must cover them.

    Measured in docs/integrations.md: patch() replaces the class objects, so a module-attribute lookup
    at call time is redirected, while a class bound by ``from ... import BLEU`` *before* the patch is
    not - which is why that is asserted here as the documented limit, not as a bug.
    """
    import sacrebleu
    from sacrebleu.metrics import BLEU as bound_before_patch        # the limit case
    import swadesh

    hyps = ["the cat sat on the mat", "a dog barked"]
    refs = [["the cat is on the mat", "a dog barked"]]

    def shapes():
        return [
            ("module attribute class (lighteval)", sacrebleu.BLEU().corpus_score(hyps, refs).score),
            ("module attribute fn (lm-eval, comet)", sacrebleu.corpus_bleu(hyps, refs).score),
            ("sacrebleu.metrics class", sacrebleu.metrics.BLEU().corpus_score(hyps, refs).score),
            ("bound before the patch", bound_before_patch().corpus_score(hyps, refs).score),
        ]

    before = dict(shapes())
    swadesh.patch()
    try:
        assert sacrebleu.BLEU.__module__ == "swadesh"
        assert sacrebleu.metrics.BLEU.__module__ == "swadesh"
        after = dict(shapes())
        # patching can only move where a number is computed, never the number
        assert all(before[k] == after[k] for k in before), (before, after)
        # the three shapes the dependents use are swadesh's now; the import-bound one is not
        assert sacrebleu.BLEU.__mro__[1].__module__.startswith("swadesh")
        assert bound_before_patch.__module__ == "sacrebleu.metrics.bleu"
    finally:
        swadesh.unpatch()
    assert sacrebleu.BLEU.__module__ == "sacrebleu.metrics.bleu"
