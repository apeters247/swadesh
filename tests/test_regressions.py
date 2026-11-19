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
