"""Regression tests for the 2026-11-21 num_refs change.

A speed-up to `_cache_references` briefly made master compute `num_refs` from `len(streams)`, which
is only equal to sacreBLEU's per-segment reference count when no reference stream contains `None`.
The parity suite caught it; these pin the exact inputs so a future rewrite fails here first, and
fails in a few lines rather than 38.
"""
from __future__ import annotations

import pytest

import sacrebleu
import swadesh

H = ["the cat sat on the mat", "a dog ran .", "hello world", "", "x y z"]
R_ONE = ["the cat sat on a mat", "a dog ran .", "hello world", "", "x y z"]
R_NONE_TAIL = [R_ONE, [None, None, None, None, None]]
R_MIXED = [["the cat sat on a mat", None, "hello world", None, "x y z"],
           ["a cat", "a dog ran .", None, None, None]]
R_NONE_FIRST = [[None, "a dog ran .", "hello world", None, "x y z"]]


def _outcome(fn):
    try:
        s = fn()
        return (str(s.score), s.format(signature=""), getattr(s, "num_refs", None))
    except Exception as e:  # noqa: BLE001 -- sacreBLEU's exception is the expectation
        return (type(e).__name__, str(e))


@pytest.mark.parametrize("cls_name", ["BLEU", "CHRF", "TER"])
def test_num_refs_with_none_references_matches_sacrebleu(cls_name):
    """num_refs must be sacreBLEU's per-segment count, including when streams contain None."""
    for refs in (R_NONE_TAIL, R_MIXED, R_NONE_FIRST):
        sbm = getattr(sacrebleu, cls_name)()
        m = getattr(swadesh, cls_name)()
        assert _outcome(lambda: sbm.corpus_score(H, refs)) == _outcome(lambda: m.corpus_score(H, refs)), \
            (cls_name, refs)
        assert m.num_refs == sbm.num_refs, (cls_name, refs, m.num_refs, sbm.num_refs)


@pytest.mark.parametrize("cls_name", ["BLEU", "CHRF"])
def test_compat_functions_with_none_references(cls_name):
    fn = "corpus_" + cls_name.lower()
    sb, sw = getattr(sacrebleu, fn), getattr(swadesh, fn)
    for refs in (None, R_NONE_TAIL, R_MIXED):
        assert _outcome(lambda: sb(H, refs)) == _outcome(lambda: sw(H, refs)), refs


def test_num_refs_attribute_after_none_free_cache():
    """With no None anywhere, num_refs is the number of reference streams (sacreBLEU agrees)."""
    for module in (sacrebleu, swadesh):
        metric = module.BLEU()
        metric.corpus_score(H, [R_ONE])
        assert metric.num_refs == 1, module.__name__
