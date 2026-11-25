"""The ported metric base must behave exactly like sacreBLEU's.

``swadesh/base.py`` is a direct port of ``sacrebleu/metrics/base.py`` (see its docstring and
``NOTICE``). It exists so the metrics stop inheriting from the sacreBLEU package, but it is a
drop-in: signatures, score formatting and the argument-check errors have to stay byte-identical.
These tests diff it against the real thing.
"""

import random
import re
from pathlib import Path

import pytest
import sacrebleu
import sacrebleu.metrics.base as SB

import swadesh.base as OURS

ROOT = Path(__file__).resolve().parents[1]
BASE_SRC = ROOT / "python" / "swadesh" / "base.py"

SIG_ARGS = [
    {"num_refs": 1},
    {"num_refs": 3, "n_bootstrap": 100, "seed": 12345},
    {"num_refs": -1},
    {"num_refs": 1, "test_set": "wmt22", "langpair": "en-de", "origlang": None, "subset": None},
    {"num_refs": 2, "n_bootstrap": 1000, "seed": 7, "test_set": "t", "langpair": "l"},
]


def grab(fn):
    """Run `fn`, returning the exception as (type name, message) or None."""
    try:
        fn()
        return None
    except Exception as e:  # noqa: BLE001 -- the exception is the expectation
        return (type(e).__name__, str(e))


@pytest.mark.parametrize("args", SIG_ARGS, ids=[str(a) for a in SIG_ARGS])
def test_signature_matches_sacrebleu(args):
    ours, theirs = OURS.Signature(dict(args)), SB.Signature(dict(args))
    assert ours.info == theirs.info
    assert ours._abbr == theirs._abbr
    assert ours.format() == theirs.format()
    assert ours.format(short=True) == theirs.format(short=True)
    assert str(ours) == str(theirs)
    assert repr(ours) == repr(theirs)


def test_signature_update_matches_sacrebleu():
    ours, theirs = OURS.Signature({"num_refs": 1}), SB.Signature({"num_refs": 1})
    ours.update("bs", 100)
    theirs.update("bs", 100)
    assert ours.format() == theirs.format()


def test_signature_missing_num_refs_raises_like_sacrebleu():
    assert grab(lambda: OURS.Signature({})) == grab(lambda: SB.Signature({}))


@pytest.mark.parametrize("name,score", [("BLEU", 12.3456), ("chrF2", 0.5), ("TER", 0.0), ("X", 99.999)])
def test_score_format_matches_sacrebleu(name, score):
    for width in (0, 1, 2, 4):
        for score_only in (False, True):
            for sig in ("", "nrefs:1|version:2.6.0"):
                ours, theirs = OURS.Score(name, score), SB.Score(name, score)
                assert ours.format(width, score_only, sig) == theirs.format(width, score_only, sig)
    assert repr(OURS.Score(name, score)) == repr(SB.Score(name, score))

    ours, theirs = OURS.Score(name, score), SB.Score(name, score)
    ours._verbose = theirs._verbose = " (foo)"
    assert ours.format() == theirs.format()
    assert grab(lambda: ours.format(is_json=True)) == grab(lambda: theirs.format(is_json=True))
    assert grab(lambda: OURS.Score(name, score).format(is_json=True)) == \
        grab(lambda: SB.Score(name, score).format(is_json=True))


def test_score_estimate_ci_matches_sacrebleu():
    rnd = random.Random(0)
    vals = [rnd.uniform(0.0, 100.0) for _ in range(400)]
    ours, theirs = OURS.Score("BLEU", 12.0), SB.Score("BLEU", 12.0)
    ours.estimate_ci([OURS.Score("BLEU", v) for v in vals])
    theirs.estimate_ci([SB.Score("BLEU", v) for v in vals])
    assert ours._mean == theirs._mean
    assert ours._ci == theirs._ci
    assert ours.format() == theirs.format()


class _OursMetric(OURS.Metric):
    def _aggregate_and_compute(self, stats):
        pass

    def _compute_score_from_stats(self, stats):
        pass

    def _preprocess_segment(self, sent):
        return sent

    def _extract_reference_info(self, refs):
        return {"ref_words": list(refs)}

    def _compute_segment_statistics(self, hypothesis, ref_kwargs):
        return [0]


class _TheirsMetric(SB.Metric):
    def _aggregate_and_compute(self, stats):
        pass

    def _compute_score_from_stats(self, stats):
        pass

    def _preprocess_segment(self, sent):
        return sent

    def _extract_reference_info(self, refs):
        return {"ref_words": list(refs)}

    def _compute_segment_statistics(self, hypothesis, ref_kwargs):
        return [0]


# The error message is prefixed with the class name; real metrics share the name "TER".
_OursMetric.__name__ = _TheirsMetric.__name__ = "TER"

SENT_CASES = [(1, ["a"]), ("a", "a"), ("a", [1]), ("a", [None]), ("a", ["a"]), ("a", ())]
CORPUS_CASES = [
    ("a", [["a"]]), ([1], [["a"]]), ([None], [["a"]]), (["a"], "x"),
    (["a"], [1]), (["a"], [[1]]), (["a"], [[None]]), (["a"], [["a"]]),
]


@pytest.mark.parametrize("hyp,refs", SENT_CASES, ids=[f"{h!r}-{r!r}" for h, r in SENT_CASES])
def test_sentence_arg_checks_match_sacrebleu(hyp, refs):
    assert grab(lambda: _OursMetric()._check_sentence_score_args(hyp, refs)) == \
        grab(lambda: _TheirsMetric()._check_sentence_score_args(hyp, refs))


@pytest.mark.parametrize("hyps,refs", CORPUS_CASES, ids=[f"{h!r}-{r!r}" for h, r in CORPUS_CASES])
def test_corpus_arg_checks_match_sacrebleu(hyps, refs):
    assert grab(lambda: _OursMetric()._check_corpus_score_args(hyps, refs)) == \
        grab(lambda: _TheirsMetric()._check_corpus_score_args(hyps, refs))


def test_base_module_is_standalone():
    """The whole point: no sacreBLEU import, and the signature version is the compat literal."""
    src = BASE_SRC.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:import\s+sacrebleu|from\s+sacrebleu)", src, re.M)
    assert OURS.COMPAT_VERSION == sacrebleu.__version__
