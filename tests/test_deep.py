"""Deep differential fuzzing: BLEU (every tokenizer), chrF, chrF++ and TER.

Each example is one fuzzed case, built from a Hypothesis-drawn seed (see
deep_gen.py). The case runs every public code path of the metric (sufficient
statistics, corpus score, sentence scores, signature), and the exception type
and message must match when sacreBLEU raises. Case counts are written per
metric and configuration (see fuzzsupport.CASES).

A failing case is written to ``$SWADESH_FAIL_DIR`` as JSON, so that
tests/shrink.py can reduce it to a minimal repro.

Run the default budget with ``pytest tests/test_deep.py``, or the 1M-per-test
budget with ``HYPOTHESIS_PROFILE=deep pytest tests/test_deep.py -k <name>``.
"""
import json
import os
import random

import pytest
import sacrebleu
from hypothesis import given
from hypothesis import strategies as st

import swadesh
from deep_gen import make_case, make_ter_case, make_ter_long_case
from fuzzsupport import budget, count
from test_differential import assert_same_score

TOKENIZERS = ["13a", "intl", "zh", "char", "none"]
SEEDS = st.integers(0, 2**63 - 1)


def _call(fn):
    try:
        return ("ok", fn())
    except Exception as e:  # any sacreBLEU or swadesh exception is an outcome to compare
        return ("err", type(e).__name__, str(e))


def check(ours_fn, theirs_fn, compare):
    """Same outcome: both return (then compare values) or both raise the same exception."""
    b = _call(theirs_fn)
    a = _call(ours_fn)
    if b[0] == "err":
        assert a == b, f"sacreBLEU raised {b}, swadesh gave {a}"
    else:
        assert a[0] == "ok", f"swadesh raised {a}, sacreBLEU returned"
        compare(a[1], b[1])


def _eq(a, b):
    assert a == b


def _ter_key(score):
    return (score.score, score.num_edits, score.ref_length, str(score))


def record_failure(kind, kw, hyps, refs):
    """Write a failing case for tests/shrink.py, if SWADESH_FAIL_DIR is set."""
    d = os.environ.get("SWADESH_FAIL_DIR")
    if d:
        path = os.path.join(d, f"{kind}-{os.getpid()}-{len(os.listdir(d))}.json")
        with open(path, "w") as f:
            json.dump({"kind": kind, "kw": kw, "hyps": hyps, "refs": refs}, f, ensure_ascii=False)


# --- case checks: raise AssertionError on any mismatch ----------------------

def bleu_case(kw, hyps, refs):
    ours, theirs = swadesh.BLEU(**kw), sacrebleu.BLEU(**kw)
    check(lambda: ours._extract_corpus_statistics(hyps, refs).tolist(),
          lambda: theirs._extract_corpus_statistics(hyps, refs), _eq)
    check(lambda: ours.corpus_score(hyps, refs), lambda: theirs.corpus_score(hyps, refs),
          assert_same_score)
    assert str(ours.get_signature()) == str(theirs.get_signature())
    check(lambda: ours.sentence_scores(hyps, refs),
          lambda: [theirs.sentence_score(h, [r_[i] for r_ in refs]).score for i, h in enumerate(hyps)],
          _eq)


def chrf_case(kw, hyps, refs):
    ours, theirs = swadesh.CHRF(**kw), sacrebleu.CHRF(**kw)
    check(lambda: ours._extract_corpus_statistics(hyps, refs).tolist(),
          lambda: theirs._extract_corpus_statistics(hyps, refs), _eq)
    check(lambda: ours.corpus_score(hyps, refs), lambda: theirs.corpus_score(hyps, refs),
          assert_same_score)
    assert str(ours.get_signature()) == str(theirs.get_signature())
    check(lambda: ours.sentence_scores(hyps, refs),
          lambda: [theirs.sentence_score(h, [r_[i] for r_ in refs]).score for i, h in enumerate(hyps)],
          _eq)


def ter_case(kw, hyps, refs):
    ours, theirs = swadesh.TER(**kw), sacrebleu.TER(**kw)
    check(lambda: ours._extract_corpus_statistics(hyps, refs).tolist(),
          lambda: theirs._extract_corpus_statistics(hyps, refs), _eq)
    check(lambda: _ter_key(ours.corpus_score(hyps, refs)),
          lambda: _ter_key(theirs.corpus_score(hyps, refs)), _eq)
    assert str(ours.get_signature()) == str(theirs.get_signature())
    check(lambda: [_ter_key(ours.sentence_score(h, [r_[i] for r_ in refs])) for i, h in enumerate(hyps)],
          lambda: [_ter_key(theirs.sentence_score(h, [r_[i] for r_ in refs])) for i, h in enumerate(hyps)],
          _eq)


def _run(kind, check_fn, kw, hyps, refs):
    try:
        check_fn(kw, hyps, refs)
    except AssertionError:
        record_failure(kind, kw, hyps, refs)
        raise


# --- BLEU: one parametrized test per tokenizer ------------------------------

@pytest.mark.parametrize("tok", TOKENIZERS)
@budget(200)
@given(seed=SEEDS)
def test_deep_bleu(tok, seed):
    count(f"BLEU tokenize={tok}", seed)
    r = random.Random(seed)
    hyps, refs = make_case(r)
    kw = dict(
        tokenize=tok, lowercase=r.random() < 0.5,
        smooth_method=r.choice(["exp", "floor", "add-k", "none"]),
        smooth_value=r.choice([None, 0.1, 0.5, 1, 2]),
        max_ngram_order=r.randint(1, 6),
        effective_order=r.random() < 0.5,
    )
    _run("bleu", bleu_case, kw, hyps, refs)


# --- chrF and chrF++ --------------------------------------------------------

@pytest.mark.parametrize("word_order", [0, 2], ids=["chrF", "chrF++"])
@budget(200)
@given(seed=SEEDS)
def test_deep_chrf(word_order, seed):
    count("chrF" if word_order == 0 else "chrF++", seed)
    r = random.Random(seed)
    hyps, refs = make_case(r)
    kw = dict(
        char_order=r.choice([1, 2, 4, 6, 7, 9]),
        word_order=word_order,
        beta=r.choice([0, 1, 2, 3]),
        lowercase=r.random() < 0.5,
        whitespace=r.random() < 0.5,
        eps_smoothing=r.random() < 0.5,
    )
    _run("chrf", chrf_case, kw, hyps, refs)


# --- TER: options drawn per case; shifted (permuted) references -------------

def _ter_opts(r):
    return {k: r.random() < 0.5 for k in ("normalized", "no_punct", "asian_support", "case_sensitive")}


def _opt_label(opts):
    return "TER " + " ".join(f"{k}={int(v)}" for k, v in opts.items())


@budget(100)
@given(seed=SEEDS)
def test_deep_ter(seed):
    r = random.Random(seed)
    opts = _ter_opts(r)
    count(_opt_label(opts), seed)
    hyps, refs = make_ter_case(r)
    _run("ter", ter_case, opts, hyps, refs)


# TER with 20-200 word sentences and block-shifted references. sacreBLEU spends
# 0.3-1.7 s per long sentence, so this runs fewer cases than the main TER test.
@budget(20, deep_examples=int(os.environ.get("SWADESH_TER_LONG_EXAMPLES", "10000")))
@given(seed=SEEDS)
def test_deep_ter_long_shifts(seed):
    r = random.Random(seed)
    opts = _ter_opts(r)
    count("TER-long " + _opt_label(opts)[4:], seed)
    hyps, refs = make_ter_long_case(r)
    _run("ter-long", ter_case, opts, hyps, refs)
