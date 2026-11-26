"""API and CLI parity with sacreBLEU 2.6.0 (criterion 5, `parity` track).

Every test runs the same call on sacrebleu and on swadesh and asserts the same
outcome: the same value (scores compared exactly, including every formatted
string, attribute and signature) or the same exception type and message.
Documented, intentional differences are in docs/parity-report.md and are not
asserted as equal here.

Environment:
  SACREBLEU_BIN   path to the sacrebleu CLI (default: next to sys.executable)
  SWADESH_DATA    dir with wmt16/wmt17 caches (default: ~/.sacrebleu)
"""

from __future__ import annotations

import copy
import inspect
import logging
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import sacrebleu
import sacrebleu.compat as sb_compat
import sacrebleu.significance as sb_sig
from sacrebleu.metrics import BLEU as SB_BLEU, CHRF as SB_CHRF, TER as SB_TER
from sacrebleu.metrics.bleu import BLEU as ORIG_BLEU_CLS
from sacrebleu.metrics.chrf import CHRF as ORIG_CHRF_CLS
from sacrebleu.metrics.ter import TER as ORIG_TER_CLS

import swadesh

SACREBLEU_BIN = Path(os.environ.get("SACREBLEU_BIN", Path(sys.executable).with_name("sacrebleu")))
SWADESH_DATA = Path(os.environ.get("SWADESH_DATA", Path.home() / ".sacrebleu"))

H = ["the cat sat on the mat", "a dog ran.", "Hello, World!", "", "x y z w"]
R = [
    ["the cat sat on a mat", "a dog ran .", "hello world", "", "x y z"],
    ["a cat", "a dog ran .", "hello, world!", "", "x"],
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def outcome(fn):
    """Run fn; return ("ok", value) or ("exc", type name, message)."""
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001 -- we compare every exception
        return ("exc", type(e).__name__, str(e))


_SCORE_ATTRS = ("name", "score", "counts", "totals", "precisions", "bp", "sys_len", "ref_len",
                "char_order", "word_order", "beta", "num_edits", "ref_length", "_mean", "_ci",
                "_verbose")
_FORMAT_CALLS = (
    dict(), dict(width=4), dict(width=0), dict(score_only=True), dict(width=0, score_only=True),
    dict(signature="sig|x:1"), dict(signature="sig|x:1", score_only=True),
    dict(is_json=True, signature="a:b"), dict(is_json=True, signature="a:b", width=3),
    dict(is_json=True),
)


def fingerprint(obj):
    """Everything observable about a Score / Signature / PairedTest result."""
    if hasattr(obj, "format") and hasattr(obj, "score"):
        attrs = {a: getattr(obj, a) for a in _SCORE_ATTRS if hasattr(obj, a)}
        fmts = [outcome(lambda kw=kw: obj.format(**kw)) for kw in _FORMAT_CALLS]
        return (type(obj).__name__, attrs, fmts, str(obj))
    if hasattr(obj, "format") and hasattr(obj, "info"):
        return ("Signature", str(obj), obj.format(short=True), dict(obj.info))
    if hasattr(obj, "p_value") and hasattr(obj, "score"):
        return ("Result", obj.score, obj.p_value, getattr(obj, "mean", None), getattr(obj, "ci", None))
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict):  # PairedTest output
        sigs, scores = obj
        return ({k: str(v) for k, v in sigs.items()},
                {k: [fingerprint(x) if not isinstance(x, str) else x for x in v] for k, v in scores.items()})
    return obj


def same(make):
    """Run make(module) on sacrebleu and on swadesh; assert identical outcome."""
    a = outcome(lambda: make(sacrebleu))
    b = outcome(lambda: make(swadesh))
    fa = (a[0], fingerprint(a[1])) if a[0] == "ok" else a
    fb = (b[0], fingerprint(b[1])) if b[0] == "ok" else b
    assert fa == fb, f"\n  sacrebleu: {fa}\n  swadesh:   {fb}"
    return a


def _sig_shape(fn):
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


# --------------------------------------------------------------------------
# 1. constructors: arguments and validation
# --------------------------------------------------------------------------

BLEU_KWARGS = [
    {}, dict(lowercase=True), dict(force=True), dict(tokenize="intl"), dict(tokenize="char"),
    dict(tokenize="none"), dict(tokenize="zh"), dict(smooth_method="none"),
    dict(smooth_method="floor"), dict(smooth_method="floor", smooth_value=0.3),
    dict(smooth_method="add-k", smooth_value=2), dict(smooth_method="exp", smooth_value=0.5),
    dict(max_ngram_order=1), dict(max_ngram_order=6), dict(effective_order=True),
    dict(smooth_method="floor", smooth_value=1),
    # invalid / unusual values: must raise or behave identically
    dict(smooth_method="bogus"), dict(tokenize="bogus"), dict(tokenize="ja-mecab"),
    dict(max_ngram_order=0), dict(max_ngram_order=-1), dict(max_ngram_order=2.0),
    dict(max_ngram_order="3"), dict(max_ngram_order=None), dict(smooth_value="x"),
    dict(lowercase="yes"), dict(force=None), dict(bogus=1),
]


@pytest.mark.parametrize("kw", BLEU_KWARGS, ids=lambda d: repr(d))
def test_bleu_constructor_and_score(kw):
    same(lambda m: m.BLEU(**kw).corpus_score(H, R))


@pytest.mark.parametrize("kw", BLEU_KWARGS, ids=lambda d: repr(d))
def test_bleu_constructor_signature_str(kw):
    same(lambda m: str(m.BLEU(**kw).get_signature()))


CHRF_KWARGS = [
    {}, dict(word_order=2), dict(beta=1), dict(char_order=4, word_order=1), dict(lowercase=True),
    dict(whitespace=True), dict(eps_smoothing=True), dict(char_order=1),
    dict(char_order=0), dict(char_order=-1), dict(word_order=-2), dict(beta=0), dict(beta=2.5),
    dict(beta="2"), dict(char_order=2.0), dict(char_order=20), dict(bogus=1),
]


@pytest.mark.parametrize("kw", CHRF_KWARGS, ids=lambda d: repr(d))
def test_chrf_constructor_and_score(kw):
    same(lambda m: m.CHRF(**kw).corpus_score(H, R))


@pytest.mark.parametrize("kw", CHRF_KWARGS, ids=lambda d: repr(d))
def test_chrf_signature(kw):
    same(lambda m: str(m.CHRF(**kw).get_signature()))


TER_KWARGS = [
    {}, dict(normalized=True), dict(no_punct=True), dict(asian_support=True),
    dict(case_sensitive=True), dict(normalized=True, no_punct=True, asian_support=True,
                                    case_sensitive=True), dict(normalized="yes"), dict(bogus=1),
]


@pytest.mark.parametrize("kw", TER_KWARGS, ids=lambda d: repr(d))
def test_ter_constructor_and_score(kw):
    same(lambda m: m.TER(**kw).corpus_score(H, R))


@pytest.mark.parametrize("kw", TER_KWARGS, ids=lambda d: repr(d))
def test_ter_signature(kw):
    same(lambda m: str(m.TER(**kw).get_signature()))


def test_bleu_positional_arguments_including_references():
    # sacreBLEU takes `references` as the 9th positional argument.
    same(lambda m: m.BLEU(False, False, None, "exp", None, 4, False, "", R).corpus_score(H, None).score)


def test_chrf_positional_arguments_including_references():
    same(lambda m: m.CHRF(4, 2, 1, True, True, True, R).corpus_score(H, None).score)


def test_ter_positional_arguments_including_references():
    same(lambda m: m.TER(True, False, True, False, R).corpus_score(H, None).score)


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_constructor_signatures_match(cls):
    a = _sig_shape(getattr(sacrebleu, cls).__init__)
    b = _sig_shape(getattr(swadesh, cls).__init__)
    assert [x[:2] + (x[2],) for x in a] == [x[:2] + (x[2],) for x in b]


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_constructor_parameter_annotations_match(cls):
    a = inspect.signature(getattr(sacrebleu, cls).__init__)
    b = inspect.signature(getattr(swadesh, cls).__init__)
    assert [p.annotation for p in a.parameters.values()] == [p.annotation for p in b.parameters.values()]


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_corpus_and_sentence_score_signatures(cls):
    for meth in ("corpus_score", "sentence_score"):
        a = _sig_shape(getattr(getattr(sacrebleu, cls), meth))
        b = _sig_shape(getattr(getattr(swadesh, cls), meth))
        assert [x[:2] for x in a] == [x[:2] for x in b], meth


def test_class_constants():
    assert swadesh.BLEU.SMOOTH_DEFAULTS == SB_BLEU.SMOOTH_DEFAULTS
    assert swadesh.BLEU.TOKENIZERS == SB_BLEU.TOKENIZERS
    assert swadesh.BLEU.TOKENIZER_DEFAULT == SB_BLEU.TOKENIZER_DEFAULT
    assert (swadesh.CHRF.CHAR_ORDER, swadesh.CHRF.WORD_ORDER, swadesh.CHRF.BETA) == (
        SB_CHRF.CHAR_ORDER, SB_CHRF.WORD_ORDER, SB_CHRF.BETA)


@pytest.mark.parametrize("tok,trg", [
    (None, "zh"), ("13a", "zh"), (None, "ja"), ("13a", "ja"), (None, "ko"), ("13a", "ko"),
])
def test_trg_lang_tokenizer_selection_and_warnings(tok, trg, caplog):
    def make(m):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="sacrebleu"):
            sig = str(m.BLEU(tokenize=tok, trg_lang=trg).get_signature())
        return sig, [r.getMessage() for r in caplog.records]
    same(make)


def test_unknown_smooth_method_message_is_sacrebleus_literal():
    # sacreBLEU's assert message is not an f-string; swadesh must reproduce it verbatim.
    a = outcome(lambda: SB_BLEU(smooth_method="x"))
    b = outcome(lambda: swadesh.BLEU(smooth_method="x"))
    assert a == b and a[0] == "exc" and a[1] == "AssertionError"


# --------------------------------------------------------------------------
# 2. corpus_score / sentence_score inputs
# --------------------------------------------------------------------------

CORPUS_INPUTS = {
    "plain": (H, R),
    "tuples": (tuple(H), tuple(tuple(x) for x in R)),
    "hyps_numpy": (np.array(H), R),
    "refs_numpy": (H, np.array(R)),
    "hyps_numpy_obj": (np.array(H, dtype=object), R),
    "hyps_str": ("abc", [["a", "b", "c"]]),
    "refs_str": (["a", "b"], "ab"),
    "refs_list_of_str": (["a"], ["ab"]),
    "empty_hyps": ([], R),
    "empty_refs_list": (H, []),
    "empty_ref_stream": (H[:2], [[]]),
    "empty_strings_only": ([""], [[""]]),
    "hyp_none_first": ([None, "a"], [["a", "b"]]),
    "hyp_none_later": (["a", None], [["a", "b"]]),
    "hyp_int_first": ([3, "a"], [["a", "b"]]),
    "hyp_int_later": (["a", 5], [["a", "b"]]),
    "hyp_bytes_later": (["a", b"b"], [["a", "b"]]),
    "ref_int_first": (["a"], [[3]]),
    "ref_int_later": (["a", "b"], [["a", "b"], ["a", 3]]),
    "ref_int_in_first_stream_later_segment": (["a", "b"], [["a", 7]]),
    "ref_none_first_nonempty_hyp": (["a", "b"], [[None, "b"]]),
    "ref_none_first_empty_hyp": (["", "b"], [[None, "b"]]),
    "ref_none_in_second_stream": (["a", "b"], [["a", "b"], [None, None]]),
    "ref_all_none_segment": (["a", "b"], [[None, "b"], [None, None]]),
    "hyps_longer_than_refs": (["a", "b", "c"], [["a", "b"]]),
    "hyps_shorter_than_refs": (["a"], [["a", "b"]]),
    "streams_mismatched": (["a", "b"], [["a", "b"], ["a"]]),
    "bad_ref_container": (["a"], [None]),
    "refs_as_flat_list_of_lists": (["a"], [["a"], "a"]),
}

@pytest.mark.parametrize("name", list(CORPUS_INPUTS))
@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_corpus_score_inputs(name, cls):
    hyps, refs = CORPUS_INPUTS[name]
    same(lambda m: getattr(m, cls)().corpus_score(hyps, refs))


@pytest.mark.parametrize("name", ["hyps_generator", "refs_generator"])
@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_corpus_score_generators(name, cls):
    def make(m):
        hyps, refs = {
            "hyps_generator": (iter(H), R),
            "refs_generator": (H, iter(R)),
        }[name]
        return getattr(m, cls)().corpus_score(hyps, refs)
    same(make)


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_corpus_score_references_none_without_cache(cls):
    same(lambda m: getattr(m, cls)().corpus_score(H, None))


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
@pytest.mark.parametrize("refs", [R, [], [[]], [[None, None]]])
def test_references_cache_truthiness(cls, refs):
    # references=[] must leave the cache empty (sacreBLEU), so corpus_score(refs=None) raises.
    same(lambda m: getattr(m, cls)(references=refs).corpus_score(H, None))


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_cached_references_then_explicit_references(cls):
    same(lambda m: (lambda x: (x.corpus_score(H, R).score, x.corpus_score(H, None).score))(
        getattr(m, cls)(references=R)))


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_cache_num_refs_signature(cls):
    same(lambda m: str(getattr(m, cls)(references=[["a", None], ["b", "c"]]).get_signature()))
    same(lambda m: str(getattr(m, cls)(references=[["a", "x"], ["b", "c"]]).get_signature()))


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_num_refs_after_scoring(cls):
    def make(m):
        x = getattr(m, cls)()
        x.corpus_score(H, [["a", None, "c", "d", "e"], ["b", "x", None, "d", "e"]])
        return x.get_signature().format(), vars(x).get("num_refs")
    same(make)


SENT_CASES = {
    "plain": ("a dog ran .", ["a dog ran ."]),
    "multi": ("a dog ran .", ["a dog ran .", "a cat ran ."]),
    "none_first": ("a dog ran .", [None, "a dog ran ."]),
    "none_later": ("a dog", ["a dog", None]),
    "all_none": ("a dog", [None]),
    "all_none_empty_hyp": ("", [None]),
    "two_none": ("a", [None, None, "a"]),
    "empty_refs": ("a", []),
    "empty_hyp": ("", ["a b"]),
    "empty_ref": ("a b", [""]),
    "int_ref": ("a", [3]),
    "int_ref_later": ("a", ["a", 3]),
    "int_hyp": (5, ["a"]),
    "none_hyp": (None, ["a"]),
    "list_hyp": (["a"], ["a"]),
    "str_refs": ("a", "a"),
    "tuple_refs": ("a b", ("a b",)),
    "numpy_refs": ("a", np.array(["a"])),
    "generator_refs": ("a", (x for x in ["a"])),
    "refs_none_entry_only": ("a", None),
}


@pytest.mark.parametrize("name", list(SENT_CASES))
@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_sentence_score_inputs(name, cls):
    hyp, refs = SENT_CASES[name]

    def make(m):
        x = getattr(m, cls)()
        return x.sentence_score(hyp, refs)
    same(make)


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_sentence_score_with_cache_and_none_refs(cls):
    same(lambda m: getattr(m, cls)(references=[["a b"]]).sentence_score("a b", [None]))


@pytest.mark.parametrize("eff", [False, True])
def test_bleu_sentence_effective_order_warning_is_logged_once(eff, caplog):
    def make(m):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="sacrebleu"):
            x = m.BLEU(effective_order=eff)
            x.sentence_score("a b", ["a b"])
        return [r.getMessage() for r in caplog.records]
    same(make)


@pytest.mark.parametrize("cls", ["BLEU", "CHRF"])
def test_bootstrap_corpus_score_ci(cls):
    same(lambda m: getattr(m, cls)().corpus_score(H, R, n_bootstrap=40).format())


@pytest.mark.parametrize("n", [0, 1, 2])
def test_n_bootstrap_edge_values(n):
    same(lambda m: m.BLEU().corpus_score(H, R, n_bootstrap=n).format())


# --------------------------------------------------------------------------
# 3. Score attributes, format(), signatures
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls,kw", [
    ("BLEU", {}), ("BLEU", dict(effective_order=True, smooth_method="floor")),
    ("CHRF", {}), ("CHRF", dict(word_order=2)), ("TER", {}), ("TER", dict(normalized=True)),
])
def test_score_objects_fully(cls, kw):
    same(lambda m: getattr(m, cls)(**kw).corpus_score(H, R))


def test_score_objects_from_sentence_and_bootstrap():
    same(lambda m: m.BLEU().sentence_score("the cat sat", ["the cat sat", "a cat"]))
    same(lambda m: m.CHRF().sentence_score("the cat sat", ["the cat sat", "a cat"]))
    same(lambda m: m.TER().sentence_score("the cat sat", ["the cat sat", "a cat"]))
    same(lambda m: m.CHRF(word_order=2).corpus_score(H, R, n_bootstrap=30))


@pytest.mark.parametrize("kwargs", [
    dict(), dict(width=4), dict(score_only=True), dict(is_json=True, signature="a:b|c:d"),
])
def test_format_kwargs_on_bootstrapped_score(kwargs):
    same(lambda m: m.BLEU().corpus_score(H, R, n_bootstrap=30).format(**kwargs))


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_signature_format_short(cls, short):
    same(lambda m: getattr(m, cls)(references=R).get_signature().format(short=short))


def test_signature_after_paired_test_style_update():
    def make(m):
        x = m.BLEU(references=R)
        sig = x.get_signature()
        sig.update("seed", "12345")
        sig.update("bs", 1000)
        return str(sig), sig.format(short=True)
    same(make)


def test_score_name_and_verbose():
    same(lambda m: (m.CHRF(beta=3, word_order=2).corpus_score(H, R).name,
                    m.BLEU().corpus_score(H, R)._verbose))


# --------------------------------------------------------------------------
# 4. compat functions
# --------------------------------------------------------------------------

COMPAT_CALLS = {
    "corpus_bleu": lambda c: c.corpus_bleu(H, R),
    "corpus_bleu_kw": lambda c: c.corpus_bleu(H, [R[0]], smooth_method="floor", smooth_value=0.3,
                                             force=True, lowercase=True, tokenize="intl",
                                             use_effective_order=True),
    "corpus_bleu_bad_tok": lambda c: c.corpus_bleu(H, R, tokenize="bogus"),
    "corpus_bleu_empty": lambda c: c.corpus_bleu([], [[]]),
    "corpus_bleu_none_refs": lambda c: c.corpus_bleu(H, [[None] * len(H)]),
    "raw_corpus_bleu": lambda c: c.raw_corpus_bleu(H, R, 0.1),
    "raw_corpus_bleu_default": lambda c: c.raw_corpus_bleu(H, R),
    "sentence_bleu": lambda c: c.sentence_bleu("a dog ran .", [R[0][1]]),
    "sentence_bleu_multi": lambda c: c.sentence_bleu("a dog ran .", [R[0][1], R[1][1]],
                                                    smooth_method="add-k"),
    "sentence_bleu_none": lambda c: c.sentence_bleu("a dog", [None]),
    "corpus_chrf": lambda c: c.corpus_chrf(H, R),
    "corpus_chrf_pos": lambda c: c.corpus_chrf(H, R, 6, 2, 2, False, False),
    "corpus_chrf_kw": lambda c: c.corpus_chrf(H, R, word_order=2, eps_smoothing=True),
    "sentence_chrf": lambda c: c.sentence_chrf("a dog", ["a dog", "a cat"]),
    "corpus_ter": lambda c: c.corpus_ter(H, R),
    "corpus_ter_kw": lambda c: c.corpus_ter(H, R, normalized=True, no_punct=True, asian_support=True,
                                           case_sensitive=True),
    "sentence_ter": lambda c: c.sentence_ter("a dog", ["a dog", "a cat"]),
    "sentence_ter_none": lambda c: c.sentence_ter("a dog", [None, "a cat"]),
    "corpus_bleu_nested_none": lambda c: c.corpus_bleu(H, [[None] * 5, R[0]]),
}


@pytest.mark.parametrize("name", list(COMPAT_CALLS))
def test_compat_functions(name):
    # Both modules expose the compat functions under the same names.
    same(lambda m: COMPAT_CALLS[name](m))


@pytest.mark.parametrize("name", ["corpus_bleu", "raw_corpus_bleu", "sentence_bleu", "corpus_chrf",
                                  "sentence_chrf", "corpus_ter", "sentence_ter"])
def test_compat_function_signatures(name):
    assert _sig_shape(getattr(sb_compat, name)) == _sig_shape(getattr(swadesh, name)), name


def test_compat_module_public_names():
    sb = {n for n in dir(sb_compat) if not n.startswith("_")}
    sw = {n for n in dir(swadesh) if not n.startswith("_")}
    assert {"corpus_bleu", "raw_corpus_bleu", "sentence_bleu", "corpus_chrf", "sentence_chrf",
            "corpus_ter", "sentence_ter"} <= sw
    assert {"corpus_bleu", "sentence_bleu"} <= sb


# --------------------------------------------------------------------------
# 5. significance: PairedTest and the kernels
# --------------------------------------------------------------------------

def _systems():
    return [("base", H), ("sys", ["the cat sat on a mat", "a dog ran .", "hello world", "", "x y q"])]


def _metrics(m):
    return {"bleu": m.BLEU(), "chrf": m.CHRF(word_order=2)}


@pytest.mark.parametrize("test_type,n", [("ar", 300), ("bs", 200)])
def test_paired_test_results(test_type, n):
    same(lambda m: m.significance.PairedTest(_systems(), _metrics(m), R, test_type=test_type,
                                             n_samples=n)())


def test_paired_test_ar_with_confidence_and_ter():
    same(lambda m: m.significance.PairedTest(_systems(), {"ter": m.TER()}, R, test_type="ar",
                                             n_samples=200, n_ar_confidence=50)())


def test_paired_test_default_samples_and_zero_confidence():
    same(lambda m: m.significance.PairedTest(_systems(), _metrics(m), R, test_type="ar",
                                             n_samples=0, n_ar_confidence=0).n_samples)


def test_paired_test_three_systems_and_attributes():
    def make(m):
        p = m.significance.PairedTest([("base", H), ("a", _systems()[1][1]), ("b", H[::-1])],
                                      _metrics(m), R, test_type="bs", n_samples=150)
        return (p.n_samples, p.n_ar_confidence, p.n_systems, p.test_type, fingerprint(p()))
    same(make)


def test_paired_test_cached_references():
    same(lambda m: m.significance.PairedTest(_systems(), {"bleu": m.BLEU(references=R)}, None,
                                             test_type="bs", n_samples=100)())


def test_paired_test_bad_type_message():
    same(lambda m: m.significance.PairedTest(_systems(), _metrics(m), R, test_type="zz"))


def test_paired_test_mismatched_system_length_raises_like_sacrebleu():
    systems = [("base", H), ("sys", H[:3])]
    same(lambda m: m.significance.PairedTest(systems, {"bleu": m.BLEU()}, R, test_type="ar",
                                             n_samples=50)())


def test_paired_bs_mismatched_system_length_raises_like_sacrebleu():
    systems = [("base", H), ("sys", H[:3])]
    same(lambda m: m.significance.PairedTest(systems, {"bleu": m.BLEU()}, R, test_type="bs",
                                             n_samples=50)())


def test_paired_test_is_standalone():
    """Phase 5: the test manager is swadesh's own, not a sacreBLEU subclass.

    The behaviour is still compared with the oracle elsewhere in this file; this asserts the
    dependency itself is gone, which is what the dual licence needs.
    """
    p = swadesh.significance.PairedTest(_systems(), _metrics(swadesh), R)
    assert not isinstance(p, sacrebleu.significance.PairedTest), (
        "PairedTest still inherits from sacreBLEU; the clean-room rule wants it standalone"
    )


def test_sacrebleu_result_and_estimate_ci():
    arr = np.array([1.0, 2.0, 3.5, 7.0, 2.2, 9.9, 4.4])
    same(lambda m: m.significance.estimate_ci(arr))
    same(lambda m: repr(m.significance.Result(1.2345, 0.03, 1.1, 0.4)))


# BLEU statistics for max order 4: [sys_len, ref_len, 4 correct, 4 totals]
BLEU_STATS = np.array([
    [9, 8, 4, 3, 2, 1, 9, 7, 6, 5],
    [7, 7, 3, 2, 1, 0, 8, 6, 4, 2],
    [10, 9, 5, 4, 3, 2, 10, 9, 8, 7],
])


@pytest.mark.parametrize("n", [1, 7, 40])
def test_bootstrap_resample_scores(n):
    same(lambda m: [x.score for x in m.significance._bootstrap_resample(BLEU_STATS, m.BLEU(), n)[1]])


@pytest.mark.parametrize("width", [8, 11])
def test_bootstrap_resample_malformed_width_raises_like_sacrebleu(width):
    stats = np.ones((3, width))
    same(lambda m: [x.score for x in m.significance._bootstrap_resample(stats, m.BLEU(), 5)[1]])


def test_bleu_np2_kernel_malformed_rows_raise_valueerror_not_panic():
    rows = np.ones((2, 8), dtype=np.float32).tobytes()
    with pytest.raises(ValueError):
        swadesh._swadesh.bleu_scores_np2_f32(rows, 8, 4, "exp", None, False)


def test_mismatched_ar_kernel_raises_valueerror_not_panic():
    """Rust bug repro: `ar_sums_i64` panics on mismatched shapes (BaseException); see report.

    The Python layer must never reach it with mismatched shapes (it falls back to numpy and
    raises sacreBLEU's ValueError), so this test pins the direct kernel and is an xfail."""
    bl = np.zeros((4, 8), dtype=np.int64).tobytes()
    sy = np.zeros((3, 8), dtype=np.int64).tobytes()
    sel = np.zeros((5, 3), dtype=np.uint8).tobytes()
    with pytest.raises(ValueError):
        swadesh._swadesh.ar_sums_i64(bl, sy, 8, sel, 4)


def test_bootstrap_kernel_out_of_range_index_raises_not_panics():
    stats = np.ones((3, 10), dtype=np.float32).tobytes()
    idxs = np.array([[0, 5, 1]], dtype=np.int64).tobytes()
    with pytest.raises((ValueError, IndexError)):
        swadesh._swadesh.bootstrap_sums_f32(stats, 10, idxs, 3)


def test_seeded_bootstrap_kernel_width_mismatch_raises_not_panics():
    stats = np.ones((3, 8), dtype=np.float32).tobytes()
    with pytest.raises(ValueError):
        swadesh._swadesh.bootstrap_sums_f32_seeded(stats, 10, 3, 4, 1)


def _run_isolated(code: str, timeout: int = 120) -> str:
    """Run `code` in a fresh interpreter in its own process group; kill the whole group on hang
    (forked pool workers that deadlock would otherwise outlive the test)."""
    import signal
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise AssertionError(f"hung: forked PairedTest workers did not finish within {timeout}s")
    assert p.returncode == 0, err[-2000:]
    return out


_PT_SCRIPT = """
import json, logging, sys
logging.disable(logging.CRITICAL)
import numpy as np
import {module} as m
import {module}.significance
if {warm}:
    import swadesh
    swadesh._swadesh.ar_sums_i64(np.zeros((2, 3), np.int64).tobytes(), np.zeros((2, 3), np.int64).tobytes(),
                                 3, np.zeros((4, 2), np.uint8).tobytes(), 2)
H = {H!r}
R = {R!r}
systems = [("base", H), ("sys", {H2!r})]
metrics = {{"bleu": m.BLEU(), "chrf": m.CHRF(word_order=2)}}
out = m.significance.PairedTest(systems, metrics, R, test_type="ar", n_samples=120, n_jobs=2)()
sigs, scores = out
print(json.dumps({{"sigs": {{k: str(v) for k, v in sigs.items()}},
                  "scores": {{k: [x if isinstance(x, str) else [x.score, x.p_value, x.mean, x.ci] for x in v] for k, v in scores.items()}}}}))
"""


_FORK_MIN_REPRO = """
import multiprocessing as mp
import numpy as np
from swadesh import _swadesh as N

def ar():
    N.ar_sums_i64(np.zeros((2, 3), np.int64).tobytes(), np.zeros((2, 3), np.int64).tobytes(), 3,
                  np.zeros((4, 2), np.uint8).tobytes(), 2)
    return "ok"

if __name__ == "__main__":
    ar()  # the parent runs the rayon-parallel kernel first ...
    with mp.get_context("fork").Pool(1) as pool:
        print(pool.apply_async(ar).get(timeout=20))  # ... then the forked child runs it too
"""


def test_forked_child_after_parent_parallel_kernel_does_not_hang():
    out = _run_isolated(_FORK_MIN_REPRO, timeout=60)
    assert out.strip() == "ok"


def _pt_parallel_script(module: str, warm: bool) -> str:
    return _PT_SCRIPT.format(module=module, warm=warm, H=H, R=R, H2=_systems()[1][1])


def _json_roundtrip(x):
    import json
    return json.loads(json.dumps(x))


def test_paired_test_parallel_workers_match_sacrebleu_fresh_process():
    # n_jobs=2 forks worker processes that run the swadesh kernels (multiprocessing users).
    import json
    expected = json.loads(_run_isolated(_pt_parallel_script("sacrebleu", warm=False)))
    got = json.loads(_run_isolated(_pt_parallel_script("swadesh", warm=False)))
    assert got == expected


def test_paired_test_parallel_workers_after_native_use_do_not_hang():
    import json
    expected = json.loads(_run_isolated(_pt_parallel_script("sacrebleu", warm=False)))
    got = json.loads(_run_isolated(_pt_parallel_script("swadesh", warm=True), timeout=90))
    assert got == expected


# --------------------------------------------------------------------------
# 6. pickling, copying, multiprocessing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("protocol", list(range(0, pickle.HIGHEST_PROTOCOL + 1)))
@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_pickle_all_protocols_with_cache(cls, protocol):
    def make(m):
        x = getattr(m, cls)(references=R)
        y = pickle.loads(pickle.dumps(x, protocol=protocol))
        return fingerprint(y.corpus_score(H, None)), str(y.get_signature())
    same(make)


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_pickle_without_cache_then_score(cls):
    def make(m):
        kw = {"smooth_method": "add-k"} if cls == "BLEU" else {}
        y = pickle.loads(pickle.dumps(getattr(m, cls)(**kw)))
        return fingerprint(y.corpus_score(H, R))
    same(make)


@pytest.mark.parametrize("cls", ["BLEU", "CHRF", "TER"])
def test_deepcopy_and_copy(cls):
    same(lambda m: fingerprint(copy.deepcopy(getattr(m, cls)(references=R)).corpus_score(H, None)))
    same(lambda m: fingerprint(copy.copy(getattr(m, cls)(references=R)).corpus_score(H, None)))


def test_pickled_score_objects():
    same(lambda m: fingerprint(pickle.loads(pickle.dumps(m.BLEU().corpus_score(H, R)))))
    same(lambda m: pickle.loads(pickle.dumps(m.BLEU(references=R).get_signature())).format())


def test_pickled_bootstrapped_score():
    same(lambda m: fingerprint(pickle.loads(pickle.dumps(m.BLEU().corpus_score(H, R, n_bootstrap=30)))))


def test_instance_attribute_names_match():
    assert sorted(vars(swadesh.BLEU(references=R))) == sorted(vars(SB_BLEU(references=R)))
    assert sorted(vars(swadesh.CHRF())) == sorted(vars(SB_CHRF()))
    assert sorted(vars(swadesh.TER())) == sorted(vars(SB_TER()))


def _fork_worker(args):
    cls, hyps, refs = args
    return fingerprint(getattr(swadesh, cls)().corpus_score(hyps, refs))


def test_fork_pool_corpus_scores():
    # multiprocessing users: a worker pool that scores with swadesh must match sacreBLEU.
    with mp.get_context("fork").Pool(2) as pool:
        got = pool.map(_fork_worker, [("BLEU", H, R), ("CHRF", H, R)])
    assert got == [fingerprint(SB_BLEU().corpus_score(H, R)), fingerprint(SB_CHRF().corpus_score(H, R))]


# --------------------------------------------------------------------------
# 7. patch() / unpatch()
# --------------------------------------------------------------------------

@pytest.fixture
def patched():
    swadesh.patch()
    try:
        yield
    finally:
        swadesh.unpatch()


def test_patch_rebinds_every_public_name():
    before = {
        "sacrebleu.BLEU": sacrebleu.BLEU, "sacrebleu.metrics.BLEU": sacrebleu.metrics.BLEU,
        "sacrebleu.metrics.bleu.BLEU": sacrebleu.metrics.bleu.BLEU,
        "sacrebleu.compat.BLEU": sb_compat.BLEU,
    }
    swadesh.patch()
    try:
        assert sacrebleu.BLEU is swadesh.BLEU and sacrebleu.CHRF is swadesh.CHRF and sacrebleu.TER is swadesh.TER
        assert sacrebleu.metrics.METRICS["BLEU"] is swadesh.BLEU
        assert sacrebleu.metrics.METRICS["CHRF"] is swadesh.CHRF
        assert sacrebleu.metrics.METRICS["TER"] is swadesh.TER
        assert sb_compat.BLEU is swadesh.BLEU
        assert sacrebleu.metrics.bleu.BLEU is swadesh.BLEU
        assert sacrebleu.significance._bootstrap_resample is swadesh.significance._bootstrap_resample
    finally:
        swadesh.unpatch()
    assert sacrebleu.BLEU is before["sacrebleu.BLEU"]
    assert sacrebleu.metrics.bleu.BLEU is before["sacrebleu.metrics.bleu.BLEU"]
    assert sacrebleu.metrics.METRICS["BLEU"] is ORIG_BLEU_CLS
    assert sacrebleu.metrics.METRICS["CHRF"] is ORIG_CHRF_CLS
    assert sacrebleu.metrics.METRICS["TER"] is ORIG_TER_CLS
    assert sb_compat.BLEU is ORIG_BLEU_CLS


def test_patch_is_idempotent_and_unpatch_is_idempotent():
    swadesh.patch()
    swadesh.patch()
    try:
        assert sacrebleu.BLEU is swadesh.BLEU
    finally:
        swadesh.unpatch()
        swadesh.unpatch()
    assert sacrebleu.BLEU is ORIG_BLEU_CLS
    assert sacrebleu.metrics.METRICS["BLEU"] is ORIG_BLEU_CLS


def test_unpatch_without_patch_is_noop():
    swadesh.unpatch()
    assert sacrebleu.BLEU is ORIG_BLEU_CLS


def test_patch_isinstance_checks(patched):
    assert isinstance(sacrebleu.BLEU(), sacrebleu.metrics.bleu.BLEU)
    assert isinstance(sacrebleu.BLEU(), ORIG_BLEU_CLS)  # subclass of the original class
    assert isinstance(swadesh.BLEU(), ORIG_BLEU_CLS)
    assert isinstance(sacrebleu.CHRF(), ORIG_CHRF_CLS)
    assert isinstance(sacrebleu.TER(), ORIG_TER_CLS)
    assert isinstance(ORIG_BLEU_CLS(), ORIG_BLEU_CLS)
    assert not isinstance(ORIG_BLEU_CLS(), swadesh.BLEU)  # objects made before patch stay plain


def test_patch_hf_evaluate_style_calls(patched):
    # HF `evaluate`'s bleu metric calls the module-level functions with lists of predictions
    # and a list of reference lists.
    got = sacrebleu.corpus_bleu(H, [R[0]])
    expected = SB_BLEU().corpus_score(H, [R[0]])
    assert fingerprint(got) == fingerprint(expected)
    assert fingerprint(sacrebleu.corpus_chrf(H, [R[0]], word_order=2)) == fingerprint(
        SB_CHRF(word_order=2).corpus_score(H, [R[0]]))
    assert sacrebleu.sentence_bleu("a dog ran .", [R[0][1]]).score == SB_BLEU().sentence_score(
        "a dog ran .", [R[0][1]]).score


def test_patch_significance_and_paired_test(patched):
    p = sacrebleu.significance.PairedTest(_systems(), {"bleu": sacrebleu.BLEU()}, R,
                                          test_type="bs", n_samples=100)
    _, scores = p()
    assert scores["BLEU"][0].score == SB_BLEU().corpus_score(H, R).score


def test_patched_module_functions_equal_unpatched_on_edge_inputs():
    expected = [outcome(lambda: sb_compat.corpus_bleu(*a)) for a in [([], [[]]), (H, R)]]
    swadesh.patch()
    try:
        got = [outcome(lambda: sacrebleu.compat.corpus_bleu(*a)) for a in [([], [[]]), (H, R)]]
    finally:
        swadesh.unpatch()
    assert [e[0] for e in got] == [e[0] for e in expected]
    assert [fingerprint(e[1]) if e[0] == "ok" else e[1:] for e in got] == [
        fingerprint(e[1]) if e[0] == "ok" else e[1:] for e in expected]


# --------------------------------------------------------------------------
# 8. CLI: `python -m swadesh.cli` against the sacrebleu binary
# --------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Mask the program name (sacrebleu vs swadesh, the one intentional CLI difference).

    argparse wraps usage lines at the program name's length, so usage and error text is
    compared with whitespace collapsed as well. Plain output (scores, tables, echoed data)
    is compared byte for byte."""
    # "python -m swadesh.cli" (3.12+ sets __main__.__spec__) and the bare entry-point names.
    text = re.sub(r"\bpython(\d+([.]\d+)*)? -m (sacrebleu|swadesh(cli)?(\.[a-z_]+)?)(?=[: ])", "PROG", text)
    text = re.sub(r"\b(sacrebleu|swadesh|swadesh\.cli|cli\.py)(?=[: ])", "PROG", text)
    text = re.sub(r"\A(PROG) (\d+\.\d+\.\d+)\n", r"PROG \2\n", text)
    if "usage:" in text or "error:" in text:
        text = " ".join(text.split())
    return text


def run_cli(argv, prog, stdin=None, cwd=None):
    env = {k: v for k, v in os.environ.items() if k not in ("SACREBLEU_FORMAT", "NO_COLOR")}
    env["SACREBLEU_SEED"] = "12345"
    p = subprocess.run(prog + argv, input=stdin, capture_output=True, text=True, env=env, cwd=cwd,
                       timeout=600)
    return p.returncode, p.stdout, p.stderr


SB_PROG = [str(SACREBLEU_BIN)]
SW_PROG = [sys.executable, "-m", "swadesh.cli"]


@pytest.fixture(scope="module")
def cli_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    refs = d / "refs.txt"
    hyps = d / "hyps.txt"
    refs2 = d / "refs2.txt"
    src = SWADESH_DATA / "wmt16" / "wmt16.en-de.ref"
    if not src.exists():
        pytest.skip(f"wmt16 en-de not cached at {src}")
    lines = src.read_text(encoding="utf-8").splitlines()[:200]
    refs.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # perturbed hypotheses: drop the last word of every other line, and one tokenized-looking line
    hyp_lines = [" ".join(l.split()[:-1]) if i % 2 else l for i, l in enumerate(lines)]
    hyp_lines[3] = hyp_lines[3] + " ."
    hyps.write_text("\n".join(hyp_lines) + "\n", encoding="utf-8")
    refs2.write_text("\n".join(l.upper() for l in lines) + "\n", encoding="utf-8")
    return {"refs": refs, "hyps": hyps, "refs2": refs2, "dir": d}


CLI_ARGS = [
    ["--help"], ["-h"], ["--version"], ["-V"], ["--citation"], ["--cite"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "src"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "ref"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "docid"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "ref", "src"],
    ["-t", "wmt16", "-l", "en-de", "--echo", "all"],
    ["-t", "wmt16", "-l", "en-de", "-ol", "en", "--echo", "src"],
    ["-t", "wmt16", "-l", "en-de", "-ol", "non-en", "--echo", "src"],
    ["-t", "wmt16", "-l", "en-de", "--origlang", "en", "--echo", "docid"],
    ["-t", "wmt16", "-l", "en-de", "--subset", "europarl", "--echo", "src"],
    ["-t", "wmt16", "--list"],
    ["--list"],
    ["-t", "nonexistent-set", "-l", "en-de", "--echo", "src"],
    ["-t", "wmt16", "-l", "en-xx", "--echo", "src"],
]

CLI_SCORING = [
    [], ["-m", "bleu", "chrf", "ter"], ["-s", "none"], ["-s", "floor", "-sv", "0.2"],
    ["-s", "add-k"], ["-s", "add-k", "-sv", "3"], ["-tok", "intl"], ["-tok", "char"],
    ["-tok", "zh"], ["-tok", "none"], ["-lc"], ["--force"], ["-sl"], ["-sl", "-m", "chrf"],
    ["-cc", "4", "-cw", "2", "--chrf-beta", "1"], ["--chrf-whitespace", "--chrf-lowercase"],
    ["--chrf-eps-smoothing", "-cw", "2"], ["-m", "ter", "--ter-normalized"],
    ["-m", "ter", "--ter-case-sensitive", "--ter-asian-support", "--ter-no-punct"],
    ["-ci", "-cin", "50"], ["-ci"], ["-q"], ["-sh"], ["-b"], ["-w", "4"], ["-d"], ["-nc"],
    ["-f", "text"], ["-f", "latex"], ["-f", "json"], ["-sh", "-f", "latex"], ["--short"],
    ["-b", "-w", "5"], ["-d", "-m", "bleu", "chrf"], ["-e", "utf-8"], ["-nr", "1"],
    ["-s", "bogus"], ["-tok", "bogus"], ["-m", "nope"], ["-w", "x"], ["-cc", "0"],
    ["-cw", "-1"], ["--chrf-beta", "0"], ["-ci", "-cin", "0"],
    ["-par", "-parn", "50", "-j", "1"], ["-pbs", "-pbsn", "50"],
    ["-par", "-pbs"], ["-par", "-ci"], ["-m", "bleu", "-sv", "0.1", "-s", "exp"],
]


def _cli_compare(argv, cli_files, stdin=None):
    rc_sb, out_sb, err_sb = run_cli(argv, SB_PROG, stdin=stdin, cwd=cli_files["dir"])
    rc_sw, out_sw, err_sw = run_cli(argv, SW_PROG, stdin=stdin, cwd=cli_files["dir"])
    assert rc_sw == rc_sb, (argv, rc_sb, rc_sw, err_sb[-400:], err_sw[-400:])
    assert _norm(out_sw) == _norm(out_sb), argv
    if "Traceback (most recent call last)" in err_sb:
        # Uncaught exceptions: the frames differ (different entry point); the exception line must not.
        assert err_sw.strip().splitlines()[-1] == err_sb.strip().splitlines()[-1], argv
    else:
        assert _norm(err_sw) == _norm(err_sb), argv


@pytest.mark.parametrize("argv", CLI_ARGS, ids=lambda a: " ".join(a))
def test_cli_informational_flags(argv):
    _cli_compare(argv, {"dir": Path(".")})


@pytest.mark.parametrize("argv", CLI_SCORING, ids=lambda a: " ".join(a) or "default")
def test_cli_scoring_flags(argv, cli_files):
    base = [str(cli_files["refs"]), "-i", str(cli_files["hyps"])]
    _cli_compare(base + argv, cli_files)


@pytest.mark.xfail(strict=True, reason=(
    "documented divergence: sacreBLEU accepts `-tok ja-mecab` and fails later, at import time, with "
    "exit code 1 and a traceback ending in `    pip install sacrebleu[ja]`; swadesh refuses an "
    "unbuilt tokenizer up front with a hint and exit code 2 (docs/parity-report.md)"))
def test_cli_unbuilt_tokenizer(cli_files):
    _cli_compare([str(cli_files["refs"]), "-i", str(cli_files["hyps"]), "-tok", "ja-mecab"], cli_files)


@pytest.mark.parametrize("argv", [
    ["-sl", "-m", "ter", "-tok", "intl"],
    ["-sl", "-m", "bleu", "-s", "floor"],
    ["-m", "bleu", "chrf", "-par", "-parn", "200"],
    ["-m", "bleu", "-pbs", "-pbsn", "200"],
    ["-m", "chrf", "-ci", "-cin", "30", "-f", "json"],
], ids=lambda a: " ".join(a))
def test_cli_stdin_and_multi_refs(argv, cli_files):
    stdin = cli_files["hyps"].read_text(encoding="utf-8")
    _cli_compare([str(cli_files["refs"]), str(cli_files["refs2"])] + argv, cli_files, stdin=stdin)


def test_cli_dataset_flow_with_cached_wmt16(cli_files):
    if not (SWADESH_DATA / "wmt16" / "wmt16.en-de.ref").exists():
        pytest.skip("wmt16 en-de not cached")
    ref_full = run_cli(["-t", "wmt16", "-l", "en-de", "--echo", "ref"], SB_PROG,
                       cwd=cli_files["dir"])[1]
    hyp = cli_files["dir"] / "wmt16_hyp.txt"
    hyp.write_text("\n".join(l[: max(1, len(l) // 2)] for l in ref_full.splitlines()) + "\n",
                   encoding="utf-8")
    _cli_compare(["-t", "wmt16", "-l", "en-de", "-i", str(hyp)], cli_files)
    _cli_compare(["-t", "wmt16", "-l", "en-de", "-i", str(hyp), "-m", "bleu", "chrf", "ter"],
                 cli_files)
    _cli_compare(["-t", "wmt16", "-l", "en-de", "-i", str(hyp), "-sl", "-m", "chrf"], cli_files)
    _cli_compare(["-t", "wmt16", "-l", "en-de", "-i", str(hyp), "-ci", "-cin", "100"], cli_files)


def test_cli_invalid_utf8_exit_code_matches(cli_files, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe bad\n")
    _cli_compare([str(cli_files["refs"]), "-i", str(bad)], cli_files)


def test_cli_missing_input_file_exit_code_matches(cli_files):
    _cli_compare([str(cli_files["refs"]), "-i", str(cli_files["dir"] / "nope.txt")], cli_files)

