"""The fallback statistics methods are swadesh's own and reproduce the oracle exactly.

`BLEU` and `CHRF` now define `_extract_reference_info` and `_compute_segment_statistics` themselves,
so the fallback loop in `_FastMixin._py_statistics` no longer runs sacreBLEU's method bodies. That
loop is reached only when a reference stream leaves a segment with no reference, when a parameter is
outside what the native kernel takes, or when the kernel raises; the native corpus path never calls
either method.

Every constant below was measured from the inherited methods before the port (black-box calls on
objects only; no source was read) and re-checked against a live `sacrebleu.BLEU`/`sacrebleu.CHRF`.
Comparisons are exact: rows are integer lists, scores are `str()` plus `repr()` of the float, and
errors are the exception type and message.
"""
from __future__ import annotations

import numpy as np
import pytest
import sacrebleu
import sacrebleu.metrics.bleu as sacre_bleu
import sacrebleu.metrics.chrf as sacre_chrf

import swadesh
from swadesh.metrics import BLEU, CHRF

H = ["the cat sat on the mat", "a dog ran .", "hello world"]
R_ONE = ["the cat sat on a mat", "a dog ran .", "hello world"]
R_EXACT = ["the cat sat on the mat", "a dog ran .", "hello world"]
R_TWO = [R_ONE, R_EXACT]
# A segment with no reference at all in any stream: the corpus-level fallback trigger. Its
# hypothesis is empty, so BLEU still has a well-formed row for it.
H_EMPTY = ["the cat sat on the mat", "", "hello world"]
R_EMPTY = [["the cat sat on a mat", None, "hello world"], [None, None, None]]
R_EMPTY_TWO = [["the cat sat on a mat", None, "hello world"],
               ["the cat sat on the mat", None, "hello world"]]
# A four-stream corpus with a None tail leaves each segment one reference: no fallback, only the
# None entries being dropped.
R_NONE_TAIL = [R_ONE, [None, None, None]]


def _rows_ours(metric, hyps, streams):
    try:
        return [[int(x) for x in row] for row in metric._py_statistics(hyps, streams)]
    except Exception as e:  # the exception is part of the behaviour being pinned
        return (type(e).__name__, str(e))


def _rows_live(metric, hyps, streams):
    try:
        return [[int(x) for x in row]
                for row in metric._extract_corpus_statistics(hyps, streams)]
    except Exception as e:  # the exception is part of the behaviour being pinned
        return (type(e).__name__, str(e))


def _none_membership_message():
    try:
        "x" in None  # noqa: B015 -- the oracle's error comes from exactly this expression
    except TypeError as e:
        return str(e)
    raise AssertionError("membership against None stopped raising")


NONE_MEMBERSHIP_MESSAGE = _none_membership_message()


def _text(fn):
    """The printed score, or the (type, message) of the error; both are version-stable."""
    try:
        return (str(fn()),)
    except Exception as e:  # the exception is part of the behaviour being pinned
        return (type(e).__name__, str(e))


def _outcome(fn):
    try:
        score = fn()
        return (str(score), repr(float(score.score)))
    except Exception as e:  # the exception is part of the behaviour being pinned
        return (type(e).__name__, str(e))


def _error(fn):
    try:
        fn()
    except Exception as e:  # the exception is the observable
        return (type(e).__name__, str(e))
    return "no error"


# --------------------------------------------------------- the methods are swadesh's own

def test_statistics_methods_belong_to_swadesh():
    for cls, parent in ((BLEU, sacre_bleu.BLEU), (CHRF, sacre_chrf.CHRF)):
        for name in ("_extract_reference_info", "_compute_segment_statistics"):
            assert name in cls.__dict__, name
            assert getattr(cls, name) is not getattr(parent, name), name


def test_compute_score_from_stats_is_swadeshs_own():
    """Both scorers are ours now.

    They used to be sacreBLEU's. The bootstrap fast path keys off the attributes the kernel reads
    rather than on class or function identity (see `significance._native_bleu_scores`), so porting
    them does not take BLEU or chrF off that path -- `test_compute_score_from_stats_matches_sacrebleu`
    below holds the ported arithmetic to sacreBLEU's.
    """
    for cls, parent in ((BLEU, sacre_bleu.BLEU), (CHRF, sacre_chrf.CHRF)):
        assert "_compute_score_from_stats" in cls.__dict__, cls.__name__
        assert getattr(cls, "_compute_score_from_stats") is not getattr(
            parent, "_compute_score_from_stats"), cls.__name__


def test_compute_score_from_stats_matches_sacrebleu():
    """The ported fallback scorers have to agree with sacreBLEU's on the rows they exist for.

    Rows too short for the metric's order are included on purpose: both implementations then fail
    the same way, and the exception is part of the contract.
    """
    def grab(fn):
        try:
            return ("ok", fn())
        except Exception as e:  # noqa: BLE001 -- the exception is the expectation
            return ("exc", type(e).__name__, str(e))

    def bleu_rows(n):
        w = 2 + 2 * n
        rows = [
            [3, 4] + [1, 0, 0, 0][:n] + [1, 2, 3, 4][:n],          # no matches (issue #141 path)
            [12, 10] + [8, 6, 4, 0][:n] + [9, 7, 5, 0][:n],        # ordinary
            [0, 0] + [0] * n + [0] * n,                            # both empty
            [5, 5] + [3.5, 2.25, 0.0, 0.1][:n] + [4.0, 3.5, 0.0, 0.0][:n],  # fractional float32 row
            [7, 3] + [1, 0, 0, 0][:n] + [4, 0, 0, 0][:n],          # short segment, zero higher orders
        ]
        for r in rows:
            assert len(r) == w
        rows.append([1, 1, 1])                                     # deliberately too short
        return rows

    for kw in (dict(), dict(smooth_method="add-k"), dict(smooth_method="floor"),
               dict(smooth_method="none"), dict(smooth_method="add-k", smooth_value=0.5),
               dict(smooth_method="floor", smooth_value=0.25), dict(max_ngram_order=2),
               dict(max_ngram_order=1), dict(effective_order=True),
               dict(effective_order=True, smooth_method="add-k")):
        ours, theirs = BLEU(**kw), sacre_bleu.BLEU(**kw)
        for row in bleu_rows(ours.max_ngram_order):
            a = grab(lambda: ours._compute_score_from_stats(list(row)))
            b = grab(lambda: theirs._compute_score_from_stats(list(row)))
            if a[0] == "ok" and b[0] == "ok":
                assert (a[1].score, repr(a[1])) == (b[1].score, repr(b[1])), (kw, row)
            else:
                assert a[0] == b[0], (kw, row, a, b)

    for kw in (dict(), dict(eps_smoothing=True), dict(beta=1), dict(beta=3),
               dict(char_order=2, word_order=2, beta=3, lowercase=True, whitespace=True)):
        ours, theirs = CHRF(**kw), sacre_chrf.CHRF(**kw)
        order = ours.order
        for row in ([4, 4, 2] * order, [0] * (3 * order), [3, 0, 0] * order, [0, 3, 0] * order,
                    [1, 1, 1]):
            a = grab(lambda: ours._compute_score_from_stats(list(row)))
            b = grab(lambda: theirs._compute_score_from_stats(list(row)))
            if a[0] == "ok" and b[0] == "ok":
                assert (a[1].score, repr(a[1])) == (b[1].score, repr(b[1])), (kw, row)
            else:
                assert a[0] == b[0], (kw, row, a, b)


def test_the_fallback_loop_is_the_path_under_test():
    metric = BLEU()
    stats = metric._extract_corpus_statistics(H_EMPTY, R_EMPTY)
    assert isinstance(stats, list) and isinstance(stats[0], list)
    assert metric.num_refs == -1
    assert isinstance(BLEU()._extract_corpus_statistics(H, [R_ONE]), np.ndarray)


# --------------------------------------------------------- BLEU rows

BLEU_ROW_CASES = [
    ("perfect", {}, ["the cat sat on the mat"], [["the cat sat on the mat"]],
     [[6, 6, 6, 5, 4, 3, 6, 5, 4, 3]]),
    ("multi_ref_clips_to_best", {}, ["a a a a"], [["a a"], ["a a a"]],
     [[4, 3, 3, 2, 1, 0, 4, 3, 2, 1]]),
    ("closest_ref_len_tie_goes_short", {}, ["a b c d"], [["a b c"], ["a b c d e"]],
     [[4, 3, 4, 3, 2, 1, 4, 3, 2, 1]]),
    ("none_dropped_before_counting", {}, ["a b"], [[None], ["a b"]],
     [[2, 2, 2, 1, 0, 0, 2, 1, 0, 0]]),
    ("zero_references_empty_hypothesis", {}, [""], [[None]],
     [[0, -1, 0, 0, 0, 0, 0, 0, 0, 0]]),
    ("counts_precede_totals_char_tokenizer", {"tokenize": "char"}, ["a b"], [["a b"]],
     [[2, 2, 2, 1, 0, 0, 2, 1, 0, 0]]),
    ("lowercase", {"lowercase": True}, ["The Cat"], [["the cat"]],
     [[2, 2, 2, 1, 0, 0, 2, 1, 0, 0]]),
    ("intl", {"tokenize": "intl"}, ["a-b 3.5%"], [["a-b 3.5%"]],
     [[4, 4, 4, 3, 2, 1, 4, 3, 2, 1]]),
    ("zh", {"tokenize": "zh"}, ["你好 世界"], [["你好世界"]],
     [[4, 4, 4, 3, 2, 1, 4, 3, 2, 1]]),
    ("two_orders", {"max_ngram_order": 2}, ["a b c"], [["a b c"]],
     [[3, 3, 3, 2, 3, 2]]),
    ("zero_orders", {"max_ngram_order": 0}, ["a b c"], [["a b"]], [[3, 2]]),
    ("negative_orders", {"max_ngram_order": -1}, ["a b c"], [["a b"]], [[3, 2]]),
    ("force_tokenized_period", {"force": True}, ["word 1 ."], [["word 1"]],
     [[3, 2, 2, 1, 0, 0, 3, 2, 1, 0]]),
    ("smooth_method_does_not_change_the_row", {"smooth_method": "none"}, ["a b"], [["a c"]],
     [[2, 2, 1, 0, 0, 0, 2, 1, 0, 0]]),
    ("zero_references_nonempty_hypothesis", {}, ["a"], [[None]],
     ("TypeError", NONE_MEMBERSHIP_MESSAGE)),
    ("fractional_order", {"max_ngram_order": 2.5}, ["a"], [["a"]],
     ("TypeError", "'float' object cannot be interpreted as an integer")),
    ("string_order", {"max_ngram_order": "x"}, ["a"], [["a"]],
     ("TypeError", 'can only concatenate str (not "int") to str')),
]

CHRF_ROW_CASES = [
    ("default_six_char_orders", {}, ["abc"], [["abc"]],
     [[3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    ("best_reference_not_a_merge", {"char_order": 2, "word_order": 0}, ["ab"],
     [["ab"], ["abc"]], [[2, 2, 2, 1, 1, 1]]),
    ("best_reference_order_does_not_matter", {"char_order": 2, "word_order": 0}, ["ab"],
     [["abc"], ["ab"]], [[2, 2, 2, 1, 1, 1]]),
    ("f_tie_keeps_first_reference", {"char_order": 1, "word_order": 0}, ["acaa"],
     [["abcba"], ["abb"], ["a"]], [[4, 5, 3]]),
    ("f_tie_keeps_first_reference_words", {"char_order": 1, "word_order": 2}, ["baabcc"],
     [["caaabc"], ["bab"], ["bb"]], [[6, 6, 5, 1, 1, 0, 0, 0, 0]]),
    ("word_orders", {"char_order": 2, "word_order": 2}, ["ab cd ef"], [["ab cd ef"]],
     [[6, 6, 6, 5, 5, 5, 3, 3, 3, 2, 2, 2]]),
    ("whitespace_kept", {"char_order": 2, "word_order": 0, "whitespace": True}, [" ab "],
     [[" ab "]], [[4, 4, 4, 3, 3, 3]]),
    ("whitespace_dropped", {"char_order": 2, "word_order": 0}, [" ab "], [[" ab "]],
     [[2, 2, 2, 1, 1, 1]]),
    ("hypothesis_total_zero_without_reference_order", {"char_order": 2, "word_order": 0},
     ["ab"], [[""]], [[0, 0, 0, 0, 0, 0]]),
    ("eight_char_orders", {"char_order": 8, "word_order": 0}, ["abc"], [["abc"]],
     [[3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    ("word_orders_only", {"char_order": 0, "word_order": 2}, ["ab cd"], [["ab cd"]],
     [[2, 2, 2, 1, 1, 1]]),
    ("negative_char_order_with_words", {"char_order": -1, "word_order": 2}, ["ab"], [["ab"]],
     [[1, 1, 1, 0, 0, 0]]),
    ("empty_hypothesis", {}, [""], [["abc"]],
     [[0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    ("three_references", {"char_order": 6, "word_order": 2}, ["the cat sat"],
     [["the cat sat"], ["a cat sat"], ["the dog sat"]],
     [[9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2]]),
    ("fractional_beta_changes_the_winner", {"char_order": 1, "word_order": 0, "beta": 0.5},
     ["baa"], [["acbacb"], ["a"]], [[3, 6, 3]]),
    ("integral_beta_keeps_the_other_reference", {"char_order": 1, "word_order": 0, "beta": 2},
     ["baa"], [["acbacb"], ["a"]], [[3, 1, 1]]),
    ("whitespace_bytes", {"whitespace": True, "char_order": 1, "word_order": 0}, ["a", b"b"],
     [["a", "b"]], [[1, 1, 1], [1, 1, 0]]),
    ("integer_hypothesis", {}, [3], [["ab"]],
     ("AttributeError", "'int' object has no attribute 'split'")),
    ("bytes_hypothesis", {}, [b"ab"], [["ab"]],
     ("TypeError", "sequence item 0: expected str instance, bytes found")),
    ("integer_reference", {}, ["a"], [[3]],
     ("AttributeError", "'int' object has no attribute 'split'")),
    ("fractional_char_order", {"char_order": 2.5}, ["a"], [["a"]],
     ("TypeError", "'float' object cannot be interpreted as an integer")),
    ("fractional_word_order", {"word_order": 2.5}, ["a"], [["a"]],
     ("TypeError", "'float' object cannot be interpreted as an integer")),
]


@pytest.mark.parametrize("kwargs,hypotheses,streams,expected", [c[1:] for c in BLEU_ROW_CASES],
                         ids=[c[0] for c in BLEU_ROW_CASES])
def test_bleu_fallback_rows(kwargs, hypotheses, streams, expected):
    assert _rows_ours(BLEU(**kwargs), hypotheses, streams) == expected
    assert _rows_live(sacrebleu.BLEU(**kwargs), hypotheses, streams) == expected


@pytest.mark.parametrize("kwargs,hypotheses,streams,expected", [c[1:] for c in CHRF_ROW_CASES],
                         ids=[c[0] for c in CHRF_ROW_CASES])
def test_chrf_fallback_rows(kwargs, hypotheses, streams, expected):
    assert _rows_ours(CHRF(**kwargs), hypotheses, streams) == expected
    assert _rows_live(sacrebleu.CHRF(**kwargs), hypotheses, streams) == expected


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 8])
def test_bleu_row_layout(n):
    rows = BLEU(max_ngram_order=n)._py_statistics(["a b c d e f"], [["a b c d e f"]])
    assert len(rows[0]) == 2 + 2 * n


@pytest.mark.parametrize("char_order,word_order", [(1, 0), (6, 0), (0, 2), (6, 2), (3, 3), (8, 0)])
def test_chrf_row_layout(char_order, word_order):
    rows = CHRF(char_order=char_order, word_order=word_order)._py_statistics(
        ["ab cd"], [["ab cd"]])
    assert len(rows[0]) == 3 * (char_order + word_order)


def test_bleu_reference_info_shape():
    metric = BLEU()
    assert metric._extract_reference_info([]) == {"ref_ngrams": None, "ref_lens": []}
    info = metric._extract_reference_info(["a a a", "a"])
    assert info["ref_lens"] == [3, 1]
    # the maximum count over references, not their sum ('a' appears 4 times across both refs)
    assert dict(info["ref_ngrams"]) == {("a",): 3, ("a", "a"): 2, ("a", "a", "a"): 1}
    assert metric._extract_reference_info(["the cat", ""])["ref_lens"] == [2, 0]


def test_chrf_reference_info_shape():
    info = CHRF(char_order=2, word_order=1)._extract_reference_info(["ab cd", ""])
    assert [[len(counter) for counter in per_ref] for per_ref in info["ref_ngrams"]] == \
        [[4, 3, 2], [0, 0, 0]]
    assert CHRF()._extract_reference_info([]) == {"ref_ngrams": []}


# --------------------------------------------------------- scores through the fallback

SCORE_CASES = [
    ("bleu_has_empty_segment", "BLEU", {}, H_EMPTY, R_EMPTY,
     ("BLEU = 55.84 87.5/66.7/50.0/33.3 (BP = 1.000 ratio = 1.143 hyp_len = 8 ref_len = 7)",)),
    ("bleu_has_empty_segment_two_refs", "BLEU", {}, H_EMPTY, R_EMPTY_TWO,
     ("BLEU = 100.00 100.0/100.0/100.0/100.0 (BP = 1.000 ratio = 1.143 hyp_len = 8 ref_len = 7)",)),
    ("bleu_zero_orders", "BLEU", {"max_ngram_order": 0}, H, [R_ONE],
     ("BLEU = 0.00  (BP = 1.000 ratio = 1.000 hyp_len = 12 ref_len = 12)",)),
    ("bleu_negative_orders", "BLEU", {"max_ngram_order": -1}, H, [R_ONE],
     ("BLEU = 0.00  (BP = 1.000 ratio = 1.000 hyp_len = 12 ref_len = 12)",)),
    ("chrf_reads_ragged_rows", "CHRF", {}, H_EMPTY, R_EMPTY,
     ("IndexError", "list index out of range")),
    ("chrf_fractional_beta", "CHRF", {"beta": 2.5}, H, [R_ONE],
     ("chrF2.5 = 85.92",)),
    ("chrf_fractional_beta_pp", "CHRF", {"beta": 2.5, "char_order": 6, "word_order": 2}, H, R_TWO,
     ("chrF2.5++ = 100.00",)),
    ("chrf_fractional_beta_whitespace", "CHRF", {"beta": 2.5, "whitespace": True}, H, [R_ONE],
     ("chrF2.5 = 88.76",)),
    ("chrf_fractional_beta_eps", "CHRF", {"beta": 2.5, "eps_smoothing": True}, H, [R_ONE],
     ("chrF2.5 = 85.92",)),
    ("chrf_zero_beta", "CHRF", {"beta": 0}, H, [R_ONE], ("chrF0 = 80.42",)),
    ("chrf_word_orders_only", "CHRF", {"char_order": 0, "word_order": 2}, H, [R_ONE],
     ("chrF2++ = 84.72",)),
    ("chrf_whitespace_bytes", "CHRF", {"whitespace": True}, ["a", b"b"], [["a", "b"]],
     ("chrF2 = 50.00",)),
]


@pytest.mark.parametrize("cls_name,kwargs,hyps,refs,expected", [c[1:] for c in SCORE_CASES],
                         ids=[c[0] for c in SCORE_CASES])
def test_fallback_corpus_score_constants(cls_name, kwargs, hyps, refs, expected):
    cls = getattr(swadesh, cls_name)
    assert _text(lambda: cls(**kwargs).corpus_score(hyps, refs)) == expected


TRAP_CASES = [
    ("bleu_smooth_value_divides", "BLEU", {"smooth_method": "floor", "smooth_value": "x"},
     ["a"], [["b"]],
     ("BLEU = 0.00 0.0/0.0/0.0/0.0 (BP = 1.000 ratio = 1.000 hyp_len = 1 ref_len = 1)",)),
    ("bleu_add_k_none_value", "BLEU", {"smooth_method": "add-k", "smooth_value": None},
     ["a"], [["b"]],
     ("BLEU = 0.00 0.0/0.0/0.0/0.0 (BP = 1.000 ratio = 1.000 hyp_len = 1 ref_len = 1)",)),
    ("bleu_int_hypothesis", "BLEU", {}, ["a", 5], [["a", "b"]],
     ("AttributeError", "'int' object has no attribute 'endswith'")),
    ("chrf_int_hypothesis", "CHRF", {}, ["a", 5], [["a", "b"]],
     ("AttributeError", "'int' object has no attribute 'split'")),
]


@pytest.mark.parametrize("cls_name,kwargs,hyps,refs,expected", [c[1:] for c in TRAP_CASES],
                         ids=[c[0] for c in TRAP_CASES])
def test_fallback_trap_constants(cls_name, kwargs, hyps, refs, expected):
    cls = getattr(swadesh, cls_name)
    assert _text(lambda: cls(**kwargs).corpus_score(hyps, refs)) == expected


SENTENCE_CASES = [
    ("bleu_none_tail_falls_back_to_the_real_ref", "BLEU", {}, "a b", ["a b", None],
     ("BLEU = 0.00 100.0/100.0/0.0/0.0 (BP = 1.000 ratio = 1.000 hyp_len = 2 ref_len = 2)",)),
    ("bleu_only_none_is_the_error_path", "BLEU", {}, "a b", [None],
     ("TypeError", NONE_MEMBERSHIP_MESSAGE)),
    ("chrf_only_none_is_the_error_path", "CHRF", {}, "a b", [None],
     ("ValueError", "not enough values to unpack (expected 3, got 0)")),
    ("chrf_best_reference", "CHRF", {"char_order": 2, "word_order": 0}, "ab", ["abc", "ab"],
     ("chrF2 = 100.00",)),
]


@pytest.mark.parametrize("cls_name,kwargs,hyp,refs,expected", [c[1:] for c in SENTENCE_CASES],
                         ids=[c[0] for c in SENTENCE_CASES])
def test_fallback_sentence_score_constants(cls_name, kwargs, hyp, refs, expected):
    cls = getattr(swadesh, cls_name)
    assert _text(lambda: cls(**kwargs).sentence_score(hyp, refs)) == expected


SCORE_EQUALITY = [
    ("bleu_has_empty", "BLEU", {}, H_EMPTY, R_EMPTY),
    ("bleu_has_empty_two_refs", "BLEU", {}, H_EMPTY, R_EMPTY_TWO),
    ("bleu_lowercase", "BLEU", {"lowercase": True}, H, R_TWO),
    ("bleu_zero_orders", "BLEU", {"max_ngram_order": 0}, H, [R_ONE]),
    ("bleu_floor", "BLEU", {"smooth_method": "floor", "smooth_value": 0.1}, H, [R_ONE]),
    ("bleu_add_k", "BLEU", {"smooth_method": "add-k", "smooth_value": 0.5}, H, [R_ONE]),
    ("bleu_effective", "BLEU", {"effective_order": True}, H, [R_ONE]),
    ("bleu_none_tail", "BLEU", {}, H, R_NONE_TAIL),
    ("chrf_has_empty", "CHRF", {}, H_EMPTY, R_EMPTY),
    ("chrf_fractional_beta", "CHRF", {"beta": 2.5}, H, R_TWO),
    ("chrf_fractional_beta_half", "CHRF", {"beta": 0.5}, H, [R_ONE]),
    ("chrf_zero_beta", "CHRF", {"beta": 0}, H, [R_ONE]),
    ("chrf_fractional_beta_whitespace", "CHRF", {"beta": 2.5, "whitespace": True}, H, [R_ONE]),
    ("chrf_fractional_beta_eps", "CHRF", {"beta": 2.5, "eps_smoothing": True}, H, [R_ONE]),
    ("chrf_fractional_beta_pp", "CHRF", {"beta": 2.5, "char_order": 6, "word_order": 2}, H, R_TWO),
    ("chrf_word_orders_only", "CHRF", {"char_order": 0, "word_order": 2}, H, [R_ONE]),
    ("chrf_lowercase", "CHRF", {"lowercase": True}, H, R_TWO),
]


@pytest.mark.parametrize("cls_name,kwargs,hyps,refs", [c[1:] for c in SCORE_EQUALITY],
                         ids=[c[0] for c in SCORE_EQUALITY])
def test_fallback_corpus_scores_match_live_sacrebleu(cls_name, kwargs, hyps, refs):
    ours = getattr(swadesh, cls_name)(**kwargs)
    live = getattr(sacrebleu, cls_name)(**kwargs)
    assert _outcome(lambda: ours.corpus_score(hyps, refs)) == _outcome(
        lambda: live.corpus_score(hyps, refs))


SENTENCE_EQUALITY = [
    ("bleu", "BLEU", {}), ("bleu_zero_orders", "BLEU", {"max_ngram_order": 0}),
    ("chrf_has_empty", "CHRF", {}),
    ("chrf_fractional_beta", "CHRF", {"beta": 2.5}),
    ("chrf_fractional_beta_pp", "CHRF", {"beta": 2.5, "char_order": 6, "word_order": 2}),
]


@pytest.mark.parametrize("cls_name,kwargs", [c[1:] for c in SENTENCE_EQUALITY],
                         ids=[c[0] for c in SENTENCE_EQUALITY])
@pytest.mark.parametrize("hyps,streams", [(H_EMPTY, R_EMPTY), (H, R_NONE_TAIL)],
                         ids=["empty_segment", "none_tail"])
def test_fallback_sentence_scores_match_live_sacrebleu(cls_name, kwargs, hyps, streams):
    ours = getattr(swadesh, cls_name)(**kwargs)
    live = getattr(sacrebleu, cls_name)(**kwargs)
    for i, hyp in enumerate(hyps):
        seg_refs = [stream[i] for stream in streams]
        assert _outcome(lambda: ours.sentence_score(hyp, seg_refs)) == _outcome(
            lambda: live.sentence_score(hyp, seg_refs)), (hyp, seg_refs)


# --------------------------------------------------------- properties of the port

def test_fallback_loop_does_not_execute_sacrebleu_bodies(monkeypatch):
    """Replace the four inherited bodies with raisers; the fallback still returns the pinned values."""
    def boom(*args, **kwargs):
        raise AssertionError("sacreBLEU's fallback method body ran")

    monkeypatch.setattr(sacre_bleu.BLEU, "_extract_reference_info", boom)
    monkeypatch.setattr(sacre_bleu.BLEU, "_compute_segment_statistics", boom)
    monkeypatch.setattr(sacre_chrf.CHRF, "_extract_reference_info", boom)
    monkeypatch.setattr(sacre_chrf.CHRF, "_compute_segment_statistics", boom)

    assert _rows_ours(BLEU(), H_EMPTY, R_EMPTY) == [
        [6, 6, 5, 3, 2, 1, 6, 5, 4, 3],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 1, 0, 0, 2, 1, 0, 0],
    ]
    assert _rows_ours(CHRF(char_order=2, word_order=0), ["ab"], [["ab"], ["abc"]]) == \
        [[2, 2, 2, 1, 1, 1]]
    assert _text(lambda: BLEU().corpus_score(H_EMPTY, R_EMPTY)) == (
        "BLEU = 55.84 87.5/66.7/50.0/33.3 (BP = 1.000 ratio = 1.143 hyp_len = 8 ref_len = 7)",)
    assert _text(lambda: CHRF(beta=2.5).corpus_score(H, [R_ONE])) == ("chrF2.5 = 85.92",)
    assert _text(lambda: CHRF(beta=2.5, char_order=6, word_order=2).corpus_score(H, R_TWO)) == (
        "chrF2.5++ = 100.00",)


def test_non_native_tokenizer_fallback_matches_live_sacrebleu():
    """A pre-tokenising, non-native signature takes the Python path; the fallback loop then runs."""
    def fake_tokenizer(sentence):
        return " <w> ".join(sentence.replace(".", " . ").split())

    hyps = ["AB .", "x y"]
    streams = [["AB .", "x y"], [None, "x y"]]
    ours = BLEU()
    ours.tokenizer = fake_tokenizer
    ours.tokenizer_signature = "fake-mecab"
    live = sacrebleu.BLEU()
    live.tokenizer = fake_tokenizer
    live.tokenizer_signature = "fake-mecab"
    assert ours._native_tokenizer is None
    our_rows = _rows_ours(ours, hyps, streams)
    assert our_rows == [[3, 3, 3, 2, 1, 0, 3, 2, 1, 0], [3, 3, 3, 2, 1, 0, 3, 2, 1, 0]]
    assert our_rows == _rows_live(live, hyps, streams)
    assert _outcome(lambda: ours.corpus_score(hyps, streams)) == _outcome(
        lambda: live.corpus_score(hyps, streams))


def test_cached_references_through_the_fallback_path():
    for cls_name in ("BLEU", "CHRF"):
        ours = getattr(swadesh, cls_name)(references=R_EMPTY)
        live = getattr(sacrebleu, cls_name)(references=R_EMPTY)
        assert _outcome(lambda: ours.corpus_score(H_EMPTY, None)) == _outcome(
            lambda: live.corpus_score(H_EMPTY, None))
        assert ours.num_refs == live.num_refs == -1


def test_bootstrap_over_fallback_rows_matches_live():
    """The fast path identifies the untouched scorer by identity; these rows must still resample."""
    ours = BLEU().corpus_score(H_EMPTY, R_EMPTY, n_bootstrap=20)
    live = sacrebleu.BLEU().corpus_score(H_EMPTY, R_EMPTY, n_bootstrap=20)
    assert (str(ours), repr(ours._mean), repr(ours._ci)) == \
        (str(live), repr(live._mean), repr(live._ci))


DIRECT_ERROR_CASES = [
    ("bleu_extract_int", "BLEU", lambda m: m._extract_reference_info([5])),
    ("bleu_compute_int", "BLEU",
     lambda m: m._compute_segment_statistics(5, {"ref_ngrams": None, "ref_lens": []})),
    ("bleu_extract_bytes", "BLEU", lambda m: m._extract_reference_info([b"ab"])),
    ("chrf_extract_int", "CHRF", lambda m: m._extract_reference_info([5])),
    ("chrf_compute_int", "CHRF", lambda m: m._compute_segment_statistics(5, {"ref_ngrams": []})),
    ("chrf_extract_bytes", "CHRF", lambda m: m._extract_reference_info([b"ab"])),
    ("chrf_compute_bytes", "CHRF",
     lambda m: m._compute_segment_statistics(b"ab", {"ref_ngrams": []})),
]


@pytest.mark.parametrize("cls_name,make", [c[1:] for c in DIRECT_ERROR_CASES],
                         ids=[c[0] for c in DIRECT_ERROR_CASES])
def test_method_errors_match_live_sacrebleu(cls_name, make):
    assert _error(lambda: make(getattr(swadesh, cls_name)())) == \
        _error(lambda: make(getattr(sacrebleu, cls_name)()))