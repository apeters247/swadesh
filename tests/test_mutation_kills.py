"""Python-level differential tests written for the mutation-survivor analysis.

Each test compares swadesh's public API with sacreBLEU 2.6.0 on inputs that reach a branch a
surviving mutant had changed: sentence-level TER with shifts and long/short length ratios,
BLEU with orders above 4 (the wide n-gram key path, which merges multiple references),
chrF with char_order 7 and word_order 5 (wide keys), and BLEU/chrF smoothing edge cases.
Every comparison is exact (`==` on floats and on the score objects' attributes).
"""

import random

import pytest
import sacrebleu
from sacrebleu.metrics import BLEU as SBLEU
from sacrebleu.metrics import CHRF as SCHRF
from sacrebleu.metrics import TER as STER

import swadesh

VOCAB = ["a", "b", "c", "d", "the", "cat", "dog", "x"]


def _words(r, n, vocab=VOCAB):
    return " ".join(r.choice(vocab) for _ in range(n))


def _ter_pairs(seed, count):
    r = random.Random(seed)
    for _ in range(count):
        kind = r.random()
        n = r.randint(1, 30) if kind < 0.6 else r.randint(40, 70)
        ref = _words(r, n)
        words = ref.split()
        if r.random() < 0.5 and len(words) > 3:
            # Move a block far away: exercises shift distance, beam and candidate limits.
            s = r.randrange(len(words) - 1)
            ln = r.randint(1, min(6, len(words) - s))
            block = words[s:s + ln]
            rest = words[:s] + words[s + ln:]
            t = r.randint(0, len(rest))
            words = rest[:t] + block + rest[t:]
        hyp = " ".join(words)
        if kind > 0.9:
            hyp = _words(r, r.randint(0, 3))  # very short hypothesis vs long reference
        refs = [ref] if r.random() < 0.7 else [ref, _words(r, r.randint(1, 40))]
        yield hyp, refs


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ter_sentence_scores_match_sacrebleu(seed):
    ours = swadesh.TER()
    theirs = STER()
    for hyp, refs in _ter_pairs(seed, 40):
        a = ours.sentence_score(hyp, refs)
        b = theirs.sentence_score(hyp, refs)
        assert a.score == b.score, (hyp, refs)
        assert str(a) == str(b), (hyp, refs)


def test_ter_empty_and_identical_inputs():
    ours, theirs = swadesh.TER(), STER()
    for hyp, refs in [("", ["a b c"]), ("a b", [""]), ("", [""]), ("a a a", ["a a a"]),
                      ("c a b", ["a b c"]), ("b c a", ["a b c"])]:
        assert ours.sentence_score(hyp, refs).score == theirs.sentence_score(hyp, refs).score


@pytest.mark.parametrize("order", [5, 6])
def test_bleu_orders_above_four_match_sacrebleu(order):
    """max_ngram_order > 4 leaves the packed u128 path: multi-reference clipping takes the
    wide-key route, which must clip exactly as sacreBLEU does."""
    r = random.Random(40 + order)
    hyps, refs = [], [[], []]
    for _ in range(60):
        base = [r.choice("ab") for _ in range(r.randint(5, 14))]
        hyps.append(" ".join(base))
        refs[0].append(" ".join(r.choice("ab") for _ in range(r.randint(5, 14))))
        refs[1].append(" ".join(base[:r.randint(0, len(base))] + ["a", "b"]))
    ours = swadesh.BLEU(max_ngram_order=order)
    theirs = SBLEU(max_ngram_order=order)
    assert ours.corpus_score(hyps, refs).score == theirs.corpus_score(hyps, refs).score
    for h, a, b in zip(hyps, refs[0], refs[1]):
        assert ours.sentence_score(h, [a, b]).score == theirs.sentence_score(h, [a, b]).score


def test_chrf_wide_char_and_word_orders_match_sacrebleu():
    r = random.Random(77)
    hyps, refs = [], [[], []]
    for _ in range(60):
        hyps.append(" ".join("".join(r.choice("abé.,x") for _ in range(r.randint(1, 5)))
                             for _ in range(r.randint(0, 9))))
        refs[0].append(" ".join("".join(r.choice("abé.,x") for _ in range(r.randint(1, 5)))
                                for _ in range(r.randint(0, 9))))
        refs[1].append(hyps[-1][: r.randint(0, len(hyps[-1]))] + " ab.")
    for kwargs in [dict(char_order=7, word_order=0), dict(char_order=6, word_order=5),
                   dict(char_order=7, word_order=5, beta=3)]:
        ours = swadesh.CHRF(**kwargs)
        theirs = SCHRF(**kwargs)
        assert ours.corpus_score(hyps, refs).score == theirs.corpus_score(hyps, refs).score, kwargs
        for h, a, b in zip(hyps, refs[0], refs[1]):
            assert ours.sentence_score(h, [a, b]).score == theirs.sentence_score(h, [a, b]).score


@pytest.mark.parametrize("smooth,value", [("floor", 0), ("floor", 0.1), ("add-k", 1), ("exp", None),
                                          ("none", None)])
def test_bleu_smoothing_edges_match_sacrebleu(smooth, value):
    kwargs = dict(smooth_method=smooth, effective_order=True)
    if value is not None:
        kwargs["smooth_value"] = value
    ours, theirs = swadesh.BLEU(**kwargs), SBLEU(**kwargs)
    cases = [("a b c d", ["x y z w"]), ("a", ["a b c d e"]), ("a b c d e f", ["a b"]),
             ("", ["a b"]), ("a b", [""]), ("a a a a", ["a a a a"]), ("a b c", ["c b a"])]
    for h, refs in cases:
        a, b = ours.sentence_score(h, refs), theirs.sentence_score(h, refs)
        assert a.score == b.score, (smooth, value, h, refs)
        assert a.bp == b.bp, (smooth, value, h, refs)


def test_bleu_bp_when_both_lengths_are_zero():
    # sacreBLEU: `if sys_len < ref_len` is false when both are 0, so BP is 1.0.
    assert SBLEU().sentence_score("", [""]).bp == swadesh.BLEU().sentence_score("", [""]).bp == 1.0


def test_chrf_first_best_reference_wins_on_ties():
    """Two references with the same F-score but different statistics: the first one counts."""
    ours, theirs = swadesh.CHRF(), SCHRF()
    hyp = "ab"
    refs = ["ab ab", "ab"]
    a = ours.sentence_score(hyp, refs)
    b = theirs.sentence_score(hyp, refs)
    assert a.score == b.score
    assert str(a) == str(b)
