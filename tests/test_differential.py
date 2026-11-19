"""Differential tests: swadesh must be bit-identical to sacreBLEU.

Every test compares *exact* values (statistics as ints, scores via ==, and the
formatted strings / signatures), never approximations.
"""
import random

import pytest
import sacrebleu
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import swadesh

# An adversarial alphabet: everything the tokenizers special-case.
ALPHABET = (
    list("abcXYZ019 .,-'\"!?;:()[]{}<>/\\@#$%^&*_+=|~`")
    + ["&amp;", "&quot;", "&lt;", "&gt;", "<skipped>", "-\n", "\n", "\t", "  "]
    + list("    　  \x1c\x85​")  # unicode spaces (+ZWSP which is not)
    + list("中文字符測試日本語한국어ｆｕｌｌ，。⺀㇀⾁⾡⩭⩮…€→")
    + list("ΣσςİIıßẞǅﬁ́é٣²Ⅷ½¿¡«»„“”‘’–—")
    + ["\U0001F600", "\U00020001", "\U0002A6D6", "\U0001D7CE"]
)

segment = st.lists(st.sampled_from(ALPHABET), min_size=0, max_size=25).map("".join)


def corpus(min_size=1, max_size=6):
    return st.lists(segment, min_size=min_size, max_size=max_size)


TOKENIZERS = ["13a", "intl", "zh", "char", "none"]

FAST = settings(max_examples=300, deadline=None, suppress_health_check=list(HealthCheck))


def _refs_like(hyps, data, n_refs, allow_none):
    streams = []
    for _ in range(n_refs):
        s = data.draw(st.lists(segment, min_size=len(hyps), max_size=len(hyps)))
        if allow_none:
            s = [None if (x == "" and random.random() < 0.5) else x for x in s]
        streams.append(s)
    # sacreBLEU crashes if every reference of a segment is None (invalid input):
    # keep stream 0 always present.
    streams[0] = ["" if x is None else x for x in streams[0]]
    return streams


def assert_same_score(a, b):
    assert type(a).__name__ == type(b).__name__
    assert a.score == b.score, (a.score, b.score)
    assert str(a) == str(b)
    for attr in ("counts", "totals", "precisions", "bp", "sys_len", "ref_len"):
        if hasattr(b, attr):
            assert getattr(a, attr) == getattr(b, attr), attr


@pytest.mark.parametrize("tok", TOKENIZERS)
@pytest.mark.parametrize("lowercase", [False, True])
@FAST
@given(data=st.data(), hyps=corpus())
def test_bleu_corpus(tok, lowercase, data, hyps):
    n_refs = data.draw(st.integers(1, 3))
    refs = _refs_like(hyps, data, n_refs, allow_none=n_refs > 1)
    smooth = data.draw(st.sampled_from(["exp", "floor", "add-k", "none"]))
    order = data.draw(st.integers(1, 6))
    eff = data.draw(st.booleans())
    sv = data.draw(st.sampled_from([None, 0.1, 0.5, 1, 2]))
    kw = dict(tokenize=tok, lowercase=lowercase, smooth_method=smooth,
              smooth_value=sv, max_ngram_order=order, effective_order=eff)
    ours, theirs = swadesh.BLEU(**kw), sacrebleu.BLEU(**kw)
    s_ours = ours._extract_corpus_statistics(hyps, refs).tolist()
    s_theirs = theirs._extract_corpus_statistics(hyps, refs)
    assert s_ours == s_theirs
    assert_same_score(ours.corpus_score(hyps, refs), theirs.corpus_score(hyps, refs))
    assert str(ours.get_signature()) == str(theirs.get_signature())


@pytest.mark.parametrize("tok", TOKENIZERS)
@FAST
@given(data=st.data(), hyps=corpus())
def test_bleu_sentence_scores(tok, data, hyps):
    refs = _refs_like(hyps, data, data.draw(st.integers(1, 2)), allow_none=False)
    smooth = data.draw(st.sampled_from(["exp", "floor", "add-k", "none"]))
    eff = data.draw(st.booleans())
    lc = data.draw(st.booleans())
    kw = dict(tokenize=tok, smooth_method=smooth, effective_order=eff, lowercase=lc)
    ours = swadesh.BLEU(**kw).sentence_scores(hyps, refs)
    m = sacrebleu.BLEU(**kw)
    theirs = [m.sentence_score(h, [r[i] for r in refs]).score for i, h in enumerate(hyps)]
    assert ours == theirs


@FAST
@given(data=st.data(), hyps=corpus())
def test_chrf_corpus(data, hyps):
    n_refs = data.draw(st.integers(1, 3))
    refs = _refs_like(hyps, data, n_refs, allow_none=n_refs > 1)
    kw = dict(
        char_order=data.draw(st.sampled_from([1, 2, 4, 6, 7, 9])),
        word_order=data.draw(st.sampled_from([0, 1, 2, 3, 5])),
        beta=data.draw(st.sampled_from([0, 1, 2, 3])),
        lowercase=data.draw(st.booleans()),
        whitespace=data.draw(st.booleans()),
        eps_smoothing=data.draw(st.booleans()),
    )
    ours, theirs = swadesh.CHRF(**kw), sacrebleu.CHRF(**kw)
    assert ours._extract_corpus_statistics(hyps, refs).tolist() == theirs._extract_corpus_statistics(hyps, refs)
    assert_same_score(ours.corpus_score(hyps, refs), theirs.corpus_score(hyps, refs))
    assert str(ours.get_signature()) == str(theirs.get_signature())


@FAST
@given(data=st.data(), hyps=corpus())
def test_chrf_sentence_scores(data, hyps):
    refs = _refs_like(hyps, data, data.draw(st.integers(1, 2)), allow_none=False)
    kw = dict(word_order=data.draw(st.sampled_from([0, 2])), lowercase=data.draw(st.booleans()),
              eps_smoothing=data.draw(st.booleans()), whitespace=data.draw(st.booleans()))
    ours = swadesh.CHRF(**kw).sentence_scores(hyps, refs)
    m = sacrebleu.CHRF(**kw)
    theirs = [m.sentence_score(h, [r[i] for r in refs]).score for i, h in enumerate(hyps)]
    assert ours == theirs


def test_ref_cache_and_compat():
    hyps = ["The cat sat on the mat.", "A dog barked."]
    refs = [["The cat is on the mat.", "The dog barked."], ["A cat sat on a mat.", "Dogs bark."]]
    for name in ["corpus_bleu", "corpus_chrf"]:
        assert str(getattr(swadesh, name)(hyps, refs)) == str(getattr(sacrebleu, name)(hyps, refs))
    for name in ["sentence_bleu", "sentence_chrf"]:
        assert str(getattr(swadesh, name)(hyps[0], [r[0] for r in refs])) == \
            str(getattr(sacrebleu, name)(hyps[0], [r[0] for r in refs]))
    a = swadesh.BLEU(references=refs).corpus_score(hyps, None)
    b = sacrebleu.BLEU(references=refs).corpus_score(hyps, None)
    assert str(a) == str(b)


def test_bootstrap_identical():
    rnd = random.Random(0)
    words = "the a cat dog sat on mat ran fast slow red blue".split()
    hyps = [" ".join(rnd.choices(words, k=rnd.randint(1, 12))) for _ in range(200)]
    refs = [[" ".join(rnd.choices(words, k=rnd.randint(1, 12))) for _ in range(200)]]
    for M, S in [(swadesh.BLEU, sacrebleu.BLEU), (swadesh.CHRF, sacrebleu.CHRF)]:
        a = M().corpus_score(hyps, refs, n_bootstrap=100)
        b = S().corpus_score(hyps, refs, n_bootstrap=100)
        assert str(a) == str(b) and a._mean == b._mean and a._ci == b._ci
