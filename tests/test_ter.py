"""Differential tests: swadesh TER must be bit-identical to sacrebleu.TER.

Segment statistics are compared as exact values (lists of floats via ``==``),
and scores through ``.score``, ``num_edits``, ``ref_length``, ``str()`` and the
signature. Nothing is compared approximately.
"""
import itertools
import os
import random

import pytest
import sacrebleu
from hypothesis import given
from hypothesis import strategies as st

from fuzzsupport import budget

import swadesh

# Adversarial alphabet for the TER tokenizer. Same idea as test_differential.py,
# plus the TER-specific tokens (possessives, newline-hyphen, XML entities, CJK
# punctuation and full-width forms).
ALPHABET = (
    list("abcXYZ019 .,-'\"!?;:()[]{}<>/\\@#$%^&*_+=|~`")
    + ["&amp;", "&quot;", "&lt;", "&gt;", "<skipped>", "-\n", "\n", "\n-", "\t", "  "]
    + ["'s", "'s ", " 's", "s'", "'S", "'s\n"]
    + list("    　  \x1c\x85​")
    + list("中文字符測試日本語한국어ｆｕｌｌ，。、？！（）⺀㇀⾁⾡⩭⩮…€→")
    + list("ΣσςİIıßẞǅﬁ́é٣²Ⅷ½¿¡«»„“”‘’–—")
    + list("ぁあいアイㇰ㌀㈱㊤㋿ﾞ．：；＂")
    + ["\U0001F600", "\U00020001", "\U0002A6D6", "\U0001D7CE"]
)

segment = st.lists(st.sampled_from(ALPHABET), min_size=0, max_size=25).map("".join)

# Small vocabulary so that the same words recur and shifts actually trigger.
VOCAB = ["the", "cat", "sat", "on", "mat", "a", "dog", "is", "of", "it", "and", "to", "x,", "y.", "'s", "3-4"]

FAST = budget(150)
SLOW = budget(40)

OPTION_COMBOS = list(itertools.product([False, True], repeat=4))


def _refs_like(hyps, data, n_refs):
    """Reference streams. Stream 0 is always defined; later streams may have None."""
    streams = []
    for k in range(n_refs):
        s = data.draw(st.lists(segment, min_size=len(hyps), max_size=len(hyps)))
        if k > 0:
            s = [None if (x == "" and data.draw(st.booleans())) else x for x in s]
        streams.append(s)
    return streams


def assert_same(ours, theirs):
    # The score objects are swadesh's own now, so the types differ by design; compare the
    # observable fields the module docstring promises instead.
    assert ours.score == theirs.score, (ours.score, theirs.score)
    assert ours.num_edits == theirs.num_edits
    assert ours.ref_length == theirs.ref_length
    assert str(ours) == str(theirs)


def assert_same_stats(ours_metric, theirs_metric, hyps, refs):
    ours_stats = ours_metric._extract_corpus_statistics(hyps, refs).tolist()
    theirs_stats = theirs_metric._extract_corpus_statistics(hyps, refs)
    assert ours_stats == theirs_stats


def _make(options):
    normalized, no_punct, asian_support, case_sensitive = options
    kw = dict(normalized=normalized, no_punct=no_punct, asian_support=asian_support, case_sensitive=case_sensitive)
    return swadesh.TER(**kw), sacrebleu.TER(**kw)


@pytest.mark.parametrize("options", OPTION_COMBOS, ids=lambda o: "".join("1" if x else "0" for x in o))
@FAST
@given(data=st.data(), hyps=st.lists(segment, min_size=1, max_size=6))
def test_ter_adversarial(options, data, hyps):
    n_refs = data.draw(st.integers(1, 3))
    refs = _refs_like(hyps, data, n_refs)
    ours, theirs = _make(options)
    assert_same_stats(ours, theirs, hyps, refs)
    assert_same(ours.corpus_score(hyps, refs), theirs.corpus_score(hyps, refs))
    assert str(ours.get_signature()) == str(theirs.get_signature())


def test_ter_possessive_and_reference_double_pass():
    """Spec ter.md step 8 and section 5: the possessive rule, and the reference preprocessed
    twice (once in Python, once by the kernel). Pinned on the measured cases."""
    cases = [
        ("a's.", "a's."),
        ("a's,", "a's,"),
        ("a's", "a's"),
        ("", "'s'"),
        ("dont''s", "dont''s"),
        ("a's?", "a's?"),
    ]
    for options in [(True, False, False, False), (False, False, False, False)]:
        ours, theirs = _make(options)
        for hyp, ref in cases:
            assert_same_stats(ours, theirs, [hyp], [[ref]])
            assert_same(ours.corpus_score([hyp], [[ref]]), theirs.corpus_score([hyp], [[ref]]))


@pytest.mark.parametrize("options", OPTION_COMBOS, ids=lambda o: "".join("1" if x else "0" for x in o))
@FAST
@given(data=st.data(), hyp=segment, n_refs=st.integers(1, 3))
def test_ter_sentence_score(options, data, hyp, n_refs):
    refs = [data.draw(segment) for _ in range(n_refs)]
    ours, theirs = _make(options)
    assert_same(ours.sentence_score(hyp, refs), theirs.sentence_score(hyp, refs))


@pytest.mark.parametrize("options", OPTION_COMBOS, ids=lambda o: "".join("1" if x else "0" for x in o))
@SLOW
@given(data=st.data(), hyps=st.lists(st.lists(st.sampled_from(VOCAB), min_size=20, max_size=60).map(" ".join), min_size=1, max_size=3), mode=st.sampled_from(["independent", "perturbed"]))
def test_ter_long_sentences_shifts(options, data, hyps, mode):
    """20-60 word sentences over a small vocabulary, so that shifts trigger."""
    n_refs = data.draw(st.integers(1, 3))
    refs = []
    for _ in range(n_refs):
        stream = []
        for h in hyps:
            if mode == "independent":
                words = data.draw(st.lists(st.sampled_from(VOCAB), min_size=20, max_size=60))
            else:
                # Block moves and drops applied to the hypothesis words.
                words = h.split()
                for _ in range(data.draw(st.integers(1, 4))):
                    if len(words) < 4:
                        break
                    a = data.draw(st.integers(0, len(words) - 1))
                    ln = data.draw(st.integers(1, min(10, len(words) - a)))
                    block = words[a:a + ln]
                    rest = words[:a] + words[a + ln:]
                    at = data.draw(st.integers(0, len(rest)))
                    words = rest[:at] + block + rest[at:]
                if data.draw(st.booleans()) and len(words) > 3:
                    del words[data.draw(st.integers(0, len(words) - 1))]
            stream.append(" ".join(words))
        refs.append(stream)
    ours, theirs = _make(options)
    assert_same_stats(ours, theirs, hyps, refs)
    assert_same(ours.corpus_score(hyps, refs), theirs.corpus_score(hyps, refs))


def _perturb(line, rnd):
    """Word swaps and drops, deterministic for a given Random."""
    words = line.split()
    for i in range(len(words) - 1):
        if rnd.random() < 0.08:
            words[i], words[i + 1] = words[i + 1], words[i]
    words = [w for w in words if rnd.random() >= 0.05]
    return " ".join(words)


REAL_DATA = "/var/www/llmeval/bench/data/wmt22.en-de.ref"


@pytest.mark.skipif(not os.path.exists(REAL_DATA), reason="WMT22 data not present (bench/get_data.sh)")
@pytest.mark.parametrize("normalized", [False, True])
def test_ter_real_wmt22_perturbed(normalized):
    with open(REAL_DATA, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f][:500]
    rnd = random.Random(1234)
    hyps = [_perturb(ln, rnd) for ln in lines]
    refs = [lines]
    ours, theirs = _make((normalized, False, False, False))
    assert_same_stats(ours, theirs, hyps, refs)
    assert_same(ours.corpus_score(hyps, refs), theirs.corpus_score(hyps, refs))


def test_ter_no_references_raises_like_sacrebleu():
    for metric in _make((False, False, False, False)):
        with pytest.raises(ZeroDivisionError):
            metric._extract_corpus_statistics(["a b"], [[None]])


def test_ter_empty_strings_and_cached_refs():
    hyps = ["", "a b", "The cat sat on the mat."]
    refs = [["", "a b c", "The cat is on the mat."], ["x", "a", "The dog sat."]]
    for options in OPTION_COMBOS:
        ours, theirs = _make(options)
        assert_same(ours.corpus_score(hyps, refs), theirs.corpus_score(hyps, refs))
    cached = swadesh.TER(normalized=True, references=refs)
    ref = sacrebleu.TER(normalized=True, references=refs)
    assert_same(cached.corpus_score(hyps, None), ref.corpus_score(hyps, None))


def test_ter_module_functions_and_bootstrap():
    hyps = ["the cat sat on the mat .", "b c a", "dogs bark loudly"]
    refs = [["the cat is on the mat .", "a b c", "dogs barked"]]
    for name in ["corpus_ter", "sentence_ter"]:
        ours = getattr(swadesh, name)
        theirs = getattr(sacrebleu, name)
        if name == "corpus_ter":
            assert str(ours(hyps, refs, normalized=True)) == str(theirs(hyps, refs, normalized=True))
        else:
            assert str(ours(hyps[0], [r[0] for r in refs], asian_support=True)) == \
                str(theirs(hyps[0], [r[0] for r in refs], asian_support=True))
    a = swadesh.TER().corpus_score(hyps, refs, n_bootstrap=50)
    b = sacrebleu.TER().corpus_score(hyps, refs, n_bootstrap=50)
    assert str(a) == str(b) and a._mean == b._mean and a._ci == b._ci
