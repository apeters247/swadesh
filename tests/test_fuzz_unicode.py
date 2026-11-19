"""Differential fuzzing over the full Unicode range (surrogates excluded).

The character distribution is weighted toward punctuation (P), symbols (S),
numbers (N), separators (Z), control characters (Cc) and the Han, Hangul and
Kana blocks, so that the CJK tokenizer ranges and whitespace handling are hit
often. Segments are up to 300 characters.

Budget: UNI_EXAMPLES env var (default 1500 per test).
"""
import os
import unicodedata

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import sacrebleu
import swadesh
from test_differential import TOKENIZERS, _refs_like, assert_same_score

EXAMPLES = int(os.environ.get("UNI_EXAMPLES", "1500"))
SURR_LO, SURR_HI = 0xD800, 0xDFFF


_CAT = {c: [] for c in ("P", "S", "N", "Z", "Cc")}
for _cp in range(0x110000):
    if SURR_LO <= _cp <= SURR_HI:
        continue
    _cat = unicodedata.category(chr(_cp))
    if _cat == "Cc":
        _CAT["Cc"].append(_cp)
    elif _cat[0] in "PSNZ":
        _CAT[_cat[0]].append(_cp)

_P = st.sampled_from(_CAT["P"]).map(chr)
_S = st.sampled_from(_CAT["S"]).map(chr)
_N = st.sampled_from(_CAT["N"]).map(chr)
_Z = st.sampled_from(_CAT["Z"]).map(chr)
_CC = st.sampled_from(_CAT["Cc"]).map(chr)
_HAN = st.one_of(st.integers(0x4E00, 0x9FFF), st.integers(0x3400, 0x4DBF), st.integers(0x20000, 0x2A6DF),
                 st.integers(0xF900, 0xFAFF), st.integers(0x2E80, 0x2FDF)).map(chr)
_HANGUL = st.one_of(st.integers(0xAC00, 0xD7A3), st.integers(0x1100, 0x11FF), st.integers(0x3130, 0x318F)).map(chr)
_KANA = st.one_of(st.integers(0x3040, 0x309F), st.integers(0x30A0, 0x30FF), st.integers(0x31F0, 0x31FF),
                  st.integers(0xFF66, 0xFF9D)).map(chr)
_ANY = st.integers(0, 0x10FFFF - 0x800).map(lambda x: x if x < SURR_LO else x + 0x800).map(chr)

# Weighted by repetition in one_of.
CHAR = st.one_of(*([_P] * 15 + [_S] * 10 + [_N] * 10 + [_Z] * 10 + [_CC] * 10
                   + [_HAN] * 15 + [_HANGUL] * 8 + [_KANA] * 8 + [_ANY] * 8))
segment = st.lists(CHAR, min_size=0, max_size=300).map("".join)
FUZZ = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=list(HealthCheck))


def corpus(min_size=1, max_size=5):
    return st.lists(segment, min_size=min_size, max_size=max_size)


@FUZZ
@given(data=st.data(), hyps=corpus())
def test_uni_bleu_corpus(data, hyps):
    tok = data.draw(st.sampled_from(TOKENIZERS))
    lc = data.draw(st.booleans())
    refs = _refs_like(hyps, data, data.draw(st.integers(1, 2)), allow_none=False)
    kw = dict(tokenize=tok, lowercase=lc, effective_order=data.draw(st.booleans()))
    ours, theirs = swadesh.BLEU(**kw), sacrebleu.BLEU(**kw)
    assert ours._extract_corpus_statistics(hyps, refs).tolist() == theirs._extract_corpus_statistics(hyps, refs)
    assert_same_score(ours.corpus_score(hyps, refs), theirs.corpus_score(hyps, refs))


@FUZZ
@given(data=st.data(), hyps=corpus())
def test_uni_bleu_sentence(data, hyps):
    tok = data.draw(st.sampled_from(TOKENIZERS))
    refs = _refs_like(hyps, data, data.draw(st.integers(1, 2)), allow_none=False)
    kw = dict(tokenize=tok, lowercase=data.draw(st.booleans()), effective_order=data.draw(st.booleans()),
              smooth_method=data.draw(st.sampled_from(["exp", "floor", "add-k", "none"])))
    ours = swadesh.BLEU(**kw).sentence_scores(hyps, refs)
    m = sacrebleu.BLEU(**kw)
    assert ours == [m.sentence_score(h, [r[i] for r in refs]).score for i, h in enumerate(hyps)]


@FUZZ
@given(data=st.data(), hyps=corpus())
def test_uni_chrf_corpus(data, hyps):
    refs = _refs_like(hyps, data, data.draw(st.integers(1, 2)), allow_none=False)
    kw = dict(char_order=data.draw(st.sampled_from([1, 3, 6, 7])), word_order=data.draw(st.sampled_from([0, 2])),
              lowercase=data.draw(st.booleans()), whitespace=data.draw(st.booleans()),
              eps_smoothing=data.draw(st.booleans()))
    ours, theirs = swadesh.CHRF(**kw), sacrebleu.CHRF(**kw)
    assert ours._extract_corpus_statistics(hyps, refs).tolist() == theirs._extract_corpus_statistics(hyps, refs)
    assert_same_score(ours.corpus_score(hyps, refs), theirs.corpus_score(hyps, refs))


@FUZZ
@given(data=st.data(), hyps=corpus())
def test_uni_chrf_sentence(data, hyps):
    refs = _refs_like(hyps, data, data.draw(st.integers(1, 2)), allow_none=False)
    kw = dict(word_order=data.draw(st.sampled_from([0, 2])), lowercase=data.draw(st.booleans()),
              whitespace=data.draw(st.booleans()), eps_smoothing=data.draw(st.booleans()))
    ours = swadesh.CHRF(**kw).sentence_scores(hyps, refs)
    m = sacrebleu.CHRF(**kw)
    assert ours == [m.sentence_score(h, [r[i] for r in refs]).score for i, h in enumerate(hyps)]
