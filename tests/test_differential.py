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


def _rand_corpus(seed, n, vocab=None):
    rnd = random.Random(seed)
    words = vocab or "the a cat dog sat on mat ran fast slow red blue , . big small house tree".split()
    mk = lambda: " ".join(rnd.choices(words, k=rnd.randint(0, 25)))
    return [mk() for _ in range(n)], [[mk() for _ in range(n)]]


@pytest.mark.parametrize("metric", ["bleu", "bleu-addk", "bleu-floor", "chrf", "chrf++"])
@pytest.mark.parametrize("n", [1, 2, 37, 1500])
def test_bootstrap_exact(metric, n):
    hyps, refs = _rand_corpus(n, n)
    mk = {"bleu": lambda M: M.BLEU(), "bleu-addk": lambda M: M.BLEU(smooth_method="add-k"),
          "bleu-floor": lambda M: M.BLEU(smooth_method="floor", effective_order=True),
          "chrf": lambda M: M.CHRF(), "chrf++": lambda M: M.CHRF(word_order=2)}[metric]
    a = mk(swadesh).corpus_score(hyps, refs, n_bootstrap=300)
    b = mk(sacrebleu).corpus_score(hyps, refs, n_bootstrap=300)
    assert str(a) == str(b)
    assert a._mean == b._mean and a._ci == b._ci and type(a._mean) is type(b._mean)


def test_bootstrap_float32_rounding_regime():
    # > 2**24 total characters: float32 sums are no longer exact, so the
    # summation order must match numpy's exactly.
    rnd = random.Random(7)
    words = ["supercalifragilistic" * 4, "antidisestablishment" * 4, "x"]
    hyps = [" ".join(rnd.choices(words, k=40)) for _ in range(6000)]
    refs = [[" ".join(rnd.choices(words, k=40)) for _ in range(6000)]]
    a = swadesh.CHRF().corpus_score(hyps, refs, n_bootstrap=20)
    b = sacrebleu.CHRF().corpus_score(hyps, refs, n_bootstrap=20)
    assert a._mean == b._mean and a._ci == b._ci


@pytest.mark.parametrize("test_type", ["bs", "ar"])
def test_paired_tests_exact(test_type):
    from sacrebleu.significance import PairedTest as SPT
    hyps, refs = _rand_corpus(11, 400)
    sys2, _ = _rand_corpus(12, 400)
    sys3, _ = _rand_corpus(13, 400)
    named = [("baseline", hyps), ("sys2", sys2), ("sys3", sys3)]
    kw = dict(test_type=test_type, n_samples=0 if test_type == "bs" else 500, n_ar_confidence=200)
    ours = swadesh.PairedTest(named, {"BLEU": swadesh.BLEU(references=refs), "chrF": swadesh.CHRF(references=refs)}, None, **kw)()
    theirs = SPT(named, {"BLEU": sacrebleu.BLEU(references=refs), "chrF": sacrebleu.CHRF(references=refs)}, None, **kw)()
    assert {k: str(v) for k, v in ours[0].items()} == {k: str(v) for k, v in theirs[0].items()}
    for metric in theirs[1]:
        for x, y in zip(ours[1][metric], theirs[1][metric]):
            assert repr(x) == repr(y), (metric, x, y)


@FAST
@given(data=st.data(), cands=corpus(1, 5), refs=corpus(1, 4))
def test_pairwise_matrices(data, cands, refs):
    lc = data.draw(st.booleans())
    tok = data.draw(st.sampled_from(TOKENIZERS))
    wo = data.draw(st.sampled_from([0, 2]))
    for ours, theirs in [
        (swadesh.CHRF(word_order=wo, lowercase=lc), sacrebleu.CHRF(word_order=wo, lowercase=lc)),
        (swadesh.BLEU(tokenize=tok, lowercase=lc, effective_order=True),
         sacrebleu.BLEU(tokenize=tok, lowercase=lc, effective_order=True)),
    ]:
        m = ours.pairwise(cands, refs)
        exp = [[theirs.sentence_score(c, [r]).score for r in refs] for c in cands]
        assert m.tolist() == exp
    sq = swadesh.CHRF().pairwise(cands)
    assert sq.shape == (len(cands), len(cands))


def test_mbr():
    cands = ["the cat sat on the mat", "a cat sat on the mat", "the cat sat on a mat", "dogs are great"]
    best, util = swadesh.mbr(cands)
    assert best in (0, 1, 2) and util.shape == (4,)


def test_rewards():
    from swadesh.rewards import bleu_reward, chrf_reward
    hyps = ["Das ist gut.", "Ein Hund bellt."]
    refs = ["Das ist sehr gut.", "Der Hund bellt laut."]
    r = chrf_reward()(completions=hyps, reference=refs)
    assert r == [sacrebleu.CHRF().sentence_score(h, [x]).score * 0.01 for h, x in zip(hyps, refs)]
    conv = [[{"role": "assistant", "content": h}] for h in hyps]
    assert chrf_reward()(completions=conv, reference=refs) == r
    b = bleu_reward()(completions=hyps, reference=[[x, x + " ja"] for x in refs])
    m = sacrebleu.BLEU(effective_order=True)
    assert b == [m.sentence_score(h, [x, x + " ja"]).score * 0.01 for h, x in zip(hyps, refs)]
def _sum_same(a, b):
    import math
    import struct
    if type(a) is not type(b):
        return False
    if isinstance(a, int):
        return a == b
    return struct.pack("<d", a) == struct.pack("<d", b) or (math.isnan(a) and math.isnan(b))


def test_py_sum_matches_builtin_sum():
    """The emulated builtin sum() must agree bit-for-bit on this interpreter.

    CPython changed float summation in 3.12 and again in 3.14 (compensated ints),
    so this test is what catches a wrong SumMode for a new interpreter.
    """
    import math
    rnd = random.Random(7)
    hard = [1e16, -1e16, 1.0, -1.0, 0.1, 0.2, 0.3, 3.0, 1e-300, 2.5, -7.25]
    for _ in range(20000):
        items = []
        for _ in range(rnd.randint(0, 8)):
            r = rnd.random()
            if r < 0.15:
                items.append(-9999999999)
            elif r < 0.25:
                items.append(rnd.randint(-5, 5))
            elif r < 0.55:
                items.append(rnd.choice(hard))
            else:
                items.append(math.log(rnd.uniform(1e-3, 100.0)))
        assert _sum_same(sum(items), swadesh._swadesh._py_sum(items)), items


PUNCT_HEAVY = st.lists(st.sampled_from(list(".,-0123456789 a.,-é中&;'\"()$\n\t ") + ["&amp;", "<skipped>", "-\n", "&quot;"]),
                       max_size=40).map("".join)


def _sacre_tok(name):
    from sacrebleu.metrics.bleu import _get_tokenizer
    return _get_tokenizer(name)()


@pytest.mark.parametrize("name", TOKENIZERS)
@settings(max_examples=3000, deadline=None, suppress_health_check=list(HealthCheck))
@given(line=st.one_of(PUNCT_HEAVY, segment))
def test_tokenizers_string_exact(name, line):
    ref = _sacre_tok(name)
    assert swadesh.tokenize(line, name) == ref(line)


def test_tokenize_batch_real_data():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "bench", "data", "wmt22.en-de.ref")
    if not os.path.exists(path):
        pytest.skip("bench data not downloaded")
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    for name in TOKENIZERS:
        ref = _sacre_tok(name)
        assert swadesh.tokenize(lines, name) == [ref(l) for l in lines]


@pytest.mark.parametrize("seed", ["12345", "0", "1", "987654321987654321", "none"])
def test_bootstrap_seeds(seed, monkeypatch):
    monkeypatch.setenv("SACREBLEU_SEED", seed)
    hyps, refs = _rand_corpus(3, 700)
    a = swadesh.BLEU().corpus_score(hyps, refs, n_bootstrap=200)
    b = sacrebleu.BLEU().corpus_score(hyps, refs, n_bootstrap=200)
    if seed == "none":  # OS entropy: only the format can be compared
        assert a.score == b.score
    else:
        assert str(a) == str(b) and a._mean == b._mean and a._ci == b._ci


@pytest.mark.skipif(int(__import__("numpy").__version__.split(".")[0]) < 2, reason="NumPy >= 2 path")
@settings(max_examples=2000, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_native_bootstrap_bleu_scoring_rowwise(data):
    import numpy as np
    from swadesh.significance import _native_bleu_scores
    order = data.draw(st.integers(1, 6))
    smooth = data.draw(st.sampled_from(["exp", "floor", "add-k", "none"]))
    sv = data.draw(st.sampled_from([None, 0.1, 0.5, 1, 2, 0.01]))
    eff = data.draw(st.booleans())
    big = data.draw(st.sampled_from([10, 1000, 2**24 + 7, 2**30]))
    rows = []
    for _ in range(data.draw(st.integers(1, 8))):
        total = [data.draw(st.integers(0, big)) for _ in range(order)]
        correct = [data.draw(st.integers(0, t)) for t in total]
        rows.append([data.draw(st.integers(0, big)), data.draw(st.integers(0, big))] + correct + total)
    sums = np.array(rows, dtype=np.float32)
    m = sacrebleu.BLEU(smooth_method=smooth, smooth_value=sv, max_ngram_order=order, effective_order=eff)
    ours = [s.score for s in _native_bleu_scores(m, sums)]
    theirs = [m._compute_score_from_stats(r.copy()).score for r in sums]
    assert ours == theirs
