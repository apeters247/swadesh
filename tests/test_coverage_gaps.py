"""Differential tests for code paths the coverage run found untested.

Every value is compared with sacreBLEU 2.6.0. The reference classes are bound at
import time, before any test calls ``swadesh.patch()``, so they stay original.
"""
import asyncio
import io
import json
import logging
import os
import contextlib
import runpy
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest
import sacrebleu
import sacrebleu.sacrebleu as sb_cli
from sacrebleu.metrics import BLEU as REF_BLEU
from sacrebleu.metrics import CHRF as REF_CHRF
from sacrebleu.metrics import TER as REF_TER
from sacrebleu.significance import PairedTest as REF_PairedTest

import swadesh
import swadesh.cli
import swadesh.rewards as sw_rewards
import swadesh.verify as sw_verify

H = ["The cat sat on the mat.", "A dog barked loudly.", "Prices rose by 3.2% in October.", ""]
R = [["The cat is on the mat.", "The dog barked.", "Prices increased 3.2% in October.", "x"],
     ["A cat sat on a mat.", "The dogs barked loudly.", "Prices rose 3.2 percent in October.", ""]]
REFS_BY_CLASS = {swadesh.BLEU: REF_BLEU, swadesh.CHRF: REF_CHRF, swadesh.TER: REF_TER}


# --------------------------------------------------------------------------- patch / unpatch


@contextlib.contextmanager
def _sacrebleu_stdout():
    """sacreBLEU's CLI does ``sys.stdout = open(sys.stdout.fileno(), ...)``: a second file
    object that owns fd 1 and closes it when garbage-collected, which breaks pytest's
    fd capture for every later test. Hand it a duplicate fd instead, and close that
    ourselves after flushing (the duplicate writes to the same captured file)."""
    import os

    saved = sys.stdout
    saved.flush()
    fd = os.dup(saved.fileno())
    ours = open(fd, "w", encoding="utf-8", closefd=False)
    sys.stdout = ours
    try:
        yield
    finally:
        cur = sys.stdout
        cur.flush()
        sys.stdout = saved
        if cur is not ours:
            cur.close()  # sacreBLEU's re-opened file owns the duplicate fd
        else:
            ours.close()
            os.close(fd)

def test_patch_swaps_classes_and_unpatch_restores():
    import sacrebleu.compat as sb_compat
    import sacrebleu.metrics as m
    import sacrebleu.metrics.bleu as mb
    try:
        swadesh.patch()
        assert sacrebleu.BLEU is swadesh.BLEU and sacrebleu.CHRF is swadesh.CHRF and sacrebleu.TER is swadesh.TER
        assert m.METRICS["BLEU"] is swadesh.BLEU and m.METRICS["CHRF"] is swadesh.CHRF and m.METRICS["TER"] is swadesh.TER
        assert mb.BLEU is swadesh.BLEU and sb_compat.BLEU is swadesh.BLEU
        swadesh.patch()  # second call is a no-op
        assert sacrebleu.BLEU is swadesh.BLEU
        # Patched sacreBLEU entry points give sacreBLEU's numbers.
        assert str(sacrebleu.corpus_bleu(H, R)) == str(REF_BLEU().corpus_score(H, R))
        assert str(sacrebleu.corpus_chrf(H, R)) == str(REF_CHRF().corpus_score(H, R))
        assert str(sacrebleu.corpus_ter(H, R)) == str(REF_TER().corpus_score(H, R))
    finally:
        swadesh.unpatch()
    assert sacrebleu.BLEU is REF_BLEU and m.METRICS["BLEU"] is REF_BLEU and sb_compat.BLEU is REF_BLEU
    swadesh.unpatch()  # second call is a no-op


def test_paired_test_while_patched_matches_reference():
    """Under patch() the significance hooks are already installed (the `_installed` early return)."""
    named = [("baseline", H), ("sys2", [h.upper() for h in H])]
    kw = dict(test_type="bs", n_samples=200)
    theirs = REF_PairedTest(named, {"BLEU": REF_BLEU(references=R)}, None, **kw)()
    try:
        swadesh.patch()
        ours = swadesh.PairedTest(named, {"BLEU": swadesh.BLEU(references=R)}, None, **kw)()
    finally:
        swadesh.unpatch()
    assert {k: str(v) for k, v in ours[0].items()} == {k: str(v) for k, v in theirs[0].items()}
    for metric in theirs[1]:
        for x, y in zip(ours[1][metric], theirs[1][metric]):
            assert repr(x) == repr(y), (metric, x, y)


# --------------------------------------------------------------------------- significance

@pytest.mark.parametrize("test_type", ["ar", "bs"])
def test_paired_tests_for_float_statistics_ter(test_type):
    """TER statistics are float64, so the integer AR fast path does not apply (fallback branches)."""
    named = [("baseline", H), ("sys2", [h + " extra" for h in H])]
    kw = dict(test_type=test_type, n_samples=300, n_ar_confidence=100)
    ours = swadesh.PairedTest(named, {"TER": swadesh.TER(references=R)}, None, **kw)()
    theirs = REF_PairedTest(named, {"TER": REF_TER(references=R)}, None, **kw)()
    assert {k: str(v) for k, v in ours[0].items()} == {k: str(v) for k, v in theirs[0].items()}
    for x, y in zip(ours[1]["TER"], theirs[1]["TER"]):
        assert repr(x) == repr(y), (x, y)


def test_chrf_bootstrap_matches_reference():
    """sacreBLEU 2.6.0 on CPython 3.14.3 raises TypeError in estimate_ci for this input (an
    empty hypothesis segment; statistics.mean cannot mix numpy float32 and float). swadesh runs the
    same code path, so the exact-parity requirement is the same exception type and message."""
    with pytest.raises(TypeError) as theirs:
        REF_CHRF().corpus_score(H, R, n_bootstrap=100)
    with pytest.raises(TypeError) as ours:
        swadesh.CHRF().corpus_score(H, R, n_bootstrap=100)
    assert str(ours.value) == str(theirs.value)


def test_chrf_bootstrap_matches_reference_without_empty_segment():
    """Same path as above without the empty segment, where both return a score and a CI."""
    hyps, refs = H[:3], [s[:3] for s in R]
    a = swadesh.CHRF().corpus_score(hyps, refs, n_bootstrap=100)
    b = REF_CHRF().corpus_score(hyps, refs, n_bootstrap=100)
    assert str(a) == str(b) and a._mean == b._mean and a._ci == b._ci


class _BLEUOverride(swadesh.BLEU):
    """A BLEU subclass that changes score arithmetic must not take the native bootstrap path."""

    def _compute_score_from_stats(self, stats):
        return super()._compute_score_from_stats(stats)


class _RefBLEUOverride(REF_BLEU):
    def _compute_score_from_stats(self, stats):
        return super()._compute_score_from_stats(stats)


def test_bootstrap_with_overridden_score_matches_reference():
    a = _BLEUOverride().corpus_score(H, R, n_bootstrap=100)
    b = _RefBLEUOverride().corpus_score(H, R, n_bootstrap=100)
    assert str(a) == str(b) and a._mean == b._mean and a._ci == b._ci


# --------------------------------------------------------------------------- metrics.py helpers

def test_ref_cache_length_matches_reference():
    ours = swadesh.BLEU(references=R)._ref_cache
    theirs = REF_BLEU(references=R)._ref_cache
    assert bool(ours) and bool(theirs)
    assert len(ours) == len(theirs) == len(H)


@pytest.mark.parametrize("cls", [swadesh.BLEU, swadesh.CHRF, swadesh.TER])
@pytest.mark.parametrize("hyps,refs", [
    (H, [s[:3] for s in R]),          # hypotheses longer than the references
    (H[:2], R),                       # hypotheses shorter: references are cut
    (H, [R[0], R[1][:2]]),            # streams of different length
])
def test_truncation_matches_reference(cls, hyps, refs):
    ref_cls = REFS_BY_CLASS[cls]
    assert str(cls().corpus_score(hyps, refs)) == str(ref_cls().corpus_score(hyps, refs))


@pytest.mark.parametrize("cls", [swadesh.BLEU, swadesh.CHRF, swadesh.TER])
def test_no_references_error_matches_reference(cls):
    with pytest.raises(RuntimeError) as theirs:
        REFS_BY_CLASS[cls]().corpus_score(H, None)
    with pytest.raises(RuntimeError) as ours:
        cls().corpus_score(H, None)
    assert str(ours.value) == str(theirs.value)


def test_tokenized_period_warning_matches_reference(caplog):
    hyps = [f"word {i} ." for i in range(120)]
    refs = [[f"word {i}" for i in range(120)]]
    caplog.set_level(logging.WARNING, logger="sacrebleu")
    caplog.clear()
    REF_BLEU().corpus_score(hyps, refs)
    theirs = [r.getMessage() for r in caplog.records]
    caplog.clear()
    swadesh.BLEU().corpus_score(hyps, refs)
    ours = [r.getMessage() for r in caplog.records]
    assert theirs and ours == theirs
    caplog.clear()
    swadesh.BLEU(force=True).corpus_score(hyps, refs)
    assert not caplog.records


def test_chrf_tokenized_warning_when_force_disabled_matches_reference(caplog):
    """CHRF sets _force=True in sacreBLEU's base class, so the warning path needs it cleared by hand."""
    hyps = [f"word {i} ." for i in range(120)]
    refs = [[f"word {i}" for i in range(120)]]
    caplog.set_level(logging.WARNING, logger="sacrebleu")
    ref = REF_CHRF()
    ref._force = False
    caplog.clear()
    ref.corpus_score(hyps, refs)
    theirs = [r.getMessage() for r in caplog.records]
    ours_m = swadesh.CHRF()
    ours_m._force = False
    caplog.clear()
    ours_m.corpus_score(hyps, refs)
    ours = [r.getMessage() for r in caplog.records]
    assert ours == theirs


def test_sentence_bleu_without_effective_order_warns_like_reference(caplog):
    caplog.set_level(logging.WARNING, logger="sacrebleu")
    caplog.clear()
    a = swadesh.BLEU(effective_order=False).sentence_score(H[0], [R[0][0]]).score
    ours = [r.getMessage() for r in caplog.records]
    caplog.clear()
    b = REF_BLEU(effective_order=False).sentence_score(H[0], [R[0][0]]).score
    theirs = [r.getMessage() for r in caplog.records]
    assert a == b and ours == theirs and ours


def test_aggregate_accepts_lists_and_1d_arrays():
    m = swadesh.BLEU()
    arr = m._extract_corpus_statistics(H, R)
    ref = REF_BLEU()
    # list input goes to sacreBLEU's own aggregation
    assert str(m._aggregate_and_compute(arr.tolist())) == str(ref._aggregate_and_compute(arr.tolist()))
    # a single 1-D statistics vector
    assert str(m._aggregate_and_compute(arr.sum(0))) == str(ref._compute_score_from_stats(arr.sum(0).tolist()))


# Python-side preprocessing path: what ja-mecab / ko-mecab / spm / flores use. Those tokenizers need
# optional models that are not installed here, so the native-tokenizer check is switched off and the
# 13a/intl Python tokenization is used instead; results must still equal sacreBLEU's.
@pytest.fixture
def python_tokenization_path(monkeypatch):
    monkeypatch.setattr(swadesh.BLEU, "_native_tokenizer", property(lambda self: None))


@pytest.mark.parametrize("tok", ["13a", "intl"])
@pytest.mark.parametrize("lowercase", [False, True])
def test_python_tokenization_path_corpus_and_sentence(python_tokenization_path, tok, lowercase):
    hyps = H + ["Ends with a tokenized period ."]
    refs = [r + ["The cat sat."] for r in R]
    ours, theirs = swadesh.BLEU(tokenize=tok, lowercase=lowercase), REF_BLEU(tokenize=tok, lowercase=lowercase)
    assert str(ours.corpus_score(hyps, refs)) == str(theirs.corpus_score(hyps, refs))
    s_ours = ours.sentence_score(hyps[0], [refs[0][0]])
    s_theirs = theirs.sentence_score(hyps[0], [refs[0][0]])
    assert str(s_ours) == str(s_theirs)


SEED_HYPS = ["the cat sat on the mat", "a dog barked loudly in the yard",
             "prices rose by three percent"]
SEED_REFS = [["the cat is on the mat", "a dog barked in the yard", "prices rose by 3 percent"]]


def test_ter_explain_is_the_sentence_score_broken_apart():
    """`explain()` reports the parts of the same search: edits, shifts, and what is left wrong."""
    hyps = ["on the mat the cat sat", "the cat sat on the mat", "a zebra galloped past the gate"]
    refs = [["the cat sat on the mat", "the cat is on the mat", "a zebra ran past the gate"]]
    ter = swadesh.TER()
    rows = ter.explain(hyps, refs)
    scores = ter.sentence_scores(hyps, refs)
    assert [r["ter"] for r in rows] == scores, "ter is the sentence score, computed the same way"
    for i, (h, r, s) in enumerate(zip(hyps, rows, scores)):
        assert r["ter"] == REF_TER().sentence_score(h, [refs[0][i]]).score
        assert r["edits"] == r["shifts"] + r["distance"]
        c = r["counts"]
        assert r["distance"] == c["substitutions"] + c["insertions"] + c["deletions"]
        assert len(r["accepted_shifts"]) == r["shifts"]
        assert r["n_refs"] == 1 and r["best_ref"] == 0
        assert r["best_ref_len"] == len(ter._preprocess_segment(refs[0][i]).split())
        assert r["hyp_len"] == len(ter._preprocess_segment(h).split())

    # The first pair is a reordering: the search accepts a shift, and the distance is what is left.
    assert rows[0]["shifts"] >= 1
    assert rows[0]["distance"] == rows[0]["edits"] - rows[0]["shifts"]

    # Several references: the denominator is their mean length, the numerator the best reference's.
    two = swadesh.TER().explain(["the cat sat on the mat"],
                               [["the cat is on the mat"], ["a cat sat on a mat"]])[0]
    a = swadesh.TER().explain(["the cat sat on the mat"], [["the cat is on the mat"]])[0]
    b = swadesh.TER().explain(["the cat sat on the mat"], [["a cat sat on a mat"]])[0]
    assert two["n_refs"] == 2
    assert two["ref_len"] == (a["best_ref_len"] + b["best_ref_len"]) / 2
    assert two["edits"] == min(a["edits"], b["edits"])
    assert two["best_ref"] == (0 if a["edits"] <= b["edits"] else 1)

    # Reference caching and the length check behave like the scoring calls.
    cached = swadesh.TER(references=refs)
    assert [r["ter"] for r in cached.explain(hyps)] == scores
    with pytest.raises(swadesh.LengthMismatchError):
        with swadesh.strict():
            swadesh.TER().explain(hyps, [["only one"]])


def test_accumulator_is_corpus_score_in_constant_memory():
    """A streamed score is `corpus_score`; the state is one row wide, whatever the corpus size."""
    hyps = ["the cat sat on the mat", "a dog barked at the moon",
            "prices rose by 3.2% in October", "she sells sea shells",
            "the quick brown fox jumps over the lazy dog", "edition twelve is out now"]
    refs = [["the cat is on the mat", "a dog barked at the moon",
             "prices increased 3.2% in October", "she sells seashells",
             "the quick brown fox jumped over a lazy dog", "edition 12 is out now"]]
    for metric in (swadesh.BLEU(), swadesh.CHRF(), swadesh.CHRF(word_order=2), swadesh.TER()):
        whole = metric.corpus_score(hyps, refs)
        acc = swadesh.Accumulator(metric)
        for i, h in enumerate(hyps):
            acc.add(h, [s[i] for s in refs])
        assert str(acc.score()) == str(whole)
        assert acc.n_segments == len(hyps)

    # Incremental `add` == one `add_corpus` == a different segmentation of the same stream.
    a = swadesh.Accumulator(swadesh.BLEU())
    for i, h in enumerate(hyps):
        a.add(h, [refs[0][i]])
    b = swadesh.Accumulator(swadesh.BLEU()).add_corpus(hyps, refs)
    c = swadesh.Accumulator(swadesh.BLEU())
    for i in range(3):
        c.add_corpus(hyps[i::3], [refs[0][i::3]])
    assert str(a.score()) == str(b.score()) == str(c.score())
    assert b._rows is None and len(b._row) == len(a._row) == len(c._row)

    # A bootstrap interval resamples segments, so it needs the rows; the default says so.
    with pytest.raises(ValueError):
        b.score(n_bootstrap=100)
    kept = swadesh.Accumulator(swadesh.BLEU(), keep_rows=True).add_corpus(hyps, refs)
    with swadesh.seed(7):
        assert str(kept.score(n_bootstrap=50)) == str(
            swadesh.BLEU().corpus_score(hyps, refs, n_bootstrap=50))
    assert len(kept._rows) == len(hyps)

    # An empty accumulator takes the metric's own empty-input path, exactly as sacreBLEU does.
    with pytest.raises(IndexError):
        swadesh.Accumulator(swadesh.BLEU()).score()

    # strict mode still checks the lengths, and reset() lets the object be reused.
    with pytest.raises(swadesh.LengthMismatchError):
        with swadesh.strict():
            swadesh.Accumulator(swadesh.BLEU()).add_corpus(["a", "b"], [["x"]])
    assert a.reset().n_segments == 0


def test_n_jobs_sets_the_thread_count_without_changing_a_score():
    """`n_jobs` sizes the native pool; a score is the same at any count."""
    hyps = ["the cat sat on the mat number %d" % i for i in range(400)]
    refs = [["the cat is on the mat number %d" % i for i in range(400)]]
    assert swadesh.get_n_jobs() >= 1
    default = str(swadesh.BLEU().corpus_score(hyps, refs)) + str(swadesh.CHRF().corpus_score(hyps, refs))
    try:
        assert swadesh.set_n_jobs(1) == 1
        assert swadesh.get_n_jobs() == 1
        after = str(swadesh.BLEU().corpus_score(hyps, refs)) + str(swadesh.CHRF().corpus_score(hyps, refs))
        assert after == default, "the thread count must not change a score"
        with swadesh.n_jobs(2):
            assert swadesh.get_n_jobs() == 2
            assert str(swadesh.TER().corpus_score(hyps, refs)) == str(swadesh.TER().corpus_score(hyps, refs))
        assert swadesh.get_n_jobs() == 1, "the context manager restores the previous count"
    finally:
        swadesh.set_n_jobs(0)
    assert swadesh.get_n_jobs() >= 1, "0 restores the machine default"
    assert swadesh.set_n_jobs(None) >= 1, "None does too"


def test_evaluate_matches_the_individual_calls():
    """`evaluate()` is one call over the same metrics, not a second implementation.

    Idea 2's *speed* premise does not hold - chrF is character-based and shares no tokenization with
    BLEU, and the reference preparation is about 1% of the work (measured; see the module docstring
    and docs/feature-ideas.md) - so this pins the part that is real: the dict, the metric names, the
    one length check, and equality with each metric's own `corpus_score`.
    """
    hyps = ["The cat sat on the mat.", "A dog barked loudly.", "Prices rose by 3.2% in October."]
    refs = [["The cat is on the mat.", "The dog barked.", "Prices increased 3.2% in October."]]

    scores = swadesh.evaluate(hyps, refs)
    assert list(scores) == ["BLEU", "chrF2", "TER"], "the metrics' own names, in the order given"
    for name, metric in (("BLEU", swadesh.BLEU()), ("chrF2", swadesh.CHRF()), ("TER", swadesh.TER())):
        single = metric.corpus_score(hyps, refs)
        assert scores[name].score == single.score
        assert str(scores[name]) == str(single)
        assert scores[name].score == metric.corpus_score(hyps, refs).score

    # The result is a dict of score objects, so it formats and attributes like any other score.
    assert str(scores["BLEU"]).startswith("BLEU = ")
    assert scores["TER"].score >= 0

    # Classes, instances and a mapping all work, and nothing is reordered.
    assert list(swadesh.evaluate(hyps, refs, metrics=[swadesh.TER])) == ["TER"]
    assert list(swadesh.evaluate(hyps, refs, metrics={"anything": swadesh.BLEU()})) == ["BLEU"]
    assert list(swadesh.evaluate(hyps, refs, metrics=[swadesh.CHRF(word_order=2)])) == ["chrF2++"]

    # One length check for all of them, under strict mode.
    with pytest.raises(swadesh.LengthMismatchError):
        with swadesh.strict():
            swadesh.evaluate(hyps, [["a"]])

    # The bootstrap path is the metrics' own, seed and all: inside one seed block the resampled
    # interval is identical, and the printed interval proves the bootstrap path ran at all.
    with swadesh.seed(12345):
        booted = str(swadesh.evaluate(hyps, refs, metrics=[swadesh.BLEU], n_bootstrap=50)["BLEU"])
        direct = str(swadesh.BLEU().corpus_score(hyps, refs, n_bootstrap=50))
    assert booted == direct and "\u03bc =" in booted

    # And it agrees with sacreBLEU exactly.
    assert swadesh.evaluate(hyps, refs)["BLEU"].score == REF_BLEU().corpus_score(hyps, refs).score


def test_compare_is_the_pieces_assembled():
    """`swadesh.compare()` reports the same scores, intervals and p-values as the machinery it wraps.

    The p-value check is the point: it is compared against `PairedTest` run directly, so `compare`
    cannot drift into a second significance implementation.
    """
    base = ["The cat sat on the mat.", "A dog barked loudly.", "Prices rose by 3.2% in October."]
    new = ["The cat is on the mat.", "A dog barked.", "Prices rose by 3%."]
    refs = [["The cat is on the mat.", "The dog barked.", "Prices increased 3.2% in October."]]

    result = swadesh.compare([("base", base), ("new", new)], refs, n_samples=200)
    # The metric names are the metrics' own (`chrF2++`), which is what the CLI prints.
    assert result.metrics == ["BLEU", "chrF2++"]
    assert result.baseline == "base"
    assert result.row("base").is_baseline and not result.row("new").is_baseline
    assert result.row("base").p_values == {}, "the baseline carries no p-value"

    # The score is a plain corpus score; the interval and the p-value come from the paired resample.
    assert result.row("base").scores["BLEU"] == swadesh.BLEU().corpus_score(base, refs).score
    _sig, scores = swadesh.PairedTest([("base", base), ("new", new)], {"BLEU": swadesh.BLEU()}, refs,
                                      test_type="bs", n_samples=200)()
    assert result.row("new").p_values["BLEU"] == float(scores["BLEU"][1].p_value)
    assert result.row("new").intervals["BLEU"] == float(scores["BLEU"][1].ci)
    assert 0.0 <= result.row("new").p_values["BLEU"] <= 1.0

    # The table names the baseline, carries a header per metric, and the ranking is by score.
    table = result.table()
    assert "Baseline: base" in table and "BLEU (μ ± 95% CI)" in table and "p (BLEU)" in table
    assert len(table.splitlines()) == 3, table
    assert set(result.ranking()) == {"base", "new"}
    assert result.best() in {"base", "new"}
    assert isinstance(result.meaningful(), list)
    assert json.loads(json.dumps(result.to_dict()))["systems"][0]["system"] == "base"

    # A mapping works as well as a list of pairs, and a single system needs no pairing.
    same = swadesh.compare({"base": base, "new": new}, refs, n_samples=200)
    assert same.row("new").p_values == result.row("new").p_values
    alone = swadesh.compare([("only", base)], refs, n_samples=100)
    assert alone.rows[0].p_values == {} and alone.row("only").intervals["BLEU"] >= 0
    assert alone.ranking() == ["only"]

    with pytest.raises(ValueError):
        swadesh.compare([("a", base)], refs, baseline="nope")
    with pytest.raises(ValueError):
        swadesh.compare([], refs)


def test_seed_from_the_process(monkeypatch):
    """`swadesh.seed()` is the in-process seed, and every signature reports the one in effect.

    Reproducibility is the point: two runs inside the same seed give the same numbers and say so in
    their own signature line, and a different seed says something different.
    """
    monkeypatch.delenv("SACREBLEU_SEED", raising=False)

    def run(value):
        with swadesh.seed(value):
            m = swadesh.BLEU()
            score = m.corpus_score(SEED_HYPS, SEED_REFS, n_bootstrap=200)
            return str(m.get_signature()), str(score)

    sig7a, text7a = run(7)
    sig7b, text7b = run(7)
    assert "seed:7" in sig7a
    assert (sig7a, text7a) == (sig7b, text7b), "the same seed must reproduce the same interval"
    sig8, text8 = run(8)
    assert "seed:8" in sig8
    assert text8 != text7a, "a different seed must move the resampled interval"

    # The environment is the fallback; a block overrides it, and leaving the block restores it.
    monkeypatch.setenv("SACREBLEU_SEED", "99")
    m = swadesh.BLEU()
    m.corpus_score(SEED_HYPS, SEED_REFS, n_bootstrap=100)
    assert "seed:99" in str(m.get_signature())
    with swadesh.seed(7):
        m = swadesh.BLEU()
        m.corpus_score(SEED_HYPS, SEED_REFS, n_bootstrap=100)
        assert "seed:7" in str(m.get_signature())
    m = swadesh.BLEU()
    m.corpus_score(SEED_HYPS, SEED_REFS, n_bootstrap=100)
    assert "seed:99" in str(m.get_signature())

    # The default is still sacreBLEU's 12345, and it still agrees with sacreBLEU exactly.
    monkeypatch.delenv("SACREBLEU_SEED", raising=False)
    ours = swadesh.BLEU().corpus_score(SEED_HYPS, SEED_REFS, n_bootstrap=100)
    theirs = REF_BLEU().corpus_score(SEED_HYPS, SEED_REFS, n_bootstrap=100)
    assert str(ours) == str(theirs)
    plain = swadesh.BLEU()
    plain.corpus_score(SEED_HYPS, SEED_REFS)          # no bootstrap: no bs: and no seed:
    assert str(plain.get_signature()) == "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0"

    # `PairedTest` captures the seed when it is constructed, so the block has to wrap the constructor.
    with swadesh.seed(5):
        paired = swadesh.PairedTest([("a", SEED_HYPS), ("b", list(reversed(SEED_HYPS)))],
                                    {"bleu": swadesh.BLEU()}, SEED_REFS, test_type="bs", n_samples=50)
    assert paired._seed == 5

    swadesh.set_seed(11)
    try:
        m = swadesh.BLEU()
        m.corpus_score(SEED_HYPS, SEED_REFS, n_bootstrap=50)
        assert "seed:11" in str(m.get_signature())
    finally:
        swadesh.set_seed(None)


def test_strict_mode_raises_instead_of_truncating(monkeypatch):
    """`swadesh.strict()` refuses the truncation sacreBLEU performs silently.

    A one-element reference stream against six hypotheses produces a score from the first segment
    under sacreBLEU; that is reproduced by default (parity), and refused inside `strict()`, with the
    lengths named. The mode is a context manager, so it cannot change any call's signature.
    """
    monkeypatch.delenv("SWADESH_STRICT", raising=False)
    hyps = ["a", "b", "c", "d", "e", "f"]
    refs = [["a"]]

    # Default: sacreBLEU's own behaviour, to the last bit.
    assert swadesh.BLEU().corpus_score(hyps, refs).score == REF_BLEU().corpus_score(hyps, refs).score

    with pytest.raises(swadesh.LengthMismatchError) as caught:
        with swadesh.strict():
            swadesh.BLEU().corpus_score(hyps, refs)
    message = str(caught.value)
    assert "hypotheses: 6" in message and "reference 0: 1" in message, message

    for call in (lambda: swadesh.CHRF().corpus_score(hyps, refs),
                 lambda: swadesh.CHRF().sentence_scores(hyps, refs),
                 lambda: swadesh.TER().corpus_score(hyps, refs),
                 lambda: swadesh.TER().sentence_scores(hyps, refs)):
        with pytest.raises(swadesh.LengthMismatchError):
            with swadesh.strict():
                call()

    # A nested `strict(False)` asks for sacreBLEU's behaviour for one call, and the block restores
    # the default on exit.
    with swadesh.strict():
        with swadesh.strict(False):
            assert swadesh.BLEU().corpus_score(hyps, refs).score == REF_BLEU().corpus_score(hyps, refs).score
    assert swadesh.BLEU().corpus_score(hyps, refs).score == REF_BLEU().corpus_score(hyps, refs).score

    # Matching streams never raise, strict or not, and a second reference may be None.
    with swadesh.strict():
        assert swadesh.BLEU().corpus_score(["a", "b"], [["a", "b"]]) is not None
        assert swadesh.TER().corpus_score(["a b", "c d"], [["a b", None], ["x", "c d"]]) is not None


def test_strict_mode_from_the_environment(monkeypatch):
    """`SWADESH_STRICT` is the process-wide switch, for pipelines that cannot add a `with`."""
    monkeypatch.setenv("SWADESH_STRICT", "1")
    with pytest.raises(swadesh.LengthMismatchError):
        swadesh.BLEU().corpus_score(["a", "b"], [["a"]])
    monkeypatch.setenv("SWADESH_STRICT", "0")
    assert swadesh.BLEU().corpus_score(["a", "b"], [["a"]]).score == \
        REF_BLEU().corpus_score(["a", "b"], [["a"]]).score
    swadesh.set_strict(True)
    try:
        with pytest.raises(swadesh.LengthMismatchError):
            swadesh.BLEU().corpus_score(["a", "b"], [["a"]])
    finally:
        swadesh.set_strict(None)


def test_python_tokenization_path_pairwise(python_tokenization_path):
    ours = swadesh.BLEU(tokenize="intl").pairwise(H)
    theirs = np.array([[REF_BLEU(tokenize="intl").sentence_score(c, [d]).score for d in H] for c in H])
    assert np.array_equal(ours, theirs)


def test_mbr_without_self_exclusion_matches_reference():
    best, util = swadesh.mbr(H, exclude_self=False)
    m = np.array([[REF_CHRF().sentence_score(c, [d]).score for d in H] for c in H])
    assert np.array_equal(util, m.mean(1))
    assert best == int(np.argmax(m.mean(1)))


# --------------------------------------------------------------------------- rewards.py

def test_reward_completion_formats_match_reference():
    refs = ["The cat sat on the mat.", "Prices rose.", "The cat sat.", "42 apples"]
    completions = [
        "The cat is on the mat.",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Prices rose."}],
        {"content": "The cat sat."},
        42,
    ]
    got = sw_rewards.chrf_reward()(completions=completions, reference=refs)
    texts = ["The cat is on the mat.", "Prices rose.", "The cat sat.", "42"]
    expect = [REF_CHRF().sentence_score(t, [r]).score * 0.01 for t, r in zip(texts, refs)]
    assert got == expect


def test_reward_missing_column_raises():
    with pytest.raises(KeyError, match="reference"):
        sw_rewards.chrf_reward()(completions=["a"], answer=["b"])


# --------------------------------------------------------------------------- cli.py and verify.py

@pytest.fixture
def corpus_files(tmp_path):
    hyp, ref = tmp_path / "hyp.txt", tmp_path / "ref.txt"
    hyp.write_text("\n".join(H) + "\n", encoding="utf-8")
    ref.write_text("\n".join(R[0]) + "\n", encoding="utf-8")
    return hyp, ref


def _fresh_stdin(monkeypatch):
    # sacreBLEU's CLI reopens fd 0/1 (sys.stdin.fileno()), which pytest's capture objects lack, and
    # it takes ownership of the fd it wraps, so every CLI invocation needs its own stdin.
    monkeypatch.setattr(sys, "stdin", open(os.devnull))


def test_cli_main_output_matches_sacrebleu_cli(corpus_files, monkeypatch, capfd):
    hyp, ref = corpus_files
    argv = [str(ref), "-i", str(hyp), "-m", "bleu", "chrf", "ter", "-f", "json"]
    monkeypatch.setattr(sys, "argv", ["sacrebleu"] + argv)
    _fresh_stdin(monkeypatch)
    with _sacrebleu_stdout():
        sb_cli.main()
    theirs = capfd.readouterr().out
    _fresh_stdin(monkeypatch)
    try:
        with _sacrebleu_stdout():
            swadesh.cli.main(argv)
    finally:
        swadesh.unpatch()
    ours = capfd.readouterr().out
    assert theirs and ours == theirs


def test_cli_module_entrypoint(corpus_files, monkeypatch, capfd):
    hyp, ref = corpus_files
    monkeypatch.setattr(sys, "argv", ["swadesh", str(ref), "-i", str(hyp), "-m", "bleu"])
    _fresh_stdin(monkeypatch)
    try:
        with pytest.raises(SystemExit) as ei, _sacrebleu_stdout():
            runpy.run_module("swadesh.cli", run_name="__main__")
    finally:
        swadesh.unpatch()
    assert not ei.value.code
    assert capfd.readouterr().out.strip()


class _WrongBLEU(swadesh.BLEU):
    """Deliberately wrong, to check that the verifier reports mismatches."""

    def sentence_scores(self, hypotheses, references):
        return [s + 1.0 for s in super().sentence_scores(hypotheses, references)]

    def corpus_score(self, hypotheses, references, n_bootstrap=1):
        return REF_CHRF().corpus_score(hypotheses, references)


def test_verify_identical_reports_ok(capsys):
    assert sw_verify.verify(H, R) is True
    out = capsys.readouterr().out
    assert "identical" in out and "DIFFERENT" not in out
    assert sw_verify.verify(H, R, verbose=False) is True
    assert capsys.readouterr().out == ""


def test_verify_detects_mismatches(monkeypatch, capsys):
    monkeypatch.setattr(swadesh, "BLEU", _WrongBLEU)
    assert sw_verify.verify(H, R) is False
    out = capsys.readouterr().out
    assert "DIFFERENT" in out and "first sentence mismatch at line 1" in out


def test_verify_main_exit_codes(corpus_files, monkeypatch, capsys):
    hyp, ref = corpus_files
    args = ["-r", str(ref), "-i", str(hyp)]
    assert sw_verify.main(args) == 0
    assert "ALL IDENTICAL" in capsys.readouterr().out
    assert sw_verify.main(args + ["-tok", "intl", "-lc"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(swadesh, "BLEU", _WrongBLEU)
    assert sw_verify.main(args) == 1
    assert "MISMATCH FOUND" in capsys.readouterr().out


def test_verify_module_entrypoint(corpus_files, monkeypatch, capsys):
    hyp, ref = corpus_files
    monkeypatch.setattr(sys, "argv", ["verify", "-r", str(ref), "-i", str(hyp)])
    with pytest.raises(SystemExit) as ei:
        runpy.run_module("swadesh.verify", run_name="__main__")
    assert ei.value.code == 0


# --------------------------------------------------------------------------- mcp_server.py

def _mcp_call(name, args):
    from swadesh.mcp_server import server
    return asyncio.run(server.call_tool(name, args))


def test_mcp_errors_for_bad_arguments():
    pytest.importorskip("mcp")
    bad = [
        ("corpus_score", {"hypotheses": H, "references": R, "metrics": ["nope"]}, "unknown metric"),
        ("corpus_score", {"hypotheses": H, "references": [R[0][:2]]}, "as long as hypotheses"),
        ("sentence_scores", {"hypotheses": H, "references": R, "metric": "nope"}, "unknown metric"),
        ("mbr_select", {"candidates": ["only one"]}, "at least two"),
    ]
    from mcp.server.mcpserver.exceptions import ToolError
    for name, args, msg in bad:
        # In-process, the SDK re-raises the ToolError; over a transport it becomes an
        # is_error result. Either way the agent must see our message.
        try:
            res = _mcp_call(name, args)
            assert res.is_error, name
            text = res.content[0].text
        except ToolError as e:
            text = str(e)
        assert msg in text, (name, text)


def test_mcp_confidence_interval_matches_reference():
    pytest.importorskip("mcp")
    import json
    res = _mcp_call("corpus_score", {"hypotheses": H, "references": R, "metrics": ["bleu"], "confidence": True})
    out = json.loads(res.content[0].text)["bleu"]
    ref = REF_BLEU().corpus_score(H, R, n_bootstrap=1000)
    assert out["mean"] == float(ref._mean) and out["ci95"] == float(ref._ci)


def test_mcp_sentence_scores_bleu_and_ter_match_reference():
    pytest.importorskip("mcp")
    import json
    out = json.loads(_mcp_call("sentence_scores", {"hypotheses": H, "references": R, "metric": "ter"}).content[0].text)
    assert out["scores"] == [REF_TER().sentence_score(h, [r[i] for r in R]).score for i, h in enumerate(H)]
    out = json.loads(_mcp_call("sentence_scores", {"hypotheses": H, "references": R, "metric": "bleu"}).content[0].text)
    assert out["scores"] == [REF_BLEU(effective_order=True).sentence_score(h, [r[i] for r in R]).score
                             for i, h in enumerate(H)]


def test_mcp_main_and_entrypoint_run_stdio(monkeypatch):
    mcp_mod = pytest.importorskip("mcp.server.mcpserver")
    from swadesh import mcp_server
    seen = []
    monkeypatch.setattr(mcp_mod.MCPServer, "run", lambda self, transport: seen.append(transport))
    mcp_server.main()
    runpy.run_module("swadesh.mcp_server", run_name="__main__")
    assert seen == ["stdio", "stdio"]


# --------------------------------------------------------------------------- round 2: remaining Python lines

def _score_outcome(fn):
    """Score or exception of a corpus call, compared exactly (type, message and score string)."""
    try:
        return ("ok", str(fn()))
    except Exception as e:  # the exception itself is the outcome being compared
        return ("error", type(e).__name__, str(e))


def _paired_outcome(fn):
    """Result or exception of a paired test, in a form that compares exactly (type, message, every value)."""
    try:
        res = fn()
    except Exception as e:  # the exception itself is the outcome being compared
        return ("error", type(e).__name__, str(e))
    return ("ok", {k: str(v) for k, v in res[0].items()},
            {m: [repr(x) for x in vals] for m, vals in res[1].items()})


def test_py_statistics_no_references_and_cached_references(python_tokenization_path):
    """metrics._py_statistics: no references and an empty cache is an error; with a cache it is reused."""
    ours, theirs = swadesh.BLEU(tokenize="13a"), REF_BLEU(tokenize="13a")
    steps = [lambda m: m.corpus_score(H, None), lambda m: m.corpus_score(H, R), lambda m: m.corpus_score(H, None)]
    for step in steps:
        assert _score_outcome(lambda: step(ours)) == _score_outcome(lambda: step(theirs))
    # references given at construction are cached; a later call without references reuses them
    a, b = swadesh.BLEU(tokenize="13a", references=R), REF_BLEU(tokenize="13a", references=R)
    assert _score_outcome(lambda: a.corpus_score(H, None)) == _score_outcome(lambda: b.corpus_score(H, None))


def test_py_statistics_tokenized_hypothesis_without_force_warns_like_reference(python_tokenization_path, caplog):
    hyps = [f"word {i} ." for i in range(120)]
    refs = [[f"word {i}" for i in range(120)]]
    caplog.set_level(logging.WARNING, logger="sacrebleu")
    caplog.clear()
    ref_score = REF_BLEU(tokenize="13a").corpus_score(hyps, refs)
    theirs = [r.getMessage() for r in caplog.records]
    caplog.clear()
    our_score = swadesh.BLEU(tokenize="13a").corpus_score(hyps, refs)
    ours = [r.getMessage() for r in caplog.records]
    assert str(our_score) == str(ref_score)
    assert theirs and ours == theirs


def test_resampled_sums_out_of_range_and_negative_indices_match_numpy():
    from swadesh import significance as sig
    stats = np.arange(12, dtype=np.float64).reshape(4, 3)
    s32 = np.array(stats, dtype="float32")
    wrap = np.array([[-1, 2], [0, 1]])
    assert np.array_equal(sig._resampled_sums(stats, wrap), s32[wrap].sum(1))  # negative: wrap-around
    bad = np.array([[0, 9]])
    with pytest.raises(IndexError) as theirs:
        s32[bad]
    with pytest.raises(IndexError) as ours:
        sig._resampled_sums(stats, bad)
    assert str(ours.value) == str(theirs.value)


@pytest.mark.parametrize("paired_metric", ["bs", "ar"])
def test_paired_chrf_without_numpy2_matches_reference(monkeypatch, paired_metric):
    """significance._NUMPY2 False makes the native CHRF kernels decline; sacreBLEU's own code runs."""
    from swadesh import significance as sig
    named = [("baseline", H), ("sys2", [h.upper() for h in H])]
    kw = dict(test_type=paired_metric, n_samples=200)
    theirs = _paired_outcome(lambda: REF_PairedTest(named, {"CHRF": REF_CHRF(references=R)}, None, **kw)())
    monkeypatch.setattr(sig, "_NUMPY2", False)
    try:
        swadesh.patch()
        ours = _paired_outcome(lambda: swadesh.PairedTest(named, {"CHRF": swadesh.CHRF(references=R)}, None, **kw)())
    finally:
        swadesh.unpatch()
    assert ours == theirs


class _ChrFOverride(swadesh.CHRF):
    """Overriding the F-score arithmetic must disable the native CHRF significance kernel."""

    def _compute_f_score(self, statistics):
        return super()._compute_f_score(statistics)


def test_paired_chrf_subclass_with_overridden_score_matches_reference():
    named = [("baseline", H), ("sys2", [h + " extra" for h in H])]
    kw = dict(test_type="bs", n_samples=200)
    theirs = _paired_outcome(lambda: REF_PairedTest(named, {"CHRF": REF_CHRF(references=R)}, None, **kw)())
    ours = _paired_outcome(lambda: swadesh.PairedTest(named, {"CHRF": _ChrFOverride(references=R)}, None, **kw)())
    assert ours == theirs


def test_paired_chrf_zero_order_matches_reference():
    named = [("baseline", H), ("sys2", [h + " extra" for h in H])]
    kw = dict(test_type="bs", n_samples=200)
    theirs = _paired_outcome(lambda: REF_PairedTest(named, {"CHRF": REF_CHRF(char_order=0, word_order=0, references=R)}, None, **kw)())
    ours = _paired_outcome(lambda: swadesh.PairedTest(named, {"CHRF": swadesh.CHRF(char_order=0, word_order=0, references=R)}, None, **kw)())
    assert ours == theirs


def test_mcp_argument_validators_raise_clear_messages():
    pytest.importorskip("mcp")
    from swadesh import mcp_server as ms
    with pytest.raises(ValueError, match="hypotheses is empty"):
        ms._check([], [["a"]])
    with pytest.raises(ValueError, match="references must be"):
        ms._check(["a"], [])
    with pytest.raises(ValueError, match="as long as hypotheses"):
        ms._check(["a"], [[]])
    with pytest.raises(ValueError, match="unknown metric"):
        ms.sentence_scores(["a"], [["a"]], metric="nope")
    with pytest.raises(ValueError, match="at least two"):
        ms.mbr_select(["only one"])


def test_py_statistics_tokenized_hypothesis_with_empty_reference_warns_like_reference(caplog):
    """A None reference segment sets has_empty, which sends the call to _py_statistics."""
    hyps = [f"word {i} ." for i in range(120)]
    refs = [[None if i % 3 == 0 else f"word {i}" for i in range(120)]]
    caplog.set_level(logging.WARNING, logger="sacrebleu")
    caplog.clear()
    ref_outcome = _score_outcome(lambda: REF_BLEU().corpus_score(hyps, refs))
    theirs = [r.getMessage() for r in caplog.records]
    caplog.clear()
    our_outcome = _score_outcome(lambda: swadesh.BLEU().corpus_score(hyps, refs))
    ours = [r.getMessage() for r in caplog.records]
    assert our_outcome == ref_outcome
    assert ours == theirs


def test_native_bleu_declines_chrf_rows_it_does_not_understand():
    """Malformed rows make the native significance kernel return None, so sacreBLEU's scorer runs."""
    from swadesh import significance as sig
    m = swadesh.CHRF()
    assert sig._native_bleu_scores(m, np.zeros(18, dtype=np.float32)) is None  # 1-D rows
    assert sig._native_bleu_scores(m, np.zeros((2, 5), dtype=np.float32)) is None  # wrong width


# --------------------------------------------------------------------------- round 3: native kernels by config

@pytest.mark.parametrize("make_ours,make_ref", [
    (lambda: swadesh.BLEU(max_ngram_order=1), lambda: REF_BLEU(max_ngram_order=1)),   # resample width 4
    (lambda: swadesh.BLEU(max_ngram_order=2), lambda: REF_BLEU(max_ngram_order=2)),   # width 6
    (lambda: swadesh.BLEU(max_ngram_order=3), lambda: REF_BLEU(max_ngram_order=3)),   # width 8
    (lambda: swadesh.BLEU(max_ngram_order=5), lambda: REF_BLEU(max_ngram_order=5)),   # 5-grams: wide keys, multi-ref merge
    (lambda: swadesh.BLEU(max_ngram_order=6), lambda: REF_BLEU(max_ngram_order=6)),   # width 14
    (lambda: swadesh.CHRF(char_order=1, word_order=0), lambda: REF_CHRF(char_order=1, word_order=0)),  # width 3
    (lambda: swadesh.CHRF(char_order=2, word_order=0), lambda: REF_CHRF(char_order=2, word_order=0)),  # width 6
    (lambda: swadesh.CHRF(char_order=4, word_order=0), lambda: REF_CHRF(char_order=4, word_order=0)),  # width 12
    (lambda: swadesh.CHRF(char_order=3, word_order=0), lambda: REF_CHRF(char_order=3, word_order=0)),  # width 9
    (lambda: swadesh.CHRF(char_order=7, word_order=0), lambda: REF_CHRF(char_order=7, word_order=0)),  # 21-bit keys x 7: wide
    (lambda: swadesh.CHRF(char_order=6, word_order=2), lambda: REF_CHRF(char_order=6, word_order=2)),  # chrF++ wide mix
], ids=["bleu1", "bleu2", "bleu3", "bleu5", "bleu6", "chrf1", "chrf2", "chrf4", "chrf3", "chrf7wide", "chrf6pp"])
def test_native_kernel_config_matches_reference(make_ours, make_ref):
    hyps = H + ["A cat is sitting on a mat today."]
    refs = [[*R[0], "tail"], [*R[1], "tail"]]  # two streams, each as long as hypotheses
    assert _score_outcome(lambda: make_ours().corpus_score(hyps, refs)) == \
        _score_outcome(lambda: make_ref().corpus_score(hyps, refs))
    assert _score_outcome(lambda: make_ours().corpus_score(hyps, refs, n_bootstrap=50)) == \
        _score_outcome(lambda: make_ref().corpus_score(hyps, refs, n_bootstrap=50))


def test_chrf_whitespace_pairwise_matches_reference():
    ours = swadesh.CHRF(whitespace=True).pairwise(H)
    ref = np.array([[REF_CHRF(whitespace=True).sentence_score(c, [d]).score for d in H] for c in H])
    assert np.array_equal(ours, ref)


_TER_CASES = [
    ("the dog's", ["the dog's"]),                                   # possessive at the end of a sentence
    ("a", [" ".join(f"w{i}" for i in range(50))]),                  # hypothesis far shorter: beam width grows
    ("你好世界", ["你好世界吗"]),  # all CJK: asian_support wrap
]


@pytest.mark.parametrize("hyp,refs", _TER_CASES, ids=["possessive", "short-hyp", "cjk"])
def test_ter_edge_cases_match_reference(hyp, refs):
    for kw in ({}, {"asian_support": True}, {"normalized": True}):
        ours = _score_outcome(lambda: swadesh.TER(**kw).sentence_score(hyp, refs))
        theirs = _score_outcome(lambda: REF_TER(**kw).sentence_score(hyp, refs))
        assert ours == theirs, (kw, ours, theirs)


def test_ter_repeated_prefixes_corpus_matches_reference():
    """Many segments sharing word prefixes exercise the TER beam-search cache hit path."""
    hyps = [f"the cat sat on the mat number {i} today" for i in range(60)]
    refs = [[f"the cat sat on a mat number {i}" for i in range(60)]]
    assert str(swadesh.TER().corpus_score(hyps, refs)) == str(REF_TER().corpus_score(hyps, refs))
