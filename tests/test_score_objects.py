"""The score objects are swadesh's own and format identically to sacreBLEU's.

These are the objects the CLI prints, so their text, score-only and JSON forms are part of the
output contract. The broader parity suite compares whole CLI output; this pins the objects
themselves, including the JSON key order.
"""

import pytest

import sacrebleu
import swadesh.metrics as ours

BLEU_ARGS = dict(
    score=34.5,
    counts=[10, 8, 6, 4],
    totals=[12, 10, 8, 6],
    precisions=[83.3, 80.0, 75.0, 66.7],
    bp=0.98,
    sys_len=100,
    ref_len=102,
)


def _pairs():
    import sacrebleu.metrics.bleu as bl
    import sacrebleu.metrics.chrf as ch
    import sacrebleu.metrics.ter as te

    return [
        ("BLEU", lambda m: m.BLEUScore(**BLEU_ARGS), bl.BLEUScore),
        ("chrF++", lambda m: m.CHRFScore(66.6, 6, 2, 2), ch.CHRFScore),
        ("chrF", lambda m: m.CHRFScore(66.6, 6, 0, 2), ch.CHRFScore),
        ("TER", lambda m: m.TERScore(50.0, 25, 50.0), te.TERScore),
    ]


@pytest.mark.parametrize("name,make,oracle_cls", _pairs(), ids=lambda x: x if isinstance(x, str) else "")
def test_score_repr_matches(name, make, oracle_cls):
    assert repr(make(ours)) == repr(make(sacrebleu.metrics))


@pytest.mark.parametrize("name,make,oracle_cls", _pairs(), ids=lambda x: x if isinstance(x, str) else "")
def test_score_formats_match(name, make, oracle_cls):
    sig = "nrefs:1|case:mixed|version:2.6.0"
    for kwargs in (
        dict(),
        dict(width=1),
        dict(width=4),
        dict(score_only=True),
        dict(is_json=True, signature=sig),
        dict(width=1, is_json=True, signature=sig),
    ):
        got = make(ours).format(**kwargs)
        want = make(sacrebleu.metrics).format(**kwargs)
        assert got == want, f"{name} format{kwarqs_suffix(kwargs)}"


def kwarqs_suffix(kwargs):
    return f"({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"


def test_estimate_ci_matches():
    a = ours.Score("BLEU", 0.0)
    b = sacrebleu.metrics.base.Score("BLEU", 0.0)
    scores = [ours.Score("BLEU", float(x)) for x in range(80)]
    oracle_scores = [sacrebleu.metrics.base.Score("BLEU", float(x)) for x in range(80)]
    a.estimate_ci(scores)
    b.estimate_ci(oracle_scores)
    assert (a._mean, a._ci) == (b._mean, b._ci)


def test_score_is_not_a_sacrebleu_subclass():
    assert not isinstance(ours.Score("BLEU", 1.0), sacrebleu.metrics.base.Score)