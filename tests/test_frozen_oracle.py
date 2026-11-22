"""Exactness against the frozen oracle corpus (tests/oracle/), without needing sacreBLEU.

tools/freeze_oracle.py recorded sacreBLEU 2.6.0's outputs once; these tests replay them against
swadesh. Fuzz cases carry their own hypotheses; WMT22 and model cases are rebuilt from
bench/data and tests/corpus/model with the same deterministic perturbations as the freezer.

    pytest tests/test_frozen_oracle.py -q
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import swadesh

ORACLE = Path(__file__).resolve().parent / "oracle"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "bench" / "data"
CORPUS = Path(__file__).resolve().parent / "corpus" / "model"

METRICS = {
    "BLEU": lambda tok: swadesh.BLEU(tokenize=tok),
    "BLEU floor0.1": lambda tok: swadesh.BLEU(tokenize=tok, smooth_method="floor", smooth_value=0.1),
    "BLEU intl": lambda tok: swadesh.BLEU(tokenize="intl"),
    "chrF": lambda tok: swadesh.CHRF(),
    "chrF b3": lambda tok: swadesh.CHRF(beta=3),
    "chrF++": lambda tok: swadesh.CHRF(word_order=2),
    "TER": lambda tok: swadesh.TER(),
    "TER norm": lambda tok: swadesh.TER(normalized=True),
}


def perturbed(refs, n, seed=1234):
    """Identical to tools/freeze_oracle.py:perturbed."""
    rnd = random.Random(seed)
    out = []
    while len(out) < n:
        for r in refs:
            w = r.split()
            for _ in range(max(1, len(w) // 6)):
                if not w:
                    break
                op = rnd.random()
                k = rnd.randrange(len(w))
                if op < 0.3:
                    w[k] = rnd.choice(["the", "a", "is", "nicht", "und"])
                elif op < 0.5:
                    del w[k]
                elif op < 0.7:
                    w.insert(k, rnd.choice(["sehr", "ganz", "so"]))
                elif op < 0.85:
                    w[k] = w[k].upper()
            out.append(" ".join(w))
            if len(out) >= n:
                break
    return out


def _case_data(entry):
    """(hyps, refs) for one oracle entry."""
    if "hyps" in entry:
        return entry["hyps"], entry["refs"]
    stem = entry["case"].split(":", 1)[0]
    if (DATA / f"wmt22.{stem}.ref").exists():
        refs = (DATA / f"wmt22.{stem}.ref").read_text(encoding="utf-8").splitlines()[:400]
        return perturbed(refs, len(refs)), [refs]
    model = stem[len("model-"):] if stem.startswith("model-") else stem
    jf = CORPUS / f"{model}.jsonl"
    recs = sorted((json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines()), key=lambda r: r["i"])
    rt = CORPUS / jf.name.replace("real-", "ref-").replace(".jsonl", ".txt")
    refs = [[l.rstrip("\n") for l in rt.read_text(encoding="utf-8").splitlines()]]
    return [r["hyp"] for r in recs], refs


def _compare(entry):
    name = entry["case"]
    base = name.split(":", 1)[1]
    builder = METRICS[base]
    tok = entry["tokenizer"]
    want = entry["expect"]
    hyps, refs = _case_data(entry)
    m = builder(tok)

    if isinstance(want, dict) and want.get("exc"):
        with pytest.raises(Exception) as ei:  # noqa: B017 -- the type is asserted below
            m.corpus_score(hyps, refs)
        assert type(ei.value).__name__ == want["exc"], name
        return

    got = m.corpus_score(hyps, refs)
    assert got.score.hex() == want["score"], f"{name}: corpus score"
    assert str(got) == want["str"], f"{name}: formatted corpus"
    assert got.format(signature="sig|x:1") == want["format"], f"{name}: format()"
    rows = [list(x) for x in zip(*refs)]
    for i, (h, r, w) in enumerate(zip(hyps, rows, want["sentences"])):
        if isinstance(w, dict):
            with pytest.raises(Exception):  # noqa: B017
                m.sentence_score(h, r)
            continue
        assert m.sentence_score(h, r).score.hex() == w, f"{name}[{i}]: sentence score"


def _files():
    return sorted(ORACLE.glob("*.jsonl")) if ORACLE.exists() else []


_FILES = sorted(ORACLE.glob("*.jsonl")) if ORACLE.exists() else []


@pytest.mark.parametrize("path", _FILES, ids=[p.stem for p in _FILES])
# NOTE: _files() is evaluated at import; if tests/oracle is empty at import time the
# parametrization is empty, which is why test_oracle_present exists.
def test_frozen_oracle(path):
    """swadesh reproduces every frozen sacreBLEU output exactly."""
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = []
    for entry in entries:
        if entry["case"].split(":", 1)[1] not in METRICS:
            continue
        try:
            _compare(entry)
        except AssertionError as e:
            failures.append(str(e))
    assert not failures, "\n".join(failures[:5])


def test_oracle_present():
    assert _files(), "tests/oracle is empty: run tools/freeze_oracle.py"