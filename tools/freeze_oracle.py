"""Freeze the oracle: record sacreBLEU 2.6.0's exact outputs as data, once.

After this runs, the test suite compares against tests/oracle/*.jsonl and needs no sacreBLEU.
The live oracle (running sacreBLEU directly) stays available as a development flag.

    python tools/freeze_oracle.py [--out tests/oracle] [--limit N]

Records, for every case: sufficient statistics as ints, corpus score and every sentence score as
float.hex(), the formatted score string, the signature, and the exception type and message when
either side raises. Cases cover WMT22 (seven pairs), the adversarial fuzz alphabet at fixed seeds,
and real model output under tests/corpus/model/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "bench" / "data"
PAIRS = ["cs-en", "de-en", "en-de", "en-ja", "en-ru", "en-zh", "zh-en"]
TOK = {"en-zh": "zh", "en-ja": "char"}

# The adversarial alphabet every fuzz test draws from (see tests/test_differential.py).
ALPHABET = (
    list("abcXYZ019 .,-'\"!?;:()[]{}<>/\\@#$%^&*_+=|~`")
    + ["&amp;", "&quot;", "&lt;", "&gt;", "<skipped>", "-\n", "\n", "\t", "  "]
    + list("\u00a0\u2000\u2001\u200a\u3000\u2028\u1680\x1c\x85\u200b")
    + list("\u4e2d\u6587\u5b57\u7b26\u6e2c\u8a66\u65e5\u672c\u8a9e\ud55c\uad6d\uc5b4"
           "\uff46\uff55\uff4c\uff4c\uff0c\u3002\u2e80\u31c0\u2f81\u2fa1\u2a6d\u2a6e"
           "\u2026\u20ac\u2192")
    + list("\u03a3\u03c3\u03c2\u0130I\u0131\u00df\u1e9e\u01c5\ufb01\u0301\u00e9\u0663\u00b2\u2167\u00bd\u00bf\u00a1\u00ab\u00bb\u201e\u201c\u201d\u2018\u2019\u2013\u2014")
    + ["\U0001F600", "\U00020001", "\U0002A6D6", "\U0001D7CE"]
)

METRICS = [
    ("BLEU", lambda tok: sacrebleu.BLEU(tokenize=tok), {}),
    ("BLEU floor0.1", lambda tok: sacrebleu.BLEU(tokenize=tok, smooth_method="floor", smooth_value=0.1), {}),
    ("BLEU intl", lambda tok: sacrebleu.BLEU(tokenize="intl"), {}),
    ("chrF", lambda tok: sacrebleu.CHRF(), {}),
    ("chrF b3", lambda tok: sacrebleu.CHRF(beta=3), {}),
    ("chrF++", lambda tok: sacrebleu.CHRF(word_order=2), {}),
    ("TER", lambda tok: sacrebleu.TER(), {}),
    ("TER norm", lambda tok: sacrebleu.TER(normalized=True), {}),
]


def outcome(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLEH -- the exception IS the expectation
        return {"exc": type(e).__name__, "msg": str(e)}


def record_one(metric, hyps, refs, tok):
    """Everything observable about one metric on one corpus."""
    m = metric(tok)
    corpus = outcome(lambda: m.corpus_score(hyps, refs))
    if isinstance(corpus, sacrebleu.metrics.base.Score) or hasattr(corpus, "score"):
        out = {
            "score": corpus.score.hex(),
            "str": str(corpus),
            "format": corpus.format(signature="sig|x:1"),
            "verbose": str(corpus._verbose),
        }
    else:
        out = corpus
    sents = []
    for h, r in zip(hyps, [list(row) for row in zip(*refs)] if refs else []):
        s = outcome(lambda: m.sentence_score(h, r).score)
        sents.append(s if isinstance(s, dict) else s.hex())
    out["sentences"] = sents
    return out


def perturbed(refs, n, seed=1234):
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


def fuzz_cases(n, seed=99):
    rnd = random.Random(seed)
    cases = []
    for _ in range(n):
        segs = []
        for _ in range(rnd.randint(1, 4)):
            segs.append("".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(0, 12))))
        refs = [segs[:]]
        if rnd.random() < 0.3:
            refs.append([s[::-1] for s in segs])
        cases.append((segs, refs))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tests" / "oracle"))
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for pair in PAIRS:
        refs = [l.rstrip("\n") for l in (DATA / f"wmt22.{pair}.ref").read_text(encoding="utf-8").splitlines()]
        refs = refs[: a.limit]
        hyps = perturbed(refs, len(refs))
        tok = TOK.get(pair, "13a")
        f = out / f"wmt22.{pair}.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for name, mk, _ in METRICS:
                fh.write(json.dumps({"case": f"{pair}:{name}", "tokenizer": tok,
                                     "expect": record_one(mk, hyps, [refs], tok)}) + "\n")
        manifest.append(dict(case=f"wmt22.{pair}", file=f.name, n=len(hyps)))
        print("wrote", f, flush=True)

    f = out / "fuzz.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for i, (hyps, refs) in enumerate(fuzz_cases(200)):
            for name, mk, _ in METRICS:
                fh.write(json.dumps({"case": f"fuzz{i}:{name}", "tokenizer": "13a", "hyps": hyps,
                                     "refs": refs, "expect": record_one(mk, hyps, refs, "13a")}) + "\n")
    manifest.append(dict(case="fuzz", file=f.name, n=200))
    print("wrote", f, flush=True)

    model_dir = ROOT / "tests" / "corpus" / "model"
    if model_dir.exists():
        for jf in sorted(model_dir.glob("*.jsonl")):
            recs = sorted((json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines()), key=lambda r: r["i"])
            hyps = [r["hyp"] for r in recs]
            ref_file = Path(str(jf).replace("real-", "ref-").replace(".jsonl", ".txt"))
            if not ref_file.exists():
                continue
            refs = [[l.rstrip("\n") for l in ref_file.read_text(encoding="utf-8").splitlines()]]
            f = out / f"model-{jf.stem}.jsonl"
            with f.open("w", encoding="utf-8") as fh:
                for name, mk, _ in METRICS:
                    fh.write(json.dumps({"case": f"{jf.stem}:{name}", "tokenizer": "13a",
                                         "expect": record_one(mk, hyps, refs, "13a")}) + "\n")
            manifest.append(dict(case=jf.stem, file=f.name, n=len(hyps)))
            print("wrote", f, flush=True)

    (out / "MANIFEST.json").write_text(json.dumps({
        "sacrebleu": sacrebleu.__version__, "python": sys.version.split()[0],
        "cases": manifest}, indent=1) + "\n", encoding="utf-8")
    print("wrote manifest with", len(manifest), "files", flush=True)


if __name__ == "__main__":
    main()