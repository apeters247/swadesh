"""Does sacreBLEU 2.6.0 itself give different BLEU scores on different CPython versions?

sacreBLEU's ``compute_bleu`` sums ``my_log(p)`` values with the builtin ``sum()``.
CPython changed ``sum()`` over floats in 3.12 (Neumaier compensation) and again in
3.14 (compensated ints too), so the same statistics can produce different scores.

This script does two things:

  score    Record sentence-level and corpus-level BLEU from the *installed*
           sacreBLEU, for each WMT22 reference file, using deterministic
           perturbed hypotheses. Scores are stored as ``float.hex()`` so that
           nothing is rounded.
  compare  Compare two such records: count sentence scores that differ, and
           measure the difference in units in the last place (ulps).

Hypotheses are the reference with deterministic word drops and adjacent swaps.
Randomness uses only ``random.random()``, whose output is stable across
CPython versions, so every interpreter sees the same hypotheses.

Usage:
  python tools/py_version_drift.py score --out FILE
  python tools/py_version_drift.py compare A.json B.json
"""
import argparse
import json
import os
import platform
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get(
    "SWADESH_WMT22_DIR",
    os.path.join(HERE, "..", "bench", "data"),
)
PAIRS = ["cs-en", "de-en", "en-de", "en-ja", "en-ru", "en-zh", "zh-en"]
SEED = 20221122
DROP_P = 0.15
SWAP_P = 0.10


def perturb(line, rnd):
    words = line.split()
    if len(words) <= 1:
        return line
    kept = [w for w in words if rnd.random() >= DROP_P]
    if not kept:
        kept = [words[0]]
    i = 0
    while i < len(kept) - 1:
        if rnd.random() < SWAP_P:
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
            i += 2
        else:
            i += 1
    return " ".join(kept)


def load_pair(pair):
    path = os.path.join(DATA, f"wmt22.{pair}.ref")
    with open(path, encoding="utf-8") as f:
        refs = [line.rstrip("\n") for line in f]
    rnd = random.Random(SEED)
    hyps = [perturb(r, rnd) for r in refs]
    return hyps, refs


def cmd_score(args):
    import sacrebleu
    from sacrebleu.metrics import BLEU

    out = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "sacrebleu": sacrebleu.__version__,
        "seed": SEED,
        "drop_p": DROP_P,
        "swap_p": SWAP_P,
        "pairs": {},
    }
    for pair in PAIRS:
        hyps, refs = load_pair(pair)
        m = BLEU()
        sent = [m.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
        corpus = BLEU().corpus_score(hyps, [refs]).score
        out["pairs"][pair] = {
            "n": len(hyps),
            "sentence": [float(s).hex() for s in sent],
            "corpus": float(corpus).hex(),
        }
    with open(args.out, "w") as f:
        json.dump(out, f)
    total = sum(p["n"] for p in out["pairs"].values())
    print(f"wrote {args.out}: {len(PAIRS)} pairs, {total} segments, "
          f"python {out['python']}, sacrebleu {out['sacrebleu']}")
    return 0


def ulp_key(x):
    """Map a float to an integer so that adjacent floats differ by 1."""
    bits = struct.unpack("<q", struct.pack("<d", x))[0]
    return bits if bits >= 0 else -(bits & 0x7FFFFFFFFFFFFFFF)


def cmd_compare(args):
    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)
    print(f"A: python {a['python']} sacrebleu {a['sacrebleu']}")
    print(f"B: python {b['python']} sacrebleu {b['sacrebleu']}")
    tot_n = tot_diff = 0
    max_ulp = 0
    corpus_diff = 0
    report = {}
    for pair in PAIRS:
        pa, pb = a["pairs"][pair], b["pairs"][pair]
        diffs = []
        for i, (x, y) in enumerate(zip(pa["sentence"], pb["sentence"])):
            if x != y:
                fx, fy = float.fromhex(x), float.fromhex(y)
                diffs.append((abs(ulp_key(fx) - ulp_key(fy)), fx - fy, i))
        ulps = [d[0] for d in diffs]
        pair_max = max(ulps, default=0)
        max_ulp = max(max_ulp, pair_max)
        tot_n += pa["n"]
        tot_diff += len(diffs)
        cd = pa["corpus"] != pb["corpus"]
        corpus_diff += cd
        report[pair] = {
            "segments": pa["n"],
            "sentence_differ": len(diffs),
            "max_ulp": pair_max,
            "corpus_differ": cd,
            "corpus_A": float.fromhex(pa["corpus"]),
            "corpus_B": float.fromhex(pb["corpus"]),
        }
        print(f"{pair}: segments={pa['n']} sentence_differ={len(diffs)} "
              f"max_ulp={pair_max} corpus_differ={cd} "
              f"corpus A={float.fromhex(pa['corpus']):.17g} B={float.fromhex(pb['corpus']):.17g}")
    print(f"TOTAL: segments={tot_n} sentence_differ={tot_diff} "
          f"({100.0 * tot_diff / max(tot_n, 1):.3f}%) max_ulp={max_ulp} "
          f"corpus_pairs_differ={corpus_diff}/{len(PAIRS)}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"A": a["python"], "B": b["python"], "total_segments": tot_n,
                       "total_sentence_differ": tot_diff, "max_ulp": max_ulp,
                       "corpus_pairs_differ": corpus_diff, "pairs": report}, f, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
