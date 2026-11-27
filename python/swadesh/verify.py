"""Check swadesh against sacreBLEU on *your* data.

    python -m swadesh.verify -r refs.txt [-r refs2.txt] -i hyps.txt [--tokenize 13a] [--lowercase]

Runs both implementations (corpus BLEU / chrF / chrF++, every sentence score,
statistics and signatures) and reports any difference. Exit code 0 means every
value was identical, 1 means a mismatch was found (please report it!), 2 means
sacreBLEU is not installed, which this comparison cannot do without.
"""

from __future__ import annotations

import argparse
import sys
import time

import swadesh

try:
    import sacrebleu
except ModuleNotFoundError:
    # This command is the README's call to action, so a missing oracle has to read as an
    # instruction rather than a traceback.
    print(
        "swadesh.verify compares swadesh against sacreBLEU, which is not installed here.\n"
        'Install it first:  pip install "swadesh[sacrebleu]"',
        file=sys.stderr,
    )
    # 2, not 1: exit 1 already means "a mismatch was found", which this is not.
    sys.exit(2)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def verify(hyps, refs, tokenize=None, lowercase=False, verbose=True) -> bool:
    ok = True
    checks = [
        ("BLEU", dict(tokenize=tokenize, lowercase=lowercase), swadesh.BLEU, sacrebleu.BLEU),
        ("chrF", dict(lowercase=lowercase), swadesh.CHRF, sacrebleu.CHRF),
        ("chrF++", dict(lowercase=lowercase, word_order=2), swadesh.CHRF, sacrebleu.CHRF),
    ]
    for name, kw, Ours, Theirs in checks:
        ours, theirs = Ours(**kw), Theirs(**kw)
        t0 = time.perf_counter()
        a = ours.corpus_score(hyps, refs)
        t1 = time.perf_counter()
        b = theirs.corpus_score(hyps, refs)
        t2 = time.perf_counter()
        same_corpus = (a.score == b.score and str(a) == str(b)
                       and str(ours.get_signature()) == str(theirs.get_signature()))
        sent_kw = dict(kw)
        if name == "BLEU":
            sent_kw["effective_order"] = True
        so, st = Ours(**sent_kw), Theirs(**sent_kw)
        ss = so.sentence_scores(hyps, refs)
        mismatches = 0
        first = None
        for i, h in enumerate(hyps):
            v = st.sentence_score(h, [r[i] for r in refs]).score
            if v != ss[i]:
                mismatches += 1
                first = first if first is not None else (i, ss[i], v)
        good = same_corpus and mismatches == 0
        ok &= good
        if verbose:
            speed = (t2 - t1) / max(t1 - t0, 1e-9)
            print(f"{name:7s} corpus {'identical' if same_corpus else 'DIFFERENT'}  "
                  f"sentences {len(hyps) - mismatches}/{len(hyps)} identical  "
                  f"({speed:.0f}x faster corpus scoring)")
            if not same_corpus:
                print(f"    swadesh:   {a}\n    sacrebleu: {b}")
            if first:
                print(f"    first sentence mismatch at line {first[0] + 1}: {first[1]!r} vs {first[2]!r}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m swadesh.verify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--ref", action="append", required=True, help="reference file (repeatable)")
    ap.add_argument("-i", "--input", required=True, help="hypothesis file")
    ap.add_argument("-tok", "--tokenize", default=None)
    ap.add_argument("-lc", "--lowercase", action="store_true")
    a = ap.parse_args(argv)
    hyps = _read(a.input)
    refs = [_read(r) for r in a.ref]
    print(f"swadesh {swadesh.__version__} vs sacrebleu {sacrebleu.__version__}: "
          f"{len(hyps)} segments, {len(refs)} reference stream(s)")
    ok = verify(hyps, refs, a.tokenize, a.lowercase)
    print("ALL IDENTICAL" if ok else "MISMATCH FOUND - please report at the swadesh issue tracker")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
