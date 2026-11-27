"""Time swadesh against sacreBLEU on *your* files, and check every number is identical.

    python bench/bench_own.py -r ref.txt -i hyp.txt
    python bench/bench_own.py -r ref.txt -r ref2.txt -i hyp.txt --threads 8 --reps 3

sacreBLEU is timed first, on cold inputs, because its tokenizers are lru_cached per string: timing it
second would flatter it. swadesh has no such cache, so its order does not matter.
"""
import argparse
import time

import sacrebleu
import swadesh


def read(path):
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def timed(fn, reps):
    best, value = None, None
    for _ in range(reps):
        start = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - start
        best = dt if best is None else min(best, dt)
    return value, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-r", "--references", action="append", required=True,
                    help="a reference file; repeat for multiple references")
    ap.add_argument("-i", "--input", required=True, help="the hypothesis file")
    ap.add_argument("--threads", type=int, default=0,
                    help="worker threads for swadesh (default: the machine)")
    ap.add_argument("--reps", type=int, default=1, help="best of N timings per implementation")
    ap.add_argument("--tokenize", default=None, help="BLEU tokenizer (default: sacreBLEU's choice)")
    ap.add_argument("--lowercase", action="store_true")
    cfg = ap.parse_args()

    hyps = read(cfg.input)
    streams = [read(p) for p in cfg.references]
    n = min([len(hyps)] + [len(s) for s in streams])
    if n != len(hyps) or any(len(s) != n for s in streams):
        print("note: truncating to %d lines, as sacreBLEU does" % n)

    kwargs = {}
    if cfg.tokenize:
        kwargs["tokenize"] = cfg.tokenize
    if cfg.lowercase:
        kwargs["lowercase"] = True

    swadesh.set_n_jobs(cfg.threads or None)
    metrics = [
        ("BLEU", lambda: sacrebleu.BLEU(**kwargs).corpus_score(hyps, streams),
         lambda: swadesh.BLEU(**kwargs).corpus_score(hyps, streams)),
        ("chrF", lambda: sacrebleu.CHRF(**kwargs).corpus_score(hyps, streams),
         lambda: swadesh.CHRF(**kwargs).corpus_score(hyps, streams)),
        ("chrF++", lambda: sacrebleu.CHRF(word_order=2, **kwargs).corpus_score(hyps, streams),
         lambda: swadesh.CHRF(word_order=2, **kwargs).corpus_score(hyps, streams)),
        ("TER", lambda: sacrebleu.TER().corpus_score(hyps, streams),
         lambda: swadesh.TER().corpus_score(hyps, streams)),
    ]

    print("%d segments, %d reference(s), %d threads" % (n, len(streams),
                                                        swadesh.get_n_jobs()))
    print("%-7s %12s %12s %9s  %s" % ("metric", "sacrebleu", "swadesh", "speedup", "identical"))
    failures = 0
    for name, theirs_fn, ours_fn in metrics:
        theirs, secs_theirs = timed(theirs_fn, cfg.reps)
        ours, secs_ours = timed(ours_fn, cfg.reps)
        same = str(theirs) == str(ours)
        failures += 0 if same else 1
        print("%-7s %11.4fs %11.4fs %8.1fx  %s"
              % (name, secs_theirs, secs_ours, secs_theirs / max(secs_ours, 1e-9),
                 "yes" if same else "NO\n  theirs: %s\n  ours:   %s" % (theirs, ours)))
    print("\n%d of %d metrics differ" % (failures, len(metrics)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())