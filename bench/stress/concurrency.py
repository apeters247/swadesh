"""Thread-safety and Python-thread scaling for swadesh (run on the 100k corpus).

A. Independent instances: T threads each build their own BLEU/CHRF/TER and score
   the corpus; every result must equal the serial result (str and score repr).
B. Shared instance: one BLEU / CHRF / TER object used by T threads, each with a
   different reference stream (rotations of the references), so any shared
   mutable state (num_refs, the ref cache) would show up as a wrong score.
C. Shared sentence_scores: T threads on one CHRF object, each with its own slice.

    python bench/stress/concurrency.py [--lines 100000] [--iters 20]
"""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_inputs import WORK  # noqa: E402

import swadesh  # noqa: E402


def read(p, n):
    with open(p, encoding="utf-8", newline="\n") as f:
        return [line.rstrip("\n") for _, line in zip(range(n), f)]


def key(res):
    return (repr(res.score), str(res))


def run_threads(n, fn):
    errors, results = [], [None] * n
    barrier = threading.Barrier(n)

    def worker(i):
        try:
            barrier.wait()
            results[i] = fn(i)
        except BaseException as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t = time.perf_counter()
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    return time.perf_counter() - t, results, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=100_000)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--threads", default="1,2,4,8,16")
    a = ap.parse_args()
    hyp = read(f"{WORK}/c{a.lines}.hyp.txt", a.lines)
    ref = read(f"{WORK}/c{a.lines}.ref.txt", a.lines)
    out = {"lines": a.lines, "cpus": os.cpu_count(), "A_independent": [], "B_shared": [], "C_sentence": []}
    ts = [int(x) for x in a.threads.split(",")]

    serial = {"bleu": key(swadesh.BLEU().corpus_score(hyp, [ref])),
              "chrf": key(swadesh.CHRF().corpus_score(hyp, [ref])),
              "ter": key(swadesh.TER().corpus_score(hyp, [ref]))}
    make = {"bleu": swadesh.BLEU, "chrf": swadesh.CHRF, "ter": swadesh.TER}

    # A: independent instances, every thread does one full corpus_score per iteration.
    for t in ts:
        for metric in ("bleu", "chrf", "ter"):
            def job(i, metric=metric):
                got = []
                for _ in range(a.iters):
                    got.append(key(make[metric]().corpus_score(hyp, [ref])))
                return got
            wall, res, errs = run_threads(t, job)
            bad = sum(1 for r in res if r is None) + \
                sum(1 for r in res if r is not None for k in r if k != serial[metric])
            rec = dict(threads=t, metric=metric, iters=a.iters, wall_s=round(wall, 4),
                       jobs=t * a.iters, jobs_per_s=round(t * a.iters / wall, 2), mismatches=bad, errors=errs[:3])
            out["A_independent"].append(rec)
            print("A", rec, flush=True)

    # B: one shared instance, per-thread reference rotation (expected value computed serially).
    for metric in ("bleu", "chrf", "ter"):
        n_rot = 8
        rots = [ref[i * 7:] + ref[:i * 7] for i in range(n_rot)]
        expect = [key(make[metric]().corpus_score(hyp, [r])) for r in rots]
        shared = make[metric]()

        def job(i, metric=metric):
            bad = 0
            for _ in range(a.iters):
                k = key(shared.corpus_score(hyp, [rots[i % n_rot]]))
                bad += k != expect[i % n_rot]
            return bad
        wall, res, errs = run_threads(8, job)
        rec = dict(threads=8, metric=metric, iters=a.iters, wall_s=round(wall, 4),
                   mismatches=sum(r or 0 for r in res), errors=errs[:3])
        out["B_shared"].append(rec)
        print("B", rec, flush=True)

    # C: shared CHRF instance, sentence_scores on 20k slices.
    n_s = min(20_000, a.lines)
    shared = swadesh.CHRF()
    expect = shared.sentence_scores(hyp[:n_s], [ref[:n_s]])
    for t in ts:
        def job(i):
            return shared.sentence_scores(hyp[:n_s], [ref[:n_s]]) == expect
        wall, res, errs = run_threads(t, job)
        rec = dict(threads=t, metric="chrf sentence_scores", slice=n_s, wall_s=round(wall, 4),
                   mismatches=sum(1 for r in res if r is not True), errors=errs[:3])
        out["C_sentence"].append(rec)
        print("C", rec, flush=True)

    with open(os.path.join(HERE, "results", "concurrency.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
