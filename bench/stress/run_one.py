"""Time one corpus-level metric call in a fresh process (used by run_all.py).

Reads hypothesis and reference files (one segment per line, '\\n' only), times
``corpus_score`` (best of --repeat), and prints one JSON line with the score,
its full ``str()``, peak RSS (ru_maxrss, KiB) and load averages. Exceptions are
reported as JSON ``error`` fields; the driver enforces timeouts externally.

    python bench/stress/run_one.py --impl swadesh --metric bleu --hyp h.txt --ref r.txt
"""
import argparse
import json
import os
import resource
import time


def maxrss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def read(path):
    with open(path, encoding="utf-8", newline="\n") as f:
        return [line.rstrip("\n") for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["swadesh", "sacrebleu"], required=True)
    ap.add_argument("--metric", choices=["bleu", "chrf", "chrf++", "ter", "bleu-bs"], required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tok", default="13a")
    a = ap.parse_args()

    t = time.perf_counter()
    hyp = read(a.hyp)
    refs = [read(r) for r in a.ref]
    load_s = time.perf_counter() - t
    rss_load = maxrss_kb()

    if a.impl == "sacrebleu":
        import sacrebleu as mod
    else:
        import swadesh as mod

    def build():
        if a.metric == "chrf":
            return mod.CHRF()
        if a.metric == "chrf++":
            return mod.CHRF(word_order=2)
        if a.metric == "ter":
            return mod.TER()
        return mod.BLEU(tokenize=a.tok)

    out = dict(impl=a.impl, metric=a.metric, n=len(hyp), nrefs=len(refs), load_s=round(load_s, 4),
               rss_after_load_kb=rss_load, threads=os.environ.get("RAYON_NUM_THREADS", "default"))
    la0 = os.getloadavg()
    # PyO3 panics derive from BaseException, so catch that too.
    try:
        best, res = float("inf"), None
        for _ in range(a.repeat):
            m = build()
            t = time.perf_counter()
            if a.metric == "bleu-bs":
                res = m.corpus_score(hyp, refs, n_bootstrap=1000)
            else:
                res = m.corpus_score(hyp, refs)
            best = min(best, time.perf_counter() - t)
        out.update(time_s=best, score=repr(res.score), str=str(res), rss_peak_kb=maxrss_kb())
    except BaseException as e:  # noqa: BLE001
        out.update(time_s=None, error=f"{type(e).__name__}: {str(e)[:300]}", rss_peak_kb=maxrss_kb())
    out["loadavg"] = [list(la0), list(os.getloadavg())]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
