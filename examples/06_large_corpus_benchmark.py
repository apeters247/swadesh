"""Time swadesh against sacrebleu on a large corpus, and check the results match.

The corpus is built from WMT22 en-de references (bench/data, downloaded by
bench/get_data.sh) with random word-level perturbations, or from synthetic
sentences when that data is not present. Timings depend on the machine and on
the number of threads (RAYON_NUM_THREADS). This script prints what it measures
and does not claim a speedup for any machine.

Run: python examples/06_large_corpus_benchmark.py --lines 20000
     RAYON_NUM_THREADS=1 python examples/06_large_corpus_benchmark.py --lines 20000
"""
import argparse
import os
import random
import time
from pathlib import Path

import sacrebleu

import swadesh

# Override the location with SWADESH_DATA=/path/to/dir containing wmt22.en-de.ref
DATA = Path(os.environ.get("SWADESH_DATA", Path(__file__).resolve().parent.parent / "bench" / "data")) / "wmt22.en-de.ref"


def build_corpus(n: int, seed: int = 1234):
    rnd = random.Random(seed)
    if DATA.exists():
        refs = [line.rstrip("\n") for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
        source = f"WMT22 en-de references ({DATA.name})"
    else:
        vocab = "the a cat dog sat on mat ran fast slow red blue river train market bread".split()
        refs = [" ".join(rnd.choices(vocab, k=rnd.randint(6, 25))) for _ in range(2000)]
        source = "synthetic sentences (bench/data not found)"
    hyps, out_refs = [], []
    while len(hyps) < n:
        for r in refs:
            w = r.split()
            for _ in range(max(1, len(w) // 6)):
                if not w:
                    break
                op, i = rnd.random(), rnd.randrange(len(w))
                if op < 0.33:
                    del w[i]
                elif op < 0.66 and len(w) > 1:
                    j = rnd.randrange(len(w))
                    w[i], w[j] = w[j], w[i]
                else:
                    w[i] = w[i][::-1]
            hyps.append(" ".join(w))
            out_refs.append(r)
            if len(hyps) == n:
                break
    return source, hyps, [out_refs]


def best_of(fn, repeat):
    best, result = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=20000)
    ap.add_argument("--repeat", type=int, default=3, help="best of N runs")
    a = ap.parse_args()

    source, hyps, refs = build_corpus(a.lines)
    threads = os.environ.get("RAYON_NUM_THREADS", "default")
    print(f"{a.lines} segments from {source}; RAYON_NUM_THREADS={threads}; cpus={os.cpu_count()}")
    print(f"sacrebleu {sacrebleu.__version__}, swadesh {swadesh.__version__}")
    print(f"{'case':<16}{'sacrebleu s':>13}{'swadesh s':>12}{'speedup':>9}  identical")

    cases = [
        ("BLEU corpus", lambda m: m.BLEU().corpus_score(hyps, refs)),
        ("chrF corpus", lambda m: m.CHRF().corpus_score(hyps, refs)),
        ("chrF++ corpus", lambda m: m.CHRF(word_order=2).corpus_score(hyps, refs)),
    ]
    for name, fn in cases:
        t_ref, r_ref = best_of(lambda: fn(sacrebleu), a.repeat)
        t_sw, r_sw = best_of(lambda: fn(swadesh), a.repeat)
        same = str(r_ref) == str(r_sw)
        print(f"{name:<16}{t_ref:>13.3f}{t_sw:>12.4f}{t_ref / t_sw:>8.1f}x  {'yes' if same else 'NO'}")
        if not same:
            raise SystemExit(f"{name}: swadesh and sacrebleu differ:\n  {r_sw}\n  {r_ref}")


if __name__ == "__main__":
    main()
