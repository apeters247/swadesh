"""Benchmark swadesh vs sacreBLEU on real WMT22 references.

Hypotheses are realistic perturbations of the references (word drops, swaps,
substitutions), deterministic via a fixed seed. Every timed swadesh result is
also checked for exact equality with sacreBLEU.

    python bench/bench.py --lines 100000 --json bench/results.json
"""
import argparse
import json
import os
import platform
import random
import time

import sacrebleu

import swadesh

HERE = os.path.dirname(__file__)


def load(langpair, n):
    path = os.path.join(HERE, "data", f"wmt22.{langpair}.ref")
    refs = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    rnd = random.Random(1234)
    out_r, out_h = [], []
    while len(out_r) < n:
        for r in refs:
            w = r.split()
            for _ in range(max(1, len(w) // 6)):
                if not w:
                    break
                op = rnd.random()
                i = rnd.randrange(len(w))
                if op < 0.33:
                    del w[i]
                elif op < 0.66 and len(w) > 1:
                    j = rnd.randrange(len(w))
                    w[i], w[j] = w[j], w[i]
                else:
                    w[i] = w[i][::-1]
            out_r.append(r)
            out_h.append(" ".join(w))
            if len(out_r) == n:
                break
    return out_h, [out_r]


def timeit(fn, repeat=3):
    best, res = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t)
    return best, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=20000)
    ap.add_argument("--langpair", default="en-de")
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="skip slow sacreBLEU sentence/bootstrap cases")
    a = ap.parse_args()
    hyps, refs = load(a.langpair, a.lines)
    tok = {"en-zh": "zh", "en-ja": "char"}.get(a.langpair, "13a")

    cases = [
        ("BLEU corpus", lambda m: m.BLEU(tokenize=tok).corpus_score(hyps, refs).score, 3),
        ("chrF corpus", lambda m: m.CHRF().corpus_score(hyps, refs).score, 3),
        ("chrF++ corpus", lambda m: m.CHRF(word_order=2).corpus_score(hyps, refs).score, 3),
    ]
    nt = min(a.lines, 10000)
    th, tr = hyps[:nt], [refs[0][:nt]]
    cases.append((f"TER corpus x{nt}", lambda m: m.TER().corpus_score(th, tr).score, 1))
    n_sent = min(a.lines, 20000)
    sh, sr = hyps[:n_sent], [refs[0][:n_sent]]

    def sent_sacre(metric):
        return [metric.sentence_score(h, [r]).score for h, r in zip(sh, sr[0])]

    cases += [
        (f"sentence BLEU x{n_sent}",
         lambda m: m.BLEU(tokenize=tok, effective_order=True).sentence_scores(sh, sr) if m is swadesh
         else sent_sacre(m.BLEU(tokenize=tok, effective_order=True)), 1),
        (f"sentence chrF x{n_sent}",
         lambda m: m.CHRF().sentence_scores(sh, sr) if m is swadesh else sent_sacre(m.CHRF()), 1),
    ]
    if not a.quick:
        nb = min(a.lines, 5000)
        bh, br = hyps[:nb], [refs[0][:nb]]
        cases.append((f"BLEU + bootstrap(1000) x{nb}",
                      lambda m: str(m.BLEU().corpus_score(bh, br, n_bootstrap=1000)), 1))

    rows = []
    print(f"{a.lines} lines of WMT22 {a.langpair}, tokenizer={tok}, {os.cpu_count()} CPUs, {platform.processor() or platform.machine()}")
    print(f"{'case':34s} {'sacrebleu':>10s} {'swadesh':>10s} {'speedup':>8s}  exact")
    for name, fn, rep in cases:
        t_s, r_s = timeit(lambda: fn(sacrebleu), rep)
        t_w, r_w = timeit(lambda: fn(swadesh), rep)
        exact = r_s == r_w
        rows.append(dict(case=name, sacrebleu_s=t_s, swadesh_s=t_w, speedup=t_s / t_w, exact=exact))
        print(f"{name:34s} {t_s:9.3f}s {t_w:9.4f}s {t_s / t_w:7.1f}x  {'yes' if exact else 'NO!'}")
    if a.json:
        json.dump(dict(lines=a.lines, langpair=a.langpair, cpus=os.cpu_count(),
                       machine=platform.machine(), rows=rows), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
