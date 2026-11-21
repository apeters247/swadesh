"""Extra speed tests for the blog post: the workloads other people actually run, timed both ways.

Every case compares the *result* with == (scores), == (formatted strings) or exact list equality,
and reports it. Run on a quiet machine.

    python extra_speed.py [--json out.json]

Cases:
  sentence_scores (batch)      per-sample reward vectors, the RL-training shape
  sentence loop (naive)        the same as a naive loop of sentence_score calls, which is what code
                               in the wild does; swadesh has no shortcut for this, so it shows the
                               cost people pay without batch mode
  MBR pairwise                 chrF matrix over N candidates per input (n^2 sentence scores)
  significance (bootstrap CI)  corpus score with 1000 bootstrap resamples
  PairedTest bs / ar           two systems, many resamples, one core
  CLI process wall time        the sacrebleu command vs python -m swadesh.cli, file in, stdout out
"""
from __future__ import annotations

import json
import tempfile
import os
import subprocess
import sys
import time
from pathlib import Path

import sacrebleu
import sacrebleu.significance as sb_sig
import swadesh


REF = Path(os.environ.get("SWADESH_REF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wmt22.en-de.ref")))
refs_all = [l.rstrip("\n") for l in REF.read_text(encoding="utf-8").splitlines()]


def perturbed(n):
    import random

    rnd = random.Random(1234)
    out = []
    while len(out) < n:
        for r in refs_all:
            w = r.split()
            for _ in range(max(1, len(w) // 6)):
                if not w:
                    break
                op = rnd.random()
                k = rnd.randrange(len(w))
                if op < 0.4:
                    del w[k]
                elif op < 0.7:
                    w.insert(k, rnd.choice(["sehr", "ganz", "und"]))
            out.append(" ".join(w))
            if len(out) >= n:
                break
    return out


def best(fn, repeat=3):
    best_t, res = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        res = fn()
        best_t = min(best_t, time.perf_counter() - t)
    return best_t, res


rows = []


def record(name, t_sb, t_sw, exact, note=""):
    rows.append(dict(case=name, sacrebleu_s=t_sb, swadesh_s=t_sw, speedup=t_sb / t_sw if t_sw else None,
                     exact=exact, note=note))
    print(f"{name:34s} {t_sb:9.3f}s {t_sw:9.4f}s {t_sb / max(t_sw, 1e-9):7.1f}x  {'yes' if exact else 'NO!'}  {note}", flush=True)


# 1. sentence_scores batch, 20k
n = 20000
H, R = perturbed(n), [refs_all[:n]]
sb = sacrebleu.BLEU(effective_order=True)
sw = swadesh.BLEU(effective_order=True)
t_sb, a = best(lambda: [sb.sentence_score(h, [r]).score for h, r in zip(H, R[0])], 1)
t_sw, b = best(lambda: sw.sentence_scores(H, R), 1)
record(f"BLEU sentence_scores batch x{n}", t_sb, t_sw, a == b)

t_sb, a = best(lambda: [sacrebleu.CHRF().sentence_score(h, [r]).score for h, r in zip(H, R[0])], 1)
t_sw, b = best(lambda: swadesh.CHRF().sentence_scores(H, R), 1)
record(f"chrF sentence_scores batch x{n}", t_sb, t_sw, a == b)

# 2. naive sentence_score loop (no batch API), 2000 segments: what existing code pays
n2 = 2000
H2, R2 = H[:n2], [R[0][:n2]]
sb2, sw2 = sacrebleu.BLEU(effective_order=True), swadesh.BLEU(effective_order=True)
t_sb, a = best(lambda: [sb2.sentence_score(h, [r]).score for h, r in zip(H2, R2[0])])
t_sw, b = best(lambda: [sw2.sentence_score(h, [r]).score for h, r in zip(H2, R2[0])])
record(f"BLEU sentence_score loop x{n2}", t_sb, t_sw, a == b, "per-call Python overhead on both sides")

# 3. MBR pairwise: 20 inputs x 10 candidates, chrF++ matrix
import random

rnd = random.Random(7)
n_in, cand = 20, 10
base = H[:n_in]
cands = [base]
for _ in range(cand - 1):
    cands.append([" ".join(w[: max(1, len(w) - rnd.randrange(0, 3))]) for w in base])
flat = [c for col in zip(*cands) for c in col]


def pairwise_sb(group):
    m = sacrebleu.CHRF(word_order=2)
    return [[m.sentence_score(a, [b]).score for b in group] for a in group]


def pairwise_sw(group):
    import numpy as np

    return np.asarray(swadesh.CHRF(word_order=2).pairwise(group)).tolist()


def all_pairs(fn):
    out = []
    for group in zip(*cands):
        out.append(fn(list(group)))
    return out


try:
    t_sb, a = best(lambda: all_pairs(pairwise_sb), 1)
    t_sw, b = best(lambda: all_pairs(pairwise_sw), 1)
    record(f"MBR chrF++ pairwise {cand}x{cand}", t_sb, t_sw, a == b)
except Exception as e:  # noqa: BLE001
    print("pairwise skipped:", type(e).__name__, e, flush=True)

# 4. bootstrap CIs, 5000 segments, 1000 resamples
n4 = 5000
H4, R4 = H[:n4], [R[0][:n4]]
sb4, sw4 = sacrebleu.BLEU(), swadesh.BLEU()
t_sb, a = best(lambda: sb4.corpus_score(H4, R4, n_bootstrap=1000), 1)
t_sw, b = best(lambda: sw4.corpus_score(H4, R4, n_bootstrap=1000), 1)
record(f"BLEU + bootstrap(1000) x{n4}", t_sb, t_sw, a.score == b.score and a._mean == b._mean and a._ci == b._ci)

# 5. PairedTest, two systems
n5 = 2000
sysA = H[:n5]
sysB = [" ".join(w[:-1]) if i % 2 else h for i, h, w in ((i, h, h.split()) for i, h in enumerate(sysA))]
pairs = [("baseline", sysA), ("candidate", sysB)]
refs5 = [R[0][:n5]]
for kind in ("bs", "ar"):
    metrics = {"BLEU": sacrebleu.BLEU(references=refs5), "chrF": sacrebleu.CHRF(references=refs5)}
    t_sb, a = best(lambda: sb_sig.PairedTest(pairs, {k: v for k, v in metrics.items()}, None,
                                             test_type=kind, n_samples=500, n_jobs=1)(), 1)
    t_sw, b = best(lambda: swadesh.PairedTest(pairs, {"BLEU": sacrebleu.BLEU(references=refs5),
                                                      "chrF": sacrebleu.CHRF(references=refs5)}, None,
                                              test_type=kind, n_samples=500, n_jobs=1)(), 1)
    fa = {k: [(str(x), getattr(x, "p_value", None)) for x in v] for k, v in a[1].items()}
    fb = {k: [(str(x), getattr(x, "p_value", None)) for x in v] for k, v in b[1].items()}
    record(f"PairedTest {kind} x{n5}, 2 metrics", t_sb, t_sw, fa == fb)

# 6. CLI wall time, 20k lines
n6 = 20000
hyp = Path(tempfile.mkdtemp()) / "hyp.txt"
hyp.write_text("\n".join(H[:n6]) + "\n", encoding="utf-8")
ref = hyp.with_name("ref.txt")
ref.write_text("\n".join(R[0][:n6]) + "\n", encoding="utf-8")


def cli(cmd):
    def run():
        p = subprocess.run(cmd + [str(ref), "-i", str(hyp)], capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    return run


t_sb, a = best(cli([str(Path(sys.executable).with_name("sacrebleu"))]), 3)
t_sw, b = best(cli([sys.executable, "-m", "swadesh.cli"]), 3)
record(f"CLI BLEU corpus x{n6}", t_sb, t_sw, a == b, "process start + read + score + print")
t_sb, a = best(cli([str(Path(sys.executable).with_name("sacrebleu")), "-m", "chrf", "-cw", "2"]), 1)
t_sw, b = best(cli([sys.executable, "-m", "swadesh.cli", "-m", "chrf", "-cw", "2"]), 1)
record(f"CLI chrF++ corpus x{n6}", t_sb, t_sw, a == b)

out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else ""
if out:
    json.dump(dict(rows=rows), open(out, "w"), indent=1)
    print("wrote", out)
print("MISMATCHES", sum(0 if r["exact"] else 1 for r in rows))