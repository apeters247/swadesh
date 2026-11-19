"""Fixtures for the Rust tests of the NumPy-2 float32 bootstrap scorers
(`compute_bleu_np2_f32`, `chrf_f_score_np2_f32`): sacreBLEU's own
`_compute_score_from_stats` applied to random float32 rows. Expected values are
stored as exact f64 bit patterns plus whether sacreBLEU returned a np.float32.

    python tools/gen_np2_fixtures.py   # -> crates/swadesh-core/tests/fixtures/np2_*.txt
"""
import random
import struct
from pathlib import Path

import numpy as np
import sacrebleu

assert int(np.__version__.split(".")[0]) >= 2, "fixtures encode NumPy >= 2 semantics"
OUT = Path(__file__).resolve().parents[1] / "crates/swadesh-core/tests/fixtures"
rnd = random.Random(2026)


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def f32bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def big():
    return rnd.choice([3, 20, 1000, 100_000, 2**24 + rnd.randint(1, 99), 2**30])


lines = []
for _ in range(4000):
    order = rnd.randint(1, 6)
    smooth = rnd.choice(["exp", "floor", "add-k", "none"])
    sv = rnd.choice([None, 0.1, 0.5, 1, 2, 0.01, 3.7])
    eff = rnd.random() < 0.5
    hi = big()
    total = [rnd.choice([0, rnd.randint(0, hi)]) for _ in range(order)]
    correct = [rnd.randint(0, t) if rnd.random() < 0.85 else 0 for t in total]
    row = np.array([rnd.randint(0, hi), rnd.randint(0, hi)] + correct + total, dtype=np.float32)
    m = sacrebleu.BLEU(smooth_method=smooth, smooth_value=sv, max_ngram_order=order, effective_order=eff)
    score = m._compute_score_from_stats(row.copy()).score
    sv_s = "none" if sv is None else repr(float(sv))
    lines.append(f"{order} {smooth} {sv_s} {int(eff)} {bits(score)} " + " ".join(str(f32bits(v)) for v in row))
(OUT / "np2_bleu.txt").write_text("\n".join(lines) + "\n")

lines = []
for _ in range(4000):
    co, wo = rnd.randint(0, 7), rnd.randint(0, 3)
    if co + wo == 0:
        co = 1
    beta = rnd.randint(0, 4)
    eps = rnd.random() < 0.5
    hi = big()
    row = []
    for _ in range(co + wo):
        h, r = rnd.choice([0, rnd.randint(0, hi)]), rnd.choice([0, rnd.randint(0, hi)])
        row += [h, r, rnd.randint(0, min(h, r))]
    row = np.array(row, dtype=np.float32)
    m = sacrebleu.CHRF(char_order=co, word_order=wo, beta=beta, eps_smoothing=eps)
    s = m._compute_score_from_stats(row.copy()).score
    is32 = isinstance(s, np.float32)
    lines.append(f"{co + wo} {beta} {int(eps)} {int(is32)} {bits(s)} " + " ".join(str(f32bits(v)) for v in row))
(OUT / "np2_chrf.txt").write_text("\n".join(lines) + "\n")
print("wrote", OUT / "np2_bleu.txt", OUT / "np2_chrf.txt")
