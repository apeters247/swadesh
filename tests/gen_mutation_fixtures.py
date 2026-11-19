#!/usr/bin/env python3
"""Generate sacreBLEU / numpy / CPython oracle fixtures for the mutation-kill tests.

Run with the venv that has sacrebleu 2.6.0 and numpy installed:

    .venv/bin/python tests/gen_mutation_fixtures.py [--ter-cases N]

Writes plain-text fixtures to crates/swadesh-core/tests/fixtures/mut_*.txt, which
the `#[cfg(test)]` modules in swadesh-core read with `include_str!`. Every expected
value is produced by the reference implementation (sacreBLEU, numpy, CPython
`statistics`), never by swadesh, and floats are stored as exact IEEE bit patterns
so the Rust tests compare with `==` on bits.

Files:
  mut_ter.txt          TER segment stats: edits, avg ref len (sacreBLEU TER)
  mut_bleu_score.txt   BLEU compute_bleu(stats) score bits (sacreBLEU _compute_score_from_stats)
  mut_chrf_score.txt   chrF _compute_f_score bits
  mut_chrf_seg.txt     chrF segment statistics (best-ref, first strict best)
  mut_chrf_pairs.txt   chrF sentence scores for pairwise matrices
  mut_punct.txt        CHRF._remove_punctuation
  mut_int_edge.txt     numpy integers(0, n) draws for Lemire edge cases (rng near 2**63)
  mut_next_below.txt   numpy draws for next_below(n) (uint64 dtype)
  mut_stat_mean.txt    CPython statistics.mean over wide-exponent floats
"""

import math
import os
import random
import statistics
import struct
import sys

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "crates", "swadesh-core", "tests", "fixtures")

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
MASK128 = (1 << 128) - 1
PCG_MULT = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645


def f64_bits(x: float) -> str:
    return format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "016x")


def write(name, lines):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"wrote {name}: {len(lines)} lines")


# --------------------------------------------------------------- TER -------

WORDS = ["a", "b", "c", "d", "e", "f", "the", "cat"]


def move_block(words, start, length, target):
    """Move words[start:start+length] so it begins at `target` (of the remainder)."""
    block = words[start:start + length]
    rest = words[:start] + words[start + length:]
    target = max(0, min(target, len(rest)))
    return rest[:target] + block + rest[target:]


def ter_pair(r: random.Random):
    """Return (hyp_words, [ref_words...]) for one TER case."""
    kind = r.choice(["perm", "repeat", "far", "short_hyp", "long_ref", "random", "perm",
                     "extreme", "ties", "extreme", "ties", "beam", "beam", "edge", "edge", "edge"])
    if kind == "perm":
        ref = [r.choice(WORDS[:r.randint(2, 8)]) for _ in range(r.randint(1, 35))]
        hyp = ref[:]
        for _ in range(r.randint(1, 4)):
            if len(hyp) < 2:
                break
            s = r.randrange(len(hyp))
            hyp = move_block(hyp, s, r.randint(1, min(10, len(hyp) - s)), r.randint(0, len(hyp)))
        if r.random() < 0.5:  # a substitution or deletion
            if hyp:
                hyp[r.randrange(len(hyp))] = r.choice(WORDS)
        return hyp, [ref]
    if kind == "repeat":
        V = ["a", "b"][: r.randint(1, 2)]
        ref = [r.choice(V) for _ in range(r.randint(5, 45))]
        hyp = [r.choice(V) for _ in range(r.randint(5, 45))]
        return hyp, [ref]
    if kind == "far":
        # Block moved more than MAX_SHIFT_DIST (50) positions; exercises the distance cutoff.
        n = r.randint(55, 85)
        ref = [WORDS[i % len(WORDS)] if r.random() < 0.7 else r.choice(WORDS) for i in range(n)]
        s = r.randint(0, 10)
        length = r.randint(2, 11)
        hyp = move_block(ref, s, length, min(n, s + r.randint(50, 75)))
        if r.random() < 0.5:
            hyp = move_block(hyp, s, length, max(0, s + r.randint(-75, -50)))
        return hyp, [ref]
    if kind == "extreme":
        # Length ratio above 50 (beam width grows with the ratio), hypothesis made of reference words.
        ref = [r.choice(WORDS[:4]) for _ in range(r.randint(52, 120))]
        k = r.randint(1, 4)
        hyp = [r.choice(ref) for _ in range(k)]
        if r.random() < 0.5:
            hyp = [r.choice(ref[:]) for _ in range(k)]
        return hyp, [ref]
    if kind == "edge":
        # Long filler reference with distinctive words only near its ends, and a short hypothesis:
        # the beam band must exclude the matching cells (beam width formula for ratio > 50).
        m = r.randint(52, 160)
        ref = ["x"] * m
        words = ["a", "b", "c"]
        for w in words[: r.randint(2, 3)]:
            p = r.randint(0, 5) if r.random() < 0.5 else m - 1 - r.randint(0, 5)
            ref[p] = w
        hyp = [r.choice(words) for _ in range(r.randint(1, 3))]
        return hyp, [ref]
    if kind == "beam":
        # Length ratio m / n_hyp well above 50: the beam width formula is in use.
        V = ["a", "b", "c"][: r.randint(2, 3)]
        ref = [r.choice(V) for _ in range(r.randint(52, 130))]
        hyp = [r.choice(V) for _ in range(r.randint(1, 3))]
        if r.random() < 0.5:
            hyp = [r.choice(ref) for _ in range(r.randint(1, 3))]
        return hyp, [ref]
    if kind == "ties":
        # Two-word vocabulary and permuted content: many shift candidates with equal scores.
        V = ["a", "b"]
        ref = [r.choice(V) for _ in range(r.randint(8, 22))]
        hyp = ref[:]
        r.shuffle(hyp)
        if r.random() < 0.5:
            hyp = hyp[: r.randint(3, len(hyp))]
        return hyp, [ref]
    if kind == "short_hyp":
        ref = [r.choice(WORDS) for _ in range(r.randint(30, 80))]
        hyp = [r.choice(WORDS) for _ in range(r.randint(0, 4))]
        return hyp, [ref]
    if kind == "long_ref":
        ref = [r.choice(WORDS[:3]) for _ in range(r.randint(40, 75))]
        hyp = ref[:]
        for _ in range(r.randint(1, 3)):
            s = r.randrange(len(hyp))
            hyp = move_block(hyp, s, r.randint(1, min(6, len(hyp) - s)), r.randint(0, len(hyp)))
        return hyp, [ref]
    return [r.choice(WORDS) for _ in range(r.randint(0, 20))], [
        [r.choice(WORDS) for _ in range(r.randint(0, 20))]]


def gen_ter(n_cases: int):
    r = random.Random(20261119)
    ter = TER()
    lines = []
    for i in range(n_cases):
        hyp, refs = ter_pair(r)
        if i % 7 == 0:  # multi-reference cases
            refs = refs + [[r.choice(WORDS) for _ in range(r.randint(1, 30))]]
        if i % 53 == 0:
            refs = [[]] + refs  # empty reference (sacreBLEU special case)
        refs_s = [" ".join(x) for x in refs]
        hyp_s = " ".join(hyp)
        kw = ter._extract_reference_info(refs_s)
        edits, avg = ter._compute_segment_statistics(hyp_s, kw)
        lines.append("\t".join([str(int(edits)), repr(float(avg)), hyp_s] + refs_s))
    # Inputs found by probing the beam-width mutants (`<` -> `==`/`<=`, `/` -> `%`, `+` -> `*`).
    probe_inputs = [
        ("b a", "b b b b c c a c c b a b a a a c a b c a b a c a a a a b b b c a a a b b c b c b a a c c c c c c a b b b a b b c b c c b b a b c c b b a a b a a a b c b a c b c b a b c a c a b b c b b b b b a a c c a b a b c a"),
        ("a b", "a a a b a b b b a a b a b b b b a a b a a b b a b b b a b a a b a a b a b a b a a b b b b a b b a b b b b b a a b a a b a b b a b b a b b b a a b a a a b a a a b a b b b a b a b b b a b a a b a b a a a b b b b b b"),
        ("a c", "c b c c c c a c b c c a a b a c a b b a b c b c c a c c b c c c a a c c c a c b a c a c a b b c b c b a a a c b b a a a c c a c b a b c c b b a b c b c c a b b c c b a a a b a b b b a a a a b b c c a"),
        ("c c", "d b d b b d a c a b c a c d c c d b c d d d b b c d c d c a c c c a d a c d a b c a a a d b d d a c d d a c b b c b d a d a d c b d c d a b d c c b d a d b a c b a a b a a d b c b b a d a d c c a c b"),
    ]
    for hyp, ref in probe_inputs:
        refs_s = [ref]
        kw = ter._extract_reference_info(refs_s)
        edits, avg = ter._compute_segment_statistics(hyp, kw)
        lines.append("\t".join([str(int(edits)), repr(float(avg)), hyp] + refs_s))
    write("mut_ter.txt", lines)


# ---------------------------------------------------------- BLEU / chrF ---

def gen_bleu_score():
    r = random.Random(7)
    lines = []
    smooths = [("none", None), ("floor", 0.1), ("floor", 0.0), ("floor", 2.0),
               ("add-k", 1), ("add-k", 0.5), ("exp", None)]
    for _ in range(1500):
        n = r.randint(1, 4)
        smooth, sv = r.choice(smooths)
        eff = r.random() < 0.5
        sys_len = r.randint(0, 30)
        ref_len = r.randint(0, 30)
        total = [r.randint(0, 12) for _ in range(n)]
        if r.random() < 0.3:
            total = [0 if r.random() < 0.5 else t for t in total]
        correct = [r.randint(0, t) for t in total]
        if r.random() < 0.3:
            correct = [0] * n
        if r.random() < 0.05:  # the all-zero-precision Int sum path (inconsistent stats)
            total = [r.randint(1, 5) for _ in range(n)]
            correct = [0] * n
        if r.random() < 0.08:  # hyp and ref both empty: bp is 1 (not 0) since sys == ref
            sys_len = ref_len = 0
        if r.random() < 0.04:  # inconsistent: matches counted with zero total (break path)
            total = [0] * n
            correct = [r.randint(0, 3) for _ in range(n)]
        stats = [sys_len, ref_len] + correct + total
        kwargs = dict(smooth_method=smooth, max_ngram_order=n, effective_order=eff)
        if sv is not None:
            kwargs["smooth_value"] = sv
        m = BLEU(**kwargs)
        score = m._compute_score_from_stats(list(stats)).score
        lines.append(" ".join([smooth, "none" if sv is None else repr(float(sv)), "1" if eff else "0",
                               str(n)] + [str(x) for x in stats] + [f64_bits(score)]))
    write("mut_bleu_score.txt", lines)


def gen_chrf_score():
    r = random.Random(11)
    lines = []
    for _ in range(1500):
        co = r.randint(1, 6)
        wo = r.randint(0, 3)
        beta = r.choice([1, 2, 3])
        eps = r.random() < 0.5
        order = co + wo
        stats = []
        for _ in range(order):
            kind = r.random()
            if kind < 0.15:
                h, rf = 0, r.randint(0, 9)
            elif kind < 0.3:
                h, rf = r.randint(0, 9), 0
            elif kind < 0.4:
                h, rf = 0, 0
            else:
                h, rf = r.randint(1, 40), r.randint(1, 40)
            mt = r.randint(0, min(h, rf))
            stats += [h, rf, mt]
        m = CHRF(char_order=co, word_order=wo, beta=beta, eps_smoothing=eps)
        score = m._compute_f_score(stats)
        lines.append(" ".join([str(co), str(wo), str(beta), "1" if eps else "0"] + [str(x) for x in stats]
                              + [f64_bits(score)]))
    write("mut_chrf_score.txt", lines)


def gen_chrf_seg():
    r = random.Random(13)
    vocab = ["ab", "abc", "a", "b.", "(x)", "c,", "a-b", "cat", "dog!", "x?", "xy", "é"]
    seps = [" ", " ", "  ", ""]

    def sent():
        k = r.randint(0, 6)
        out = ""
        for i in range(k):
            out += r.choice(vocab) + (r.choice(seps) if i < k - 1 else "")
        return out

    lines = []
    # Random cases.
    for _ in range(800):
        co = r.randint(1, 6)
        wo = r.randint(0, 3)
        ws = r.random() < 0.5
        eps = r.random() < 0.3
        beta = r.choice([1, 2, 3])
        hyp = sent()
        refs = [sent() for _ in range(r.randint(1, 3))]
        m = CHRF(char_order=co, word_order=wo, whitespace=ws, eps_smoothing=eps, beta=beta)
        stats = m._compute_segment_statistics(hyp, m._extract_reference_info(refs))
        lines.append("\t".join(["seg", str(co), str(wo), "1" if ws else "0", "1" if eps else "0", str(beta), hyp]
                               + refs + [" ".join(str(x) for x in stats)]))
    # Cases where two references tie on F-score with different statistics: sacreBLEU keeps the first.
    ties = 0
    tries = 0
    while ties < 60 and tries < 200000:
        tries += 1
        co, wo = r.randint(1, 3), r.randint(0, 2)
        hyp = sent()
        refs = [sent(), sent(), sent()]
        m = CHRF(char_order=co, word_order=wo)
        kw = m._extract_reference_info(refs)
        per = []
        for i in range(len(refs)):
            kwi = m._extract_reference_info([refs[i]])
            st = m._compute_segment_statistics(hyp, kwi)
            per.append((m._compute_f_score(st), st))
        best = max(p[0] for p in per)
        winners = [p for p in per if p[0] == best]
        if len(winners) >= 2 and winners[0][1] != winners[1][1]:
            stats = m._compute_segment_statistics(hyp, kw)
            assert stats == winners[0][1], "sacreBLEU does not keep the first best reference"
            lines.append("\t".join(["seg", str(co), str(wo), "0", "0", "2", hyp] + refs
                                   + [" ".join(str(x) for x in stats)]))
            ties += 1
    print("chrF tie cases found:", ties, "after", tries, "tries")
    write("mut_chrf_seg.txt", lines)

    punct = []
    for s in ["word!", "(a),b", "a", "", "!!", "x.", ".x", "a-b-c", "(", "a.b", "a..", "é.", "ab?c!",
              "ab'c'", "word.é", "x", "?x?", "1.5,", "end.", "éé.", "a b (c)", " x "]:
        parts = CHRF()._remove_punctuation(s)
        punct.append("\t".join([s] + parts))
    write("mut_punct.txt", punct)

    pairs = []
    rr = random.Random(17)
    for _ in range(400):
        co, wo = rr.randint(1, 7), rr.randint(0, 3)
        beta = rr.choice([1, 2, 3])
        eps = rr.random() < 0.3
        ws = rr.random() < 0.3
        cand = [sent() for _ in range(rr.randint(1, 3))]
        refs = [sent() for _ in range(rr.randint(1, 3))]
        m = CHRF(char_order=co, word_order=wo, beta=beta, eps_smoothing=eps, whitespace=ws)
        cells = [m.sentence_score(c, [rf]).score for c in cand for rf in refs]
        pairs.append("\t".join([str(co), str(wo), str(beta), "1" if eps else "0", "1" if ws else "0",
                                "|".join(cand), "|".join(refs), " ".join(f64_bits(x) for x in cells)]))
    write("mut_chrf_pairs.txt", pairs)


# ------------------------------------------------------------- numpy ------

def pcg_state_for_first_output(x: int, inc: int) -> int:
    """Return the pre-step state p so that the first next_u64 (with inc) is x (hi word 0)."""
    s1 = x & MASK128
    return ((s1 - inc) * pow(PCG_MULT, -1, 1 << 128)) & MASK128


def numpy_bitgen(state: int, inc: int):
    bg = np.random.PCG64(0)
    bg.state = {"bit_generator": "PCG64", "state": {"state": state, "inc": inc},
                "has_uint32": 0, "uinteger": 0}
    return bg


def gen_int_edge():
    lines = []
    # Lemire 64-bit with rng = 2**63 (excl = 2**63+1): rejection happens about half the time.
    for seed in [12345, 0, 99]:
        g = np.random.default_rng(seed)
        vals = g.integers(-(2 ** 62), 2 ** 62 + 1, size=600)
        lines.append("int64 " + str(seed) + " " + " ".join(str(int(v)) for v in vals))
    # Lemire 32-bit with a crafted first draw that equals the threshold exactly (no rejection in
    # numpy, since the test is `left < threshold`). Crafted through the PCG64 state.
    excl = 1000003
    rng = excl - 1
    thr = ((MASK32) - rng) % excl
    inv = pow(excl, -1, 1 << 32)
    x = (thr * inv) & MASK32
    inc = (0x0123_4567_89AB_CDEF << 1) | 1
    state = pcg_state_for_first_output(x, inc)
    bg = numpy_bitgen(state, inc)
    assert bg.random_raw() == x, "crafted PCG state does not yield the intended first output"
    bg = numpy_bitgen(state, inc)
    out = np.random.Generator(bg).integers(0, excl, size=8)
    lines.append("crafted32 " + str(excl) + " " + str(state) + " " + str(inc) + " "
                 + " ".join(str(int(v)) for v in out))
    # Crafted 32-bit case where the first draw is rejected (left < threshold).
    x2 = ((thr - 1) * inv) & MASK32  # left = thr - 1 < thr -> rejection
    state2 = pcg_state_for_first_output(x2, inc)
    bg = numpy_bitgen(state2, inc)
    assert bg.random_raw() == x2
    bg = numpy_bitgen(state2, inc)
    out2 = np.random.Generator(bg).integers(0, excl, size=8)
    lines.append("crafted32 " + str(excl) + " " + str(state2) + " " + str(inc) + " "
                 + " ".join(str(int(v)) for v in out2))
    write("mut_int_edge.txt", lines)


def gen_next_below():
    lines = []
    for n in [2, 3, 5, 1000, 2 ** 31 + 5, 2 ** 32, 2 ** 40 + 3, 2 ** 63 + 1, 2 ** 64 - 1]:
        for seed in [12345, 3]:
            g = np.random.default_rng(seed)
            vals = g.integers(0, n, size=300, dtype=np.uint64)
            lines.append(f"{n} {seed} " + " ".join(str(int(v)) for v in vals))
    write("mut_next_below.txt", lines)


def gen_stat_mean():
    rnd = random.Random(99)
    lines = []
    for _ in range(400):
        n = rnd.randint(1, 6)
        data = []
        for _ in range(n):
            e = rnd.randint(-1074, 1023)
            mant = rnd.random() + 1.0 if e > -1022 else rnd.random()
            try:
                v = math.ldexp(mant, e)
            except OverflowError:
                v = 1.0
            if rnd.random() < 0.5:
                v = -v
            if not math.isfinite(v) or v == 0.0:
                v = 2.0 ** -1000
            data.append(v)
        expected = statistics.mean(data)
        lines.append(f64_bits(expected) + " " + " ".join(f64_bits(x) for x in data))
    lines.append(f64_bits(statistics.mean([2.0 ** -1000])) + " " + f64_bits(2.0 ** -1000))
    # Values of similar magnitude: the rounding decision of the exact quotient matters here
    # (wide-exponent data almost never needs rounding).
    for _ in range(600):
        n = rnd.randint(2, 7)
        data = [math.ldexp(rnd.uniform(1.0, 2.0), rnd.randint(-3, 3)) * rnd.choice([1, -1]) for _ in range(n)]
        lines.append(f64_bits(statistics.mean(data)) + " " + " ".join(f64_bits(x) for x in data))
    # Exact halfway cases (ties to even, and a tiny excess that needs the sticky bit).
    ties = [
        [1.0, 2.0 ** -53],
        [1.0, 2.0 ** -53, 2.0 ** -200, 0.0],
        [-1.0, -(2.0 ** -53), -(2.0 ** -200), 0.0],
        [1.0, 2.0 ** -53, 0.0, 0.0],
        [1.0 + 2.0 ** -52, 2.0 ** -53],
        [2.0 ** 52 + 1.0, 1.0],
        [3.0, 2.0 ** -52, 0.0],
    ]
    for data in ties:
        lines.append(f64_bits(statistics.mean(data)) + " " + " ".join(f64_bits(x) for x in data))
    # Near the top of the range: results with binary exponent 1022 and 1023.
    big = [[sys.float_info.max], [sys.float_info.max, sys.float_info.max],
           [sys.float_info.max, sys.float_info.max, -sys.float_info.max / 2],
           [2.0 ** 1023], [1.5 * 2.0 ** 1022], [1.5 * 2.0 ** 1022, 1.5 * 2.0 ** 1022, 2.0 ** 1020]]
    for data in big:
        lines.append(f64_bits(statistics.mean(data)) + " " + " ".join(f64_bits(x) for x in data))
    write("mut_stat_mean.txt", lines)


PY_SUM_RUNNER = r"""
import struct, sys
for line in sys.stdin:
    items = []
    for t in line.split():
        if t[0] == "i":
            items.append(int(t[1:]))
        else:
            items.append(struct.unpack(">d", bytes.fromhex(t[1:]))[0])
    s = sum(items)
    if isinstance(s, int):
        print("i" + str(s))
    else:
        print("f" + struct.pack(">d", s).hex())
"""


def _encode_item(x):
    if isinstance(x, int):
        return "i" + str(x)
    return "f" + struct.pack(">d", x).hex()


def _run_sum(interp, encoded_lines):
    import subprocess
    out = subprocess.run([interp, "-c", PY_SUM_RUNNER], input="\n".join(encoded_lines) + "\n",
                         capture_output=True, text=True, check=True)
    return out.stdout.split("\n")[:len(encoded_lines)]


def gen_py_sum():
    """CPython `sum()` over mixed int/float items, for the 3.12 (Neumaier) and 3.14 (NeumaierInts)
    summation modes, plus a plain left-to-right reduction (Naive, CPython <= 3.11)."""
    import functools
    import operator
    r = random.Random(5)
    hard = [1e16, -1e16, 1.0, -1.0, 0.1, 0.2, 0.3, 3.0, 1e-300, 2.5, -7.25, 1e308, -1e308, 5e-324]
    cases = [[1e308, 1e308], [1e308, 1e308, -1e308], [1e308, 1e308, -1e308, 2.0], [0.1, 0.2, 0.3],
             [-0.0, 0.0], [1e16, 1, -1e16], [2, 1e16, 3, -1e16, 1.0], [1e16, 2, 2.0, -1e16]]
    for _ in range(3000):
        items = []
        for _ in range(r.randint(0, 9)):
            x = r.random()
            if x < 0.15:
                items.append(-9999999999)
            elif x < 0.25:
                items.append(r.randint(-5, 5))
            elif x < 0.55:
                items.append(r.choice(hard))
            else:
                items.append(math.log(r.uniform(1e-3, 100.0)))
        cases.append(items)
    encoded = [" ".join(_encode_item(x) for x in items) for items in cases]
    py312 = _run_sum("/usr/bin/python3", encoded)
    py314 = _run_sum(sys.executable, encoded)
    lines = []
    for items, e, a, b in zip(cases, encoded, py312, py314):
        naive = functools.reduce(operator.add, items, 0)
        lines.append("\t".join([_encode_item(naive), a, b, e]))
    write("mut_py_sum.txt", lines)


def main():
    n_ter = 1200
    if "--ter-cases" in sys.argv:
        n_ter = int(sys.argv[sys.argv.index("--ter-cases") + 1])
    print("numpy", np.__version__, "sacrebleu", sacrebleu.__version__, "python", sys.version.split()[0])
    os.makedirs(OUT, exist_ok=True)
    gen_bleu_score()
    gen_chrf_score()
    gen_chrf_seg()
    gen_int_edge()
    gen_next_below()
    gen_stat_mean()
    gen_py_sum()
    gen_ter(n_ter)


if __name__ == "__main__":
    main()
