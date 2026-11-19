"""Generate stress inputs under WORK (default ~/swadesh-stress/work).

Scaling corpora: prefixes of one deterministic stream, bench.load(10M) (WMT22
en-de references, perturbed, seed 1234). The 2037 WMT22 references cycle, so
large corpora contain repeated references with different hypotheses.

Pathological inputs: see gen_pathological().
"""
import argparse
import os
import random
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)
from bench.bench import load  # noqa: E402  (bench/bench.py, same generator as the benchmark)

SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
WORK = os.environ.get("SWADESH_STRESS_WORK", os.path.expanduser("~/swadesh-stress/work"))


def write(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def wmt_refs():
    with open(os.path.join(REPO, "bench", "data", "wmt22.en-de.ref"), encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def perturb(words, rnd, rate):
    """Word deletions, swaps, reversals and block moves (block moves matter for TER)."""
    w = list(words)
    for _ in range(max(1, int(len(w) * rate))):
        if not w:
            break
        op = rnd.random()
        i = rnd.randrange(len(w))
        if op < 0.25:
            del w[i]
        elif op < 0.5 and len(w) > 1:
            j = rnd.randrange(len(w))
            w[i], w[j] = w[j], w[i]
        elif op < 0.75 and len(w) > 3:
            length = rnd.randint(1, min(5, len(w) - 1))
            block = w[i:i + length]
            del w[i:i + length]
            j = rnd.randrange(len(w) + 1)
            w[j:j] = block
        else:
            w[i] = w[i][::-1]
    return w


def gen_scaling():
    os.makedirs(WORK, exist_ok=True)
    hyps, refs = load("en-de", max(SIZES))
    refs = refs[0]
    for n in SIZES:
        write(os.path.join(WORK, f"c{n}.hyp.txt"), hyps[:n])
        write(os.path.join(WORK, f"c{n}.ref.txt"), refs[:n])
        print("wrote corpus", n, flush=True)
    del hyps, refs


def gen_pathological():
    out = os.path.join(WORK, "path")
    os.makedirs(out, exist_ok=True)
    rnd = random.Random(7)
    base = wmt_refs()
    vocab = [w for r in base for w in r.split()]

    def sent(n):
        return [rnd.choice(vocab) for _ in range(n)]

    def pair(name, hyps, refs):
        write(os.path.join(out, f"{name}.hyp.txt"), hyps)
        write(os.path.join(out, f"{name}.ref.txt"), refs)
        print("wrote", name, len(hyps), flush=True)

    # 1. One ~1 MB line (about 1e6 characters) with 15% perturbation.
    words = []
    while sum(len(w) + 1 for w in words) < 1_000_000:
        words.extend(sent(1000))
    ref = " ".join(words)
    hyp = " ".join(perturb(words, rnd, 0.15))
    pair("oneline_1mb", [hyp], [ref])

    # 2. Ten lines of 100k characters without spaces (ASCII letters and digits).
    alpha = string.ascii_letters + string.digits
    refs = ["".join(rnd.choice(alpha) for _ in range(100_000)) for _ in range(10)]
    hyps = []
    for r in refs:
        chars = list(r)
        for _ in range(1000):
            chars[rnd.randrange(len(chars))] = rnd.choice(alpha)
        hyps.append("".join(chars))
    pair("nospace_100k", hyps, refs)

    # 3. One million tiny segments (0-2 tokens each, some empty).
    choices = ["", "a", "b", "a b", "the", "a a", "b a b"]
    refs = [rnd.choice(choices) for _ in range(1_000_000)]
    hyps = [rnd.choice(choices) for _ in range(1_000_000)]
    pair("tiny_1m", hyps, refs)

    # 4. 10k segments with 50 references each (refs written as 50 files).
    n = 10_000
    src = [base[i % len(base)] for i in range(n)]
    hyps = [" ".join(perturb(s.split(), rnd, 0.15)) for s in src]
    ref_files = []
    for k in range(50):
        rk = [" ".join(perturb(s.split(), rnd, 0.15)) for s in src]
        name = os.path.join(out, f"refs50_r{k:02d}.txt")
        write(name, rk)
        ref_files.append(name)
    write(os.path.join(out, "refs50.hyp.txt"), hyps)
    print("wrote refs50 (50 refs x", n, ")", flush=True)

    # 5. TER on 1000-word sentences: 50 and 200 segments.
    for count in (50, 200):
        refs, hyps = [], []
        for _ in range(count):
            r = sent(1000)
            refs.append(" ".join(r))
            hyps.append(" ".join(perturb(r, rnd, 0.1)))
        pair(f"ter1000w_{count}", hyps, refs)

    # 6. All-identical corpus: 1M copies cycling over 1000 WMT references.
    same = [base[i % 1000] for i in range(1_000_000)]
    pair("identical_1m", same, same)

    # 7. Empty strings.
    pair("empty_hyp_1k", [""] * 1000, base[:1000])
    pair("all_empty_1k", [""] * 1000, [""] * 1000)
    pair("mixed_empty_1k",
         [("" if rnd.random() < 0.3 else base[i]) for i in range(1000)],
         [("" if rnd.random() < 0.3 else base[i]) for i in range(1000)])
    pair("zero_segments", [], [])
    pair("one_empty_line", [""], [""])
    return out, ref_files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["scaling", "pathological", "all"])
    a = ap.parse_args()
    if a.what in ("scaling", "all"):
        gen_scaling()
    if a.what in ("pathological", "all"):
        gen_pathological()
