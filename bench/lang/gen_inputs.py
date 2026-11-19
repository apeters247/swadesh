"""Generate benchmark inputs: references repeated to N lines, perturbed hypotheses.

    python3 -I gen_inputs.py OUTDIR [DATADIR] [N]

Writes OUTDIR/<lp>.ref.txt and OUTDIR/<lp>.hyp.txt (one segment per line, LF only).
Perturbations are deterministic (seed per language pair); each repeated copy of a
reference gets a different perturbation so copies are not byte-identical hypotheses.
"""
import os
import random
import sys

LANGPAIRS = ["en-de", "zh-en"]


def perturb_words(line, rnd):
    w = line.split(" ")
    if len(w) < 2:
        return line
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
    return " ".join(w)


def perturb_chars(line, rnd):
    c = list(line)
    for _ in range(max(1, len(c) // 8)):
        if not c:
            break
        op = rnd.random()
        i = rnd.randrange(len(c))
        if op < 0.4:
            del c[i]
        elif op < 0.7:
            c[i] = c[rnd.randrange(len(c))]
        else:
            c.insert(i, c[i])
    return "".join(c)


def main():
    out = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else "/var/www/llmeval/bench/data"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    os.makedirs(out, exist_ok=True)
    for lp in LANGPAIRS:
        with open(os.path.join(data, f"wmt22.{lp}.ref"), encoding="utf-8") as f:
            refs = [l for l in f.read().split("\n") if l][:2000]
        if lp.startswith("zh"):
            # Chinese has no spaces: perturb characters instead of words.
            fn = perturb_chars
        else:
            fn = perturb_words
        rnd = random.Random(1234 + len(lp))
        with open(os.path.join(out, f"{lp}.ref.txt"), "w", encoding="utf-8", newline="\n") as fr, \
             open(os.path.join(out, f"{lp}.hyp.txt"), "w", encoding="utf-8", newline="\n") as fh:
            for i in range(n):
                r = refs[i % len(refs)]
                fr.write(r + "\n")
                fh.write(fn(r, rnd) + "\n")
        print(lp, len(refs), "base lines ->", n, "segments")


if __name__ == "__main__":
    main()
