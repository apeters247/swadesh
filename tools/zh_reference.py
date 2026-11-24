"""Reference model of the oracle's `zh` tokenizer, derived by black-box measurement.

The oracle is sacreBLEU 2.6.0's `TokenizerZh`; it is used only as a black box. The model, measured
2026-11-24 (see `docs/spec/tokenizers.md` section 5):

  1. strip leading/trailing Python whitespace;
  2. space a `.`/`,` that does not follow a digit, consuming the pair;
  3. space a `.`/`,` that does not precede a digit, consuming the pair;
  4. split the `ZH_RANGES` characters as single tokens (but not `.`/`,`), and a `-` after a digit;
  5. accumulate everything else and join on Python whitespace.

Run as a measurement battery: `python tools/zh_reference.py` exercises the built-in battery and the
corpus under `tests/`, and exits non-zero on any difference.
"""
import glob
import itertools
import json
import random
import sys

import sacrebleu
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

ZH_RANGES = [(0x21, 0x26), (0x28, 0x2C), (0x2E, 0x2F), (0x3A, 0x40), (0x5B, 0x60), (0x7B, 0x7E),
             (0x200B, 0x2027), (0x202A, 0x202E), (0x2030, 0x205E), (0x2060, 0x2A6D),
             (0x2E80, 0x2FDF), (0x2FF0, 0x2FFF), (0x3001, 0x303F), (0x3100, 0x312F),
             (0x31A0, 0x31EF), (0x3200, 0x4DB5), (0x4E00, 0x9FBB), (0xF900, 0xFA2D),
             (0xFA30, 0xFA6A), (0xFA70, 0xFAD9), (0xFE10, 0xFE1F), (0xFE30, 0xFE4F),
             (0xFF00, 0xFFEF)]


def zh_splits(c):
    return any(a <= ord(c) <= b for a, b in ZH_RANGES)


def is_digit(c):
    return c.isascii() and c.isdigit() or ('\uff10' <= c <= '\uff19')


def split_left(s):
    out, i, n = [], 0, len(s)
    while i < n:
        if i + 1 < n and not is_digit(s[i]) and s[i + 1] in ".,":
            out.append(s[i] + " " + s[i + 1] + " ")
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def split_right(s):
    out, i, n = [], 0, len(s)
    while i < n:
        if i + 1 < n and s[i] in ".," and not is_digit(s[i + 1]):
            out.append(" " + s[i] + " " + s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def tokenize(line):
    s = split_right(split_left(line.strip()))
    toks, run, prev = [], [], None

    def flush():
        if run:
            toks.append("".join(run))
            run.clear()

    for c in s:
        splits = c not in ".," and (zh_splits(c) or (c == '-' and prev is not None and is_digit(prev)))
        if splits:
            flush()
            toks.append(c)
        else:
            run.append(c)
        prev = c
    flush()
    return " ".join(" ".join(toks).split())


def main(argv):
    raw = TokenizerZh()
    bad = 0

    def check(label, cases, want, got):
        nonlocal bad
        wrong = [(c, want(c), got(c)) for c in cases if want(c) != got(c)]
        bad += len(wrong)
        print(f"{label}: {len(wrong)}/{len(cases)} wrong")
        for c, w, g in wrong[:5]:
            print(f"   {c!r} oracle={w!r} model={g!r}")

    alpha = ['.', ',', '0', 'a', '\u4e2d', '-', ' ', '%']
    cases = []
    for length in range(1, 5):
        cases += ["".join(c) for c in itertools.product(alpha, repeat=length)]
    rng = random.Random(3)
    cases += ["".join(rng.choice(alpha) for _ in range(rng.randint(5, 9))) for _ in range(5000)]
    cases += [".0", "..0", "...0", "a..0", "0...0", "5.,6", "0.0", "0.", "1. x", "10, 15",
              "a,b", "1,000", "5-6", "x-1", "3.5", "..,0", "\u4f60\u597dabc\u4e16\u754c"]
    check("battery", cases, raw, tokenize)

    texts = []
    for path in argv or (glob.glob("tests/oracle/*.jsonl") + glob.glob("tests/corpus/model/*.txt")):
        data = open(path, encoding="utf-8").read()
        if path.endswith(".jsonl"):
            for line in data.splitlines():
                for value in json.loads(line).values():
                    if isinstance(value, str):
                        texts.append(value)
                    elif isinstance(value, list):
                        texts.extend(v for v in value if isinstance(v, str))
        else:
            texts.extend(data.splitlines())
    if texts:
        check("corpus", texts, raw, tokenize)

    print("total wrong:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
