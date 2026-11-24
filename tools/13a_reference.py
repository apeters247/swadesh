"""Reference model of the oracle's `13a` tokenizer, derived by black-box measurement.

The oracle is sacreBLEU 2.6.0's `Tokenizer13a`; it is used only as a black box and its source was
not read. Two layers are measured separately, as for `intl`:

  * the **tokenizer** (`Tokenizer13a.__call__`), and
  * the **metric layer** (`Metric._preprocess_segment`), which strips trailing whitespace first.

Stages, from `docs/spec/13a-algorithm.md`:

  1. normalise: drop `<skipped>`, drop `-\\n`, newline -> space, decode four HTML entities;
  2. pad the 30-character ASCII class `!"#$%&()*+/:;<=>?@[\\]^_`{|}~`;
  3. the measured `.`/`,` run rule (see below);
  4. a `-` that follows a digit gets spaces around it;
  5. join on Python whitespace.

The `.`/`,` run rule, measured 2026-11-24. For each maximal run of `[.,]` of length `m`, with `p`
the character before it and `f` the character after it:

  * the run is padded on both sides, except that
  * the run's **last** character glues to `f` iff `f` is a digit and `(m + [p is a digit])` is even;
  * the run's **first** character glues to `p` iff `p` is a digit, `m == 1`, and `f` is a digit.

`..0` -> `. .0` (m=2, even); `...0` -> `. . . 0` (m=3, odd); `0...0` -> `0 . . .0` (digit before,
m=3, even); `0.0` -> `0.0` (m=1 between digits); `5.,6` -> `5 . , 6` (m=2, digit before, odd);
`0.a` -> `0 . a` (m=1 but `f` is not a digit). Run as a measurement battery:

    python tools/13a_reference.py [corpus.txt|corpus.jsonl ...]

It prints one line per battery and exits non-zero on any difference.
"""
import glob
import itertools
import json
import random
import re
import sys

import sacrebleu
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

PAD = set('!"#$%&()*+/:;<=>?@[\\]^_`{|}~')
DASH = re.compile(r'([0-9])(-)')


def is_attach_digit(c):
    return c.isascii() and c.isdigit() or ('\uff10' <= c <= '\uff19')


def normalise(line):
    s = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in s:
        s = (s.replace("&quot;", '"').replace("&amp;", "&")
               .replace("&lt;", "<").replace("&gt;", ">"))
    return s


def pad_class(s):
    return "".join((" " + c + " ") if c in PAD else c for c in s)


def mark_runs(s):
    res, i, n = [], 0, len(s)
    while i < n:
        if s[i] in ".,":
            j = i
            while j < n and s[j] in ".,":
                j += 1
            m = j - i
            p = s[i - 1] if i > 0 else None
            f = s[j] if j < n else None
            p_digit = p is not None and is_attach_digit(p)
            f_digit = f is not None and is_attach_digit(f)
            left_glue = p_digit and m == 1 and f_digit
            right_glue = f_digit and (m + (1 if p_digit else 0)) % 2 == 0
            if not left_glue:
                res.append(" ")
            for k in range(i, j):
                res.append(s[k])
                if k < j - 1 or not right_glue:
                    res.append(" ")
            i = j
        else:
            res.append(s[i])
            i += 1
    return "".join(res)


def tokenize(line):
    """The tokenizer layer (no trailing-whitespace strip)."""
    return " ".join(DASH.sub(r'\1 \2 ', mark_runs(pad_class(normalise(line)))).split())


def preprocess(line):
    """The metric layer: strip trailing whitespace, then tokenize."""
    return tokenize(line.rstrip())


def main(argv):
    raw = Tokenizer13a()
    metric = sacrebleu.BLEU(tokenize="13a")
    bad = 0

    def check(label, cases, want, got):
        nonlocal bad
        wrong = [(c, want(c), got(c)) for c in cases if want(c) != got(c)]
        bad += len(wrong)
        print(f"{label}: {len(wrong)}/{len(cases)} wrong")
        for c, w, g in wrong[:5]:
            print(f"   {c!r} oracle={w!r} model={g!r}")

    alpha = ['.', ',', '0', 'a', '&', '(', '-', ' ']
    cases = []
    for length in range(1, 5):
        cases += ["".join(c) for c in itertools.product(alpha, repeat=length)]
    rng = random.Random(11)
    cases += ["".join(rng.choice(alpha) for _ in range(rng.randint(5, 12))) for _ in range(6000)]
    cases += [".0", "..0", "...0", "....0", "0...0", "5.,6", "0.0", "0.", "0.a", "1,000.50",
              "It costs $1,000.50!", "Hello, world.", "1990-1995", "37.", "9,",
              "`\u2e80'&", "a'b", "AT&T", "&quot;hi&quot;", "x<skipped>y", "a-\nb"]
    check("battery", cases, raw, tokenize)

    texts = []
    for path in argv:
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
    if not texts:
        texts = []
        for path in sorted(glob.glob("tests/oracle/*.jsonl")):
            for line in open(path, encoding="utf-8"):
                for value in json.loads(line).values():
                    if isinstance(value, str):
                        texts.append(value)
                    elif isinstance(value, list):
                        texts.extend(v for v in value if isinstance(v, str))
        for path in sorted(glob.glob("tests/corpus/model/*.txt")):
            texts.extend(open(path, encoding="utf-8").read().splitlines())
    check("corpus-tokenizer", texts, raw, tokenize)
    check("corpus-metric", texts, metric._preprocess_segment, preprocess)

    print("total wrong:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
