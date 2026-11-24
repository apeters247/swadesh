"""Reference model of the oracle's `intl` tokenizer, derived by black-box measurement.

The oracle is sacreBLEU 2.6.0's `TokenizerV14International`, used only as a black box. Two layers
are measured separately:

  * the **tokenizer** itself (`TokenizerV14International.__call__`), and
  * the **metric layer** (`Metric._preprocess_segment`), which strips trailing whitespace before
    calling the tokenizer.

The tokenizer model is three passes, each established by measurement (see
`docs/spec/intl-absorption.md`):

  1. pad every Unicode symbol (category S*) with a space on both sides;
  2. split between a non-number character and a following punctuation character (category P*),
     consuming both, scanning left to right without overlap;
  3. split between a punctuation character and a following non-number character, same scan;

then join on Python whitespace. Symbols are never glued to a number; punctuation is glued to a
number. The non-overlapping scan is what makes two leading quotes behave differently from two
leading marks: `‘’/1` pairs `‘` with `’`, resumes at `/`, and `/1` therefore glues, while `./1`
pairs `.` with `/` and splits.

Run as a measurement battery:

    python tools/intl_reference.py [corpus.txt|corpus.jsonl ...]

It prints one line per battery and exits non-zero if any case differs from the oracle.
"""
import json
import sys
import unicodedata

import sacrebleu
from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International


def is_punct(c):
    return unicodedata.category(c)[0] == "P"


def is_symbol(c):
    return unicodedata.category(c)[0] == "S"


def is_number(c):
    return unicodedata.category(c)[0] == "N"


def pad_symbols(s):
    return "".join((" " + c + " ") if is_symbol(c) else c for c in s)


def split_left(s):
    """Pass 2: a non-number followed by punctuation; the match consumes both characters."""
    out, i, n = [], 0, len(s)
    while i < n:
        if i + 1 < n and not is_number(s[i]) and is_punct(s[i + 1]):
            out.append(s[i] + " " + s[i + 1] + " ")
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def split_right(s):
    """Pass 3: punctuation followed by a non-number; the match consumes both characters."""
    out, i, n = [], 0, len(s)
    while i < n:
        if i + 1 < n and is_punct(s[i]) and not is_number(s[i + 1]):
            out.append(" " + s[i] + " " + s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def tokenize(line):
    """The tokenizer layer, as a space-joined string (no trailing-whitespace strip)."""
    return " ".join(split_right(split_left(pad_symbols(line))).split())


def preprocess(line):
    """The metric layer: strip trailing whitespace, then tokenize."""
    return tokenize(line.rstrip())


def main(argv):
    raw = TokenizerV14International()
    metric = sacrebleu.BLEU(tokenize="intl")
    bad = 0

    def check(label, cases, want, got):
        nonlocal bad
        wrong = [(c, want(c), got(c)) for c in cases if want(c) != got(c)]
        bad += len(wrong)
        print(f"{label}: {len(wrong)}/{len(cases)} wrong")
        for c, w, g in wrong[:5]:
            print(f"   {c!r} oracle={w!r} model={g!r}")

    fixed = ["/1", " /1", "\u2018\u2019/1", "a\u2018\u2019/1", "''/1", "'''/1", '"/1', '""/1',
             "\u201c\u201d/1", "\u201c\u201d?1", "\u2018\u2019.1", "\u2018\u2019a1", "'/1",
             "a'/1", "\u2018/1", "\u2018\u2019%/1", "\u2018\u2019  /1", "3.5%", "a1/1", "2,000",
             "3.5.", "5-6", "x-1", "1/x", "%3", "a%b", "3%4", "a./1", "./1", "a../1", "2./9",
             "a.2/1", "a?/1", "a,/1", "x./1", "2/1", "$5", "5$", "\u00a56F", "G#OV6\u26063",
             "R\u20ac\u21905mmB", "M\u2605\u00a17\u00a5\u00a3y", "w?~5%:|$", "3.5", "U.S.A.",
             "(x)", "a_b", "hello, world!", "\u00ab2", "\u201c6", "9\u30029u", "7. ", "7.",
             "a. ", "a.", "a ? ", "a.  b", "a .b", "7 . 5", "7. 5", "a_1", "_1", "1_", "a_",
             "_a", "a\U0001f605", "\U0001f468\u200d\U0001f469", "0. ", "7  ", " 7"]
    check("fixed", fixed, raw, tokenize)

    alpha = [chr(c) for c in range(0x20, 0x7F)]
    uni = [chr(c) for c in range(0xA0, 0x2100) if unicodedata.category(chr(c))[0] in ("P", "S")]
    cjk = [chr(c) for c in (0x3001, 0x3002, 0x3008, 0x3009, 0x300C, 0x300D, 0xFF01, 0xFF0C,
                            0xFF1F, 0x30FB)]
    pool = [c for c in alpha + uni + cjk if c not in ("\n", "\r", "\t")]
    rng = __import__("random").Random(5)
    cases = ["".join(rng.choice(pool) for _ in range(rng.randint(1, 10))) for _ in range(5000)]
    words = ["hello", "world", "3.5", "2024", "a", "x1", "U.S.", "e-mail", "don't", "  "]
    cases += [rng.choice(["", " "]).join(rng.choice(words + [rng.choice(pool)])
                                        for _ in range(rng.randint(1, 5)))
              for _ in range(1500)]
    check("random-6500", cases, raw, tokenize)

    # Every P/S character in the three contexts where pass 2 cannot fire.
    chars = [chr(cp) for cp in range(0x110000) if unicodedata.category(chr(cp))[0] in ("P", "S")]
    chars.append("_")
    check("char-matrix-start", [c + "1" for c in chars], raw, tokenize)
    check("char-matrix-digit", ["1" + c + "1" for c in chars], raw, tokenize)
    check("char-matrix-end", ["1" + c for c in chars], raw, tokenize)

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
    if texts:
        check("corpus-tokenizer", texts, raw, tokenize)
        check("corpus-metric", texts, metric._preprocess_segment, preprocess)

    print("total wrong:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
