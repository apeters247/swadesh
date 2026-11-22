"""Record TER oracle behaviour as frozen fixtures (development-time tool; tests never call sacreBLEU).

Writes `tests/oracle/ter_probe.jsonl`:

  {"kind": "tokenize", "cfg": {...}, "input": "...", "expect": "..."}
  {"kind": "segment", "cfg": {...}, "hyp": "...", "refs": [...], "expect": ["0x...", "0x..."]}

Run with sacreBLEU installed:  python tools/measure_ter.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from sacrebleu.metrics import TER

OUT = Path(__file__).resolve().parents[1] / "tests" / "oracle" / "ter_probe.jsonl"

CONFIGS = [
    {"normalized": False, "no_punct": False, "asian_support": False, "case_sensitive": False},
    {"normalized": True, "no_punct": False, "asian_support": False, "case_sensitive": False},
    {"normalized": True, "no_punct": False, "asian_support": True, "case_sensitive": False},
    {"normalized": False, "no_punct": True, "asian_support": False, "case_sensitive": False},
    {"normalized": True, "no_punct": False, "asian_support": False, "case_sensitive": True},
]

TOKENIZE_CASES = [
    "Hello, world!", "a.b", "a...b", "end.", "3.14", "1,000", "1.2.3", "abc.,", ".,.,",
    "a;b", "a:b", "a?b", "a!b", 'a"b', "a(b)c", "a{b}c", "a[b]c", "a<b>c", "a-b", "a--b",
    "12-13", "a-13", "13-a", "x-1", "1-x", "a-b", "A-1", "\u00e9-1", "a-\u00e9", "a-", "-a",
    "a . b", "x . 3.14", ".3.14", "3.14.", "3.14-1",
    "a\u2014b", "a\u2013b", "a\u00abb\u00bb", "\u201cquoted\u201d", "don't", "'single'",
    '"double"', "x'y", "x''y", "a\u2019b", "1990's",
    "50$", "@user", "#tag", "100%", "a&b", "a*b", "x^2", "a|b", "~t", "under_score",
    "a`b", "a\\b", "a/b", "a=b", "a+b",
    "a\u00a0b", "a\tb", "  a   b  ", "a\u2003b", "Hello   World", "MiXeD CaSe",
    "a\x1fb", "a\x7eb", "a\x7fb", "a\x80b",
    # asian
    "\u4e2d\u6587\u5b57", "\u65e5\u672c\u8a9e\u3002", "\u4e2d\uff0c\u6587", "\u4e2d\u6587 \u3002",
    "\u4e2d\u6587 , \u82f1\u6587", "\u30a2\u30a4\u30a6", "\uff10\uff11", "\u30fb",
    "\u3001\u3002\u300c\u300d\u300a\u300b\u3010\u3011", "\u3042\u3044\u3046",
    "\u2f00\u2f01", "\u3000", "\uac00\ub098\ub2e4",
    "\uff01\uff08\uff09\uff0c\uff0e\uff1a\uff1f\uff61",
    "\u4e00" * 3 + "\u3002" + "\u4e8c" * 3,
]

SEGMENT_CASES = [
    ("the cat sat on the mat", ["the cat sat on the mat"]),
    ("the cat sat on the mat", ["the mat the cat sat on"]),
    ("a b c d e f g h i j k", ["a b c d e f g h i j k"]),
    ("b a", ["a b"]),
    ("a b", ["b a"]),
    ("", ["a b"]),
    ("a b", [""]),
    ("a", ["a b c"]),
    ("hello , world !", ["hello , world !"]),
    ("3.14 1,000", ["3.14 1,000"]),
    ("\u4e2d\u6587\u5b57\u3002", ["\u4e2d\u6587 \u5b57 \u3002"]),
    # multi-reference: mean of edits and of lengths
    ("a b c", ["a b c", "a b c d e"]),
    ("a b c", ["x y z", "a b c"]),
]

rnd = random.Random(20261122)
BLOCKS = ["the", "cat", "sat", "on", "mat", "a", "dog", "ran", "far"]
for _ in range(120):
    n = rnd.randrange(1, 14)
    ref = [rnd.choice(BLOCKS) for _ in range(n)]
    hyp = list(ref)
    for _ in range(rnd.randrange(1, 3)):
        if len(hyp) < 2:
            break
        i = rnd.randrange(len(hyp))
        j = min(len(hyp), i + rnd.randrange(1, 5))
        blk = hyp[i:j]
        del hyp[i:j]
        k = rnd.randrange(len(hyp) + 1)
        hyp[k:k] = blk
    SEGMENT_CASES.append((" ".join(hyp), [" ".join(ref)]))

# long repeated blocks / max shift length boundaries
for length in (9, 10, 11, 12):
    blk = " ".join(["w"] * length)
    SEGMENT_CASES.append((blk + " x y z", ["x y z " + blk]))
for dist in (49, 50, 51):
    words = ["w%d" % i for i in range(dist)]
    SEGMENT_CASES.append((" ".join(words) + " p", ["p " + " ".join(words)]))


def cfg_kwargs(cfg):
    return dict(cfg)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    metrics = {(c["normalized"], c["no_punct"], c["asian_support"], c["case_sensitive"]): TER(**c) for c in CONFIGS}
    for cfg in CONFIGS:
        key = (cfg["normalized"], cfg["no_punct"], cfg["asian_support"], cfg["case_sensitive"])
        m = metrics[key]
        for text in TOKENIZE_CASES:
            got = m._preprocess_segment(text)
            lines.append({"kind": "tokenize", "cfg": cfg, "input": text, "expect": got})
        for hyp, refs in SEGMENT_CASES:
            # The row function expects an already-preprocessed hypothesis, as the public API
            # guarantees; preprocessing here keeps the recorded rows comparable.
            hyp_pp = m._preprocess_segment(hyp)
            stats = m._compute_segment_statistics(hyp_pp, m._extract_reference_info(refs))
            lines.append({
                "kind": "segment",
                "cfg": cfg,
                "hyp": hyp,
                "refs": refs,
                "expect": [float(x).hex() for x in stats],
            })
    OUT.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} records to {OUT}")


if __name__ == "__main__":
    main()