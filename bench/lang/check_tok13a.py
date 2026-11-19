"""Ground truth for the tok kernel: sacreBLEU 2.6.0's Tokenizer13a on the same lines.

    python3 -I check_tok13a.py HYP REF [N]

Prints the checksum the C/Rust `--kernel tok` runs print (FNV-1a 64 over each
tokenised line's UTF-8 bytes, plus the byte length, summed mod 2^64 over
hypotheses and references). Run with a python that has sacrebleu installed.
"""
import sys

from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a


def lines(path):
    with open(path, encoding="utf-8", newline="") as f:  # no newline translation: keep \r
        text = f.read()
    v = text.split("\n")
    if v and v[-1] == "":
        v.pop()
    return v


def main():
    hyp, ref = lines(sys.argv[1]), lines(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else len(hyp)
    tok = Tokenizer13a()
    mask = (1 << 64) - 1
    chk = 0
    for side in (hyp[:n], ref[:n]):
        for line in side:
            b = tok(line).encode("utf-8")
            h = 1469598103934665603
            for x in b:
                h ^= x
                h = (h * 1099511628211) & mask
            chk = (chk + h + len(b)) & mask
    print(f"sacrebleu13a checksum={chk} segments={n}")


if __name__ == "__main__":
    main()
