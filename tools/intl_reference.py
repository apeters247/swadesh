"""intl as an explicit left-to-right scanner (no regex alternatives for clusters).

The previous attempts failed because a single regex had to decide which of several mutually
exclusive number shapes applied. A scanner decides at each position, consumes what it matched, and
never revisits. Measured rules it encodes:

  * a cluster starts at a digit, and continues while the next position "belongs to the number":
    a separator (. , : / -) followed by a digit, or a letter run followed by a digit
    -- so "6.152.524", "3-4", "12/12/19", "16x12", "5,000.50" are clusters,
    while "COVID-19" is not (the hyphen follows a letter, and a cluster must start at a digit)
  * a cluster may continue past a separator to end-of-token: "1986." at line end, "2?" at line end
  * every other class member is padded on its own
"""
import json
import sys

import sacrebleu.tokenizers.tokenizer_intl as m

ORACLE = getattr(m, [n for n in dir(m) if n.startswith("Tokenizer")][0])()

NONASCII = set(json.load(open("/var/www/llmeval-wt/intl_splits.json")))
ASCII_CLASS = set('!"#$%&\'()*+-/;<=>?@[\\]^_`{|}~=')
SEP = set(".,:/-\uff0c\uff1a\u3001\u2010\u2011\u2013\u2014\u30fb\uff0f\uff0e")
DASHES = set("-\u2010\u2011\u2013\u2014\u30fb")
PYSPACE = set("\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004"
              "\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u205f\u3000")


def is_class(c):
    return ord(c) in NONASCII or c in ASCII_CLASS or c in DASHES or c in ".,:"


def _is_emoji(c):
    cp = ord(c)
    return 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF


def is_digit(c):
    return c.isdigit()


def scan(line):
    s = line.rstrip("".join(PYSPACE))
    toks, cur = [], []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in PYSPACE:
            if cur:
                toks.append("".join(cur))
                cur = []
            i += 1
            continue
        if is_digit(c):
            # Consume the whole number cluster.
            j = i + 1
            while j < n:
                d = s[j]
                if is_digit(d):
                    j += 1
                elif d in SEP and j + 1 < n and is_digit(s[j + 1]):
                    j += 1
                elif d in "([\uff08) ]\uff09" and j + 1 < n and is_digit(s[j + 1]):
                    # A bracket between digits continues the run: "24（1営業日".
                    j += 1
                elif d.isalpha() and j + 1 < n and is_digit(s[j + 1]):
                    k = j
                    while k < n and s[k].isalpha():
                        k += 1
                    if k < n and is_digit(s[k]):
                        j = k
                    else:
                        break
                else:
                    break
            # A trailing "." or "," stays with the number only at end of the line ("7." ->
            # "7."), never before a space ("7. x" -> "7 . x"). Other separators always split.
            # Measured: at end of line, ONE class character stays glued to the number
            # ("7." -> "7.", "1699?" -> "1699?"); before a space it splits.
            # Restrict to punctuation/symbols that sit adjacent in text; an emoji is its own
            # token even at end of line ("19:06" -> "19:06😅" splits).
            if j + 1 == n and j < n and is_class(s[j]) and not _is_emoji(s[j]):
                j += 1
            cur.append(s[i:j])
            i = j
            continue
        if is_class(c):
            # Measured (probe series + corpus): a punctuation character glues the following
            # digit run when the punctuation STARTS A TOKEN, i.e. the character before it is a
            # space, or the line begins there. "HD)、２つ" glues because ")" split off, leaving
            # "、" at a token start; "x，4个" and "x —2300" split because "x"/" " came first;
            # "，4个" and "—2300" at line start glue.
            # Only the glossing punctuation leads a digit run; "$" and ordinary quotes do not.
            # Measured: the CJK comma glues a following digit run at any token start
            # ("HD)、２つ"); every other leading punctuation glues only at the very start of
            # the line ("-80" yes; "50 - 80" splits into "- 80").
            cjk_comma = c in "\u3001\uff0c"
            # A dash that directly follows another dash glues a digit run anywhere:
            # "心碎——2300" -> "心碎 — —2300". Brackets and quotes glue only at line start.
            dash_after_dash = c in "\u2014\u2013-" and i >= 1 and s[i - 1] in "\u2014\u2013-"
            other_at_line_start = (c in "([\uff08\u201c\u2014\u2013-" and i == 0) or dash_after_dash
            starts_token = not cur
            lead_class = (cjk_comma and starts_token) or (other_at_line_start and starts_token)
            if lead_class and starts_token and i + 1 < n and is_digit(s[i + 1]):
                j = i + 1
                while j < n and is_digit(s[j]):
                    j += 1
                # The run continues into letters when the punctuation led it: "—2300万人",
                # "、２つのHDMI端子".
                while j < n and s[j].isalpha() and not is_class(s[j]) and not _is_emoji(s[j]):
                    j += 1
                if cur:
                    toks.append("".join(cur))
                    cur = []
                toks.append(s[i:j])
                i = j
                continue
            if cur:
                toks.append("".join(cur))
                cur = []
            toks.append(c)
            i += 1
            continue
        cur.append(c)
        i += 1
    if cur:
        toks.append("".join(cur))
    return " ".join(toks)


bad = total = 0
for path in sys.argv[1:]:
    for i, line in enumerate(open(path, encoding="utf-8").read().splitlines()):
        total += 1
        want = str(ORACLE(line))
        got = scan(line)
        if want.split() != got.split():
            bad += 1
            if bad <= 3:
                print(f"{path}:{i}\n w {' '.join(want.split())[:95]}\n g {' '.join(got.split())[:95]}")
print(f"{total} lines, {bad} mismatches")