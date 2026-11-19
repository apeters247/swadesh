"""Input generators for the deep differential run (tests/test_deep.py).

Hypothesis draws one integer seed per example, and everything else comes from
``random.Random(seed)``. The reason is cost: a Hypothesis ``draw`` costs about
0.7 ms, and one corpus needs 25 or more of them, so a draw per character would
make generation slower than the metric itself. ``make_case(random.Random(seed))``
rebuilds any case from its seed, and tests/shrink.py reduces a failing case to a
minimal repro.

Segment families target the areas the earlier suites covered only lightly:
long segments (1k-10k characters), whitespace-only and empty segments,
punctuation-only and digit-only strings, mixed scripts, every Unicode
whitespace, long runs of ``. , - 0-9`` (13a regex), chained XML entities and
``<skipped>``. Reference streams can hold up to 8 references with ``None``
holes. TER hypotheses are 20-200 word sentences, and their references are
block-shifted copies, so TER's shift search fires.
"""
import unicodedata

from test_differential import ALPHABET

SURR = range(0xD800, 0xE000)


def _chars(pred):
    return [chr(c) for c in range(0x110000) if c not in SURR and pred(chr(c))]


# Every Unicode White_Space code point, plus everything Python's isspace() accepts.
_WHITE_SPACE_PROPERTY = [
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
    *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
]
WS_CHARS = sorted(set(_chars(str.isspace)) | {chr(c) for c in _WHITE_SPACE_PROPERTY})
PUNCT_CHARS = sorted(set(_chars(lambda ch: unicodedata.category(ch).startswith("P")))
                     | set("$+<=>^`|~"))
DIGIT_CHARS = sorted(set(_chars(lambda ch: unicodedata.category(ch) in ("Nd", "Nl", "No")))
                     | set("0123456789"))

SCRIPT_RANGES = {
    "latin": [(0x41, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)],
    "greek": [(0x370, 0x3FF), (0x1F00, 0x1FFF)],
    "cyrillic": [(0x400, 0x52F)],
    "armenian": [(0x531, 0x58F)],
    "hebrew": [(0x591, 0x5F4)],
    "arabic": [(0x600, 0x6FF), (0x750, 0x77F)],
    "devanagari": [(0x900, 0x97F)],
    "bengali": [(0x980, 0x9FF)],
    "tamil": [(0xB80, 0xBFF)],
    "thai": [(0xE00, 0xE7F)],
    "georgian": [(0x10A0, 0x10FF)],
    "ethiopic": [(0x1200, 0x137F)],
    "cherokee": [(0x13A0, 0x13FF)],
    "hangul": [(0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "kana": [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D)],
    "han": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x2E80, 0x2FDF)],
    "ext_han": [(0x20000, 0x2A6DF)],
    "emoji": [(0x1F300, 0x1FAFF)],
    "combining": [(0x300, 0x36F), (0x1AB0, 0x1AFF)],
    "fullwidth": [(0xFF01, 0xFF5E)],
    "special": [0x130, 0x131, 0x1E9E, 0xDF, 0x1C5, 0xFB01, 0x2D3],
}
SCRIPT_CHARS = {}
for _name, _rs in SCRIPT_RANGES.items():
    _out = []
    for _r in _rs:
        if isinstance(_r, tuple):
            _out.extend(chr(c) for c in range(_r[0], _r[1] + 1) if c not in SURR)
        else:
            _out.append(chr(_r))
    SCRIPT_CHARS[_name] = _out
SCRIPT_NAMES = sorted(SCRIPT_CHARS)

ENTITY_TOKENS = ["&amp;", "&amp;amp;", "&amp;amp;amp;", "&quot;", "&amp;quot;", "&lt;", "&gt;",
                 "&lt;skipped&gt;", "&amp;lt;skipped&amp;gt;", "<skipped>", "<skipped", "skipped>",
                 "&", ";", "amp;", "x", " ", "1.", "&amp;&amp;", "&#38;"]
TER_WORDS = ["the", "cat", "sat", "on", "mat", "a", "dog", "is", "of", "it", "and", "to",
             "x,", "y.", "'s", "3-4", "-", "1,000", "&amp;", "<skipped>", "..."]
_RUN_CHARS = ".,-0123456789"
_WS_MIXED_EXTRA = ["a", "1", ".", "ß"]


def _join(r, pick, lo, hi):
    return "".join(pick(r) for _ in range(r.randint(lo, hi)))


def _uni(r):
    while True:
        cp = r.randrange(0x110000)
        if cp not in SURR:
            return chr(cp)


def _mixed(r):
    parts = []
    for _ in range(r.randint(1, 6)):
        chars = SCRIPT_CHARS[r.choice(SCRIPT_NAMES)]
        parts.append("".join(r.choice(chars) for _ in range(r.randint(1, 8))))
    return "".join(parts)


def _run(r, hi):
    return "".join(r.choice(_RUN_CHARS) * r.randint(1, hi) for _ in range(r.randint(1, 3)))


def _long(r):
    """1000-10000 characters, built from 10-40 character chunks."""
    return "".join(_join(r, lambda q: q.choice(ALPHABET), 10, 40) for _ in range(r.randint(100, 250)))


def _ws_mixed(r):
    pool = WS_CHARS + _WS_MIXED_EXTRA
    return _join(r, lambda q: q.choice(pool), 1, 20)


# (name, weight, generator). The common families are short. The rare families
# (100-400 character runs and 1k-10k characters) are drawn in ``segment`` at 0.2%.
_FAMILY_FNS = [
    ("adv", 8, lambda r: _join(r, lambda q: q.choice(ALPHABET), 0, 25)),
    ("uni", 3, lambda r: _join(r, _uni, 0, 30)),
    ("ws", 3, lambda r: _join(r, lambda q: q.choice(WS_CHARS), 0, 12)),
    ("ws_mixed", 2, _ws_mixed),
    ("punct", 3, lambda r: _join(r, lambda q: q.choice(PUNCT_CHARS), 1, 30)),
    ("digit", 3, lambda r: _join(r, lambda q: q.choice(DIGIT_CHARS), 1, 30)),
    ("mixed", 3, _mixed),
    ("run", 3, lambda r: _run(r, 40)),
    ("entity", 3, lambda r: _join(r, lambda q: q.choice(ENTITY_TOKENS), 1, 12)),
    ("empty", 2, lambda r: ""),
]
_FAMILY_NAMES = [f[0] for f in _FAMILY_FNS]
_FAMILY_WEIGHTS = [f[1] for f in _FAMILY_FNS]
_FAMILY_BY_NAME = {f[0]: f[2] for f in _FAMILY_FNS}
RARE_PROB = 0.002


def short_segment(r):
    """A segment from the common families (at most a few hundred characters)."""
    return _FAMILY_BY_NAME[r.choices(_FAMILY_NAMES, _FAMILY_WEIGHTS)[0]](r)


def segment(r):
    """Any segment: common families, or rarely a 100-400 char run or a 1k-10k char text."""
    if r.random() < RARE_PROB:
        return _run(r, 400) if r.random() < 0.5 else _long(r)
    return short_segment(r)


def shifted(r, text, max_moves=6, max_block=30):
    """Copy of ``text`` with 1..max_moves block moves and an optional word deletion."""
    words = text.split()
    if len(words) < 4:
        return text
    for _ in range(r.randint(1, max_moves)):
        a = r.randrange(len(words))
        ln = r.randint(1, min(max_block, len(words) - a))
        block = words[a:a + ln]
        rest = words[:a] + words[a + ln:]
        at = r.randint(0, len(rest))
        words = rest[:at] + block + rest[at:]
    if r.random() < 0.5 and len(words) > 4:
        del words[r.randrange(len(words))]
    return " ".join(words)


def word_sentence(r, lo=1, hi=15):
    return " ".join(r.choice(TER_WORDS) for _ in range(r.randint(lo, hi)))


def make_case(r, max_segments=8, max_refs=8):
    """(hyps, refs) for BLEU/chrF. Stream 0 is always defined; later streams may hold None."""
    n = r.randint(1, max_segments)
    hyps = [segment(r) for _ in range(n)]
    refs = []
    for k in range(r.randint(1, max_refs)):
        stream = []
        for _ in range(n):
            if k > 0 and r.random() < 0.25:
                stream.append(None)
            else:
                stream.append(segment(r))
        refs.append(stream)
    return hyps, refs


def make_ter_case(r):
    """(hyps, refs) for TER: short word sentences, shifted copies, None holes.

    sacreBLEU's TER costs 1-300 ms per sentence at 10-50 words, so this stays short;
    make_ter_long_case covers 20-200 words at a smaller count.
    """
    hyps = [word_sentence(r) if r.random() < 0.8 else short_segment(r) for _ in range(r.randint(1, 2))]
    refs = []
    for k in range(r.randint(1, 2)):
        stream = []
        for h in hyps:
            choice = r.randrange(6)
            if choice == 1 and k > 0:
                stream.append(None)  # stream 0 is always defined
            elif choice == 2:
                stream.append(short_segment(r))
            elif choice == 3:
                stream.append(word_sentence(r))
            else:
                stream.append(shifted(r, h))
        refs.append(stream)
    return hyps, refs


def make_ter_long_case(r):
    """(hyps, refs) for TER with 20-200 word sentences and block-shifted references."""
    hyp = word_sentence(r, 20, 200)
    refs = []
    for _ in range(r.randint(1, 2)):
        refs.append([shifted(r, hyp) if r.random() < 0.8 else word_sentence(r, 20, 200)])
    return [hyp], refs
