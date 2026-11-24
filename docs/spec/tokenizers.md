# Tokenizer specification

This document defines the behaviour of every tokenizer in `swadesh-core`, written **before** the
implementations, from published sources only. The implementations are written against this
document; where a rule here is ambiguous, the frozen oracle corpus (`tests/oracle/`) decides, and
this document is corrected.

**Governing sourcing rule (owner instruction, 2026-11-22):** derive the algorithm from published
prose and from measured behaviour of the oracle as a black box. **Do not read or copy any
implementation source** — not sacreBLEU's, not Moses `tokenizer.perl` (LGPL-2.1; its rule sequence
must not be transcribed), not any other tokenizer. Where this document cites "Moses behaviour", it
means *the documented behaviour in the literature*, not its code.

**Sources used** (none of them sacreBLEU source code):

- Koehn & Knowles, *Six Challenges for Neural Machine Translation* (WMT 2017), which states the
  rule set used for the `13a` tokenizer and which Moses implements in `moses/baseline/moses-scripts`.
- The Moses tokenizer's *documented* behaviour as described in the literature and the manual —
  never its source, which is LGPL-2.1 and may not be transcribed.
- The Unicode Standard, Annex #29 (text segmentation) and the character database, for whitespace
  and the punctuation/symbol/number general categories.
- Python's `str` documentation for the semantics of `str.split()`, `str.strip()` and `str.lower()`,
  which the tokenizers depend on (the oracle's language is Python, so these define the target).
- The BLEU and chrF papers (Papineni et al. 2002; Popović 2015) for the char-level tokenizer and
  what its input is expected to look like.
- Public cross-check only, never copied: `LoicGrobol/spertiniite`, an independent Rust
  reimplementation of the same tokenizers (licence `NOASSERTION`). It was consulted to confirm that
  the rules below match what an independent implementer arrived at; no code was taken from it.

Everything the tokenizers take as input comes in already **stripped** (`str.rstrip()`) and,
for `--lowercase`, lowered by the caller. Tokenizers do not lowercase.

## Common postcondition

Each tokenizer maps a string to a string. For `13a`, `intl` and `zh` the result is
`" ".join(result.split())`: tokens separated by exactly one ASCII space, no leading or trailing
space. Where a rule can produce adjacent spaces, they are collapsed by that final step.

**`char` is the exception, and the first draft of this document got it wrong.** `char` emits one
token per Unicode scalar value and keeps whitespace as its own token, so runs do not collapse:
measured `"ab cd"` → `"a b   c d"`, `"a  b"` → `"a     b"`, `" x "` → `"  x  "`. Collapsing would
change the token count and therefore the score.

`none` is the identity, whitespace included: `"ab cd"` → `"ab cd"`. At token level this agrees with
the postcondition, because the consumer splits on Python whitespace anyway.

The "unjoined" variants emit the same tokens separated by `\0` and are an internal optimisation;
they are not observable.

## 1. `none`

The identity function.

## 2. `char`

The input, with each whitespace-separated run split into its individual characters, and those
characters joined by single spaces. Unicode-aware: a character is one scalar value.

## 3. `13a`

The rule set of Koehn & Knowles. Applied in this order; each rule is a global substitution.

1. For every character in the **punctuation/symbol class** of rule A below, surround it with
   spaces.
2. Attach a `.` or `,` to the token before it, unless that token's last character is a digit. Formally:
   for every occurrence of a non-digit character immediately followed by `.` or `,`, insert a space
   between them and a space after the `.`/`,`.
3. Separately, for every `.` or `,` immediately followed by a non-digit, insert a space before the
   `.`/`,` and between it and the following character.
4. For a digit immediately followed by `-`, insert a space between them and after the `-`.
5. **Entity decoding and cleanup, applied before the substitutions above:**

   a. Remove every occurrence of `<skipped>`.
   b. Remove every occurrence of `-\n` (hyphen followed by a newline).
   c. Replace every `\n` with a single space.
   d. If the line contains `&`, replace every `&quot;` with `"`, every `&amp;` with `&`,
      every `&lt;` with `<`, every `&gt;` with `>`. There is **no** `&` → `&amp;` rule and no
      interchange; a bare `&` stays bare.

   Measured: `"a&amp;b"` → `a & b`; `"x<skipped>y"` → `xy`; `"line-\nnext"` → `linenext`;
   `"&quot;hi&quot;"` → `" hi "`; `"AT&T"` → `AT & T` (the `&` is split by rule A).
6. Every single character that is a Unicode **number** (`\p{N}`: decimal digits, letter numbers,
   other numbers) is surrounded by spaces.

Rule A's punctuation/symbol class, in the notation of the Unicode general category:
`\p{P}` (all punctuation) plus `\p{S}` (all symbols), as defined by the Unicode version of the
interpreter the oracle runs on.

Whitespace: splitting into tokens uses **Python whitespace**, not Unicode `\p{Z}`. The Python set is
`\t \n \v \f \r \x1c \x1d \x1e \x1f \x20 \x85 \xa0 \u1680 \u2000-\u200a \u2028 \u2029 \u205f \u3000`.
Notably `\u200b` (zero-width space) is **not** whitespace for this purpose and stays inside a token.

Two consequences that an implementation must get right:

- Rule 2 says "the token before it", i.e. the run of non-space characters immediately preceding. A
  `.` at the start of the string has no preceding token, so rule 2 does not apply, but rule 3 may.
- Rules 2 and 3 scan the *same* string in their own passes and each performs its own
  non-overlapping, left-to-right replacement. Whether a character `.,` is consumed by rule 2 or
  rule 3 when both could match at nearby positions is determined by that ordering; an
  implementation that merges them into one pass will differ on inputs such as `5.,6`.

## 4. `intl`

The full measured model, with its evidence, is `docs/spec/intl-absorption.md`. In summary, given a
segment:

0. Drop trailing Python whitespace.
1. Pad every Unicode symbol (`S*`) with a space on both sides.
2. Scan left to right: a non-number followed by punctuation (`P*`) gets a space around both, and
   the scan resumes after the pair.
3. Scan again: punctuation followed by a non-number gets a space around both, resuming after the
   pair.
4. Split on Python whitespace.

Consequences: punctuation glues to numbers (`3%4`, `3.5%`, `_1`) while symbols never do (`$5`,
`¥6F`, `→`); punctuation is separated from every non-number neighbour (`a%1`); the
pair-consuming scan is why `‘’/1` → `‘ ’ /1` while `./1` → `. / 1`. `%` and `→` differ from `13a` exactly as
measured before, and both follow from this model.

No entity decoding and no `<skipped>`/`-\n` removal: `"a&amp;b"` becomes `a & amp ; b` and
`"a<skipped>b"` becomes `a < skipped > b`, whereas `13a` gives `a & b` and `ab`. Measured on both.

Whitespace handling is identical to `13a`, except that trailing whitespace is dropped before the
passes (so `"7. "` glues like `"7."`).

## 5. `zh`

Measured 2026-11-24 (`tools/zh_reference.py`). The line is stripped of leading and trailing Python
whitespace **first**, and then:

1. A `.` or `,` that does **not** follow a digit is spaced on both sides, and the match consumes both
   characters (so a run of `.`/`,` interacts at its boundaries: `..0` -> `. . 0`, `...0` -> `. . .0`,
   `a..0` -> `a . .0`). This is the `intl` pair-consuming rule with `[.,]` as the whole mark set.
2. Then a `.` or `,` that does **not** precede a digit is spaced on both sides, same consumption.
3. The range table below splits its characters off as single-character tokens, but it must **not**
   decide `.` or `,`: their behaviour is contextual and rules 1-2 are authoritative.
4. A `-` splits when the character before it is a digit (`1-2` -> `1 - 2`, `1-` -> `1 -`) but not
   otherwise (`a-b`, `a-1`, `-1` stay whole).
5. Everything else accumulates into a run: Latin letters, digits, and arrows such as `→` do not
   split, so `"café 12"` -> `café 12` and `3.5` -> `3.5`.

**Corrections to this section.** The earlier statement said "every character in `\p{N}` starts and
ends its own token" and "everything else ... accumulates": both are contradicted by the oracle.
`3.5` stays `3.5` and `1,000` stays `1,000` (so digits do not split individually), while `.`/`,` and
`-` are contextual, not plain accumulators. Measured: 0 wrong on 9,696 battery cases and 13,289
corpus texts.

**CJK ranges** (inclusive, code points), which the implementation must treat as data:

| start | end |
|---|---|
| U+2E80 | U+2EFF |
| U+3000 | U+303F |
| U+3040 | U+30FF |
| U+3400 | U+4DBF |
| U+4E00 | U+9FFF |
| U+F900 | U+FAFF |
| U+FF00 | U+FFEF |
| U+20000 | U+2A6DF |
| U+2A6E0 | U+2A6FF |
| U+2F800 | U+2FA1F |

Observed confirmations from the oracle (`13a`/`intl`/`zh` on edge inputs): `"5.,6"` → `5 . , 6`
for all three; `"x\u200by"` keeps the zero-width space inside the token; `"café 12"` → `café 12`;
`zh` treats `→` as non-CJK (only `13a` leaves it unsplit, `intl` splits it).

**Known defect, reproduced deliberately.** The widely used range table contains the *string*
`u'\u20000'` as an end point, which is a two-character string, not a code point. The consequence is
that the comparison used in practice admits every character from U+2001 up to U+2A6D — arrows,
mathematical operators, currency symbols — as "Chinese". This is observable in the oracle's output
and must be reproduced: the effective range is U+2001–U+2A6D, not U+20000–U+2A6DF.

This is not a licence to invent the defect; it is recorded here because faithful reproduction
requires it, and because a consumer changing it would change scores.

## 6. Lowercase (`--lowercase` / `lowercase=True`)

Lowercasing happens **before** tokenization and means Python's `str.lower()`: full Unicode case
mapping including special casing (final sigma, circled letters, ligatures) as of the interpreter's
Unicode version. Rust's standard library uses a newer Unicode version and will differ on some code
points, so the implementation must match the interpreter's tables or delegate to CPython.

## 7. Whitespace and stripping

`str.strip()` and `str.rstrip()` use the Python whitespace set above. `' '.join(s.split())` uses the
same set. These are not the same as Rust's `char::is_whitespace()` (which includes U+200B-adjacent
behaviour differences and excludes `\x1c`–`\x1f` from some versions) and the difference is observable.

## 8. Acceptance criteria

An implementation of this document is acceptable iff:

1. `tests/test_frozen_oracle.py` passes with zero mismatches over the frozen corpus.
2. The fuzz corpus (200 cases over the adversarial alphabet) tokenizes identically, which exercises
   every quirk above: ampersand escapes, `&lt;`/`&gt;`, `-\n`, zero-width space, `\x1c`–`\x1f`,
   Turkish dotted I, final sigma, astral plane.
3. The Rust unit tests in `crates/swadesh-core` that cover tokenizers pass.

Only then may the previous implementation file be deleted.