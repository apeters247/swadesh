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

Same skeleton as `13a`, with the internationalised class for rule A and **no** `&amp;` rule:

1. Surround with spaces every character in the class `\p{P} \p{S} \p{N}`.
2. Attach a trailing `.`/`,` to the preceding token unless it ends in a digit; detach a leading
   `.`/`,` from a following non-digit.
3. No separate `%` rule and no entity decoding. `%` is split by rule 1 only where rule 1
   applies; observed: `"100%"` stays `100%` under `intl` but becomes `100 %` under `13a`, because
   `%` is in `13a`'s rule-A class but not in `intl`'s.
   (`→` behaves the other way: not split by `13a`, split by `intl`.) The two classes therefore
   differ, and the implementation must not treat `intl` as `\p{P}\p{S}\p{N}` wholesale.
4. No entity decoding and no `<skipped>`/`-\n` removal: `"a&amp;b"` becomes `a & amp ; b`
   (the `;` splits by rule 1) and `"a<skipped>b"` becomes `a < skipped > b`, whereas `13a` gives
   `a & b` and `ab`. Measured on both.

Whitespace handling is identical to `13a`.

The difference between `13a` and `intl` is therefore: `intl` splits on numbers and all
punctuation/symbols in one pass and leaves ampersand escapes alone.

## 5. `zh`

Character-level segmentation for Chinese, with alphanumeric runs kept whole:

1. Split the line into runs separated by Python whitespace.
2. Within a run, every character in the **CJK ranges** below starts and ends its own token.
3. Every character in `\p{N}` likewise.
4. Everything else (Latin letters, punctuation, and arrows such as `→`) accumulates into a token:
   `"你 好 abc 世 界"` tokenizes to `你 好 abc 世 界`. No entity decoding, no `<skipped>` removal.

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