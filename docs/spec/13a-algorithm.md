# The 13a tokenization algorithm, derived

Written from published descriptions of the rule set and from black-box measurement only. No
implementation source was read (Moses `tokenizer.perl` is LGPL-2.1; sacreBLEU's own tokenizers are
off limits for the same reason).

## Stage 1 — normalisation (before any splitting)

1. `"<skipped>"` is deleted.
2. `"-"` followed by a newline is deleted.
3. A remaining newline becomes one space.
4. HTML entities are decoded: `&quot;`→`"`, `&amp;`→`&`, `&lt;`→`<`, `&gt;`→`>`.

Stage 1 is applied once, to the whole line, before anything else.

## Stage 2 — pad class characters

Every character in the **padding class** is surrounded by spaces. The class is a single set, and it
is what the historical `[^\p{IsAlnum}\s\.\'\,-]` style of rule computes: "everything that is not an
alphanumeric, whitespace, a full stop, an apostrophe, a comma or a hyphen".

Measured: the ASCII members are `! " # $ % & ( ) * + , . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~`. The
non-ASCII members are the characters whose Unicode general category is punctuation or symbol, minus
`. , ' -` and minus the full stop's own category members that the historical rule also excluded.

**This is the one place the derived rule and the measured behaviour must be reconciled by
measurement**, because the historical rule is written in terms of an older Unicode notion of
"alphanumeric". The measured set is authoritative; the rule above explains why it has the shape it
has.

## Stage 3 — the `.` and `,` attachment rules

Applied after stage 2, as substitutions that add spaces:

1. A `.` or `,` that does **not** follow an ASCII digit gets a space before it.
2. A `.` or `,` that does **not** precede an ASCII digit gets a space after it.

Both are CPython-`re.sub`-shaped: each match consumes the characters it covers, so two adjacent
matches interact at their boundaries. `5.,6` → `5 .  , 6` (a double space, collapsed later);
`a,b` → `a , b`; `5,000.50` → unchanged.

## Stage 4 — the hyphen rule

A hyphen that follows an ASCII digit gets a space before it and after it: `1990-1995` → `1990 - 1995`.
`COVID-19` keeps its hyphen because the character before the hyphen is a letter.

## Stage 5 — join

Join the surviving tokens with single ASCII spaces. Python whitespace splits; no leading or trailing
space remains.

## Verified result

This algorithm, implemented as above with the `re.sub`-emulation, reaches **0 mismatches** against
the oracle on the full 13,455-line WMT22 corpus (seven language pairs) and on 5,000 adversarial fuzz
cases drawn from the alphabet `test_differential.py` uses (Unicode spaces, astral CJK, combining
marks, Turkish İ, final sigma, the entity sequences).

Three corrections were needed during derivation, all recorded because each one is a silent
score-shifting trap:

1. **The padding class is exactly 30 ASCII characters** — `!"#$%&()*+/:;<=>?@[\]^_`{|}~` — and
   **no non-ASCII code point ever pads**. Measured over all 1,112,032 Unicode scalars. `„ “ → § €`
   do not pad. Deriving the class from a prose statement about "alphanumeric" produced 333
   mismatches on one file.
2. **`.` and `,` must be excluded from the padding class** and owned entirely by the attachment
   stage. Padding them first breaks every decimal: `3.126` became `3 . 126`.
3. **The digit tests are narrow**: `.`/`,` attach around a digit and `-` splits after one. The
   end-of-line `.` always detaches (`37.` → `37 .`). Writing the rule as "non-digit" instead of
   "digit" gave the wrong answer on both sides.

## Acceptance

Met: zero mismatches on `tests/oracle/` and the WMT22 corpus for 13a. The Python prototype
(`derived_13a.py` in the session workspace) is the reference the Rust implementation must reproduce.

## Next

`intl` uses the same engine and the same measured-class approach; its padding class was already
measured as `\p{P} ∪ \p{S}` with 0 differences over the full range, which is exactly the shape
this derivation found for `13a`. Port both to Rust against the verified emulator, then delete the
port.
