# `intl` tokenizer: the measured model

Written **before** the implementation, from black-box measurement of the oracle (sacreBLEU 2.6.0,
`TokenizerV14International`). The oracle was probed only through its public tokenizer output; no
oracle source was read.

This supersedes the earlier "number-absorption rule" in this file. That rule was assembled from
single probes and got three things wrong: it claimed a mark after a word ending in a letter glues
(the probes show the opposite), it could not explain the two-adjacent-quote case, and it had no
account of why `a.2/1` leaves `2/1` glued while `2/1` alone splits. The model below explains all
three as one mechanism, and the reference implementation in `tools/intl_reference.py` matches the
oracle on 58,383 measured cases with zero differences.

## Two layers, measured separately

The oracle has a tokenizer and a metric layer, and they differ on trailing whitespace:

| input | tokenizer | `Metric._preprocess_segment` |
|---|---|---|
| `"7. "` | `7 .` | `7.` |
| `"7."` | `7.` | `7.` |
| `"7  "` | `7` | `7` |

So the **tokenizer** does not strip: pass 3 sees the space after `.` and splits. The **metric
layer** strips trailing whitespace before calling the tokenizer, which is why `"7. "` scores like
`"7."`. A clean-room implementation must keep the layers apart: `swadesh.tokenize` mirrors the
tokenizer, while `_preprocess_segment` mirrors the metric layer.

## The tokenizer model

Given a segment `s`:

1. Pad every Unicode **symbol** (general category `S*`) with a space on both sides.
2. Scan left to right. When a character that is **not a number** (category `N`) is immediately
   followed by **punctuation** (category `P*`), insert a space around both characters and resume
   the scan after the pair.
3. Scan left to right again. When punctuation is immediately followed by a **non-number**, insert a
   space around both characters and resume after the pair.
4. Join the result on Python whitespace with single spaces.

The scan consumes both characters of a match. That single detail is the whole story of the cases
that defeated the earlier rule.

The metric layer is: strip trailing Python whitespace, then the four steps above.

## What the model predicts, and why

* **Punctuation glues to numbers; symbols never do.** `3%4` → `3%4`, `3.5%` → `3.5%`, `9。9u` →
  `9。9u`, `_1` → `_1`, but `$5` → `$ 5`, `¥6F` → `¥ 6F`, `☆3` → `☆ 3`, `→` after a digit splits.
  Punctuation is separated from every non-number neighbour: `a%1` → `a % 1`, `a?/1` → `a ? /1`.
* **Pair consumption.** In `‘’/1` pass 2 pairs `‘` with `’` and resumes at `/`; `/` is not
  followed by punctuation, so `/1` glues: `‘’/1` → `‘ ’ /1`. In `./1` pass 2 pairs `.` with `/`
  and resumes at `1`: `./1` → `. / 1`. Two marks split, two quotes glue, and neither needs a
  special case.
* **`a./1`** → pass 2 pairs `a` with `.`, resumes at `/`, `/1` glues → `a . /1`.
  **`2./9`** → `2` is a number so pass 2 cannot pair it with `.`; `.` pairs with `/` → `2 . / 9`.
  **`a.2/1`** → `a` pairs with `.`, resumes at `2`; `2`/`/` cannot pair (number), `/1` glues →
  `a . 2/1`.
* **Trailing whitespace** is not stripped by the tokenizer: `7. ` → `7 .`, `7.` → `7.`, and
  `a.` → `a .` (pairing). The metric layer strips first, so `7. ` scores as `7.`.
* **`_`** is `Pc`, i.e. punctuation: `_1` glues, `a_1` → `a _ 1`, `1_` glues.
* **ZWJ** (U+200D, category `Cf`) is neither punctuation, symbol nor number, so it is its own
  token between the padded symbols: `👨‍👩` → `👨` `‍` `👩`, exactly as the oracle does.

## Measurement

`tools/intl_reference.py` is the model as executable code. It imports the oracle only to compare
outputs and exits non-zero on any difference. Batteries and results (2026-11-24):

| battery | layer | cases | wrong |
|---|---|---|---|
| fixed probe list (limbs, quotes, digits, trailing space, `_`, ZWJ, emoji) | tokenizer | 70 | 0 |
| random strings over ASCII, all `P`/`S` in U+00A0–U+20FF, CJK punctuation, seed 5 | tokenizer | 6,500 | 0 |
| every `P`/`S` scalar (8,560) plus `_` as `c1` (start of segment) | tokenizer | 8,561 | 0 |
| same scalars as `1c1` (between numbers) | tokenizer | 8,561 | 0 |
| same scalars as `1c` (after a number, end of segment) | tokenizer | 8,561 | 0 |
| corpus: all strings in `tests/oracle/*.jsonl` plus `tests/corpus/model/*.txt` | tokenizer | 13,065 | 0 |
| same corpus | metric | 13,065 | 0 |
| **total** | | **58,383** | **0** |

The exhaustive matrix is the load-bearing evidence for the number rule: across all 8,561
punctuation and symbol scalars, every `P` character glues to an adjacent number in all three
contexts and every `S` character splits in all three, with no other output shape.

For comparison, the committed hand-rolled scanner (before this change) differs from the oracle on
224 of 14,376 texts from the same corpus extraction.

## Status

The implementation follows in the next commit: `crates/swadesh-core/src/tokenize_intl_clean.rs` is
replaced by the passes above. The earlier limb-1 and lead-path trailing-absorption clauses are
subsumed by the model rather than extended.
