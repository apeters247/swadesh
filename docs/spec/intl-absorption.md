# intl tokenizer: the number-absorption rule

Written before the implementation, from black-box measurement of the oracle (sacreBLEU 2.6.0,
`TokenizerV14International`). The implementation is **not** in the tree: see "Status".

## The question

When a mark sits immediately before a run of digits, does the mark join the number or stand alone?

    '?1'      -> '?1'        glued
    'x?1'     -> 'x ? 1'     split
    'a./1'    -> 'a . /1'    glued
    '2./9'    -> '2 . / 9'   split
    '?𝟎符'    -> '?𝟎符'       glued
    '‘’/1'    -> '‘ ’ /1'    glued

## Measured facts

**One mark at position 0.** Every ASCII mark glues except `$ + < = > ^ ` | ~`, all of which are
Unicode `\p{S}`. Every `\p{P}` mark glues, ASCII and non-ASCII alike (`‘ ’ “ ” « » （ 、`). This is
exactly the set `splits_after_digit` already holds.

**One mark after a word.** Glues only when the word ends in a letter: `a./1`, `ab./1`. Splits after
a digit (`2./9`, `22./9`, `x2./9`, `a2./9`, `1./2`), and splits after a letter for every other mark
(`a?1`, `a,1`, `a/1`, `a'1`, `a)1`, `a-1`, `a’1`, `x!Ⅷ`).

**Two marks in a row.** `'?1`, `‘/1`, `’/1`, `./1`, `!/1`, `?/1`, `//1`, `,/1`, `&/1`, `@/1`,
`(/1`, `\/1`, `_/1` all split.

**Except two adjacent quotes.** `‘’/1`, `’’/1`, `‘‘/1`, `''/1`, `'’/1`, `‘'/1` all glue. This is the
one case that resisted a predicate.

**After a digit** the mark never glues, in any position.

## Rule

A mark joins a following digit run when the mark is not `\p{S}` **and** one of:

1. it is the first character of the segment, or
2. it follows a word that ends in a letter, or
3. it is the second of two adjacent quote marks at the start of a word.

Everything else splits.

## Status: limb 1 implemented; limbs 2 and 3 pending

Limb 1 shipped on 2026-11-24 as a single guarded clause in `tokenize_intl_clean.rs`:
`i == 0 && (is_punct(c) || c == '_')`, under the existing `\p{S}` exclusions and the outer
"followed by a digit" test. Measured: the fixed probe set went from 12 of 29 wrong to 3 of 29, and a
3,000-case random sweep over `\p{P}`/`\p{N}`/`\p{S}`/letter characters from 611 to 372. The suite
counts did not move, because each of those hypothesis tests reports its first counterexample and the
BLEU one's counterexample simply moved on to limb 2.

Still open: limb 2 (a mark after a word that ends in a letter), limb 3 (the quote pairs), and the
trailing-punctuation rule in the lead path (`?1%` glues in the oracle, `?1` `%` here).

The rest of this section is the original attempt record, kept because it names the blocker.

I attempted this rule nine times on 2026-11-22. Attempts 1-5 are recorded in the lab notebook; each
fixed the cases it was measured against and regressed others, and the final state was worse than the
committed one, so the tokenizer is back to its committed form. The blocker is limb 3: the quote-pair
case is the only one whose predicate I could not derive, and without it the other two limbs produce
`‘’/1` -> `‘ ’ / 1` where the oracle gives `‘ ’ /1`.

**What the corpus needs.** The remaining oracle failure (`fuzz42:BLEU intl`) needs limbs 1 and 2
only: `?𝟎符` (limb 1) and `a./1` (limb 2). Limb 3 appears in `fuzz66`, which the committed
tokenizer already passes. So a partial implementation covering limbs 1 and 2 would close the open
case — the reason I did not ship it is that my limb-1 implementation also re-admitted `./1`, `’/1`
and `!/1` through a path I had not separated from the quote case.

**Next step.** Determine why two adjacent quotes behave differently from other two-mark chains;
then implement all three limbs together and re-run the 1,177-text sweep.