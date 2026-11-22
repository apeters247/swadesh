# BLEU and chrF segment statistics — specification

Phase 2 of the clean-room plan. Defines the per-segment sufficient statistics that the score stage
consumes. Written before any implementation, from published definitions and from black-box
measurement of the oracle; no implementation source was read.

**Governing sourcing rule** (from `docs/spec/tokenizers.md`): derive from published prose and
measured oracle behaviour only. sacreBLEU's `_compute_segment_statistics` and any other
implementation are off limits.

Sources: Papineni et al. 2002 (BLEU), Lin & Och 2004 (closest reference length), Post 2018
(sacreBLEU's smoothing methods), Popović 2015 (chrF), and the documented meanings of the options.

## Why statistics, not scores

BLEU and chrF both factor into a per-segment integer phase and a corpus-level scoring phase. Only
the segment phase needs to be fast, and it needs no floating point at all:

- BLEU: per segment, the hyp length, the reference length, and for each n from 1 to N the number of
  clipped matching n-grams and the number of hyp n-grams of that order. Integer pairs.
- chrF: per segment, per n-gram order, the number of matching char n-grams in hyp and ref and the
  totals of each. Integers as well.

If these match, the corpus score is the scoring stage's business. The integer-only property is what
makes exactness tractable: there is nothing to round.

## 1. Tokenisation (input to both)

The segment is tokenised by the configured tokenizer (`none`, `13a`, `intl`, `zh`, `char`), after
`str.rstrip()`. The result is a sequence of tokens split on Python whitespace. See
`docs/spec/tokenizers.md`; the four implementations there verify at zero on 13,455 corpus lines.

## 2. BLEU statistics

For a hypothesis token sequence `h` of length `H` and reference sequences `r_1..r_k`:

1. **Statistics vector.** `[H, R, c_1..c_N, t_1..t_N]` where `R` is the closest reference length
   (below), `c_n` the count of clipped matching n-grams of order n and `t_n` the total number of
   hypothesis n-grams of order n, `N` = `max_ngram_order` (default 4).

   **Not interleaved.** Counts come first as a block, then totals as a block; the scoring stage
   indexes the two halves separately. My first draft said `c_1, t_1, c_2, t_2…`, and the differential
   run showed every one of 13,455 rows differing because of it.
2. **Clipped counts.** For each n-gram `g` of order n, `c_n = Σ_g min(count_H(g), max_i count_{r_i}(g))`.
   The maximum is taken over references *before* clipping against the hypothesis, so a hypothesis
   count of 3 against references of 1 and 2 clips to 2, not to 1.
3. **Closest reference length.** With `H` the hyp length, `R` is the reference length minimising
   `|L - H|`. **On a tie the SHORTER reference wins** — provided the references arrive as separate
   streams, which is the corpus layout:

   | hyp | references (separate streams) | chosen |
   |---|---|---|
   | 5 | `[7]`, `[3]` | 3 |
   | 5 | `[3]`, `[7]` | 3 |
   | 10 | `[12]`, `[8]`, `[20]` | 8 |

   **A caveat that cost a session to find:** several references packed into ONE stream behave
   differently — `[r6, r2]` for hyp 4 gives 6, the first element. Two standard layouts therefore
   disagree on ties, and a probe written with one layout will produce a rule that breaks the other.

4. **Empty segments.** A segment with no tokens contributes `[0, R, 0…0]` with all `c_n` and `t_n`
   zero; the scoring stage handles it.
5. **Effective order.** With `effective_order=True`, the scoring stage uses only orders below the
   hyp length. This is a scoring-stage decision, so the statistics stage always fills all `N` pairs.

## 3. chrF statistics

For a hypothesis string and reference strings:

1. **Character sequence.** By default each character is a unit; whitespace characters participate.
   With `whitespace=False`, whitespace is removed **before** n-gram extraction, so a removed space
   does not separate adjacent characters into different n-grams — the characters become adjacent.
2. **Word sequence** (only when `word_order > 0`, i.e. chrF++). **Punctuation is NOT stripped.**
   Words are the whitespace-delimited tokens with punctuation left intact. Measured with
   `word_order=3, char_order=1`, so the word component dominates:

   | hypothesis | reference | score | reading |
   |---|---|---|---|
   | `a-b c` | `ab c` | 48.08 | punctuation retained: `a-b` ≠ `ab` |
   | `a-b c` | `a-b c` | 100.00 | |
   | `a , b` | `a b` | 60.61 | a standalone `,` counts as a word |
   | `a  b` | `a b` | 100.00 | whitespace runs collapse, as for words elsewhere |

   Two earlier drafts of this document guessed the opposite (punctuation stripped), twice.
   The measurement above is the specification.
3. **Orders.** Char orders `1..char_order` and, for chrF++, word orders `1..word_order`. **Each
   order produces the triple `[hyp_total, ref_total, matches]`** — hypothesis total first, then
   reference total, then matches. Measured on a WMT22 segment with a 194-character hypothesis and a
   202-character reference, whose first order reports `[194, 202, 194]`.

   Two earlier readings of this triple in this session were wrong because the test strings happened
   to have the lengths I expected; measure with a segment whose lengths plainly differ.
4. **Matching count.** For chrF the per-order match count is `min(count_H, count_r)` summed over
   n-grams, per reference. **With several references the row kept is the one whose f-score is
   highest** (the first strictly better), NOT a merge of counts across references. Measured: a
   hypothesis matching one of two references perfectly scores 100.0 against `[good, bad]` and
   against `[bad, good]` alike.

## 4. What the scoring stage does with these (for orientation, not implemented here)

BLEU combines the clipped counts into precisions and applies brevity penalty from `H` and `R`;
chrF builds precision and recall per order and a harmonic mean with `beta`. Both are the scoring
stage's concern and are specified in `docs/spec/scoring.md` when Phase 2 reaches them.

## 5. Acceptance criteria

An implementation is acceptable iff:

1. `tests/test_frozen_oracle.py` passes with zero mismatches: those fixtures carry the expected
   sufficient statistics for every corpus line and fuzz case, so statistics exactness is checked
   directly, not only through scores.
2. The Rust unit tests `np2_scoring.rs`, `resample_serial.rs` and `coverage_gaps.rs` pass.
3. A differential run against the oracle over the seven WMT22 pairs reports zero differing
   statistics vectors across all five tokenizers.

## Questions settled by measurement (2026-11-22)

1. **Closest-reference-length tie-break: the SHORTER reference wins, with separate streams.** Two
   drafts of this document were wrong in opposite directions ("shorter wins", then "first in
   order"), and a third attempt to "fix" the implementation broke the parity suite. The table in
   section 2.3 is the settled answer, including the single-stream caveat.
2. **`whitespace=False` and `lowercase=True` are compatible in either order** for the case tested
   (`"A  B"` vs `"ab"` scores 100.0): whitespace removal happens on the raw text and lowercasing is
   independent, because deleting a space never changes a character's case.
3. **chrF word order: punctuation is retained, words are whitespace tokens.** Settled with
   `word_order=3, char_order=1` so the word component dominates; the table in section 3.2 is the
   evidence. Two earlier drafts of this document asserted the opposite.
