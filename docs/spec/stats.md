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

1. **Statistics vector.** `[H, R, c_1, t_1, c_2, t_2, ..., c_N, t_N]` where `R` is the closest
   reference length (below), `c_n` is the count of clipped matching n-grams of order n and `t_n` the
   total number of hypothesis n-grams of order n. `N` is `max_ngram_order`, default 4.
2. **Clipped counts.** For each n-gram `g` of order n, `c_n = Σ_g min(count_H(g), max_i count_{r_i}(g))`.
   The maximum is taken over references *before* clipping against the hypothesis, so a hypothesis
   count of 3 against references of 1 and 2 clips to 2, not to 1.
3. **Closest reference length.** With `H` the hyp length, `R` is the reference length minimising
   `|L - H|`. **On a tie the FIRST reference in order wins** — measured, and the opposite of the
   commonly cited "shorter wins":

   | hyp | references | chosen |
   |---|---|---|
   | 5 | 4, 6 | 4 |
   | 5 | 6, 4 | 6 |
   | 5 | 7, 4 | 7 |
   | 5 | 4, 7 | 4 |

   Ordering matters, so the implementation must preserve reference order.
4. **Empty segments.** A segment with no tokens contributes `[0, R, 0…0]` with all `c_n` and `t_n`
   zero; the scoring stage handles it.
5. **Effective order.** With `effective_order=True`, the scoring stage uses only orders below the
   hyp length. This is a scoring-stage decision, so the statistics stage always fills all `N` pairs.

## 3. chrF statistics

For a hypothesis string and reference strings:

1. **Character sequence.** By default each character is a unit; whitespace characters participate.
   With `whitespace=False`, whitespace is removed **before** n-gram extraction, so a removed space
   does not separate adjacent characters into different n-grams — the characters become adjacent.
2. **Word sequence** (only when `word_order > 0`, i.e. chrF++). Punctuation is removed from the
   word sequence; the removed set is defined in the specification as the character sequence with
   every punctuation character deleted, where "punctuation" is the set the oracle uses (measured
   separately, see the acceptance criteria).
3. **Orders.** Char orders `1..char_order` and, for chrF++, word orders `1..word_order`. Each order
   produces a triple: matching n-grams, hyp total, ref total.
4. **Matching count.** For chrF the per-order match count is again `min(count_H, max_i count_{r_i})`
   summed over n-grams — the same clipping rule as BLEU, over characters or words.

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

1. **Closest-reference-length tie-break: first reference in order wins.** The table above; my draft
   said "shorter wins" and that was wrong. This is why the spec is measured, not remembered.
2. **`whitespace=False` and `lowercase=True` are compatible in either order** for the case tested
   (`"A  B"` vs `"ab"` scores 100.0): whitespace removal happens on the raw text and lowercasing is
   independent, because deleting a space never changes a character's case.
3. **chrF word-order punctuation removal is NOT a simple punctuation strip**: `"a b, c"` against
   `"a b c"` scores 58.26, not 100, so punctuation is retained somewhere in the comparison. The
   exact behaviour must be established by a dedicated probe series before implementing chrF++; until
   then the implementation should follow the existing behaviour and the frozen oracle decides.