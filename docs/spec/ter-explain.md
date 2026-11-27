# `TER.explain()`: an additive inspection surface

Status: additive API of the native TER kernel. Written 2026-11-27 from the implementation's own
measurements (`crates/swadesh-core/src/ter_clean.rs`, kernel `ter_explain`), not from any oracle
source. Nothing in this document changes a score.

## Why it exists

TER is a single number that hides three decisions: how many word blocks the TERCOM shift search
moved, how far the shifted hypothesis is from the reference after those moves, and which reference
won when a segment has several. `sentence_score` reports only the ratio. A researcher debugging a
`0.42` needs the parts, and the parts are already computed inside the search - they are simply
dropped before the ratio is formed. `explain()` returns them instead of dropping them.

## What it is (and is not)

* It is the **same search** as `corpus_score` / `sentence_scores`: the same tokenizer, the same
  `best_shift` candidate loop, the same `MAX_SHIFT_DIST`/`MAX_SHIFT_SIZE`/`MAX_CANDIDATES` limits,
  the same final banded Levenshtein distance. It is not a second implementation of TER.
* `ter` is recomputed from `edits` and `ref_len` with the metric's own `_score_from_row` rule, so it
  is exactly the number `sentence_score(...).score` returns, including the empty-reference fallbacks.
* It is **additive**. `TER`'s constructor and `corpus_score`/`sentence_score` signatures are
  unchanged, so the parity harness that compares constructor signatures against sacreBLEU never sees
  it. `explain` is a new method, not a new keyword.
* There is **no Python fallback**. The numbers describe the native search; a Python re-derivation of
  the shift loop would be a second implementation and could drift. Where the native extension is
  missing, `explain` is missing.

## The per-segment record

One dict per segment, in input order:

| key | meaning |
|---|---|
| `ter` | the segment's TER in percent, identical to `sentence_scores(...)[i]` |
| `edits` | `shifts + distance`; the numerator of TER for this segment |
| `shifts` | how many word-block shifts the search accepted |
| `distance` | the final banded Levenshtein distance, `edits - shifts` |
| `ref_len` | mean reference length in tokens over **all** references: TER's denominator |
| `best_ref` | index of the reference the `edits` came from (the fewest edits; ties keep the first) |
| `best_ref_len` | that reference's token count |
| `hyp_len` | hypothesis token count |
| `n_refs` | how many references the segment had |
| `counts` | the final alignment's `matches`/`substitutions`/`insertions`/`deletions` |
| `accepted_shifts` | one `{hyp_start, ref_start, len}` per accepted shift |

## Invariants (each holds by construction, and is asserted by a test)

1. `edits == shifts + distance`.
2. `distance == substitutions + insertions + deletions` (Levenshtein distance counts the non-matches
   of the final alignment).
3. `len(accepted_shifts) == shifts`.
4. `ter == TER().sentence_scores(hyps, refs)[i]` and `== TER().sentence_score(hyps[i], [refs[j][i]]).score`.
5. With several references, `ref_len` is their **mean** length while `edits` and `best_ref` come from
   the **minimum-edit** reference - the same asymmetry `ter_segment` documents (spec `ter.md` 5).
6. `best_ref` is the argmin over references of the single-reference `edits`, ties to the first.

## Shift coordinates (the one subtlety)

An accepted shift is reported in **working coordinates**: positions in the hypothesis as the earlier
shifts have already rearranged it. The `k`-th entry is relative to the sequence after the first
`k - 1` shifts, not to the original tokenization, because that is the sequence the search actually
operated on and the only coordinates in which the decision is meaningful. The search is
deterministic, so the same pair always yields the same shifts in the same order.

## Measured example (the reason the field exists)

`"on the mat the cat sat"` against `"the cat sat on the mat"` scores TER `16.7` - `(1 shift + 0
edits) / 6`. `explain()` reports `edits: 1, shifts: 1, distance: 0, counts: {matches: 6, ...}`, which
is the whole story of the number: one block moved, nothing else wrong.
