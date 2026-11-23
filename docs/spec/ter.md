# TER: derived specification

This document defines the behaviour of `crates/swadesh-core/src/ter.rs` (Phase 3). It was written
**before** the implementation, from the published literature and from black-box measurement of the
oracle (sacreBLEU 2.6.0, called but never read). Where a rule below was not derivable from the
published algorithm, the sentence says so and cites the probe that established it.

**Sourcing rule.** Same as `tokenizers.md`: only the papers and oracle calls. `tokenizer.perl` is
LGPL-2.1 and was not read or transcribed; the tokenization rules here were read off the oracle's
behaviour alone.

## 1. What TER is

Translation Edit Rate (Snover et al. 2006): the minimum number of edits, expressed per reference
word, that turns the hypothesis into the reference, where an edit is an insertion, a deletion or a
substitution of one word, **or a shift of a contiguous block of words to a different place**.

    TER = (insertions + deletions + substitutions + shifts) / average reference length

The plain edit-distance part is standard. The shifts are what make TER different from WER: moving a
phrase is one edit rather than `2 * length` insert/delete pairs. TERCOM's search does not solve the
shift placement optimally (it is a greedy beam search), so *which* shifts are found, and the
tie-breaking inside the beam, are part of the observable behaviour and must be reproduced exactly.

## 2. Public surface

```rust
pub struct TerConfig {
    pub normalized: bool,
    pub no_punct: bool,
    pub asian_support: bool,
    pub case_sensitive: bool,
}

pub fn ter_segment(cfg: &TerConfig, hyp: &str, refs: &[&str]) -> Result<[f64; 2], TerError>;
```

`ter_segment` returns `[num_edits, avg_ref_len]`. `avg_ref_len` is the **mean** of the reference
lengths (not the closest reference length, unlike BLEU): with references of 10 and 20 words it is
15, so TER can exceed 1.0. An empty `refs` is an error, not a perfect score.

The corpus score is `sum(num_edits) / sum(avg_ref_len)`, which is TER weighted by reference length.

## 3. Tokenization

Two tokenizers, selected by `normalized`.

### 3.1 `normalized = false` (the default; TERCOM-style)

1. Lowercase unless `case_sensitive`.
2. Replace every Python whitespace character with a single space, collapse runs, strip the ends.
3. Split on spaces. **No punctuation is separated.**

Measured: `"a.b"` stays `"a.b"`, `"a;b"` stays `"a;b"`, `"end."` stays `"end."`. Only whitespace
splits. With `no_punct = true`, one more step runs before the split: a fixed set of ASCII
punctuation is **deleted** (not separated), so `"a.b"` becomes `"ab"` and `"Hello, world!"` becomes
`"hello world"`. The set is exactly `!` `"` `(` `)` `,` `.` `:` `;` `?` — nine characters. The
other ASCII punctuation (`#` `$` `%` `&` `'` `*` `+` `-` `/` `<` `=` `>` `@` `[` `\` `]` `^` `_`
`` ` `` `{` `|` `}` `~`) survives and stays attached to its neighbours. Measured by sweeping
`string.punctuation`: `a&b` stays `a&b`, `a#b` stays `a#b`. This is why the entity step of
section 3.2 matters: the `&` it produces survives `no_punct`.

### 3.2 `normalized = true` (the sacreBLEU "normalized" tokenizer)

1. Lowercase unless `case_sensitive`.
2. **Unescape XML entities.** Four sequential global replacements, in this order: `&amp;` → `&`,
   `&lt;` → `<`, `&gt;` → `>`, `&quot;` → `"`. The pass runs after lowercasing, so the four
   entities are covered case-insensitively when `case_sensitive` is false; with
   `case_sensitive = true` only the lowercase spellings are replaced. Measured on
   `_preprocess_segment`: `&amp;` → `&`, `&AMP;` → `&` (default) but `&AMP;` → `& AMP ;`
   (`case_sensitive`), `&amp;lt;` → `<`, `&amp;amp;` → `& amp ;`, `&amp;AMP;` → `& amp ;`. The
   order matters: `&amp;` runs first, so `&amp;lt;` collapses to `<` while `&amp;amp;` leaves the
   second `amp;` for the tokenizer to split. The default tokenizer (section 3.1) does not unescape
   entities at all: `&amp;` stays `&amp;`.
3. Normalise headline quotes and apostrophes: `“ ” „ ‟ « » ‘ ’ ‚ ‛` are mapped so that the
   *apostrophe-like* marks become `'` and the quote-like marks stay themselves. Measured: `“x”`
   survives as one token `“x”`, while `"x"` becomes `" x "`. The distinction matters: `"` (U+0022)
   is in the separating set, `“ ` `”` (U+201C/201D) are not.
4. Python whitespace → single space, collapse, strip.
5. **Protect numbers.** A maximal run of ASCII digits, optionally with internal `,` or `.` between
   digits (`3.14`, `1,000`, `1.2.3`), is a single token and is not split at its internal marks.
   Measured: `3.14` stays, `1-2-3` splits (hyphens are *not* protected inside a digit run), `a.14`
   splits, `.3.14` keeps the digits together but leaves the leading dot separate.
6. **Protect letters-hyphen.** A hyphen with a letter-like character immediately on both sides is
   part of the word: `a-b` stays, `12-13` splits, `a-13` stays, `13-a` splits, `1-x` splits, `x-1`
   stays. So the condition is "both neighbours are non-space, non-digit", not "both are letters":
   measured `é-1` stays and `a-é` stays. Treat it as: a hyphen is protected when the character
   before it is not a space or digit and the character after it is not a space or digit.
7. **Separate the remaining ASCII punctuation.** Every ASCII character in `!` `"` `#` `$` `%`
   `&` `(` `)` `*` `+` `,` `/` `:` `;` `<` `=` `>` `?` `@` `[` `\` `]` `^` `_` `` ` `` `{` `|`
   `}` `~` · U+0021..U+007E is put on its own token, **except**:
   - `'` (U+0027), which never separates (`don't` stays one word), and
   - `.` and `,` protected by rule 5, and `-` protected by rule 6.

   None of the non-ASCII punctuation separates: `—`, `–`, `«`, `»`, `、`, `。`, `，`, `…`, `§`, `°`,
   `¡`, `¿` all stay attached to their neighbours. This is a real asymmetry of the oracle and is
   reproduced deliberately.

   The class is exactly ASCII, code points `0x00`–`0x7F`. Measured: `0x7E` (`~`) separates, `0x7F`
   (DEL) and `0x80` do not.

8. Collapse spaces again and split.

With `no_punct = true`, the deletion set of section 3.1 runs first, but **in normalized mode a
deleted mark is replaced by a space, not removed**, so it still separates tokens. The exception is
`.` and `,` between ASCII digits: those are part of the protected number token (step 5), so deleting
them joins the digits. Measured on `_preprocess_segment`:

| input | `no_punct` | `normalized, no_punct` |
|---|---|---|
| `a?b` | `ab` | `a b` |
| `a?b?c` | `abc` | `a b c` |
| `a.b` | `ab` | `a b` |
| `2.2` | `22` | `22` |
| `1,000` | `1000` | `1000` |
| `12.34.56` | `123456` | `123456` |
| `2?2` | `22` | `2 2` |
| `a (b) c` | `a b c` | `a b c` |
| `a[b]c` | `a[b]c` | `a [ b ] c` |

The default mode removes every mark outright, so `a?b` stays one token; the normalized mode leaves a
separator, so it becomes two. `[` and `]` are never deleted in either mode. The same rule applies to
the fullwidth forms when `asian_support` is on (section 3.3.2): `a。b` becomes `ab` in the default
mode and `a b` in normalized mode, and `2。2` becomes `2 2` in both — only the ASCII `.` and `,`
join digits.

### 3.3 East Asian support (`asian_support = true`)

The flag does two things, and they are gated differently. The per-character split only happens in
**normalized** mode; the extra punctuation set for `no_punct` applies in both modes. The original
measurement missed the first gate, and the implementation split per character in both modes.

#### 3.3.1 The per-character split requires `normalized = true`

With `normalized = false`, `asian_support = true` is a no-op for the split: the segment is tokenized
exactly as with the flag off. Measured on 400 random segments mixing CJK and ASCII
(`你 好 日 本 語 。 ！ ？ 、 a b c 1 2`), with `no_punct = false`: zero score differences between
`TER(asian_support=True)` and `TER(asian_support=False)` in sacreBLEU 2.6.0. With `normalized = true`
the flag changes 207 of 300 random segments. The minimal pair is `你好` against `你好吗`:

| flags | sacreBLEU | split per character |
|---|---|---|
| `asian_support=True` | 100.0 | no |
| `asian_support=True, normalized=True` | 33.3 | yes |

So the condition is `normalized && asian_support && any(is_asian)`, not `asian_support && any(is_asian)`.

#### 3.3.2 `no_punct` uses the fullwidth set in both modes

`no_punct = true` with `asian_support = true` deletes the fullwidth forms even when `normalized =
false`. Measured on `_preprocess_segment`:

| input | `no_punct` | `no_punct, asian_support` | `no_punct, asian_support, normalized` |
|---|---|---|---|
| `a。b` | `a。b` | `ab` | `a b` |
| `a（b）c` | `a（b）c` | `abc` | `a b c` |
| `你好。世界` | `你好。世界` | `你好世界` | `你 好 世 界` |

So the `asian` flag passed to the punctuation stripper is `asian_support && any(is_asian)`, without
the `normalized` gate; only the per-character split and its whitespace exemption carry that gate.

#### 3.3.3 The ranges

Applied when any character of the segment falls in one of these ranges; then **every character of
the Asian ranges becomes its own token**, and runs of spaces are dropped so that `中文。` becomes
`中 文 。`.

The ranges below were found by exhaustive scan for the exact boundaries (each code point in
`0x2E00..0x3100`, `0x3100..0x3200`, `0x3200..0x3500`, `0x3400..0x4DC0`, `0x9FF0..0xA050`,
`0xF8F0..0xFB10`, `0xFE00..0xFF70`, `0xFF60..0xFFF0` was tested individually):

| range | content |
|---|---|
| U+2E80 – U+2EFF | CJK radicals supplement |
| U+3001 – U+3002 | CJK punctuation: `、` `。` |
| U+3008 – U+3011 | CJK brackets: `《` `》` `「` `」` `『` `』` `【` `】` |
| U+3014 – U+301F | CJK brackets, continued |
| U+30FB | katakana middle dot |
| U+31C0 – U+31EF | CJK strokes |
| U+3200 – U+33FF | enclosed CJK, CJK compatibility |
| U+3400 – U+4DBF | CJK extension A |
| U+4E00 – U+9FFF | CJK unified ideographs |
| U+F900 – U+FAFF | CJK compatibility ideographs |
| U+FE30 – U+FE4F | CJK compatibility forms |
| U+FF01, U+FF02, U+FF08 – U+FF09, U+FF0C, U+FF0E, U+FF1A, U+FF1B, U+FF1F | fullwidth punctuation, individually |
| U+FF61 – U+FF65 | halfwidth punctuation (ideographic full stop and friends) |

Notable **exclusions**, each measured: U+2F00–U+2FDF (Kangxi radicals) do **not** split; U+3000
(the ideographic space) does not; U+3003–U+3007 do not; the Hiragana and Katakana blocks
(U+3040–U+30FF except U+30FB) and Hangul (U+AC00–U+D7A3) do **not** split, so `アイウ` stays one
token while `日本語。` splits because of the `。`; fullwidth digits (U+FF10–U+FF19) do not split.

The presence of any **one** of these characters switches the whole segment to per-character
splitting; it is not applied per character.

The punctuation subset is exactly 39 code points: the 128 radical code points of U+2E80–U+2EFF
split per character but are not deleted by `no_punct`. A second exhaustive scan of
`0x2E00..0x3100` and `0xFF00..0xFFF0`, comparing `_preprocess_segment` with and without
`asian_support`, found the split set at 167 code points (128 radicals plus 39 punctuation) and the
`no_punct` deletion set at exactly those 39 punctuation code points. U+FF1B `；` is in both sets;
U+FF0D `－` is in neither.

`no_punct` in Asian mode deletes the fullwidth forms (`，` `。` `？` `！` and the CJK brackets) in
addition to the ASCII set.

## 4. Edit distance with shifts

### 4.1 The edit distance

Standard Levenshtein matrix over word ids, costs 1 each for insert, delete, substitute. The matrix
is recomputed for the reference against each shifted hypothesis, but only the **last row** is
needed when no trace is requested, so an implementation may keep just that row. When the trace is
needed (for the alignment that becomes the next shift candidate search), each cell stores one
operation byte: `i`nsert, `d`elete, `s`ubstitute, ` ` (match), `x` (undefined).

**Tie-breaking inside a cell** is observable and must match: when several predecessors reach the
same minimal cost the operation recorded is chosen in a fixed order, measured as
diagonal (match/substitute) first, then insertion, then deletion. Implementations that pick a
different cell in a tie produce different alignments, hence different shift candidates, hence a
different TER.

### 4.2 Shifts

A shift takes a contiguous block `[start, start+len)` of the hypothesis and moves it so that it
begins at position `target`. The block must be found in the reference at `target`:

    hyp[start .. start+len] == ref[target .. target+len]

Constraints, all measured as the point where the oracle stops producing a better TER:

* `1 <= len <= 50`. **Corrected at measurement time:** I first wrote the literature's `len <= 10`,
  and the probe refuted it — a block of 50 words shifts as a single edit, 51 does not. So the length
  cap is 50, the same number as the distance cap. (Moving the same block by 50 positions also
  shifts; by 51 it costs 2 instead of 1.)
* The move distance `|target - start|` must be at most 50 and non-zero.
* A shift costs **1**, regardless of block length.
* A shift that does not reduce the edit distance is not taken: the search only accepts a shift whose
  resulting cost is strictly below the previous best.
* The search is greedy and bounded: at most **25** candidate shifts are examined per iteration
  (`BEAM_WIDTH`), and at most **1,000** candidate pairs are collected for one iteration
  (`MAX_SHIFT_CANDIDATES`). When there are more candidates than the cap the oracle stops collecting
  early, so *which* candidates are seen depends on iteration order — that order is part of the
  behaviour and must be reproduced: the oracle walks the alignment left to right, pairing each
  hypothesis position with the reference positions it aligns to.

Each accepted shift adds 1 to the edit count and is applied to the hypothesis before the next
iteration. Iteration ends when no shift reduces the cost, or when 1,000 candidate evaluations have
been made in total across the whole segment (the counter is deliberately not reset between
iterations — measured, §7).

### 4.3 Counting

`num_edits = edit_distance(after final shift) + number_of_shifts`. The final edit distance is
computed against the final hypothesis, so substitutions caused by a shift are included.

An empty reference has reference length 0 and contributes the hypothesis length as edits, with no
shift search at all; the corpus aggregation then divides by a ref length that may be 0.

## 5. Segment statistics and aggregation

For each segment, with hypothesis and the non-missing references:

1. Tokenize the hypothesis and every reference with the configured tokenizer.
2. `num_edits` is the **minimum** over references of the per-reference edit count, all against the
   same hypothesis: the best reference wins.
3. `avg_ref_len` is the **mean** of the reference lengths. The asymmetry is the oracle's and was
   measured: `a b c` against `[a b c, a b c d e]` reports `[0, 4.0]`, not `[1, 4]` (a mean of the
   edits) and not `[0, 3]` (the best reference's length).

**Corrected at implementation time.** I first wrote "mean over references" for the edits, which
the segment fixtures rejected on 138 of 700 rows. `num_edits` is therefore integral; corpus TER is
`sum(num_edits) / sum(avg_ref_len)` over segments.

**Measured: the reference path preprocesses each reference twice.** The reference implementation
runs the tokenizer once when references are cached and once when they are extracted. That is only
observable where the tokenizer is not idempotent. It is idempotent on the apostrophe-heavy inputs
first probed (`dont''s`, `the 's`, `44 -0038' s`), but **not** on a possessive followed by a lone
`.` or `,`: `a's.` → `a's .` → `a 's .`, and `a's,` → `a's ,` → `a 's ,`. The second pass splits the
`'s` that the first pass left attached. So the double preprocessing is observable for that narrow
class. The native kernel currently preprocesses once and therefore disagrees with the oracle on
`a's.` and `a's,` (reference length 2 against the oracle's 3, measured 2026-11-22). That gap is
recorded in the lab notebook and not yet closed; the fixtures (§6) enforce the outcome, so a
double-pass implementation would be accepted without changing them.

## 6. Verification plan

1. `tools/measure_ter.py` probes the oracle for every rule above and writes
   `tests/oracle/ter_probe.jsonl` (tokenization strings, segment statistics as `float.hex()`).
   It carries 1,135 records: five configurations × 80 tokenization inputs and × 120 random
   shift-heavy segments plus block-length and move-distance boundaries.
2. The clean-room tokenizer and segment statistics are compared against that file on every corpus
   segment (WMT22, all seven pairs) and on adversarial inputs: long repeated blocks, a block of 50
   and of 51, distances 49/50/51, ties in the matrix, empty segments.
3. Corpus scores and per-sentence scores replayed through `tests/test_frozen_oracle.py`.

## 7. Findings recorded during fixture generation

* The shift length cap is 50, not the literature's 10 (§4.2). Bisected: a 50-word block shifts as
  one edit, 51 costs 6.
* The two caps interact in a way worth stating: a 50-word block **and** a 50-position move are each
  individually permitted, so the limits are not a combined budget.
* A cap of 10 per block still reaches the same total for the pure-shift case, because the search
  iterates and can chain shifts. So the cap changes *which* shifts are found, not necessarily the
  final cost — it is a search-efficiency parameter, and §6 tests the outcome rather than the cap.
* The candidate counter is not reset per iteration. This is a hypothesis from the observed
  behaviour of the published loop; §6 tests it rather than assuming it, and it only differs on
  segments that evaluate many candidates.
* Whether the shift search ever accepts a block that overlaps the previous one is not stated in the
  literature; the tests decide it empirically.# TER: the beam edit distance

Appended to `docs/spec/ter.md` section 4.1 after measuring why the clean-room TER disagrees with the
oracle on long-reference segments.

## Why a full matrix is wrong

The clean-room TER computed a full Levenshtein matrix. On `real-nxkd1-k-q5` segment 136 (36
hypothesis words against an 87-word reference) the full matrix gives **86**, whose correctness is
confirmed by an independent reference implementation, and the oracle reports **87**.

The oracle's distance is not exact. It is computed inside a **band** around the anti-diagonal, and
when the hypothesis and reference differ greatly in length the true optimal path falls outside the
band, so the reported distance is larger than the true minimum. Matching the oracle therefore means
reproducing the band, not improving on it.

## The band, as measured

    length_ratio = n_ref / n_hyp            (1 when the hypothesis is empty)
    beam_width   = ceil(length_ratio / 2 + 25)  if 25 < length_ratio / 2
                 = 25                        otherwise

For each hypothesis step `i` from 1 to `n_hyp`:

    pseudo_diag = floor(i * length_ratio)
    min_j       = max(0, pseudo_diag - beam_width)
    max_j       = min(n_ref + 1, pseudo_diag + beam_width)

Only cells `j` in `[min_j, max_j)` are computed; every other cell keeps the "infinite" distance of
`10^16`. Row 0 is `0..=n_ref` and column 0 is `0..=n_hyp`, as usual.

For the 36-vs-87 case: ratio 2.42, so beam 25; at `i = 36`, `pseudo_diag = 87`, `min_j = 62`,
`max_j = 88`. Cells with `j < 62` are uncomputed at that row, which is why the result is 87.

## Consequence for the shift search

The shift search evaluates candidates with the same banded distance, so a narrow band can also make
a shift look better or worse than it is. Both the candidate scoring and the final distance must use
the band, or the search picks different shifts.

## Verification

The three failing corpus scores (`real-granite4_small-h`, `real-nxkd1-k-q5`, `real-qwen2.5vl_7b`)
and the CLI WMT16 case are all this. The frozen fuzz and WMT22 fixtures pass with the full matrix
because their segments are short enough that the band covers the optimal path.