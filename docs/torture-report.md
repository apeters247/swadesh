# swadesh torture report (2026-11-19)

Branch `agent/torture` at `b6d09ea` (master at the time). Goal: find any input where
swadesh differs from sacreBLEU 2.6.0. **Result: no mismatch found** across everything below.

## Data
Downloaded with sacreBLEU's own downloader (shared cache `bench/.sacrebleu`, about 1.5 GB new).
Extracted files are in `bench/data/torture/` (gitignored). Per-pair JSON results are in
`bench/data/torture/results/py311|py312|py314/`.

- 45 language-pair files, 64,625 source lines (WMT20, WMT21, WMT22, WMT23, WMT24, IWSLT17).
- Languages: ar, bn, cs, de, en, ha, he, hi, is, iu (Inuktitut syllabics), ja, km, ko, liv, ps,
  ru, sah, ta, uk, xh, zh, zu.
- Not available in sacreBLEU 2.6.0 datasets (not tested with real data): Thai, Vietnamese,
  Greek, Turkish, Georgian, Armenian, Amharic. FLORES is not in this version's listing.
  Turkish dotted/dotless I is covered only by synthetic insertion of U+0130 and U+0131.

## Hypothesis types per pair
- `sys0_vs_ref0`, `sys1_vs_ref0` (real systems, where the set provides them)
- `sys0_vs_ref0+ref1`, `ref0_vs_ref1` (where a second reference exists)
- `sys0_vs_perturbed_ref0` (perturbed references)
- `perturbed_ref0_as_hyp` (perturbed hypothesis)
- `src_as_hyp` (source text as hypothesis, cross-script noise)

Perturbation (seeded, about 8% of characters): inserts or replaces with RTL/LTR marks, ZWJ/ZWNJ,
ZWSP, BOM, NBSP variants, ideographic space, U+2028/2029, U+0085, soft hyphen, combining marks,
U+0130/0131, U+1E9E, emoji, ZWJ sequences, control chars, PUA, regional indicators, Kana/Hangul/Han
variants, NFC/NFD/NFKC/NFKD, case swaps, space deletion, and empty segments (about 5%).

## Configurations compared
- `verify()` for tokenizers {13a, intl, zh, char, none} x lowercase {False, True}. Each runs BLEU,
  chrF and chrF++, comparing corpus score, str, signature and every sentence score.
- Stricter checks: CHRF(whitespace=True), CHRF(eps_smoothing=True), CHRF(whitespace, word_order=2,
  lowercase), CHRF(eps_smoothing, beta=3), BLEU(add-k, effective_order), BLEU(intl, floor, order 2).
  Each compares corpus statistics, corpus score and sentence scores.
- Bootstrap CI (`corpus_score(n_bootstrap=200)`): BLEU 13a, BLEU zh, CHRF on three hypothesis
  types for all 45 pairs.

## Totals (Python 3.11, main code)
- 189 hypothesis types, 256,894 hypothesis segments.
- About 9.25M sentence-score comparisons, plus 7,167 corpus-level comparisons.
- Mismatches: 0. Errors: 0 (after the re-run described below).

## Python version subset
- en-ja (wmt24) and he-en (wmt23), all tokenizers and configurations, on Python 3.12.12 and 3.14.3
  (built with `maturin develop --release` in this worktree, abi3 extension).
- 12 hypothesis types per version, 18,355 segments each, about 660k sentence comparisons each.
- Mismatches: 0.

## Long hypothesis fuzzing
- Existing differential suite at `max_examples=20000` (copy in `tests/test_differential_long.py`):
  `test_chrf_corpus`, `test_chrf_sentence`, `test_pairwise_matrices`, and
  `test_bleu_corpus` restricted to tokenizers zh and char. All passed.
- The full parametrized suite at 20k would take hours, so the other tokenizers were not run at 20k.
- New Unicode-weighted generator (`tests/test_fuzz_unicode.py`). Full non-surrogate range,
  weighted toward P, S, N, Z, Cc, Han, Hangul and Kana, segments up to 300 characters. Four tests
  (BLEU corpus/sentence, chrF corpus/sentence) at 50,000 examples each. All passed.

## Environment notes (not sacreBLEU mismatches)
- The first Python 3.11 pass used the shared `.venv` editable install, whose `_swadesh` binary was
  rebuilt by another agent during the run. One pair crashed with a missing symbol and the results
  were mixed. That pass was discarded and the whole 3.11 set was re-run against this worktree's
  rebuilt binary at `b6d09ea`.
- The worktree branched from an older master. It was fast-forwarded to `b6d09ea` before the
  final runs.

## Conclusion
No difference from sacreBLEU 2.6.0 was found in these inputs or configurations. No xfail regression
tests were added, because there were no minimal reproducers to record.
