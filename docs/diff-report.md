# swadesh differential report (diff track, criterion 1)

Date: 2026-11-19. Oracle: sacreBLEU 2.6.0 on CPython 3.12.3, aarch64 (GX10).
Branch `test/diff`. Machine: GX10 only (shared; see "Environment").

## Result

| Metric | Configuration | Cases | Distinct seeds | Mismatches |
|---|---|---:|---:|---:|
| BLEU | tokenize=13a (lowercase drawn per case) | 1,000,000 | 1,000,000 | 0 |
| BLEU | tokenize=intl | 1,000,000 | 1,000,000 | 0 |
| BLEU | tokenize=zh | 1,000,000 | 1,000,000 | 0 |
| BLEU | tokenize=char | 1,000,000 | 1,000,000 | 0 |
| BLEU | tokenize=none | 1,000,000 | 1,000,000 | 0 |
| chrF | word_order=0 (char_order, beta, lowercase, whitespace, eps_smoothing drawn per case) | 1,000,000 | 1,000,000 | 0 |
| chrF++ | word_order=2 (other options drawn per case) | 1,000,000 | 1,000,000 | 0 |
| TER | short sentences (1-15 words), shifted references, options drawn per case | 1,000,000 | 1,000,000 | 0 |
| TER | 20-200 word sentences, block-shifted references (separate test) | 1,800 | 1,800 | 0 |

**Mismatches: 1 (state only, not scores).** See "Open mismatch" below. All scores, statistics,
signatures and exception types/messages matched in every other case.

### TER option combinations (short-sentence test, 1,000,000 cases)
The four booleans (normalized, no_punct, asian_support, case_sensitive) are drawn per case. All 16
combinations ran, with 62,220 to 62,678 cases each. The long-shift test ran 97 to 140 cases per
combination.

## Configurations and generators

Every case runs the full public surface of the metric: `_extract_corpus_statistics`, `corpus_score`
(score, `str`, and per-metric attributes), `get_signature()`, and per-segment `sentence_score` /
`sentence_scores`. When sacreBLEU raises, swadesh must raise the same exception type and message.

Hypothesis draws one seed per case (`st.integers(0, 2**63-1)`). A `random.Random(seed)` builds the
case, so each case is reproducible from its seed.

Segment families (`tests/deep_gen.py`), with weights:
- Adversarial alphabet from `test_differential.py` (entities, `<skipped>`, CJK, emoji, unicode spaces).
- Arbitrary non-surrogate code points, 0-30 characters.
- Whitespace-only and empty segments. Whitespace covers every Unicode `White_Space` code point and
  everything `str.isspace()` accepts.
- Punctuation-only strings (all Unicode P\* plus ASCII symbols) and digit-only strings (Nd/Nl/No).
- Mixed scripts: runs drawn from 20 script sets (Latin, Greek, Cyrillic, Han, Hangul, Kana, Arabic,
  Hebrew, Devanagari, Thai, Ethiopic, Cherokee, emoji, combining marks, full-width and more).
- Long runs of `. , - 0-9` (1-40 characters, 1-3 runs), which exercise the 13a regex.
- Chained entities (`&amp;amp;amp;`, `&amp;quot;`, `&lt;skipped&gt;`) and `<skipped>` fragments.
- Rare long segments: 0.2% of segments are 100-400 character runs, and 0.2% are 1,000-10,000
  characters.

Reference streams: 1-8 streams. Stream 0 is always defined. Later streams have a 25% chance of a
`None` hole per segment.

TER: hypotheses and references are word sentences drawn from a small vocabulary (so shifts fire),
plus block-shifted copies (1-6 moves of 1-30 words, optional word deletion).

## Environment and deviations

- **TER length.** sacreBLEU's TER costs about 12 ms at 20 words, 300 ms at 50 words, and 1.7 s at 200
  words per sentence (measured on GX10). 1,000,000 cases at 20-200 words would take about 50
  CPU-hours, so the 1M TER cases use 1-15 word sentences. The 20-200 word shift cases are a
  separate test of 1,800 cases (6 shards of 300). That is a smaller count than the other metrics.
- **Process layout.** Each configuration ran as its own single-worker pytest process. Nine ran at
  once during the main run. The coordinator's 6-worker limit arrived partway through; at that point
  only two processes (TER and long-shift TER) were still running. The long-shift shards then ran at
  six workers.
- **Shared machine.** GX10 is shared with another job. Its load average was 45-57 while the run
  started, which made throughput uneven. The default-profile suite was run with 6 workers after a
  memory check (about 20 GB available).
- **Deep profile.** `HYPOTHESIS_PROFILE=deep` sets `max_examples=1,000,000` for every test that
  calls `fuzzsupport.budget()`. That covers the new tests and the existing `test_differential.py`,
  `test_ter.py` and `test_fuzz_unicode.py` tests. The default profile keeps each test's own count,
  so the default run is unchanged.

## Open mismatch

**`BLEU.sentence_scores()` (batch) leaves `num_refs` unchanged; sacreBLEU's per-segment
`sentence_score()` sets it.** Scores are identical. Only `get_signature()` differs afterwards
(`nrefs:var` in swadesh, `nrefs:2` in sacreBLEU). This needs variable reference counts (`None`
holes in a later stream) and a signature query after sentence scoring.

Minimal repro (regression test `tests/test_regressions.py::test_sentence_scores_signature_state_matches_sacrebleu`):

```python
hyps = ["a b", "c d"]
refs = [["a b", "c d"], [None, "c d"]]
# swadesh:   corpus_score -> nrefs:var ; sentence_scores -> nrefs:var
# sacreBLEU: corpus_score -> nrefs:var ; per-segment sentence_score -> nrefs:2
```

Suspected cause: `python/swadesh/metrics.py` sets `self.num_refs` in `sentence_score` and in
`_cache_references`, but the batch `sentence_scores` path never sets it. Severity is low (state, not
values), but it is a drop-in difference, so it is left as a strict xfail for the lead to decide.

## Tooling

- `tests/fuzzsupport.py`: `budget()` (profile-aware `max_examples`) and per-process case counts.
- `tests/conftest.py`: registers the `default` and `deep` Hypothesis profiles and dumps case
  counts at exit when `SWADESH_CASE_DIR` is set.
- `tests/deep_gen.py`: generators. `tests/test_deep.py`: the differential tests.
- `tests/shrink.py`: delta-debugging reducer. Failing cases are written to `SWADESH_FAIL_DIR` as JSON
  and reduced to a minimal repro. The reducer was self-tested on an injected mismatch.

Reproduce: `HYPOTHESIS_PROFILE=deep SWADESH_CASE_DIR=... pytest tests/test_deep.py -k <name>`.
Default run: `pytest tests/test_deep.py` (about 2 min at 12 workers).
