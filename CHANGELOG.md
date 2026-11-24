# Changelog

All notable changes to swadesh are listed here. Versions follow semantic
versioning. swadesh 0.x is pinned to sacreBLEU 2.6.0 (`swadesh.SACREBLEU_COMPAT_VERSION`).

## 0.1.0

First release.

### Added

- Native BLEU and chrF/chrF++ sufficient statistics and score arithmetic. Corpus
  scores, signatures and formatting still come from sacreBLEU's Python code, so
  results are `==` to sacreBLEU 2.6.0 on the same interpreter.
- `swadesh.BLEU` and `swadesh.CHRF`: subclasses of sacreBLEU's classes with the same
  constructors and API. They add `sentence_scores(hyps, refs)` (parallel) and
  `pairwise(candidates, references=None)`.
- `swadesh.mbr()` for minimum Bayes risk selection.
- Native bootstrap resampling sums and `swadesh.PairedTest` (paired bootstrap and
  approximate randomization), with sacreBLEU's p-values and intervals.
- `swadesh.rewards`: chrF, chrF++ and sentence-BLEU reward functions for GRPO, PPO
  and RLOO, including conversational completions.
- `swadesh.patch()` and `swadesh.unpatch()`: route existing `import sacrebleu`
  code (Hugging Face `evaluate`, fairseq, and similar) to swadesh.
- The `swadesh` command: sacreBLEU's CLI with swadesh patched in, with the same flags.
- `python -m swadesh.verify`: compare swadesh and sacreBLEU on any reference and
  hypothesis files.
- A standalone Rust binary, `swadesh` (crate `swadesh-cli`), that prints
  sacreBLEU-compatible BLEU and chrF output without Python.
- Native tokenizers: `13a`, `intl`, `zh`, `char` and `none`. The other sacreBLEU
  tokenizers (`ja-mecab`, `ko-mecab`, `spm`, `flores101`, `flores200`, `spBLEU-1K`)
  are run by sacreBLEU in Python, and the matching is native.

### Known limitations

- TER is native: `swadesh.TER` and `swadesh.corpus_ter` run the Rust tercom implementation, and
  the Rust binary accepts `-m ter`. Byte-exactness is not complete on every case yet; see
  `TESTING.md` for the divergences being closed.
- Exactness holds on one interpreter at a time. sacreBLEU's own corpus scores
  differ in the last bits between CPython 3.11 and 3.12. See
  [docs/python-version-drift.md](docs/python-version-drift.md).
- Random draws for bootstrap and significance still come from NumPy. A Rust port
  of NumPy's generator is tested against NumPy fixtures but not yet used.
- The Rust binary has no interpreter, so its `sum()` semantics come from
  `--python-version` (default 3.12).

### Verification

- Differential property tests (hypothesis, with an adversarial character alphabet)
  against sacreBLEU 2.6.0 in CI, on Python 3.10 to 3.14 with NumPy 2, and on 3.10 to
  3.12 with NumPy 1.26.
- Cross-machine benchmarks on an x86_64 Linux host, an aarch64 Linux host and an
  arm64 macOS host. Each timed result was checked for exact equality with sacreBLEU
  on the same machine. See [docs/guide.md](docs/guide.md#performance).
