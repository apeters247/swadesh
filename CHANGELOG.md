# Changelog

All notable changes to swadesh are listed here. Versions follow semantic versioning. swadesh 0.x is
pinned to sacreBLEU 2.6.0 (`swadesh.SACREBLEU_COMPAT_VERSION`).

## 0.1.0 — 2026-11-26

First release. BLEU, chrF, chrF++ and TER match sacreBLEU 2.6.0 exactly, in a drop-in Python package
and a standalone Rust CLI.

### Added

- **The metrics, natively.** BLEU, chrF/chrF++ and TER - including the banded edit distance and the
  shift search - as Rust sufficient statistics with sacreBLEU's score arithmetic, signatures and
  formatting. Every returned number compares `==` with sacreBLEU 2.6.0 on the same interpreter.
- **A drop-in Python package.** `import swadesh as sacrebleu`; `swadesh.BLEU`/`CHRF`/`TER` with
  sacreBLEU's constructors, attributes and score objects, plus `sentence_scores(hyps, refs)` and
  `pairwise(...)` for the batch and MBR shapes sacreBLEU has no shortcut for.
- **`swadesh.patch()`** and `unpatch()`, which route existing `import sacrebleu` code (Hugging Face
  `evaluate`, fairseq, and similar) to swadesh.
- **A standalone Rust CLI that needs no Python.** Every sacreBLEU flag, with byte-identical
  stdout/stderr and exit codes: the significance group (`-ci`, `-cin`, `-par`, `-pbs`, `-parn`,
  `-pbsn`, `-j`), the test-set group (`-t` including comma-separated lists, `--list`, `--download`,
  `--echo`, `-ol`, `--subset`), `--citation` and `-d/--detail`. All 67 catalogue sets fetch and parse
  - tar+gzip, zip, plain `.gz`, a plain `.ref` - across five parser families.
- **The `swadesh` console script runs the compiled code**, not sacreBLEU's argument parser, so the CLI
  needs sacreBLEU absent and `python -m swadesh.cli` is byte-identical to the standalone binary.
- **`swadesh.PairedTest`**: paired bootstrap and approximate randomization with sacreBLEU's p-values
  and intervals. The bootstrap draw emulates `numpy.random.default_rng(seed).choice` natively.
- **`swadesh.rewards`**: chrF, chrF++ and sentence-BLEU reward functions for GRPO, PPO and RLOO,
  including conversational completions.
- **`python -m swadesh.verify`**: compare swadesh with sacreBLEU on your own reference and hypothesis
  files and print either "identical" or the first mismatch.
- **Native tokenizers** `13a`, `intl`, `zh`, `char` and `none`. The exotic ones (`ja-mecab`,
  `ko-mecab`, `spm`, `flores101`, `flores200`, `spBLEU-1K`) are delegated to sacreBLEU's Python
  implementations on use and need `pip install "swadesh[sacrebleu]"`.
- **Input handling**: `-e/--encoding` for utf-8, utf-8-sig, ascii, latin-1 and cp1252 (with Python's
  own error lines), and gzip input by name.
- **An MCP server** (`swadesh-mcp`) and a WebAssembly build of the CLI for the browser.

### Known limitations

- Exactness holds **one interpreter at a time**: sacreBLEU's own corpus scores differ in the last bits
  between CPython 3.11 and 3.12, and swadesh reproduces whichever interpreter you run. The standalone
  binary takes `--python-version` (default 3.12).
- The exotic tokenizers and `patch()` need sacreBLEU installed (`pip install "swadesh[sacrebleu]"`); the
  metrics themselves do not.
- `-tok ja-mecab` and the other unbuilt tokenizers exit 2 with a hint, where sacreBLEU exits 1 after a
  traceback. It is the one registered CLI divergence; the rest are documented in
  [docs/parity-report.md](docs/parity-report.md).

### Verification

- `cargo test --release --workspace`: 13 suites, 0 failed.
- `tests/test_cli.py`: **685 passed**, 5 skipped (opt-in download tests), 4 xfailed, 0 failed - it runs
  the Rust binary against the sacrebleu 2.6.0 CLI and byte-diffs both.
- `tests/test_parity.py`: 494 cases, 0 failed. `test_frozen_oracle.py`, `test_ter.py`, `test_deep.py`,
  `test_differential*.py`, `test_fuzz_unicode.py`, `test_coverage_gaps.py`, `test_mutation_kills.py`,
  `test_mcp.py`: 0 failed.
- **Real model output**: qwen3.8-27b and llama3.3:70b, 150 English-German segments each, 12
  configurations apiece - 24/24 corpus scores bit-exact, 0 of 300 sentence scores differing, 10/10 CLI
  invocations identical, `total mismatches: 0`.
- **Ten more language pairs** - Gujarati, Kazakh, Lithuanian, Arabic in both directions, Japanese,
  Korean, German-to-French and Turkish: CLI byte-identical on every metric, 0 of 4,000 sentence scores
  differing. See [docs/multilingual-parity.md](docs/multilingual-parity.md).
- **`tools/smoke_install.sh`** passes for the wheel and for the sdist built from source, on CPython
  3.12 (Linux): import as sacrebleu, `patch()`, the console script, the standalone binary,
  `swadesh.verify` and the MCP server.
- **Speed**: up to 316x on 24 threads and 11-36x on one core, every timed case checked `==`; see the
  README and [`bench/results/SUMMARY.md`](bench/results/SUMMARY.md).