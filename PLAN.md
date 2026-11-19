# swadesh — work plan

**Goal:** a bit-exact, 20–200× faster drop-in replacement for sacreBLEU 2.6.0, written in Rust.
Get it known first: benchmarks, a strong README, then launch posts.

Named after the **Swadesh list**, the core-vocabulary lists that the anthropologist-linguist
Morris Swadesh used to measure how much two languages share (lexicostatistics). BLEU and
chrF do the same for a translation and its reference.

## Non-negotiables
1. **Bit-exact.** Every statistic and score must be `==` to sacreBLEU 2.6.0 on the same
   Python interpreter, including quirks (the buggy `zh` CJK ranges, CPython 3.12's
   Neumaier `sum()`, numpy float32 bootstrap arithmetic). Differential tests are the oracle.
2. **Drop-in.** Same classes, constructor arguments, `Score` objects, signatures and CLI flags.
3. **Measured claims only.** Every speedup in the README comes from `bench/` output on a
   named machine.

## Architecture
- `crates/swadesh-core`: pure Rust tokenizers (13a, intl, zh, char, none), n-gram
  statistics, score arithmetic, (WIP) TER, bootstrap/paired tests. Uses rayon.
- `crates/swadesh-py`: PyO3 bindings (`swadesh._swadesh`), abi3 (Python 3.10+).
- `python/swadesh`: subclasses of sacreBLEU's `BLEU`/`CHRF`. Only sufficient-statistics
  extraction is native, so corpus scores are computed by sacreBLEU's own code.
- `tests/test_differential.py`: hypothesis fuzzing against sacreBLEU, with an adversarial alphabet.
- `bench/`: real WMT22 data (`bench/get_data.sh`), `bench/bench.py`.

## Task board
| Track | Owner | Status |
|---|---|---|
| Core BLEU/chrF stats + bindings + diff tests | lead | done (v0, 24–60× speedup) |
| Core perf: ASCII fast paths, alloc-free tokenizers, faster counting | lead | in progress |
| Bootstrap / paired-test (bs, ar) native and bit-exact | lead + `rng` agent | in progress |
| Native TER (tercom shifts) | `ter` agent | in progress |
| numpy PCG64/SeedSequence/choice emulation | `rng` agent | in progress |
| Standalone Rust CLI (`swadesh` binary, sacreBLEU-identical output) | `cli` agent | in progress |
| Python 3.10–3.14 × numpy 1/2 matrix, CI and wheel workflows | `compat` agent | in progress |
| Rust vs C vs asm evaluation (micro-benchmarks) | `lang` agent | in progress |
| Cross-machine benchmarks (VPS x86, troy-1, GX10 ARM, Mac mini M4) | `bench` agent | later |
| README, docs, launch posts | `docs` agent | later |

## Features beyond sacreBLEU
- `sentence_scores()`: parallel segment-level scores (RL rewards, MBR, data filtering).
- `swadesh.patch()`: accelerates every existing `import sacrebleu` user (HF evaluate, fairseq…).
- Native bootstrap and paired significance tests, bit-identical p-values.
- Pairwise matrices (`chrf_matrix`) for MBR decoding / consensus.
- Standalone static binary and a Rust crate (no Python needed; good for agents and CI).

## Agent conventions
- Each agent works in its own worktree `/var/www/llmeval-wt/<name>` on branch `agent/<name>`,
  with its own `.venv` and `target/`. The lead merges.
- Keep temp output small; `cargo clean` before finishing; check `df -h /` (the VPS disk is ~92% full).
- Use `cargo build -j2`. This VPS runs production services.
