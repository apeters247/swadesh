# Rust vs C vs hand-written assembly for swadesh's hot kernels

**Recommendation: Rust.** On the same algorithms, Rust and C are within noise of each other on
the tokenizer and word n-gram kernels. Rust is faster on character n-grams with sort-based
counting, and C is about 6% faster with hash counting. Hand-written assembly did not beat
compiler-generated AVX2 intrinsics. The deciding factors are not speed: memory safety on
untrusted input, PyO3 + maturin for Python wheels, WebAssembly and cross-platform builds, and
rayon. The largest measured win in this project came from algorithmic changes, not from the
language: the memchr-driven `13a` passes made tokenization about 2.6× faster in Rust after this
benchmark was taken.

## Results

Single thread, ns per segment (hypothesis + reference), WMT22 en-de, 200,000 segments.
Each value is the median of 3 rounds, each round itself a median of 5 runs
(`bench/lang/results/focused_en-de.tsv`, load average 3.6–6.4 on the shared 8-core VPS).
All implementations produced identical checksums before timing.

| kernel | C | Rust | faster |
|---|---:|---:|---|
| `13a` tokenizer (hyp + ref) | 3,811 | 4,027 | C by 5% (Rust ASCII-path variant: 3,800, a tie) |
| chrF char n-grams 1–6, sort + RLE | 35,335 | 23,874 | Rust by 32% |
| chrF char n-grams 1–6, open-addressing hash | 19,874 | 21,175 | C by 6% |
| BLEU word n-grams 1–4, sort + RLE | 2,950 | 3,501 | C by 16% (noisy: Rust ranged 3,317–4,472) |

The 13a class-1 byte scan (the tightest inner loop), ns per segment:

| variant | C | Rust |
|---|---:|---:|
| scalar loop | 224 | 32 (auto-vectorized by LLVM) |
| 256-entry lookup table | 84 | 22 |
| SSE2 intrinsics | 16 | 14 |
| AVX2 intrinsics | 8 | 9 |
| hand-written AVX2 assembly (same `.s` file linked into both) | 8 | 7 |

Four threads (rayon vs OpenMP), ns per segment, from the earlier full run
(`bench/lang/results/results.tsv`; load average 7–18, so treat these as rough):

| kernel | en-de C | en-de Rust | zh-en C | zh-en Rust |
|---|---:|---:|---:|---:|
| `13a` tokenizer | 2,630 | 1,682 | 1,798 | 1,560 |
| chrF char n-grams (sort) | 18,565 | 10,011 | 15,747 | 9,518 |
| BLEU word n-grams (sort) | 1,601 | 1,650 | 1,672 | 1,698 |

## What the numbers say

- **Rust ≈ C.** With the same algorithms, the two compile to similar code through the same
  family of optimizers (LLVM for Rust; GCC 13 `-O3 -march=native` for C). The differences that remain
  go both ways and are about the size of the run-to-run noise on a shared machine.
- **Assembly doesn't pay.** The hand-written AVX2 scan ties the intrinsics in both languages.
  The intrinsics, in turn, beat the plain loop by only a few ns per segment, out of a
  several-thousand-ns segment. The scan is not where time goes.
- **The C scalar loop is the outlier.** GCC did not vectorize the branchy class test
  (224 ns), while LLVM vectorized the equivalent Rust loop (32 ns). This is a compiler
  difference, not a language difference; clang might close it.
- **Algorithms dominate.** Hash vs sort counting changed chrF by 15–40% in each language, more
  than the language did. After this benchmark, swadesh-core moved to an open-addressing count
  table with O(1) generation clearing and to memchr-driven tokenizer passes. Those changes made
  `13a` about 2.6× faster (1,344 → 516 ns per line) and BLEU segment statistics 1.7× faster.
  The Rust numbers above predate them.

## Non-speed factors

| | Rust | C |
|---|---|---|
| Untrusted input (arbitrary Unicode from users and agents) | bounds-checked; `unsafe` limited to two audited spots | every buffer and index must be checked by hand |
| Python bindings | PyO3 + maturin; abi3 wheels for all platforms from one CI job | CPython C API or cffi; manual refcounting |
| WebAssembly | `cargo build --target wasm32-unknown-unknown` (the playground) | possible via Emscripten/clang, with more setup |
| Parallelism | rayon, data-race free by construction | OpenMP/pthreads |
| Packaging | crates.io + PyPI from the same workspace | per-platform build scripts |
| Contributors | large, growing pool in ML tooling (tokenizers, polars, ruff) | large but shrinking for new tooling |

## Caveats

- Shared production VPS (8 cores, AVX-512 capable); other workloads ran during every
  measurement. Each row records the 1-minute load average.
- One input language pair for the single-thread table; zh-en single-thread numbers are in
  `results.tsv` (taken at load 13–17).
- Multi-reference max-merge was not benchmarked.
- C was compiled with GCC; clang was not available.

## Reproduce

```bash
python bench/lang/gen_inputs.py        # builds the 200k-segment inputs from bench/data
bash bench/lang/run_focused.sh         # C, Rust and asm variants; checksums enforced
```
