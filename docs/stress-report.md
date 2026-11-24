# Stress, robustness and performance report

> **Status (2026-11-24): the 2026-11-19 campaign below is historical; a speed and capability re-run on the current tree is appended at the end of this file.**
>
> **Originally:** This report records the 2026-11-19 testing campaign. It has
> not been re-run since the 2026-11-22 clean-room rewrite of the tokenizers and the segment
> statistics, so the figures below describe the tree as it was then. `TESTING.md` carries the
> re-measured state and the list of known divergences; `docs/lab-notebook.md` records what was
> measured and when.

Criterion 6 of [TESTING.md](../TESTING.md). The raw data is in `bench/stress/results/` (`scaling.jsonl`,
`pathological.jsonl`, `chain-troy1.log`); the scripts are in `bench/stress/`.

**Where it ran.** Scaling and pathological inputs ran on troy-1 (Ryzen 9 7900X, 60 GB RAM) on 2026-11-19
while other jobs ran on the machine (load 16–28), so absolute times are noisy. The run ended when troy-1 ran
out of memory: sacreBLEU's own runs needed 40–46 GB at 1M segments, together with a runaway coverage job.
The memory, thread and concurrency phases were re-run on the VPS under a 4–6 GB cgroup memory cap.
Every result where both tools finished compared `==`.

Note (2026-11-21, re-run on a quiet troy-1 with `ulimit -v 20000000`): at 1M segments sacreBLEU's
chrF raises `MemoryError` after 19.9 GB of RSS; swadesh completes in 1.04 s with a 1.06 GB peak.
SacreBLEU has never produced a chrF score at this size, so there is no exactness comparison to
make for it. BLEU at 1M was re-confirmed bit-identical (score 64.60594655357025, sacreBLEU
9.02 GB vs swadesh 0.97 GB).

## 1. Scaling (corpus scores; one reference; WMT22 en-de perturbed)

| segments | metric | sacreBLEU | swadesh | speedup | peak RSS sacreBLEU / swadesh |
|---:|---|---:|---:|---:|---:|
| 10k | BLEU | 1.04 s | 0.012 s | 87× | 140 / 52 MB |
| 10k | chrF | 2.56 s | 0.020 s | 128× | 431 / 54 MB |
| 10k | TER | 9.22 s | 0.068 s | 135× | 89 / 134 MB |
| 100k | BLEU | 9.56 s | 0.111 s | 86× | 954 / 134 MB |
| 100k | chrF | 28.86 s | 0.135 s | 214× | 4,063 / 148 MB |
| 100k | TER | 85.35 s | 0.498 s | 171× | 276 / 253 MB |
| 1M | BLEU | 106.7 s | 1.69 s | 63× | 8,770 / 951 MB |
| 1M | chrF | MemoryError (needs > 20 GB of RAM; no score produced) | 1.04 s | — | **> 19,882** / 1,063 MB |
| 1M | TER | 859.1 s | 6.32 s | 136× | 2,156 / 1,200 MB |
| 1M | BLEU + 1000 bootstrap | 270.7 s | 6.10 s → **8.4 s / 717 MB after fix** | 44× | **45,601** / 8,520 → **717** MB |
| 10M | BLEU | crashed (OOM) | 10.8 s | — | — / 9,136 MB |
| 10M | chrF | not run | 15.7 s | — | — / 10,339 MB |
| 10M | TER | not run | 56.2 s | — | — / 10,724 MB |

Time grows linearly in both tools; swadesh has no super-linear segment.

## 2. Pathological inputs

| input | result |
|---|---|
| 1 MB single line, 100k-char lines without spaces | both finish, identical; swadesh ≤ sacreBLEU in time and memory |
| 1M tiny segments | BLEU 0.54 s vs 7.84 s; chrF 0.52 s vs 23.6 s; TER 1.02 s vs 9.14 s, all identical |
| TER, 50 sentences of 1,000 words | swadesh 16.1 s vs sacreBLEU 509.9 s, identical; swadesh peak RSS 4.06 GB (see below) |
| TER, 200 sentences of 1,000 words | swadesh 53.4 s; **sacreBLEU crashed** |
| identical 1M corpora | not measured; killed by the machine-wide OOM, not a swadesh failure |

## 3. Findings and fixes

1. **Fixed: bootstrap memory.** With threads, the seeded bootstrap materialized the full `n_samples × n` index matrix:
   8 GB at 1M segments and 80 GB at 10M (the process aborted). It now draws indices in batches of at most 16M
   (`resample.rs`). At 1M × 1000 the peak RSS is 717 MB, and the result is identical to numpy across batch boundaries.
2. **Known limit: TER memory on very long sentences.** sacreBLEU's TER algorithm keeps a prefix cache of up to 10,000
   edit-distance rows per sentence; swadesh reproduces it exactly (16 bytes per cell). At 1,000-word sentences that is
   ~160 MB per sentence, times the number of threads (~4 GB at 24 threads). sacreBLEU keeps the same cache as Python
   tuples and runs out of memory first. Limit threads with `RAYON_NUM_THREADS` for such inputs.
3. **Memory in general:** swadesh's peak RSS is below sacreBLEU's everywhere except TER at ≤100k, where both are small
   (swadesh holds per-thread TER caches).

## 4. Thread scaling (VPS, 8 vCPUs, shared; 200k segments; TER on 50k)

| threads | BLEU | chrF | TER | identical to 1 thread |
|---:|---:|---:|---:|:---:|
| 1 | 0.922 s | 1.340 s | 1.445 s | — |
| 2 | 0.614 s (1.5×) | 0.727 s (1.8×) | 0.825 s (1.8×) | yes |
| 4 | 0.397 s (2.3×) | 0.520 s (2.6×) | 0.491 s (2.9×) | yes |
| 8 | 0.328 s (2.8×) | 0.343 s (3.9×) | 0.356 s (4.1×) | yes |

BLEU scales least because reading the Python strings is serial (it needs the GIL); the parallel part is the n-gram work.

## 5. Concurrency from Python threads (VPS, 100k segments, 5 iterations)

- Independent metric objects in 4, 8 and 16 threads: 0 mismatches, 0 errors (BLEU, chrF, TER).
- One shared object per metric across 8 threads with different references: 0 mismatches.
- Shared `CHRF.sentence_scores` from 1–16 threads: 0 mismatches.
- Fork safety: `PairedTest(n_jobs=2)` after native calls no longer deadlocks (per-process thread pool), covered by
  `tests/test_parity.py`.

## Re-measured 2026-11-24 (speed and capability, current tree)

Run on troy-1, Python 3.11.14, 24 cores, while the differential suite was not running. Each row is
the best of up to three runs; the corpus is `tests/corpus/model/ref-*.txt` (1,050 real segments),
scored with the hypothesis equal to the reference. Absolute times are still noisy (the machine is
shared), and the TER rows vary between runs, so read the ratios as order-of-magnitude.

**Corpus scores**

| metric | segments | swadesh | sacreBLEU 2.6.0 | speed-up |
|---|---|---|---|---|
| BLEU `13a` | 200 | 3.4 ms | 28.7 ms | 8.3x |
| BLEU `13a` | 1,000 | 11.1 ms | 112.7 ms | 10.2x |
| BLEU `intl` | 1,000 | 17.2 ms | 110.5 ms | 6.4x |
| chrF | 1,000 | 18.7 ms | 531.2 ms | 28.4x |
| chrF++ | 1,000 | 26.0 ms | 588.2 ms | 22.6x |
| TER | 1,000 | 7.5-32.6 ms | 952.2 ms | 29-118x |
| TER `normalized` | 1,000 | 14.8 ms | 1,128.6 ms | 76.5x |

**Sentence-level**

| call | segments | swadesh | sacreBLEU | speed-up |
|---|---|---|---|---|
| `sentence_scores`, BLEU `13a` | 1,000 | 26.0 ms | 154.6 ms | 5.9x |
| `sentence_scores`, chrF | 1,000 | 22.0 ms | 475.2 ms | 21.6x |
| `sentence_scores`, chrF++ | 1,000 | 29.6 ms | 528.7 ms | 17.9x |
| `TER().sentence_score` | 200 | 12.8 ms | 197.5 ms | 15.5x |

**Capability (the sizes sacreBLEU struggles with)**

| case | swadesh | sacreBLEU | speed-up |
|---|---|---|---|
| one 20,000-token segment, BLEU | 3.0 ms | 43.6 ms | 14.6x |
| one 20,000-token segment, chrF | 1.9 ms | 43.8 ms | 23.5x |
| 5,000 segments, BLEU | 5.8 ms | 604.9 ms | 103.7x |
| 5,000 segments, chrF | 7.9 ms | 2,906.3 ms | 366.8x |
| one 500-token segment, TER, identical | 12.4 ms | 27.3 ms | 2.2x |
| one 500-token segment, TER, block-shifted | 71.8 ms | 8,536.2 ms | 118.8x |

**Exactness during the benchmark.** At 1,000 segments the scores compared `==` for BLEU, chrF and
TER, so the speed did not come from cutting corners.

The TER rows are the widest spread because the shift search dominates: an identical pair costs one
alignment, a block-shifted pair costs the whole iterative search, and that is where the native
implementation is 100x faster than the reference loop.
