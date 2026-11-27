# Reproducing the numbers, or measuring on your own data

Two ways, both runnable in a couple of minutes. Nothing here is a claim you have to take on faith: the
first script prints a table and checks every result for exact equality, and the second does the same on
files you choose.

## 1. The WMT22 harness (what the README and the blog quote)

```bash
bench/get_data.sh                        # ~37 MB of WMT22 references, from the sacrebleu cache
python bench/bench.py --quick --lines 2000
python bench/bench.py --lines 20000      # the headline sizes
python bench/bench.py --lines 100000 --json bench/results.json
```

`bench/bench.py` takes the WMT22 references and derives hypotheses from them by dropping, swapping and
substituting words with a fixed seed, so every run sees the same corpus. `--quick` skips the slow
paired-test and bootstrap cases.

Sample output (troy-1, 24 cores, 2026-11-27, `--quick --lines 2000`):

```
2000 lines of WMT22 en-de, tokenizer=13a, 24 CPUs, x86_64
case                                sacrebleu    swadesh  speedup  exact
BLEU corpus                            0.131s    0.0086s    15.3x  yes
chrF corpus                            0.321s    0.0122s    26.3x  yes
chrF++ corpus                          0.365s    0.0172s    21.3x  yes
TER corpus x2000                       1.582s    0.0143s   110.5x  yes
sentence BLEU x2000                    0.137s    0.0090s    15.3x  yes
sentence chrF x2000                    0.275s    0.0134s    20.6x  yes
```

## 2. Your own files

```bash
python bench/bench_own.py -r ref.txt -i hyp.txt
python bench/bench_own.py -r ref.txt -r ref2.txt -i hyp.txt --threads 8 --reps 3
```

It times sacreBLEU first (its tokenizers are `lru_cache`d per string, so timing it second would flatter
it), times swadesh, and prints `identical` only when the formatted score strings match exactly — the
same comparison `swadesh.verify` makes, plus the clock. Exit code 1 if any metric differs.

For correctness alone, without timing:

```bash
python -m swadesh.verify -r ref.txt -i hyp.txt     # "identical", or the first mismatch
```

## The ratio depends on the corpus size, so quote both

Same machine, same generator, minimum of 5 runs (one for TER above 1,000 segments):

| segments | BLEU | chrF | TER |
|---:|---:|---:|---:|
| 10 | 10.3× | 19.3× | 236.9× |
| 100 | 13.9× | 15.6× | 551.3× |
| 1,000 | 14.2× | 18.9× | 730.9× |
| 10,000 | 179.2× | 270.7× | *(slow; not run here)* |

Two honest readings of that table:

- **Bigger is better**, but not smoothly: at 1,000 segments the per-call overhead (thread pool, corpus
  marshalling between Python and Rust) is a large share of a small total, and at 10,000 the parallel
  work dominates. A single 100,000-segment run shows the same effect even more strongly.
- **At tens of segments the ratio is meaningless in absolute terms.** Both sides take hundreds of
  microseconds; a timer reading there is noise. Use the library because it is exact, not because a
  10-segment call is 10× faster.

## Method notes, so a discrepancy can be traced

- **sacreBLEU is single-threaded and swadesh is not.** `swadesh.set_n_jobs(1)` (`--threads 1`) pins it to
  one worker if you want a like-for-like comparison; the score is identical either way.
- **Peak memory** is not in these tables; `docs/stress-report.md` has the 1M-segment measurements, where
  sacreBLEU's chrF raises `MemoryError` under a 20 GB cap and swadesh finishes in ~1 GB.
- **The machine matters.** These numbers came from a Ryzen 9 7900X with other tenants using ~4 of 24
  cores; the load average is recorded in `bench/results/*.json`. On a busy laptop expect smaller ratios.
- **Exactness is per interpreter.** sacreBLEU's own scores move in the last bits between CPython 3.11 and
  3.12, so compare on the interpreter you will report from, and name it (`docs/python-version-drift.md`).

If a number disagrees with what is published here, the useful report is the two output lines, the
segment count, `python -V`, `sacrebleu.__version__`, and the core count — not the ratio on its own.