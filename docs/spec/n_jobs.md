# `n_jobs`: the worker-thread count for the native corpus kernels

Status: additive API (idea 5 of `docs/feature-ideas.md`), written 2026-11-27 from a measurement on
troy-1. No score changes; the thread count never affects a number.

## The measurement that shapes it (and corrects the idea's premise)

The idea assumed `corpus_score` was single-threaded and that extending `n_jobs` to it would "turn a
24-core machine into roughly a 20x corpus scorer". It is **already parallel**: the statistics kernels
run on a rayon pool sized to the machine (`swadesh-core`'s default `parallel` feature, `par::install`).
Measured on troy-1 (Ryzen 9 7900X, 24 threads), 40,000 segments x 25 words, one reference stream,
`taskset` capping the CPUs to emulate the thread count:

| CPUs | BLEU | chrF++ | TER |
|---|---|---|---|
| 1 | 0.201 s | 0.476 s | 5.964 s |
| 4 | 0.107 s | 0.138 s | 2.017 s |
| 8 | 0.043 s | 0.073 s | 0.920 s |
| 16 | 0.031 s | 0.048 s | 0.648 s |

The scaling is the point: TER is ~9x at 16 CPUs, BLEU ~6x, chrF++ ~10x, with **identical scores at
every count**. So the knob is not "turn parallelism on"; it is "choose the thread count", which
already defaults to the machine.

## What ships

`swadesh.n_jobs(k)` (a context manager), `swadesh.set_n_jobs(k)` and `swadesh.get_n_jobs()`, backed by
a new `_swadesh.set_num_threads` / `num_threads` pair over `swadesh_core::par::set_num_threads`, which
replaces the process's rayon pool.

* `set_n_jobs(n)` with `n >= 1` builds a pool of `n` threads and returns `n`.
* `set_n_jobs(0)` or `set_n_jobs(None)` drops the pool; the next call rebuilds the machine default.
  Returns the default count.
* `n_jobs(n)` restores the previous count on exit.
* `get_n_jobs()` is the count in effect.

## Relationship to sacreBLEU's `n_jobs`

Not the same thing and deliberately so. sacreBLEU's `PairedTest(n_jobs=)` spawns **processes** for its
resampling (`multiprocessing`, which is why `swadesh-core`'s pool is fork-safe). This one sizes the
**threads inside one process** for the corpus kernels. The name is shared because the meaning - "how
much parallelism to use" - is shared; the mechanism and the scope are not.

## Invariants (each asserted by a test)

1. A score is identical at any thread count: `set_n_jobs(1)` and the default give the same string for
   BLEU, chrF and TER.
2. `set_n_jobs(n)` returns `n` for `n >= 1`, and `get_n_jobs()` agrees.
3. `n_jobs(n)` restores the previous count on exit, including on an exception.
4. `set_n_jobs(0)` / `set_n_jobs(None)` restore the machine default (a count `>= 1`).
5. Without the `parallel` feature `set_num_threads` is a no-op and `num_threads()` is 1; the Python
   API still exists and reports a count `>= 1`.
