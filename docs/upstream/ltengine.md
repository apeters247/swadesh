# Upstream: a comparative benchmark for LTEngine

`LibreTranslate/LTEngine` — Rust, **AGPL-3.0** (so a permissive crate is trivially compatible), 216★,
last push 2026-11-20. Its README has this unchecked roadmap item:

> - [ ] Create comparative benchmarks between LTEngine and proprietary software.

That is the hook: the project already wants the thing, and `swadesh-core` supplies the scoring half of it
with numbers that stay comparable to published work.

## Why a benchmark, and why this crate

A "comparative benchmark" needs two decisions: a corpus, which is theirs, and a scorer, which is the part
everyone gets wrong. If the scorer is not sacreBLEU, the numbers are not comparable to the WMT results
people quote, and a 0.3 chrF difference cannot be defended. `swadesh-core` produces **sacreBLEU 2.6.0's
numbers exactly** — same floats, same signatures, same tokenizer behaviour — in-process, in Rust, with no
Python toolchain in the image they already ship.

It is also fast enough to run in their CI: on their scale (a few thousand segments per language pair) the
whole benchmark is milliseconds.

## The dependency

```toml
# workspace Cargo.toml
[workspace.dependencies]
swadesh-core = "0.1.1"
```

```toml
# ltengine/Cargo.toml
[dependencies]
swadesh-core = { workspace = true }
```

`swadesh-core` is **Apache-2.0 OR MIT**, so their AGPL-3.0 project can take it under either, and no
licence scanner will object. (Before 0.1.1 it was AGPL, which is why this is worth doing now.)

## The whole integration, measured

50 lines, against the published crate, producing exactly what the CLI prints for the same files:

```
   chrF 61.9252     (sacreBLEU: 61.9)
 chrF++ 64.3703
    TER 16.6667     (sacreBLEU: 16.7)
   BLEU 37.9918     (sacreBLEU: 38.0)
```

The shape is: read files, call `bleu_corpus_stats` / `chrf_corpus_stats` / `ter_corpus_stats`, sum the
per-segment rows, then `compute_bleu` / `chrf_f_score` on the single summed row. The statistics rows are
already `[sys_len, ref_len, correct.., total..]`, so summing them *is* the corpus aggregate — there is no
second implementation of anything.

One decision the host project has to make explicitly, because it changes the last two digits of a score:
which CPython's `sum()` semantics to reproduce. `BleuScoreConfig { sum_mode: SumMode::for_python(3, 12) }`
matches 3.12/3.13; `SumMode::Naive` matches 3.11 and earlier. sacreBLEU's own scores move between those
interpreters, so the benchmark should record which one it reproduces.

## Proposed issue (paste as-is)

**Title**

```
A sacreBLEU-identical scorer for the comparative benchmarks on the roadmap?
```

**Body**

```
The roadmap has "Create comparative benchmarks between LTEngine and proprietary software". I can supply
the scoring half of that, if you want it.

I maintain swadesh, a Rust implementation of sacreBLEU 2.6.0's BLEU / chrF / chrF++ / TER. The point of it
is that it returns the same numbers to the last float — verified with `==` on 13k WMT22 segments, 10 more
language pairs and real model output — so a benchmark built on it stays comparable with published WMT
results, and a 0.3 chrF difference is a claim you can defend rather than an artefact of a different
tokenizer or a different `sum()`.

It is Apache-2.0 OR MIT (0.1.1, on crates.io), so there is nothing to negotiate for an AGPL project. No
Python is involved: it is the same crate that backs the standalone `swadesh` CLI and the Python package.

The integration is small — statistics, one element-wise sum per corpus, then the score arithmetic — and I
have a 50-line binary that scores ref/hyp files and prints the four metrics, producing exactly what the
sacreBLEU CLI prints for the same files:

    BLEU 37.9918   chrF 61.9252   chrF++ 64.3703   TER 16.6667

If you'd like it, tell me the shape you prefer and I'll open the PR:

1. a `bench` binary (`src/bin/bench.rs`) taking `--ref/--hyp/--lang`, printing the four metrics and the
   signature line;
2. a `--score-dir` mode that walks a corpus directory in the layout you already use for test sets;
3. or just the dependency and I will stay out of your CLI design.

Happy to also add the "which corpus, which interpreter" note to the output, since the numbers only
reproduce exactly if that is pinned.

If comparative benchmarks are not a direction you want to take yet, no problem — say so and I'll leave it
there.
```

## Notes before posting

- It is an **issue first**, deliberately: adding a binary to someone's server crate is a feature, and a
  surprise PR for a feature is the fastest route to a closed PR.
- The second candidate, `somastllc/SomNLP-Translate` (Rust, 33★), has **no licence file**, so a
  contribution there has unclear terms. Worth a look only after they add one.
- If they accept, the patch is the 50-line proof above plus argument parsing in their house style, and the
  dependency line. The proof is in this repository's history as `docs/upstream/ltengine-bench.rs`.