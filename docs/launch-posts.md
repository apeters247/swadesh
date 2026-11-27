# Launch: the posts, in order, and the rules for answering

Written 2026-11-27. Every number below is one of ours and has a source in the repo; nothing here
should be posted with a figure that `bench/results/` or `TESTING.md` cannot back.

The one claim to lead with, everywhere: **the numbers are identical, and you can check that on your own
data in one command.** Everything else (speed, memory, the features) is secondary and only matters
because the first is true.

```bash
python -m swadesh.verify -r ref.txt -i hyp.txt    # prints "identical" or the first mismatch
```

## The sequence

The order matters: the package has to exist before anything links to it, and the blog has to exist
before the technical post, because the blog is where you send the sceptics.

| day | channel | what goes up | why in this order |
|---|---|---|---|
| 0 | PyPI | `swadesh` 0.1.0 (wheels + sdist) | every other post needs an install that works |
| 0 | GitHub | `v0.1.0` tag, release notes from `CHANGELOG.md` | the release page is the stable link |
| 1 | HuggingFace blog | the post in `docs/blog-post.md` | the citable technical artefact |
| 2 | Hacker News | Show HN (below) | the widest technical audience, and it tolerates a project |
| 3 | r/LocalLLaMA | the eval-speed framing (below) | they run evals constantly and feel the cost |
| 3 | r/TranslationStudies | the MT framing | small, but the most relevant readers per post |
| 4 | mt-list / corpora mailing lists | a 10-line technical note | the WMT crowd, no algorithm to fight |
| 5 | Lobsters | same as HN, shorter | invite-only, high signal |
| 7 | follow-up | the first weekly download number, and any bug found | shows the thing is alive |

Do not post to r/MachineLearning's front page: project self-posts are removed there. Use its weekly
thread if it has one. Do not cross-post the same title to three subreddits within an hour; it reads as
spam and gets removed.

## Show HN

**Title** (58 chars, under the 80 limit):

```
Show HN: Bit-exact sacreBLEU in Rust – the same floats, 100x faster
```

**Body:**

```
sacreBLEU is the reference implementation for BLEU/chrF/TER, and in this project's
ledger its numbers are the historical record. So a faster replacement is only
useful if it returns exactly the same floats, not merely close ones.

swadesh is that: the integer half (tokenising, n-gram counting, the TER shift
search) in Rust, the floating-point score arithmetic ported from sacreBLEU so it
cannot drift. On WMT22 en-de with 24 threads it is 117-316x faster, single-digit
GB -> under 1 GB, and 1M segments of chrF goes from a MemoryError at a 20 GB cap
to about 1 s.

The claim is falsifiable on your data:

    pip install "git+https://github.com/apeters247/swadesh"   (PyPI imminent)
    python -m swadesh.verify -r ref.txt -i hyp.txt

That prints either "identical" or the first mismatch, every sentence score
included. If it prints a mismatch, that is a bug and I want the two lines.

Two things worth knowing before you try it: exactness is per-interpreter,
because sacreBLEU's own scores move in the last bits between CPython 3.11 and
3.12 (2,132 of 13,455 sentence BLEU scores differ by up to 9 ulps on our test
data) - swadesh detects the interpreter and matches it. And the exotic
tokenisers (ja-mecab, spm) still delegate to sacreBLEU; the five main ones are
native.

Repo: https://github.com/apeters247/swadesh  (AGPL-3.0, commercial licence free
below $5M revenue).
```

Notes: no "revolutionary", no adjective stacking, and the first paragraph concedes the point a critic
would raise. That is deliberate.

## r/LocalLLaMA

**Title:**

```
Bit-exact sacreBLEU in Rust: scoring evals 100–300x faster without changing a score
```

**Body:** the Show HN body, with the opening changed to the eval-pipeline problem:

```
If you score a lot of generations, the metric step is usually the slow part:
BLEU/chrF in Python is ~10^4 segments/second per core on a good day, and chrF at
1M segments needs ~20 GB.

swadesh is a bit-exact replacement: same floats (verified with ==, every sentence
score, on 13k WMT22 segments and ten more language pairs), 100-300x faster on 24
threads, 1M segments of chrF in ~1 s and ~1 GB.

    python -m swadesh.verify -r ref.txt -i hyp.txt

...tells you in one command whether it agrees on your data. It is also a
drop-in: `import swadesh as sacrebleu`, or `swadesh.patch()` to redirect existing
code, including HuggingFace `evaluate` and lm-evaluation-harness.
```

## r/TranslationStudies

**Title:**

```
A drop-in sacreBLEU that returns identical scores, 100× faster (and how to check it)
```

**Body:** lead with the ledger argument, not the speed: agencies and MT teams re-score archives, and
the archive has to stay comparable. Mention COMET/TER pipeline use, `n_jobs`, and the
`paired_difference` power question ("how many segments before a 1.0 chrF gap is real?"). Same verify
call to action.

## mt-list / corpora

Subject: `bit-exact sacreBLEU in Rust (identical scores, ~100x faster)`

Ten lines, no marketing: what it is, the parity evidence (`test_parity.py` 494 cases, the frozen
oracle, the interpreter-drift table), the verify command, and the repo. This audience will read the
`docs/` and find the holes, which is the point.

## Answering the comments

The same five questions arrive every time. Answer them with the artefact, not with confidence.

| question | answer |
|---|---|
| "Is it *really* identical, or 1e-9 close?" | `==`, per float, per sentence score. Point at `tests/test_parity.py` and the verify command; invite the counterexample. |
| "Why is exactness so hard? It's just arithmetic." | The drift table: CPython changed `sum()` in 3.12; the `zh` tokeniser's range table has a two-character `u'\u20000'`; the bootstrap is float32 numpy. Links to `docs/python-version-drift.md` and `docs/how-its-exact.md`. |
| "Why not just use COMET / an LLM judge?" | Different job. COMET is a learned metric and needs a model; these are the cheap, reproducible reference metrics that every paper still reports. swadesh makes the *standard* baseline fast, not a new metric. |
| "AGPL? I can't use that." | That is exactly what the commercial licence is for; free under $5M revenue, and `docs/licensing.md` states the boundary. |
| "sacreBLEU is Apache-2.0 and fine. Why replace it?" | Because it is 100-300x slower and needs 9-20 GB at 1M segments where this needs 1 GB. Same output, less waiting, and a memory ceiling removed. If your evals are small, keep sacreBLEU; that is an honest answer. |

Never argue that a benchmark is wrong. Ask for the input pair and reproduce it.

## Assets

- blog post (Markdown, Hub format): `docs/blog-post.md`; thumbnail `docs/img/thumbnail.png` (1300x650)
- figures: `docs/img/fig1-thread-scaling.png`, `fig2-size-scaling.png`, `fig3-interpreter-drift.png`
- reproducible benchmarks: `bench/bench.py`, `bench/results/`
- the release notes: `CHANGELOG.md` 0.1.0