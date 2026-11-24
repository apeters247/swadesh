# swadesh user guide

swadesh is a Rust implementation of sacreBLEU 2.6.0's BLEU, chrF and chrF++
scoring, with a Python package that drops into existing sacreBLEU code. Its
scores are checked for exact equality (`==`) against sacreBLEU 2.6.0 on the same
interpreter. The Python classes subclass sacreBLEU's own. The per-segment
statistics and the batch sentence scores are native. Corpus scores, signatures
and score formatting still come from sacreBLEU's Python code. See
[how-its-exact.md](how-its-exact.md) for the details.

Contents

1. [Install](#install)
2. [Three ways to use it](#three-ways-to-use-it)
3. [Batch sentence scores](#batch-sentence-scores)
4. [Pairwise matrices and MBR](#pairwise-matrices-and-mbr)
5. [Significance tests](#significance-tests)
6. [Reward functions for RL](#reward-functions-for-rl)
7. [Checking your data: `python -m swadesh.verify`](#checking-your-data-python--m-swadeshverify)
8. [The standalone Rust binary](#the-standalone-rust-binary)
9. [Tokenizers](#tokenizers)
10. [Threading](#threading)
11. [Limits](#limits)

## Install

swadesh needs Python 3.10 or newer, and it builds with a Rust toolchain through
maturin. The runtime dependencies are `sacrebleu==2.6.0` and `numpy`.

From a checkout of the repository:

```bash
python -m venv .venv
. .venv/bin/activate
pip install .
python -c "import swadesh; print(swadesh.__version__, swadesh.SACREBLEU_COMPAT_VERSION)"
```

The second version number is the sacreBLEU version swadesh is pinned to (2.6.0).
The exactness claims in this guide cover sacreBLEU 2.6.0 only.

## Three ways to use it

### 1. `import swadesh`, with the same API as sacrebleu

The classes take the same constructor arguments and return the same `Score`
objects. Changing `import sacrebleu` to `import swadesh` is enough for most code.

```python
import swadesh

hyps = ["The cat sat on the mat.", "It is raining in Lisbon today."]
refs = [["The cat is sitting on the mat.", "It rains in Lisbon today."]]  # one list per reference stream

print(swadesh.BLEU().corpus_score(hyps, refs))
print(swadesh.CHRF(word_order=2).corpus_score(hyps, refs))   # chrF++
print(swadesh.corpus_bleu(hyps, refs))
print(swadesh.sentence_bleu(hyps[0], [refs[0][0]]))
```

`refs` is a list of reference streams: one list per reference set, each with one
string per segment. This is the sacreBLEU layout, and it is the most common
mistake when you first use the API.

The module-level functions `corpus_bleu`, `raw_corpus_bleu`, `sentence_bleu`,
`corpus_chrf`, `sentence_chrf`, `corpus_ter` and `sentence_ter` are sacreBLEU's
functions rebound to swadesh's classes.

### 2. `swadesh.patch()` for existing code

`swadesh.patch()` replaces sacreBLEU's `BLEU` and `CHRF` classes in place, along
with the compat functions, the metric registry, and the significance kernels.
Code that you do not control, and that already does `import sacrebleu`, then uses
swadesh with no changes. Call it once at program start, before that code runs.

```python
import swadesh

swadesh.patch()

import sacrebleu

hyps = ["The cat sat on the mat."]
refs = [["The cat is sitting on the mat."]]
print(sacrebleu.corpus_bleu(hyps, refs))   # now computed by swadesh
print(sacrebleu.BLEU is swadesh.BLEU)      # True

swadesh.unpatch()                          # restores sacreBLEU's classes
print(sacrebleu.BLEU is swadesh.BLEU)      # False
```

For Hugging Face `evaluate`, see `examples/02_patch_hf_evaluate.py`. It calls
`swadesh.patch()` before `evaluate.load("sacrebleu")`.

### 3. The `swadesh` command line

The `swadesh` command is sacreBLEU's CLI with swadesh patched in, so it accepts
the same flags and writes the same output. The reference files come before `-i`,
as in sacreBLEU's CLI.

```bash
printf '%s\n' "The cat sat on the mat." "It is raining in Lisbon today." > hyp.txt
printf '%s\n' "The cat is sitting on the mat." "It rains in Lisbon today." > ref.txt

swadesh ref.txt -i hyp.txt                          # BLEU, JSON output
swadesh ref.txt -i hyp.txt -m bleu chrf -f text     # several metrics, text output
swadesh ref.txt -i hyp.txt -m chrf -cw 2 -b         # chrF++, score only
swadesh ref.txt -i hyp.txt -sl -m bleu -f text      # one line per segment
cat hyp.txt | swadesh ref.txt -tok 13a -lc          # hypotheses on stdin
```

Every option of sacreBLEU's CLI is available, because the command runs
sacreBLEU's own code. That includes the test-set options (`-t`, `--list`,
`--echo`), which download their data as sacreBLEU does.

## Batch sentence scores

`sentence_scores` returns one sentence-level score per segment, computed in
parallel. Use it for RL rewards, MBR, and data filtering. The values equal
`[sentence_score(h, [r[i] for r in refs]).score for i, h in enumerate(hyps)]`,
but there is no Python loop and no per-segment object.

```python
import swadesh

hyps = [
    "The cat sat on the mat.",
    "A dog barked.",
    "It is raining in Lisbon today.",
]
refs = [[
    "The cat is sitting on the mat.",
    "The dog barked loudly.",
    "It rains in Lisbon today.",
]]

bleu_scores = swadesh.BLEU(effective_order=True).sentence_scores(hyps, refs)
chrf_scores = swadesh.CHRF().sentence_scores(hyps, refs)
print([round(s, 1) for s in bleu_scores])
print([round(s, 1) for s in chrf_scores])
```

The scores are on sacreBLEU's 0 to 100 scale. Use `effective_order=True` for
sentence-level BLEU. With it on, an n-gram order that has no n-grams in a short
segment is left out of the geometric mean. Without it, sacreBLEU warns that
sentence-level BLEU should use effective order.

If the hypothesis list and the reference streams have different lengths,
`sentence_scores` truncates to the shortest, as sacreBLEU's own `zip` does.

## Pairwise matrices and MBR

`pairwise(candidates, references=None)` returns the NumPy matrix
`M[i, j] = sentence_score(candidates[i], [references[j]]).score`. When
`references` is omitted, the candidates are scored against each other. Each
sentence's n-grams are extracted once, not once per pair.

`swadesh.mbr(candidates, metric=None)` returns `(best_index, utilities)`, where
each candidate's utility is its mean score against the other candidates. The
default metric is `CHRF()`.

```python
import swadesh

candidates = [
    "The committee approved the new budget on Tuesday.",
    "The committee approved the budget on Tuesday.",
    "On Tuesday the committee passed the new budget.",
    "The committee rejected the new budget on Tuesday.",
]

matrix = swadesh.CHRF().pairwise(candidates)
print(matrix.shape)                        # (4, 4)

best, utility = swadesh.mbr(candidates)
print("chrF MBR choice:", candidates[best])

best_bleu, _ = swadesh.mbr(candidates, metric=swadesh.BLEU(effective_order=True))
print("BLEU MBR choice:", candidates[best_bleu])
```

## Significance tests

`swadesh.PairedTest` has the same interface as
`sacrebleu.significance.PairedTest`. Its resampling kernels are native. The
random draws come from NumPy, exactly as sacreBLEU draws them, so p-values and
intervals match. The first system in the list is the baseline.

```python
import swadesh

refs = [[
    "The cat is sitting on the mat.",
    "A dog barked loudly at the postman.",
    "It rains in Lisbon today.",
    "The meeting moved to Thursday because of the holiday.",
]]
baseline = [
    "The cat sat on the mat.",
    "A dog barked loudly at the mailman.",
    "It is raining in Lisbon today.",
    "The meeting moved to Thursday due to the holiday.",
]
candidate = [
    "The cat is sitting on the mat.",
    "A dog barked loudly at the postman.",
    "Rain in Lisbon today.",
    "The meeting moved to Thursday because of a holiday.",
]

test = swadesh.PairedTest(
    named_systems=[("baseline", baseline), ("candidate", candidate)],
    metrics={"BLEU": swadesh.BLEU(), "chrF": swadesh.CHRF()},
    references=refs,
    test_type="bs",     # "bs": paired bootstrap; "ar": approximate randomization
    n_samples=1000,     # 0 uses the sacreBLEU default (1000 for bs, 10000 for ar)
)
signatures, results = test()
for i, name in enumerate(results["System"]):
    for metric in (k for k in results if k != "System"):
        r = results[metric][i]
        print(name, metric, round(r.score, 2), "p =", r.p_value)
```

`test()` returns `(signatures, results)`. `results["System"]` holds the system
names, and each metric key holds one result per system: `.score`, `.p_value`
(`None` for the baseline), and for bootstrap runs `.mean` and `.ci`, the 95%
interval half-width.

Two settings control reproducibility:

- `SACREBLEU_SEED` sets the seed (default `12345`). Set it to `none` to take
  entropy from the OS, which makes runs differ.
- The random draws use NumPy's `default_rng` with the same calls sacreBLEU makes.
  sacreBLEU uses the same generator, so a change in NumPy's generator would
  change both.

For a single corpus, `corpus_score` with `n_bootstrap` returns the bootstrap
mean and interval. `str(score)` shows them as `μ ± ci`:

```python
import swadesh

hyps = ["The cat sat on the mat.", "It is raining in Lisbon today.", "A dog barked."]
refs = [["The cat is sitting on the mat.", "It rains in Lisbon today.", "The dog barked loudly."]]

print(swadesh.BLEU().corpus_score(hyps, refs, n_bootstrap=1000))
```

## Reward functions for RL

`swadesh.rewards` builds reward functions for GRPO, PPO, RLOO and best-of-n.
Each one takes the batch of completions, plus the reference column as a
keyword argument, and returns one float per completion. Rewards are scaled to
`[0, 1]` by default (`scale=0.01`).

```python
from swadesh.rewards import bleu_reward, chrf_reward

reward = chrf_reward(ref_column="reference")       # chrF
print(reward(completions=["Das ist gut.", "Katze."],
             reference=["Das ist sehr gut.", "Die Katze sitzt."]))

chrfpp = chrf_reward(word_order=2)                 # chrF++
bleu = bleu_reward()                               # sentence BLEU, effective order on
```

With TRL, pass the same objects as `reward_funcs`. Here `reward_funcs` is the
list you would give to `GRPOTrainer`, and `reference` is a column in your
dataset:

```python
from swadesh.rewards import chrf_reward

reward_funcs = [chrf_reward(ref_column="reference")]

# GRPOTrainer(model=..., reward_funcs=reward_funcs, train_dataset=dataset, ...)
# TRL calls each reward function with the completions and the dataset columns
# as keyword arguments. Calling it directly looks like this:
rewards = reward_funcs[0](
    completions=["Das ist sehr gut.", "Das ist gut."],
    reference=["Das ist sehr gut.", "Das ist sehr gut."],
)
print([round(r, 3) for r in rewards])   # [1.0, 0.447]
```

Conversational completions (a list of messages per sample) are supported. The
content of the last message is scored. Several references per example are
passed as a list of references for each sample.

## Checking your data: `python -m swadesh.verify`

Before you trust swadesh on a new kind of data, run the verifier on your files.
It computes BLEU, chrF and chrF++ with both swadesh and sacreBLEU, and it
compares the corpus scores, the signatures, and every sentence score.

```bash
python -m swadesh.verify -r ref.txt -i hyp.txt
python -m swadesh.verify -r ref.txt -i hyp.txt -tok intl -lc
```

Pass `-r` more than once for several reference streams.

The verifier prints one line per metric and exits with 0 when every value is
identical, or 1 when it finds a mismatch. A mismatch is a bug. Report it with
the input files.

## The standalone Rust binary

`crates/swadesh-cli` builds a `swadesh` binary with no Python dependency. It is
for shell pipelines, CI and agents. It prints the same output as sacreBLEU's CLI
for the options it supports. Build it from the repository root:

```bash
cargo build --release -p swadesh-cli
target/release/swadesh ref.txt -i hyp.txt -m bleu chrf -f text
```

The binary is a separate build from the Python wheel. It accepts the BLEU and
chrF options of the Python CLI, with these differences:

- `--python-version X.Y` selects the `sum()` semantics that sacreBLEU would use
  on that CPython. The `--help` text gives the default as 3.12, because the
  binary has no Python interpreter of its own. Pass the version of the sacreBLEU
  run you want to match.
- Tokenizers: `none`, `13a`, `intl`, `zh` and `char` are built in. Other
  tokenizers (`-l ja`, `-l ko`, `ja-mecab`, `spm`, `flores*`) are rejected with
  a pointer to the Python package.
- Not supported, rejected with a message: test sets (`-t`, `--list`,
  `--echo`), `--detail`, bootstrap confidence (`-ci`), paired significance
  (`-par`, `-pbs`), TER (`-m ter`), several systems, gzip input, and encodings
  other than UTF-8.

`tests/test_cli.py` compares the binary's stdout byte for byte with sacreBLEU's
CLI, so that is the reference for what the binary should print.

Check the options with `target/release/swadesh --help`.

## Tokenizers

BLEU takes a `tokenize` argument, and sacreBLEU's default is `13a`. The
tokenizers split into two groups.

**Native** (tokenized in Rust): `13a`, `intl`, `zh`, `char`, `none`. Tokenization,
n-gram extraction and matching all run natively. These are the tokenizers in the
benchmark numbers below.

**Python-tokenized fallback**: `ja-mecab`, `ko-mecab`, `spm`, `flores101`,
`flores200` and `spBLEU-1K`. sacreBLEU tokenizes each segment in Python
(these need sacreBLEU's optional dependencies, such as MeCab or
sentencepiece, and some download a model on first use), and swadesh matches
the tokens natively. No benchmark numbers are published for these tokenizers.
The examples in this guide did not exercise them, because MeCab and
sentencepiece are not installed where they were run.

chrF has no tokenizer argument. Its character and word n-grams are always native.

## Threading

Corpus statistics (which also feed `sentence_scores`) use
[rayon](https://github.com/rayon-rs/rayon) once a batch has 2,048 or more
segments; below that they run on one thread. The pairwise matrix and the
bootstrap and approximate-randomization kernels always use rayon.

To limit threads, set `RAYON_NUM_THREADS` before the first swadesh call. rayon
reads it when its pool is first created, so setting it in the shell is the
simplest option:

```bash
RAYON_NUM_THREADS=4 python examples/06_large_corpus_benchmark.py --lines 20000
```

Changing the variable after the first call has no effect in that process.

## Performance

These numbers come from `bench/results/SUMMARY.md`. Hypotheses there are
synthetic perturbations of WMT22 en-de references, not real system output. Each
case was checked for exact equality with sacreBLEU 2.6.0 on the same machine.
The speedup is sacreBLEU's time divided by swadesh's time, truncated to one
decimal place. Times are best of three for corpus cases, and one run for
sentence and bootstrap cases.

For 20,000 lines on troy-1 (AMD Ryzen 9 7900X, 24 logical CPUs, Linux, x86_64):
BLEU corpus 98.5x, chrF corpus 261.2x, chrF++ corpus 309.3x, sentence chrF
297.8x, and BLEU with 1,000 bootstrap resamples over 5,000 lines 12.2x.

For the same case on GX10 (Cortex-X925 and Cortex-A725, 20 CPUs, Linux,
aarch64): BLEU corpus 54.0x, chrF corpus 373.7x, chrF++ corpus 125.1x.

On the Mac mini (Apple M4, 10 CPUs, macOS, arm64): BLEU corpus 79.1x, chrF
corpus 208.8x, chrF++ corpus 168.6x.

Speedups vary with the machine, the thread count, the case, and the input. The
runs labelled `20k-1thread` in the same file show the lower end:
BLEU corpus 22.1x, chrF corpus 54.4x on troy-1, and 6.8x for BLEU with
bootstrap on the Mac mini. Measure on your own data with
`examples/06_large_corpus_benchmark.py`.

## Limits

- `swadesh.TER` is native (tercom shifts and beam edit distance). The standalone Rust CLI accepts
  `-m ter` too, but neither path is byte-identical to sacreBLEU on every case yet - `TESTING.md`
  lists what is open.
- Corpus scores are computed by sacreBLEU's Python code on the interpreter that
  runs them. So swadesh's corpus scores equal sacreBLEU's on any one
  interpreter, and they can differ in the last bits between CPython versions.
  See [how-its-exact.md](how-its-exact.md#34-cpython-sum-semantics) and
  [python-version-drift.md](python-version-drift.md).
- `sentence_scores` and `pairwise` use the `sum()` semantics of the interpreter
  they run on, so they match sacreBLEU on that interpreter.
- For the exact guarantees, the supported versions, and what is tested, see
  [how-its-exact.md](how-its-exact.md).
