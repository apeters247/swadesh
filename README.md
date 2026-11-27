# swadesh

**sacreBLEU, bit for bit, in Rust.** BLEU, chrF, chrF++ and TER with the same scores, the same signatures
and the same CLI, and much faster: up to 316× on a 24-thread desktop and 11–36× on one core (measured 2026-11-26).

**Status (2026-11-26).** Bit-exact on the WMT-style corpora, on the 494-case parity suite and on the
seven-file differential selection (**0 failing cases**): TER, BLEU, chrF and chrF++ match sacreBLEU on
every input exercised, including the long unrelated-reference cases that were the last divergence.
Ten more language pairs - Gujarati, Kazakh, Lithuanian, Arabic in both directions, Japanese and
Korean, German-to-French and Turkish - are byte-identical too, on the evidence in
`docs/multilingual-parity.md`. `TESTING.md` records the evidence; `docs/lab-notebook.md` records
the work. The CLI suite is green
too - `tests/test_cli.py` 685 passed, 5 skipped, 4 xfailed - when its fixtures are pointed at the checkout's data
with `SWADESH_DATA=bench/data` (they default to `/var/www/llmeval/bench/data`).

<p align="center"><img src="docs/img/benchmark.svg" alt="swadesh vs sacreBLEU on 100k segments: BLEU 117x, chrF 314x, chrF++ 283x faster, identical scores" width="720"></p>

```bash
pip install swadesh
```

The library needs only NumPy. BLEU, chrF, chrF++, TER and the paired significance tests all run
with sacreBLEU absent, and so does the `swadesh` command line - it runs the compiled code in process
(`_swadesh.run_cli`), so there is no Python-side CLI and nothing to keep in step. `sacrebleu` is an
extra, needed by `patch()`, `verify.py` and the six tokenizers that use mecab or sentencepiece
(`pip install "swadesh[sacrebleu]"`).

```python
import swadesh as sacrebleu   # that's the whole migration

bleu = sacrebleu.corpus_bleu(hyps, [refs])
print(bleu)            # the same BLEUScore object, value and string sacreBLEU gives you
```

Or keep your code and every library that imports sacreBLEU (Hugging Face `evaluate`,
fairseq, Marian/OpenNMT scripts…) unchanged:

```python
import swadesh
swadesh.patch()        # sacrebleu.* now runs on swadesh
```

```bash
swadesh ref.txt -i hyp.txt -m bleu chrf      # the sacreBLEU CLI, same flags, same output
```

## Identical, not "close"

swadesh is a differential re-implementation of [sacreBLEU](https://github.com/mjpost/sacrebleu) 2.6.0.
Every statistic, score, signature string and printed line is compared with `==` against
sacreBLEU, never with a tolerance:

- Property-based fuzzing (hypothesis) over an adversarial alphabet: Unicode whitespace,
  CJK, combining marks, `&amp;`/`<skipped>`/`-\n`, Turkish `İ`, final sigma, astral characters.
- Real WMT22 test sets across 7 language pairs and all native tokenizers.
- Python 3.10–3.14 × numpy 1.26 / 2.x, on x86-64 Linux, ARM64 Linux and Apple M4.
- A byte-for-byte CLI comparison suite of 448 cases against the `sacrebleu` CLI; the current
  pass count is in `TESTING.md`.

Don't take our word for it. Check it on your own data:

```bash
python -m swadesh.verify -r ref.txt -i hyp.txt
# BLEU    corpus identical  sentences 3000/3000 identical  (29x faster corpus scoring)
# chrF    corpus identical  sentences 3000/3000 identical  (46x faster corpus scoring)
# chrF++  corpus identical  sentences 3000/3000 identical  (35x faster corpus scoring)
# ALL IDENTICAL
```

Bit-exact means reproducing sacreBLEU's quirks too. Some of them:

* The `zh` tokenizer's range table contains `u'\u20000'`, which in Python is the
  two-character string `'\u2000' + '0'`. String comparison therefore treats
  U+2001–U+2A6D (general punctuation, arrows, math symbols…) as "Chinese characters". swadesh does too.
* The `intl` tokenizer's `\p{P}`/`\p{S}`/`\p{N}` classes come from tables generated from
  the `regex` module itself, and Python's `str.split()` whitespace set is reproduced exactly.
* CPython 3.12 changed `sum()` of floats to compensated summation, and 3.14 changed it again.
  BLEU's geometric mean goes through `sum()`, so sacreBLEU's own scores differ in the last
  bits between Python versions ([details](docs/python-version-drift.md)). swadesh detects the
  interpreter and matches it.
* Bootstrap resampling uses numpy's PCG64 stream and float32 accumulation. swadesh draws
  the same indices and reproduces numpy's float32 summation order, so confidence intervals
  and paired-test p-values are identical too.

How this works is described in [docs/how-its-exact.md](docs/how-its-exact.md).

## Benchmarks

Hypotheses are perturbed WMT22 en-de references (word drops, swaps and reversals, fixed seed).
Every swadesh result below was checked `==` against sacreBLEU 2.6.0 on the same machine.
Times are best of 3 (BLEU/chrF corpus) or a single run (TER, sentence-level, bootstrap).
Source: [`bench/results/SUMMARY.md`](bench/results/SUMMARY.md) · reproduce with `bench/get_data.sh && python bench/bench.py`.

**AMD Ryzen 9 7900X, idle machine** (`bench/results/speed-2026-11-26-100k.json`,
`speed-2026-11-26-20k.json`, `speed-2026-11-26-20k-1core.json`; load average 0.39 at the start)

| | sacreBLEU | swadesh, 24 threads | speedup | swadesh, 1 core (20k) | speedup |
|---|---:|---:|---:|---:|---:|
| corpus BLEU (100k) | 5.82 s | 0.050 s | **117×** | 0.085 s | 13.7× |
| corpus chrF (100k) | 18.56 s | 0.059 s | **314×** | 0.121 s | 28.4× |
| corpus chrF++ (100k) | 20.80 s | 0.074 s | **283×** | 0.164 s | 24.1× |
| corpus TER (10k) | 6.39 s | 0.056 s | **115×** | 0.181 s | 35.6× |
| sentence BLEU ×20k | 0.97 s | 0.0108 s | **90×** | 0.087 s | 11.0× |
| sentence chrF ×20k | 2.82 s | 0.0120 s | **234×** | 0.123 s | 22.7× |
| BLEU + 1000 bootstrap resamples (5k) | 0.47 s | 0.0102 s | **46×** | 0.031 s | 15.0× |

The single-core column is the 20k run pinned with `taskset -c 0`, against sacreBLEU's 20k timings.
A feature-by-feature comparison with sacreBLEU (52 rows: every tokenizer and metric option, sentence-level,
significance tests, MBR, the three CLIs, import time and peak memory; all identical) is in
[`bench/results/compare-macmini-m4-v2.md`](bench/results/compare-macmini-m4-v2.md); regenerate it with
`python bench/compare.py`. Scaling to 10M segments, pathological inputs and concurrency are in
[`docs/stress-report.md`](docs/stress-report.md).
Earlier builds were also benchmarked on an ARM Cortex-X925 (GX10, up to 379× on chrF) and an
Apple M4 (up to 242×); see [`bench/results/SUMMARY.md`](bench/results/SUMMARY.md).

The corpus repeats the 2,037 WMT22 references, which helps sacreBLEU's tokenizer
`lru_cache`, so these numbers understate the speedup on non-repetitive data.

## Beyond sacreBLEU

Things sacreBLEU doesn't have, all bit-identical to doing the same thing with sacreBLEU in a loop:

```python
import swadesh
from swadesh import BLEU, CHRF, mbr

# Every sentence score at once, in parallel (e.g. data filtering, analysis)
scores = CHRF().sentence_scores(hyps, [refs])                  # == [sentence_score(h, [r]).score ...]

# n x m score matrices for MBR decoding / best-of-n reranking (128x128 chrF: 4.9 s -> 31 ms)
M = CHRF().pairwise(candidates)                                 # candidates vs candidates
best, utility = mbr(candidates)                                 # minimum Bayes risk pick

# Reward functions for RL fine-tuning (TRL GRPO-compatible signature)
from swadesh.rewards import chrf_reward, bleu_reward
reward = chrf_reward(ref_column="reference")                    # completions=[...], reference=[...] -> [0..1]

# Native bootstrap CIs and paired significance tests (identical p-values)
BLEU().corpus_score(hyps, [refs], n_bootstrap=1000)
swadesh.PairedTest([("base", base), ("new", new)], {"BLEU": BLEU(references=[refs])}, None, test_type="bs")()

# Refuse the silent truncation instead of publishing a number from the wrong segments
with swadesh.strict():                    # or SWADESH_STRICT=1
    BLEU().corpus_score(hyps, [refs])     # LengthMismatchError if the streams differ in length

# A reproducible bootstrap interval, from the process; its own signature says which seed
with swadesh.seed(7):                     # or SACREBLEU_SEED=7
    BLEU().corpus_score(hyps, [refs], n_bootstrap=1000)   # ...|bs:1000|seed:7|...

# Every system scored and paired in one call: scores, intervals, p-values, a ranking
from swadesh import compare
table = compare({"base": base_hyps, "new": new_hyps}, [refs])   # print(table)
table.best()                             # best system by the first metric
table.meaningful(alpha=0.05)             # systems that differ from the baseline
table.to_dict()                          # JSON-able, for a report or a DataFrame
```

- **MCP server for agents.** `pip install "swadesh[mcp]"`, then `claude mcp add swadesh -- swadesh-mcp`.
  Agents get `corpus_score` (with sacreBLEU signatures and optional bootstrap CIs), `sentence_scores`,
  `mbr_select` and `tokenize` tools, all returning sacreBLEU-identical numbers.
- **`compare()` for a system table.** The paired-test pieces assembled: per-system score, bootstrap
  interval, p-value against the baseline and a ranking, in text or as a dict. Every number comes from
  the same kernels and resamplers, so a `compare()` line is reproducible from its signature.
- **Exact tokenizers.** `swadesh.tokenize(lines, "13a")` returns sacreBLEU's tokenization, in parallel.
- **In the browser.** The CLI compiles to a roughly 349 KB WebAssembly module; `docs/playground/`
  is a static page that runs it.
- **Standalone binary.** `crates/swadesh-cli` is an executable with no Python, for CI pipelines and
  agents. It implements every sacreBLEU flag and is byte-identical to the sacreBLEU 2.6.0 CLI on
  `tests/test_cli.py` (685 passed, 0 failed) and on the parity harness's CLI section (76 passed, one
  registered divergence: the exit code for an unbuilt tokenizer).
- **Free-threaded Python.** The extension declares `gil_used = false`; on CPython 3.14t with the
  GIL disabled, concurrent threads score in parallel and stay bit-exact. (sacreBLEU's own `lxml`
  import re-enables the GIL unless you run with `PYTHON_GIL=0`.)
- **Rust crate.** `swadesh-core` gives tokenizers, statistics and score arithmetic for Rust programs.

## Coverage

| | status |
|---|---|
| BLEU: tokenizers `13a`, `intl`, `zh`, `char`, `none` | native |
| BLEU: `ja-mecab`, `ko-mecab`, `spm`, `flores101/200`, `spBLEU-1K` | tokenized by sacreBLEU in Python, then matched natively |
| chrF, chrF++ (all options) | native |
| bootstrap CIs, paired bootstrap, paired approximate randomization | native kernels, sacreBLEU's statistics |
| TER (all options: normalized, no-punct, asian support, case-sensitive) | native (tercom shifts + beam edit distance) |
| `-t` test-set downloads, all other CLI features | sacreBLEU (the `swadesh` Python CLI *is* the sacreBLEU CLI, accelerated) |

## Why "swadesh"?

In the 1950s the linguist and anthropologist Morris Swadesh compiled lists of core vocabulary
and compared languages by counting how many of those words they share. That technique,
lexicostatistics, measures similarity by shared units. BLEU and chrF do the same for a
translation and its reference: they count shared n-grams.

## Credits

swadesh reproduces [sacreBLEU](https://github.com/mjpost/sacrebleu) by Matt Post and contributors
(Apache-2.0). The Python metric layer used to inherit from sacreBLEU and has since been ported, so
the library API needs no sacreBLEU at all; what still touches it -- `patch()`, the command line,
`verify.py`, `significance.install()` and the mecab/sentencepiece tokenizers -- does so on use, and
`NOTICE` names each one. The Rust implementations were written from published specifications and
from measurement of sacreBLEU's output, not from its source; `docs/spec/` records each one. chrF is due to
Maja Popović; BLEU to Papineni et al. (2002). Please keep citing sacreBLEU
([Post, 2018](https://aclanthology.org/W18-6319/)) when you report scores: they *are* sacreBLEU scores.

## Licence

swadesh is dual-licensed:

* **AGPL-3.0-or-later** — free. Running it inside your organisation to score your own data, in CI,
  in research and in open-source work all cost nothing. See `LICENSE`.
* **Commercial licence** — for shipping swadesh inside a closed product, or offering it (or a
  modified version) as a network service without publishing that service's source. **Free below
  USD 5 million annual revenue**, paid above it; see `LICENSE.commercial`.

`docs/licensing.md` sets out which side of that line a given use falls on.
