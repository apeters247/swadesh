# Getting swadesh into the stacks that already use sacreBLEU

Phase 1 of the launch plan. The goal is not a PR in every repo; it is to make the one-line switch
work, prove it on the libraries people actually run, and only then ask upstream for two lines.

## The switch

```python
import swadesh
swadesh.patch()          # sacrebleu.* now resolves to swadesh; unpatch() restores
```

`patch()` replaces the **class objects** in `sacrebleu`, `sacrebleu.metrics`, `sacrebleu.metrics.bleu`
/ `.chrf` / `.ter` and `sacrebleu.compat` with subclasses of both swadesh's class and sacreBLEU's, so
`isinstance` checks downstream keep working. It does not touch the module-level functions
(`sacrebleu.corpus_bleu`), because it does not have to: those look `BLEU` up in their module globals
at call time, so they pick up the replaced class.

### Which call shapes it reaches (measured, 2026-11-27)

`tools`-free check, run against the 0.1.0 wheel and sacreBLEU 2.6.0:

| shape, as the libraries write it | reached | note |
|---|---|---|
| `sacrebleu.BLEU(...)` — lighteval | **yes** | attribute looked up at call time |
| `sacrebleu.corpus_bleu(hyps, refs)` — lm-eval, COMET | **yes** | the compat function's global lookup |
| `sacrebleu.metrics.BLEU(...)` | **yes** | |
| `from sacrebleu import corpus_bleu` *before* the patch — fairseq's example scripts | **yes** | a bound *function* still looks its class up at call time: measured 1.0881 s -> 0.0076 s, 144.1x, identical |
| `from sacrebleu.metrics import BLEU` *before* the patch | **no** | a bound *class* is the one case that escapes; patch before that import, or import `swadesh` instead |

Every shape returns the identical number either way, which is the point: patching cannot change a
result, only where it is computed.

## lm-evaluation-harness — verified, 34–638× on its own functions

`lm_eval/api/metrics.py` calls `sacrebleu.corpus_bleu(preds, refs).score`, `corpus_chrf`,
`corpus_chrf(..., word_order=2)` and `corpus_ter`. Measured through those functions, unfrozen corpus
generated per run, single process, 24 threads, lm-eval 0.4.13 installed with `--no-deps`:

| metric | segments | sacreBLEU | swadesh | speedup | identical |
|---|---:|---:|---:|---:|---|
| BLEU | 20,000 | 1.4475 s | 0.0417 s | **34.7×** | yes |
| chrF | 20,000 | 2.9863 s | 0.0404 s | **74.0×** | yes |
| chrF++ | 20,000 | 3.4658 s | 0.0446 s | **77.7×** | yes |
| TER | 1,000 | 5.0463 s | 0.0079 s | **638×** | yes |

Scope, stated plainly: this is the harness's *metric layer*, called directly on the same inputs. A
full `lm_eval` run needs a model, `datasets` and torch, which is why the verification used a
`--no-deps` install and drove the functions the harness calls.

The harness's multi-reference path (`_sacreformat` with a list of references per segment, which is how
WMT-style multi-ref corpora arrive) was checked the same way: 3,000 segments with two references each,
BLEU 5.590394, chrF 42.694656, chrF++ 41.898980 - identical before and after the patch.

### Three ways to apply it, in increasing order of commitment

**1. Two lines in your own runner** (works today, no fork):

```python
import swadesh; swadesh.patch()      # before lm_eval imports its metrics
from lm_eval import evaluator
```

**2. `sitecustomize`, to accelerate the `lm_eval` CLI without touching it** — the interpreter imports
`sitecustomize` before anything else:

```bash
mkdir -p /tmp/swadesh-patch
printf 'import swadesh\nswadesh.patch()\n' > /tmp/swadesh-patch/sitecustomize.py
PYTHONPATH=/tmp/swadesh-patch lm_eval --model hf --tasks wmt16-en-ro ...
```

Verified: with that on `PYTHONPATH`, `sacrebleu.BLEU` is swadesh's class at startup and lm-eval's
own `bleu()` runs at ~0.021 s for 20,000 segments instead of ~1.45 s.

**3. Two lines upstream**, only after 1 and 2 have users. The patch is written and checked:
`docs/upstream/lm-eval-swadesh.patch` (applies cleanly to `lm-evaluation-harness` at `d6de8164`) and the
PR text beside it. Verified against a clone: without the variable `sacrebleu.BLEU` is sacreBLEU's and
`bleu()` takes 0.132 s; with `SWADESH_PATCH=1` it is swadesh's and takes 0.010 s, on the same 5.105133.

```python
# lm_eval/api/metrics.py, top of file
if os.environ.get("SWADESH_PATCH"):
    import swadesh; swadesh.patch()
```

Optional, off by default, no new dependency. That is the smallest change worth proposing, and it is
worth proposing only with adoption numbers behind it.

## lighteval — call shape verified, execution not

`src/lighteval/metrics/metrics_corpus.py` does `import sacrebleu` and then `sacrebleu.BLEU(...)`,
`sacrebleu.CHRF(...)`, `sacrebleu.CHRF(word_order=2)` **inside a method**, i.e. attribute lookup at
call time. That is the shape patch() reaches (row 1 of the table above). Not executed inside lighteval
here: its install pulls torch, and the shape check above is the part that could have failed.

One risk specific to lighteval is cleared: it passes `trg_lang` (`"zh"`, `"ja"`, `"ko"`) to the BLEU
constructor, so the patched class has to accept it and keep choosing the tokenizer by language. It
does: `sacrebleu.BLEU(trg_lang="zh")` after the patch returns the same
`BLEU = 43.01 100.0/80.0/50.0/16.7 (BP = 0.846 ...)` as before, through a class whose module is
`swadesh.metrics`. `tokenize=`, `lowercase=` and `effective_order=` were checked the same way.

## COMET (Unbabel) — there is nothing to accelerate, and the dependency is a red herring

`unbabel-comet` 2.2.7 does declare `sacrebleu>=2.0.0,<3.0.0` as a hard requirement (PyPI metadata), so
it looked like a target. Reading the installed package (2.2.7, `--no-deps`) settles it differently:

- the **only** imports are in `comet/cli/score.py` and `comet/cli/compare.py`, both
  `from sacrebleu.utils import get_reference_files, get_source_file` - the `-d/--sacrebleu_dataset`
  convenience for fetching a test set;
- there is no `chrf`/`CHRF` anywhere in the package. Its metrics are `RegressionMetrics`,
  `MCCMetric` and `WMTKendall`, all torch.

So `patch()` has nothing to reach: COMET's numbers come from a model, and patching would change nothing
(and no number). COMET belongs in the *ecosystem* column, not the integration column - the right ask
there is not a patch but a docs line recommending swadesh for the BLEU/chrF baselines that get reported
next to a COMET score.

## fairseq — both of its paths verified

`fairseq` 0.12.2 requires `sacrebleu>=1.4.12`, and it reaches sacreBLEU two different ways. Both are
covered:

- `fairseq/scoring/bleu.py` does `import sacrebleu` inside the scorer's constructor and keeps the
  module, then calls through it - a module attribute at call time (row 2 of the table).
- `fairseq/examples/translation_moe/score.py` does `from sacrebleu import corpus_bleu as _corpus_bleu`
  at import time. That is the bound-function case, measured 1.0881 s -> 0.0076 s on 20,000 segments
  (144.1x) with an identical score.

## NeMo — dependency verified, usage not inspected

`nemo-toolkit` 3.0.0 pulls sacrebleu in its `asr`, `tts`, `speechlm2` and `all` extras (PyPI metadata).
Not inspected and not run: the package is large and its metric paths are ASR-side (WER/BLEU on speech
outputs). Treat the mechanism as expected, not as verified.

## What this does not reach

- **Vendored arithmetic** — torchmetrics' BLEU, nltk's `bleu_score`, `jiwer`: they have their own
  implementations, are not bit-identical to sacreBLEU, and are not ours to patch.
- **The `-t` test-set downloads and the mecab/spm tokenizers** still call sacreBLEU (and its extras);
  `swadesh[sacrebleu]` provides them.
- **A class imported before the patch** — see the last row of the table.

## Order of work

1. lm-eval two-liner and sitecustomize recipe: documented here; ask their maintainers to link it in
   the harness docs.
2. lighteval: run one task with the patch, then the same two-line PR.
3. NeMo: inspect and run one ASR scoring path before claiming anything.
4. Only then the upstream PRs, each with a number. COMET is not in this list any more - see above.