# Upstream PR: opt-in `SWADESH_PATCH` for lm-evaluation-harness

Ready to submit, but submitted only once the two-liner has users — the numbers below are the ask's
justification, and they are better with a download count next to them.

## The patch

`lm-eval-swadesh.patch` in this directory, generated against `EleutherAI/lm-evaluation-harness` at
commit `d6de8164` (main, 2026-11-27), touching only `lm_eval/api/metrics.py`:

```python
if os.environ.get("SWADESH_PATCH"):
    try:
        import swadesh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SWADESH_PATCH is set but swadesh is not installed. Install it with `pip install swadesh` "
            "or unset SWADESH_PATCH."
        ) from exc
    swadesh.patch()
```

Why this shape:

- **off by default**, so nothing changes for anyone who does not ask;
- **no new dependency** — `swadesh` is imported only when the variable is set, and the harness keeps
  requiring exactly what it requires today;
- **loud and specific** if the variable is set and the package is missing, because a silent no-op is
  the one failure mode that would waste someone's afternoon.

## Verification, run against a clone with the patch applied

```
PYTHONPATH=<clone> python -c "import lm_eval.api.metrics as M; ..."      # 2,000 segments
```

| | `sacrebleu.BLEU.__module__` | `bleu(items)` | wall time |
|---|---|---|---|
| without `SWADESH_PATCH` | `sacrebleu.metrics.bleu` | 5.105133 | 0.132 s |
| with `SWADESH_PATCH=1` | `swadesh` | **5.105133** | **0.010 s** |

`git apply --check` clean against the commit above. With `SWADESH_PATCH=1` in an environment that lacks
swadesh, the import stops with the message in the patch.

At the corpus sizes that matter (24 threads, fresh corpus per run, lm-eval's own aggregation functions):

| metric | segments | sacreBLEU | swadesh | speedup | identical |
|---|---:|---:|---:|---:|---|
| BLEU | 20,000 | 1.4475 s | 0.0417 s | 34.7× | yes |
| chrF | 20,000 | 2.9863 s | 0.0404 s | 74.0× | yes |
| chrF++ | 20,000 | 3.4658 s | 0.0446 s | 77.7× | yes |
| TER | 1,000 | 5.0463 s | 0.0079 s | 638× | yes |

Multi-reference corpora (the `_sacreformat` path with a list of references per segment) are identical
too: 3,000 segments, two references each, BLEU 5.590394 / chrF 42.694656 / chrF++ 41.898980.

## PR title and body

**Title**

```
metrics: opt-in SWADESH_PATCH to route the BLEU/chrF/TER aggregation through swadesh
```

**Body**

```
The translation metrics in `lm_eval/api/metrics.py` call `sacrebleu.corpus_bleu(...).score` and its
chrF/chrF++/TER siblings, and on a 20,000-segment corpus those calls are the slowest part of a
translation eval.

This adds an opt-in environment variable that routes them through swadesh, a Rust implementation that
returns bit-identical results (same floats, same signature strings, compared with `==`). It is off by
default, adds no dependency, and changes no number:

| metric | segments | sacrebleu | with SWADESH_PATCH=1 | identical |
|---|---:|---:|---:|---|
| BLEU    | 20,000 | 1.4475 s | 0.0417 s | yes |
| chrF    | 20,000 | 2.9863 s | 0.0404 s | yes |
| chrF++  | 20,000 | 3.4658 s | 0.0446 s | yes |
| TER     |  1,000 | 5.0463 s | 0.0079 s | yes |

Multi-reference corpora (the `_sacreformat` path) were checked the same way and are identical.

If `SWADESH_PATCH` is set and swadesh is not installed, the import raises a message naming the fix
rather than silently not patching.

swadesh is AGPL-3.0 (free for research use; commercial licence free below USD 5M revenue). Because it
is optional and off by default, nothing about the harness's licensing changes.
```

## Before submitting

- Rebase the patch onto whatever main is then; the hunk is 13 lines in a stable corner of the file, so
  it will apply with trivial fuzz.
- Quote the adoption number next to the speed table (PyPI downloads, or the repos that adopted it).
- Offer the alternative if they prefer it: a `--swadesh` CLI flag instead of an environment variable.
  The env var was chosen because it needs no argument parsing and works for the library API too.