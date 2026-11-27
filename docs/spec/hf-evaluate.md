# HuggingFace `evaluate` module: `evaluate.load(<path>)`

Status: additive adapter (shortlist item s1), written 2026-11-27 from HuggingFace's published
"Creating an EvaluationModule" documentation. No score changes; the adapter calls swadesh's own
`corpus_score`.

## Why it exists, and what it can and cannot be

`evaluate.load("bleu")` is how much of the community scores a model. A swadesh module lets the same
call reach the Rust kernels. There is one hard constraint: **`evaluate.load("name")` resolves `name`
on the HuggingFace Hub**, and this project does not publish anything there. So the shipping form is
`evaluate.load("<path-to-evaluate_module.py>")`, which `evaluate` supports for a local file, and the
Hub route is left to whoever wants to own that publication.

## What ships

**`swadesh.hf_metric.compute(predictions, references, metric="bleu")`** - the logic, with **no**
HuggingFace import, so it is testable and usable without `evaluate` installed. It:

* transposes HuggingFace's `references` (a list per prediction) into swadesh's corpus layout (a
  *stream* per reference slot), and raises a clear `ValueError` if the reference sets are ragged -
  swadesh's corpus statistics are rectangular by construction;
* maps `metric` (`"bleu"`, `"chrf"`, `"chrf++"`, `"ter"`, case-insensitive) to the metric class;
* returns `{name: score, "signature": str(metric.get_signature())}` where `name` is the metric's own
  (`BLEU`, `chrF2++`, `TER`), so a caller can read the number and its provenance.

**`swadesh/evaluate_module.py`** - the `evaluate.EvaluationModule` subclass, importing `datasets` and
`evaluate` and delegating to `hf_metric.compute`. It is the file a user points `evaluate.load` at.
Only this file needs the HuggingFace stack; nothing else in the package imports it.

**An optional extra.** `pip install "swadesh[hf]"` pulls `evaluate` for users who want the module.
It is optional, never a runtime dependency: the core libraries still need numpy and nothing else.

## Invariants (each asserted by a test)

1. For each metric, `hf_metric.compute` equals `corpus_score` on the same data (and sacreBLEU's).
2. `references` in HuggingFace layout (`[[r1], [r2], ...]`) and in swadesh's layout both produce the
   same number.
3. Ragged references raise `ValueError` rather than silently truncating.
4. An unknown `metric` raises `ValueError`.
5. With `evaluate` installed, `evaluate.load(<the module file>)` returns a module whose `compute`
   equals `hf_metric.compute` (this test skips when `evaluate` is absent).
