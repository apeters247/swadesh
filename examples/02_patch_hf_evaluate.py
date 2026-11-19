"""Accelerate code that already calls sacrebleu, without editing it.

swadesh.patch() replaces sacrebleu's BLEU and CHRF classes (and its compat
functions) in place. Any library that does `import sacrebleu` afterwards gets
the swadesh implementation. Hugging Face `evaluate` is one example: its
"sacrebleu" metric calls sacrebleu at compute time.

Requires: pip install evaluate   (the metric script is fetched from the Hub
the first time; it needs network access once)

Run: python examples/02_patch_hf_evaluate.py
"""
import swadesh

swadesh.patch()  # call this before the code that uses sacrebleu runs

try:
    import evaluate
except ImportError:
    raise SystemExit(
        "This example needs Hugging Face evaluate. Install it with:\n"
        "    pip install evaluate\n"
        "swadesh.patch() itself works without it: sacrebleu.BLEU is now swadesh.BLEU."
    )

import sacrebleu.compat as compat

# sacrebleu's compat functions look up BLEU from their module globals at call
# time, so patch() is what routes them to swadesh.
assert compat.BLEU is swadesh.BLEU

hyps = ["The cat sat on the mat.", "A dog barked loudly at the mailman."]
# evaluate's sacrebleu metric takes one list of references per prediction.
refs_per_segment = [["The cat is sitting on the mat."], ["A dog barked loudly at the postman."]]

metric = evaluate.load("sacrebleu")
result = metric.compute(predictions=hyps, references=refs_per_segment)
print("BLEU from evaluate:", round(result["score"], 2))
print("n-gram counts:", result["counts"], "totals:", result["totals"], "brevity penalty:", result["bp"])

swadesh.unpatch()  # restore sacrebleu's original classes if you need them back
