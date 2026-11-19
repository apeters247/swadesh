"""Minimum Bayes risk (MBR) selection among sampled translations.

For one source sentence, sample N candidate translations from your model. MBR
picks the candidate with the highest mean metric score against all the others,
treating each as a pseudo-reference. swadesh.pairwise() computes the N x N
matrix natively and in parallel, with n-grams extracted once per candidate.

Run: python examples/03_mbr_decoding.py
"""
import swadesh

# In practice these come from num_return_sequences=N sampling.
candidates = [
    "The committee approved the new budget on Tuesday.",
    "The committee approved the budget on Tuesday.",
    "On Tuesday the committee passed the new budget.",
    "The board approved the new budget on Tuesday.",
    "Tuesday: the committee approved a budget.",
    "The committee rejected the new budget on Tuesday.",
]

# chrF is the default utility, and it works well on short segments.
best, utility = swadesh.mbr(candidates)
print("chrF MBR choice:", candidates[best])
print("expected utilities:", [round(float(u), 2) for u in utility])

# The raw matrix: M[i, j] = sentence chrF of candidates[i] against candidates[j].
matrix = swadesh.CHRF().pairwise(candidates)
print("matrix shape:", matrix.shape, "diagonal is 100:", bool((matrix.diagonal() == 100).all()))

# Sentence-level BLEU needs effective_order=True, or short segments score 0.
bleu = swadesh.BLEU(effective_order=True)
best_bleu, _ = swadesh.mbr(candidates, metric=bleu)
print("BLEU MBR choice:", candidates[best_bleu])

# pairwise() against an explicit reference list (for example, a pivot set):
pivots = ["The committee approved the new budget on Tuesday."]
print("chrF of each candidate against one pivot:", swadesh.CHRF().pairwise(candidates, pivots).ravel().round(1).tolist())
