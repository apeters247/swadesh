"""Paired bootstrap test between two systems on the same test set.

swadesh.PairedTest has the same interface as sacrebleu.significance.PairedTest
and gives the same p-values and confidence intervals. The resampling is native.
The first system in the list is the baseline.

Run: python examples/05_significance.py
     SACREBLEU_SEED=42 python examples/05_significance.py   (change the seed)
"""
import swadesh

refs = [[
    "The cat is sitting on the mat.",
    "A dog barked loudly at the postman.",
    "It rains in Lisbon today.",
    "The meeting moved to Thursday because of the holiday.",
    "She bought fresh bread at the market this morning.",
    "The river floods every spring after heavy rain.",
    "Our train was delayed by two hours near Porto.",
    "He reads the newspaper before breakfast.",
]]

baseline = [
    "The cat sat on the mat.",
    "A dog barked loudly at the mailman.",
    "It is raining in Lisbon today.",
    "The meeting moved to Thursday due to the holiday.",
    "She bought fresh bread at the market in the morning.",
    "The river floods each spring after heavy rains.",
    "Our train was delayed two hours near Porto.",
    "He reads a newspaper before breakfast.",
]
candidate = [
    "The cat is sitting on the mat.",
    "A dog barked loudly at the postman.",
    "Rain in Lisbon today.",
    "The meeting moved to Thursday because of a holiday.",
    "She bought bread at the market this morning.",
    "The river floods every spring.",
    "The train was late near Porto.",
    "He reads the paper before breakfast.",
]

test = swadesh.PairedTest(
    named_systems=[("baseline", baseline), ("candidate", candidate)],
    metrics={"BLEU": swadesh.BLEU(), "chrF": swadesh.CHRF()},
    references=refs,
    test_type="bs",       # paired bootstrap resampling ("ar" for approximate randomization)
    n_samples=1000,       # resamples; 0 uses the sacreBLEU default (1000 for bs, 10000 for ar)
)
signatures, results = test()

for name, sig in signatures.items():
    print(f"{name} signature: {sig}")
print()
print(f"{'system':<12}{'metric':<8}{'score':>8}{'p-value':>10}{'95% CI':>14}")
for i, sys_name in enumerate(results["System"]):
    for metric in (k for k in results if k != "System"):
        r = results[metric][i]
        p = "baseline" if r.p_value is None else f"{r.p_value:.3f}"
        ci = "-" if r.ci is None else f"+/-{r.ci:.2f}"
        print(f"{sys_name:<12}{metric:<8}{r.score:>8.2f}{p:>10}{ci:>14}")
