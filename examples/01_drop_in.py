"""Drop-in use: swadesh has the same classes and functions as sacrebleu.

Change `import sacrebleu` to `import swadesh` and the calls below work unchanged.
The last lines check that the scores are identical to sacreBLEU 2.6.0 on this
interpreter.

Run: python examples/01_drop_in.py
"""
import sacrebleu

import swadesh

hyps = [
    "The cat sat on the mat.",
    "A dog barked loudly at the mailman.",
    "It is raining in Lisbon today.",
]
# refs is a list of reference *streams*: one list per reference set,
# with one string per segment. Here there is a single reference stream.
refs = [[
    "The cat is sitting on the mat.",
    "A dog barked loudly at the postman.",
    "It rains in Lisbon today.",
]]

bleu = swadesh.BLEU()
chrf = swadesh.CHRF()
chrfpp = swadesh.CHRF(word_order=2)

print(bleu.corpus_score(hyps, refs))
print(chrf.corpus_score(hyps, refs))
print(chrfpp.corpus_score(hyps, refs))
print(swadesh.corpus_bleu(hyps, refs))
print(swadesh.sentence_bleu(hyps[0], [refs[0][0]]))

# Same scores, same signatures, as sacreBLEU 2.6.0.
assert str(bleu.corpus_score(hyps, refs)) == str(sacrebleu.BLEU().corpus_score(hyps, refs))
assert str(chrfpp.corpus_score(hyps, refs)) == str(sacrebleu.CHRF(word_order=2).corpus_score(hyps, refs))
ref_bleu = sacrebleu.BLEU()
ref_bleu.corpus_score(hyps, refs)  # a signature needs the reference count first
assert str(bleu.get_signature()) == str(ref_bleu.get_signature())
print("identical to sacrebleu:", sacrebleu.__version__)
