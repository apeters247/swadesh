# Decision: do not add native `ja-mecab` / `ko-mecab` / `spm` tokenizers

Date: 2026-11-27. Status: decided, with measurements below. Affects the shortlist item s3 in
`docs/feature-ideas.md`.

## The question

swadesh's native tokenizer set is `{none, 13a, intl, zh, char}`. The remaining sacreBLEU tokenizers -
`ja-mecab`, `ko-mecab`, `spm`, `spm32k` - are served by delegating to sacreBLEU's own tokenizer when
it is installed. Should the Rust core implement them, to remove that last dependency for those
configurations?

## Measurements (troy-1, 2026-11-27)

1. **The native API refuses them, loudly.** `swadesh.tokenize(lines, "ja-mecab")` raises
   `ValueError: unsupported native tokenizer "ja-mecab"` (same for `ko-mecab`, `spm`, `spm32k`). The
   metric classes route those names to sacreBLEU's tokenizer instead.

2. **The project's own environment cannot even exercise them.** `~/venv311` has sacreBLEU 2.6.0 but
   no `mecab`/`ipadic`/`sentencepiece`, so the *reference* cannot run `ja-mecab`/`spm` there; there is
   nothing to compare a port against, and the clean-room rule forbids implementing an oracle behaviour
   that has not been measured.

3. **With the extras installed, the fallback is already byte-exact.** In a throwaway venv with
   `mecab-python3==1.0.12` + `ipadic==1.0.0`, `swadesh.BLEU(tokenize="ja-mecab")` equals
   `sacrebleu.BLEU(tokenize="ja-mecab")` exactly: identical input gives
   `BLEU = 100.00 100.0/100.0/100.0/100.0` from both, and a reworded reference gives
   `65.80370064762461` from both. The reference segmentation of
   `こんにちは世界。私の名前はアンドリューです。` is
   `こんにちは 世界 。 私 の 名前 は アンドリュー です 。`

4. **That segmentation is a dictionary artefact.** The tokens above are ipadic entries; the
   reference pins its own versions of `mecab-python3` and `ipadic`. A byte-exact Rust port would have
   to reproduce a *specific* dictionary and MeCab's Viterbi behaviour, and stay exact across
   dictionary upgrades - the same class of problem as transcribing Moses' `tokenizer.perl`, which
   this project does not do. `spm`/`spm32k` additionally need a model file whose exact bytes matter.

## Decision

**Do not implement them natively now.** Reasons, in order of weight:

1. **Parity cannot be promised.** The product is bit-exactness; a port whose output depends on a
   dictionary version we do not control would be exact today and silently wrong after any dictionary
   bump. That is worse than an honest delegation.
2. **The dependency is already optional and already exact.** `sacrebleu` is an extra
   (`swadesh[sacrebleu]`), the core needs only numpy, and where the extra is present the numbers are
   identical (measurement 3). Nothing is missing for a user who needs Japanese or a sentencepiece
   model today.
3. **Licence and build weight.** `mecab-python3` is tri-licensed (BSD/LGPL/GPL); static linking into
   a dual-licensed AGPL+commercial product needs real care, and `sentencepiece` pulls a C++ toolchain
   into every wheel build.
4. **Unverifiable in this environment** (measurement 2), so it would ship unmeasured - against the
   project's own rules.

## What would change the decision

A concrete user need, *plus* a pinned, licence-clean dictionary with a documented exact-match harness
that compares the port against the reference on a fixed corpus and fails loudly on any dictionary
change. Absent that, delegation is the honest answer.

## Consequences, recorded

* `swadesh.tokenize` keeps refusing these names with `ValueError` (never a silent fallback to a
  different segmentation).
* The metric classes keep delegating to sacreBLEU's tokenizer for these names.
* The Rust CLI (which has no sacreBLEU) cannot run these tokenizers; it reports that as its one known
  CLI divergence.
