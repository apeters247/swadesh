**Title:** A bit-exact Rust backend for sacreBLEU (BLEU/chrF/TER, 22–46× per core) — interest in upstreaming or linking?

Hi Matt and sacreBLEU maintainers,

I built **swadesh**, a re-implementation of sacreBLEU 2.6.0's hot paths in Rust. It returns exactly
the same numbers: every statistic and score compares `==`, signatures and CLI output are
byte-identical, and it is 22–46× faster on one core (up to 331× multi-threaded).

It is built *on* sacreBLEU rather than beside it: `swadesh.BLEU` subclasses `sacrebleu.BLEU` and only
replaces `_extract_corpus_statistics`, so corpus scores are still computed by sacreBLEU's own code.
Testing is differential: hypothesis fuzzing, WMT20–24 data in 22 languages (about 9M sentence comparisons,
zero mismatches), Python 3.10–3.14 × numpy 1/2, and x86/ARM/Apple Silicon.

Reproducing sacreBLEU exactly turned up a few things you may want to know about:

1. `tokenizer_zh._UCODE_RANGES` contains `u' 0'` and `u'⾀0'`. These are two-character
   strings, so U+2001–U+2A6D (general punctuation, arrows, math, currency) are treated as Chinese
   characters. swadesh reproduces this, since changing it would change published `tok:zh` scores.
2. CPython 3.12 (and again 3.14) changed float `sum()`, which `compute_bleu` uses for the
   geometric mean. sacreBLEU 2.6.0's sentence BLEU differs between Python 3.11 and 3.12 on 15.8% of
   WMT22 segments (≤ 9 ulps), and some corpus scores differ too. `math.fsum` would make it
   version-independent, but it would also change scores at the ulp level.
3. TER preprocesses references twice (`_cache_references`, then `_extract_reference_info`), and the
   TER tokenizer isn't idempotent (`'s` is re-split).

Would you be open to (a) a link from the README, or (b) an optional `sacrebleu[fast]` extra that
uses swadesh for statistics when installed? I'm happy to adapt it to whatever you'd accept, and to
upstream the findings above as separate issues.

Code: https://github.com/apeters247/swadesh · write-up: docs/blog-post.md · check on your data:
`python -m swadesh.verify -r ref.txt -i hyp.txt`
