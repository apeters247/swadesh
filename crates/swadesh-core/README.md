# swadesh-core

Bit-exact Rust implementation of [sacreBLEU](https://github.com/mjpost/sacrebleu) 2.6.0's
BLEU, chrF and chrF++: tokenizers (`13a`, `intl`, `zh`, `char`, `none`), segment statistics,
score arithmetic (including CPython's version-specific `sum()` semantics), pairwise score
matrices and bootstrap kernels. Parallel via rayon (feature `parallel`, on by default;
disable it for `wasm32-unknown-unknown`).

```rust
use swadesh_core::{bleu_corpus_stats, compute_bleu, BleuConfig, BleuScoreConfig, Smooth, SumMode, Tokenizer};

let hyps = ["The cat sat on the mat."];
let refs = vec![vec!["The cat is on the mat."]];
let cfg = BleuConfig { tokenizer: Tokenizer::T13a, max_order: 4 };
let stats = bleu_corpus_stats(&cfg, &hyps, &refs);          // [sys_len, ref_len, correct.., total..] per segment
let summed: Vec<f64> = stats.iter().map(|&x| x as f64).collect();
let score = compute_bleu(&summed, &BleuScoreConfig {
    smooth: Smooth::Exp, smooth_value: 0.0, effective_order: false, max_order: 4,
    sum_mode: SumMode::for_python(3, 12),
});
println!("BLEU = {:.2}", score.score);
```

Results are identical (`==`) to sacreBLEU's on the same inputs; see the repository's
differential test suite. Python users want the `swadesh` package on PyPI instead.

License: Apache-2.0.
