#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use swadesh_core::stats::{bleu_segment, BleuConfig};
use swadesh_core::tokenize::Tokenizer;

#[derive(Arbitrary, Debug)]
struct Input {
    tok: u8,
    max_order: u8,
    hyp: String,
    refs: Vec<String>,
}

fuzz_target!(|i: Input| {
    let refs: Vec<&str> = i.refs.iter().take(4).map(String::as_str).collect();
    if refs.is_empty() {
        return;
    }
    let tokenizer = match i.tok % 5 {
        0 => Tokenizer::T13a,
        1 => Tokenizer::Intl,
        2 => Tokenizer::Zh,
        3 => Tokenizer::Char,
        _ => Tokenizer::None,
    };
    let max_order = (i.max_order % 9) as usize; // 0..=8
    let cfg = BleuConfig { tokenizer, max_order };
    let mut out = Vec::new();
    bleu_segment(&cfg, &i.hyp, &refs, &mut out);
    // Invariants of the statistics row: [hyp_len, ref_len, correct_1..N, total_1..N].
    assert_eq!(out.len(), 2 + 2 * max_order);
    assert!(out[0] >= 0 && out[1] >= 0);
    for n in 1..=max_order {
        let correct = out[1 + n];
        let total = out[1 + max_order + n];
        assert!(total >= 0 && correct >= 0 && correct <= total, "n={n} correct={correct} total={total}");
    }
});
