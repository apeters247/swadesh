#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use swadesh_core::stats::{chrf_segment, ChrfConfig};

#[derive(Arbitrary, Debug)]
struct Input {
    char_order: u8,
    word_order: u8,
    beta: i64,
    whitespace: bool,
    eps: bool,
    hyp: String,
    refs: Vec<String>,
}

fuzz_target!(|i: Input| {
    let refs: Vec<&str> = i.refs.iter().take(4).map(String::as_str).collect();
    if refs.is_empty() {
        return;
    }
    // char_order up to 10 (21-bit keys: wide fallback above 6), word_order up to 6 (32-bit keys: wide above 4).
    let char_order = (i.char_order % 11) as usize;
    let word_order = (i.word_order % 7) as usize;
    let cfg = ChrfConfig {
        char_order,
        word_order,
        beta: i.beta,
        whitespace: i.whitespace,
        eps_smoothing: i.eps,
    };
    let mut out = Vec::new();
    chrf_segment(&cfg, &i.hyp, &refs, &mut out);
    let order = char_order + word_order;
    assert_eq!(out.len(), 3 * order);
    for k in 0..order {
        let (h, r, m) = (out[3 * k], out[3 * k + 1], out[3 * k + 2]);
        assert!(h >= 0 && r >= 0 && m >= 0 && m <= h, "order {k}: hyp={h} ref={r} match={m}");
    }
});
