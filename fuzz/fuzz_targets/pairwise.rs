#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use swadesh_core::pairwise::{bleu_matrix, chrf_matrix};
use swadesh_core::score::{compute_bleu, BleuScoreConfig, Smooth, SumMode};
use swadesh_core::stats::{bleu_segment, chrf_segment, BleuConfig, ChrfConfig};
use swadesh_core::tokenize::Tokenizer;

#[derive(Arbitrary, Debug)]
struct Input {
    chrf: bool,
    tok: u8,
    char_order: u8,
    word_order: u8,
    beta: i64,
    whitespace: bool,
    eps: bool,
    smooth: u8,
    smooth_value: f64,
    effective_order: bool,
    max_order: u8,
    cands: Vec<String>,
    refs: Vec<String>,
}

fuzz_target!(|i: Input| {
    let cands: Vec<&str> = i.cands.iter().take(8).map(String::as_str).collect();
    let refs: Vec<&str> = i.refs.iter().take(8).map(String::as_str).collect();
    let (n, m) = (cands.len(), refs.len());
    if i.chrf {
        let cfg = ChrfConfig {
            char_order: (i.char_order % 11) as usize,
            word_order: (i.word_order % 5) as usize,
            beta: i.beta,
            whitespace: i.whitespace,
            eps_smoothing: i.eps,
        };
        let out = chrf_matrix(&cfg, &cands, &refs);
        assert_eq!(out.len(), n * m);
        // Consistency with the per-pair segment path (single reference each).
        let order = cfg.char_order + cfg.word_order;
        for (a, &c) in cands.iter().enumerate() {
            for (b, &r) in refs.iter().enumerate() {
                let mut st = Vec::new();
                chrf_segment(&cfg, c, &[r], &mut st);
                let want = swadesh_core::score::chrf_f_score(&st, order, cfg.beta, cfg.eps_smoothing);
                let got = out[a * m + b];
                assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()), "chrf cell ({a},{b}): {got} vs {want}");
            }
        }
    } else {
        let tokenizer = match i.tok % 5 {
            0 => Tokenizer::T13a,
            1 => Tokenizer::Intl,
            2 => Tokenizer::Zh,
            3 => Tokenizer::Char,
            _ => Tokenizer::None,
        };
        let max_order = 1 + (i.max_order % 8) as usize;
        let scfg = BleuScoreConfig {
            smooth: [Smooth::None, Smooth::Floor, Smooth::AddK, Smooth::Exp][(i.smooth % 4) as usize],
            smooth_value: i.smooth_value,
            effective_order: i.effective_order,
            max_order,
            sum_mode: SumMode::Neumaier,
        };
        let out = bleu_matrix(tokenizer, &scfg, &cands, &refs);
        assert_eq!(out.len(), n * m);
        let bcfg = BleuConfig { tokenizer, max_order };
        for (a, &c) in cands.iter().enumerate() {
            for (b, &r) in refs.iter().enumerate() {
                let mut st = Vec::new();
                bleu_segment(&bcfg, c, &[r], &mut st);
                let fst: Vec<f64> = st.iter().map(|&x| x as f64).collect();
                let want = compute_bleu(&fst, &scfg).score;
                let got = out[a * m + b];
                assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()), "bleu cell ({a},{b}): {got} vs {want}");
            }
        }
    }
});
