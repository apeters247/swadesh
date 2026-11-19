use swadesh_core::{bleu_corpus_stats, compute_bleu, BleuConfig, BleuScoreConfig, Smooth, SumMode, Tokenizer};

fn main() {
    let hyps = ["The cat sat on the mat."];
    let refs = vec![vec!["The cat is on the mat."]];
    let cfg = BleuConfig { tokenizer: Tokenizer::T13a, max_order: 4 };
    let stats = bleu_corpus_stats(&cfg, &hyps, &refs);
    let summed: Vec<f64> = stats.iter().map(|&x| x as f64).collect();
    let score = compute_bleu(
        &summed,
        &BleuScoreConfig { smooth: Smooth::Exp, smooth_value: 0.0, effective_order: false, max_order: 4, sum_mode: SumMode::for_python(3, 12) },
    );
    println!("BLEU = {:.2}", score.score);
}
