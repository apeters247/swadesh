// Proof of integration: BLEU / chrF / chrF++ / TER from swadesh-core in ~50 lines.
//
//   cargo new lte-bench && cd lte-bench
//   cargo add swadesh-core
//   # paste this into src/main.rs, then: cargo run -- ref.txt hyp.txt
//
// The numbers are the ones sacreBLEU prints for the same files, bit for bit: the per-segment work is
// native and the score arithmetic reproduces CPython's, including the interpreter-specific `sum()`.

use swadesh_core::{
    bleu_corpus_stats, chrf_corpus_stats, chrf_f_score, compute_bleu, ter_corpus_stats, BleuConfig,
    BleuScoreConfig, ChrfConfig, Smooth, SumMode, TerConfig, Tokenizer,
};

fn read(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(str::to_owned)
        .collect()
}

/// Element-wise sum of `n` rows of `width` columns: the whole corpus aggregation.
fn sum(rows: &[i64], width: usize) -> Vec<i64> {
    let mut total = vec![0i64; width];
    for row in rows.chunks(width) {
        for (t, v) in total.iter_mut().zip(row) {
            *t += v;
        }
    }
    total
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (refs_file, hyps_file) = (&args[1], &args[2]);
    let hyps = read(hyps_file);
    let refs = vec![read(refs_file)];
    let h: Vec<&str> = hyps.iter().map(String::as_str).collect();
    let r: Vec<Vec<&str>> = refs.iter().map(|v| v.iter().map(String::as_str).collect()).collect();

    // BLEU: rows are [sys_len, ref_len, correct.., total..]; summing them is the aggregate.
    let bleu_cfg = BleuConfig { tokenizer: Tokenizer::T13a, max_order: 4 };
    let total = sum(&bleu_corpus_stats(&bleu_cfg, &h, &r), 2 + 2 * 4);
    let stats: Vec<f64> = total.iter().map(|&v| v as f64).collect();
    let bleu = compute_bleu(
        &stats,
        &BleuScoreConfig {
            smooth: Smooth::Exp,
            smooth_value: 0.0,
            effective_order: false,
            max_order: 4,
            sum_mode: SumMode::Naive, // CPython <= 3.11; for_python(3, 12) matches 3.12/3.13
        },
    );

    // chrF / chrF++: three counters per order, then sacreBLEU's F-score.
    for (label, word_order) in [("chrF", 0usize), ("chrF++", 2)] {
        let cfg = ChrfConfig { char_order: 6, word_order, beta: 2, whitespace: false, eps_smoothing: false };
        let order = cfg.char_order + cfg.word_order;
        let total = sum(&chrf_corpus_stats(&cfg, &h, &r), 3 * order);
        let score = chrf_f_score(&total, order, cfg.beta, cfg.eps_smoothing);
        println!("{label:>7} {score:.4}");
    }

    // TER: one pair of (edits, reference length) per segment; the ratio is over the sums.
    let ter_cfg = TerConfig { normalized: false, no_punct: false, asian_support: false, case_sensitive: false };
    let rows = ter_corpus_stats(&ter_cfg, &h, &r).expect("each segment needs a reference");
    let (mut edits, mut len) = (0.0, 0.0);
    for pair in rows.chunks(2) {
        edits += pair[0];
        len += pair[1];
    }
    println!("{:>7} {:.4}", "TER", if len > 0.0 { 100.0 * edits / len } else { 0.0 });
    println!("{:>7} {:.4}", "BLEU", bleu.score);
}