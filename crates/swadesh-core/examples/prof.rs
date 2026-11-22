use std::time::Instant;
use swadesh_core::*;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let refs: Vec<String> = std::fs::read_to_string(path).unwrap().lines().map(|s| s.to_string()).collect();
    let mut lines = Vec::new();
    while lines.len() < 200_000 { lines.extend(refs.iter().cloned()); }
    let hyps: Vec<String> = lines.iter().map(|l| l.split(' ').rev().collect::<Vec<_>>().join(" ")).collect();
    let n = lines.len() as f64;
    let mut out = String::new();
    let t = Instant::now();
    let mut acc = 0usize;
    for l in &lines { tokenize_dispatch::tokenize_unjoined(Tokenizer::T13a, l, &mut out); acc += out.len(); }
    println!("13a tokenize      {:6.0} ns/line ({acc})", t.elapsed().as_nanos() as f64 / n);
    let t = Instant::now();
    let mut toks = 0usize;
    for l in &lines { tokenize_dispatch::tokenize_unjoined(Tokenizer::T13a, l, &mut out); toks += unicode::py_split(&out).count(); }
    println!("13a tok+split     {:6.0} ns/line ({toks} toks)", t.elapsed().as_nanos() as f64 / n);
    let cfg = BleuConfig { tokenizer: Tokenizer::T13a, max_order: 4 };
    let mut row = Vec::new();
    let t = Instant::now();
    for (h, r) in hyps.iter().zip(&lines) { swadesh_core::bleu_stats_segment(&cfg, h, &[r.as_str()], &mut row); }
    println!("bleu_segment 13a  {:6.0} ns/pair", t.elapsed().as_nanos() as f64 / n);
    let cfg0 = BleuConfig { tokenizer: Tokenizer::None, max_order: 4 };
    let t = Instant::now();
    for (h, r) in hyps.iter().zip(&lines) { swadesh_core::bleu_stats_segment(&cfg0, h, &[r.as_str()], &mut row); }
    println!("bleu_segment none {:6.0} ns/pair", t.elapsed().as_nanos() as f64 / n);
    let ccfg = ChrfConfig { char_order: 6, word_order: 0, beta: 2, whitespace: false, eps_smoothing: false };
    let t = Instant::now();
    for (h, r) in hyps.iter().zip(&lines) { swadesh_core::chrf_stats_segment(&ccfg, h, &[r.as_str()], &mut row); }
    println!("chrf_segment      {:6.0} ns/pair", t.elapsed().as_nanos() as f64 / n);
}
