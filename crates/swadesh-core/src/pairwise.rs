//! Pairwise (n x m) sentence-level score matrices, e.g. for minimum Bayes risk
//! (MBR) decoding, best-of-n reranking or consensus scoring. Every cell equals
//! sacreBLEU's `sentence_score(cands[i], [refs[j]]).score` exactly, but n-gram
//! counts are built once per sentence instead of once per pair.

use rustc_hash::FxHashMap;

use crate::ngram::{clipped_matches, count, AnyRuns};
use crate::score::{chrf_f_score, compute_bleu, BleuScoreConfig};
use crate::stats::{remove_punctuation, ChrfConfig};
use crate::tokenize_dispatch::{tokenize_unjoined, Tokenizer};
use crate::unicode::{is_py_space, py_rstrip, py_split};

struct Profile {
    runs: Vec<AnyRuns>,
    totals: Vec<i64>,
    len: i64,
}

fn chrf_profile(cfg: &ChrfConfig, s: &str, words: &mut FxHashMap<String, u32>) -> Profile {
    let chars: Vec<u32> = if cfg.whitespace {
        s.chars().map(|c| c as u32).collect()
    } else {
        s.chars().filter(|&c| !is_py_space(c)).map(|c| c as u32).collect()
    };
    let mut runs = Vec::with_capacity(cfg.char_order + cfg.word_order);
    for n in 1..=cfg.char_order {
        runs.push(count(&chars, n, 21));
    }
    if cfg.word_order > 0 {
        let ids: Vec<u32> = remove_punctuation(s)
            .into_iter()
            .map(|w| {
                let k = words.len() as u32;
                *words.entry(w.to_owned()).or_insert(k)
            })
            .collect();
        for n in 1..=cfg.word_order {
            runs.push(count(&ids, n, 32));
        }
    }
    let totals = runs.iter().map(|r| r.total() as i64).collect();
    Profile { runs, totals, len: 0 }
}

/// chrF matrix, row-major `cands.len() x refs.len()`.
pub fn chrf_matrix(cfg: &ChrfConfig, cands: &[&str], refs: &[&str]) -> Vec<f64> {
    let mut words = FxHashMap::default();
    let cp: Vec<Profile> = cands.iter().map(|s| chrf_profile(cfg, s, &mut words)).collect();
    let rp: Vec<Profile> = refs.iter().map(|s| chrf_profile(cfg, s, &mut words)).collect();
    let order = cfg.char_order + cfg.word_order;
    let m = refs.len();
    let mut out = vec![0f64; cands.len() * m];
    crate::par::for_each_chunk(&mut out, m.max(1), |(i, row)| {
        let mut stats = Vec::with_capacity(3 * order);
        for (j, cell) in row.iter_mut().enumerate() {
            stats.clear();
            for o in 0..order {
                let r_total = rp[j].totals[o];
                stats.push(if r_total > 0 { cp[i].totals[o] } else { 0 });
                stats.push(r_total);
                stats.push(clipped_matches(&cp[i].runs[o], &rp[j].runs[o]) as i64);
            }
            *cell = chrf_f_score(&stats, order, cfg.beta, cfg.eps_smoothing);
        }
    });
    out
}

fn bleu_profile(tok: Tokenizer, max_order: usize, s: &str, words: &mut FxHashMap<String, u32>) -> Profile {
    let mut buf = String::new();
    tokenize_unjoined(tok, py_rstrip(s), &mut buf);
    let ids: Vec<u32> = py_split(&buf)
        .map(|w| {
            let k = words.len() as u32;
            *words.entry(w.to_owned()).or_insert(k)
        })
        .collect();
    let runs: Vec<AnyRuns> = (1..=max_order).map(|n| count(&ids, n, 32)).collect();
    let totals = runs.iter().map(|r| r.total() as i64).collect();
    Profile { runs, totals, len: ids.len() as i64 }
}

/// Sentence-BLEU matrix, row-major `cands.len() x refs.len()`.
pub fn bleu_matrix(tok: Tokenizer, scfg: &BleuScoreConfig, cands: &[&str], refs: &[&str]) -> Vec<f64> {
    let n_max = scfg.max_order;
    let mut words = FxHashMap::default();
    let cp: Vec<Profile> = cands.iter().map(|s| bleu_profile(tok, n_max, s, &mut words)).collect();
    let rp: Vec<Profile> = refs.iter().map(|s| bleu_profile(tok, n_max, s, &mut words)).collect();
    let m = refs.len();
    let mut out = vec![0f64; cands.len() * m];
    crate::par::for_each_chunk(&mut out, m.max(1), |(i, row)| {
        let mut stats = vec![0f64; 2 + 2 * n_max];
        for (j, cell) in row.iter_mut().enumerate() {
            stats[0] = cp[i].len as f64;
            stats[1] = rp[j].len as f64;
            for o in 0..n_max {
                stats[2 + o] = clipped_matches(&cp[i].runs[o], &rp[j].runs[o]) as f64;
                stats[2 + n_max + o] = cp[i].totals[o] as f64;
            }
            *cell = compute_bleu(&stats, scfg).score;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every cell equals sacreBLEU's `CHRF.sentence_score(cand, [ref]).score` bit for bit.
    #[test]
    fn chrf_matrix_cells_match_sacrebleu() {
        for line in include_str!("../tests/fixtures/mut_chrf_pairs.txt").lines() {
            let c: Vec<&str> = line.split('\t').collect();
            let cfg = ChrfConfig {
                char_order: c[0].parse().unwrap(),
                word_order: c[1].parse().unwrap(),
                beta: c[2].parse().unwrap(),
                eps_smoothing: c[3] == "1",
                whitespace: c[4] == "1",
            };
            let cands: Vec<&str> = c[5].split('|').collect();
            let refs: Vec<&str> = c[6].split('|').collect();
            let want: Vec<u64> = c[7].split_whitespace().map(|h| u64::from_str_radix(h, 16).unwrap()).collect();
            let got = chrf_matrix(&cfg, &cands, &refs);
            assert_eq!(got.len(), want.len(), "{line}");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.to_bits(), *w, "{line}");
            }
        }
    }
}
