//! Segment-level sufficient statistics for BLEU and chrF, identical to
//! sacreBLEU's `_compute_segment_statistics`.

use std::collections::HashMap;

use crate::ngram::{clipped_matches, count, max_merge, AnyRuns};
use crate::score::chrf_f_score;
use crate::tokenize::{preprocess_bleu, Tokenizer};
use crate::unicode::{is_py_space, py_split};

/// Interns string tokens into dense u32 ids (per segment).
#[derive(Default)]
pub struct Interner<'a> {
    map: HashMap<&'a str, u32>,
}

impl<'a> Interner<'a> {
    pub fn clear(&mut self) {
        self.map.clear();
    }
    #[inline]
    pub fn id(&mut self, s: &'a str) -> u32 {
        let n = self.map.len() as u32;
        *self.map.entry(s).or_insert(n)
    }
}

// --------------------------------------------------------------------- BLEU

#[derive(Clone, Copy, Debug)]
pub struct BleuConfig {
    pub tokenizer: Tokenizer,
    pub max_order: usize,
}

/// sacreBLEU `BLEU._get_closest_ref_len`.
pub fn closest_ref_len(hyp_len: i64, ref_lens: &[i64]) -> i64 {
    let (mut closest_diff, mut closest_len) = (-1i64, -1i64);
    for &r in ref_lens {
        let diff = (hyp_len - r).abs();
        if closest_diff == -1 || diff < closest_diff {
            closest_diff = diff;
            closest_len = r;
        } else if diff == closest_diff && r < closest_len {
            closest_len = r;
        }
    }
    closest_len
}

/// Statistics row: `[hyp_len, ref_len, correct_1..N, total_1..N]`.
/// `hyp` and `refs` must already be lowercased if requested.
pub fn bleu_segment(cfg: &BleuConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    let n_max = cfg.max_order;
    let mut buf = String::new();
    // Tokenize references.
    let mut ref_strings: Vec<String> = Vec::with_capacity(refs.len());
    for r in refs {
        preprocess_bleu(cfg.tokenizer, r, &mut buf);
        ref_strings.push(std::mem::take(&mut buf));
    }
    let mut hyp_string = String::new();
    preprocess_bleu(cfg.tokenizer, hyp, &mut hyp_string);

    let mut interner = Interner::default();
    let hyp_ids: Vec<u32> = py_split(&hyp_string).map(|t| interner.id(t)).collect();
    let mut ref_lens = Vec::with_capacity(refs.len());
    let mut ref_ids: Vec<Vec<u32>> = Vec::with_capacity(refs.len());
    for r in &ref_strings {
        let ids: Vec<u32> = py_split(r).map(|t| interner.id(t)).collect();
        ref_lens.push(ids.len() as i64);
        ref_ids.push(ids);
    }
    let hyp_len = hyp_ids.len() as i64;
    let ref_len = closest_ref_len(hyp_len, &ref_lens);

    out.clear();
    out.push(hyp_len);
    out.push(ref_len);
    let mut correct = vec![0i64; n_max];
    let mut total = vec![0i64; n_max];
    for n in 1..=n_max {
        let h = count(&hyp_ids, n, 32);
        total[n - 1] = h.total() as i64;
        if h.is_empty() {
            continue;
        }
        let r = max_merge(ref_ids.iter().map(|ids| count(ids, n, 32)).collect());
        correct[n - 1] = clipped_matches(&h, &r) as i64;
    }
    out.extend_from_slice(&correct);
    out.extend_from_slice(&total);
}

// --------------------------------------------------------------------- chrF

#[derive(Clone, Copy, Debug)]
pub struct ChrfConfig {
    pub char_order: usize,
    pub word_order: usize,
    pub beta: i64,
    pub whitespace: bool,
    pub eps_smoothing: bool,
}

const PUNCTS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

#[inline]
fn is_chrf_punct(c: char) -> bool {
    c.is_ascii() && PUNCTS.contains(c)
}

/// sacreBLEU `CHRF._remove_punctuation`.
pub fn remove_punctuation(sent: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for w in py_split(sent) {
        let mut chars = w.chars();
        let first = chars.next().unwrap();
        let last = w.chars().next_back().unwrap();
        if chars.next().is_none() {
            // len(w) == 1
            out.push(w);
        } else if is_chrf_punct(last) {
            out.push(&w[..w.len() - last.len_utf8()]);
            out.push(&w[w.len() - last.len_utf8()..]);
        } else if is_chrf_punct(first) {
            out.push(&w[..first.len_utf8()]);
            out.push(&w[first.len_utf8()..]);
        } else {
            out.push(w);
        }
    }
    out
}

/// Character n-gram count sets (orders 1..=char_order) for one sentence.
fn chrf_char_ngrams(cfg: &ChrfConfig, s: &str) -> Vec<AnyRuns> {
    let chars: Vec<u32> = if cfg.whitespace {
        s.chars().map(|c| c as u32).collect()
    } else {
        s.chars().filter(|&c| !is_py_space(c)).map(|c| c as u32).collect()
    };
    let mut out = Vec::with_capacity(cfg.char_order + cfg.word_order);
    for n in 1..=cfg.char_order {
        out.push(count(&chars, n, 21));
    }
    out
}

/// Char + word n-grams for hypothesis and references. Word ids are interned
/// jointly so that equal words get equal ids across hyp/refs.
fn chrf_ngrams_pair(cfg: &ChrfConfig, hyp: &str, refs: &[&str]) -> (Vec<AnyRuns>, Vec<Vec<AnyRuns>>) {
    let mut hyp_ng = chrf_char_ngrams(cfg, hyp);
    let mut refs_ng: Vec<Vec<AnyRuns>> = refs.iter().map(|r| chrf_char_ngrams(cfg, r)).collect();
    if cfg.word_order > 0 {
        let mut words = Interner::default();
        let hw: Vec<u32> = remove_punctuation(hyp).into_iter().map(|w| words.id(w)).collect();
        let rws: Vec<Vec<u32>> = refs
            .iter()
            .map(|r| remove_punctuation(r).into_iter().map(|w| words.id(w)).collect())
            .collect();
        for n in 1..=cfg.word_order {
            hyp_ng.push(count(&hw, n, 32));
            for (k, rw) in rws.iter().enumerate() {
                refs_ng[k].push(count(rw, n, 32));
            }
        }
    }
    (hyp_ng, refs_ng)
}

/// Statistics row: `[hyp, ref, match] * (char_order + word_order)` of the
/// best-matching reference (sacreBLEU semantics: first strictly-best f-score).
pub fn chrf_segment(cfg: &ChrfConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    let (hyp_ng, refs_ng) = chrf_ngrams_pair(cfg, hyp, refs);
    let order = cfg.char_order + cfg.word_order;
    let hyp_totals: Vec<i64> = hyp_ng.iter().map(|h| h.total() as i64).collect();
    let mut best_f = -1.0f64;
    let mut stats = Vec::with_capacity(3 * order);
    out.clear();
    for r in &refs_ng {
        stats.clear();
        for i in 0..order {
            let rr = &r[i];
            let m = clipped_matches(&hyp_ng[i], rr) as i64;
            let ref_total = rr.total() as i64;
            stats.push(if rr.is_empty() { 0 } else { hyp_totals[i] });
            stats.push(ref_total);
            stats.push(m);
        }
        let f = chrf_f_score(&stats, order, cfg.beta, cfg.eps_smoothing);
        if f > best_f {
            best_f = f;
            out.clear();
            out.extend_from_slice(&stats);
        }
    }
}
