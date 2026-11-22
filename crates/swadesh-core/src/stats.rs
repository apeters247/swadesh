//! Segment-level sufficient statistics for BLEU and chrF, identical to
//! sacreBLEU's `_compute_segment_statistics`.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::ngram::{clipped_matches, count, max_merge};
use crate::score::chrf_f_score;
use crate::table::{fits, packed_windows, CountTable};
use crate::tokenize_dispatch::{tokenize_unjoined, Tokenizer};
use crate::unicode::{is_py_space, py_rstrip, py_split};

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
    SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        let n_max = cfg.max_order;
        // Tokenize hyp + refs into reusable buffers.
        while sc.strings.len() < refs.len() + 1 {
            sc.strings.push(String::new());
        }
        tokenize_unjoined(cfg.tokenizer, py_rstrip(hyp), &mut sc.strings[0]);
        for (k, r) in refs.iter().enumerate() {
            tokenize_unjoined(cfg.tokenizer, py_rstrip(r), &mut sc.strings[k + 1]);
        }
        let strings = &sc.strings[..refs.len() + 1];
        let mut interner: FxHashMap<&str, u32> = FxHashMap::default();
        let mut id = |t| {
            let n = interner.len() as u32;
            *interner.entry(t).or_insert(n)
        };
        let mut ids: Vec<Vec<u32>> = strings.iter().map(|s| py_split(s).map(&mut id).collect()).collect();
        let ref_ids = ids.split_off(1);
        let hyp_ids = ids.pop().unwrap();
        let ref_lens: Vec<i64> = ref_ids.iter().map(|r| r.len() as i64).collect();
        let hyp_len = hyp_ids.len() as i64;
        let ref_len = closest_ref_len(hyp_len, &ref_lens);

        out.clear();
        out.push(hyp_len);
        out.push(ref_len);
        out.resize(2 + 2 * n_max, 0);
        for n in 1..=n_max {
            let total = (hyp_ids.len() + 1).saturating_sub(n) as i64;
            out[2 + n_max + n - 1] = total;
            if total == 0 {
                continue;
            }
            let correct = if fits(n, 32) {
                let main = &mut sc.t1;
                if ref_ids.len() == 1 {
                    main.reset(ref_ids[0].len());
                    for k in packed_windows(&ref_ids[0], n, 32) {
                        main.add(k);
                    }
                } else {
                    let cap: usize = ref_ids.iter().map(|r| r.len()).sum();
                    main.reset(cap);
                    for r in &ref_ids {
                        let tmp = &mut sc.t2;
                        tmp.reset(r.len());
                        for k in packed_windows(r, n, 32) {
                            tmp.add(k);
                        }
                        for k in packed_windows(r, n, 32) {
                            main.set_max(k, tmp.get(k));
                        }
                    }
                }
                packed_windows(&hyp_ids, n, 32).filter(|&k| main.take(k)).count() as i64
            } else {
                let h = count(&hyp_ids, n, 32);
                let r = max_merge(ref_ids.iter().map(|ids| count(ids, n, 32)).collect());
                clipped_matches(&h, &r) as i64
            };
            out[2 + n - 1] = correct;
        }
    });
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

fn chars_of(whitespace: bool, s: &str, out: &mut Vec<u32>) {
    out.clear();
    if whitespace {
        out.extend(s.chars().map(|c| c as u32));
    } else {
        out.extend(s.chars().filter(|&c| !is_py_space(c)).map(|c| c as u32));
    }
}

/// `[hyp_total_if_ref_nonempty, ref_total, clipped_matches]` for one order.
#[inline]
fn order_stats(t: &mut CountTable, hyp: &[u32], r: &[u32], n: usize, bits: u32, out: &mut Vec<i64>) {
    let h_total = (hyp.len() + 1).saturating_sub(n) as i64;
    let r_total = (r.len() + 1).saturating_sub(n) as i64;
    let m = if h_total == 0 || r_total == 0 {
        0
    } else if fits(n, bits) {
        t.reset(r.len());
        for k in packed_windows(r, n, bits) {
            t.add(k);
        }
        packed_windows(hyp, n, bits).filter(|&k| t.take(k)).count() as i64
    } else {
        clipped_matches(&count(hyp, n, bits), &count(r, n, bits)) as i64
    };
    out.push(if r_total > 0 { h_total } else { 0 });
    out.push(r_total);
    out.push(m);
}

/// Statistics row: `[hyp, ref, match] * (char_order + word_order)` of the
/// best-matching reference (sacreBLEU semantics: first strictly-best f-score).
pub fn chrf_segment(cfg: &ChrfConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        let order = cfg.char_order + cfg.word_order;
        let mut hc = std::mem::take(&mut sc.u1);
        let mut rc = std::mem::take(&mut sc.u2);
        chars_of(cfg.whitespace, hyp, &mut hc);
        // Word ids interned jointly across hyp and refs.
        let mut interner: FxHashMap<&str, u32> = FxHashMap::default();
        let (hw, rws): (Vec<u32>, Vec<Vec<u32>>) = if cfg.word_order > 0 {
            let mut id = |t| {
                let n = interner.len() as u32;
                *interner.entry(t).or_insert(n)
            };
            let hw = remove_punctuation(hyp).into_iter().map(&mut id).collect();
            let rws = refs.iter().map(|r| remove_punctuation(r).into_iter().map(&mut id).collect()).collect();
            (hw, rws)
        } else {
            (Vec::new(), Vec::new())
        };
        let mut best_f = -1.0f64;
        let mut stats: Vec<i64> = Vec::with_capacity(3 * order);
        out.clear();
        for (ri, r) in refs.iter().enumerate() {
            stats.clear();
            chars_of(cfg.whitespace, r, &mut rc);
            for n in 1..=cfg.char_order {
                order_stats(&mut sc.t1, &hc, &rc, n, 21, &mut stats);
            }
            for n in 1..=cfg.word_order {
                order_stats(&mut sc.t1, &hw, &rws[ri], n, 32, &mut stats);
            }
            let f = chrf_f_score(&stats, order, cfg.beta, cfg.eps_smoothing);
            if f > best_f {
                best_f = f;
                out.clear();
                out.extend_from_slice(&stats);
            }
        }
        sc.u1 = hc;
        sc.u2 = rc;
    });
}

struct Scratch {
    strings: Vec<String>,
    t1: CountTable,
    t2: CountTable,
    u1: Vec<u32>,
    u2: Vec<u32>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch {
        strings: Vec::new(),
        t1: CountTable::new(),
        t2: CountTable::new(),
        u1: Vec::new(),
        u2: Vec::new(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interner_clear_restarts_ids() {
        let mut i = Interner::default();
        assert_eq!(i.id("a"), 0);
        assert_eq!(i.id("b"), 1);
        assert_eq!(i.id("a"), 0);
        i.clear();
        assert_eq!(i.id("c"), 0);
    }

    #[test]
    fn closest_ref_len_matches_sacrebleu() {
        // sacreBLEU `_get_closest_ref_len`: no references -> -1; ties go to the shorter reference.
        assert_eq!(closest_ref_len(5, &[]), -1);
        assert_eq!(closest_ref_len(5, &[3, 7]), 3);
        assert_eq!(closest_ref_len(5, &[7, 3]), 3);
        assert_eq!(closest_ref_len(10, &[12, 8, 20]), 8);
        assert_eq!(closest_ref_len(10, &[8, 12]), 8);
    }

    #[test]
    fn remove_punctuation_matches_sacrebleu() {
        for line in include_str!("../tests/fixtures/mut_punct.txt").lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(remove_punctuation(cols[0]), cols[1..].to_vec(), "input {:?}", cols[0]);
        }
    }

    /// Segment statistics against sacreBLEU's `CHRF._compute_segment_statistics`, including
    /// cases where two references tie on F-score: sacreBLEU keeps the first strict best.
    #[test]
    fn chrf_segment_matches_sacrebleu() {
        for line in include_str!("../tests/fixtures/mut_chrf_seg.txt").lines() {
            let c: Vec<&str> = line.split('\t').collect();
            let cfg = ChrfConfig {
                char_order: c[1].parse().unwrap(),
                word_order: c[2].parse().unwrap(),
                whitespace: c[3] == "1",
                eps_smoothing: c[4] == "1",
                beta: c[5].parse().unwrap(),
            };
            let hyp = c[6];
            let refs: Vec<&str> = c[7..c.len() - 1].to_vec();
            let want: Vec<i64> = c[c.len() - 1].split_whitespace().map(|x| x.parse().unwrap()).collect();
            let mut out = Vec::new();
            chrf_segment(&cfg, hyp, &refs, &mut out);
            assert_eq!(out, want, "hyp {hyp:?} refs {refs:?}");
        }
    }
}
