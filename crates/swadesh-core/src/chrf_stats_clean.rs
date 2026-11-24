//! chrF segment statistics: clean-room implementation.
//!
//! Specification: `docs/spec/stats.md` section 3, refined by measurement. Written from that
//! document and Popović 2015; no implementation source was read.
//!
//! Row layout, measured on real corpus data: `[hyp_total_n, ref_total_n, match_n]` per order, char
//! orders first then word orders. For one WMT22 segment the hyp has 194 characters and the reference
//! 202, and the first order reports `[194, 202, 194]`: hypothesis total, reference total, matches.
//! ("abc xyz" vs "abcd" reports `[6, 4, 3]` for the same reason and reads as ref-first if the
//! strings' lengths are guessed wrong -- measure with a segment whose lengths differ.)
//!
//! **Multi-reference rule, measured:** for several references the row kept is that of the reference
//! with the highest f-score (the first strictly better one), NOT a merge of their counts. A
//! hypothesis matching one reference perfectly scores 100.0 against `[good, bad]` in either order.

use crate::stats::ChrfConfig;
use crate::table::{fits, packed_windows, CountTable};
use rustc_hash::FxHashMap;
use crate::unicode::{is_py_space, py_split, py_strip};

/// Characters of a string as a `Vec<char>`, honouring the whitespace-removal option.

/// Bits per element when packing n-gram keys. Characters fit in 21 (Unicode tops out at
/// 0x10FFFF) and word ids in 32.
const CHAR_BITS: u32 = 21;
const WORD_BITS: u32 = 32;

const PUNCTS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

#[inline]
fn is_chrf_punct(c: char) -> bool {
    c.is_ascii() && PUNCTS.contains(c)
}

/// chrF word splitting: a word whose final character is ASCII punctuation is split into
/// `(word minus that character, the character)`; failing that, the same with the leading
/// character. Single-character words and words with no ASCII punctuation split at either end are
/// returned unchanged. (spec `stats.md` 3.2, measured.)
pub fn remove_punctuation(sent: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for w in py_split(sent) {
        let mut chars = w.chars();
        let first = chars.next().expect("py_split never yields an empty word");
        let last = w.chars().next_back().expect("a word has at least one char");
        if chars.next().is_none() {
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

fn char_seq(s: &str, keep_whitespace: bool) -> Vec<u32> {
    // Measured 2026-11-24: with `whitespace=True` the characters participate as they are, without
    // any strip -- a reference "a-\n" has three characters. With `whitespace=False` every Python
    // whitespace character is removed.
    if keep_whitespace {
        s.chars().map(|c| c as u32).collect()
    } else {
        s.chars().filter(|c| !is_py_space(*c)).map(|c| c as u32).collect()
    }
}

/// Word sequence: Python whitespace split of the stripped string. Lowercasing is the caller's job
/// (it happens before extraction, for both the char and word sequences).
fn word_seq(s: &str, ids: &mut FxHashMap<String, u32>) -> Vec<u32> {
    // Word order counts split on the chrF punctuation rule, not plain whitespace: `Dank,` is two
    // tokens and `#PRS_ORG#` is `#PRS_ORG` plus `#`. Char orders never call this.
    remove_punctuation(py_strip(s))
        .into_iter()
        .map(|w| {
            let n = ids.len() as u32;
            *ids.entry(w.to_string()).or_insert(n)
        })
        .collect()
}

/// `(matches, hyp_total, ref_total)` for one order between one hyp sequence and one ref sequence.
///
/// Counting uses the packed-key table, original to this project; measurement showed it is far faster
/// than sorting keys (chrF went from 104 us to a few us per pair).
fn order_stats(hyp: &[u32], refs: &[u32], n: usize, bits: u32) -> (i64, i64, i64) {
    let h = hyp;
    let r = refs;
    let h_total = (h.len() + 1).saturating_sub(n) as i64;
    let r_total = (r.len() + 1).saturating_sub(n) as i64;
    if h_total == 0 || r_total == 0 {
        return (0, h_total, r_total);
    }
    if !fits(n, bits) {
        // Too many elements for a u128 key (orders above six at 21 bits, above four at 32), so
        // count the wide n-grams through the general run-length representation instead.
        let hc = crate::ngram::count(h, n, bits);
        let rc = crate::ngram::count(r, n, bits);
        return (crate::ngram::clipped_matches(&hc, &rc) as i64, h_total, r_total);
    }
    let mut rtab = CountTable::new();
    rtab.reset(r.len().saturating_sub(n) + 1);
    for k in packed_windows(r, n, bits) {
        rtab.add(k);
    }
    let mut m = 0i64;
    for k in packed_windows(h, n, bits) {
        if rtab.take(k) {
            m += 1;
        }
    }
    (m, h_total, r_total)
}

/// Compute the statistics row for one segment.
pub fn chrf_stats_segment(cfg: &ChrfConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    let order = cfg.char_order + cfg.word_order;
    let hc = char_seq(hyp, cfg.whitespace);
    let mut wids: FxHashMap<String, u32> = FxHashMap::default();
    let hw = if cfg.word_order > 0 { word_seq(hyp, &mut wids) } else { Vec::new() };

    let mut best_f = -1.0f64;
    let mut stats: Vec<i64> = Vec::with_capacity(3 * order);
    out.clear();
    for r in refs {
        stats.clear();
        let rc = char_seq(r, cfg.whitespace);
        // Six character n-grams at 21 bits each still fit a u128 (Unicode tops out at 0x10FFFF);
        // packing at 32 bits silently dropped every order above four.
        for n in 1..=cfg.char_order {
            let (m, h, rt) = order_stats(&hc, &rc, n, CHAR_BITS);
            stats.push(if rt > 0 { h } else { 0 });
            stats.push(rt);
            stats.push(m);
        }
        if cfg.word_order > 0 {
            let rw = word_seq(r, &mut wids);
            for n in 1..=cfg.word_order {
                let (m, h, rt) = order_stats(&hw, &rw, n, WORD_BITS);
                stats.push(if rt > 0 { h } else { 0 });
                stats.push(rt);
                stats.push(m);
            }
        }
        // The reference chosen is the one the metric's own score prefers, so selection must use
        // the scoring f-score (eps-aware), not a separate formula. Measured 2026-11-24 on
        // `hyp='' refs=['a','']` and `hyp='b' refs=['','a']`: the oracle's row is the one with the
        // higher `chrf_f_score`, ties keeping the first reference.
        let f = crate::score::chrf_f_score(&stats, order, cfg.beta, cfg.eps_smoothing);
        if f > best_f {
            best_f = f;
            out.clear();
            out.extend_from_slice(&stats);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(hyp: &str, refs: &[&str], char_order: usize, word_order: usize) -> Vec<i64> {
        let cfg = ChrfConfig {
            char_order,
            word_order,
            beta: 2,
            whitespace: false,
            eps_smoothing: false,
        };
        let mut out = Vec::new();
        chrf_stats_segment(&cfg, hyp, refs, &mut out);
        out
    }

    #[test]
    fn perfect_match() {
        // "abc xyz" with whitespace removed is "abcxyz": 6 chars.
        let r = row("abc xyz", &["abc xyz"], 2, 0);
        assert_eq!(r, vec![6, 6, 6, 5, 5, 5]);
    }

    #[test]
    fn best_reference_wins_not_merged() {
        // Two references: 6 chars and 3 chars. The 6-char one wins, so the row is its row.
        let r = row("abc xyz", &["abc xyz", "qqq"], 2, 0);
        assert_eq!(r, vec![6, 6, 6, 5, 5, 5]);
    }

    #[test]
    fn word_order_appends_word_orders() {
        let r = row("a b", &["a b"], 1, 1);
        assert_eq!(r.len(), 6, "two orders: 3 values each");
    }
}
