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

use crate::unicode::{py_split, py_strip};

#[derive(Clone, Copy, Debug)]
pub struct ChrfStatsConfig {
    pub char_order: usize,
    pub word_order: usize,
    pub beta: f64,
    pub whitespace: bool,
    pub eps_smoothing: bool,

}

/// Characters of a string as a `Vec<char>`, honouring the whitespace-removal option.
fn char_seq(s: &str, keep_whitespace: bool) -> Vec<char> {
    let s = py_strip(s);
    if keep_whitespace {
        s.chars().collect()
    } else {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }
}

/// Word sequence: Python whitespace split of the stripped string. Lowercasing is the caller's job
/// (it happens before extraction, for both the char and word sequences).
fn word_seq(s: &str) -> Vec<String> {
    py_split(py_strip(s)).map(|w| w.to_string()).collect()
}

/// Run-length counts of the n-grams of a sequence, keyed by the sequence itself.
fn run_length<T: Ord + Clone>(seq: &[T], n: usize) -> Vec<(Vec<T>, u32)> {
    if seq.len() < n {
        return Vec::new();
    }
    let mut keys: Vec<&[T]> = seq.windows(n).collect();
    keys.sort_unstable();
    let mut out: Vec<(Vec<T>, u32)> = Vec::with_capacity(keys.len());
    for k in keys {
        match out.last_mut() {
            Some((last, c)) if last.as_slice() == k => *c += 1,
            _ => out.push((k.to_vec(), 1)),
        }
    }
    out
}

/// `(matches, hyp_total, ref_total)` for one order between one hyp sequence and one ref sequence.
fn order_stats<T: Ord + Clone>(hyp: &[T], refs: &[T], n: usize) -> (i64, i64, i64) {
    let h = run_length(hyp, n);
    let r = run_length(refs, n);
    let h_total: i64 = h.iter().map(|(_, c)| *c as i64).sum();
    let r_total: i64 = r.iter().map(|(_, c)| *c as i64).sum();
    let mut matches = 0i64;
    for (k, hc) in &h {
        if let Ok(pos) = r.binary_search_by(|(rk, _)| rk.cmp(k)) {
            matches += (*hc).min(r[pos].1) as i64;
        }
    }
    (matches, h_total, r_total)
}

/// The f-score chrF uses to choose among references, computed from a stats row.
fn f_score(stats: &[i64], order: usize, beta: f64, eps: bool) -> f64 {
    let mut sum_p = 0.0;
    let mut sum_r = 0.0;
    let mut cnt = 0;
    for n in 0..order {
        let h = stats[3 * n] as f64;
        let r = stats[3 * n + 1] as f64;
        let m = stats[3 * n + 2] as f64;
        let (p, rc) = if eps {
            if h > 0.0 && r > 0.0 {
                ((m + 1.0) / (h + 1.0), (m + 1.0) / (r + 1.0))
            } else {
                continue;
            }
        } else if h > 0.0 && r > 0.0 {
            (m / h, m / r)
        } else {
            continue;
        };
        sum_p += p;
        sum_r += rc;
        cnt += 1;
    }
    if cnt == 0 {
        return 0.0;
    }
    let p = sum_p / cnt as f64;
    let r = sum_r / cnt as f64;
    if p + r == 0.0 {
        return 0.0;
    }
    (1.0 + beta * beta) * (p * r) / (beta * beta * p + r)
}

/// Compute the statistics row for one segment.
pub fn chrf_stats_segment(cfg: &ChrfStatsConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    let order = cfg.char_order + cfg.word_order;
    let hc = char_seq(hyp, cfg.whitespace);
    let hw = if cfg.word_order > 0 { word_seq(hyp) } else { Vec::new() };

    let mut best_f = -1.0f64;
    let mut stats: Vec<i64> = Vec::with_capacity(3 * order);
    out.clear();
    for r in refs {
        stats.clear();
        let rc = char_seq(r, cfg.whitespace);
        for n in 1..=cfg.char_order {
            let (m, h, rt) = order_stats(&hc, &rc, n);
            stats.push(if rt > 0 { h } else { 0 });
            stats.push(rt);
            stats.push(m);
        }
        if cfg.word_order > 0 {
            let rw = word_seq(r);
            for n in 1..=cfg.word_order {
                let (m, h, rt) = order_stats(&hw, &rw, n);
                stats.push(if rt > 0 { h } else { 0 });
                stats.push(rt);
                stats.push(m);
            }
        }
        let f = f_score(&stats, order, cfg.beta, cfg.eps_smoothing);
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
        let cfg = ChrfStatsConfig {
            char_order,
            word_order,
            beta: 2.0,
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
