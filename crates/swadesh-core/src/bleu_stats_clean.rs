//! BLEU segment statistics: clean-room implementation.
//!
//! Specification: `docs/spec/stats.md`, section 2. Written from that document and from
//! Papineni et al. 2002 plus Lin & Och 2004 (closest reference length); no implementation source
//! was read. The previous `stats.rs` counted through a packed-u32 table; this one counts through a
//! sorted-key run-length, which is simpler to verify and independently derived.
//!
//! Output row: `[hyp_len, ref_len, correct_1..N, total_1..N]`, all integers.

use crate::tokenize_dispatch::{tokenize_unjoined, Tokenizer};

/// Config for BLEU statistics.
#[derive(Clone, Copy, Debug)]
pub struct BleuStatsConfig {
    pub tokenizer: Tokenizer,
    pub max_order: usize,
}

/// Closest reference length: the length minimising `|L - hyp|`; **a tie goes to the shorter
/// reference** when references arrive as separate streams (spec section 2.3). Measured; a packed
/// single stream behaves differently and is not this function's job.
pub fn closest_ref_len(hyp_len: i64, ref_lens: &[i64]) -> i64 {
    let mut best = -1i64;
    let mut best_diff = -1i64;
    for &r in ref_lens {
        let diff = (hyp_len - r).abs();
        if best_diff == -1 || diff < best_diff || (diff == best_diff && r < best) {
            best_diff = diff;
            best = r;
        }
    }
    best
}

/// Count clipped n-gram matches for one order.
///
/// For each hypothesis n-gram, the clipped count is `min(count_hyp, max over references)`. The max
/// across references is taken BEFORE clipping against the hypothesis (spec section 2.2).
fn clipped_matches(hyps: &[u32], refs: &[&[u32]], n: usize) -> i64 {
    if n == 0 || hyps.len() < n {
        return 0;
    }
    // Reference counts: for each n-gram key, the maximum count over all reference streams.
    let mut ref_keys: Vec<(Vec<u32>, u32)> = Vec::new();
    for r in refs {
        ref_keys.extend(run_length(r, n));
    }
    // Merge: max count per key across references.
    ref_keys.sort();
    let mut merged: Vec<(Vec<u32>, u32)> = Vec::with_capacity(ref_keys.len());
    for (k, c) in ref_keys {
        match merged.last_mut() {
            Some((lk, lc)) if *lk == k => *lc = (*lc).max(c),
            _ => merged.push((k, c)),
        }
    }
    // Hypothesis counts, clipped against the merged reference maxima.
    let hyp_runs = run_length(hyps, n);
    let mut total = 0i64;
    for (k, c) in hyp_runs {
        if let Ok(pos) = merged.binary_search_by(|(mk, _)| mk.cmp(&k)) {
            total += (c.min(merged[pos].1)) as i64;
        }
    }
    total
}

/// Run-length counts of the n-grams of `seq`, sorted by key.
fn run_length(seq: &[u32], n: usize) -> Vec<(Vec<u32>, u32)> {
    if seq.len() < n {
        return Vec::new();
    }
    let mut keys: Vec<&[u32]> = seq.windows(n).collect();
    keys.sort_unstable();
    let mut out: Vec<(Vec<u32>, u32)> = Vec::with_capacity(keys.len());
    for k in keys {
        match out.last_mut() {
            Some((last, c)) if last.as_slice() == k => *c += 1,
            _ => out.push((k.to_vec(), 1)),
        }
    }
    out
}

/// Compute the statistics row for one segment.
///
/// `hyp` and `refs` must already be lowercased if the metric is lowercase; tokenisation and
/// rstripping happen here, as the oracle does.
pub fn bleu_stats_segment(cfg: &BleuStatsConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    let n_max = cfg.max_order;

    // Tokenise: one buffer per stream, then re-split into tokens.
    let mut buf = String::new();
    let mut intern: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut interned: Vec<Vec<u32>> = Vec::with_capacity(refs.len() + 1);
    let mut next_id: u32 = 0;

    let mut push_stream = |s: &str,
                           buf: &mut String,
                           interned: &mut Vec<Vec<u32>>,
                           intern: &mut std::collections::HashMap<String, u32>,
                           next_id: &mut u32| {
        tokenize_unjoined(cfg.tokenizer, s, buf);
        let mut ids = Vec::new();
        for tok in crate::unicode::py_split(buf) {
            let id = *intern.entry(tok.to_string()).or_insert_with(|| {
                let v = *next_id;
                *next_id += 1;
                v
            });
            ids.push(id);
        }
        interned.push(ids);
    };

    push_stream(hyp, &mut buf, &mut interned, &mut intern, &mut next_id);
    let mut ref_ids: Vec<Vec<u32>> = Vec::with_capacity(refs.len());
    for r in refs {
        let mut one: Vec<Vec<u32>> = Vec::new();
        push_stream(r, &mut buf, &mut one, &mut intern, &mut next_id);
        ref_ids.push(one.pop().unwrap());
    }

    let hyp_ids = interned.pop().unwrap();
    let hyp_len = hyp_ids.len() as i64;
    let ref_lens: Vec<i64> = ref_ids.iter().map(|r| r.len() as i64).collect();
    let ref_len = closest_ref_len(hyp_len, &ref_lens);

    out.clear();
    out.push(hyp_len);
    out.push(ref_len);
    out.resize(2 + 2 * n_max, 0);
    let ref_slices: Vec<&[u32]> = ref_ids.iter().map(|r| r.as_slice()).collect();
    for n in 1..=n_max {
        // Row layout, measured from the port and confirmed by the oracle's consumers:
        //   [hyp_len, ref_len, correct_1..correct_N, total_1..total_N]
        // NOT interleaved -- the scoring stage indexes the two halves separately.
        out[2 + (n - 1)] = clipped_matches(&hyp_ids, &ref_slices, n);
        out[2 + n_max + (n - 1)] = (hyp_ids.len() + 1).saturating_sub(n) as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(hyp: &str, refs: &[&str], order: usize) -> Vec<i64> {
        let cfg = BleuStatsConfig { tokenizer: Tokenizer::None, max_order: order };
        let mut out = Vec::new();
        bleu_stats_segment(&cfg, hyp, refs, &mut out);
        out
    }

    #[test]
    fn perfect_match_gives_full_counts() {
        // 5 tokens, one reference of 5: orders 1..4 all match fully.
        let r = row("a b c d e", &["a b c d e"], 4);
        assert_eq!(r[0], 5);
        assert_eq!(r[1], 5);
        assert_eq!(&r[2..], &[5, 4, 3, 2, 5, 4, 3, 2]);
    }

    #[test]
    fn clipping_uses_max_over_references_before_hypothesis() {
        // Hyp "a a a"; refs "a", "a a". Max ref count for "a" is 2, so clip 3 -> 2.
        let r = row("a a a", &["a", "a a"], 1);
        assert_eq!(r[2], 2, "unigram matches clip to the best reference, not the first");
        assert_eq!(r[3], 3);
    }

    #[test]
    fn closest_ref_len_ties_go_to_the_shorter() {
        assert_eq!(closest_ref_len(5, &[]), -1);
        assert_eq!(closest_ref_len(5, &[3, 7]), 3);
        assert_eq!(closest_ref_len(5, &[7, 3]), 3);
        assert_eq!(closest_ref_len(10, &[20, 8]), 8);
    }

    #[test]
    fn empty_segment() {
        let r = row("", &["a b"], 4);
        assert_eq!(r[0], 0);
        assert_eq!(r[1], 2);
        assert!(r[2..].iter().all(|&v| v == 0));
    }
}