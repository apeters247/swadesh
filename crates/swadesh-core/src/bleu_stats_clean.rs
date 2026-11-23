//! BLEU segment statistics.
//!
//! Specification: `docs/spec/stats.md`, section 2. Written from that document and from
//! Papineni et al. 2002 plus Lin & Och 2004 (closest reference length); no implementation source
//! was read. Counts n-gram matches through the project's packed-key table.
//!
//! Output row: `[hyp_len, ref_len, correct_1..N, total_1..N]`, all integers.

use crate::stats::BleuConfig;
use crate::table::{fits, packed_windows, CountTable};
use crate::tokenize_dispatch::{preprocess_bleu, Tokenizer};
use crate::unicode::{py_rstrip, py_split};
use rustc_hash::FxHashMap;


/// Closest reference length: the length minimising `|L - hyp|`; a tie goes to the shorter
/// reference when references arrive as separate streams (spec section 2.3).
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

/// Per-thread buffers, so a segment allocates nothing after warm-up.
struct Scratch {
    strings: Vec<String>,
    ids: Vec<Vec<u32>>,
    ref_lens: Vec<i64>,
    ref_max: CountTable,
    per_ref: CountTable,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch {
        strings: Vec::new(),
        ids: Vec::new(),
        ref_lens: Vec::new(),
        ref_max: CountTable::new(),
        per_ref: CountTable::new(),
    });
}

/// Clipped matches for one order, filling `ref_max` with "how many of each hypothesis n-gram are
/// available to match". Single-reference segments skip the per-reference merge (spec 2.2's rule
/// degenerates to plain clip against that one reference).
fn fill_ref_max(hyp: &[u32], refs: &[Vec<u32>], n: usize, sc: &mut Scratch) -> bool {
    if n == 0 || hyp.len() < n || !fits(n, 32) {
        return false;
    }
    let main = &mut sc.ref_max;
    if refs.len() == 1 {
        main.reset(refs[0].len());
        for k in packed_windows(&refs[0], n, 32) {
            main.add(k);
        }
    } else {
        let cap: usize = refs.iter().map(|r| r.len()).sum();
        main.reset(cap);
        for r in refs {
            let tmp = &mut sc.per_ref;
            tmp.reset(r.len());
            for k in packed_windows(r, n, 32) {
                tmp.add(k);
            }
            for k in packed_windows(r, n, 32) {
                main.set_max(k, tmp.get(k));
            }
        }
    }
    true
}

/// Compute the statistics row for one segment.
///
/// Tokenisation and rstripping happen here, as the oracle does.
pub fn bleu_stats_segment(cfg: &BleuConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        let n_max = cfg.max_order;

        while sc.strings.len() < refs.len() + 1 {
            sc.strings.push(String::new());
        }
        while sc.ids.len() < refs.len() + 1 {
            sc.ids.push(Vec::new());
        }
        // Tokenise into reusable buffers. Words come out as a split of the tokenised line, so there
        // is no intermediate space-join-and-resplit round trip.
        preprocess_bleu(cfg.tokenizer, hyp, &mut sc.strings[0]);
        for (k, r) in refs.iter().enumerate() {
            preprocess_bleu(cfg.tokenizer, r, &mut sc.strings[k + 1]);
        }
        let strings = &sc.strings[..refs.len() + 1];
        // Take the per-segment buffers out of `sc` for the rest of the call so the counting tables
        // inside `sc` can be borrowed mutably at the same time; they go back at the end.
        let mut ids = std::mem::take(&mut sc.ids);

        // Intern with borrowed keys into reused per-segment buffers.
        let mut interner: FxHashMap<&str, u32> = FxHashMap::default();
        for buf in ids.iter_mut().take(strings.len()) {
            buf.clear();
        }
        for (buf, s) in ids.iter_mut().zip(strings.iter()) {
            for t in py_split(py_rstrip(s)) {
                let n = interner.len() as u32;
                let id = *interner.entry(t).or_insert(n);
                buf.push(id);
            }
        }
        drop(interner);

        // Drop buffers left over from a previous segment that had more references. Without this
        // the extra buffers are treated as additional references, which silently changes the
        // closest-reference length and the clipped counts.
        ids.truncate(refs.len() + 1);
        let (hyp_idv, ref_ids): (&[u32], &[Vec<u32>]) = match ids.split_first() {
            Some((h, r)) => (h.as_slice(), r),
            None => {
                sc.ids = ids;
                out.clear();
                return;
            }
        };
        let hyp_len = hyp_idv.len() as i64;
        let mut ref_lens: Vec<i64> = std::mem::take(&mut sc.ref_lens);
        ref_lens.clear();
        ref_lens.extend(ref_ids.iter().map(|r| r.len() as i64));
        let ref_len = closest_ref_len(hyp_len, &ref_lens);

        out.clear();
        out.push(hyp_len);
        out.push(ref_len);
        out.resize(2 + 2 * n_max, 0);
        for n in 1..=n_max {
            let total = (hyp_idv.len() + 1).saturating_sub(n) as i64;
            out[2 + n_max + (n - 1)] = total;
            if total == 0 {
                continue;
            }
            let correct = if fits(n, 32) {
                fill_ref_max(hyp_idv, ref_ids, n, sc);
                let main = &mut sc.ref_max;
                let mut matched = 0i64;
                for k in packed_windows(hyp_idv, n, 32) {
                    if main.take(k) {
                        matched += 1;
                    }
                }
                matched
            } else {
                // spec 2.6: orders above four do not fit a u128 key at 32 bits. Counting through
                // the general wide-key representation is exact; returning "no key" here used to
                // report zero matches and silently depress the score.
                let hyp_runs = crate::ngram::count(hyp_idv, n, 32);
                let ref_runs = crate::ngram::max_merge(
                    ref_ids.iter().map(|r| crate::ngram::count(r, n, 32)).collect());
                crate::ngram::clipped_matches(&hyp_runs, &ref_runs) as i64
            };
            out[2 + (n - 1)] = correct;
        }
        sc.ids = ids;
        sc.ref_lens = ref_lens;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(hyp: &str, refs: &[&str], order: usize) -> Vec<i64> {
        let cfg = BleuConfig { tokenizer: Tokenizer::None, max_order: order };
        let mut out = Vec::new();
        bleu_stats_segment(&cfg, hyp, refs, &mut out);
        out
    }

    #[test]
    fn perfect_match_gives_full_counts() {
        let r = row("a b c d e", &["a b c d e"], 4);
        assert_eq!(r[0], 5);
        assert_eq!(r[1], 5);
        assert_eq!(&r[2..], &[5, 4, 3, 2, 5, 4, 3, 2]);
    }

    #[test]
    fn orders_above_four_count_exactly() {
        // spec 2.6: 5- and 6-grams exceed the u128 packed budget and must use wide keys.
        let r = row("a b c d e f g", &["a b c d e f g"], 6);
        assert_eq!(r[0], 7);
        assert_eq!(r[1], 7);
        assert_eq!(&r[2..8], &[7, 6, 5, 4, 3, 2], "correct 1..6");
        assert_eq!(&r[8..14], &[7, 6, 5, 4, 3, 2], "total 1..6");

        let r = row("a b c d e f g", &["a b c d e f X"], 5);
        assert_eq!(&r[2..7], &[6, 5, 4, 3, 2], "correct 1..5");
        assert_eq!(&r[7..12], &[7, 6, 5, 4, 3], "total 1..5");
    }

    #[test]
    fn clipping_uses_max_over_references_before_hypothesis() {
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

    #[test]
    fn repeated_call_reuses_buffers_and_stays_correct() {
        let mut out = Vec::new();
        let cfg = BleuConfig { tokenizer: Tokenizer::None, max_order: 2 };
        bleu_stats_segment(&cfg, "a a b", &["a b", "a a b"], &mut out);
        let first = out.clone();
        bleu_stats_segment(&cfg, "a a b", &["a b", "a a b"], &mut out);
        assert_eq!(first, out);
    }
}
