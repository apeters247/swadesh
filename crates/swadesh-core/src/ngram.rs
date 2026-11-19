//! Exact (collision-free) n-gram counting.
//!
//! Tokens / characters are mapped to integer ids and n-grams are packed
//! losslessly into a `u128` when they fit (word ids: 4 x 32 bit, Unicode code
//! points: 6 x 21 bit). Longer n-grams fall back to `Vec<u32>` keys. Counting is
//! done by sort + run-length encoding, which is cache friendly for the short
//! sequences typical of MT/LLM evaluation and needs no hashing at all.

use std::cmp::Ordering;

/// Sorted `(key, count)` runs.
pub type Runs<K> = Vec<(K, u32)>;

#[derive(Clone, Debug)]
pub enum AnyRuns {
    Packed(Runs<u128>),
    Wide(Runs<Vec<u32>>),
}

impl AnyRuns {
    pub fn total(&self) -> u64 {
        match self {
            AnyRuns::Packed(r) => r.iter().map(|x| x.1 as u64).sum(),
            AnyRuns::Wide(r) => r.iter().map(|x| x.1 as u64).sum(),
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            AnyRuns::Packed(r) => r.is_empty(),
            AnyRuns::Wide(r) => r.is_empty(),
        }
    }
}

#[inline]
fn rle<K: Ord + Clone>(mut keys: Vec<K>) -> Runs<K> {
    keys.sort_unstable();
    let mut out: Runs<K> = Vec::with_capacity(keys.len());
    for k in keys {
        match out.last_mut() {
            Some((last, c)) if *last == k => *c += 1,
            _ => out.push((k, 1)),
        }
    }
    out
}

/// Count all n-grams of order `n` over `seq`, with each element using `bits` bits.
pub fn count(seq: &[u32], n: usize, bits: u32) -> AnyRuns {
    if n == 0 || seq.len() < n {
        return AnyRuns::Packed(Vec::new());
    }
    if (n as u32) * bits <= 128 {
        let mut keys = Vec::with_capacity(seq.len() + 1 - n);
        for w in seq.windows(n) {
            let mut k: u128 = 0;
            for &x in w {
                k = (k << bits) | x as u128;
            }
            keys.push(k);
        }
        AnyRuns::Packed(rle(keys))
    } else {
        let keys: Vec<Vec<u32>> = seq.windows(n).map(|w| w.to_vec()).collect();
        AnyRuns::Wide(rle(keys))
    }
}

fn clipped_matches_t<K: Ord>(h: &Runs<K>, r: &Runs<K>) -> u64 {
    let (mut i, mut j, mut m) = (0usize, 0usize, 0u64);
    while i < h.len() && j < r.len() {
        match h[i].0.cmp(&r[j].0) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                m += h[i].1.min(r[j].1) as u64;
                i += 1;
                j += 1;
            }
        }
    }
    m
}

/// Sum over shared keys of `min(hyp_count, ref_count)`.
pub fn clipped_matches(h: &AnyRuns, r: &AnyRuns) -> u64 {
    match (h, r) {
        (AnyRuns::Packed(a), AnyRuns::Packed(b)) => clipped_matches_t(a, b),
        (AnyRuns::Wide(a), AnyRuns::Wide(b)) => clipped_matches_t(a, b),
        // Same order always uses the same representation; an empty set may
        // be represented either way.
        _ => 0,
    }
}

fn max_merge_t<K: Ord + Clone>(parts: Vec<Runs<K>>) -> Runs<K> {
    if parts.len() == 1 {
        return parts.into_iter().next().unwrap();
    }
    let mut all: Runs<K> = parts.into_iter().flatten().collect();
    all.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut out: Runs<K> = Vec::with_capacity(all.len());
    for (k, c) in all {
        match out.last_mut() {
            Some((last, lc)) if *last == k => *lc = (*lc).max(c),
            _ => out.push((k, c)),
        }
    }
    out
}

/// Per-key maximum across several reference count sets (BLEU multi-reference clipping).
pub fn max_merge(parts: Vec<AnyRuns>) -> AnyRuns {
    let mut packed = Vec::new();
    let mut wide = Vec::new();
    for p in parts {
        match p {
            AnyRuns::Packed(r) => {
                if !r.is_empty() {
                    packed.push(r)
                }
            }
            AnyRuns::Wide(r) => {
                if !r.is_empty() {
                    wide.push(r)
                }
            }
        }
    }
    if !wide.is_empty() {
        AnyRuns::Wide(max_merge_t(wide))
    } else if packed.is_empty() {
        AnyRuns::Packed(Vec::new())
    } else {
        AnyRuns::Packed(max_merge_t(packed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn packed(r: &AnyRuns) -> Vec<(u128, u32)> {
        match r {
            AnyRuns::Packed(v) => v.clone(),
            AnyRuns::Wide(_) => panic!("expected packed keys"),
        }
    }

    fn naive_packed(seq: &[u32], n: usize, bits: u32) -> BTreeMap<u128, u32> {
        let mut m = BTreeMap::new();
        if seq.len() >= n {
            for w in seq.windows(n) {
                let k = w.iter().fold(0u128, |k, &x| (k << bits) | x as u128);
                *m.entry(k).or_insert(0) += 1;
            }
        }
        m
    }

    #[test]
    fn count_merges_repeated_ngrams() {
        assert_eq!(packed(&count(&[1, 1, 1], 1, 32)), vec![(1, 3)]);
        let seq = [3u32, 1, 2, 1, 2, 1, 3, 3];
        for n in 1..=4 {
            let got: BTreeMap<u128, u32> = packed(&count(&seq, n, 32)).into_iter().collect();
            assert_eq!(got, naive_packed(&seq, n, 32), "n {n}");
        }
        assert!(count(&[1, 2], 3, 32).is_empty());
        assert!(count(&[1, 2], 0, 32).is_empty());
    }

    #[test]
    fn count_wide_keys_for_long_orders() {
        let seq: Vec<u32> = (0..12).map(|i| (i % 3) as u32).collect();
        match count(&seq, 5, 32) {
            AnyRuns::Wide(r) => {
                assert_eq!(r.iter().map(|x| x.1 as usize).sum::<usize>(), seq.len() - 4);
                for w in r.windows(2) {
                    assert!(w[0].0 < w[1].0, "keys must be sorted and unique");
                }
                for (k, c) in &r {
                    let want = seq.windows(5).filter(|w| *w == k.as_slice()).count() as u32;
                    assert_eq!(*c, want);
                }
            }
            AnyRuns::Packed(_) => panic!("5 x 32 bits must use wide keys"),
        }
    }

    #[test]
    fn clipped_matches_sums_min_counts() {
        let h = [1u32, 2, 1, 2, 1, 3, 2, 2];
        let r = [1u32, 2, 2, 2, 4];
        for n in 1..=3 {
            let hc = naive_packed(&h, n, 32);
            let rc = naive_packed(&r, n, 32);
            let want: u64 = hc
                .iter()
                .map(|(k, c)| (*c).min(*rc.get(k).unwrap_or(&0)) as u64)
                .sum();
            assert_eq!(clipped_matches(&count(&h, n, 32), &count(&r, n, 32)), want, "n {n}");
        }
        // Wide keys, and mixed representations (an empty side is always 0).
        let hw: Vec<u32> = (0..30).map(|i| (i * i % 4) as u32).collect();
        let rw: Vec<u32> = (0..30).map(|i| (i % 3) as u32).collect();
        let (a, b) = (count(&hw, 6, 32), count(&rw, 6, 32));
        let mut distinct: Vec<&[u32]> = hw.windows(6).collect();
        distinct.sort();
        distinct.dedup();
        let want: u64 = distinct
            .iter()
            .map(|k| {
                let hit = hw.windows(6).filter(|w| w == k).count() as u64;
                let rhit = rw.windows(6).filter(|w| w == k).count() as u64;
                hit.min(rhit)
            })
            .sum();
        assert_eq!(clipped_matches(&a, &b), want);
        assert_eq!(clipped_matches(&count(&[9], 1, 32), &count(&r, 1, 32)), 0);
    }

    #[test]
    fn max_merge_takes_per_key_maximum() {
        let a = count(&[1, 1, 2], 1, 32);
        let b = count(&[1, 3, 3, 3], 1, 32);
        let c = count(&[], 1, 32);
        assert_eq!(packed(&max_merge(vec![a, b, c])), vec![(1, 2), (2, 1), (3, 3)]);
        assert_eq!(packed(&max_merge(vec![count(&[5, 5], 1, 32)])), vec![(5, 2)]);
        assert!(max_merge(vec![count(&[], 1, 32), count(&[1], 2, 32)]).is_empty());
    }

    #[test]
    fn max_merge_wide_parts_take_per_key_maximum() {
        let parts = [
            vec![1u32, 2, 3, 4, 5, 6],
            vec![1u32, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5],
            vec![1u32, 2, 3],
        ];
        let runs: Vec<AnyRuns> = parts.iter().map(|p| count(p, 5, 32)).collect();
        let mut want: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
        for p in &parts {
            let mut here: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
            if p.len() >= 5 {
                for w in p.windows(5) {
                    *here.entry(w.to_vec()).or_insert(0) += 1;
                }
            }
            for (k, c) in here {
                let e = want.entry(k).or_insert(0);
                *e = (*e).max(c);
            }
        }
        match max_merge(runs) {
            AnyRuns::Wide(r) => {
                let got: Vec<(Vec<u32>, u32)> = r;
                let want: Vec<(Vec<u32>, u32)> = want.into_iter().collect();
                assert_eq!(got, want);
            }
            AnyRuns::Packed(_) => panic!("wide parts must merge to wide keys"),
        }
    }

    #[test]
    fn runs_report_totals_and_emptiness() {
        assert_eq!(count(&[4, 4, 5], 1, 32).total(), 3);
        assert_eq!(count(&[4, 4, 5, 6, 7, 8], 5, 32).total(), 2);
        assert!(AnyRuns::Packed(Vec::new()).is_empty());
        assert!(AnyRuns::Wide(Vec::new()).is_empty());
        assert!(!count(&[1], 1, 32).is_empty());
        assert!(!count(&[1, 2, 3, 4, 5], 5, 32).is_empty());
    }
}
