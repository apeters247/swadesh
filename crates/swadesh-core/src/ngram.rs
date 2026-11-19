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
