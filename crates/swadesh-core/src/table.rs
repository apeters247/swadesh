//! A tiny, allocation-free (after warm-up) open-addressing count table keyed by
//! packed n-grams. Cleared in O(1) via generation stamps. Exact: keys are the
//! full packed n-gram, never a lossy hash.

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u128,
    count: u32,
    gen: u32,
}

pub struct CountTable {
    entries: Vec<Entry>,
    mask: usize,
    shift: u32,
    gen: u32,
}

impl Default for CountTable {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn mix(key: u128) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    (lo ^ hi.rotate_left(31)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

impl CountTable {
    pub fn new() -> Self {
        let mut t = CountTable { entries: Vec::new(), mask: 0, shift: 64, gen: 1 };
        t.resize(64);
        t
    }

    fn resize(&mut self, cap: usize) {
        let cap = cap.next_power_of_two().max(16);
        self.entries = vec![Entry::default(); cap];
        self.mask = cap - 1;
        self.shift = 64 - cap.trailing_zeros();
        self.gen = 1;
    }

    /// Prepare for up to `n` distinct keys; logically empties the table.
    #[inline]
    pub fn reset(&mut self, n: usize) {
        if n * 2 > self.entries.len() {
            self.resize(n * 2);
            return;
        }
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            for e in self.entries.iter_mut() {
                e.gen = 0;
            }
            self.gen = 1;
        }
    }

    #[inline(always)]
    fn find(&self, key: u128) -> usize {
        let mut i = (mix(key) >> self.shift) as usize;
        loop {
            let e = &self.entries[i];
            if e.gen != self.gen || e.key == key {
                return i;
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline(always)]
    pub fn add(&mut self, key: u128) {
        let i = self.find(key);
        let g = self.gen;
        let e = &mut self.entries[i];
        if e.gen != g {
            *e = Entry { key, count: 1, gen: g };
        } else {
            e.count += 1;
        }
    }

    #[inline(always)]
    pub fn get(&self, key: u128) -> u32 {
        let i = self.find(key);
        let e = &self.entries[i];
        if e.gen == self.gen {
            e.count
        } else {
            0
        }
    }

    /// Set `count[key] = max(count[key], c)`.
    #[inline(always)]
    pub fn set_max(&mut self, key: u128, c: u32) {
        let i = self.find(key);
        let g = self.gen;
        let e = &mut self.entries[i];
        if e.gen != g {
            *e = Entry { key, count: c, gen: g };
        } else if c > e.count {
            e.count = c;
        }
    }

    /// Consume one occurrence of `key` if available (clipped matching).
    #[inline(always)]
    pub fn take(&mut self, key: u128) -> bool {
        let i = self.find(key);
        let e = &mut self.entries[i];
        if e.gen == self.gen && e.count > 0 {
            e.count -= 1;
            true
        } else {
            false
        }
    }
}

/// Iterate packed keys of all length-`n` windows of `seq` (`bits` bits/element).
/// Requires `n * bits <= 128`.
#[inline]
pub fn packed_windows(seq: &[u32], n: usize, bits: u32) -> impl Iterator<Item = u128> + '_ {
    let total = n as u32 * bits;
    let mask: u128 = if total >= 128 { u128::MAX } else { (1u128 << total) - 1 };
    let mut k: u128 = 0;
    seq.iter().enumerate().filter_map(move |(i, &x)| {
        k = ((k << bits) | x as u128) & mask;
        if i + 1 >= n {
            Some(k)
        } else {
            None
        }
    })
}

#[inline]
pub fn fits(n: usize, bits: u32) -> bool {
    n as u32 * bits <= 128
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn fits_is_the_128_bit_budget() {
        assert!(fits(4, 32));
        assert!(!fits(5, 32));
        assert!(fits(6, 21));
        assert!(!fits(7, 21));
        assert!(fits(128, 1));
        assert!(!fits(129, 1));
        assert!(!fits(1, 129));
    }

    #[test]
    fn packed_windows_are_exact_big_endian_keys() {
        let seq: Vec<u32> = (0..50u32).map(|i| i * 7919 % 1000).collect();
        for n in 1..=4 {
            let got: Vec<u128> = packed_windows(&seq, n, 32).collect();
            let want: Vec<u128> = seq
                .windows(n)
                .map(|w| w.iter().fold(0u128, |k, &x| (k << 32) | x as u128))
                .collect();
            assert_eq!(got, want, "n {n}");
        }
        let cps: Vec<u32> = "héllo wörld 中文 𝔸ok".chars().map(|c| c as u32).collect();
        for n in 1..=6 {
            let got: Vec<u128> = packed_windows(&cps, n, 21).collect();
            let want: Vec<u128> = cps
                .windows(n)
                .map(|w| w.iter().fold(0u128, |k, &x| (k << 21) | x as u128))
                .collect();
            assert_eq!(got, want, "code points n {n}");
        }
    }

    /// Random operation sequences across resets (including growth) against a HashMap model.
    #[test]
    fn count_table_matches_hashmap_model() {
        let mut t = CountTable::new();
        let mut x = 1u64;
        let mut next = move || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            x >> 33
        };
        for round in 0..40u64 {
            let n = if round % 9 == 8 { 3000 } else { (2 + round * 7) as usize };
            // Keys are distinct per round up to `m` values in two 'planes', so at most n keys.
            let m = (n / 2).max(1) as u64;
            t.reset(n);
            let mut model: HashMap<u128, u32> = HashMap::new();
            for _ in 0..(3 * n) {
                let key = (next() % m) as u128 | (((next() & 1) as u128) << 100);
                match next() % 4 {
                    0 => {
                        t.add(key);
                        *model.entry(key).or_insert(0) += 1;
                    }
                    1 => {
                        let c = (next() % 5) as u32;
                        t.set_max(key, c);
                        model
                            .entry(key)
                            .and_modify(|v| *v = (*v).max(c))
                            .or_insert(c);
                    }
                    2 => {
                        let want = match model.get_mut(&key) {
                            Some(v) if *v > 0 => {
                                *v -= 1;
                                true
                            }
                            _ => false,
                        };
                        assert_eq!(t.take(key), want, "take round {round}");
                    }
                    _ => {
                        let want = model.get(&key).copied().unwrap_or(0);
                        assert_eq!(t.get(key), want, "get round {round}");
                    }
                }
            }
            for (k, v) in &model {
                assert_eq!(t.get(*k), *v, "final round {round}");
            }
        }
    }

    /// The generation counter wraps to 0 after 2^32 resets. Entries stamped with generation 0
    /// must not be read as live afterwards.
    #[test]
    fn generation_wraparound_forgets_old_entries() {
        let mut t = CountTable::new();
        for e in t.entries.iter_mut() {
            *e = Entry { key: 7, count: 5, gen: 0 };
        }
        t.gen = u32::MAX;
        t.reset(4);
        assert_eq!(t.gen, 1);
        assert_eq!(t.get(7), 0);
        assert!(!t.take(7));
        t.add(7);
        assert_eq!(t.get(7), 1);
    }
}
