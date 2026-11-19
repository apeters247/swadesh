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
