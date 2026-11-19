//! Bit-exact emulation of the numpy (2.x) random and reduction pieces that
//! sacreBLEU 2.6.0's bootstrap and paired significance tests rely on.
//!
//! Covered, each verified bit-for-bit against numpy in `tests/nprng.rs`:
//! - [`SeedSequence`]: numpy's hash-mixing seed sequence (pool size 4).
//! - [`Pcg64`]: numpy's PCG64 bit generator (128-bit LCG, XSL-RR output),
//!   seeded the way `np.random.default_rng(seed)` seeds it.
//! - [`Generator`]: `integers(low, high, size)` (Lemire bounded integers),
//!   `choice(n, size=(rows, cols), replace=True)`, and `integers(2, dtype=bool)`.
//! - [`mean_f64`] (numpy pairwise summation), [`sort_f64`] (NaN-last),
//!   [`statistics_mean`] (CPython `statistics.mean`, exact then correctly
//!   rounded), and [`sum_rows_f32`] (numpy `arr[idx].sum(0)` in float32).
//!
//! Not emulated: `seed=None` (sacreBLEU's `SACREBLEU_SEED=none`). It draws OS
//! entropy, so there is no fixed output to reproduce.

use std::ops::Add;

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

/// PCG 128-bit default multiplier: `(2549297995355413924 << 64) + 4865540595714422341`.
const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

#[inline]
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

#[inline]
fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// numpy's `SeedSequence(entropy)` for an integer entropy (no spawn key).
#[derive(Clone, Debug)]
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    /// `np.random.SeedSequence(seed)` for a non-negative integer seed.
    pub fn new(entropy: u128) -> Self {
        // numpy coerces an int to little-endian uint32 words, at least one word.
        let mut words = [0u32; POOL_SIZE];
        let mut len = 0usize;
        let mut e = entropy;
        loop {
            words[len] = e as u32;
            len += 1;
            e >>= 32;
            if e == 0 {
                break;
            }
        }
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            let v = if i < len { words[i] } else { 0 };
            *slot = hashmix(v, &mut hash_const);
        }
        for src in 0..POOL_SIZE {
            for dst in 0..POOL_SIZE {
                if src != dst {
                    let h = hashmix(pool[src], &mut hash_const);
                    pool[dst] = mix(pool[dst], h);
                }
            }
        }
        // Entropy never exceeds the pool for a u128 seed, so no extra mixing loop.
        Self { pool }
    }

    /// `generate_state(n_words, dtype=np.uint32)`.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut v = self.pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^ (v >> XSHIFT)
            })
            .collect()
    }

    /// `generate_state(n_words, dtype=np.uint64)`: each u64 is two consecutive
    /// u32 words, low word first (little-endian view).
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let s = self.generate_state_u32(2 * n_words);
        (0..n_words)
            .map(|i| (s[2 * i] as u64) | ((s[2 * i + 1] as u64) << 32))
            .collect()
    }
}

/// numpy's PCG64 bit generator.
#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    /// `np.random.PCG64(seed)` for an integer seed.
    pub fn new(seed: u128) -> Self {
        let v = SeedSequence::new(seed).generate_state_u64(4);
        let initstate = ((v[0] as u128) << 64) | v[1] as u128;
        let initseq = ((v[2] as u128) << 64) | v[3] as u128;
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// `random_raw`: the next 64-bit output. Steps first, then outputs the new state.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        let x = ((s >> 64) as u64) ^ (s as u64);
        let rot = (s >> 122) as u32;
        x.rotate_right(rot)
    }

    /// Buffered 32-bit output: returns the low half of a 64-bit draw, then
    /// the cached high half on the next call (numpy's `next_uint32`).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let x = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (x >> 32) as u32;
        x as u32
    }
}

/// `np.random.Generator` over PCG64, as returned by `np.random.default_rng(seed)`.
#[derive(Clone, Debug)]
pub struct Generator {
    bitgen: Pcg64,
}

/// Lemire's method with a 32-bit range (`rng` is inclusive max, `rng < u32::MAX`).
#[inline]
fn lemire_u32(bg: &mut Pcg64, rng: u32) -> u32 {
    let excl = rng + 1;
    let mut m = (bg.next_u32() as u64) * (excl as u64);
    let mut left = m as u32;
    if left < excl {
        let thr = (u32::MAX - rng) % excl;
        while left < thr {
            m = (bg.next_u32() as u64) * (excl as u64);
            left = m as u32;
        }
    }
    (m >> 32) as u32
}

/// Lemire's method with a 64-bit range (`rng < u64::MAX`).
#[inline]
fn lemire_u64(bg: &mut Pcg64, rng: u64) -> u64 {
    let excl = rng + 1;
    let mut m = (bg.next_u64() as u128) * (excl as u128);
    let mut left = m as u64;
    if left < excl {
        let thr = (u64::MAX - rng) % excl;
        while left < thr {
            m = (bg.next_u64() as u128) * (excl as u128);
            left = m as u64;
        }
    }
    (m >> 64) as u64
}

/// numpy's `random_bounded_uint64_fill` for one value: uniform in `0..=rng`.
#[inline]
fn bounded_u64(bg: &mut Pcg64, rng: u64) -> u64 {
    if rng == 0 {
        0
    } else if rng <= u32::MAX as u64 {
        if rng == u32::MAX as u64 {
            bg.next_u32() as u64
        } else {
            lemire_u32(bg, rng as u32) as u64
        }
    } else if rng == u64::MAX {
        bg.next_u64()
    } else {
        lemire_u64(bg, rng)
    }
}

impl Generator {
    /// `np.random.default_rng(seed)` for an integer seed.
    pub fn new(seed: u128) -> Self {
        Self {
            bitgen: Pcg64::new(seed),
        }
    }

    /// Access to the underlying PCG64 (e.g. for `random_raw`-style draws).
    pub fn bit_generator(&mut self) -> &mut Pcg64 {
        &mut self.bitgen
    }

    /// `rng.integers(low, high, size=n, dtype=np.int64)` with numpy's default
    /// `endpoint=False`, so values are uniform in `low..high`. Returns a flat
    /// vector in C order.
    pub fn integers(&mut self, low: i64, high: i64, n: usize) -> Vec<i64> {
        assert!(low < high, "integers: low must be < high");
        let rng = (high - 1).wrapping_sub(low) as u64;
        let off = low as u64;
        (0..n)
            .map(|_| off.wrapping_add(bounded_u64(&mut self.bitgen, rng)) as i64)
            .collect()
    }

    /// `rng.choice(n, size=(rows, cols), replace=True)`, flattened in C order.
    /// Equivalent to `integers(0, n, size=(rows, cols))`.
    pub fn choice(&mut self, n: usize, rows: usize, cols: usize) -> Vec<usize> {
        assert!(n >= 1, "choice: population must be non-empty");
        self.integers(0, n as i64, rows * cols)
            .into_iter()
            .map(|v| v as usize)
            .collect()
    }

    /// `rng.integers(2, size=(rows, cols), dtype=bool)`, flattened in C order.
    /// numpy draws one `next_uint32` per 32 booleans and consumes its bits
    /// least-significant first. The buffer is local to each call.
    pub fn integers_bool(&mut self, rows: usize, cols: usize) -> Vec<bool> {
        let n = rows * cols;
        let mut out = Vec::with_capacity(n);
        let mut buf = 0u32;
        for i in 0..n {
            if i % 32 == 0 {
                buf = self.bitgen.next_u32();
            }
            out.push((buf >> (i % 32)) & 1 == 1);
        }
        out
    }
}

/// `np.random.default_rng(seed)`.
pub fn default_rng(seed: u128) -> Generator {
    Generator::new(seed)
}

/// numpy's `pairwise_sum` (loops_utils.h.src): blocks of 128, 8-way unrolled,
/// recursive halving. Works for float32 and float64 accumulation.
fn pairwise<T>(a: &[T]) -> T
where
    T: Copy + Default + Add<Output = T>,
{
    let n = a.len();
    if n < 8 {
        let mut res = T::default();
        for &x in a {
            res = res + x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        let blocked_end = n - (n % 8);
        while i < blocked_end {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj = *rj + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res = res + a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// `np.mean(x)` for a 1-D float64 array: pairwise sum divided by `n`.
/// Returns NaN for an empty slice, as numpy does.
pub fn mean_f64(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    pairwise(xs) / xs.len() as f64
}

/// Sum of a float32 slice with numpy's pairwise order (used for a 1-column reduction).
pub fn pairwise_sum_f32(xs: &[f32]) -> f32 {
    pairwise(xs)
}

/// `np.sort` for float64: ascending, NaNs last. The order of `-0.0` and `0.0`
/// is unspecified (they compare equal), and it cannot change a later mean.
pub fn sort_f64(xs: &mut [f64]) {
    xs.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap(),
    });
}

/// Power of two `2^e` as an exact f64, for `-1074 <= e <= 1023`.
fn pow2(e: i64) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// Correctly rounded (round-half-even) value of `num / den`, `den > 0`.
fn round_ratio(num: &BigInt, den: &BigUint) -> f64 {
    if num.sign() == Sign::NoSign {
        return 0.0;
    }
    let neg = num.sign() == Sign::Minus;
    let a = num.magnitude();
    // Scale so the quotient has >= 56 significant bits and the dropped
    // bit count is >= 2 for every representable result.
    let k = std::cmp::max(
        56 + den.bits() as i64 - a.bits() as i64,
        1076,
    );
    let shifted = a << (k as usize);
    let q = &shifted / den;
    let sticky = !(&shifted % den).is_zero();
    let e_lead = q.bits() as i64 - 1 - k;
    let lsb_exp = std::cmp::max(e_lead - 52, -1074);
    let d = (lsb_exp + k) as usize;
    let mant = &q >> d;
    let rem = &q - (&mant << d);
    let half = BigUint::one() << (d - 1);
    let mut m = mant.iter_u64_digits().next().unwrap_or(0);
    let odd = m & 1 == 1;
    if rem > half || (rem == half && (sticky || odd)) {
        m += 1;
    }
    let top = lsb_exp + (64 - m.leading_zeros() as i64) - 1;
    let mag = if top > 1023 {
        f64::INFINITY
    } else {
        (m as f64) * pow2(lsb_exp)
    };
    if neg {
        -mag
    } else {
        mag
    }
}

/// CPython 3.11 `statistics.mean(list_of_floats)`.
///
/// The finite case is the exact rational mean `sum(xs) / n`, correctly rounded
/// to f64 (round-half-even), as `float(Fraction)` gives. If any input is
/// non-finite, CPython sums only the non-finite values in input order and
/// divides by `n`. Returns `None` for an empty slice (`StatisticsError` in Python).
pub fn statistics_mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let n = xs.len();
    if xs.iter().any(|x| !x.is_finite()) {
        let total = xs
            .iter()
            .filter(|x| !x.is_finite())
            .fold(0.0f64, |acc, &x| acc + x);
        return Some(total / n as f64);
    }
    // Every finite f64 is m * 2^e with e >= -1074, so scale by 2^1074 to an integer.
    let mut sum = BigInt::zero();
    for &x in xs {
        let bits = x.to_bits();
        let neg = bits >> 63 == 1;
        let biased = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1u64 << 52) - 1);
        let (m, e) = if biased == 0 {
            (frac, -1074i64)
        } else {
            (frac | (1u64 << 52), biased - 1075)
        };
        if m == 0 {
            continue;
        }
        let term = BigInt::from(m) << ((e + 1074) as usize);
        if neg {
            sum -= term;
        } else {
            sum += term;
        }
    }
    let den = BigUint::from(n as u64) << 1074usize;
    Some(round_ratio(&sum, &den))
}

/// `arr[idx].sum(0)` for a C-contiguous float32 matrix `mat` of `width` columns,
/// where `idx` selects rows (with repeats). numpy adds rows sequentially in
/// float32 (an elementwise add per row, not a pairwise tree). For `width == 1`
/// the single column is reduced with pairwise summation instead.
pub fn sum_rows_f32(mat: &[f32], width: usize, idx: &[usize]) -> Vec<f32> {
    if width == 1 {
        let col: Vec<f32> = idx.iter().map(|&i| mat[i]).collect();
        return vec![pairwise(&col)];
    }
    let mut out = vec![0.0f32; width];
    for &i in idx {
        let row = &mat[i * width..(i + 1) * width];
        for (o, &v) in out.iter_mut().zip(row) {
            *o += v;
        }
    }
    out
}
