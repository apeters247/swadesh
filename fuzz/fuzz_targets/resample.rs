#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use swadesh_core::resample::{ar_sums_i64, bootstrap_sums_f32, bootstrap_sums_f32_seeded};

// Structured inputs: `n` segments, `width` columns, and every index is drawn
// from 0..n (the documented precondition). Stats are bounded so i64 sums cannot
// overflow (sufficient statistics are counts far below 2^40 in practice).
#[derive(Arbitrary, Debug)]
struct Input {
    width: u8,
    n: u8,
    n_samples: u8,
    seed: u128,
    f32_bits: Vec<u32>,
    idx_bytes: Vec<u8>,
    i64_stats: Vec<u32>,
    sel: Vec<u8>,
}

fuzz_target!(|i: Input| {
    let width = (i.width % 26) as usize; // includes 0 and non-specialized widths
    let n = (i.n % 48) as usize;
    let n_samples = (i.n_samples % 48) as usize;

    // --- f32 bootstrap sums against a naive sequential reference.
    let stats: Vec<f32> = (0..n * width)
        .map(|k| f32::from_bits(*i.f32_bits.get(k % i.f32_bits.len().max(1)).unwrap_or(&0)))
        .collect();
    let idxs: Vec<i64> = (0..n_samples * n)
        .map(|k| {
            if n == 0 {
                0
            } else {
                (*i.idx_bytes.get(k % i.idx_bytes.len().max(1)).unwrap_or(&0) as usize % n) as i64
            }
        })
        .collect();
    let got = bootstrap_sums_f32(&stats, width, &idxs, n);
    let ns = if n == 0 { 0 } else { n_samples };
    assert_eq!(got.len(), ns * width);
    for s in 0..ns {
        for j in 0..width {
            let mut acc = 0f32;
            for k in 0..n {
                acc += stats[idxs[s * n + k] as usize * width + j];
            }
            let g = got[s * width + j];
            assert!(g.to_bits() == acc.to_bits() || (g.is_nan() && acc.is_nan()), "f32 mismatch s={s} j={j}");
        }
    }

    // --- seeded bootstrap: shape and determinism.
    let seeded = bootstrap_sums_f32_seeded(&stats, width, n, n_samples, i.seed);
    let seeded_again = bootstrap_sums_f32_seeded(&stats, width, n, n_samples, i.seed);
    assert_eq!(seeded.len(), if n == 0 { 0 } else { n_samples * width });
    assert!(seeded.iter().zip(&seeded_again).all(|(a, b)| a.to_bits() == b.to_bits()));

    // --- paired approximate randomization: exact i64 sums against a naive reference.
    let bl: Vec<i64> = (0..n * width).map(|k| i.i64_stats.get(k % i.i64_stats.len().max(1)).map_or(0, |&x| x as i64)).collect();
    let sys: Vec<i64> = bl.iter().rev().map(|&x| x / 2 + 1).collect();
    let trials = if n == 0 { 0 } else { (i.sel.len() / n).min(48) };
    let sel: Vec<u8> = i.sel.iter().take(trials * n).map(|&b| b & 1).collect();
    let (a, b) = ar_sums_i64(&bl, &sys, width, &sel, n);
    assert_eq!(a.len(), trials * width);
    assert_eq!(b.len(), trials * width);
    for t in 0..trials {
        for j in 0..width {
            let (mut wa, mut wb) = (0i64, 0i64);
            for k in 0..n {
                let picked = sel[t * n + k] != 0;
                let (x, y) = if picked { (bl[k * width + j], sys[k * width + j]) } else { (sys[k * width + j], bl[k * width + j]) };
                wa += x;
                wb += y;
            }
            assert_eq!((a[t * width + j], b[t * width + j]), (wa, wb));
        }
    }
});
