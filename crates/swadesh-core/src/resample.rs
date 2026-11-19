//! Kernels for bootstrap resampling and paired significance tests.
//!
//! These replace numpy's `stats_f32[idxs]` fancy-index materialization
//! (an `n_samples x n_segments x width` float32 tensor) followed by `.sum(0)`.
//! numpy reduces axis 0 of each `(n_segments, width)` slice by *sequential*
//! float32 addition (verified empirically), which we reproduce exactly.

use rayon::prelude::*;

/// For every sample row of `idxs` (`n_samples x n`), sum the selected rows of
/// `stats` (`n x width`, already cast to f32) sequentially in f32.
pub fn bootstrap_sums_f32(stats: &[f32], width: usize, idxs: &[i64], n: usize) -> Vec<f32> {
    let n_samples = if n == 0 { 0 } else { idxs.len() / n };
    let mut out = vec![0f32; n_samples * width];
    out.par_chunks_mut(width).enumerate().for_each(|(s, acc)| {
        let row_idx = &idxs[s * n..(s + 1) * n];
        // numpy starts the reduction from the first element (no 0.0 + x),
        // which is equal for finite values (0.0 + x == x, including -0.0 → 0.0
        // differences that cannot occur for non-negative stats).
        for &i in row_idx {
            let r = &stats[i as usize * width..(i as usize + 1) * width];
            for (a, &x) in acc.iter_mut().zip(r) {
                *a += x;
            }
        }
    });
    out
}

/// Paired approximate randomization: for each trial (row of `sel`, length n),
/// `a = sum(sel ? bl : sys)`, `b = sum(sel ? sys : bl)` (exact integers).
pub fn ar_sums_i64(bl: &[i64], sys: &[i64], width: usize, sel: &[u8], n: usize) -> (Vec<i64>, Vec<i64>) {
    let n_trials = if n == 0 { 0 } else { sel.len() / n };
    let mut a = vec![0i64; n_trials * width];
    let mut b = vec![0i64; n_trials * width];
    a.par_chunks_mut(width)
        .zip(b.par_chunks_mut(width))
        .enumerate()
        .for_each(|(t, (ra, rb))| {
            let s = &sel[t * n..(t + 1) * n];
            for (i, &pick) in s.iter().enumerate() {
                let x = &bl[i * width..(i + 1) * width];
                let y = &sys[i * width..(i + 1) * width];
                if pick != 0 {
                    for j in 0..width {
                        ra[j] += x[j];
                        rb[j] += y[j];
                    }
                } else {
                    for j in 0..width {
                        ra[j] += y[j];
                        rb[j] += x[j];
                    }
                }
            }
        });
    (a, b)
}
