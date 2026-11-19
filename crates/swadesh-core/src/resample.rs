//! Kernels for bootstrap resampling and paired significance tests.
//!
//! These replace numpy's `stats_f32[idxs]` fancy-index materialization
//! (an `n_samples x n_segments x width` float32 tensor) followed by `.sum(0)`.
//! numpy reduces axis 0 of each `(n_segments, width)` slice by *sequential*
//! float32 addition (verified empirically), which we reproduce exactly.


/// For every sample row of `idxs` (`n_samples x n`), sum the selected rows of
/// `stats` (`n x width`, already cast to f32) sequentially in f32.
pub fn bootstrap_sums_f32(stats: &[f32], width: usize, idxs: &[i64], n: usize) -> Vec<f32> {
    let n_samples = if n == 0 { 0 } else { idxs.len() / n };
    let mut out = vec![0f32; n_samples * width];
    crate::par::for_each_chunk(&mut out, width.max(1), |(s, acc)| {
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
    let fill = |out: &mut Vec<i64>, pick_bl_when: u8| {
        crate::par::for_each_chunk(out, width.max(1), |(t, row)| {
            let s = &sel[t * n..(t + 1) * n];
            for (i, &pick) in s.iter().enumerate() {
                let src = if (pick != 0) == (pick_bl_when != 0) { bl } else { sys };
                let x = &src[i * width..(i + 1) * width];
                for j in 0..width {
                    row[j] += x[j];
                }
            }
        });
    };
    // a = sel ? bl : sys ; b = sel ? sys : bl
    fill(&mut a, 1);
    fill(&mut b, 0);
    (a, b)
}
