//! Kernels for bootstrap resampling and paired significance tests.
//!
//! These replace numpy's `stats_f32[idxs]` fancy-index materialization
//! (an `n_samples x n_segments x width` float32 tensor) followed by `.sum(0)`.
//! numpy reduces axis 0 of each `(n_segments, width)` slice by *sequential*
//! float32 addition (verified empirically), which we reproduce exactly.


/// Add the selected rows of `stats` into `acc` in order, with the accumulator
/// held in registers for common widths (BLEU 10, chrF 18, chrF++ 24, TER 2...).
/// Each column is still a sequential f32 sum in the same order, so results are
/// unchanged; only the loop shape differs.
#[inline]
fn accumulate(acc: &mut [f32], stats: &[f32], width: usize, mut next: impl FnMut() -> usize, count: usize) {
    #[inline(always)]
    fn fixed<const W: usize>(acc: &mut [f32], stats: &[f32], next: &mut impl FnMut() -> usize, count: usize) {
        let mut a = [0f32; W];
        a.copy_from_slice(acc);
        for _ in 0..count {
            let i = next();
            let r: &[f32; W] = stats[i * W..(i + 1) * W].try_into().unwrap();
            for j in 0..W {
                a[j] += r[j];
            }
        }
        acc.copy_from_slice(&a);
    }
    match width {
        2 => fixed::<2>(acc, stats, &mut next, count),
        3 => fixed::<3>(acc, stats, &mut next, count),
        4 => fixed::<4>(acc, stats, &mut next, count),
        6 => fixed::<6>(acc, stats, &mut next, count),
        8 => fixed::<8>(acc, stats, &mut next, count),
        10 => fixed::<10>(acc, stats, &mut next, count),
        12 => fixed::<12>(acc, stats, &mut next, count),
        18 => fixed::<18>(acc, stats, &mut next, count),
        24 => fixed::<24>(acc, stats, &mut next, count),
        _ => {
            for _ in 0..count {
                let i = next();
                let r = &stats[i * width..(i + 1) * width];
                for (a, &x) in acc.iter_mut().zip(r) {
                    *a += x;
                }
            }
        }
    }
}

/// For every sample row of `idxs` (`n_samples x n`), sum the selected rows of
/// `stats` (`n x width`, already cast to f32) sequentially in f32.
pub fn bootstrap_sums_f32(stats: &[f32], width: usize, idxs: &[i64], n: usize) -> Vec<f32> {
    let n_samples = idxs.len().checked_div(n).unwrap_or(0);
    let mut out = vec![0f32; n_samples * width];
    crate::par::for_each_chunk(&mut out, width.max(1), |(s, acc)| {
        let row_idx = &idxs[s * n..(s + 1) * n];
        // numpy starts the reduction from the first element (no 0.0 + x),
        // which is equal for finite non-negative values.
        let mut it = row_idx.iter();
        accumulate(acc, stats, width, || *it.next().unwrap() as usize, n);
    });
    out
}

/// Paired approximate randomization: for each trial (row of `sel`, length n),
/// `a = sum(sel ? bl : sys)`, `b = sum(sel ? sys : bl)` (exact integers).
pub fn ar_sums_i64(bl: &[i64], sys: &[i64], width: usize, sel: &[u8], n: usize) -> (Vec<i64>, Vec<i64>) {
    let n_trials = sel.len().checked_div(n).unwrap_or(0);
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

/// Like [`bootstrap_sums_f32`], but draws the resampling indices itself with an
/// exact emulation of `np.random.default_rng(seed).choice(n, (n_samples, n))`
/// (verified against numpy in `tests/nprng.rs`), so no index matrix has to be
/// passed in from Python.
pub fn bootstrap_sums_f32_seeded(stats: &[f32], width: usize, n: usize, n_samples: usize, seed: u128) -> Vec<f32> {
    if n == 0 || n_samples == 0 {
        return Vec::new();
    }
    let mut rng = crate::nprng::default_rng(seed);
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        // Draw sequentially (the stream is inherently serial), sum in parallel.
        let idxs = rng.integers(0, n as i64, n_samples * n);
        return bootstrap_sums_f32(stats, width, &idxs, n);
    }
    // Single thread: fuse drawing and summing; no index matrix at all.
    let mut out = vec![0f32; n_samples * width];
    for acc in out.chunks_mut(width.max(1)) {
        accumulate(acc, stats, width, || rng.next_below(n as u64) as usize, n);
    }
    out
}
