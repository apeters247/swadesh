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
    if crate::par::num_threads() > 1 {
        // Draw sequentially (the stream is inherently serial), sum in parallel,
        // in batches of at most MAX_IDX indices so memory stays bounded
        // (a 10M-segment corpus x 1000 resamples would otherwise need 80 GB).
        const MAX_IDX: usize = 1 << 24;
        let batch = (MAX_IDX / n).max(1);
        let mut out = Vec::with_capacity(n_samples * width);
        let mut done = 0;
        let mut idxs: Vec<i64> = Vec::with_capacity(batch.min(n_samples) * n);
        while done < n_samples {
            let rows = batch.min(n_samples - done);
            idxs.clear();
            idxs.extend((0..rows * n).map(|_| rng.next_below(n as u64) as i64));
            out.extend(bootstrap_sums_f32(stats, width, &idxs, n));
            done += rows;
        }
        return out;
    }
    // Single thread: fuse drawing and summing; no index matrix at all.
    let mut out = vec![0f32; n_samples * width];
    for acc in out.chunks_mut(width.max(1)) {
        accumulate(acc, stats, width, || rng.next_below(n as u64) as usize, n);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 26-bit counts: sums exceed 2^24, so float32 rounding is exercised.
    fn counts(len: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 33) as u32 & 0x03ff_ffff) as f32
            })
            .collect()
    }

    /// numpy's `stats[idxs].sum(0)` for each sample: rows added one at a time in float32.
    fn naive_sums(stats: &[f32], width: usize, idxs: &[i64], n: usize) -> Vec<f32> {
        #[allow(clippy::manual_checked_ops)]  // n == 0 is the empty-index case, not a fallback
        let n_samples = if n == 0 { 0 } else { idxs.len() / n };
        let mut out = vec![0f32; n_samples * width];
        for s in 0..n_samples {
            for &i in &idxs[s * n..(s + 1) * n] {
                for j in 0..width {
                    out[s * width + j] += stats[i as usize * width + j];
                }
            }
        }
        out
    }

    #[test]
    fn every_width_matches_sequential_f32_sum() {
        // Specialized widths (2..24) and generic ones (1, 5, 7, 9, 11, 13, 25, ...).
        for width in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 23, 24, 25, 30] {
            let rows = 37;
            let n = 80; // more draws than rows, so rows repeat
            let stats = counts(rows * width, width as u64 + 1);
            let mut idxs = Vec::new();
            let mut x = 99u64 + width as u64;
            for _ in 0..(5 * n) {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                idxs.push(((x >> 33) % rows as u64) as i64);
            }
            let got = bootstrap_sums_f32(&stats, width, &idxs, n);
            let want = naive_sums(&stats, width, &idxs, n);
            assert_eq!(got.len(), want.len());
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "width {width} entry {k}");
            }
        }
    }

    #[test]
    fn empty_draws_give_empty_result() {
        assert!(bootstrap_sums_f32(&[1.0, 2.0], 2, &[], 3).is_empty());
        assert!(bootstrap_sums_f32_seeded(&[], 2, 0, 3, 1).is_empty());
        assert!(bootstrap_sums_f32_seeded(&[1.0, 2.0], 2, 2, 0, 1).is_empty());
    }

    #[test]
    fn seeded_sums_match_drawn_indices() {
        for (width, n, samples, seed) in [(3usize, 20usize, 9usize, 12345u128), (10, 7, 4, 0), (1, 1, 5, 9)] {
            let stats = counts(n * width, seed as u64 + 5);
            let idxs: Vec<i64> = crate::nprng::default_rng(seed)
                .choice(n, samples, n)
                .into_iter()
                .map(|v| v as i64)
                .collect();
            let want = naive_sums(&stats, width, &idxs, n);
            let got = bootstrap_sums_f32_seeded(&stats, width, n, samples, seed);
            assert_eq!(got.len(), want.len());
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "width {width} seed {seed} entry {k}");
            }
        }
    }

    #[test]
    fn seeded_sums_batch_large_draws() {
        // 17 * 2^20 indices exceed MAX_IDX (2^24), so the parallel path is split into batches.
        let (width, n, samples, seed) = (1usize, 1usize << 20, 17usize, 4242u128);
        let stats = counts(n * width, 7);
        let idxs: Vec<i64> = crate::nprng::default_rng(seed)
            .choice(n, samples, n)
            .into_iter()
            .map(|v| v as i64)
            .collect();
        let want = naive_sums(&stats, width, &idxs, n);
        let got = bootstrap_sums_f32_seeded(&stats, width, n, samples, seed);
        assert_eq!(got.len(), want.len());
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "entry {k}");
        }
    }

    #[test]
    fn paired_sums_select_by_mask_exactly() {
        for width in [1usize, 2, 3, 5] {
            let n = 9;
            let trials = 6;
            let bl: Vec<i64> = (0..n * width).map(|i| (i as i64 * 37) % 101 - 50).collect();
            let sys: Vec<i64> = (0..n * width).map(|i| (i as i64 * 53 + 7) % 97 - 40).collect();
            let sel: Vec<u8> = (0..trials * n).map(|i| ((i * 7 + i / 3) % 2) as u8).collect();
            let (a, b) = ar_sums_i64(&bl, &sys, width, &sel, n);
            assert_eq!(a.len(), trials * width);
            for t in 0..trials {
                for j in 0..width {
                    let (mut ea, mut eb) = (0i64, 0i64);
                    for i in 0..n {
                        let pick = sel[t * n + i] != 0;
                        let (x, y) = (bl[i * width + j], sys[i * width + j]);
                        ea += if pick { x } else { y };
                        eb += if pick { y } else { x };
                    }
                    assert_eq!(a[t * width + j], ea, "a width {width} trial {t} col {j}");
                    assert_eq!(b[t * width + j], eb, "b width {width} trial {t} col {j}");
                }
            }
        }
    }
}

/// sacreBLEU's `Score.estimate_ci`: the mean of the resample scores and the half-width of the
/// 95% interval (the 1/40 and 39/40 order statistics, halved).
///
/// CPython's `statistics.mean` sums floats exactly, so on the dots it can differ from any
/// floating-point sum in the last ulp. Measured 2026-11-26 over 200,000 realistic samples
/// (n = 50..10,000, scores in 0..100, 5% near-zero): a plain `f64` sum, `math.fsum` and a
/// Neumaier-compensated sum all agree with `statistics.mean` at the two decimals that are
/// printed, 0 differences in each. The compensated sum is used here because it costs nothing and
/// is far closer when it does not agree. Byte-exactness against sacreBLEU's `-ci` output is
/// checked end to end by the CLI parity tests.
pub fn estimate_ci(scores: &mut [f64]) -> (f64, f64) {
    scores.sort_by(|a, b| a.partial_cmp(b).expect("resample scores are never NaN"));
    let n = scores.len();
    let lower_idx = n / 40;
    let upper_idx = n - lower_idx - 1;
    let ci = 0.5 * (scores[upper_idx] - scores[lower_idx]);
    let mut total = 0f64;
    let mut comp = 0f64;
    for &v in scores.iter() {
        let t = total + v;
        if total.abs() >= v.abs() {
            comp += (total - t) + v;
        } else {
            comp += (v - t) + total;
        }
        total = t;
    }
    ((total + comp) / n as f64, ci)
}

#[cfg(test)]
mod estimate_ci_tests {
    use super::estimate_ci;

    #[test]
    fn order_statistics_and_mean() {
        let mut v: Vec<f64> = (0..1000).map(|i| i as f64 * 0.25).collect();
        let (mean, ci) = estimate_ci(&mut v);
        assert_eq!(mean, v.iter().sum::<f64>() / 1000.0);
        assert_eq!(ci, 0.5 * (v[974] - v[25])); // n=1000 -> lower 25, upper 974
    }

    #[test]
    fn unsorted_input_is_sorted_first() {
        let mut v = vec![10f64, 1.0, 5.0, 7.0, 3.0];
        let (mean, ci) = estimate_ci(&mut v);
        assert_eq!(mean, 26.0 / 5.0);
        assert_eq!(ci, 0.5 * (10.0 - 1.0)); // n=5 -> lower 0, upper 4
    }
}
