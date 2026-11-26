//! sacreBLEU's bootstrap confidence interval: draw resample indices, score each resampled row,
//! then take the mean and the 1/40..39/40 half-spread (`estimate_ci`).
//!
//! The index drawing and the row scoring are the emulations already in `resample` and `score`;
//! this module is the orchestration `sacrebleu.significance._bootstrap_resample` does in Python,
//! written here so both the CLI and the Python binding can call one implementation.

use crate::nprng::{default_rng, mean_f64, sort_f64};
use crate::resample::{ar_sums_i64, bootstrap_sums_f32, bootstrap_sums_f32_seeded};
use crate::score::{chrf_f_score, compute_bleu};
use crate::score::{chrf_f_score_np2_f32, compute_bleu_np2_f32, BleuScoreConfig, NpVal};

/// Which metric a statistics row belongs to. The BLEU configuration is owned, so a `ScoreKind` can
/// be returned from a helper and stored.
#[derive(Clone, Copy, Debug)]
pub enum ScoreKind {
    /// Rows are `[sys_len, ref_len, correct.., total..]`, `2 + 2 * max_order` wide.
    Bleu(BleuScoreConfig),
    /// Rows are `[hyp, ref, match]` per order, `3 * order` wide.
    Chrf { order: usize, beta: i64, eps_smoothing: bool },
    /// Rows are `[num_edits, avg_ref_len]`.
    Ter,
}

/// `100 * edits / ref_len` for a `float32` TER row: sacreBLEU's `TER._compute_score_from_stats`,
/// with the empty-reference rules (1.0 for edits without reference words, 0.0 for nothing).
pub fn ter_score_np2_f32(row: &[f32]) -> f64 {
    let total_edits = row[0];
    let sum_ref_lengths = row[1];
    let score = if sum_ref_lengths > 0.0 {
        total_edits / sum_ref_lengths
    } else if total_edits > 0.0 {
        1.0
    } else {
        0.0
    };
    (100.0f32 * score) as f64
}

/// Score one `float32` statistics row for `kind`.
pub fn score_row(kind: &ScoreKind, row: &[f32]) -> f64 {
    match kind {
        ScoreKind::Bleu(cfg) => compute_bleu_np2_f32(row, cfg),
        ScoreKind::Chrf { order, beta, eps_smoothing } => {
            match chrf_f_score_np2_f32(row, *order, *beta, *eps_smoothing) {
                NpVal::F32(x) => x as f64,
                NpVal::Py(x) => x,
            }
        }
        ScoreKind::Ter => ter_score_np2_f32(row),
    }
}

/// The bootstrap confidence interval of the corpus score: `(mean, half_width)` over `n_samples`
/// resamples of `rows` (flattened, `width` per segment), drawn with `seed`.
pub fn bootstrap_ci(
    rows: &[f64],
    width: usize,
    kind: &ScoreKind,
    n_samples: usize,
    seed: u128,
) -> (f64, f64) {
    let n = rows.len() / width;
    let as_f32: Vec<f32> = rows.iter().map(|&v| v as f32).collect();
    let sums = bootstrap_sums_f32_seeded(&as_f32, width, n, n_samples, seed);
    let mut scores: Vec<f64> = sums.chunks(width).map(|r| score_row(kind, r)).collect();
    estimate_ci_np(&mut scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{BleuScoreConfig, Smooth, SumMode};

    fn cfg() -> BleuScoreConfig {
        BleuScoreConfig {
            smooth: Smooth::Exp,
            smooth_value: 0.0,
            effective_order: false,
            max_order: 2,
            sum_mode: SumMode::Neumaier,
        }
    }

    #[test]
    fn ci_is_seeded_and_reproducible() {
        // Two segments of BLEU statistics, [sys_len, ref_len, correct, correct, total, total].
        let rows: Vec<f64> = vec![
            5.0, 5.0, 4.0, 3.0, 5.0, 4.0,
            4.0, 6.0, 3.0, 2.0, 4.0, 5.0,
        ];
        let c = cfg();
        let a = bootstrap_ci(&rows, 6, &ScoreKind::Bleu(c), 200, 12345);
        let b = bootstrap_ci(&rows, 6, &ScoreKind::Bleu(c), 200, 12345);
        assert_eq!(a, b, "same seed must give the same interval");
        let other = bootstrap_ci(&rows, 6, &ScoreKind::Bleu(c), 200, 999);
        assert_ne!(a, other, "a different seed draws different indices");
        assert!(a.1 >= 0.0 && a.0 > 0.0);
    }

    #[test]
    fn ter_rows_follow_the_empty_reference_rules() {
        assert_eq!(ter_score_np2_f32(&[3.0, 6.0]), 50.0);
        assert_eq!(ter_score_np2_f32(&[3.0, 0.0]), 100.0);
        assert_eq!(ter_score_np2_f32(&[0.0, 0.0]), 0.0);
    }
}

/// One system's result for one metric: the score, and for a paired test the p-value and — when the
/// test produces one — the resampled mean and interval.
#[derive(Clone, Copy, Debug)]
pub struct PairedResult {
    pub score: f64,
    pub p_value: Option<f64>,
    pub mean: Option<f64>,
    pub ci: Option<f64>,
}

/// sacreBLEU's `_compute_p_value`: the count that **strictly exceeds** the real difference, with
/// both `+1` corrections (a system compared with itself has a real difference of zero, and a zero
/// must not be attributed to chance).
pub fn p_value(stats: &[f64], real_difference: f64) -> f64 {
    let c = stats.iter().filter(|&&s| s > real_difference).count();
    (c as f64 + 1.0) / (stats.len() as f64 + 1.0)
}

/// numpy's `estimate_ci`: the pairwise mean and half the 1/40..39/40 spread. The paired report
/// prints the mean as a full float, so it has to be numpy's summation, not a different one.
pub fn estimate_ci_np(scores: &mut [f64]) -> (f64, f64) {
    sort_f64(scores);
    let n = scores.len();
    let li = n / 40;
    let ui = n - li - 1;
    (mean_f64(scores), 0.5 * (scores[ui] - scores[li]))
}

/// Score one `int64` statistics row (the paired AR test sums exact integers).
pub fn score_row_i64(kind: &ScoreKind, row: &[i64]) -> f64 {
    match kind {
        ScoreKind::Bleu(cfg) => {
            let as_f64: Vec<f64> = row.iter().map(|&v| v as f64).collect();
            compute_bleu(&as_f64, &cfg).score
        }
        ScoreKind::Chrf { order, beta, eps_smoothing } => {
            chrf_f_score(row, *order, *beta, *eps_smoothing)
        }
        ScoreKind::Ter => {
            let as_f32: Vec<f32> = row.iter().map(|&v| v as f32).collect();
            ter_score_np2_f32(&as_f32)
        }
    }
}

/// sacreBLEU's `_paired_bs_test` for one system and one metric: `(p_value, mean, ci)`. Both the
/// baseline and the system are resampled with the same seeded draw.
pub fn paired_bs(
    baseline_rows: &[f64],
    system_rows: &[f64],
    width: usize,
    kind: &ScoreKind,
    n_samples: usize,
    seed: u128,
    diff: f64,
) -> (f64, f64, f64) {
    let n = system_rows.len() / width;
    let bl32: Vec<f32> = baseline_rows.iter().map(|&v| v as f32).collect();
    let sys32: Vec<f32> = system_rows.iter().map(|&v| v as f32).collect();
    let bl_sums = bootstrap_sums_f32_seeded(&bl32, width, n, n_samples, seed);
    let sys_sums = bootstrap_sums_f32_seeded(&sys32, width, n, n_samples, seed);
    let bl_scores: Vec<f64> = bl_sums.chunks(width).map(|r| score_row(kind, r)).collect();
    let sys_scores: Vec<f64> = sys_sums.chunks(width).map(|r| score_row(kind, r)).collect();
    // The differences pair up resample `k` of the system with resample `k` of the baseline, so they
    // must be taken before `estimate_ci_np` sorts its input.
    let diffs: Vec<f64> = sys_scores
        .iter()
        .zip(bl_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let mut sorted_sys = sys_scores.clone();
    let (mean, ci) = estimate_ci_np(&mut sorted_sys);
    let m = mean_f64(&diffs);
    let centred: Vec<f64> = diffs.iter().map(|d| d - m).collect();
    (p_value(&centred, diff), mean, ci)
}

/// `score_row` for a `float64` row: the metrics whose statistics are not integral (TER) take the
/// same float path sacreBLEU falls back to when `_ar_shuffle` refuses the stats.
pub fn score_row_f64(kind: &ScoreKind, row: &[f64]) -> f64 {
    match kind {
        ScoreKind::Ter => {
            let total_edits = row[0];
            let sum_ref_lengths = row[1];
            let score = if sum_ref_lengths > 0.0 {
                total_edits / sum_ref_lengths
            } else if total_edits > 0.0 {
                1.0
            } else {
                0.0
            };
            100.0 * score
        }
        _ => {
            let as_f32: Vec<f32> = row.iter().map(|&v| v as f32).collect();
            score_row(kind, &as_f32)
        }
    }
}

/// The approximate-randomisation test for metrics whose statistics are not integral: sacreBLEU's
/// `pos_sel @ bl + neg_sel @ sys` and its mirror, summed in the same order.
pub fn paired_ar_f64(
    baseline: &[f64],
    system: &[f64],
    width: usize,
    kind: &ScoreKind,
    n_samples: usize,
    seed: u128,
    diff: f64,
) -> f64 {
    let n = system.len() / width;
    let mut rng = default_rng(seed);
    let mask = rng.integers_bool(n_samples, n);
    let mut row_a = vec![0f64; width];
    let mut row_b = vec![0f64; width];
    let mut stats: Vec<f64> = Vec::with_capacity(n_samples);
    for t in 0..n_samples {
        row_a.iter_mut().for_each(|v| *v = 0.0);
        row_b.iter_mut().for_each(|v| *v = 0.0);
        for s in 0..n {
            let a_first = mask[t * n + s];
            for c in 0..width {
                let bl = baseline[s * width + c];
                let sy = system[s * width + c];
                if a_first {
                    row_a[c] += bl;
                    row_b[c] += sy;
                } else {
                    row_a[c] += sy;
                    row_b[c] += bl;
                }
            }
        }
        stats.push((score_row_f64(kind, &row_a) - score_row_f64(kind, &row_b)).abs());
    }
    p_value(&stats, diff)
}

/// sacreBLEU's `_paired_ar_test` for one system and one metric: `(p_value, mean, ci)`. The mean and
/// interval are `None` unless `n_ar_confidence > 0`, and then they come from a further bootstrap
/// drawn from the same generator, as sacreBLEU does.
pub fn paired_ar(
    baseline_i64: &[i64],
    system_i64: &[i64],
    width: usize,
    kind: &ScoreKind,
    n_samples: usize,
    n_ar_confidence: usize,
    seed: u128,
    diff: f64,
) -> (f64, Option<f64>, Option<f64>) {
    let n = system_i64.len() / width;
    let mut rng = default_rng(seed);
    let mask = rng.integers_bool(n_samples, n);
    let sel: Vec<u8> = mask.iter().map(|&b| b as u8).collect();
    let (a, b) = ar_sums_i64(baseline_i64, system_i64, width, &sel, n);
    let sa: Vec<f64> = a.chunks(width).map(|r| score_row_i64(kind, r)).collect();
    let sb: Vec<f64> = b.chunks(width).map(|r| score_row_i64(kind, r)).collect();
    let stats: Vec<f64> = sa.iter().zip(sb.iter()).map(|(x, y)| (x - y).abs()).collect();
    let p = p_value(&stats, diff);
    if n_ar_confidence == 0 {
        return (p, None, None);
    }
    let idxs = rng.choice(n, n_ar_confidence, n);
    let idxs_i64: Vec<i64> = idxs.iter().map(|&i| i as i64).collect();
    let sys32: Vec<f32> = system_i64.iter().map(|&v| v as f32).collect();
    let sums = bootstrap_sums_f32(&sys32, width, &idxs_i64, n);
    let mut scores: Vec<f64> = sums.chunks(width).map(|r| score_row(kind, r)).collect();
    let (mean, ci) = estimate_ci_np(&mut scores);
    (p, Some(mean), Some(ci))
}
