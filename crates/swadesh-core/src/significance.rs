//! sacreBLEU's bootstrap confidence interval: draw resample indices, score each resampled row,
//! then take the mean and the 1/40..39/40 half-spread (`estimate_ci`).
//!
//! The index drawing and the row scoring are the emulations already in `resample` and `score`;
//! this module is the orchestration `sacrebleu.significance._bootstrap_resample` does in Python,
//! written here so both the CLI and the Python binding can call one implementation.

use crate::resample::{bootstrap_sums_f32_seeded, estimate_ci};
use crate::score::{chrf_f_score_np2_f32, compute_bleu_np2_f32, BleuScoreConfig, NpVal};

/// Which metric a statistics row belongs to.
#[derive(Clone, Copy, Debug)]
pub enum ScoreKind<'a> {
    /// Rows are `[sys_len, ref_len, correct.., total..]`, `2 + 2 * max_order` wide.
    Bleu(&'a BleuScoreConfig),
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
pub fn score_row(kind: &ScoreKind<'_>, row: &[f32]) -> f64 {
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
    kind: &ScoreKind<'_>,
    n_samples: usize,
    seed: u128,
) -> (f64, f64) {
    let n = rows.len() / width;
    let as_f32: Vec<f32> = rows.iter().map(|&v| v as f32).collect();
    let sums = bootstrap_sums_f32_seeded(&as_f32, width, n, n_samples, seed);
    let mut scores: Vec<f64> = sums.chunks(width).map(|r| score_row(kind, r)).collect();
    estimate_ci(&mut scores)
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
        let a = bootstrap_ci(&rows, 6, &ScoreKind::Bleu(&c), 200, 12345);
        let b = bootstrap_ci(&rows, 6, &ScoreKind::Bleu(&c), 200, 12345);
        assert_eq!(a, b, "same seed must give the same interval");
        let other = bootstrap_ci(&rows, 6, &ScoreKind::Bleu(&c), 200, 999);
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
