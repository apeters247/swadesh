//! Regression tests for crashes found by the `fuzz/` cargo-fuzz targets.

use swadesh_core::score::chrf_f_score;
use swadesh_core::stats::{chrf_segment, ChrfConfig};

/// fuzz target `chrf_segment` (and `pairwise`): `beta * beta` overflowed i64
/// for |beta| > 3.0e9 and panicked under overflow checks.
#[test]
fn chrf_huge_beta_does_not_overflow() {
    let stats = [3i64, 4, 2];
    for beta in [3_037_000_500i64, 4_000_000_000, i64::MAX, i64::MIN] {
        let s = chrf_f_score(&stats, 1, beta, false);
        assert!(s.is_finite() && (0.0..=100.0).contains(&s), "beta={beta}: {s}");
    }
    let cfg = ChrfConfig { char_order: 2, word_order: 0, beta: i64::MIN, whitespace: false, eps_smoothing: false };
    let mut out = Vec::new();
    chrf_segment(&cfg, "abc", &["abd"], &mut out);
    assert_eq!(out.len(), 6);
}

/// Small betas are unchanged: beta = 2 gives the usual chrF2 factor 4.
#[test]
fn chrf_small_beta_unchanged() {
    let stats = [3i64, 4, 2];
    let want = {
        let (p, r) = (2.0f64 / 3.0, 2.0f64 / 4.0);
        100.0 * (1.0 + 4.0) * p * r / (4.0 * p + r)
    };
    assert_eq!(chrf_f_score(&stats, 1, 2, false).to_bits(), want.to_bits());
}
