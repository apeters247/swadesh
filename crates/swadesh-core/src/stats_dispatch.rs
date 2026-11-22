//! One place where callers reach the segment statistics.
//!
//! BLEU and chrF were reimplemented from `docs/spec/stats.md`; the functions here are thin
//! aliases so the CLI and the Python bindings do not depend on module names. TER follows the
//! same shape once Phase 3 lands.

use crate::stats::{BleuConfig, ChrfConfig};

/// BLEU statistics for one segment.
#[inline]
pub fn bleu_stats(cfg: &BleuConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    crate::bleu_stats_clean::bleu_stats_segment(cfg, hyp, refs, out)
}

/// chrF statistics for one segment.
#[inline]
pub fn chrf_stats(cfg: &ChrfConfig, hyp: &str, refs: &[&str], out: &mut Vec<i64>) {
    crate::chrf_stats_clean::chrf_stats_segment(cfg, hyp, refs, out)
}
