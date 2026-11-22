//! swadesh-core: a bit-exact, parallel reimplementation of sacreBLEU's
//! BLEU / chrF / chrF++ sufficient statistics and scoring.

pub mod ngram;
pub mod par;
pub mod pairwise;
pub mod resample;
pub mod nprng;
pub mod score;
pub mod stats;
pub mod table;
pub mod ter;
pub mod tables;
pub mod tokenize;
pub mod tokenize_nonechar_clean;
pub mod tokenize_13a_clean;
pub mod tokenize_zh_clean;
pub mod tokenize_intl_clean;
pub mod unicode;


pub use score::{chrf_f_score, compute_bleu, BleuScoreConfig, Smooth, SumMode};
pub use stats::{bleu_segment, chrf_segment, BleuConfig, ChrfConfig};
pub use tokenize::Tokenizer;
pub use ter::{ter_segment, TerConfig, TerError};

/// Segments below this size are processed serially (thread overhead dominates).
const PAR_THRESHOLD: usize = 2048;

/// Compute a flattened `(n_segments x width)` statistics matrix in parallel.
/// `refs[i]` holds the (non-missing) references of segment `i`.
pub fn corpus_stats<F>(hyps: &[&str], refs: &[Vec<&str>], width: usize, f: F) -> Vec<i64>
where
    F: Fn(&str, &[&str], &mut Vec<i64>) + Sync,
{
    let n = hyps.len();
    let mut out = vec![0i64; n * width];
    let run = |(i, row): (usize, &mut [i64])| {
        let mut buf = Vec::with_capacity(width);
        f(hyps[i], &refs[i], &mut buf);
        if buf.len() == width {
            row.copy_from_slice(&buf);
        }
    };
    if n >= PAR_THRESHOLD {
        par::for_each_chunk(&mut out, width.max(1), run);
    } else {
        out.chunks_mut(width.max(1)).enumerate().for_each(run);
    }
    out
}

pub fn bleu_corpus_stats(cfg: &BleuConfig, hyps: &[&str], refs: &[Vec<&str>]) -> Vec<i64> {
    let width = 2 + 2 * cfg.max_order;
    corpus_stats(hyps, refs, width, |h, r, o| bleu_segment(cfg, h, r, o))
}

pub fn chrf_corpus_stats(cfg: &ChrfConfig, hyps: &[&str], refs: &[Vec<&str>]) -> Vec<i64> {
    let width = 3 * (cfg.char_order + cfg.word_order);
    corpus_stats(hyps, refs, width, |h, r, o| chrf_segment(cfg, h, r, o))
}

/// Segment-level TER statistics `[num_edits, avg_ref_len]` per segment,
/// flattened row-major (width 2). `refs[i]` holds the non-missing references of
/// segment `i`. Errors if any segment has no references.
pub fn ter_corpus_stats(cfg: &TerConfig, hyps: &[&str], refs: &[Vec<&str>]) -> Result<Vec<f64>, TerError> {
    #[cfg(feature = "parallel")]
    let rows: Vec<[f64; 2]> = {
        use rayon::prelude::*;
        par::install(|| {
            (0..hyps.len())
                .into_par_iter()
                .map(|i| ter_segment(cfg, hyps[i], &refs[i]))
                .collect::<Result<_, _>>()
        })?
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<[f64; 2]> = (0..hyps.len())
        .map(|i| ter_segment(cfg, hyps[i], &refs[i]))
        .collect::<Result<_, _>>()?;
    Ok(rows.into_iter().flatten().collect())
}
