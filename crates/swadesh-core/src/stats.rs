//! Configuration types shared by the BLEU / chrF / TER machines.
//!
//! The statistics themselves live in `bleu_stats_clean` and `chrf_stats_clean`; TER lives in
//! `ter`. This module holds only the plain parameter structs, so that the CLI and the Python
//! bindings have one place to build them.

use crate::tokenize_dispatch::Tokenizer;

/// BLEU configuration: tokenizer plus highest n-gram order.
#[derive(Clone, Copy, Debug)]
pub struct BleuConfig {
    pub tokenizer: Tokenizer,
    pub max_order: usize,
}

/// chrF configuration. `beta` is integral because the oracle restricts it that way.
#[derive(Clone, Copy, Debug)]
pub struct ChrfConfig {
    pub char_order: usize,
    pub word_order: usize,
    pub beta: i64,
    pub whitespace: bool,
    pub eps_smoothing: bool,
}

/// TER parameters, mirroring the oracle's option names.
#[derive(Clone, Copy, Debug, Default)]
pub struct TerConfig {
    /// Split punctuation off words and protect numbers (TER spec 3.2).
    pub normalized: bool,
    /// Delete punctuation instead of keeping it (TER spec 3.1).
    pub no_punct: bool,
    /// Split CJK and fullwidth characters individually (TER spec 3.3).
    pub asian_support: bool,
    /// Keep case.
    pub case_sensitive: bool,
}

/// Reported when a segment has no references, where the oracle raises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerError;
