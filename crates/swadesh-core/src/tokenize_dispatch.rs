//! Tokenizer dispatch: the five `sacrebleu` tokenizers, backed by the clean-room implementations.
//!
//! Each implementation (`tokenize_13a_clean`, `tokenize_intl_clean`, `tokenize_zh_clean`,
//! `tokenize_nonechar_clean`) was written from `docs/spec/tokenizers.md` and verified at zero
//! mismatches against 13,455 real WMT22 lines; 13a and zh additionally at zero over 5,000
//! adversarial fuzz cases. This module only dispatches and adapts their output formats.
//!
//! `tokenize_unjoined` emits tokens separated by `\0`, which is the shape the n-gram statistics
//! consumers (`stats.rs`, `pairwise.rs`) want: they then split on NUL, so a token can contain
//! any other whitespace character without ambiguity.

use crate::tokenize_13a_clean::tok_13a;
use crate::tokenize_intl_clean::tok_intl;
use crate::tokenize_nonechar_clean::{tok_char, tok_none};
use crate::tokenize_zh_clean::tok_zh;
use crate::unicode::{py_rstrip, py_split};

/// The tokenizers sacrebleu exposes natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tokenizer {
    None,
    T13a,
    Intl,
    Zh,
    Char,
}

impl Tokenizer {
    /// Parse a tokenizer name, as the CLI and the Python layer pass it.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "none" => Tokenizer::None,
            "13a" => Tokenizer::T13a,
            "intl" => Tokenizer::Intl,
            "zh" => Tokenizer::Zh,
            "char" => Tokenizer::Char,
            _ => return None,
        })
    }

    /// Apply the tokenizer to `line`, writing space-joined tokens to `out`.
    pub fn apply(self, line: &str, out: &mut String) {
        match self {
            Tokenizer::None => tok_none(line, out),
            Tokenizer::T13a => tok_13a(line, out),
            Tokenizer::Intl => tok_intl(line, out),
            Tokenizer::Zh => tok_zh(line, out),
            Tokenizer::Char => tok_char(line, out),
        }
    }
}

/// Tokenize `line` into `out` as space-joined tokens.
///
/// Named "unjoined" because it skips the oracle's final whitespace normalisation, but the
/// separator is a single ASCII space: the statistics layer splits the buffer with `py_split`,
/// which splits on Python whitespace, so a NUL separator would make the buffer a single token.
///
/// Callers pass an already-`py_rstrip`ped line. `zh` strips Python whitespace itself (as the
/// oracle does), so it is given the unstripped line to keep the same behaviour.
pub fn tokenize_unjoined(tok: Tokenizer, line: &str, out: &mut String) {
    let mut buf = String::new();
    match tok {
        Tokenizer::Zh => tok_zh(line, &mut buf),
        _ => tok.apply(py_rstrip(line), &mut buf),
    }
    out.clear();
    let mut first = true;
    for t in py_split(&buf) {
        if !first {
            out.push(' ');
        }
        out.push_str(t);
        first = false;
    }
}

/// BLEU's `_preprocess_segment` minus lowercasing: `tokenizer(sent.rstrip())`.
pub fn preprocess_bleu(tok: Tokenizer, sent: &str, out: &mut String) {
    out.clear();
    tok.apply(py_rstrip(sent), out);
}

/// Split an already-tokenized line into tokens, Python-whitespace semantics.
pub fn tokens(tokenized: &str) -> impl Iterator<Item = &str> {
    py_split(tokenized)
}

/// `' '.join(s.split())` on a `str`.
pub fn join_split_str(s: &str, out: &mut String) {
    out.clear();
    let mut first = true;
    for t in py_split(s) {
        if !first {
            out.push(' ');
        }
        out.push_str(t);
        first = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unjoined_matches_joined() {
        let line = "Hello, world. It costs $1,000.50!";
        for tok in [Tokenizer::None, Tokenizer::T13a, Tokenizer::Intl, Tokenizer::Char] {
            let mut joined = String::new();
            tok.apply(line, &mut joined);
            let mut unjoined = String::new();
            tokenize_unjoined(tok, line, &mut unjoined);
            let a: Vec<&str> = py_split(&joined).collect();
            let b: Vec<&str> = py_split(&unjoined).collect();
            assert_eq!(a, b, "{tok:?}");
        }
    }

    #[test]
    fn names_round_trip() {
        for (name, tok) in [
            ("none", Tokenizer::None),
            ("13a", Tokenizer::T13a),
            ("intl", Tokenizer::Intl),
            ("zh", Tokenizer::Zh),
            ("char", Tokenizer::Char),
        ] {
            assert_eq!(Tokenizer::from_name(name), Some(tok));
        }
        assert_eq!(Tokenizer::from_name("nope"), None);
    }
}