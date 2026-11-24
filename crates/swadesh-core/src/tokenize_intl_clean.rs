//! Tokenizer `intl`: clean-room implementation of the measured model.
//!
//! Specification: `docs/spec/intl-absorption.md` (measured 2026-11-24) and
//! `docs/spec/tokenizers.md` section 4. Derived by probing the oracle as a black box; neither
//! sacreBLEU's sources nor the earlier `tokenize.rs` port were read.
//!
//! The model is three passes over the segment:
//!
//!   0. drop trailing Python whitespace;
//!   1. pad every Unicode symbol (`\p{S}`) with a space on both sides;
//!   2. split between a non-number and a following `\p{P}` character, consuming both;
//!   3. split between a `\p{P}` character and a following non-number, consuming both;
//!
//! then split on Python whitespace. `\p{P}` includes `_` (category Pc). Punctuation glues to
//! numbers (`3%4`, `3.5%`, `_1`) and symbols never do (`$5`, `\u{a5}6F`, `\u{2192}`); punctuation is
//! separated from every non-number neighbour (`a%1`). The non-overlapping scan is why `\u{2018}\u{2019}/1`
//! keeps `/1` glued while `./1` splits. No entity decoding happens (`a&amp;b` -> `a & amp ; b`).

use crate::unicode::{is_number, is_punct, is_py_space, is_symbol, py_split};

/// Tokenize `line` into `out`, tokens separated by single ASCII spaces.
pub fn tok_intl(line: &str, out: &mut String) {
    // Pass 0: trailing Python whitespace is dropped, so "7. " glues like "7.".
    let trimmed = line.trim_end_matches(is_py_space);

    // Pass 1: symbols are always their own token.
    let mut padded: Vec<char> = Vec::with_capacity(trimmed.len() + 8);
    for c in trimmed.chars() {
        if is_symbol(c) {
            padded.push(' ');
            padded.push(c);
            padded.push(' ');
        } else {
            padded.push(c);
        }
    }

    // Pass 2: a non-number followed by punctuation.
    let mut left: Vec<char> = Vec::with_capacity(padded.len() + 8);
    let mut i = 0;
    while i < padded.len() {
        if i + 1 < padded.len() && !is_number(padded[i]) && is_punct(padded[i + 1]) {
            left.push(padded[i]);
            left.push(' ');
            left.push(padded[i + 1]);
            left.push(' ');
            i += 2;
        } else {
            left.push(padded[i]);
            i += 1;
        }
    }

    // Pass 3: punctuation followed by a non-number.
    let mut right: Vec<char> = Vec::with_capacity(left.len() + 8);
    let mut i = 0;
    while i < left.len() {
        if i + 1 < left.len() && is_punct(left[i]) && !is_number(left[i + 1]) {
            right.push(' ');
            right.push(left[i]);
            right.push(' ');
            right.push(left[i + 1]);
            i += 2;
        } else {
            right.push(left[i]);
            i += 1;
        }
    }

    let joined: String = right.into_iter().collect();
    out.clear();
    for (k, tok) in py_split(&joined).enumerate() {
        if k > 0 {
            out.push(' ');
        }
        out.push_str(tok);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> String {
        let mut o = String::new();
        tok_intl(s, &mut o);
        o
    }

    #[test]
    fn basic_rules() {
        // Expectations below are the ORACLE's measured output, not from memory.
        assert_eq!(t("Hello, world."), "Hello , world .");
        assert_eq!(t("It costs 5,000.50!"), "It costs 5,000.50!");
        assert_eq!(t("100%"), "100%");
        assert_eq!(t("AT&T"), "AT & T");
    }

    #[test]
    fn no_entity_decoding() {
        assert_eq!(t("a&amp;b"), "a & amp ; b");
    }

    #[test]
    fn non_ascii_punctuation_splits() {
        assert_eq!(t("a\u{2014}b"), "a \u{2014} b");
        assert_eq!(t("\u{4f60}\u{597d}abc"), "\u{4f60}\u{597d}abc");
    }

    #[test]
    fn pair_consumption() {
        // Measured 2026-11-24: two leading quotes pair, so the scan resumes at "/" and "/1" glues.
        assert_eq!(t("\u{2018}\u{2019}/1"), "\u{2018} \u{2019} /1");
        assert_eq!(t("''/1"), "' ' /1");
        // Two leading marks pair with each other, so this one splits.
        assert_eq!(t("./1"), ". / 1");
        assert_eq!(t("a./1"), "a . /1");
        assert_eq!(t("a.2/1"), "a . 2/1");
        assert_eq!(t("2./9"), "2 . / 9");
    }

    #[test]
    fn symbols_never_glue_to_numbers() {
        assert_eq!(t("$5"), "$ 5");
        assert_eq!(t("3%4"), "3%4");
        assert_eq!(t("3.5%"), "3.5%");
        assert_eq!(t("\u{a5}6F"), "\u{a5} 6F");
        assert_eq!(t("\u{2192}5"), "\u{2192} 5");
    }

    #[test]
    fn underscore_is_punctuation() {
        assert_eq!(t("_1"), "_1");
        assert_eq!(t("1_"), "1_");
        assert_eq!(t("a_1"), "a _ 1");
    }

    #[test]
    fn trailing_space_is_dropped_first() {
        assert_eq!(t("7. "), "7.");
        assert_eq!(t("a. "), "a .");
        assert_eq!(t("7. x"), "7 . x");
    }

    #[test]
    fn zero_width_joiner_is_its_own_token() {
        assert_eq!(t("\u{1f468}\u{200d}\u{1f469}"), "\u{1f468} \u{200d} \u{1f469}");
    }
}
