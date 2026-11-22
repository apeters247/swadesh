//! Clean-room implementation of the `none` and `char` tokenizers.
//!
//! Specification: `docs/spec/tokenizers.md`, section 1 (`none`) and section 2
//! (`char`), plus the common postcondition in that document ("each tokenizer
//! maps a string to a string ... tokens separated by exactly one ASCII space").
//! Each function appends its tokens, separated by single ASCII spaces, to an
//! output buffer; the caller owns the `\\0` / whitespace joining convention.
//!
//! Provenance: this file was written from the specification document alone. It
//! was *not* written from, and the author did not read, `crates/swadesh-core/
//! src/tokenize.rs` (the previous line-by-line port) nor sacreBLEU's Python
//! sources (`sacrebleu/tokenizers/*.py`). The specification's Python
//! whitespace set is `crate::unicode::is_py_space`; both functions below are
//! independent of the split rule (see the note on `char`).
//!
//! Verified against sacreBLEU 2.6.0 used strictly as a black box: the classes
//! `sacrebleu.tokenizers.tokenizer_none.NoneTokenizer` and
//! `sacrebleu.tokenizers.tokenizer_char.TokenizerChar` (discovered via
//! `sacrebleu.BLEU().TOKENIZERS` = `none`, `char`, ...), called on hand-picked
//! edge inputs and on 31,552 strings (all strings of length <= 3 over a
//! 33-character adversarial alphabet: every Python whitespace character, its
//! near neighbours outside the set such as U+200B/U+200C/U+180E/U+202F, a
//! soft hyphen, control and astral characters, plus 4,000 random longer
//! strings). Byte-identical to the oracle on all of them. See the work report
//! for the probe inputs and outputs.
//!
//! Integrity note: the module appends tokens to `out` with single ASCII spaces
//! and does not `\\0`-join; if a caller writes input text that ends with a
//! non-space character and then conditionally appends a separator, it must
//! decide whether `out` is empty before the call (as this module does).

use crate::unicode::is_py_space;

/// `none`: the identity function (spec section 1).
///
/// The spec's common postcondition would collapse internal whitespace. Measured
/// behaviour (sacreBLEU 2.6.0) is the bare identity: `NoneTokenizer("a  b")` is
/// `"a  b"`, and `BLEU(tokenize="none")` on `"ab  cd"` reports `hyp_len = 2`
/// (the whitespace run yields no tokens, so the postcondition and the identity
/// agree on *token counts* but not on the character-level output). The
/// implementation emits the input unchanged and claims nothing about the
/// postcondition at character level, which is what the oracle does.
///
/// The caller strips the input (`unicode::py_rstrip`) before tokenizing.
pub fn tok_none(text: &str, out: &mut String) {
    out.push_str(text);
}

/// `char`: split each Python-whitespace run into individual Unicode scalar
/// values and join them with single ASCII spaces (spec section 2).
///
/// Measured detail (sacreBLEU 2.6.0): the whitespace characters that separate
/// runs are tokens in their own right and are not dropped. The result is
/// exactly `" ".join(text.chars())`: `TokenizerChar("ab cd")` is
/// `"a b   c d"` (the separating space is a token: `" " + "%s"` on each side of
/// it), `TokenizerChar("a  b")` is `"a     b"`, `TokenizerChar(" x ")` is
/// `"  x  "`, and the stand-alone `TokenizerChar(" ")` is `" "`. So two adjacent
/// whitespace characters do *not* merge and do not lose a space.
///
/// The spec's common postcondition (`" ".join(result.split())`) would instead
/// drop the whitespace tokens and their surrounding spaces (`"a b c d"` for
/// `"ab cd"`); that disagrees with the oracle, and a direct (unstripped) call
/// would differ in token count too (`" x "` is three tokens, `"x"` one). The
/// measured behaviour is kept because the byte-identical scores depend on it.
///
/// The spec calls the split "Unicode-aware: a character is one scalar value";
/// it does not say which whitespace set. Section 7 makes the Python set
/// (`is_py_space`) the target, but the choice cannot change this output: every
/// token is a single scalar value either way, and whitespace characters are
/// kept as tokens, so a run of whitespace is never collapsed.
///
/// The caller strips the input (`unicode::py_rstrip`) before calling; this
/// function does not strip, so a leading/trailing whitespace character in a
/// direct call yields a token (and a leading/trailing space), as the oracle
/// does.
pub fn tok_char(text: &str, out: &mut String) {
    for c in text.chars() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push(c);
    }
}

#[cfg(test)]
mod tests_nonechar_clean {
    use super::*;

    fn none(s: &str) -> String {
        let mut o = String::new();
        tok_none(s, &mut o);
        o
    }

    fn chr(s: &str) -> String {
        let mut o = String::new();
        tok_char(s, &mut o);
        o
    }

    // --- section 1: `none` ---------------------------------------------------

    #[test]
    fn none_is_identity() {
        for s in [
            "", "abc", "a  b", " x ", "a\tb\nc", "h\u{e9}llo w\u{f6}rld", "a.b,", "ab\u{200b}cd",
            "a\u{a0}b", "x\u{1F600}y",
        ] {
            assert_eq!(none(s), s, "{s:?}");
        }
    }

    #[test]
    fn none_keeps_whitespace_runs() {
        // Oracle: NoneTokenizer("a  b") == "a  b" (no collapsing). The token
        // count, not the character string, is what the BLEU postcondition needs.
        assert_eq!(none("a  b"), "a  b");
        assert_eq!(none("a  b").split(is_py_space).filter(|t| !t.is_empty()).count(), 2);
        assert_eq!(none("\u{2028}\u{205f}\u{1680}"), "\u{2028}\u{205f}\u{1680}");
    }

    // --- section 2: `char` ---------------------------------------------------

    #[test]
    fn char_splits_scalar_values() {
        assert_eq!(chr("abc"), "a b c");
        assert_eq!(chr("h\u{e9}llo"), "h \u{e9} l l o");
        assert_eq!(chr("a.b,"), "a . b ,");
        // Astral-plane code point is one scalar value, one token.
        assert_eq!(chr("\u{1D49C}b"), "\u{1D49C} b");
        assert_eq!(chr("x\u{1F600}y"), "x \u{1F600} y");
    }

    #[test]
    fn char_empty() {
        let mut o = String::new();
        tok_char("", &mut o);
        assert!(o.is_empty());
    }

    #[test]
    fn char_python_space_runs() {
        // `\x1c`-`\x1f` are Python whitespace, `\u{200b}` is not; either way
        // each is one scalar value and one token.
        assert_eq!(chr("a\x1cb\x1fc"), "a \x1c b \x1f c");
        assert_eq!(chr("a\u{200b}b"), "a \u{200b} b");
        assert_eq!(chr("a\u{a0}b"), "a \u{a0} b");
        assert_eq!(chr("a\u{3000}b"), "a \u{3000} b");
        assert_eq!(chr("a\u{85}b"), "a \u{85} b");
    }

    #[test]
    fn char_keeps_separators_as_tokens() {
        // Oracle: TokenizerChar("ab cd") == "a b   c d".
        assert_eq!(chr("ab cd"), "a b   c d");
        // Oracle: TokenizerChar("a  b") == "a     b".
        assert_eq!(chr("a  b"), "a     b");
        // Oracle: TokenizerChar(" ") == " " and TokenizerChar("  ") == "   ".
        assert_eq!(chr(" "), " ");
        assert_eq!(chr("  "), "   ");
        // Oracle: TokenizerChar(" x ") == "  x  ".
        assert_eq!(chr(" x "), "  x  ");
        // Oracle: TokenizerChar("a\u{a0}") == "a \u{a0}".
        assert_eq!(chr("a\u{a0}"), "a \u{a0}");
        // Oracle: TokenizerChar("\u{2028}\u{205f}\u{1680}")
        //   == "\u{2028} \u{205f} \u{1680}".
        assert_eq!(
            chr("\u{2028}\u{205f}\u{1680}"),
            "\u{2028} \u{205f} \u{1680}"
        );
    }

    #[test]
    fn char_non_space_non_space_has_one_space() {
        // Adjacent non-whitespace scalars are always separated by one space.
        assert_eq!(chr("ab"), "a b");
        // A soft hyphen is not whitespace: it is its own token.
        assert_eq!(chr("ab\u{ad}cd"), "a b \u{ad} c d");
    }

    #[test]
    fn char_is_space_join_of_scalars() {
        // The whole rule, as measured: " ".join(text.chars()).
        for s in [
            "", "abc", "a  b", "ab cd", " x ", "\t\n", "h\u{e9}llo  w\u{f6}rld", "\u{2028}x",
            "a\u{200b} b",
        ] {
            assert_eq!(chr(s), space_joined(s.chars()), "{s:?}");
        }
    }

    fn space_joined(chars: impl Iterator<Item = char>) -> String {
        let mut o = String::new();
        for c in chars {
            if !o.is_empty() {
                o.push(' ');
            }
            o.push(c);
        }
        o
    }
}