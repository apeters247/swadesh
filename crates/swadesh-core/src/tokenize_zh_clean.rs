//! Tokenizer `zh`: clean-room implementation from the derived specification.
//!
//! Specification: `docs/spec/tokenizers.md` section 5, refined by exhaustive black-box measurement.
//! No implementation source was read.
//!
//! Behaviour, as measured over all 1,112,032 Unicode scalars (a code point "splits" iff
//! `"a<c>b"` becomes `["a", c, "b"]`): the 23 ranges in `ZH_RANGES` split off as single-character
//! tokens; everything else accumulates into runs. The set includes the historical range-table
//! defect on purpose -- U+2060..U+2A6D (arrows, maths, currency) is admitted as "Chinese" because
//! the original table's end point was a two-character string. Reproducing it is required for
//! identical scores.
//!
//! There is no entity decoding here, unlike `13a`: `"a&amp;b"` yields `a & amp ; b`, and
//! `"<skipped>"` is kept.

use crate::unicode::py_split;

/// Measured splitting ranges, inclusive. Sorted, non-overlapping.
pub static ZH_RANGES: [(u32, u32); 23] = [
    (0x21, 0x26),
    (0x28, 0x2C),
    (0x2E, 0x2F),
    (0x3A, 0x40),
    (0x5B, 0x60),
    (0x7B, 0x7E),
    (0x200B, 0x2027),
    (0x202A, 0x202E),
    (0x2030, 0x205E),
    (0x2060, 0x2A6D),
    (0x2E80, 0x2FDF),
    (0x2FF0, 0x2FFF),
    (0x3001, 0x303F),
    (0x3100, 0x312F),
    (0x31A0, 0x31EF),
    (0x3200, 0x4DB5),
    (0x4E00, 0x9FBB),
    (0xF900, 0xFA2D),
    (0xFA30, 0xFA6A),
    (0xFA70, 0xFAD9),
    (0xFE10, 0xFE1F),
    (0xFE30, 0xFE4F),
    (0xFF00, 0xFFEF),
];

/// `zh`'s second rule, measured separately from the range table: a `-` splits when the
/// character before it is a digit ("1-2" -> "1 - 2", "1-" -> "1 -"), but not otherwise
/// ("a-b", "a-1", "-1" stay whole).
#[inline]
fn hyphen_splits(prev: Option<char>) -> bool {
    prev.is_some_and(|p| p.is_ascii_digit() || ('\u{FF10}'..='\u{FF19}').contains(&p))
}

/// Measured rule for `.` and `,` (both are in the range table, but the table must not decide
/// them - their behaviour is contextual). They attach ONLY when:
///   * both neighbours are digits           "5.5", "1,000", "3.126"
///   * the previous character is a digit and the punctuation ends the word   "9.", "9,"
/// Everything else splits, including a digit followed by a space:
///   "1. x" -> "1 . x"     "10, 15" -> "10 , 15"     "a,b" -> "a , b"
#[inline]
fn dot_comma_splits(at_line_start: bool, prev: Option<char>, next: Option<char>) -> bool {
    let dig = |c: Option<char>| c.is_some_and(|x| x.is_ascii_digit());
    match (prev, next) {
        (Some(p), Some(n)) => !(dig(Some(p)) && dig(Some(n))),
        (Some(p), None) => !dig(Some(p)),
        // Nothing before it in a word that BEGINS the line: ".9" attaches, and so does ".9x".
        (None, Some(n)) if at_line_start => !dig(Some(n)),
        _ => true,
    }
}

/// True when the code point is one of the measured splitting characters.
#[inline]
pub fn zh_splits(c: char) -> bool {
    let cp = c as u32;
    // Binary search over the sorted range table.
    let mut lo = 0usize;
    let mut hi = ZH_RANGES.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (a, b) = ZH_RANGES[mid];
        if cp < a {
            hi = mid;
        } else if cp > b {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

/// Tokenize `line` into `out`, tokens separated by single ASCII spaces.
pub fn tok_zh(line: &str, out: &mut String) {
    out.clear();
    let mut first = true;
    let mut push = |tok: &str, out: &mut String| {
        if tok.is_empty() {
            return;
        }
        if !first {
            out.push(' ');
        }
        out.push_str(tok);
        first = false;
    };

    // A trailing "."/"," is attached or split according to what follows it in the ORIGINAL
    // line, not the whitespace-split word: "1." at end of line stays, "1. x" splits. So the
    // word boundaries are taken from the line, and the character after a word's last char is
    // the separating space when there is more line left.
    let words: Vec<&str> = py_split(line).collect();
    let mut run = String::new();
    for (wi, tok) in words.iter().enumerate() {
        let more_line = wi + 1 < words.len();
        let chars: Vec<char> = tok.chars().collect();
        let mut prev: Option<char> = None;
        for (idx, &c) in chars.iter().enumerate() {
            let next = match chars.get(idx + 1).copied() {
                Some(n) => Some(n),
                // End of this word: a space if the line continues, otherwise end of line.
                None => Some(' ').filter(|_| more_line),
            };
            // `.` and `,` are in the measured range table, but their behaviour is contextual,
            // so the table must not decide them: the attachment rule is authoritative.
            let splits = if c == '.' || c == ',' {
                dot_comma_splits(wi == 0 && idx == 0, prev, next)
            } else {
                zh_splits(c) || (c == '-' && hyphen_splits(prev))
            };
            if splits {
                push(&run, out);
                run.clear();
                let mut buf = [0u8; 4];
                push(c.encode_utf8(&mut buf), out);
            } else {
                run.push(c);
            }
            prev = Some(c);
        }
        push(&run, out);
        run.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> String {
        let mut o = String::new();
        tok_zh(s, &mut o);
        o
    }

    #[test]
    fn cjk_splits_latin_runs() {
        assert_eq!(t("\u{4f60}\u{597d}abc\u{4e16}\u{754c}"), "\u{4f60} \u{597d} abc \u{4e16} \u{754c}");
        assert_eq!(t("caf\u{e9} 12"), "caf\u{e9} 12");
        assert_eq!(t("a-b"), "a-b");
    }

    #[test]
    fn no_entity_decoding() {
        assert_eq!(t("a&amp;b"), "a & amp ; b");
        assert_eq!(t("<skipped>"), "< skipped >");
    }

    #[test]
    fn punctuation_and_full_width() {
        assert_eq!(t("5.,6"), "5 . , 6");
        assert_eq!(t("\u{ff11}\u{ff0c}\u{ff12}"), "\u{ff11} \u{ff0c} \u{ff12}");
        assert_eq!(t("a\u{2192}b"), "a \u{2192} b");
    }
}