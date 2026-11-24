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

/// The digit test the contextual `zh` rules use: ASCII and full-width digits.
#[inline]
fn is_zh_digit(c: char) -> bool {
    c.is_ascii_digit() || ('\u{FF10}'..='\u{FF19}').contains(&c)
}

/// `zh`'s second rule, measured separately from the range table: a `-` splits when the
/// character before it is a digit ("1-2" -> "1 - 2", "1-" -> "1 -"), but not otherwise
/// ("a-b", "a-1", "-1" stay whole).
#[inline]
fn hyphen_splits(prev: Option<char>) -> bool {
    prev.is_some_and(is_zh_digit)
}

/// The `.`/`,` passes, measured 2026-11-24 (docs/spec/tokenizers.md section 5). The oracle spaces
/// a `.`/`,` that does not follow a digit, then one that does not precede a digit, both consuming
/// the pair; that is the `intl` rule with `[.,]` as the whole mark set. The range table must not
/// decide `.` or `,`.
fn dot_passes(s: &str) -> String {
    let a: Vec<char> = s.chars().collect();
    let n = a.len();
    let is_mark = |c: char| c == '.' || c == ',';
    let mut b: Vec<char> = Vec::with_capacity(n + n / 4 + 8);
    let mut i = 0;
    while i < n {
        if i + 1 < n && !is_zh_digit(a[i]) && is_mark(a[i + 1]) {
            b.push(a[i]);
            b.push(' ');
            b.push(a[i + 1]);
            b.push(' ');
            i += 2;
        } else {
            b.push(a[i]);
            i += 1;
        }
    }
    let c = b;
    let n = c.len();
    let mut d: Vec<char> = Vec::with_capacity(n + n / 4 + 8);
    let mut i = 0;
    while i < n {
        if i + 1 < n && is_mark(c[i]) && !is_zh_digit(c[i + 1]) {
            d.push(' ');
            d.push(c[i]);
            d.push(' ');
            d.push(c[i + 1]);
            i += 2;
        } else {
            d.push(c[i]);
            i += 1;
        }
    }
    d.into_iter().collect()
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
    // The oracle strips leading and trailing Python whitespace before the `.`/`,` passes.
    let s = dot_passes(crate::unicode::py_rstrip(line).trim_start_matches(crate::unicode::is_py_space));
    let mut items: Vec<String> = Vec::new();
    let mut run = String::new();
    let mut prev: Option<char> = None;
    for c in s.chars() {
        // `.` and `,` are in the range table, but their behaviour is contextual and the passes
        // above own them.
        let splits = c != '.' && c != ','
            && (zh_splits(c) || (c == '-' && hyphen_splits(prev)));
        if splits {
            if !run.is_empty() {
                items.push(std::mem::take(&mut run));
            }
            items.push(c.to_string());
        } else {
            run.push(c);
        }
        prev = Some(c);
    }
    if !run.is_empty() {
        items.push(run);
    }
    let joined = items.join(" ");
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
    fn dot_runs_use_the_pair_consuming_passes() {
        // Measured 2026-11-24; see docs/spec/tokenizers.md section 5.
        assert_eq!(t(".0"), ".0");
        assert_eq!(t("..0"), ". . 0");
        assert_eq!(t("...0"), ". . .0");
        assert_eq!(t("0...0"), "0 . . .0");
        assert_eq!(t("a..0"), "a . .0");
        assert_eq!(t("5.,6"), "5 . , 6");
        assert_eq!(t("0."), "0.");
        assert_eq!(t("0. "), "0.");
        assert_eq!(t("1. x"), "1 . x");
        assert_eq!(t("1,000"), "1,000");
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