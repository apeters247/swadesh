//! Python's `str.lower()`, exactly.
//!
//! Rust's `str::to_lowercase` uses a newer Unicode table than CPython 3.11
//! (Unicode 14). The codepoints below are unassigned in CPython's table, so
//! `str.lower()` leaves them unchanged, but Rust maps them. Everywhere else the
//! two agree, including the context-sensitive final sigma. Strings that contain
//! one of these codepoints are lowercased in segments between them. That is
//! exact: CPython treats them as uncased and not case-ignorable, so they cut the
//! final-sigma context the same way the split does.

use std::borrow::Cow;

/// Codepoint ranges that CPython 3.11 leaves unchanged and Rust lowercases.
/// Generated from an exhaustive comparison over all codepoints.
const PY_UNCHANGED: &[(u32, u32)] = &[
    (0x1C89, 0x1C89),
    (0xA7CB, 0xA7CC),
    (0xA7CE, 0xA7CE),
    (0xA7D2, 0xA7D2),
    (0xA7D4, 0xA7D4),
    (0xA7DA, 0xA7DA),
    (0xA7DC, 0xA7DD),
    (0xA7E2, 0xA7E2),
    (0xAB6C, 0xAB6D),
    (0x10D50, 0x10D65),
    (0x16EA0, 0x16EB8),
    (0x1DF40, 0x1DF40),
    (0x1DF48, 0x1DF48),
    (0x1DF4A, 0x1DF4A),
    (0x1DF4D, 0x1DF4D),
    (0x1DF51, 0x1DF51),
    (0x1DF68, 0x1DF68),
    (0x1DF6A, 0x1DF6A),
    (0x1DF6C, 0x1DF6C),
    (0x1DF6E, 0x1DF6E),
    (0x1DF72, 0x1DF72),
    (0x1DF74, 0x1DF74),
    (0x1DF76, 0x1DF76),
    (0x1DF78, 0x1DF78),
    (0x1DF7A, 0x1DF7A),
    (0x1DF7C, 0x1DF7C),
    (0x1DF7E, 0x1DF7E),
];

fn is_py_unchanged(c: char) -> bool {
    let cp = c as u32;
    PY_UNCHANGED.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp))
}

/// `s.lower()` as CPython 3.11 computes it.
pub fn py_lower(s: &str) -> Cow<'_, str> {
    if s.is_ascii() {
        return Cow::Owned(s.to_ascii_lowercase());
    }
    if !s.chars().any(is_py_unchanged) {
        return Cow::Owned(s.to_lowercase());
    }
    let mut out = String::with_capacity(s.len());
    let mut segment_start = 0;
    for (i, c) in s.char_indices() {
        if is_py_unchanged(c) {
            out.push_str(&s[segment_start..i].to_lowercase());
            out.push(c);
            segment_start = i + c.len_utf8();
        }
    }
    out.push_str(&s[segment_start..].to_lowercase());
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython_on_known_cases() {
        assert_eq!(py_lower("ABC"), "abc");
        assert_eq!(py_lower("ΑΣ ΟΔΟΣ"), "ας οδος");
        assert_eq!(py_lower("İ"), "i\u{307}");
        // U+1C89 is unassigned in CPython 3.11: left alone.
        assert_eq!(py_lower("\u{1c89}A"), "\u{1c89}a");
    }
}
