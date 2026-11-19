//! Character-class predicates that mirror the exact semantics of the Python
//! interpreter and `regex` module used by sacreBLEU (see `tables.rs`).

use crate::tables::{PY_WHITESPACE, REGEX_N, REGEX_P, REGEX_S};

#[inline]
fn in_table(table: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    // Binary search over sorted, non-overlapping inclusive ranges.
    let mut lo = 0usize;
    let mut hi = table.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (a, b) = table[mid];
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

/// Python `str.isspace()` as used by `str.split()` / `str.strip()`.
#[inline]
pub fn is_py_space(c: char) -> bool {
    if c.is_ascii() {
        matches!(c, '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c'..='\x1f' | ' ')
    } else {
        in_table(&PY_WHITESPACE, c)
    }
}

/// `regex` module `\p{N}`.
#[inline]
pub fn is_number(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_digit()
    } else {
        in_table(&REGEX_N, c)
    }
}

/// `regex` module `\p{P}`.
#[inline]
pub fn is_punct(c: char) -> bool {
    in_table(&REGEX_P, c)
}

/// `regex` module `\p{S}`.
#[inline]
pub fn is_symbol(c: char) -> bool {
    in_table(&REGEX_S, c)
}

/// Python `str.split()` with no arguments.
#[inline]
pub fn py_split(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_py_space).filter(|t| !t.is_empty())
}

/// Python `str.rstrip()` with no arguments.
#[inline]
pub fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(is_py_space)
}

/// Python `str.strip()` with no arguments.
#[inline]
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spaces() {
        assert!(is_py_space('\u{3000}'));
        assert!(is_py_space('\x1f'));
        assert!(!is_py_space('\u{200b}'));
        assert_eq!(py_split(" a\u{a0}b  c\n").collect::<Vec<_>>(), vec!["a", "b", "c"]);
    }
    #[test]
    fn classes() {
        assert!(is_punct('.'));
        assert!(is_symbol('$'));
        assert!(is_number('٣'));
        assert!(!is_punct('a'));
    }
}
