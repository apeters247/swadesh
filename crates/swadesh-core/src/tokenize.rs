//! Bit-exact ports of sacreBLEU's built-in tokenizers.
//!
//! Every regex in sacreBLEU's `13a`, `zh` and `intl` tokenizers is either a
//! single-character class substitution or a fixed-length *two*-character
//! pattern applied with `re.sub` (leftmost, non-overlapping, scanning left to
//! right).  Such a substitution is equivalent to a single linear scan that,
//! at every position, tries the two-character match and on success emits the
//! replacement and skips both characters.  That is what [`sub2`] does.

use crate::unicode::{is_number, is_punct, is_py_space, is_symbol, py_rstrip, py_split, py_strip};

/// Tokenizers that are implemented natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tokenizer {
    None,
    T13a,
    Intl,
    Zh,
    Char,
}

impl Tokenizer {
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

    /// Apply the tokenizer to an already rstripped / lowercased line.
    pub fn apply(self, line: &str, out: &mut String) {
        out.clear();
        match self {
            Tokenizer::None => out.push_str(line),
            Tokenizer::T13a => tok_13a(line, out),
            Tokenizer::Intl => tok_intl(line, out),
            Tokenizer::Zh => tok_zh(line, out),
            Tokenizer::Char => tok_char(line, out),
        }
    }
}

/// BLEU's `_preprocess_segment` minus lowercasing: `tokenizer(sent.rstrip())`.
pub fn preprocess_bleu(tok: Tokenizer, sent: &str, out: &mut String) {
    tok.apply(py_rstrip(sent), out);
}

/// Two-character `re.sub` emulation.
#[inline]
fn sub2(
    input: &[char],
    out: &mut Vec<char>,
    m: impl Fn(char, char) -> bool,
    emit: impl Fn(char, char, &mut Vec<char>),
) {
    out.clear();
    let n = input.len();
    let mut i = 0;
    while i < n {
        if i + 1 < n && m(input[i], input[i + 1]) {
            emit(input[i], input[i + 1], out);
            i += 2;
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
}

/// Single-character class `re.sub(r'(X)', r' \1 ')`.
#[inline]
fn sub1(input: &[char], out: &mut Vec<char>, m: impl Fn(char) -> bool) {
    out.clear();
    for &c in input {
        if m(c) {
            out.push(' ');
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
    }
}

/// `' '.join(line.split())`
#[inline]
fn join_split_chars(chars: &[char], out: &mut String) {
    out.clear();
    let mut pending_space = false;
    for &c in chars {
        if is_py_space(c) {
            if !out.is_empty() {
                pending_space = true;
            }
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(c);
        }
    }
}

#[inline]
fn is_13a_class1(c: char) -> bool {
    // [\{-\~\[-\` -\&\(-\+\:-\@\/]
    matches!(c, '{'..='~' | '['..='`' | ' '..='&' | '('..='+' | ':'..='@' | '/')
}

#[inline]
fn is_ascii_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// sacreBLEU `TokenizerRegexp.__call__`.
pub fn tok_re(line: &str, out: &mut String) {
    // ASCII-only fast path identical in semantics (all patterns are ASCII
    // classes except "[^0-9]" which treats any non-digit the same).
    let a: Vec<char> = line.chars().collect();
    tok_re_chars(&a, out);
}

fn tok_re_chars(a: &[char], out: &mut String) {
    let mut b = Vec::with_capacity(a.len() * 2 + 8);
    sub1(a, &mut b, is_13a_class1);
    let mut c = Vec::with_capacity(b.len() + 8);
    // ([^0-9])([\.,]) -> \1 \2_
    sub2(&b, &mut c, |x, y| !is_ascii_digit(x) && (y == '.' || y == ','), |x, y, o| {
        o.push(x);
        o.push(' ');
        o.push(y);
        o.push(' ');
    });
    // ([\.,])([^0-9]) -> _\1 \2
    sub2(&c, &mut b, |x, y| (x == '.' || x == ',') && !is_ascii_digit(y), |x, y, o| {
        o.push(' ');
        o.push(x);
        o.push(' ');
        o.push(y);
    });
    // ([0-9])(-) -> \1 \2_
    sub2(&b, &mut c, |x, y| is_ascii_digit(x) && y == '-', |x, y, o| {
        o.push(x);
        o.push(' ');
        o.push(y);
        o.push(' ');
    });
    join_split_chars(&c, out);
}

/// sacreBLEU `Tokenizer13a.__call__`.
pub fn tok_13a(line: &str, out: &mut String) {
    let mut l = line.replace("<skipped>", "");
    l = l.replace("-\n", "");
    l = l.replace('\n', " ");
    if l.contains('&') {
        l = l.replace("&quot;", "\"");
        l = l.replace("&amp;", "&");
        l = l.replace("&lt;", "<");
        l = l.replace("&gt;", ">");
    }
    let mut a: Vec<char> = Vec::with_capacity(l.len() + 2);
    a.push(' ');
    a.extend(l.chars());
    a.push(' ');
    tok_re_chars(&a, out);
}

/// sacreBLEU `TokenizerV14International.__call__`.
pub fn tok_intl(line: &str, out: &mut String) {
    let a: Vec<char> = line.chars().collect();
    let mut b = Vec::with_capacity(a.len() * 2 + 8);
    // (\P{N})(\p{P}) -> \1 \2_
    sub2(&a, &mut b, |x, y| !is_number(x) && is_punct(y), |x, y, o| {
        o.push(x);
        o.push(' ');
        o.push(y);
        o.push(' ');
    });
    let mut c = Vec::with_capacity(b.len() + 8);
    // (\p{P})(\P{N}) -> _\1 \2
    sub2(&b, &mut c, |x, y| is_punct(x) && !is_number(y), |x, y, o| {
        o.push(' ');
        o.push(x);
        o.push(' ');
        o.push(y);
    });
    // (\p{S}) -> _\1_
    sub1(&c, &mut b, is_symbol);
    join_split_chars(&b, out);
}

/// sacreBLEU `TokenizerZh._is_chinese_char`, *including* its quirks: two of the
/// ranges in sacreBLEU are written as `u' 0'` and `u'⾀0'`, which are
/// two-character strings in Python.  String comparison then makes those ranges
/// effectively U+2001..=U+2A6D and U+2F81..=U+2FA1.  We replicate that exactly.
pub fn is_chinese_char(c: char) -> bool {
    const R: &[(u32, u32)] = &[
        (0x3400, 0x4DB5),
        (0x4E00, 0x9FA5),
        (0x9FA6, 0x9FBB),
        (0xF900, 0xFA2D),
        (0xFA30, 0xFA6A),
        (0xFA70, 0xFAD9),
        (0x2001, 0x2A6D), // u' 0' .. u'⩭6' (sic)
        (0x2F81, 0x2FA1), // u'⾀0' .. u'⾡d' (sic)
        (0xFF00, 0xFFEF),
        (0x2E80, 0x2EFF),
        (0x3000, 0x303F),
        (0x31C0, 0x31EF),
        (0x2F00, 0x2FDF),
        (0x2FF0, 0x2FFF),
        (0x3100, 0x312F),
        (0x31A0, 0x31BF),
        (0xFE10, 0xFE1F),
        (0xFE30, 0xFE4F),
        (0x2600, 0x26FF),
        (0x2700, 0x27BF),
        (0x3200, 0x32FF),
        (0x3300, 0x33FF),
    ];
    let cp = c as u32;
    if cp < 0x2001 {
        return false;
    }
    R.iter().any(|&(a, b)| a <= cp && cp <= b)
}

/// sacreBLEU `TokenizerZh.__call__`.
pub fn tok_zh(line: &str, out: &mut String) {
    let line = py_strip(line);
    let mut a: Vec<char> = Vec::with_capacity(line.len() * 2);
    for ch in line.chars() {
        if is_chinese_char(ch) {
            a.push(' ');
            a.push(ch);
            a.push(' ');
        } else {
            a.push(ch);
        }
    }
    tok_re_chars(&a, out);
}

/// sacreBLEU `TokenizerChar.__call__`: `" ".join(line)`.
pub fn tok_char(line: &str, out: &mut String) {
    let mut first = true;
    for ch in line.chars() {
        if !first {
            out.push(' ');
        }
        out.push(ch);
        first = false;
    }
}

/// Convenience: tokenize then split into tokens (what BLEU consumes).
pub fn tokens(tokenized: &str) -> impl Iterator<Item = &str> {
    py_split(tokenized)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t13a(s: &str) -> String {
        let mut o = String::new();
        tok_13a(s, &mut o);
        o
    }
    #[test]
    fn basic_13a() {
        assert_eq!(t13a("Hello, world."), "Hello , world .");
        assert_eq!(t13a("It costs $1,000.50!"), "It costs $ 1,000.50 !");
        assert_eq!(t13a("1990-1995"), "1990 - 1995");
        assert_eq!(t13a("a &amp; b"), "a & b");
    }
}
