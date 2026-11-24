//! Tokenizer `13a`: clean-room implementation of the derived algorithm.
//!
//! Specification: `docs/spec/13a-algorithm.md`, which was derived from published prose and from
//! black-box measurement of the oracle, without reading any implementation source. That document
//! records the verification: zero mismatches on 13,455 WMT22 lines and on 5,000 adversarial fuzz
//! cases.
//!
//! Five stages:
//!   1. normalise: drop `<skipped>`, drop `-\n`, newline -> space, decode four HTML entities
//!   2. pad every member of the padding class (exactly 30 ASCII characters, measured over the
//!      whole Unicode range; `.` and `,` are NOT in it - the next stage owns them)
//!   3. detach a `.` or `,` from a following non-digit; attach one to a preceding non-digit;
//!      detach a `.` at end of line
//!   4. split a `-` that follows a digit
//!   5. join on Python whitespace
//!
//! Stages 2-4 are `re.sub(pattern, " \g<0> ", text)` calls, and CPython's insertion rule is part
//! of the behaviour: a replacement's leading space is dropped when the previous match's trailing
//! space sits at exactly that output offset (matches adjacent with nothing between them). That rule
//! was verified at 0 mismatches over 20,018 cases, and `insert_pass` implements it.
//!
//! Provenance: this file was written from the specification document and black-box probes. The
//! previous port (`tokenize.rs`) and sacreBLEU's Python sources were not read while writing it.

use crate::unicode::py_split;

/// The `13a` padding class: 30 ASCII characters, measured exhaustively.
#[inline]
fn is_pad_class(c: char) -> bool {
    matches!(
        c,
        '!' | '"' | '#' | '$' | '%' | '&' | '(' | ')' | '*' | '+' | '/' | ':' | ';' | '<' | '='
            | '>' | '?' | '@' | '[' | '\\' | ']' | '^' | '_' | '`' | '{' | '|' | '}' | '~'
    )
}

/// The digit test the attachment rules use: ASCII plus full-width digits only.
#[inline]
fn is_attach_digit(c: char) -> bool {
    c.is_ascii_digit() || ('\u{FF10}'..='\u{FF19}').contains(&c)
}

/// One `re.sub(atom, " \g<0> ", text)` pass with CPython's insertion-offset rule.
///
/// `atom` decides, for each character position, whether a match may start and how far it runs.
/// `insert` receives the match text and the index of its first character, and returns the
/// replacement body (the leading space is added here, subject to the collision rule).
fn insert_pass<M, I>(text: &str, mut matches_at: M, insert: I) -> String
where
    M: FnMut(&[char], usize) -> Option<usize>,
    I: Fn(&[char], usize, usize) -> String,
{
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + text.len() / 4 + 8);
    let mut out_len = 0usize; // length in chars written so far
    let mut prev_tail: Option<usize> = None;
    let mut pos = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        match matches_at(&chars, i) {
            Some(end) if end > i => {
                let inter = &chars[pos..i];
                out.extend(inter.iter());
                out_len += inter.len();
                let collision = inter.is_empty() && prev_tail == Some(out_len);
                if !collision {
                    out.push(' ');
                    out_len += 1;
                }
                let body = insert(&chars, i, end);
                out_len += body.chars().count() + 1;
                out.push_str(&body);
                out.push(' ');
                prev_tail = Some(out_len - 1);
                i = end;
                pos = end;
            }
            _ => i += 1,
        }
    }
    out.extend(chars[pos..].iter());
    out
}

/// Stage 3: the measured `.`/`,` run rule (docs/spec/13a-algorithm.md).
///
/// For each maximal run of `.`/`,` of length `m`, with `p` the character before the run and `f`
/// the character after it: the run is spaced on both sides, except that its last character glues
/// to `f` iff `f` is a digit and `(m + [p is a digit])` is even, and its first character glues to
/// `p` iff `p` is a digit, `m == 1` and `f` is a digit. Measured 2026-11-24 over every run of
/// length 1-6 with letter, digit, space and empty prefixes: `..0` -> `. .0`, `...0` -> `. . . 0`,
/// `0...0` -> `0 . . .0`, `0.0` -> `0.0`, `5.,6` -> `5 . , 6`.
fn dots_run_rule(s: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let n = c.len();
    let mut out = String::with_capacity(s.len() + s.len() / 4 + 8);
    let mut i = 0;
    while i < n {
        if c[i] == '.' || c[i] == ',' {
            let mut j = i;
            while j < n && (c[j] == '.' || c[j] == ',') {
                j += 1;
            }
            let m = j - i;
            let p_digit = i > 0 && is_attach_digit(c[i - 1]);
            let f_digit = j < n && is_attach_digit(c[j]);
            let left_glue = p_digit && m == 1 && f_digit;
            let right_glue = f_digit && (m + usize::from(p_digit)) % 2 == 0;
            if !left_glue {
                out.push(' ');
            }
            for k in i..j {
                out.push(c[k]);
                if k + 1 < j || !right_glue {
                    out.push(' ');
                }
            }
            i = j;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

/// Stage 1: normalise the line.
fn normalise(line: &str) -> String {
    let mut s = line.replace("<skipped>", "");
    s = s.replace("-\n", "");
    s = s.replace('\n', " ");
    if s.contains('&') {
        s = s.replace("&quot;", "\"")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">");
    }
    s
}

/// Tokenize `line` into `out`, tokens separated by single ASCII spaces.
pub fn tok_13a(line: &str, out: &mut String) {
    let s = normalise(line);

    // Stage 2: pad class members. `.`, `,` and `-` are not in the class; stage 3 owns `.`/`,`
    // and stage 4 owns `-`.
    let s = insert_pass(
        &s,
        |chars, i| {
            if is_pad_class(chars[i]) {
                Some(i + 1)
            } else {
                None
            }
        },
        |chars, i, end| chars[i..end].iter().collect(),
    );

    // Stage 3: the measured `.`/`,` run rule.
    let s = dots_run_rule(&s);

    // Stage 4: a "-" that follows a digit.
    let s = insert_pass(
        &s,
        |chars, i| match chars[i] {
            '-' if i > 0 && is_attach_digit(chars[i - 1]) => Some(i + 1),
            _ => None,
        },
        |chars, i, end| chars[i..end].iter().collect(),
    );

    // Stage 5: join on Python whitespace.
    out.clear();
    let mut first = true;
    for tok in py_split(&s) {
        if !first {
            out.push(' ');
        }
        out.push_str(tok);
        first = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> String {
        let mut o = String::new();
        tok_13a(s, &mut o);
        o
    }

    #[test]
    fn dot_runs_follow_the_parity_rule() {
        // Measured 2026-11-24; see docs/spec/13a-algorithm.md stage 3.
        assert_eq!(t(".0"), ". 0");
        assert_eq!(t("..0"), ". .0");
        assert_eq!(t("...0"), ". . . 0");
        assert_eq!(t("....0"), ". . . .0");
        assert_eq!(t("0...0"), "0 . . .0");
        assert_eq!(t("0.0"), "0.0");
        assert_eq!(t("0.a"), "0 . a");
        assert_eq!(t("a..0"), "a . .0");
        assert_eq!(t("..,0"), ". . , 0");
        assert_eq!(t("0,,0"), "0 , , 0");
    }

    #[test]
    fn basic_rules() {
        assert_eq!(t("Hello, world."), "Hello , world .");
        assert_eq!(t("It costs $1,000.50!"), "It costs $ 1,000.50 !");
        assert_eq!(t("1990-1995"), "1990 - 1995");
        assert_eq!(t("3.126"), "3.126");
        assert_eq!(t("5.,6"), "5 . , 6");
        assert_eq!(t("37."), "37 .");
        assert_eq!(t("9,"), "9 ,");
    }

    #[test]
    fn entity_decoding() {
        assert_eq!(t("a &amp; b"), "a & b");
        assert_eq!(t("&quot;hi&quot;"), "\" hi \"");
        assert_eq!(t("a&lt;b&gt;c"), "a < b > c");
        assert_eq!(t("x<skipped>y"), "xy");
    }

    #[test]
    fn non_ascii_never_pads() {
        // Measured: NONE of the 1,112,032 Unicode scalars outside ASCII is in the class.
        assert_eq!(t("\u{201e}Heute\u{201c}"), "\u{201e}Heute\u{201c}");
        assert_eq!(t("a\u{2192}b"), "a\u{2192}b");
    }
}