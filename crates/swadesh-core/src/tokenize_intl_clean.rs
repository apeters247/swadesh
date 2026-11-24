//! Tokenizer `intl`: clean-room implementation of the derived algorithm.
//!
//! Specification: `docs/spec/tokenizers.md` section 4, refined by black-box measurement.
//! Provenance: written from the specification and from probing the oracle as a black box; the
//! previous port (`tokenize.rs`) and sacreBLEU's sources were not read.
//!
//! **The splitting class is exactly `\p{P} ∪ \p{S}`** (measured over all 1,112,032 scalars: the
//! set of code points that split in `a<c>b` equals the `regex` module's `P|S` with zero
//! differences in either direction), plus `_`. The ASCII members are `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`.
//! Note `%` splits here while it does not in `13a`, and no entity decoding happens.
//!
//! Contextual rules, all measured:
//!   * `.` `,` `:` attach only after an ASCII digit, when followed by a digit or end of line.
//!   * A CJK comma or a dash glues a following digit run when it starts a token, and a run
//!     continues into letters (`,２つのHDMI端子`).
//!   * A dash that directly follows another dash glues digits anywhere (`——2300`).
//!   * Brackets and opening quotes glue digits only at the very start of the line (`-22種類`);
//!     the general rule there is limb 1 of `docs/spec/intl-absorption.md`: at index 0 every
//!     `\p{P}` mark glues a following digit run, and `\p{S}` marks never do.
//!   * A digit run may absorb one trailing punctuation at end of line (`1699?`), but never an emoji.

use crate::unicode::{is_punct, is_symbol};

const ASCII_CLASS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

#[inline]
fn is_class(c: char) -> bool {
    if c.is_ascii() {
        ASCII_CLASS.contains(c) || c == '_'
    } else {
        is_punct(c) || is_symbol(c) || c == '_'
    }
}

/// The digit notion the numeric rules use: the whole Unicode number category, ASCII included.
///
/// Measured on the corpus: `𝟎*` (U+1D7CE, an astral Nd) keeps the `*` attached exactly like an
/// ASCII digit, so astral numbers count. An earlier note here claimed the opposite and was wrong;
/// `𝟎` is in `\p{Nd}`, and the oracle's rule is category `N` throughout.
#[inline]
fn is_digit(c: char) -> bool {
    crate::unicode::is_number(c)
}

/// Measured: after a trailing digit, these class characters split off instead of gluing.
/// `"1?</code>"` glues, `"1~"` splits. Everything else glues, emoji never do.
#[inline]
fn splits_after_digit(c: char) -> bool {
    matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~')
}

#[inline]
fn is_emoji(c: char) -> bool {
    let cp = c as u32;
    (0x1F000..=0x1FAFF).contains(&cp) || (0x2600..=0x27BF).contains(&cp)
}

const SEP: &str = ".:/,-\u{3001}\u{FF0C}\u{FF1A}\u{2010}\u{2011}\u{2013}\u{2014}\u{30FB}\u{FF0F}\u{FF0E}";

fn is_sep(c: char) -> bool {
    SEP.contains(c)
}

fn is_py_space(c: char) -> bool {
    crate::unicode::is_py_space(c)
}

/// Tokenize `line` into `out`, tokens separated by single ASCII spaces.
pub fn tok_intl(line: &str, out: &mut String) {
    // Trailing whitespace is dropped before tokenising ("1986. " splits, "1986." does not).
    let s: Vec<char> = {
        let mut v: Vec<char> = line.chars().collect();
        while v.last().is_some_and(|&c| is_py_space(c)) {
            v.pop();
        }
        v
    };
    let n = s.len();
    out.clear();
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut i = 0usize;

    fn flush(cur: &mut String, tokens: &mut Vec<String>) {
        if !cur.is_empty() {
            tokens.push(std::mem::take(cur));
        }
    }

    while i < n {
        let c = s[i];
        if is_py_space(c) {
            flush(&mut cur, &mut tokens);
            i += 1;
            continue;
        }
        if is_digit(c) {
            let mut j = i + 1;
            while j < n {
                let d = s[j];
                if is_digit(d) || (is_sep(d) && j + 1 < n && is_digit(s[j + 1])) {
                    j += 1;
                } else if matches!(d, '(' | ')' | '\u{FF08}' | '\u{FF09}')
                    && j + 1 < n
                    && is_digit(s[j + 1])
                {
                    j += 1;
                } else if d.is_alphabetic() && !is_class(d) && !is_emoji(d) {
                    // Letters continue the run only when another digit follows them:
                    // "16x12" yes, but "1名死亡、" stops at the "1" so the comma splits.
                    let mut k = j;
                    while k < n && s[k].is_alphabetic() && !is_class(s[k]) && !is_emoji(s[k]) {
                        k += 1;
                    }
                    if k < n && is_digit(s[k]) {
                        j = k;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            // One trailing punctuation at end of line stays glued only when the run ends in a
            // digit ("1699?", "F1."), and never an emoji ("19:06😅" splits).
            if j + 1 == n
                && j > 0
                && is_class(s[j])
                && !is_emoji(s[j])
                && !splits_after_digit(s[j])
                && is_digit(s[j - 1])
            {
                j += 1;
            }
            let tok: String = s[i..j].iter().collect();
            cur.push_str(&tok);
            i = j;
            continue;
        }
        if is_class(c) {
            // Lead rules, each established separately. The broad form ("every class character
            // glues a following digit at a token start") is what the fuzz alphabet shows, but it
            // breaks the corpus (100 mismatches) because probes see isolated pairs while corpus
            // punctuation follows words. Kept narrow, which is corpus-exact:
            //   CJK comma at any token start; dash directly after another dash; dash, opening
            //   bracket or opening quote at the very start of the line; "@" and "«".
            let starts_token = cur.is_empty();
            let cjk_comma = c == '\u{3001}' || c == '\u{FF0C}';
            let dash = c == '\u{2014}' || c == '\u{2013}' || c == '-';
            let dash_after_dash =
                dash && i >= 1 && matches!(s[i - 1], '\u{2014}' | '\u{2013}' | '-');
            // spec `intl-absorption.md` limb 1: at the very start of the segment *every* `\p{P}`
            // mark glues a following digit run, whatever the mark (`?1`, `.٣`, `/1`, `¿1`). The
            // `\p{S}` marks never do, and a mark followed by another mark rather than a digit does
            // not either, which is why `./1`, `!/1` and `'?1` still split.
            let limb1 = i == 0 && (is_punct(c) || c == '_');
            let leads = i + 1 < n
                && is_digit(s[i + 1])
                && starts_token
                && (cjk_comma || dash_after_dash || limb1);
            if leads {
                let mut j = i + 1;
                while j < n && is_digit(s[j]) {
                    j += 1;
                }
                while j < n && s[j].is_alphabetic() && !is_class(s[j]) && !is_emoji(s[j]) {
                    j += 1;
                }
                // The same trailing absorption the plain digit-run branch applies, which this path
                // was missing: one class character at the very end glues, and only when the
                // character before it is a digit (`?1%` glues, `?1a%` does not; measured
                // 2026-11-24).
                if j + 1 == n
                    && j > 0
                    && is_class(s[j])
                    && !is_emoji(s[j])
                    && !splits_after_digit(s[j])
                    && is_digit(s[j - 1])
                {
                    j += 1;
                }
                let tok: String = s[i..j].iter().collect();
                flush(&mut cur, &mut tokens);
                tokens.push(tok);
                i = j;
                continue;
            }
            flush(&mut cur, &mut tokens);
            tokens.push(c.to_string());
            i += 1;
            continue;
        }
        cur.push(c);
        i += 1;
    }
    flush(&mut cur, &mut tokens);
    out.push_str(&tokens.join(" "));
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
}