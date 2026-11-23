//! TER (translation edit rate) with TERCOM-style shifts.
//!
//! Specification: `docs/spec/ter.md`. Written from Snover et al. 2006 and from black-box
//! measurement of the oracle; no implementation source was read, and no reference tokenizer was
//! consulted. Where the spec states a rule was established by measurement, the fixture
//! `tests/oracle/ter_probe.jsonl` holds the verdict.
//!
//! Shape of the computation, in order:
//!
//! 1. tokenize (four modes, spec section 3),
//! 2. per reference, a greedy search for word-block shifts that reduce Levenshtein distance
//!    (spec section 4),
//! 3. `edits = distance + shifts`, averaged over references (spec section 5).

use crate::stats::{TerConfig, TerError};
use crate::unicode::py_rstrip;

/// Shifts move at most this many positions, and blocks are at most this long (spec 4.2).
/// How far a block may move, in positions.
const MAX_SHIFT_DIST: i64 = 50;
/// How long a shifted block may be. Distinct from the distance limit: conflating the two makes the
/// search enumerate far more candidates than the oracle does and exhausts the budget before
/// reaching the shift that helps (measured: 1000 candidates against the oracle's 71).
const MAX_SHIFT_SIZE: usize = 10;
/// A segment evaluates at most this many candidate shifts in total (spec 4.2).
const MAX_CANDIDATES: usize = 1_000;

const INF: i64 = 10_000_000_000_000_000;

const OP_UNDEF: u8 = b'x';
const OP_MATCH: u8 = b' ';
const OP_SUB: u8 = b's';
const OP_INS: u8 = b'i';
const OP_DEL: u8 = b'd';

// ------------------------------------------------------------------ character classes

/// ASCII punctuation that the normalized tokenizer separates, except `'`, and except `. , -`
/// where the number and hyphen rules protect them (spec 3.2 step 7).
#[inline]
fn separates(c: char) -> bool {
    matches!(c,
        '!' | '"' | '#' | '$' | '%' | '&' | '(' | ')' | '*' | '+' | ',' | '/' | ':' | ';' | '<'
        | '=' | '>' | '?' | '@' | '[' | '\\' | ']' | '^' | '_' | '`' | '{' | '|' | '}' | '~')
}

#[inline]
fn is_ascii_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// Fullwidth and CJK punctuation deleted in `no_punct` Asian mode (spec 3.3).
#[inline]
fn is_asian_punct(c: char) -> bool {
    // spec 3.3: exactly 39 code points, measured by comparing `_preprocess_segment` with and
    // without `asian_support` over 0x2E00..0x3100 and 0xFF00..0xFFF0. U+FF0D is not in the set.
    matches!(c,
        '\u{3001}' | '\u{3002}'
        | '\u{3008}'..='\u{3011}'
        | '\u{3014}'..='\u{301F}'
        | '\u{30FB}'
        | '\u{FF01}' | '\u{FF02}' | '\u{FF08}' | '\u{FF09}' | '\u{FF0C}' | '\u{FF0E}'
        | '\u{FF1A}' | '\u{FF1B}' | '\u{FF1F}'
        | '\u{FF61}'..='\u{FF65}')
}

/// Punctuation deleted under `no_punct`, measured as an explicit set rather than the whole
/// punctuation block: `# $ % & * + / < = > @ [ \\ ] ^ _ ` { | } ~` and `'` survive deletion.
#[inline]
fn is_ascii_punct(c: char) -> bool {
    matches!(c, '!' | '"' | ',' | '.' | ':' | ';' | '?')
}

/// Characters the Asian tokenizer splits one by one (spec 3.3, measured ranges).
#[inline]
fn is_asian(c: char) -> bool {
    let cp = c as u32;
    matches!(cp,
        0x2E80..=0x2EFF
        | 0x3001..=0x3002
        | 0x3008..=0x3011
        | 0x3014..=0x301F
        | 0x30FB
        | 0x31C0..=0x31EF
        | 0x3200..=0x33FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE4F
        | 0xFF01..=0xFF02
        | 0xFF08..=0xFF09
        | 0xFF0C
        | 0xFF0E
        | 0xFF1A
        | 0xFF1B
        | 0xFF1F
        | 0xFF61..=0xFF65)
}

/// Headline quote marks folded to their plain form before tokenizing (spec 3.2 step 3).
#[inline]

// ------------------------------------------------------------------ tokenizer

/// Push `tok` onto `out`, space-separated.
#[inline]
fn push(out: &mut Vec<String>, tok: &str) {
    if !tok.is_empty() {
        out.push(tok.to_string());
    }
}

/// Python `str.lower()` per character, accumulated into `buf`.
fn push_lower(buf: &mut String, c: char, case_sensitive: bool) {
    if case_sensitive {
        buf.push(c);
    } else {
        for l in c.to_lowercase() {
            buf.push(l);
        }
    }
}

/// Apply `no_punct`: delete the marks the measurements show are deleted.
///
/// Measured: `( )` are always deleted, `[ ] { }` never are, and everything else follows the fixed
/// small set in `is_ascii_punct`. Note the asymmetry that made this worth measuring: the plain
/// deletion set is not "all ASCII punctuation", and brackets are not a class of their own.
fn strip_punct(raw: &[char], asian: bool, normalized: bool) -> Vec<char> {
    let mut out = Vec::with_capacity(raw.len());
    for (i, &c) in raw.iter().enumerate() {
        let delete = (is_asian_punct(c) && asian) || is_ascii_punct(c) || c == '(' || c == ')';
        if delete {
            // spec 3.2: the default mode removes the mark outright, so "a?b" stays one token;
            // the normalized mode leaves a separator, so it becomes "a b". The exception is a
            // dot or comma between ASCII digits, part of the protected number token: deleting it
            // joins the digits, so "2.2" stays "22" and "1,000" stays "1000".
            if normalized {
                let joins = (c == '.' || c == ',')
                    && i > 0
                    && is_ascii_digit(raw[i - 1])
                    && raw.get(i + 1).map_or(false, |&n| is_ascii_digit(n));
                if !joins {
                    out.push(' ');
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Split `text` into TER words (spec section 3).
///
/// Structure, which the measurements forced: the segment is first cut into *pieces* on
/// **whitespace and on the digit-separator pattern only** (spec 3.2 steps 4-5), and the remaining
/// marks are then resolved inside each piece (step 7). The piece boundaries change how the hyphen
/// rule decides, so the two passes are not interchangeable.
pub fn tokenize(cfg: &TerConfig, text: &str) -> Vec<String> {
    // spec 3.2 step 2: in normalized mode the XML entities are unescaped after lowercasing, so
    // uppercase forms are covered only when `case_sensitive` is false. Four sequential global
    // passes in this order: "&amp;lt;" collapses to "<" because "&amp;" runs first, and
    // "&amp;amp;" leaves "&amp;" because the second "amp;" has no "&". The pre-lowercase only
    // runs when an entity can be present, so segments without "&" keep the original code path.
    let unescaped: String;
    let text: &str = if cfg.normalized && text.contains('&') {
        let mut s = String::with_capacity(text.len());
        for c in text.chars() {
            push_lower(&mut s, c, cfg.case_sensitive);
        }
        for (from, to) in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\"")] {
            if s.contains(from) {
                s = s.replace(from, to);
            }
        }
        unescaped = s;
        &unescaped
    } else {
        text
    };
    // Two different gates, measured (spec 3.3.1 and 3.3.2): the fullwidth punctuation set of
    // `no_punct` applies whenever `asian_support` is on, but the per-character split only happens
    // in normalized mode. Without the second gate `asian_support` alone disagrees with the oracle
    // on 227 of 300 random CJK segments.
    let asian_text = cfg.asian_support && text.chars().any(is_asian);
    let asian = cfg.normalized && asian_text;
    let raw: Vec<char> = text.chars().collect();
    let mut out: Vec<String> = Vec::new();

    // Characters deleted by `no_punct` never reach the piece scanner. Which ones is decided in
    // `strip_punct`, because it depends on what has come before.
    let kept: Vec<char> = if cfg.no_punct {
        strip_punct(&raw, asian_text, cfg.normalized)
    } else {
        raw.clone()
    };

    // ---- pass 1: cut into pieces
    let mut pieces: Vec<Vec<char>> = Vec::new();
    let mut piece: Vec<char> = Vec::new();
    let mut i = 0usize;
    // Character just consumed, and the one before it, for the digit lookaround: a lone separator
    // must look at what preceded it, not at itself.
    let mut last_char: Option<char> = None;
    let mut last_before: Option<char> = None;
    while i < kept.len() {
        let c = kept[i];
        last_before = last_char;
        last_char = Some(c);
        if asian && is_asian(c) {
            // An Asian character is its own piece and also ends the one before.
            if !piece.is_empty() {
                pieces.push(std::mem::take(&mut piece));
            }
            pieces.push(vec![c]);
            i += 1;
            continue;
        }
        if crate::unicode::is_py_space(c) && !(asian && is_asian(c)) {
            if !piece.is_empty() {
                pieces.push(std::mem::take(&mut piece));
            }
            i += 1;
            continue;
        }
        if cfg.normalized && (c == '.' || c == ',') {
            let prev_digit = i > 0 && last_before.map_or(false, is_ascii_digit);
            let next_digit = kept.get(i + 1).map_or(false, |&n| is_ascii_digit(n));
            if !(prev_digit && next_digit) {
                // A lone separator is its own piece (`a.b` -> `a` `.` `b`, `.,.,` -> four marks).
                if !piece.is_empty() {
                    pieces.push(std::mem::take(&mut piece));
                }
                pieces.push(vec![c]);
                i += 1;
                continue;
            }
        }
        piece.push(c);
        i += 1;
    }
    if !piece.is_empty() {
        pieces.push(piece);
    }

    // ---- pass 2: resolve marks inside each piece
    for p in &pieces {
        if !cfg.normalized {
            let mut w = String::new();
            for &c in p {
                push_lower(&mut w, c, cfg.case_sensitive);
            }
            push(&mut out, &w);
            continue;
        }
        if p.len() == 1 {
            let mut w = String::new();
            push_lower(&mut w, p[0], cfg.case_sensitive);
            push(&mut out, &w);
            continue;
        }
        let mut word = String::new();
        let mut k = 0usize;
        while k < p.len() {
            let c = p[k];
            let mut closes_word = false;
            let standalone = if c == '-' {
                // Measured behaviour, which is asymmetric: sticky when the character before it is
                // not a digit (`a-1`), or when nothing follows it but a mark or the end of the
                // text (`Finanz-,`, `a-`). `5-a` and `12-13` split.
                let prev = if k == 0 { None } else { Some(p[k - 1]) };
                let next = p.get(k + 1).copied();
                let non_digit_before = prev.map_or(true, |q| !is_ascii_digit(q));
                let followed_by_nothing = match next {
                    None => true,
                    Some(q) => separates(q) || q == '.' || q == ',',
                };
                let followed_by_nothing = followed_by_nothing && next.map_or(true, |q| !q.is_ascii_digit());
                !(non_digit_before || followed_by_nothing)
                        } else if c == '\'' {
                // `'s` ends the piece: the apostrophe closes the current word and the `s` begins
                // the next token. Any other apostrophe stays inside its word (`don't`, `x'y`),
                // and a lone `'` prefix is its own piece already.
                let s_next = k + 1 < p.len() && p[k + 1] == 's';
                closes_word = s_next && {
                    let after = k + 2;
                    after == p.len()
                        || matches!(p[after], '.' | ',' | '\'')
                        || separates(p[after])
                };
                false
            } else if c == '.' || c == ',' {
                // Pass 1 already cut every lone separator out, so one surviving here is inside a
                // number and stays put.
                false
            } else {
                separates(c)
            };
            if closes_word {
                // `'s` is emitted whole, so the word before it ends at the apostrophe.
                push(&mut out, &word);
                word.clear();
                let mut w = String::new();
                push_lower(&mut w, c, cfg.case_sensitive);
                push_lower(&mut w, p[k + 1], cfg.case_sensitive);
                push(&mut out, &w);
            } else if standalone {
                push(&mut out, &word);
                word.clear();
                let mut w = String::new();
                push_lower(&mut w, c, cfg.case_sensitive);
                push(&mut out, &w);
            } else {
                push_lower(&mut word, c, cfg.case_sensitive);
            }
            // `'s` consumed two characters.
            k += if closes_word { 2 } else { 1 };
        }
        push(&mut out, &word);
    }
    out
}

// ------------------------------------------------------------------ edit distance

/// Band half-width around the anti-diagonal, in cells (spec `ter.md` section 4.1).
const BEAM: i64 = 25;

/// The oracle's band: only cells near the anti-diagonal are computed, so a very short hypothesis
/// against a very long reference reports a distance above the true minimum. Reproducing it is the
/// point — see `docs/spec/ter.md`, and the note there on why a full matrix disagrees.
fn beam_width(n_hyp: usize, n_ref: usize) -> i64 {
    let ratio = if n_hyp > 0 { n_ref as f64 / n_hyp as f64 } else { 1.0 };
    if (BEAM as f64) < ratio / 2.0 {
        (ratio / 2.0 + BEAM as f64).ceil() as i64
    } else {
        BEAM
    }
}

/// Half-open `[min_j, max_j)` of reference columns computed for hypothesis step `i`.
fn band(i: usize, n_hyp: usize, n_ref: usize) -> (usize, usize) {
    let ratio = if n_hyp > 0 { n_ref as f64 / n_hyp as f64 } else { 1.0 };
    let width = beam_width(n_hyp, n_ref);
    let pseudo = (i as f64 * ratio).floor() as i64;
    let lo = (pseudo - width).max(0) as usize;
    let hi = ((pseudo + width).min(n_ref as i64 + 1)).max(0) as usize;
    (lo.min(n_ref + 1), hi)
}

/// Last row of the banded distance, used while scoring candidate shifts.
fn distance_row(refs: &[u32], hyp: &[u32], row: &mut Vec<i64>) {
    let m = refs.len();
    let n = hyp.len();
    let mut prev: Vec<i64> = (0..=m as i64).collect();
    let mut cur = vec![INF; m + 1];
    for j in 1..=n {
        let (lo, hi) = band(j, n, m);
        for v in cur.iter_mut() {
            *v = INF;
        }
        if lo == 0 {
            cur[0] = prev[0] + 1;
        }
        for i in lo.max(1)..hi {
            let cost = if refs[i - 1] == hyp[j - 1] { 0 } else { 1 };
            let sub = prev[i - 1].saturating_add(cost);
            let del = prev[i].saturating_add(1);
            let ins = cur[i - 1].saturating_add(1);
            cur[i] = sub.min(del).min(ins);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    row.clear();
    row.extend_from_slice(&prev);
}

/// Alignment of the hypothesis to the reference: one entry per reference position.
struct Trace {
    align: Vec<Option<i64>>,
    hyp_err: Vec<u8>,
    ref_err: Vec<u8>,
    cost: i64,
}

/// Turn an operation sequence into `(alignment, hypothesis-error flags, reference-error flags)`.
///
/// `align[r]` is the hypothesis position that reference position `r` was paired with (spec 4.1).
fn trace_to_alignment(ops: &[u8], refs_len: usize) -> Trace {
    let mut pos_hyp: i64 = -1;
    let mut align: Vec<Option<i64>> = Vec::with_capacity(ops.len());
    let mut hyp_err = Vec::with_capacity(ops.len());
    let mut ref_err = Vec::with_capacity(ops.len());
    for &o in ops {
        match o {
            OP_MATCH | OP_SUB => {
                pos_hyp += 1;
                align.push(Some(pos_hyp));
                let e = u8::from(o == OP_SUB);
                hyp_err.push(e);
                ref_err.push(e);
            }
            OP_INS => {
                // An insertion consumes no reference position, so there is no alignment entry for
                // it: this is the hole the shift loop breaks out at (measured).
                pos_hyp += 1;
                hyp_err.push(1);
            }
            OP_DEL => {
                // A deletion consumes a reference position and pairs it with the hypothesis
                // position just passed (measured).
                align.push(Some(pos_hyp));
                ref_err.push(1);
            }
            _ => unreachable!("trace holds only the four operations"),
        }
    }
    // One slot per reference position, so `align[r]` is `r`'s partner (or the hole the oracle
    // stores as a negative index).
    // Leave the alignment exactly as long as the walk produced: the oracle's map is keyed by the
    // reference positions it actually consumed, and only an insertion can leave one out. Padding
    // to the full reference length would make the offset loop treat a leading insertion as a
    // usable target.
    align.truncate(refs_len);
    Trace { align, hyp_err, ref_err, cost: 0 }
}

/// Banded Levenshtein with the trace that `trace_to_alignment` consumes (spec 4.1).
///
/// The trace is read back along the path of the last cell: the operations that rewrite the
/// **hypothesis** as the **reference**, which is the direction the shift search reads it in.
fn trace_align(refs: &[u32], hyp: &[u32], ops: &mut Vec<u8>) -> Trace {
    // Rows are the hypothesis and columns the reference, which is the oracle's own orientation: its
    // distance loop advances over the hypothesis. The distance is the same either way, but the
    // backtrace path is not, and the shift search reads that path.
    let n = hyp.len();
    let m = refs.len();
    let w = m + 1;

    let mut cost = vec![INF; (n + 1) * w];
    let mut op = vec![OP_UNDEF; (n + 1) * w];
    for j in 0..=m {
        cost[j] = j as i64;
        op[j] = OP_DEL;
    }
    for i in 0..=n {
        cost[i * w] = i as i64;
        op[i * w] = OP_INS;
    }
    // Only the band the oracle computes: a cell outside it keeps the infinite distance, exactly as
    // in `distance_row`.
    for i in 1..=n {
        let (lo, hi) = band(i, n, m);
        for j in lo.max(1)..hi {
            let same = hyp[i - 1] == refs[j - 1];
            let sub = cost[(i - 1) * w + j - 1] + if same { 0 } else { 1 };
            let del = cost[i * w + j - 1] + 1;
            let ins = cost[(i - 1) * w + j] + 1;
            // Strict "<" in the order the oracle tries (substitution, deletion, insertion).
            let mut best_cost = sub;
            let mut best_op = if same { OP_MATCH } else { OP_SUB };
            if del < best_cost {
                best_cost = del;
                best_op = OP_DEL;
            }
            if ins < best_cost {
                best_cost = ins;
                best_op = OP_INS;
            }
            cost[i * w + j] = best_cost;
            op[i * w + j] = best_op;
        }
    }

    ops.clear();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        let o = if j == 0 {
            OP_INS
        } else if i == 0 {
            OP_DEL
        } else {
            op[i * w + j]
        };
        ops.push(o);
        match o {
            OP_MATCH | OP_SUB => {
                i -= 1;
                j -= 1;
            }
            OP_INS => i -= 1,
            OP_DEL => j -= 1,
            _ => {
                i = i.saturating_sub(1);
                j = j.saturating_sub(1);
            }
        }
    }
    ops.reverse();

    trace_to_alignment(ops, refs.len())
}


// ------------------------------------------------------------------ shifts

/// Apply a shift of `hyp[start..start+len]` to position `target` (spec 4.2).
fn apply_shift(hyp: &[u32], start: usize, len: usize, target: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(hyp.len());
    if target < start {
        out.extend_from_slice(&hyp[..target]);
        out.extend_from_slice(&hyp[start..start + len]);
        out.extend_from_slice(&hyp[target..start]);
        out.extend_from_slice(&hyp[(start + len).min(hyp.len())..]);
    } else if target > start + len {
        out.extend_from_slice(&hyp[..start]);
        out.extend_from_slice(&hyp[(start + len).min(hyp.len())..target.min(hyp.len())]);
        out.extend_from_slice(&hyp[start..start + len]);
        out.extend_from_slice(&hyp[target.min(hyp.len())..]);
    } else {
        let lo = (start + len).min(hyp.len());
        let hi = (len + target).min(hyp.len());
        out.extend_from_slice(&hyp[..start]);
        out.extend_from_slice(&hyp[lo..hi]);
        out.extend_from_slice(&hyp[start..start + len]);
        out.extend_from_slice(&hyp[hi..]);
    }
    out
}

/// Every `(hyp_start, ref_start, len)` where the two blocks agree, in the order the search visits
/// them; each growing length is its own candidate (spec 4.2).
fn shifted_pairs(hyp: &[u32], refs: &[u32]) -> Vec<(usize, usize, usize)> {
    let (n_h, n_r) = (hyp.len(), refs.len());
    let mut out = Vec::new();
    for s_h in 0..n_h {
        for s_r in 0..n_r {
            if (s_r as i64 - s_h as i64).abs() > MAX_SHIFT_DIST {
                continue;
            }
            let mut len = 0usize;
            while s_h + len < n_h && s_r + len < n_r && hyp[s_h + len] == refs[s_r + len] {
                len += 1;
                out.push((s_h, s_r, len));
                if len == MAX_SHIFT_SIZE || n_h == s_h + len || n_r == s_r + len {
                    break;
                }
            }
        }
    }
    out
}

/// Scoring state reused across candidates.
struct Scratch {
    row: Vec<i64>,
    shifted: Vec<u32>,
    ops: Vec<u8>,
}

/// Find the best shift of `hyp` toward `refs`, or `None` when no shift reduces the distance.
///
/// Returns `(gain, shifted_hypothesis, candidates_evaluated)`.
fn best_shift(
    hyp: &[u32],
    refs: &[u32],
    scratch: &mut Scratch,
    budget: usize,
) -> Option<(i64, Vec<u32>, usize)> {
    let trace = trace_align(refs, hyp, &mut scratch.ops);
    distance_row(refs, hyp, &mut scratch.row);
    let before = scratch.row[refs.len()];

    let mut best: Option<(i64, usize, i64, i64, Vec<u32>)> = None;
    let mut used = 0usize;

    for (s_h, s_r, len) in shifted_pairs(hyp, refs) {
        if used >= budget {
            break;
        }
        // Only shift a block the hypothesis got wrong, toward a block the reference got wrong.
        if trace.hyp_err[s_h..s_h + len].iter().all(|&e| e == 0) {
            continue;
        }
        if trace.ref_err[s_r..s_r + len].iter().all(|&e| e == 0) {
            continue;
        }
        // Never move a block into the middle of itself.
        let a = match trace.align.get(s_r).copied().flatten() {
            Some(a) => a,
            // No partner for that reference position: nothing to align the block against.
            None => continue,
        };
        if (s_h as i64) <= a && a < (s_h + len) as i64 {
            continue;
        }

        let mut prev_idx: i64 = -1;
        for offset in -1i64..len as i64 {
            let pos = s_r as i64 + offset;
            let idx: i64 = if pos == -1 {
                0
            } else if let Some(Some(a)) = trace.align.get(pos as usize).copied() {
                a + 1
            } else {
                // A hole: the oracle breaks out of the offset loop here.
                break;
            };
            if idx == prev_idx {
                continue;
            }
            prev_idx = idx;

            scratch.shifted = apply_shift(hyp, s_h, len, idx as usize);
            distance_row(refs, &scratch.shifted, &mut scratch.row);
            let cost = scratch.row[refs.len()];
            used += 1;

            // Later candidates only win on a strictly better key, so the first found wins ties.
            let key = (before - cost, len, -(s_h as i64), -idx);
            let better = match &best {
                None => true,
                Some((score, l, ns, ni, _)) => key > (*score, *l, *ns, *ni),
            };
            if better {
                best = Some((key.0, key.1, key.2, key.3, scratch.shifted.clone()));
            }
            if used >= budget {
                break;
            }
        }
    }

    best.map(|(gain, _, _, _, words)| (gain, words, used))
}

/// `(edits, reference length)` for one hypothesis against one reference (spec 4.3).
pub fn translation_edit_rate(hyp_words: &[u32], ref_words: &[u32]) -> (i64, usize) {
    if ref_words.is_empty() {
        return (hyp_words.len() as i64, 0);
    }
    let mut scratch = Scratch { row: Vec::new(), shifted: Vec::new(), ops: Vec::new() };
    let mut current = hyp_words.to_vec();
    let mut shifts = 0i64;
    let mut spent = 0usize;

    loop {
        let budget = MAX_CANDIDATES.saturating_sub(spent);
        if budget == 0 {
            break;
        }
        match best_shift(&current, ref_words, &mut scratch, budget) {
            None => break,
            Some((gain, words, used)) => {
                spent += used;
                if gain <= 0 {
                    break;
                }
                shifts += 1;
                current = words;
            }
        }
    }

    distance_row(ref_words, &current, &mut scratch.row);
    (shifts + scratch.row[ref_words.len()], ref_words.len())
}

// ------------------------------------------------------------------ segment level

/// `[edits, average reference length]` for one segment (spec 5).
///
/// Lowercasing happens inside the tokenizer; the caller passes the raw strings.
pub fn ter_segment(cfg: &TerConfig, hyp: &str, refs: &[&str]) -> Result<[f64; 2], TerError> {
    if refs.is_empty() {
        return Err(TerError);
    }
    let hyp_toks = tokenize(cfg, py_rstrip(hyp));

    let mut best_edits = i64::MAX;
    let mut total_len = 0f64;
    for r in refs {
        // Each reference gets its own id space, built from the hypothesis words first so that a
        // word present in both has one id. Sharing a map across references would renumber the
        // earlier reference's words.
        let ref_toks = tokenize(cfg, py_rstrip(r));
        let mut ids: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
        let mut numbered: Vec<&str> = hyp_toks.iter().map(String::as_str).collect();
        numbered.extend(ref_toks.iter().map(String::as_str));
        let hyp_ids: Vec<u32> = hyp_toks
            .iter()
            .map(|t| {
                let n = ids.len() as u32;
                *ids.entry(t.as_str()).or_insert(n)
            })
            .collect();
        let ref_ids: Vec<u32> = ref_toks
            .iter()
            .map(|t| {
                let n = ids.len() as u32;
                *ids.entry(t.as_str()).or_insert(n)
            })
            .collect();
        let _ = numbered;
        let (edits, len) = translation_edit_rate(&hyp_ids, &ref_ids);
        // Measured: edits are the minimum over references (the best reference wins), while the
        // length is their average. Not symmetric, and not the mean of the per-reference rates.
        best_edits = best_edits.min(edits);
        total_len += len as f64;
    }
    let k = refs.len() as f64;
    Ok([best_edits as f64, total_len / k])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(cfg: &TerConfig, s: &str) -> Vec<String> {
        tokenize(cfg, s)
    }

    #[test]
    fn plain_mode_only_splits_whitespace() {
        let cfg = TerConfig::default();
        assert_eq!(words(&cfg, "a.b  a;b"), vec!["a.b", "a;b"]);
        assert_eq!(words(&cfg, "  Hello   World "), vec!["hello", "world"]);
        let np = TerConfig { no_punct: true, ..TerConfig::default() };
        assert_eq!(words(&np, "a.b"), vec!["ab"]);
        assert_eq!(words(&np, "Hello, world!"), vec!["hello", "world"]);
    }

    #[test]
    fn normalized_protects_numbers_and_letter_hyphens() {
        let cfg = TerConfig { normalized: true, ..TerConfig::default() };
        assert_eq!(words(&cfg, "3.14"), vec!["3.14"]);
        assert_eq!(words(&cfg, "1,2,3"), vec!["1,2,3"]);
        assert_eq!(words(&cfg, "a.14"), vec!["a", ".", "14"]);
        assert_eq!(words(&cfg, "12-13"), vec!["12", "-", "13"]);
        assert_eq!(words(&cfg, "a-13"), vec!["a-13"]);
        assert_eq!(words(&cfg, "13-a"), vec!["13", "-", "a"]);
        assert_eq!(words(&cfg, "don't"), vec!["don't"]);
        assert_eq!(words(&cfg, "a\u{2014}b"), vec!["a\u{2014}b"]);
    }

    #[test]
    fn asian_mode_splits_per_character() {
        let cfg = TerConfig { normalized: true, asian_support: true, ..TerConfig::default() };
        assert_eq!(words(&cfg, "\u{4e2d}\u{6587}\u{3002}"), vec!["\u{4e2d}", "\u{6587}", "\u{3002}"]);
        // Katakana is not in the Asian ranges, so it stays one token.
        assert_eq!(words(&cfg, "\u{30a2}\u{30a4}"), vec!["\u{30a2}\u{30a4}"]);
    }

    #[test]
    fn normalized_no_punct_leaves_a_separator() {
        // spec 3.2: "a?b" -> "a b" in normalized mode, "ab" in the default mode.
        let norm = TerConfig { normalized: true, no_punct: true, ..TerConfig::default() };
        assert_eq!(words(&norm, "a?b"), vec!["a", "b"]);
        assert_eq!(words(&norm, "a?b?c"), vec!["a", "b", "c"]);
        assert_eq!(words(&norm, "a.b"), vec!["a", "b"]);
        assert_eq!(words(&norm, "a[b]c"), vec!["a", "[", "b", "]", "c"]);
        let plain = TerConfig { no_punct: true, ..TerConfig::default() };
        assert_eq!(words(&plain, "a?b"), vec!["ab"]);
        assert_eq!(words(&plain, "a?b?c"), vec!["abc"]);
    }

    #[test]
    fn normalized_no_punct_joins_digits_and_spaces_other_marks() {
        // spec 3.2: only an ASCII dot or comma between digits joins after deletion.
        let cfg = TerConfig { normalized: true, no_punct: true, ..TerConfig::default() };
        assert_eq!(words(&cfg, "2.2"), vec!["22"]);
        assert_eq!(words(&cfg, "1,000"), vec!["1000"]);
        assert_eq!(words(&cfg, "12.34.56"), vec!["123456"]);
        assert_eq!(words(&cfg, "2?2"), vec!["2", "2"]);
        assert_eq!(words(&cfg, "a.b"), vec!["a", "b"]);
        assert_eq!(words(&cfg, "a[b]c"), vec!["a", "[", "b", "]", "c"]);
    }

    #[test]
    fn normalized_unescapes_xml_entities() {
        // spec 3.2 step 2: four sequential global replacements after lowercasing. "&amp;" runs
        // first, so "&amp;lt;" collapses to "<" while "&amp;amp;" leaves "&amp;" for the splitter.
        let cfg = TerConfig { normalized: true, ..TerConfig::default() };
        assert_eq!(words(&cfg, "&amp;"), vec!["&"]);
        assert_eq!(words(&cfg, "&lt;"), vec!["<"]);
        assert_eq!(words(&cfg, "&gt;"), vec![">"]);
        assert_eq!(words(&cfg, "&quot;"), vec!["\""]);
        assert_eq!(words(&cfg, "&amp;lt;"), vec!["<"]);
        assert_eq!(words(&cfg, "&amp;gt;"), vec![">"]);
        assert_eq!(words(&cfg, "&amp;amp;"), vec!["&", "amp", ";"]);
        assert_eq!(words(&cfg, "a&amp;b"), vec!["a", "&", "b"]);
        // Lowercasing runs first, so uppercase spellings are covered when case_sensitive is off.
        assert_eq!(words(&cfg, "&AMP;"), vec!["&"]);
        assert_eq!(words(&cfg, "&AMP;LT;"), vec!["<"]);
        // With case_sensitive on, only the lowercase spellings are replaced.
        let cs = TerConfig { normalized: true, case_sensitive: true, ..TerConfig::default() };
        assert_eq!(words(&cs, "&amp;"), vec!["&"]);
        assert_eq!(words(&cs, "&AMP;"), vec!["&", "AMP", ";"]);
        // The default tokenizer does not unescape entities at all.
        let plain = TerConfig::default();
        assert_eq!(words(&plain, "&amp;"), vec!["&amp;"]);
    }

    #[test]
    fn asian_punct_set_has_ff1b_and_not_ff0d() {
        // spec 3.3: 39 code points, measured; U+FF1B is in the set and U+FF0D is not.
        let cfg = TerConfig {
            normalized: true,
            no_punct: true,
            asian_support: true,
            ..TerConfig::default()
        };
        assert_eq!(words(&cfg, "a\u{FF1B}b"), vec!["a", "b"]);
        assert_eq!(words(&cfg, "a\u{FF0D}b"), vec!["a\u{FF0D}b"]);
        assert_eq!(words(&cfg, "a\u{FF61}b"), vec!["a", "b"]);
        assert_eq!(words(&cfg, "a\u{3014}b"), vec!["a", "b"]);
    }

    #[test]
    fn asian_support_without_normalized_does_not_split() {
        // spec 3.3.1: the per-character split is gated on `normalized`.
        let cfg = TerConfig { asian_support: true, ..TerConfig::default() };
        assert_eq!(words(&cfg, "\u{4f60}\u{597d}\u{3002}\u{4e16}\u{754c}"),
                   vec!["\u{4f60}\u{597d}\u{3002}\u{4e16}\u{754c}"]);
    }

    #[test]
    fn asian_support_without_normalized_still_strips_fullwidth_punct() {
        // spec 3.3.2: the `no_punct` fullwidth set is not gated on `normalized`.
        let cfg = TerConfig { asian_support: true, no_punct: true, ..TerConfig::default() };
        assert_eq!(words(&cfg, "a\u{3002}b"), vec!["ab"]);
        let plain = TerConfig { no_punct: true, ..TerConfig::default() };
        assert_eq!(words(&plain, "a\u{3002}b"), vec!["a\u{3002}b"]);
    }

    #[test]
    fn a_shifted_block_costs_one_edit() {
        let cfg = TerConfig::default();
        // "b a" against "a b": no substitution helps, one block move fixes it.
        let h = words(&cfg, "b a");
        let r = words(&cfg, "a b");
        let ids = |toks: &[String]| -> Vec<u32> { toks.iter().map(|t| if t == "a" { 0 } else { 1 }).collect() };
        assert_eq!(translation_edit_rate(&ids(&h), &ids(&r)), (1, 2));
    }

    #[test]
    fn empty_reference_counts_hypothesis_length() {
        assert_eq!(translation_edit_rate(&[0, 1, 2], &[]), (3, 0));
    }

    #[test]
    fn no_references_is_an_error() {
        assert!(ter_segment(&TerConfig::default(), "a", &[]).is_err());
    }

    #[test]
    fn identical_segments_cost_nothing() {
        let cfg = TerConfig::default();
        assert_eq!(ter_segment(&cfg, "a b c", &["a b c"]).unwrap(), [0.0, 3.0]);
    }

    #[test]
    fn blocks_up_to_twenty_words_shift_as_one_edit() {
        // Established by measurement against the oracle for these lengths. Beyond about 20 words a
        // single block move exceeds the candidate budget before it is found, which is a known
        // limitation recorded in the lab notebook rather than something this test should hide.
        let cfg = TerConfig::default();
        for n in [2usize, 5, 10, 11, 15, 20] {
            let blk = vec!["w"; n].join(" ");
            let hyp = format!("{blk} p q r");
            let refs = format!("p q r {blk}");
            let got = ter_segment(&cfg, &hyp, &[&refs]).unwrap();
            assert_eq!(got[0], 1.0, "a {n}-word block should be a single shift");
        }
    }
}