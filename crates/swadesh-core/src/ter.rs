//! Bit-exact port of sacreBLEU 2.6.0 TER:
//!
//! * `metrics/ter.py`: segment statistics (`_compute_segment_statistics`)
//! * `metrics/lib_ter.py`: beam edit distance with tercom-style shifts
//! * `tokenizers/tokenizer_ter.py`: normalization, punctuation, asian support
//!
//! The control flow, tie-breaking, iteration order and constants follow the
//! Python source line by line. Two facts about that source make some of the
//! structure safe to rely on, and both are checked by the differential tests:
//!
//! 1. Every hypothesis evaluated during one `translation_edit_rate` call has the
//!    same length (shifts permute words), so the beam width is constant within a
//!    call and a cached row depends only on its word prefix. The prefix cache is
//!    therefore purely an optimisation. It is still ported literally, including
//!    its size limit.
//! 2. Every reference index occurs in exactly one NOP/SUB/DEL op of the trace, so
//!    sacreBLEU's `align` dict is a dense vector indexed by reference position.
//!
//! Regexes in the Python tokenizer are either single-character class
//! substitutions, fixed two-character substitutions, literal replacements, or
//! anchored at the string boundary. They are emulated with explicit code-point
//! classes (sacreBLEU's own ranges, not Unicode properties), so no Unicode
//! tables are needed.

use std::collections::HashMap;

use crate::unicode::Interner;
use crate::unicode::py_split;

// ---------------------------------------------------------------- constants

// lib_ter.py
const COST_INS: i64 = 1;
const COST_DEL: i64 = 1;
const COST_SUB: i64 = 1;
const MAX_SHIFT_SIZE: usize = 10;
const MAX_SHIFT_DIST: i64 = 50;
const BEAM_WIDTH: i64 = 25;
const MAX_CACHE_SIZE: usize = 10_000;
const MAX_SHIFT_CANDIDATES: usize = 1_000;
/// `int(1e16)`
const INT_INFINITY: i64 = 10_000_000_000_000_000;

const OP_INS: u8 = b'i';
const OP_DEL: u8 = b'd';
const OP_NOP: u8 = b' ';
const OP_SUB: u8 = b's';
const OP_UNDEF: u8 = b'x';

/// `(cost, op)` cell of the edit-distance matrix.
#[derive(Clone, Copy, Debug)]
struct Cell {
    cost: i64,
    op: u8,
}

const EMPTY_CELL: Cell = Cell { cost: INT_INFINITY, op: OP_UNDEF };

// ------------------------------------------------------------ public types

/// The `TercomTokenizer` options. `lowercase` is `not case_sensitive`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerConfig {
    pub normalized: bool,
    pub no_punct: bool,
    pub asian_support: bool,
    pub case_sensitive: bool,
}

/// sacreBLEU raises `ZeroDivisionError` when a segment has no references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerError {
    NoReferences,
}

// ------------------------------------------------------------ char helpers

#[inline]
fn in_ranges(c: char, ranges: &[(u32, u32)]) -> bool {
    let cp = c as u32;
    ranges.iter().any(|&(a, b)| a <= cp && cp <= b)
}

/// `([{-~[-` -&(-+:-@/])` from `_normalize_general_and_western`.
#[inline]
fn is_13a_class(c: char) -> bool {
    matches!(c, '{'..='~' | '['..='`' | ' '..='&' | '('..='+' | ':'..='@' | '/')
}

/// `[\.,]`
#[inline]
fn is_period_or_comma(c: char) -> bool {
    c == '.' || c == ','
}

/// `re.sub(r"(X)", r" \1 ", s)` for a single-character class `X`.
fn sub_wrap(input: &[char], out: &mut Vec<char>, m: impl Fn(char) -> bool) {
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

/// `re.sub` for a fixed two-character pattern, scanning left to right with
/// non-overlapping matches (identical to Python's `re.sub`).
fn sub_pair(
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

/// `re.sub(r"(X)", "", s)` for a single-character class `X`.
fn delete_class(input: &[char], out: &mut Vec<char>, m: impl Fn(char) -> bool) {
    out.clear();
    out.extend(input.iter().copied().filter(|&c| !m(c)));
}

// ------------------------------------------------------------- tokenizer

/// `_normalize_general_and_western`.
fn normalize_general_and_western(sent: &str) -> String {
    // strip end-of-line hyphenation and join lines
    let s = sent.replace("\n-", "");
    let s = s.replace('\n', " ");
    // handle XML escaped symbols
    let s = s.replace("&quot;", "\"").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">");
    // language-dependent (Western) part
    let s = format!(" {s} ");

    // tokenize punctuation
    let chars: Vec<char> = s.chars().collect();
    let mut a = Vec::with_capacity(chars.len() * 2);
    sub_wrap(&chars, &mut a, is_13a_class);

    // handle possessives: re.sub(r"'s ", r" 's ") then re.sub(r"'s$", r" 's")
    let s: String = a.into_iter().collect();
    let mut s = s.replace("'s ", " 's ");
    // `$` is end-of-string here: the text contains no newline at this point.
    if s.ends_with("'s") {
        s.truncate(s.len() - 2);
        s.push_str(" 's");
    }
    let chars: Vec<char> = s.chars().collect();

    // tokenize period and comma unless preceded by a digit: ([^0-9])([\.,]) -> \1 \2
    let mut b = Vec::with_capacity(chars.len() * 2);
    sub_pair(
        &chars,
        &mut b,
        |x, y| !x.is_ascii_digit() && is_period_or_comma(y),
        |x, y, o| {
            o.push(x);
            o.push(' ');
            o.push(y);
            o.push(' ');
        },
    );

    // tokenize period and comma unless followed by a digit: ([\.,])([^0-9]) -> " \1 \2"
    let mut c = Vec::with_capacity(b.len() * 2);
    sub_pair(
        &b,
        &mut c,
        |x, y| is_period_or_comma(x) && !y.is_ascii_digit(),
        |x, y, o| {
            o.push(' ');
            o.push(x);
            o.push(' ');
            o.push(y);
        },
    );

    // tokenize dash when preceded by a digit: ([0-9])(-) -> \1 \2
    let mut d = Vec::with_capacity(c.len() + 8);
    sub_pair(
        &c,
        &mut d,
        |x, y| x.is_ascii_digit() && y == '-',
        |x, y, o| {
            o.push(x);
            o.push(' ');
            o.push(y);
            o.push(' ');
        },
    );
    d.into_iter().collect()
}

const CJK_IDEO_1: &[(u32, u32)] = &[(0x4E00, 0x9FFF), (0x3400, 0x4DBF)];
const CJK_STROKE: &[(u32, u32)] = &[(0x31C0, 0x31EF), (0x2E80, 0x2EFF)];
const CJK_COMPAT: &[(u32, u32)] = &[(0x3300, 0x33FF), (0xF900, 0xFAFF), (0xFE30, 0xFE4F)];
const CJK_ENCLOSED: &[(u32, u32)] = &[(0x3200, 0x3F22)];
const HIRAGANA: (u32, u32) = (0x3040, 0x309F);
const KATAKANA: (u32, u32) = (0x30A0, 0x30FF);
const KATAKANA_PHONETIC: (u32, u32) = (0x31F0, 0x31FF);
/// `TercomTokenizer.ASIAN_PUNCT`
const ASIAN_PUNCT: &[(u32, u32)] = &[
    (0x3001, 0x3001),
    (0x3002, 0x3002),
    (0x3008, 0x3011),
    (0x3014, 0x301F),
    (0xFF61, 0xFF65),
    (0x30FB, 0x30FB),
];
/// `TercomTokenizer.FULL_WIDTH_PUNCT`
const FULL_WIDTH_PUNCT: &[(u32, u32)] = &[
    (0xFF0E, 0xFF0E),
    (0xFF0C, 0xFF0C),
    (0xFF1F, 0xFF1F),
    (0xFF1A, 0xFF1A),
    (0xFF1B, 0xFF1B),
    (0xFF01, 0xFF01),
    (0xFF02, 0xFF02),
    (0xFF08, 0xFF08),
    (0xFF09, 0xFF09),
];

/// `re.sub(r"(^|^[R])([R]+)(?=$|^[R])", r"\1 \2 ", sent)`.
/// Both alternatives are anchored at position 0 and the lookahead needs the end of
/// the string, so this only fires when the whole string lies in the range.
fn wrap_whole_string_in(sent: String, (lo, hi): (u32, u32)) -> String {
    if !sent.is_empty() && sent.chars().all(|c| (lo..=hi).contains(&(c as u32))) {
        format!(" {sent} ")
    } else {
        sent
    }
}

/// `_normalize_asian`.
fn normalize_asian(sent: &str) -> String {
    let mut s: Vec<char> = sent.chars().collect();
    let mut out = Vec::with_capacity(s.len() * 2);

    for ranges in [CJK_IDEO_1, CJK_STROKE, CJK_COMPAT, CJK_ENCLOSED] {
        sub_wrap(&s, &mut out, |c| in_ranges(c, ranges));
        std::mem::swap(&mut s, &mut out);
    }

    let mut t: String = s.into_iter().collect();
    t = wrap_whole_string_in(t, HIRAGANA);
    t = wrap_whole_string_in(t, KATAKANA);
    t = wrap_whole_string_in(t, KATAKANA_PHONETIC);

    let s: Vec<char> = t.chars().collect();
    let mut out = Vec::with_capacity(s.len() * 2);
    sub_wrap(&s, &mut out, |c| in_ranges(c, ASIAN_PUNCT));
    let s = out;
    let mut out = Vec::with_capacity(s.len() * 2);
    sub_wrap(&s, &mut out, |c| in_ranges(c, FULL_WIDTH_PUNCT));
    out.into_iter().collect()
}

/// `_remove_punct`: `re.sub(r"[\.,\?:;!\"\(\)]", "", sent)`.
fn remove_punct(sent: &str) -> String {
    sent.chars().filter(|c| !matches!(c, '.' | ',' | '?' | ':' | ';' | '!' | '"' | '(' | ')')).collect()
}

/// `_remove_asian_punct`.
fn remove_asian_punct(sent: &str) -> String {
    let chars: Vec<char> = sent.chars().collect();
    let mut out = Vec::with_capacity(chars.len());
    delete_class(&chars, &mut out, |c| in_ranges(c, ASIAN_PUNCT));
    let mut out2 = Vec::with_capacity(out.len());
    delete_class(&out, &mut out2, |c| in_ranges(c, FULL_WIDTH_PUNCT));
    out2.into_iter().collect()
}

/// `TercomTokenizer.__call__` minus the lowercasing, which the caller performs
/// on the right-stripped sentence (the Python `_preprocess_segment` order).
/// Returns the tokens joined by single spaces (`' '.join(sent.split())`).
pub fn tercom_tokenize(cfg: &TerConfig, sent: &str, out: &mut String) {
    out.clear();
    if sent.is_empty() {
        return;
    }
    let mut s: String = sent.to_string();
    if cfg.normalized {
        s = normalize_general_and_western(&s);
        if cfg.asian_support {
            s = normalize_asian(&s);
        }
    }
    if cfg.no_punct {
        s = remove_punct(&s);
        if cfg.asian_support {
            s = remove_asian_punct(&s);
        }
    }
    let mut pending = false;
    for tok in py_split(&s) {
        if pending {
            out.push(' ');
        }
        out.push_str(tok);
        pending = true;
    }
}

// --------------------------------------------------------- beam edit distance

struct Node {
    children: HashMap<u32, usize>,
    row: Vec<Cell>,
}

/// `BeamEditDistance` for one reference, with its prefix cache.
struct BeamEdit {
    words_ref: Vec<u32>,
    initial: Vec<Cell>,
    /// `nodes[0]` is the root of the cache trie (no row).
    nodes: Vec<Node>,
    cache_size: usize,
}

/// Row `i` of the matrix: the initial row, a cached row, or a freshly computed one.
#[inline]
fn row_at<'a>(
    initial: &'a [Cell],
    nodes: &'a [Node],
    path: &[usize],
    fresh: &'a [Vec<Cell>],
    start: usize,
    i: usize,
) -> &'a [Cell] {
    if i == 0 {
        initial
    } else if i <= start {
        &nodes[path[i - 1]].row
    } else {
        &fresh[i - start - 1]
    }
}

impl BeamEdit {
    fn new(words_ref: &[u32]) -> Self {
        let m = words_ref.len();
        // first row corresponds to insertion operations of the reference
        let initial = (0..=m).map(|i| Cell { cost: i as i64 * COST_INS, op: OP_INS }).collect();
        BeamEdit {
            words_ref: words_ref.to_vec(),
            initial,
            nodes: vec![Node { children: HashMap::new(), row: Vec::new() }],
            cache_size: 0,
        }
    }

    /// `BeamEditDistance.__call__`: returns `(edit_distance, trace)`. The trace
    /// is only materialised when `want_trace` is set; the distance is the same
    /// either way.
    fn eval(&mut self, words_h: &[u32], want_trace: bool) -> (i64, Vec<u8>) {
        let n_h = words_h.len();
        let m = self.words_ref.len();

        // _find_cache
        let mut path: Vec<usize> = Vec::new();
        let mut node = 0usize;
        for w in words_h {
            match self.nodes[node].children.get(w) {
                Some(&c) => {
                    path.push(c);
                    node = c;
                }
                None => break,
            }
        }
        let start = path.len();

        // _edit_distance
        let length_ratio = if n_h > 0 { m as f64 / n_h as f64 } else { 1.0 };
        let beam_width = if (BEAM_WIDTH as f64) < length_ratio / 2.0 {
            (length_ratio / 2.0 + BEAM_WIDTH as f64).ceil() as i64
        } else {
            BEAM_WIDTH
        };

        let mut fresh: Vec<Vec<Cell>> = Vec::with_capacity(n_h - start);
        for i in start + 1..=n_h {
            let pseudo_diag = (i as f64 * length_ratio).floor() as i64;
            let min_j = (pseudo_diag - beam_width).max(0);
            let mut max_j = (pseudo_diag + beam_width).min(m as i64 + 1);
            if i == n_h {
                max_j = m as i64 + 1;
            }

            let mut cur = vec![EMPTY_CELL; m + 1];
            {
                let prev = row_at(&self.initial, &self.nodes, &path, &fresh, start, i - 1);
                for j in min_j..max_j {
                    let ju = j as usize;
                    if j == 0 {
                        cur[0] = Cell { cost: prev[0].cost + COST_DEL, op: OP_DEL };
                    } else {
                        let (cost_sub, op_sub) = if words_h[i - 1] == self.words_ref[ju - 1] {
                            (0, OP_NOP)
                        } else {
                            (COST_SUB, OP_SUB)
                        };
                        // Tercom prefers no-op/sub, then insertion, then deletion
                        // (here: sub, del, ins; see lib_ter.py).
                        let ops = [
                            (prev[ju - 1].cost + cost_sub, op_sub),
                            (prev[ju].cost + COST_DEL, OP_DEL),
                            (cur[ju - 1].cost + COST_INS, OP_INS),
                        ];
                        for (op_cost, op_name) in ops {
                            if cur[ju].cost > op_cost {
                                cur[ju] = Cell { cost: op_cost, op: op_name };
                            }
                        }
                    }
                }
            }
            fresh.push(cur);
        }

        let cost = row_at(&self.initial, &self.nodes, &path, &fresh, start, n_h)[m].cost;

        // get the trace
        let mut trace: Vec<u8> = Vec::new();
        if want_trace {
            let (mut i, mut j) = (n_h, m);
            while i > 0 || j > 0 {
                let op = row_at(&self.initial, &self.nodes, &path, &fresh, start, i)[j].op;
                trace.push(op);
                match op {
                    OP_SUB | OP_NOP => {
                        i -= 1;
                        j -= 1;
                    }
                    OP_INS => j -= 1,
                    OP_DEL => i -= 1,
                    _ => panic!("swadesh: unknown TER edit operation {:?}", op as char),
                }
            }
            trace.reverse();
        }

        // _add_cache
        if self.cache_size < MAX_CACHE_SIZE {
            let mut node = if start > 0 { path[start - 1] } else { 0 };
            for (k, &w) in words_h[start..].iter().enumerate() {
                let child = match self.nodes[node].children.get(&w) {
                    Some(&c) => c,
                    None => {
                        let row = std::mem::take(&mut fresh[k]);
                        self.nodes.push(Node { children: HashMap::new(), row });
                        let idx = self.nodes.len() - 1;
                        self.nodes[node].children.insert(w, idx);
                        self.cache_size += 1;
                        idx
                    }
                };
                node = child;
            }
        }

        (cost, trace)
    }
}

// ----------------------------------------------------------------- shifts

/// Python slice `words[a:b]` with clamping.
#[inline]
fn py_slice(words: &[u32], a: usize, b: usize) -> &[u32] {
    let n = words.len();
    let a = a.min(n);
    let b = b.min(n);
    if b > a {
        &words[a..b]
    } else {
        &[]
    }
}

/// `_perform_shift`.
fn perform_shift(words: &[u32], start: usize, length: usize, target: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(words.len());
    if target < start {
        // shift before previous position
        out.extend_from_slice(py_slice(words, 0, target));
        out.extend_from_slice(py_slice(words, start, start + length));
        out.extend_from_slice(py_slice(words, target, start));
        out.extend_from_slice(py_slice(words, start + length, usize::MAX));
    } else if target > start + length {
        // shift after previous position
        out.extend_from_slice(py_slice(words, 0, start));
        out.extend_from_slice(py_slice(words, start + length, target));
        out.extend_from_slice(py_slice(words, start, start + length));
        out.extend_from_slice(py_slice(words, target, usize::MAX));
    } else {
        // shift within the shifted string
        out.extend_from_slice(py_slice(words, 0, start));
        out.extend_from_slice(py_slice(words, start + length, length + target));
        out.extend_from_slice(py_slice(words, start, start + length));
        out.extend_from_slice(py_slice(words, length + target, usize::MAX));
    }
    out
}

/// `_find_shifted_pairs`: `(start_h, start_r, length)` in Python's yield order.
fn find_shifted_pairs(words_h: &[u32], words_r: &[u32]) -> Vec<(usize, usize, usize)> {
    let (n_h, n_r) = (words_h.len(), words_r.len());
    let mut out = Vec::new();
    for start_h in 0..n_h {
        for start_r in 0..n_r {
            // this is slightly different from what tercom does
            if (start_r as i64 - start_h as i64).abs() > MAX_SHIFT_DIST {
                continue;
            }
            let mut length = 0usize;
            while words_h[start_h + length] == words_r[start_r + length] && length < MAX_SHIFT_SIZE {
                length += 1;
                out.push((start_h, start_r, length));
                // If one sequence is consumed, stop processing
                if n_h == start_h + length || n_r == start_r + length {
                    break;
                }
            }
        }
    }
    out
}

/// `_flip_trace`: swap insertions and deletions.
fn flip_trace(trace: &[u8]) -> Vec<u8> {
    trace
        .iter()
        .map(|&op| match op {
            OP_INS => OP_DEL,
            OP_DEL => OP_INS,
            other => other,
        })
        .collect()
}

/// `trace_to_alignment`: `(align, ref_err, hyp_err)`. `align` has one entry per
/// reference word (sacreBLEU's dict keys are exactly `0..n_ref`).
fn trace_to_alignment(trace: &[u8]) -> (Vec<i64>, Vec<u32>, Vec<u32>) {
    let mut pos_hyp: i64 = -1;
    let mut hyp_err = Vec::new();
    let mut ref_err = Vec::new();
    let mut align: Vec<i64> = Vec::new();
    for &op in trace {
        match op {
            OP_NOP | OP_SUB => {
                pos_hyp += 1;
                align.push(pos_hyp);
                let e = u32::from(op == OP_SUB);
                hyp_err.push(e);
                ref_err.push(e);
            }
            OP_INS => {
                pos_hyp += 1;
                hyp_err.push(1);
            }
            OP_DEL => {
                align.push(pos_hyp);
                ref_err.push(1);
            }
            other => panic!("swadesh: unknown TER trace operation {:?}", other as char),
        }
    }
    (align, ref_err, hyp_err)
}

/// Best shift candidate, ranked by `(score, length, -start_h, -idx)`.
struct BestShift {
    score: i64,
    length: usize,
    neg_start_h: i64,
    neg_idx: i64,
    words: Vec<u32>,
}

/// `_shift`: returns `(best_score, shifted_words)` and updates `checked`.
fn shift(words_h: &[u32], words_r: &[u32], ed: &mut BeamEdit, checked: &mut usize) -> (i64, Vec<u32>) {
    let (pre_score, inv_trace) = ed.eval(words_h, true);

    // to get alignment, we pretend we are rewriting reference into hypothesis
    let trace = flip_trace(&inv_trace);
    let (align, ref_err, hyp_err) = trace_to_alignment(&trace);

    let mut best: Option<BestShift> = None;

    for (start_h, start_r, length) in find_shifted_pairs(words_h, words_r) {
        // don't do the shift unless both the hypothesis was wrong and the
        // reference doesn't match hypothesis at the target position
        if hyp_err[start_h..start_h + length].iter().sum::<u32>() == 0 {
            continue;
        }
        if ref_err[start_r..start_r + length].iter().sum::<u32>() == 0 {
            continue;
        }
        // don't try to shift within the subsequence
        let a = align[start_r];
        if (start_h as i64) <= a && a < (start_h + length) as i64 {
            continue;
        }

        let mut prev_idx: i64 = -1;
        for offset in -1i64..length as i64 {
            let pos = start_r as i64 + offset;
            let idx: i64 = if pos == -1 {
                0 // insert before the beginning
            } else if pos >= 0 && (pos as usize) < align.len() {
                // inserted *before* the aligned index
                align[pos as usize] + 1
            } else {
                break; // offset is out of bounds => aims past reference
            };

            if idx == prev_idx {
                continue; // skip idx if already tried
            }
            prev_idx = idx;

            let shifted = perform_shift(words_h, start_h, length, idx as usize);
            let (cost, _) = ed.eval(&shifted, false);
            let score = pre_score - cost;
            *checked += 1;

            // Python compares tuples (score, length, -start_h, -idx, words). Equal
            // first four entries imply identical `words`, so the strict comparison
            // of the first four gives the same winner.
            let cand = (score, length, -(start_h as i64), -idx);
            let better = match &best {
                None => true,
                Some(b) => cand > (b.score, b.length, b.neg_start_h, b.neg_idx),
            };
            if better {
                best = Some(BestShift {
                    score,
                    length,
                    neg_start_h: -(start_h as i64),
                    neg_idx: -idx,
                    words: shifted,
                });
            }
        }

        if *checked >= MAX_SHIFT_CANDIDATES {
            break;
        }
    }

    match best {
        None => (0, words_h.to_vec()),
        Some(b) => (b.score, b.words),
    }
}

/// `translation_edit_rate`: `(num_edits, ref_len)`.
pub fn translation_edit_rate(words_h: &[u32], words_r: &[u32]) -> (i64, usize) {
    let n_r = words_r.len();
    let n_h = words_h.len();
    if n_r == 0 {
        // special treatment of empty refs
        return (n_h as i64, 0);
    }

    let mut ed = BeamEdit::new(words_r);
    let mut shifts: i64 = 0;
    let mut input_words = words_h.to_vec();
    let mut checked_candidates = 0usize;
    loop {
        // do shifts until they stop reducing the edit distance
        let (delta, new_input_words) = shift(&input_words, words_r, &mut ed, &mut checked_candidates);
        if checked_candidates >= MAX_SHIFT_CANDIDATES {
            break;
        }
        if delta <= 0 {
            break;
        }
        shifts += 1;
        input_words = new_input_words;
    }

    let (edit_distance, _) = ed.eval(&input_words, false);
    (shifts + edit_distance, n_r)
}

// ---------------------------------------------------------- segment level

/// `TER._compute_segment_statistics` for one (already lowercased, rstripped)
/// hypothesis and its references: `[best_num_edits, avg_ref_len]`.
///
/// Returns `Err(NoReferences)` when there are no references, where sacreBLEU
/// raises `ZeroDivisionError`.
pub fn ter_segment(cfg: &TerConfig, hyp: &str, refs: &[&str]) -> Result<[f64; 2], TerError> {
    if refs.is_empty() {
        return Err(TerError::NoReferences);
    }
    let mut buf = String::new();
    let mut ref_toks: Vec<String> = Vec::with_capacity(refs.len());
    for r in refs {
        // sacreBLEU 2.6.0 tokenizes each reference twice: `_cache_references`
        // preprocesses it, then `TER._extract_reference_info` preprocesses the
        // result again. The tokenizer is not idempotent ("'s" is split again), so
        // this must be replicated. The second lowercasing is a no-op because
        // `str.lower()` is idempotent (verified over all code points).
        tercom_tokenize(cfg, r, &mut buf);
        let once = std::mem::take(&mut buf);
        tercom_tokenize(cfg, &once, &mut buf);
        ref_toks.push(std::mem::take(&mut buf));
    }
    tercom_tokenize(cfg, hyp, &mut buf);
    let hyp_tok = buf;

    let mut interner = Interner::default();
    let hyp_ids: Vec<u32> = py_split(&hyp_tok).map(|t| interner.id(t)).collect();
    let ref_ids: Vec<Vec<u32>> = ref_toks
        .iter()
        .map(|t| py_split(t).map(|w| interner.id(w)).collect())
        .collect();

    let mut ref_lengths: i64 = 0;
    let mut best_num_edits = INT_INFINITY;
    for words_ref in &ref_ids {
        let (num_edits, ref_len) = translation_edit_rate(&hyp_ids, words_ref);
        ref_lengths += ref_len as i64;
        if num_edits < best_num_edits {
            best_num_edits = num_edits;
        }
    }
    let avg_ref_len = ref_lengths as f64 / ref_ids.len() as f64;
    Ok([best_num_edits as f64, avg_ref_len])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(cfg: TerConfig, s: &str) -> String {
        let mut o = String::new();
        tercom_tokenize(&cfg, s, &mut o);
        o
    }

    fn stats(hyp: &str, refs: &[&str]) -> [f64; 2] {
        ter_segment(&TerConfig::default(), hyp, refs).unwrap()
    }

    #[test]
    fn tokenizer_basics() {
        let norm = TerConfig { normalized: true, ..Default::default() };
        assert_eq!(tok(norm, "Hello, world."), "Hello , world .");
        assert_eq!(tok(norm, "It costs $1,000.50!"), "It costs $ 1,000.50 !");
        assert_eq!(tok(norm, "1990-1995"), "1990 - 1995");
        assert_eq!(tok(norm, "the dog's bone"), "the dog 's bone");
        let np = TerConfig { no_punct: true, ..Default::default() };
        assert_eq!(tok(np, "a, b (c)!"), "a b c");
    }

    #[test]
    fn edit_counts() {
        assert_eq!(stats("a b c", &["a b c"]), [0.0, 3.0]);
        // one shift of "a" to the front: 1 edit (sacreBLEU gives 1 edit, ref length 3)
        assert_eq!(stats("b c a", &["a b c"]), [1.0, 3.0]);
        // empty hypothesis: one deletion per reference word
        assert_eq!(stats("", &["a b c"]), [3.0, 3.0]);
        // empty reference: one insertion per hypothesis word
        assert_eq!(stats("a b", &[""]), [2.0, 0.0]);
    }

    #[test]
    fn no_references_is_error() {
        assert_eq!(ter_segment(&TerConfig::default(), "a", &[]), Err(TerError::NoReferences));
    }
}

#[cfg(test)]
mod sacrebleu_tests {
    use super::*;

    /// Segment statistics from sacreBLEU 2.6.0 `TER._compute_segment_statistics` on generated
    /// cases: shifts (distance > 50, repeated words, permutations), beam-width ratios from
    /// very different lengths, multiple references and empty references.
    #[test]
    fn ter_matches_sacrebleu_on_generated_cases() {
        for (k, line) in include_str!("../tests/fixtures/mut_ter.txt").lines().enumerate() {
            let c: Vec<&str> = line.split('\t').collect();
            let edits: f64 = c[0].parse().unwrap();
            let avg: f64 = c[1].parse().unwrap();
            let hyp = c[2];
            let refs: Vec<&str> = c[3..].to_vec();
            let got = ter_segment(&TerConfig::default(), hyp, &refs).unwrap();
            assert_eq!(got, [edits, avg], "case {k}: hyp {hyp:?} refs {refs:?}");
        }
    }
}
