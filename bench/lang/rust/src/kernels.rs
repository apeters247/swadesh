//! Rust implementation of the swadesh kernels. Algorithms mirror
//! `../c/langbench.c` line for line so that checksums must match exactly.
//!
//! Two flavours:
//! * default: reusable scratch buffers (what an optimised port would do);
//! * `*_alloc`: the allocation pattern of the current `swadesh-core` code
//!   (fresh Vec/String per call), kept as an ablation.

use std::mem::swap;

pub type Run = (u128, u32);

/// Python `str.isspace()`: the set used by `str.split()` / `str.strip()`.
#[inline]
pub fn is_py_space(c: char) -> bool {
    match c as u32 {
        0x09..=0x0d | 0x1c..=0x1f | 0x20 => true,
        0x85 | 0xa0 | 0x1680 | 0x2000..=0x200a | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000 => true,
        _ => false,
    }
}

#[inline]
fn is_13a_class1(c: char) -> bool {
    matches!(c, '{'..='~' | '['..='`' | ' '..='&' | '('..='+' | ':'..='@' | '/')
}

/// Replace all occurrences of `pat` by `rep`, writing into `dst` (cleared first).
fn replace_into(src: &str, pat: &str, rep: &str, dst: &mut String) {
    dst.clear();
    let mut last = 0;
    for (i, m) in src.match_indices(pat) {
        dst.push_str(&src[last..i]);
        dst.push_str(rep);
        last = i + m.len();
    }
    dst.push_str(&src[last..]);
}

/// Reusable buffers for one worker thread.
#[derive(Default)]
pub struct Scratch<'a> {
    a: Vec<char>,
    b: Vec<char>,
    c: Vec<char>,
    s1: String,
    s2: String,
    ba: Vec<u8>,
    bb: Vec<u8>,
    bc: Vec<u8>,
    pub ob: Vec<u8>,
    cph: Vec<u32>,
    cpr: Vec<u32>,
    hid: Vec<u32>,
    rid: Vec<u32>,
    it: Vec<(&'a str, u32)>,
    kh: Vec<u128>,
    kr: Vec<u128>,
    rh: Vec<Run>,
    rr: Vec<Run>,
    ht_hk: Vec<u128>,
    ht_hv: Vec<u32>,
    ht_rk: Vec<u128>,
    ht_rv: Vec<u32>,
}

/// sacreBLEU 13a pre-steps; the result is left in `s.s1`.
fn prep13a(line: &str, s: &mut Scratch) {
    s.s1.clear();
    s.s1.push_str(line);
    // Each rule runs only when its pattern occurs.
    for (pat, rep) in [
        ("<skipped>", ""),
        ("-\n", ""),
        ("\n", " "),
        ("&quot;", "\""),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
    ] {
        if s.s1.contains(pat) {
            replace_into(&s.s1, pat, rep, &mut s.s2);
            swap(&mut s.s1, &mut s.s2);
        }
    }
}

/// sacreBLEU 13a tokeniser. Output bytes go to `s.ob`; returns their length.
pub fn tok13a(line: &str, s: &mut Scratch) -> usize {
    prep13a(line, s);
    s.a.clear();
    s.a.push(' ');
    s.a.extend(s.s1.chars());
    s.a.push(' ');

    let (a, b, c) = (&s.a, &mut s.b, &mut s.c);
    // (\s-free) class substitution X -> " X "
    b.clear();
    for &x in a.iter() {
        if is_13a_class1(x) {
            b.push(' ');
            b.push(x);
            b.push(' ');
        } else {
            b.push(x);
        }
    }
    let isdig = |x: char| x.is_ascii_digit();
    // ([^0-9])([\.,]) -> \1 \2
    c.clear();
    sub2(b, c, |x, y| !isdig(x) && (y == '.' || y == ','), |x, y, o| {
        o.extend_from_slice(&[x, ' ', y, ' '])
    });
    // ([\.,])([^0-9]) -> _\1 \2
    sub2(c, b, |x, y| (x == '.' || x == ',') && !isdig(y), |x, y, o| {
        o.extend_from_slice(&[' ', x, ' ', y])
    });
    // ([0-9])(-) -> \1 \2
    sub2(b, c, |x, y| isdig(x) && y == '-', |x, y, o| o.extend_from_slice(&[x, ' ', y, ' ']));

    // ' '.join(text.split())
    s.ob.clear();
    let mut pending = false;
    let mut tmp = [0u8; 4];
    for &x in c.iter() {
        if is_py_space(x) {
            if !s.ob.is_empty() {
                pending = true;
            }
        } else {
            if pending {
                s.ob.push(b' ');
                pending = false;
            }
            s.ob.extend_from_slice(x.encode_utf8(&mut tmp).as_bytes());
        }
    }
    s.ob.len()
}

#[inline]
fn sub2(
    input: &[char],
    out: &mut Vec<char>,
    m: impl Fn(char, char) -> bool,
    emit: impl Fn(char, char, &mut Vec<char>),
) {
    out.clear();
    let mut i = 0;
    while i < input.len() {
        if i + 1 < input.len() && m(input[i], input[i + 1]) {
            emit(input[i], input[i + 1], out);
            i += 2;
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
}

/// ASCII fast path: the same passes on bytes instead of `char`s. Valid only
/// when every byte is ASCII (then the output is ASCII too), so no UTF-8
/// decode/encode and no 4-byte `char` per code point.
fn tok13a_bytes(s: &mut Scratch) -> usize {
    let txt = s.s1.as_bytes();
    let (a, b, c) = (&mut s.ba, &mut s.bb, &mut s.bc);
    a.clear();
    a.push(b' ');
    a.extend_from_slice(txt);
    a.push(b' ');
    b.clear();
    for &x in a.iter() {
        if class1_b(x) {
            b.extend_from_slice(&[b' ', x, b' ']);
        } else {
            b.push(x);
        }
    }
    sub2_b(b, c, |x, y| !x.is_ascii_digit() && (y == b'.' || y == b','), |x, y, o| {
        o.extend_from_slice(&[x, b' ', y, b' '])
    });
    sub2_b(c, b, |x, y| (x == b'.' || x == b',') && !y.is_ascii_digit(), |x, y, o| {
        o.extend_from_slice(&[b' ', x, b' ', y])
    });
    sub2_b(b, c, |x, y| x.is_ascii_digit() && y == b'-', |x, y, o| o.extend_from_slice(&[x, b' ', y, b' ']));
    let ob = &mut s.ob;
    ob.clear();
    let mut pending = false;
    for &x in c.iter() {
        if matches!(x, 0x09..=0x0d | 0x1c..=0x1f | 0x20) {
            if !ob.is_empty() {
                pending = true;
            }
        } else {
            if pending {
                ob.push(b' ');
                pending = false;
            }
            ob.push(x);
        }
    }
    ob.len()
}

#[inline]
fn class1_b(c: u8) -> bool {
    matches!(c, b'{'..=b'~' | b'['..=b'`' | b' '..=b'&' | b'('..=b'+' | b':'..=b'@' | b'/')
}

#[inline]
fn sub2_b(
    input: &[u8],
    out: &mut Vec<u8>,
    m: impl Fn(u8, u8) -> bool,
    emit: impl Fn(u8, u8, &mut Vec<u8>),
) {
    out.clear();
    let mut i = 0;
    while i < input.len() {
        if i + 1 < input.len() && m(input[i], input[i + 1]) {
            emit(input[i], input[i + 1], out);
            i += 2;
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
}

/// `tok13a` with the ASCII fast path. Output is identical to `tok13a`.
pub fn tok13a_fast(line: &str, s: &mut Scratch) -> usize {
    if line.is_ascii() {
        prep13a(line, s);
        tok13a_bytes(s)
    } else {
        tok13a(line, s)
    }
}

/// Sort + run-length encode, in place into `rk` (reused buffer).
#[inline]
fn rle_sorted_into(keys: &mut [u128], out: &mut Vec<Run>) {
    keys.sort_unstable();
    out.clear();
    for &k in keys.iter() {
        match out.last_mut() {
            Some((last, c)) if *last == k => *c += 1,
            _ => out.push((k, 1)),
        }
    }
}

#[inline]
fn pack(p: &[u32], bits: u32) -> u128 {
    let mut k: u128 = 0;
    for &x in p {
        k = (k << bits) | x as u128;
    }
    k
}

#[inline]
fn mix128(k: u128) -> u64 {
    let mut x = (k as u64) ^ ((k >> 64) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    x
}

fn ht_reset(k: &mut Vec<u128>, v: &mut Vec<u32>, need: usize) -> usize {
    let mut sz = 16;
    while sz < 2 * need {
        sz <<= 1;
    }
    k.clear();
    k.resize(sz, 0);
    v.clear();
    v.resize(sz, 0);
    sz
}

#[inline]
fn ht_add(k: &mut [u128], v: &mut [u32], key: u128) {
    let m = k.len() - 1;
    let mut i = mix128(key) as usize & m;
    while v[i] != 0 && k[i] != key {
        i = (i + 1) & m;
    }
    k[i] = key;
    v[i] += 1;
}

#[inline]
fn ht_get(k: &[u128], v: &[u32], key: u128) -> u32 {
    let m = k.len() - 1;
    let mut i = mix128(key) as usize & m;
    while v[i] != 0 {
        if k[i] == key {
            return v[i];
        }
        i = (i + 1) & m;
    }
    0
}

/// Statistics for orders 1..=maxn; checksum contribution as in the C version.
pub fn ngram_stats(h: &[u32], r: &[u32], maxn: usize, bits: u32, use_hash: bool, s: &mut Scratch) -> u64 {
    let mut chk: u64 = 0;
    for n in 1..=maxn {
        let nh = if h.len() >= n { h.len() - n + 1 } else { 0 };
        let nr = if r.len() >= n { r.len() - n + 1 } else { 0 };
        let mut m: u64 = 0;
        if use_hash {
            let szh = ht_reset(&mut s.ht_hk, &mut s.ht_hv, nh);
            let szr = ht_reset(&mut s.ht_rk, &mut s.ht_rv, nr);
            let _ = (szh, szr);
            for i in 0..nh {
                ht_add(&mut s.ht_hk, &mut s.ht_hv, pack(&h[i..i + n], bits));
            }
            for i in 0..nr {
                ht_add(&mut s.ht_rk, &mut s.ht_rv, pack(&r[i..i + n], bits));
            }
            for j in 0..s.ht_rk.len() {
                if s.ht_rv[j] != 0 {
                    let hv = ht_get(&s.ht_hk, &s.ht_hv, s.ht_rk[j]);
                    m += hv.min(s.ht_rv[j]) as u64;
                }
            }
        } else {
            s.kh.clear();
            s.kh.extend(h.windows(n).map(|w| pack(w, bits)));
            s.kr.clear();
            s.kr.extend(r.windows(n).map(|w| pack(w, bits)));
            rle_sorted_into(&mut s.kh, &mut s.rh);
            rle_sorted_into(&mut s.kr, &mut s.rr);
            let (mut i, mut j) = (0, 0);
            while i < s.rh.len() && j < s.rr.len() {
                let (a, b) = (s.rh[i].0, s.rr[j].0);
                if a < b {
                    i += 1;
                } else if a > b {
                    j += 1;
                } else {
                    m += s.rh[i].1.min(s.rr[j].1) as u64;
                    i += 1;
                    j += 1;
                }
            }
        }
        chk = chk.wrapping_add((n as u64).wrapping_mul(m * 1_000_003 + nh as u64 * 97 + nr as u64));
    }
    chk
}

/// chrF character n-grams, orders 1..=6, whitespace removed.
pub fn chr_pair(hyp: &str, rf: &str, use_hash: bool, s: &mut Scratch) -> u64 {
    s.cph.clear();
    s.cph.extend(hyp.chars().filter(|&c| !is_py_space(c)).map(|c| c as u32));
    s.cpr.clear();
    s.cpr.extend(rf.chars().filter(|&c| !is_py_space(c)).map(|c| c as u32));
    let (h, r) = (std::mem::take(&mut s.cph), std::mem::take(&mut s.cpr));
    let chk = ngram_stats(&h, &r, 6, 21, use_hash, s);
    s.cph = h;
    s.cpr = r;
    chk
}

/// BLEU word n-grams, orders 1..=4, on tokenised text.
pub fn wrd_pair<'a>(hyp: &'a str, rf: &'a str, use_hash: bool, s: &mut Scratch<'a>) -> u64 {
    let cap = hyp.len() + rf.len() + 2;
    let mut sz = 16;
    while sz < 2 * cap {
        sz <<= 1;
    }
    s.it.clear();
    s.it.resize(sz, ("", 0));
    let mut next = 0u32;
    s.hid.clear();
    for t in hyp.split(' ').filter(|t| !t.is_empty()) {
        let id = intern(&mut s.it, t, &mut next);
        s.hid.push(id);
    }
    s.rid.clear();
    for t in rf.split(' ').filter(|t| !t.is_empty()) {
        let id = intern(&mut s.it, t, &mut next);
        s.rid.push(id);
    }
    let (h, r) = (std::mem::take(&mut s.hid), std::mem::take(&mut s.rid));
    let chk = ngram_stats(&h, &r, 4, 32, use_hash, s);
    s.hid = h;
    s.rid = r;
    chk
}

#[inline]
fn intern<'a>(tab: &mut [(&'a str, u32)], t: &'a str, next: &mut u32) -> u32 {
    let m = tab.len() - 1;
    let mut h: u64 = 1469598103934665603;
    for &b in t.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    let mut i = h as usize & m;
    loop {
        if tab[i].0.is_empty() {
            tab[i] = (t, *next);
            *next += 1;
            return tab[i].1;
        }
        if tab[i].0 == t {
            return tab[i].1;
        }
        i = (i + 1) & m;
    }
}

// ----------------------------------------------------------------- ablation

/// The allocation pattern of the current swadesh-core `tok_13a`
/// (`line.replace` chains, `Vec<char>` per pass, fresh output String).
pub fn tok13a_alloc(line: &str) -> String {
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
    let mut b = Vec::with_capacity(a.len() * 2 + 8);
    for &x in &a {
        if is_13a_class1(x) {
            b.extend_from_slice(&[' ', x, ' ']);
        } else {
            b.push(x);
        }
    }
    let isdig = |x: char| x.is_ascii_digit();
    let mut c = Vec::with_capacity(b.len() + 8);
    sub2(&b, &mut c, |x, y| !isdig(x) && (y == '.' || y == ','), |x, y, o| o.extend_from_slice(&[x, ' ', y, ' ']));
    sub2(&c, &mut b, |x, y| (x == '.' || x == ',') && !isdig(y), |x, y, o| o.extend_from_slice(&[' ', x, ' ', y]));
    sub2(&b, &mut c, |x, y| isdig(x) && y == '-', |x, y, o| o.extend_from_slice(&[x, ' ', y, ' ']));
    let mut out = String::new();
    let mut pending = false;
    for &x in &c {
        if is_py_space(x) {
            if !out.is_empty() {
                pending = true;
            }
        } else {
            if pending {
                out.push(' ');
                pending = false;
            }
            out.push(x);
        }
    }
    out
}

/// Fresh-allocation counting (swadesh-core `ngram::count` + `clipped_matches`).
pub fn ngram_stats_alloc(h: &[u32], r: &[u32], maxn: usize, bits: u32) -> u64 {
    fn count(seq: &[u32], n: usize, bits: u32) -> Vec<Run> {
        let mut keys: Vec<u128> = seq.windows(n).map(|w| pack(w, bits)).collect();
        keys.sort_unstable();
        let mut out: Vec<Run> = Vec::with_capacity(keys.len());
        for k in keys {
            match out.last_mut() {
                Some((last, c)) if *last == k => *c += 1,
                _ => out.push((k, 1)),
            }
        }
        out
    }
    let mut chk: u64 = 0;
    for n in 1..=maxn {
        let (hs, rs) = (count(h, n, bits), count(r, n, bits));
        let (mut i, mut j, mut m) = (0, 0, 0u64);
        while i < hs.len() && j < rs.len() {
            match hs[i].0.cmp(&rs[j].0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    m += hs[i].1.min(rs[j].1) as u64;
                    i += 1;
                    j += 1;
                }
            }
        }
        let nh: u64 = hs.iter().map(|x| x.1 as u64).sum();
        let nr: u64 = rs.iter().map(|x| x.1 as u64).sum();
        chk = chk.wrapping_add((n as u64).wrapping_mul(m * 1_000_003 + nh * 97 + nr));
    }
    chk
}
