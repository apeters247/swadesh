//! Rust driver for the swadesh micro-benchmarks. Same CLI and output format as
//! `../c/langbench.c`. See docs/language-choice.md for the methodology.

mod kernels;
mod scan;

use kernels::*;
use rayon::prelude::*;
use std::time::Instant;

struct Opts {
    kernel: String,
    count: String,
    scan: String,
    threads: usize,
    reps: usize,
    hyp: String,
    rf: String,
}

fn parse() -> Option<Opts> {
    let mut o = Opts {
        kernel: "tok".into(),
        count: "sort".into(),
        scan: "avx2".into(),
        threads: 1,
        reps: 5,
        hyp: String::new(),
        rf: String::new(),
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < args.len() {
        let v = args[i + 1].clone();
        match args[i].as_str() {
            "--kernel" => o.kernel = v,
            "--count" => o.count = v,
            "--scan" => o.scan = v,
            "--threads" => o.threads = v.parse().ok()?,
            "--reps" => o.reps = v.parse().ok()?,
            "--hyp" => o.hyp = v,
            "--ref" => o.rf = v,
            _ => return None,
        }
        i += 2;
    }
    if o.hyp.is_empty() || o.rf.is_empty() {
        return None;
    }
    Some(o)
}

/// Split like the C loader: LF-terminated lines, no phantom trailing line.
fn lines(text: &str) -> Vec<&str> {
    let mut v: Vec<&str> = text.split('\n').collect();
    if v.last() == Some(&"") {
        v.pop();
    }
    v
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// Sum of per-segment results. threads == 1 runs inline (no pool overhead).
fn par_sum<'a, F>(n: usize, pool: Option<&rayon::ThreadPool>, f: F) -> u64
where
    F: Fn(usize, &mut Scratch<'a>) -> u64 + Sync,
{
    match pool {
        None => {
            let mut s = Scratch::default();
            (0..n).fold(0u64, |acc, i| acc.wrapping_add(f(i, &mut s)))
        }
        Some(p) => p.install(|| {
            (0..n)
                .into_par_iter()
                .map_init(Scratch::default, |s, i| f(i, s))
                .reduce(|| 0u64, |a, b| a.wrapping_add(b))
        }),
    }
}

fn par_scan(p: &[u8], threads: usize, pool: Option<&rayon::ThreadPool>, f: fn(&[u8]) -> u64) -> u64 {
    let parts = threads * 4;
    let tot = p.len();
    let run = |k: usize| {
        let (s, e) = (tot * k / parts, tot * (k + 1) / parts);
        f(&p[s..e])
    };
    match pool {
        None => (0..parts).fold(0u64, |a, k| a.wrapping_add(run(k))),
        Some(pl) => pl.install(|| (0..parts).into_par_iter().map(run).reduce(|| 0, |a, b| a.wrapping_add(b))),
    }
}

fn main() {
    let o = match parse() {
        Some(o) => o,
        None => {
            eprintln!("usage: langbench-rs --kernel tok|chr|wrd|scan --count sort|hash|alloc --scan scalar|sse2|avx2|asm --threads T --reps R --hyp HYP --ref REF");
            std::process::exit(1);
        }
    };
    let hyp_text = std::fs::read_to_string(&o.hyp).expect("read hyp");
    let ref_text = std::fs::read_to_string(&o.rf).expect("read ref");
    let hyps = lines(&hyp_text);
    let refs = lines(&ref_text);
    assert_eq!(hyps.len(), refs.len(), "line count mismatch");
    let n = hyps.len();
    let alloc = o.count == "alloc";
    let use_hash = o.count == "hash";
    let pool = if o.threads > 1 {
        Some(rayon::ThreadPoolBuilder::new().num_threads(o.threads).build().expect("pool"))
    } else {
        None
    };
    let scan_fn: fn(&[u8]) -> u64 = match o.scan.as_str() {
        "scalar" => scan::scan_scalar,
        "lut" => scan::scan_lut,
        "sse2" => scan::scan_sse2,
        "avx2" => scan::scan_avx2,
        _ => scan::scan_asm,
    };

    // Pre-tokenised text for the word kernel (untimed setup).
    let tok_all = |ls: &[&str]| -> Vec<String> {
        let mut s = Scratch::default();
        ls.iter()
            .map(|l| {
                let k = tok13a(l, &mut s);
                String::from_utf8(s.ob[..k].to_vec()).unwrap()
            })
            .collect()
    };
    let (tokh, tokr) = if o.kernel == "wrd" { (tok_all(&hyps), tok_all(&refs)) } else { (vec![], vec![]) };

    let mut chk = 0u64;
    let mut times = Vec::with_capacity(o.reps);
    for r in 0..=o.reps {
        let t0 = Instant::now();
        chk = match o.kernel.as_str() {
            "tok" => par_sum(n, pool.as_ref(), |i, s| {
                let mut acc = 0u64;
                for side in [hyps[i], refs[i]] {
                    if alloc {
                        let out = tok13a_alloc(side);
                        acc = acc.wrapping_add(fnv(out.as_bytes())).wrapping_add(out.len() as u64);
                    } else {
                        let k = if o.count == "ascii" { tok13a_fast(side, s) } else { tok13a(side, s) };
                        acc = acc.wrapping_add(fnv(&s.ob[..k])).wrapping_add(k as u64);
                    }
                }
                acc
            }),
            "chr" => par_sum(n, pool.as_ref(), |i, s| {
                if alloc {
                    let h: Vec<u32> = hyps[i].chars().filter(|&c| !is_py_space(c)).map(|c| c as u32).collect();
                    let r: Vec<u32> = refs[i].chars().filter(|&c| !is_py_space(c)).map(|c| c as u32).collect();
                    ngram_stats_alloc(&h, &r, 6, 21)
                } else {
                    chr_pair(hyps[i], refs[i], use_hash, s)
                }
            }),
            "wrd" => par_sum(n, pool.as_ref(), |i, s| {
                if alloc {
                    wrd_alloc(&tokh[i], &tokr[i])
                } else {
                    wrd_pair(&tokh[i], &tokr[i], use_hash, s)
                }
            }),
            "scan" => par_scan(hyp_text.as_bytes(), o.threads, pool.as_ref(), scan_fn),
            other => {
                eprintln!("bad kernel {other}");
                std::process::exit(1);
            }
        };
        let dt = t0.elapsed().as_nanos() as f64;
        if r > 0 {
            times.push(dt / n as f64);
        }
    }
    let mut sorted: Vec<u64> = times.iter().map(|&t| t as u64).collect();
    sorted.sort_unstable();
    let med = sorted[sorted.len() / 2];
    let reps: Vec<String> = times.iter().map(|t| format!("{:.0}", t)).collect();
    println!(
        "impl=rust kernel={} count={} scan={} threads={} segments={} ns_per_seg={} checksum={} reps={}",
        o.kernel,
        o.count,
        o.scan,
        o.threads,
        n,
        med,
        chk,
        reps.join(",")
    );
}

/// Word n-grams with fresh allocations (ablation; mirrors the current swadesh-core flow).
fn wrd_alloc<'a>(h: &'a str, r: &'a str) -> u64 {
    use std::collections::HashMap;
    let mut words: HashMap<&'a str, u32> = HashMap::new();
    let mut id = |t: &'a str| -> u32 {
        let n = words.len() as u32;
        *words.entry(t).or_insert(n)
    };
    let hv: Vec<u32> = h.split(' ').filter(|t| !t.is_empty()).map(&mut id).collect();
    let rv: Vec<u32> = r.split(' ').filter(|t| !t.is_empty()).map(&mut id).collect();
    ngram_stats_alloc(&hv, &rv, 4, 32)
}
