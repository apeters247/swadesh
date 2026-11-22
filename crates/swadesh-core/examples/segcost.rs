//! Per-segment cost of the BLEU and chrF statistics, in nanoseconds per hypothesis-reference pair.
//!
//!     cargo run --release -q -p swadesh-core --example segcost
//!
//! Best-of-N is reported rather than the mean: the mean on a shared box mostly measures other
//! people's load, and these paths are short enough that one preemption dominates a sample.

use std::time::Instant;

use swadesh_core::{BleuConfig, ChrfConfig, Tokenizer};

/// Best of `reps` runs of `f`, divided by `n` pairs.
fn ns_per_pair(reps: usize, n: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        let ns = t.elapsed().as_nanos() as f64 / n as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

fn main() {
    const N: usize = 5000;
    let base = "Das Gesundheitsministerium des Iran hat 1.686 neue taegliche COVID-19 Infektionen \
                verzeichnet , die niedrigste Zahl in den letzten 460 Tagen .";
    let hyps: Vec<String> = (0..N).map(|i| format!("{} {}", base, i % 7)).collect();
    let refs: Vec<String> = (0..N).map(|_| base.to_string()).collect();

    let bleu = BleuConfig { tokenizer: Tokenizer::T13a, max_order: 4 };
    let mut row = Vec::new();
    let b = ns_per_pair(20, N, || {
        for (h, r) in hyps.iter().zip(refs.iter()) {
            swadesh_core::bleu_stats_segment(&bleu, h, &[r.as_str()], &mut row);
        }
    });

    let chrf = ChrfConfig {
        char_order: 6,
        word_order: 2,
        beta: 2,
        whitespace: false,
        eps_smoothing: false,
    };
    let c = ns_per_pair(20, N, || {
        for (h, r) in hyps.iter().zip(refs.iter()) {
            swadesh_core::chrf_stats_segment(&chrf, h, &[r.as_str()], &mut row);
        }
    });

    println!("bleu statistics  {b:8.1} ns/pair");
    println!("chrF++ statistics {c:7.1} ns/pair");
}