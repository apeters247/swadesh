#![no_main]

use std::time::Instant;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use swadesh_core::ter::{ter_segment, TerConfig, TerError};

// Inputs are capped so that legitimate worst-case run time stays small; a
// libFuzzer per-input timeout (-timeout=) turns any hang into a finding.
const CAP_BYTES: usize = 256;

#[derive(Arbitrary, Debug)]
struct Input {
    flags: u8,
    hyp: String,
    refs: Vec<String>,
}

fn cap(s: &str) -> &str {
    if s.len() <= CAP_BYTES {
        return s;
    }
    let mut end = CAP_BYTES;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fuzz_target!(|i: Input| {
    let cfg = TerConfig {
        normalized: i.flags & 1 != 0,
        no_punct: i.flags & 2 != 0,
        asian_support: i.flags & 4 != 0,
        case_sensitive: i.flags & 8 != 0,
    };
    let refs: Vec<&str> = i.refs.iter().take(4).map(|r| cap(r)).collect();
    let hyp = cap(&i.hyp);
    let start = Instant::now();
    let res = ter_segment(&cfg, hyp, &refs);
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 5, "ter_segment took {elapsed:?} on capped input");
    match res {
        Ok([edits, ref_len]) => {
            assert!(!refs.is_empty());
            assert!(edits.is_finite() && edits >= 0.0, "edits={edits}");
            assert!(ref_len.is_finite() && ref_len >= 0.0, "ref_len={ref_len}");
        }
        Err(TerError::NoReferences) => assert!(refs.is_empty()),
    }
});
