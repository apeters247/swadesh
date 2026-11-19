#![no_main]

use libfuzzer_sys::fuzz_target;
use swadesh_core::tokenize::{tokenize_unjoined, tokens, Tokenizer};

// Arbitrary bytes, lossily decoded: exercises invalid UTF-8 replacement too.
fuzz_target!(|data: &[u8]| {
    let line = String::from_utf8_lossy(data);
    let mut out = String::new();
    Tokenizer::Zh.apply(&line, &mut out);
    let _ = tokens(&out).count();
    tokenize_unjoined(Tokenizer::Zh, &line, &mut out);
    let _ = tokens(&out).count();
});
