#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use swadesh_cli::input::{clear_virtual_files, set_virtual_file};
use swadesh_cli::run_argv;

// Option names from swadesh-cli's option table, so generated argv reaches the
// real option handling instead of failing at the first token.
const FLAGS: &[&str] = &[
    "-m", "--metrics", "-sl", "--sentence-level", "-i", "--input", "-nr", "--num-refs", "-e", "--encoding",
    "-l", "--language-pair", "-s", "--smooth-method", "-sv", "--smooth-value", "-tok", "--tokenize", "-lc",
    "--lowercase", "--force", "--max-ngram-order", "--python-version", "-cc", "--chrf-char-order", "-cw",
    "--chrf-word-order", "--chrf-beta", "--chrf-whitespace", "--chrf-lowercase", "--chrf-eps-smoothing",
    "-q", "-sh", "-b", "-w", "--width", "-nc", "-f", "--format", "-t", "-ci", "-par", "-pbs", "-j",
    "--ter-case-sensitive", "--ter-asian-support", "--ter-no-punct", "--ter-normalized", "-h", "-V",
    "--citation", "--detail", "bleu", "chrf", "ter", "text", "json", "none", "13a", "intl", "zh", "char",
    "floor", "add-k", "exp", "yes", "no", "3", "0", "-1", "99999999999999999999", "1.5", "nan",
];

#[derive(Arbitrary, Debug)]
enum Tok {
    Flag(u8),
    Raw(String),
}

#[derive(Arbitrary, Debug)]
struct Input {
    argv: Vec<Tok>,
    files: Vec<Vec<u8>>,
}

fuzz_target!(|i: Input| {
    let argv: Vec<String> = i
        .argv
        .iter()
        .take(14)
        .map(|t| match t {
            Tok::Flag(k) => FLAGS[*k as usize % FLAGS.len()].to_string(),
            Tok::Raw(s) => s.chars().take(48).collect(),
        })
        .collect();

    // Every file the CLI could open must be an in-memory file. Option values
    // are suffixes of their tokens (`--input=X`, `-iX`), so register every
    // suffix of every token; nothing falls through to the real filesystem.
    clear_virtual_files();
    let empty = Vec::new();
    let contents: Vec<&Vec<u8>> = if i.files.is_empty() { vec![&empty] } else { i.files.iter().take(4).collect() };
    let mut k = 0usize;
    set_virtual_file("<stdin>", contents[0].clone());
    for tok in &argv {
        for (pos, _) in tok.char_indices().map(|(p, c)| (p, c)).chain(std::iter::once((tok.len(), ' '))) {
            set_virtual_file(&tok[pos..], contents[k % contents.len()].clone());
            k += 1;
        }
    }
    let mut out: Vec<u8> = Vec::new();
    let _ = run_argv(&argv, &mut out);
    clear_virtual_files();
});
