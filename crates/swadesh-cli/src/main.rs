//! `swadesh`: a standalone BLEU and chrF command line, byte-identical in its
//! stdout to `sacrebleu` 2.6.0 for the options it supports. No Python needed.

use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|s| s.to_string_lossy().into_owned())
        .collect();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let result = swadesh_cli::run_argv(&argv, &mut out);
    let flushed = out.flush();
    match result {
        Ok(()) => match flushed {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::from(1),
        },
        Err(f) => {
            // Output already written stays on stdout; the error goes to stderr, formatted by the
            // same function the in-process entry point uses.
            eprint!("{}", swadesh_cli::error_stderr(&f));
            ExitCode::from(f.code as u8)
        }
    }
}
