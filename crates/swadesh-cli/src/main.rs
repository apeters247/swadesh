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
            // Output already written stays on stdout; the error goes to stderr. An empty message
            // means the diagnostic was already printed, in sacreBLEU's own wording.
            if !f.msg.is_empty() {
                // argparse prints its usage block above an argument error, so this does too.
                if f.code == 2 && swadesh_cli::args::is_argparse_error(&f.msg) {
                    eprint!("{}", swadesh_cli::args::USAGE_BLOCK);
                }
                eprintln!("swadesh: error: {}", f.msg);
            }
            ExitCode::from(f.code as u8)
        }
    }
}
