//! `swadesh`: a standalone BLEU and chrF command line, byte-identical in its
//! stdout to `sacrebleu` 2.6.0 for the options it supports. No Python needed.

mod args;
mod input;
mod lower;
mod pyfmt;
mod report;
mod run;

use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use args::Action;

/// A failure: process exit code and a message for stderr.
#[derive(Debug)]
pub struct Fail {
    pub code: i32,
    pub msg: String,
}

/// Shorthand for returning an `Err(Fail)`.
pub fn fail<T>(code: i32, msg: impl Into<String>) -> Result<T, Fail> {
    Err(Fail { code, msg: msg.into() })
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|s| s.to_string_lossy().into_owned())
        .collect();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let result = match args::parse(&argv) {
        Ok(Action::Help) => write!(out, "{}", args::USAGE).map_err(|e| Fail { code: 1, msg: e.to_string() }),
        Ok(Action::Version) => writeln!(
            out,
            "swadesh {} (sacreBLEU {})",
            env!("CARGO_PKG_VERSION"),
            report::SACREBLEU_VERSION
        )
        .map_err(|e| Fail { code: 1, msg: e.to_string() }),
        Ok(Action::Run(cli)) => run::execute(&cli, &mut out),
        Err(f) => Err(f),
    };
    let flushed = out.flush();
    match result {
        Ok(()) => match flushed {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::from(1),
        },
        Err(f) => {
            // Output already written stays on stdout; the error goes to stderr.
            eprintln!("swadesh: error: {}", f.msg);
            ExitCode::from(f.code as u8)
        }
    }
}
