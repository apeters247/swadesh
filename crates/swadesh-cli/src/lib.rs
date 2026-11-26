//! `swadesh` command line as a library: a standalone BLEU and chrF command
//! line, byte-identical in its stdout to `sacrebleu` 2.6.0 for the options it
//! supports. Used by the `swadesh` binary and the WebAssembly playground.

pub mod args;
pub mod datasets;
pub mod input;
mod lower;
mod pyfmt;
pub mod report;
pub mod run;
pub mod tabulate;

use std::io::Write;

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

/// Run the CLI with `argv` (without the program name), writing stdout to `out`.
pub fn run_argv(argv: &[String], out: &mut dyn Write) -> Result<(), Fail> {
    match args::parse(argv) {
        Ok(Action::Help) => write!(out, "{}", args::USAGE).map_err(|e| Fail { code: 1, msg: e.to_string() }),
        Ok(Action::Version) => writeln!(
            out,
            "swadesh {} (sacreBLEU {})",
            env!("CARGO_PKG_VERSION"),
            report::SACREBLEU_VERSION
        )
        .map_err(|e| Fail { code: 1, msg: e.to_string() }),
        Ok(Action::Run(cli)) => run::execute(&cli, out),
        Err(f) => Err(f),
    }
}
