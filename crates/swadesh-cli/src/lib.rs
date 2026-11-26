//! `swadesh` command line as a library: a standalone BLEU and chrF command
//! line, byte-identical in its stdout to `sacrebleu` 2.6.0 for the options it
//! supports. Used by the `swadesh` binary and the WebAssembly playground.

pub mod args;
pub mod dataset;
pub mod plain;
pub mod iwslt;
pub mod tsv;
#[cfg(feature = "downloader")]
pub mod fetch;
pub mod sgml;
pub mod wmtxml;
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

/// The stderr text for a failure: what `main` prints, and what the in-process entry point
/// (`_swadesh.run_cli`) returns, so the two are byte-identical.
///
/// argparse prints its usage block above an argument error. Our own messages carry a
/// `swadesh: error: ` prefix; the two errors that are *also* a Python exception line
/// (`UnicodeDecodeError`, `FileNotFoundError`, ...) are printed as they stand, because the parity
/// harness compares that line against CPython's own. An empty message means the diagnostic was
/// already written to stderr directly, in sacreBLEU's wording, and nothing more is printed.
pub fn error_stderr(f: &Fail) -> String {
    if f.msg.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    if f.code == 2 && args::is_argparse_error(&f.msg) {
        out.push_str(args::USAGE_BLOCK);
    }
    if is_python_error(&f.msg) {
        out.push_str(&f.msg);
    } else {
        out.push_str("swadesh: error: ");
        out.push_str(&f.msg);
    }
    out.push('\n');
    out
}

fn is_python_error(msg: &str) -> bool {
    const CLASSES: [&str; 6] = [
        "UnicodeDecodeError: ",
        "FileNotFoundError: ",
        "PermissionError: ",
        "IsADirectoryError: ",
        "gzip.BadGzipFile: ",
        "EOFError: ",
    ];
    CLASSES.iter().any(|c| msg.starts_with(c))
}

/// Run the CLI with `argv` (without the program name), writing stdout to `out`.
pub fn run_argv(argv: &[String], out: &mut dyn Write) -> Result<(), Fail> {
    match args::parse(argv) {
        Ok(Action::Help) => write!(out, "{}", args::USAGE).map_err(|e| Fail { code: 1, msg: e.to_string() }),
        Ok(Action::Version) => writeln!(out, "swadesh {}", report::SACREBLEU_VERSION)
            .map_err(|e| Fail { code: 1, msg: e.to_string() }),
        Ok(Action::Run(cli)) => run::execute(&cli, out),
        Err(f) => Err(f),
    }
}
