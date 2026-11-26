//! Reading references and hypotheses the way sacreBLEU's CLI does.
//!
//! * Files (`-i`, reference paths) are opened with `newline="\n"`: lines end
//!   only at `\n`, and every line is `rstrip()`ed.
//! * Stdin is a `TextIOWrapper` with universal newlines: `\r\n` and a lone `\r`
//!   also end lines. An empty stdin is one empty line (`readline()` returns "").

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;

use swadesh_core::unicode::py_rstrip;

use crate::{fail, Fail};

thread_local! {
    /// In-memory files (used by the WebAssembly build, where there is no
    /// filesystem). The key `"<stdin>"` stands for standard input.
    static VIRTUAL_FILES: RefCell<HashMap<String, Vec<u8>>> = RefCell::new(HashMap::new());
}

/// Register an in-memory file; `path == "<stdin>"` sets standard input.
pub fn set_virtual_file(path: &str, contents: Vec<u8>) {
    VIRTUAL_FILES.with(|f| f.borrow_mut().insert(path.to_string(), contents));
}

/// Remove all in-memory files.
pub fn clear_virtual_files() {
    VIRTUAL_FILES.with(|f| f.borrow_mut().clear());
}

fn virtual_file(path: &str) -> Option<Vec<u8>> {
    VIRTUAL_FILES.with(|f| f.borrow().get(path).cloned())
}

/// Python's own `UnicodeDecodeError` line, which `tests/test_parity.py` compares line for line.
///
/// The reason is Python's: `unexpected end of data` when the sequence is truncated, `invalid
/// continuation byte` when a valid lead byte is followed by the wrong byte, and `invalid start byte`
/// otherwise (a lead byte that cannot begin a sequence, e.g. 0xff). Verified against CPython for
/// `b"\xff"`, `b"\x80"`, `b"\xc3"` and `b"\xc3\x28"`.
fn decode_utf8(bytes: Vec<u8>, _what: &str) -> Result<String, Fail> {
    String::from_utf8(bytes).or_else(|e| {
        let err = e.utf8_error();
        let position = err.valid_up_to();
        let byte = e.as_bytes().get(position).copied().unwrap_or(0);
        let reason = match err.error_len() {
            None => "unexpected end of data",
            Some(_) if (0xC2..=0xF4).contains(&byte) => "invalid continuation byte",
            Some(_) => "invalid start byte",
        };
        fail(
            1,
            format!(
                "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x{byte:02x} in position {position}: {reason}"
            ),
        )
    })
}

/// Python's line for a failed open, class and errno included.
fn io_error(path: &str, e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => {
            format!("FileNotFoundError: [Errno 2] No such file or directory: '{path}'")
        }
        std::io::ErrorKind::PermissionDenied => {
            format!("PermissionError: [Errno 13] Permission denied: '{path}'")
        }
        _ if std::path::Path::new(path).is_dir() => {
            format!("IsADirectoryError: [Errno 21] Is a directory: '{path}'")
        }
        // Unmeasured kinds keep a readable message rather than guessing Python's wording.
        _ => format!("{path}: {e}"),
    }
}

pub fn check_encoding(encoding: &str) -> Result<(), Fail> {
    match encoding.to_ascii_lowercase().as_str() {
        "utf-8" | "utf8" => Ok(()),
        other => fail(2, format!("encoding '{other}' is not supported by swadesh (only utf-8)")),
    }
}

/// Lines of a file, rstripped, as `for line in open(path, newline="\n")`.
pub fn file_lines(path: &str, encoding: &str) -> Result<Vec<String>, Fail> {
    check_encoding(encoding)?;
    if path.ends_with(".gz") {
        return fail(1, format!("gzip input ({path}) is not supported by swadesh; decompress it first"));
    }
    let bytes = match virtual_file(path) {
        Some(b) => b,
        None => std::fs::read(path).or_else(|e| fail(1, io_error(path, &e)))?,
    };
    let text = decode_utf8(bytes, path)?;
    Ok(text.split_inclusive('\n').map(|l| py_rstrip(l).to_string()).collect())
}

/// Lines of stdin with universal newlines, rstripped.
pub fn stdin_lines(encoding: &str) -> Result<Vec<String>, Fail> {
    check_encoding(encoding)?;
    let bytes = match virtual_file("<stdin>") {
        Some(b) => b,
        None => {
            let mut bytes = Vec::new();
            std::io::stdin()
                .lock()
                .read_to_end(&mut bytes)
                .or_else(|e| fail(1, format!("reading stdin: {e}")))?;
            bytes
        }
    };
    let text = decode_utf8(bytes, "stdin")?;
    if text.is_empty() {
        return Ok(vec![String::new()]);
    }
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    Ok(normalized.split_inclusive('\n').map(|l| py_rstrip(l).to_string()).collect())
}
