//! Reading references and hypotheses the way sacreBLEU's CLI does.
//!
//! * Files (`-i`, reference paths) are opened with `newline="\n"`: lines end
//!   only at `\n`, and every line is `rstrip()`ed.
//! * Stdin is a `TextIOWrapper` with universal newlines: `\r\n` and a lone `\r`
//!   also end lines. An empty stdin is one empty line (`readline()` returns "").

use std::io::Read;

use swadesh_core::unicode::py_rstrip;

use crate::{fail, Fail};

fn decode_utf8(bytes: Vec<u8>, what: &str) -> Result<String, Fail> {
    String::from_utf8(bytes).or_else(|e| {
        fail(
            1,
            format!(
                "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x{:02x} in {what} (position {})",
                e.as_bytes()[e.utf8_error().valid_up_to()],
                e.utf8_error().valid_up_to()
            ),
        )
    })
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
    let bytes = std::fs::read(path).or_else(|e| {
        fail(1, format!("{path}: {}", e))
    })?;
    let text = decode_utf8(bytes, path)?;
    Ok(text.split_inclusive('\n').map(|l| py_rstrip(l).to_string()).collect())
}

/// Lines of stdin with universal newlines, rstripped.
pub fn stdin_lines(encoding: &str) -> Result<Vec<String>, Fail> {
    check_encoding(encoding)?;
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .read_to_end(&mut bytes)
        .or_else(|e| fail(1, format!("reading stdin: {e}")))?;
    let text = decode_utf8(bytes, "stdin")?;
    if text.is_empty() {
        return Ok(vec![String::new()]);
    }
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    Ok(normalized.split_inclusive('\n').map(|l| py_rstrip(l).to_string()).collect())
}
