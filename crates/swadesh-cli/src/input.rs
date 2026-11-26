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
fn decode_utf8(bytes: &[u8]) -> Result<String, Fail> {
    String::from_utf8(bytes.to_vec()).or_else(|e| {
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

/// Whether `-e` names a codec swadesh can decode with; called once up front so a bad name is
/// refused before any file is opened, as sacreBLEU does.
pub fn check_encoding(encoding: &str) -> Result<(), Fail> {
    codec(encoding).map(|_| ())
}

/// The codecs swadesh decodes with. `-e` takes any Python codec name; these are the ones that need
/// no dependency beyond `std`, and they cover the encodings MT data actually ships in. Anything else
/// is refused with a hint rather than decoded wrongly (`docs/parity-report.md` records it).
#[derive(Clone, Copy)]
enum Codec {
    Utf8,
    /// UTF-8 with a leading BOM stripped, as Python's `utf-8-sig`.
    Utf8Sig,
    Ascii,
    Latin1,
    /// CP1252, whose 0x80..=0x9F block is where it differs from Latin-1.
    Cp1252,
}

/// A Python codec name: case-insensitive, with `-`, `_` and space interchangeable.
fn codec(encoding: &str) -> Result<Codec, Fail> {
    let norm: String = encoding
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c == '-' || c == '_' || c == ' ' { '_' } else { c })
        .collect();
    match norm.as_str() {
        "utf_8" | "utf8" | "u8" => Ok(Codec::Utf8),
        "utf_8_sig" => Ok(Codec::Utf8Sig),
        "ascii" | "us_ascii" | "646" => Ok(Codec::Ascii),
        "latin_1" | "latin1" | "iso_8859_1" | "iso8859_1" | "8859" | "cp819" | "l1" => {
            Ok(Codec::Latin1)
        }
        "cp1252" | "windows_1252" | "1252" => Ok(Codec::Cp1252),
        _ => fail(
            2,
            format!("encoding '{encoding}' is not built into swadesh; use the Python package for it"),
        ),
    }
}

/// CP1252's 0x80..=0x9F block; five bytes are undefined and raise in Python.
fn cp1252_char(b: u8) -> Option<char> {
    match b {
        0x80 => Some('\u{20ac}'), 0x82 => Some('\u{201a}'), 0x83 => Some('\u{0192}'),
        0x84 => Some('\u{201e}'), 0x85 => Some('\u{2026}'), 0x86 => Some('\u{2020}'),
        0x87 => Some('\u{2021}'), 0x88 => Some('\u{02c6}'), 0x89 => Some('\u{2030}'),
        0x8a => Some('\u{0160}'), 0x8b => Some('\u{2039}'), 0x8c => Some('\u{0152}'),
        0x8e => Some('\u{017d}'), 0x91 => Some('\u{2018}'), 0x92 => Some('\u{2019}'),
        0x93 => Some('\u{201c}'), 0x94 => Some('\u{201d}'), 0x95 => Some('\u{2022}'),
        0x96 => Some('\u{2013}'), 0x97 => Some('\u{2014}'), 0x98 => Some('\u{02dc}'),
        0x99 => Some('\u{2122}'), 0x9a => Some('\u{0161}'), 0x9b => Some('\u{203a}'),
        0x9c => Some('\u{0153}'), 0x9e => Some('\u{017e}'), 0x9f => Some('\u{0178}'),
        0x81 | 0x8d | 0x8f | 0x90 | 0x9d => None,
        _ => Some(b as char),
    }
}

/// Decode `bytes` with `codec`, raising Python's own error line when the data does not fit.
fn decode_with(codec: Codec, bytes: &[u8]) -> Result<String, Fail> {
    match codec {
        Codec::Utf8 => decode_utf8(bytes),
        Codec::Utf8Sig => {
            decode_utf8(bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes))
        }
        Codec::Ascii => {
            for (position, &byte) in bytes.iter().enumerate() {
                if byte > 0x7f {
                    return fail(
                        1,
                        format!(
                            "UnicodeDecodeError: 'ascii' codec can't decode byte 0x{byte:02x} in position {position}: ordinal not in range(128)"
                        ),
                    );
                }
            }
            Ok(String::from_utf8(bytes.to_vec()).expect("ascii is valid utf-8"))
        }
        // Latin-1 is the identity: every byte is its own code point, so it never fails.
        Codec::Latin1 => Ok(bytes.iter().map(|&b| b as char).collect()),
        Codec::Cp1252 => {
            let mut out = String::with_capacity(bytes.len());
            for (position, &byte) in bytes.iter().enumerate() {
                match cp1252_char(byte) {
                    Some(ch) => out.push(ch),
                    None => {
                        return fail(
                            1,
                            format!(
                                "UnicodeDecodeError: 'charmap' codec can't decode byte 0x{byte:02x} in position {position}: character maps to <undefined>"
                            ),
                        )
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Lines of a file, rstripped, as `for line in open(path, newline="\n")`.
pub fn file_lines(path: &str, encoding: &str) -> Result<Vec<String>, Fail> {
    let codec = codec(encoding)?;
    let bytes = match virtual_file(path) {
        Some(b) => b,
        None => std::fs::read(path).or_else(|e| fail(1, io_error(path, &e)))?,
    };
    // A `.gz` name means gzip, decided by the name and case-sensitively, exactly as Python's
    // `gzip.open` is reached by sacreBLEU (`.GZ` is *not* decompressed).
    let bytes = if path.ends_with(".gz") { gunzip(&bytes)? } else { bytes };
    let text = decode_with(codec, &bytes)?;
    Ok(text.split_inclusive('\n').map(|l| py_rstrip(l).to_string()).collect())
}

/// Python's `b'...'` repr of a short byte string, for `gzip.BadGzipFile`'s message.
fn py_bytes_repr(bytes: &[u8]) -> String {
    let mut out = String::from("b'");
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\'' => out.push_str("\\'"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push('\'');
    out
}

/// Decode a `.gz` stream the way Python's `gzip` module does, error lines included.
///
/// Measured: an empty file is empty (Python short-circuits before reading the header); anything else
/// must begin with the gzip magic or Python raises `BadGzipFile: Not a gzipped file (b'XX')` with the
/// first two bytes repr'd; a truncated stream is `EOFError: Compressed file ended before the
/// end-of-stream marker was reached`.
pub fn gunzip(bytes: &[u8]) -> Result<Vec<u8>, Fail> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if !bytes.starts_with(&[0x1f, 0x8b]) {
        return fail(
            1,
            format!(
                "gzip.BadGzipFile: Not a gzipped file ({})",
                py_bytes_repr(&bytes[..bytes.len().min(2)])
            ),
        );
    }
    let mut out = Vec::new();
    let mut decoder = flate2::read::GzDecoder::new(bytes);
    match std::io::Read::read_to_end(&mut decoder, &mut out) {
        Ok(_) => Ok(out),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => fail(
            1,
            "EOFError: Compressed file ended before the end-of-stream marker was reached"
                .to_string(),
        ),
        // A complete-but-corrupt stream (a bad CRC, say) is unmeasured; Python's wording for it is
        // not reproduced (`docs/parity-report.md`).
        Err(e) => fail(1, format!("gzip.BadGzipFile: {e}")),
    }
}

/// Lines of stdin with universal newlines, rstripped.
pub fn stdin_lines(encoding: &str) -> Result<Vec<String>, Fail> {
    let codec = codec(encoding)?;
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
    let text = decode_with(codec, &bytes)?;
    if text.is_empty() {
        return Ok(vec![String::new()]);
    }
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    Ok(normalized.split_inclusive('\n').map(|l| py_rstrip(l).to_string()).collect())
}
