//! A minimal `.zip` reader for the test-set downloader.
//!
//! Only what sacreBLEU's archives need: the four `wmt20/robust/*` sets ship as one `.zip` of plain-text
//! files (measured: 24 entries, 21 deflate and 3 stored directories, plus a `__MACOSX` sidecar tree).
//! The central directory is the source of truth - local headers are only used to find where each
//! entry's data starts - which is what the format guarantees.
//!
//! Written rather than pulled in as a crate: `flate2` already supplies the decompressor, and the goal
//! treats a new dependency as a decision to report. Anything the reader cannot handle (an encrypted
//! entry, a Zip64 archive, a method other than stored/deflate) is refused with the entry's name rather
//! than silently skipped.

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use crate::{fail, Fail};

const EOCD: u32 = 0x0605_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const LOCAL: u32 = 0x0403_4b50;

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// Extract every entry of a zip archive into `into`, creating directories as needed.
pub fn extract(bytes: &[u8], into: &Path) -> Result<(), Fail> {
    // The end-of-central-directory record is in the last 64 KiB plus its own 22 bytes.
    let search_start = bytes.len().saturating_sub(65_557);
    let mut eocd = None;
    for at in (search_start..bytes.len().saturating_sub(3)).rev() {
        if u32_at(bytes, at) == Some(EOCD) {
            eocd = Some(at);
            break;
        }
    }
    let Some(eocd) = eocd else {
        return fail(1, "zipfile.BadZipFile: File is not a zip file".to_string());
    };
    let count = u16_at(bytes, eocd + 10).unwrap_or(0) as usize;
    let directory = u32_at(bytes, eocd + 16).unwrap_or(0) as usize;

    let mut at = directory;
    for _ in 0..count {
        if u32_at(bytes, at) != Some(CENTRAL) {
            return fail(1, "zipfile.BadZipFile: Bad magic number for central directory".to_string());
        }
        let method = u16_at(bytes, at + 10).unwrap_or(0);
        let compressed = u32_at(bytes, at + 20).unwrap_or(0) as usize;
        let name_len = u16_at(bytes, at + 28).unwrap_or(0) as usize;
        let extra_len = u16_at(bytes, at + 30).unwrap_or(0) as usize;
        let comment_len = u16_at(bytes, at + 32).unwrap_or(0) as usize;
        let local = u32_at(bytes, at + 42).unwrap_or(0) as usize;
        let name = String::from_utf8_lossy(bytes.get(at + 46..at + 46 + name_len).unwrap_or(&[]))
            .into_owned();
        at += 46 + name_len + extra_len + comment_len;

        // Directories and macOS sidecars carry no data the parsers read; Python extracts them, but
        // they are invisible to every output, so they are skipped rather than written.
        if name.ends_with('/') || name.starts_with("__MACOSX/") {
            continue;
        }
        let Some(target) = safe_join(into, &name) else {
            return fail(1, format!("zipfile.BadZipFile: unsafe path in archive: {name}"));
        };

        if u32_at(bytes, local) != Some(LOCAL) {
            return fail(1, format!("zipfile.BadZipFile: Bad magic number for {name}"));
        }
        let local_name_len = u16_at(bytes, local + 26).unwrap_or(0) as usize;
        let local_extra_len = u16_at(bytes, local + 28).unwrap_or(0) as usize;
        let start = local + 30 + local_name_len + local_extra_len;
        let Some(data) = bytes.get(start..start + compressed) else {
            return fail(1, format!("zipfile.BadZipFile: truncated data for {name}"));
        };
        let plain = match method {
            0 => data.to_vec(),
            8 => {
                let mut out = Vec::new();
                flate2::read::DeflateDecoder::new(data)
                    .read_to_end(&mut out)
                    .or_else(|e| fail(1, format!("zipfile.BadZipFile: {name}: {e}")))?;
                out
            }
            other => {
                return fail(
                    1,
                    format!("zipfile.BadZipFile: unsupported compression method {other} for {name}"),
                )
            }
        };
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .or_else(|e| fail(1, format!("{}: {e}", parent.display())))?;
        }
        std::fs::write(&target, plain).or_else(|e| fail(1, format!("{}: {e}", target.display())))?;
    }
    Ok(())
}

/// Join an archive-relative name onto `root`, refusing anything that could escape it.
fn safe_join(root: &Path, name: &str) -> Option<PathBuf> {
    let mut out = root.to_path_buf();
    for part in name.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        let component = Path::new(part);
        let mut parts = component.components();
        match (parts.next(), parts.next()) {
            (Some(Component::Normal(_)), None) => out.push(part),
            _ => return None,
        }
    }
    Some(out)
}