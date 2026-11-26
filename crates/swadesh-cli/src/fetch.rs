//! Fetching a test set's archives: the `downloader` cargo feature (PLAN.md D-2).
//!
//! Measured from a cold cache with a small set. sacreBLEU prints two lines per archive and nothing
//! else - no progress bar, so the output is deterministic:
//!
//! ```text
//! sacreBLEU: Downloading <url> to <cache>/<set>/raw/<prefix>.<archive basename>
//! sacreBLEU: Extracting <archive> to <cache>/<set>/raw
//! ```
//!
//! where the archive name is the file-name prefix (the set name with `/` written `_`) followed by the
//! URL's basename. Extraction leaves the archive's own directory structure under `raw/`, which is
//! what the parsers in `sgml`, `wmtxml`, `plain` and `iwslt` then read.
//!
//! **All of a set's archives, in registry order.** The catalogue stores a flat list of URLs with no
//! per-pair mapping, and the oracle does not need one: measured, `-t iwslt17 -l en-fr` downloads and
//! extracts all twelve archives even though only one pair is parsed, printing the two lines once per
//! URL. So the multi-archive case is a loop over `data`, not a lookup: 27 single-archive and 31
//! multi-archive sets share this one path.
//!
//! **What this does not cover, and refuses explicitly.** Formats other than `.tar.gz`/`.tgz` - four
//! `.zip` sets and the plain-`.gz` `multi30k/2018` - are refused whole rather than partially fetched.
//! And the md5-mismatch message is ours, not measured: producing a corrupted download needs a fake
//! server, so sacreBLEU's wording for that case is still unknown (`docs/parity-report.md` records it).

use std::io::Read;
use std::path::{Path, PathBuf};

use md5::{Digest, Md5};

use crate::dataset::Found;
use crate::datasets::DATASETS;
use crate::{fail, Fail};

/// Download a set's archives when they are missing, verify each md5, and extract them.
///
/// `Ok(true)` means the raw tree is now present (or was already); `Ok(false)` means this set is not
/// one this build can fetch - a user set (nothing to fetch) or a format other than tar+gzip.
pub fn ensure_archive(found: &Found) -> Result<bool, Fail> {
    let Some(entry) = DATASETS.iter().find(|d| d.name == found.name) else { return Ok(false) };
    if entry.data.is_empty() {
        return Ok(false);
    }
    // Every URL must be a tar+gzip archive: a set that also lists another format is refused whole,
    // rather than fetched in part and left unusable.
    for url in entry.data {
        let base = url.rsplit('/').next().unwrap_or("archive");
        if !(base.ends_with(".tar.gz") || base.ends_with(".tgz") || base.ends_with(".gz")) {
            return Ok(false);
        }
    }
    let raw = found.dir.join("raw");
    std::fs::create_dir_all(&raw).or_else(|e| fail(1, format!("{}: {e}", raw.display())))?;

    // Whether anything is already unpacked. The loop is silent on a warm cache: every archive exists,
    // so nothing is downloaded, and the raw tree has directories, so nothing is re-extracted - which
    // is what the oracle prints (nothing) on a second run.
    let mut extracted = std::fs::read_dir(&raw)
        .map(|entries| entries.flatten().any(|entry| entry.path().is_dir()))
        .unwrap_or(false);
    for (index, url) in entry.data.iter().enumerate() {
        let base = url.rsplit('/').next().unwrap_or("archive");
        let archive = raw.join(format!("{}.{}", found.prefix, base));
        let downloaded = !archive.exists();
        if downloaded {
            eprintln!("sacreBLEU: Downloading {url} to {}", archive.display());
            download(url, &archive)?;
            let expected = entry.md5.get(index).copied().unwrap_or("");
            if !expected.is_empty() {
                let got = md5_of(&archive)?;
                if got != expected {
                    return fail(
                        1,
                        format!(
                            "md5 mismatch for {}: expected {expected}, got {got}",
                            archive.display()
                        ),
                    );
                }
            }
        }
        if downloaded || !extracted {
            // Measured: a plain `.gz` still prints the "Extracting" line, but it is left compressed on
            // disk and the parsers read it gzip-aware.
            eprintln!("sacreBLEU: Extracting {} to {}", archive.display(), raw.display());
            if base.ends_with(".tar.gz") || base.ends_with(".tgz") {
                extract_tar_gz(&archive, &raw)?;
            }
            extracted = true;
        }
    }
    Ok(true)
}

/// Download a set's archive by name, for `--download NAME`: the same two lines, and nothing else.
pub fn download_set(name: &str) -> Result<bool, Fail> {
    let Some(entry) = DATASETS.iter().find(|d| d.name == name) else { return Ok(false) };
    let found = Found {
        name: name.to_string(),
        prefix: name.replace('/', "_"),
        dir: crate::dataset::cache_root().join(name),
        pairs: Vec::new(),
    };
    // `--download` never reads fields, so an entry with no pairs still fetches its archive.
    let _ = entry;
    ensure_archive(&found)
}

/// A plain blocking GET, streamed to disk.
fn download(url: &str, dest: &Path) -> Result<(), Fail> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| Fail { code: 1, msg: format!("{url}: {e}") })?;
    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(dest)
        .or_else(|e| fail(1, format!("{}: {e}", dest.display())))?;
    std::io::copy(&mut reader, &mut file).or_else(|e| fail(1, format!("{}: {e}", dest.display())))?;
    Ok(())
}

fn md5_of(path: &Path) -> Result<String, Fail> {
    let mut file = std::fs::File::open(path).or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 1 << 16];
    loop {
        let read = file
            .read(&mut buffer)
            .or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_tar_gz(archive: &Path, into: &Path) -> Result<(), Fail> {
    let file =
        std::fs::File::open(archive).or_else(|e| fail(1, format!("{}: {e}", archive.display())))?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(decoder);
    tar.unpack(into).or_else(|e| fail(1, format!("{}: {e}", into.display())))?;
    Ok(())
}

/// The raw directory a set would extract into, for callers that only need to look.
pub fn raw_dir(found: &Found) -> PathBuf {
    found.dir.join("raw")
}
