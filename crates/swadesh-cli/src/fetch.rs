//! Fetching a test set's archive: the `downloader` cargo feature (PLAN.md D-2).
//!
//! Measured from a cold cache with a small set. sacreBLEU prints two lines and nothing else - no
//! progress bar, so the output is deterministic:
//!
//! ```text
//! sacreBLEU: Downloading <url> to <cache>/<set>/raw/<prefix>.<archive basename>
//! sacreBLEU: Extracting <archive> to <cache>/<set>/raw
//! ```
//!
//! and the archive name is the file-name prefix (the set name with `/` written `_`) followed by the
//! URL's basename. Extraction leaves the archive's own directory structure under `raw/`, which is
//! what the parsers in `sgml` and `wmtxml` then read.
//!
//! **What this does not cover yet, and refuses explicitly.** The catalogue stores a *flat* list of
//! URLs per set, so which archive belongs to a particular language pair is not recoverable from it;
//! 36 of the 67 sets have more than one archive and they are refused rather than guessed at (the
//! per-pair mapping is what the generator needs next). Formats other than `.tar.gz`/`.tgz` - four
//! `.zip` sets and one plain-`.gz` set - are refused for the same reason. And the md5-mismatch
//! message is ours, not measured: producing a corrupted download needs a fake server, so sacreBLEU's
//! wording for that case is still unknown (`docs/parity-report.md` records it).

use std::io::Read;
use std::path::{Path, PathBuf};

use md5::{Digest, Md5};

use crate::dataset::Found;
use crate::datasets::DATASETS;
use crate::{fail, Fail};

/// Download the set's archive when it is missing, verify its md5, and extract it.
///
/// `Ok(true)` means the raw tree is now present; `Ok(false)` means this set is not one this build can
/// fetch - a user set (nothing to fetch), a multi-archive set, or a format other than tar+gzip.
pub fn ensure_archive(found: &Found) -> Result<bool, Fail> {
    let Some(entry) = DATASETS.iter().find(|d| d.name == found.name) else { return Ok(false) };
    if entry.data.is_empty() {
        return Ok(false);
    }
    if entry.data.len() > 1 {
        return Ok(false);
    }
    let url = entry.data[0];
    let base = url.rsplit('/').next().unwrap_or("archive");
    if !(base.ends_with(".tar.gz") || base.ends_with(".tgz")) {
        return Ok(false);
    }
    let raw = found.dir.join("raw");
    std::fs::create_dir_all(&raw).or_else(|e| fail(1, format!("{}: {e}", raw.display())))?;
    let archive = raw.join(format!("{}.{}", found.prefix, base));

    if !archive.exists() {
        eprintln!("sacreBLEU: Downloading {url} to {}", archive.display());
        download(url, &archive)?;
    }
    let expected = entry.md5.first().copied().unwrap_or("");
    if !expected.is_empty() {
        let got = md5_of(&archive)?;
        if got != expected {
            return fail(
                1,
                format!("md5 mismatch for {}: expected {expected}, got {got}", archive.display()),
            );
        }
    }
    let extracted = std::fs::read_dir(&raw)
        .map(|entries| entries.flatten().any(|entry| entry.path().is_dir()))
        .unwrap_or(false);
    if !extracted {
        eprintln!("sacreBLEU: Extracting {} to {}", archive.display(), raw.display());
        extract_tar_gz(&archive, &raw)?;
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
