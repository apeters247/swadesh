//! The plain-text family (`multi30k` and friends): the processed files are copies of the two files
//! the registry names for a pair.
//!
//! Measured on `multi30k/2017`, whose archive is already fetched on this host: the registry's
//! per-pair list is `[<source>, <reference>]` (`test_2017_flickr.en`, `test_2017_flickr.de`) and
//! the processed `src`/`ref` are those two files **with each line rstripped** - checked as
//! `[line.rstrip() for line in raw] == processed` for both columns of both pairs, so it is the whole
//! rule and not a one-off: the French reference has two lines ending in a space in `raw/` and none
//! in the processed file. Nothing is parsed; the two files are found by base name under `raw/` and
//! written out line by line, which is also the reading sacreBLEU gives every other text stream.
//!
//! A set whose pair names formats another parser owns (`.sgm`, `.xml`, `.tsv`) or an archive
//! (`.gz`, `.zip`) is left to that parser or refused, so this rule cannot claim a set by accident.

use std::path::PathBuf;

use crate::dataset::{self, Found};
use crate::datasets::DATASETS;
use crate::{fail, Fail};

/// Rebuild `found`'s processed `src`/`ref` for `pair` when the registry names two plain files.
pub fn rebuild(found: &Found, pair: &str) -> Result<bool, Fail> {
    let Some(entry) = DATASETS.iter().find(|d| d.name == found.name) else { return Ok(false) };
    let Some(index) = entry.langpairs.iter().position(|p| *p == pair) else { return Ok(false) };
    let paths = entry.paths.get(index).copied().unwrap_or(&[]);
    if paths.len() != 2 {
        return Ok(false);
    }
    if paths.iter().any(|path| {
        let lower = path.to_lowercase();
        lower.ends_with(".sgm")
            || lower.ends_with(".xml")
            || lower.ends_with(".tsv")
            || lower.ends_with(".gz")
            || lower.ends_with(".zip")
    }) {
        return Ok(false);
    }
    let raw = found.dir.join("raw");
    if !raw.is_dir() {
        return Ok(false);
    }
    let files: Vec<PathBuf> = dataset::walk_files(&raw);
    let find = |want: &str| -> Option<PathBuf> {
        let base = want.rsplit('/').next().unwrap_or(want).to_string();
        files
            .iter()
            .find(|path| path.file_name().map(|n| n == base.as_str()).unwrap_or(false))
            .cloned()
    };
    let (Some(source), Some(reference)) = (find(paths[0]), find(paths[1])) else {
        return Ok(false);
    };
    let declared = found.fields_of(pair).cloned().unwrap_or_default();
    for (field, from) in [("src", &source), ("ref", &reference)] {
        if !declared.iter().any(|d| d == field) {
            continue;
        }
        let text = std::fs::read_to_string(from)
            .or_else(|e| fail(1, format!("{}: {e}", from.display())))?;
        // Each line rstripped, joined by newlines, and one trailing newline - which is what
        // `[line.rstrip() for line in raw]` compares equal to, blank-line quirks included. Every
        // measured file ends with a newline, so the absent case is unmeasured and simply gets one.
        let mut lines: Vec<&str> = text.split('\n').collect();
        if text.ends_with('\n') {
            lines.pop();
        }
        let mut body = lines.iter().map(|line| line.trim_end()).collect::<Vec<_>>().join("\n");
        if !body.is_empty() {
            body.push('\n');
        }
        let to = dataset::field_path(found, pair, field);
        std::fs::write(&to, body).or_else(|e| fail(1, format!("{}: {e}", to.display())))?;
    }
    Ok(true)
}
