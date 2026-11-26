//! The TSV family (`mtnt1.1/*`, `mtnt2019`): one tab-separated file per direction, and each field
//! is a **column** of it.
//!
//! Measured from the oracle's cache and raw tree (`docs/spec/test-set-parsers.md` has the rule and
//! the measurement). The registry's per-pair `paths` are `"<col>:<file>"` - the *same* file twice,
//! once per field - and the integer is a **0-based column index**: `mtnt1.1` writes `1:` for `src`
//! and `2:` for `ref` over a three-column `id<TAB>src<TAB>tgt` file, `mtnt2019` writes `2:`/`3:` over
//! its four-column file.
//!
//! **The `\r` is part of the value.** Some files are CRLF: `mtnt1.1/test/test.en-fr.tsv` ends every
//! line `...Kemba\r\n`, so the source column (a middle column) has no `\r` but the target column (the
//! last) keeps it, and the processed `.ref` file is itself CRLF (`...Kemba\r\n`) while `.src` is LF.
//! The rule is therefore "split the text on `\n`, split each line on `\t`, take the column verbatim" -
//! no `\r` stripping anywhere - which reproduces both the LF and the CRLF files. (`mtnt1.1`'s en-ja
//! files are LF-only, so their `.ref` is LF; both shapes come out of the one rule.)

use crate::dataset::{self, Found};
use crate::datasets::DATASETS;
use crate::{fail, Fail};

/// `"<col>:<file>"` -> `(column, file)`.
fn spec(s: &str) -> Option<(usize, &str)> {
    let (col, file) = s.split_once(':')?;
    Some((col.parse().ok()?, file))
}

/// Rebuild `found`'s processed `src`/`ref` for `pair` from its TSV, for the TSV family.
///
/// Returns `Ok(false)` when this set or pair is not one the measured rule covers - another family's
/// set, a pair whose registry paths are not two `"<col>:<file>"` specs of the same file, or no raw
/// tree at all - in which case the caller's usual refusal applies unchanged.
pub fn rebuild(found: &Found, pair: &str) -> Result<bool, Fail> {
    if !found.name.starts_with("mtnt") {
        return Ok(false);
    }
    let Some(entry) = DATASETS.iter().find(|d| d.name == found.name) else { return Ok(false) };
    let Some(index) = entry.langpairs.iter().position(|p| *p == pair) else { return Ok(false) };
    let specs = entry.paths.get(index).copied().unwrap_or(&[]);
    if specs.len() != 2 {
        return Ok(false);
    }
    let (Some((src_col, src_file)), Some((ref_col, ref_file))) = (spec(specs[0]), spec(specs[1]))
    else {
        return Ok(false);
    };
    if !src_file.to_lowercase().ends_with(".tsv") || src_file != ref_file {
        return Ok(false);
    }
    let raw_dir = found.dir.join("raw");
    if !raw_dir.is_dir() {
        return Ok(false);
    }
    let base = src_file.rsplit('/').next().unwrap_or(src_file).to_string();
    let files = dataset::walk_files(&raw_dir);
    let Some(tsv) = files
        .iter()
        .find(|path| path.file_name().map(|n| n == base.as_str()).unwrap_or(false))
        .cloned()
    else {
        return Ok(false);
    };

    let text =
        std::fs::read_to_string(&tsv).or_else(|e| fail(1, format!("{}: {e}", tsv.display())))?;
    let mut lines: Vec<&str> = text.split('\n').collect();
    if text.ends_with('\n') {
        lines.pop();
    }
    let column = |index: usize| -> Vec<String> {
        lines
            .iter()
            .map(|line| line.split('\t').nth(index).unwrap_or("").to_string())
            .collect()
    };

    let declared = found.fields_of(pair).cloned().unwrap_or_default();
    for (field, index) in [("src", src_col), ("ref", ref_col)] {
        if !declared.iter().any(|f| f == field) {
            continue;
        }
        let mut body = String::new();
        for value in column(index) {
            body.push_str(&value);
            body.push('\n');
        }
        let path = dataset::field_path(found, pair, field);
        std::fs::write(&path, body).or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
    }
    Ok(true)
}

