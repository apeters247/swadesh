//! The FakeSGML test-set family (`wmt08`..`wmt19` and their `B`/`tworefs`/`dev` variants): `<doc ...>`
//! blocks of `<seg>` lines.
//!
//! Measured from the oracle's own cache and raw tree, not from its source
//! (`docs/spec/test-set-parsers.md` has the rules). The archive holds
//! `<base>-<pair><dir>-{src,ref}.<lang>.sgm` files, each a series of
//! `<doc docid="..." genre="..." origlang="...">` blocks whose `<seg id="N">text</seg>` lines are
//! the segments in order. The processed files sacreBLEU writes - `<set>.<pair>.src|ref|docid|
//! origlang|genre`, one line per segment - are exactly that: the segment text, and the enclosing
//! doc's three attributes repeated for each of its segments.
//!
//! **The registry names the files, so nothing is guessed.** Every set in this family carries a
//! per-pair `paths` list whose entries are one `.sgm` file per `src`/`ref*` field, in field order:
//! `[src.sgm, ref.sgm]` for the plain sets and `wmt16/B`, `[src.sgm, ref0.sgm, ref1.sgm]` for
//! `tworefs` (`ref:0`, `ref:1`), and so on for `*.txt`-style variants that are deliberately left
//! refused (`wmt17/ms`'s extra references are `.txt`/`.gz`, `wmt19/google/*`'s reference is a plain
//! `.ref`, and `wmt21/B` is WMT XML and belongs to `wmtxml`) - the guard is "every path ends in
//! `.sgm` and there are as many paths as `src`/`ref*` fields", so a set whose shape is not this one
//! falls through to the ordinary refusal instead of being parsed wrongly.
//!
//! No XML entity occurs in this family's raw text (measured: the 13 `&`s in the wmt16 en-de source
//! are literal, and no `&name;` or `&#NN;` form appears at all), so segment text is taken verbatim.
//! That is *not* true of the WMT XML family, which is a different parser.

use std::path::PathBuf;

use crate::dataset::{self, Found};
use crate::datasets::DATASETS;
use crate::{fail, Fail};

/// One parsed `.sgm` file: the segments and their per-segment annotations.
pub struct Parsed {
    pub segs: Vec<String>,
    pub docids: Vec<String>,
    pub genres: Vec<String>,
    pub origlangs: Vec<String>,
}

/// The value of `name="..."` in a tag, if present.
fn attr(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse a `.sgm` file: `<doc>` attributes are carried onto every `<seg>` inside them.
pub fn parse(text: &str) -> Parsed {
    let mut parsed = Parsed {
        segs: Vec::new(),
        docids: Vec::new(),
        genres: Vec::new(),
        origlangs: Vec::new(),
    };
    let mut docid = String::new();
    let mut genre = String::new();
    let mut origlang = String::new();
    let mut rest = text;
    while let Some(at) = rest.find('<') {
        let tag_start = &rest[at..];
        if tag_start.starts_with("<doc ") || tag_start.starts_with("<doc>") {
            let end = match tag_start.find('>') {
                Some(end) => end,
                None => break,
            };
            let tag = &tag_start[..end];
            docid = attr(tag, "docid").unwrap_or_default();
            genre = attr(tag, "genre").unwrap_or_default();
            origlang = attr(tag, "origlang").unwrap_or_default();
            rest = &tag_start[end + 1..];
        } else if tag_start.starts_with("<seg ") || tag_start.starts_with("<seg>") {
            let Some(open_end) = tag_start.find('>') else { break };
            let after_open = &tag_start[open_end + 1..];
            let Some(close) = after_open.find("</seg>") else { break };
            parsed.segs.push(after_open[..close].to_string());
            parsed.docids.push(docid.clone());
            parsed.genres.push(genre.clone());
            parsed.origlangs.push(origlang.clone());
            rest = &after_open[close + 6..];
        } else {
            rest = &tag_start[1..];
        }
    }
    parsed
}

/// Write one field as `<value>\n` per segment.
fn write_field(found: &Found, pair: &str, field: &str, values: &[String]) -> Result<(), Fail> {
    let mut body = String::new();
    for value in values {
        body.push_str(value);
        body.push('\n');
    }
    let path = dataset::field_path(found, pair, field);
    std::fs::write(&path, body).or_else(|e| fail(1, format!("{}: {e}", path.display())))
}

/// Rebuild `found`'s processed files for `pair` from its raw tree, for the FakeSGML family.
///
/// Returns `Ok(false)` when this set or pair is not one the measured rule covers - another family's
/// set, a pair whose registry `paths` are not all `.sgm` or do not line up with its `src`/`ref*`
/// fields, or no raw tree at all - in which case the caller's usual refusal applies unchanged.
pub fn rebuild(found: &Found, pair: &str) -> Result<bool, Fail> {
    let raw_dir = found.dir.join("raw");
    if !raw_dir.is_dir() {
        return Ok(false);
    }
    let Some(entry) = DATASETS.iter().find(|d| d.name == found.name) else { return Ok(false) };
    let Some(index) = entry.langpairs.iter().position(|p| *p == pair) else { return Ok(false) };
    let paths = entry.paths.get(index).copied().unwrap_or(&[]);
    if paths.is_empty() || paths.iter().any(|p| !p.to_lowercase().ends_with(".sgm")) {
        return Ok(false);
    }
    let declared = found.fields_of(pair).cloned().unwrap_or_default();
    let file_backed: Vec<String> = declared
        .iter()
        .filter(|f| f.as_str() == "src" || f.starts_with("ref"))
        .cloned()
        .collect();
    if file_backed.len() != paths.len() {
        return Ok(false);
    }

    let files = dataset::walk_files(&raw_dir);
    let locate = |want: &str| -> Option<PathBuf> {
        let base = want.rsplit('/').next().unwrap_or(want).to_string();
        files
            .iter()
            .find(|path| path.file_name().map(|n| n == base.as_str()).unwrap_or(false))
            .cloned()
    };
    let mut parsed: Vec<(String, Parsed)> = Vec::new();
    for (field, want) in file_backed.iter().zip(paths.iter()) {
        let Some(path) = locate(want) else { return Ok(false) };
        let text = std::fs::read_to_string(&path)
            .or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
        parsed.push((field.clone(), parse(&text)));
    }
    let Some(source) = parsed.iter().find(|(field, _)| field == "src") else { return Ok(false) };
    let count = source.1.segs.len();
    if parsed.iter().any(|(_, p)| p.segs.len() != count) {
        return fail(
            1,
            format!("{}: the source and reference files have different segment counts", found.name),
        );
    }

    // Only the fields the set's catalogue entry declares, so a pair without `genre` gets none.
    for (field, p) in &parsed {
        if declared.iter().any(|d| d == field) {
            write_field(found, pair, field, &p.segs)?;
        }
    }
    let metadata = [
        ("docid", &source.1.docids),
        ("genre", &source.1.genres),
        ("origlang", &source.1.origlangs),
    ];
    for (name, values) in metadata {
        if declared.iter().any(|d| d == name) {
            write_field(found, pair, name, values)?;
        }
    }
    Ok(true)
}
