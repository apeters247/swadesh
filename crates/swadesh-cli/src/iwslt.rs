//! The IWSLT XML family (`iwslt17` and its `dev2010`/`tst2010`..`tst2016` subsets): TED `<mteval>`
//! XML.
//!
//! Measured from the oracle's own cache and raw tree, not from its source (`docs/spec/test-set-parsers.md`
//! has the full rule and the measurement). The archive holds one directory per direction
//! (`en-fr/`, `fr-en/`, ...) of `IWSLT17.TED.<subset>.<pair>.<lang>.xml` files, each a series of
//! `<doc docid="..." genre="...">` blocks of `<seg id="N">text</seg>` lines.
//!
//! **Two things distinguish this family from the others.**
//!
//! First, the segment text is **whitespace-collapsed**, not merely stripped: the rule is
//! `" ".join(text.split())`, and it is the whole rule - a plain strip leaves 819 of `iwslt17`
//! en-fr's 1455 source lines wrong, because the TED segments carry doubled internal spaces
//! (and Unicode spaces, NBSP and U+3000, which both Python's `split()` and Rust's
//! `split_whitespace` treat as whitespace, so the two agree).
//!
//! Second, **no XML entity is decoded**: the one `&`-bearing segment in the whole family
//! (`... Bed Bath &amp; Beyond ...`) is written out with `&amp;` literal, so sacreBLEU is not
//! running an XML parser over the text and neither may swadesh. The text is simply the bytes
//! between `<seg ...>` and `</seg>`.
//!
//! `origlang` comes from the enclosing `<doc>` (this family has no such attribute), so every
//! `origlang` line is blank. `docid` and `genre` are the `<doc>`'s, repeated onto each segment.

use std::path::{Path, PathBuf};

use crate::dataset::{self, Found};
use crate::datasets::DATASETS;
use crate::{fail, Fail};

/// One parsed TED XML file: the segments and their per-segment annotations.
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

/// `" ".join(text.split())`: strip and collapse every run of whitespace to one space.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Parse a TED XML file: `<doc>` attributes are carried onto every `<seg>` inside them, and the
/// segment text is taken verbatim (no entity decoding) and collapsed.
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
            parsed.segs.push(collapse(&after_open[..close]));
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

/// Rebuild `found`'s processed files for `pair` from its raw tree, for the IWSLT family.
///
/// Returns `Ok(false)` when this set or pair is not one the measured rule covers - another family's
/// set, a pair whose registry paths are not the two `.xml` files, or no raw tree at all - in which
/// case the caller's usual refusal applies unchanged.
pub fn rebuild(found: &Found, pair: &str) -> Result<bool, Fail> {
    if !found.name.starts_with("iwslt") {
        return Ok(false);
    }
    let Some(entry) = DATASETS.iter().find(|d| d.name == found.name) else { return Ok(false) };
    let Some(index) = entry.langpairs.iter().position(|p| *p == pair) else { return Ok(false) };
    let paths = entry.paths.get(index).copied().unwrap_or(&[]);
    if paths.len() < 2 || paths.iter().any(|p| !p.to_lowercase().ends_with(".xml")) {
        return Ok(false);
    }
    let raw_dir = found.dir.join("raw");
    if !raw_dir.is_dir() {
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
    let (Some(src_path), Some(ref_path)) = (locate(paths[0]), locate(paths[1])) else {
        return Ok(false);
    };

    let read = |path: &Path| -> Result<String, Fail> {
        std::fs::read_to_string(path).or_else(|e| fail(1, format!("{}: {e}", path.display())))
    };
    let src = parse(&read(&src_path)?);
    let reference = parse(&read(&ref_path)?);
    if src.segs.len() != reference.segs.len() {
        return fail(
            1,
            format!(
                "{} and {} have {} and {} segments",
                src_path.display(),
                ref_path.display(),
                src.segs.len(),
                reference.segs.len()
            ),
        );
    }

    // Only the fields the set's catalogue entry declares, so a pair without `genre` gets none.
    let declared = found.fields_of(pair).cloned().unwrap_or_default();
    let columns: [(&str, &Vec<String>); 5] = [
        ("src", &src.segs),
        ("ref", &reference.segs),
        ("docid", &src.docids),
        ("genre", &src.genres),
        ("origlang", &src.origlangs),
    ];
    for (field, values) in columns {
        if !declared.iter().any(|f| f == field) {
            continue;
        }
        let mut body = String::new();
        for value in values {
            body.push_str(value);
            body.push('\n');
        }
        let path = dataset::field_path(found, pair, field);
        std::fs::write(&path, body).or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
    }
    Ok(true)
}
