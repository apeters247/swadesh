//! The FakeSGML test-set family (`wmt08`..`wmt19` and friends): `<doc ...>` blocks of `<seg>` lines.
//!
//! Measured from the oracle's own cache and raw tree, not from its source. The archive holds
//! `<base>-<pair><dir>-{src,ref}.<lang>.sgm` files, each a series of
//! `<doc docid="..." genre="..." origlang="...">` blocks whose `<seg id="N">text</seg>` lines are
//! the segments in order. The processed files sacreBLEU writes - `<set>.<pair>.src|ref|docid|
//! origlang|genre`, one line per segment - are exactly that: the segment text, and the enclosing
//! doc's three attributes repeated for each of its segments. `docid`, `genre` and `origlang` in
//! `newstest2016-ende-src.en.sgm` reproduce the cached `.docid`/`.genre`/`.origlang` byte for byte,
//! which `tests/` checks by regenerating them into a scratch cache.
//!
//! No XML entity occurs in this family's raw text (measured: the 13 `&`s in the wmt16 en-de source
//! are literal, and no `&name;` or `&#NN;` form appears at all), so segment text is taken verbatim.
//! That is *not* true of the WMT XML family, which is a different parser.

use std::path::{Path, PathBuf};

use crate::dataset::{self, Found};
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

/// Every file under `dir`, recursively (the archives extract into nested directories).
fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// The processed files to write, in the order they are reported.
const FIELDS: &[&str] = &["src", "ref", "docid", "genre", "origlang"];

/// Rebuild `found`'s processed files for `pair` from its raw tree, for the SGML family.
///
/// Returns `Ok(false)` when this set or pair is not one the measured rule covers - a WMT XML set, a
/// variant whose archive layout has not been measured (`wmt16/B`, `wmt16/tworefs`, `wmt19/dev`, ...),
/// or no raw tree at all - in which case the caller's usual refusal applies unchanged.
pub fn rebuild(found: &Found, pair: &str) -> Result<bool, Fail> {
    // Only the plain `wmtNN` sets: their archive is `<base>-...sgm` with `base = newstest20NN`, and
    // the `B` variant of the same year (a separate reference, used by `wmt16/B` and `tworefs`) is a
    // different set whose layout is not measured here.
    let Some(digits) = found.name.strip_prefix("wmt") else { return Ok(false) };
    if digits.len() != 2 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return Ok(false);
    }
    let raw_dir = found.dir.join("raw");
    if !raw_dir.is_dir() {
        return Ok(false);
    }
    let Some((src_lang, trg_lang)) = pair.split_once('-') else { return Ok(false) };
    let base = format!("newstest20{digits}");
    let concat = format!("{src_lang}{trg_lang}");
    let want_src = format!("-{concat}-src.{src_lang}.sgm");
    let want_ref = format!("-{concat}-ref.{trg_lang}.sgm");

    let mut files = Vec::new();
    walk(&raw_dir, &mut files);
    let find = |suffix: &str| -> Option<&PathBuf> {
        files.iter().find(|path| {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            name.starts_with(&format!("{base}-")) && name.ends_with(suffix)
        })
    };
    let (Some(src_path), Some(ref_path)) = (find(&want_src), find(&want_ref)) else {
        return Ok(false);
    };

    let read = |path: &PathBuf| -> Result<String, Fail> {
        std::fs::read_to_string(path).or_else(|e| fail(1, format!("{}: {e}", path.display())))
    };
    let src = parse(&read(src_path)?);
    let reference = parse(&read(ref_path)?);
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
        if !FIELDS.contains(&field) || !declared.iter().any(|f| f == field) {
            continue;
        }
        let mut body = String::new();
        for value in values {
            body.push_str(value);
            body.push('\n');
        }
        let path = dataset::field_path(found, pair, field);
        std::fs::write(&path, body)
            .or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
    }
    Ok(true)
}
