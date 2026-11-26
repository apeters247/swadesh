//! The WMT XML family (`wmt20`..`wmt24`): `<doc>` blocks containing `<src>`, `<ref>` and `<hyp>`
//! streams of `<seg>` lines.
//!
//! Measured from the oracle's cache and raw tree. The archive holds
//! `xml/wmttest20<NN>.<pair>.all.xml` (for wmt24, `raw/xml/...` directly), whose structure is
//!
//! ```text
//! <dataset id="wmttest2022">
//!   <collection id="general">
//!     <doc origlang="de" id="en_de_CLIENT-01_..._doc08" domain="conversation">
//!       <src lang="de">      <p><seg id="1">text</seg></p> </src>
//!       <ref lang="fr" translator="A"> ... </ref>
//!       <hyp system="Online-A" lang="fr"> ... </hyp>
//! ```
//!
//! so the segment text comes from the `<seg>` lines, `docid`/`origlang`/`domain` from the enclosing
//! `<doc>`, a reference is named `ref:<translator>`, and a system is named by `<hyp system=...>`.
//!
//! **Two measured traps.** The `txt/` tree beside the XML is *not* the source of the processed
//! files: `txt/sources/generaltest2022.de-fr.src.de` differs from the processed `.src` in whitespace
//! ("von  ihrem" against "von ihrem"), and `txt/system-outputs/...hyp.Online-A.fr` differs from the
//! processed `.Online-A` the same way, while `LT22`'s happens to be identical there. The XML is the
//! source of truth for every stream, so this parser ignores `txt/` entirely - a copy-based shortcut
//! would have been wrong for some systems and right for others.
//!
//! Entities: `&amp;` and `&gt;` both occur and are unescaped to `&` and `>`; no nested form
//! (`&amp;gt;`, `&amp;quot;`) occurs anywhere in the cached wmt22/wmt24 XML, so a single pass is
//! exact for the measured data and a nested form is left as a gap rather than guessed at.

use std::path::{Path, PathBuf};

use crate::dataset::{self, Found};
use crate::{fail, Fail};

/// One stream of a document set: its name (`src`, `ref:A`, `Online-A`) and its segments.
struct Stream {
    name: String,
    segs: Vec<String>,
    docids: Vec<String>,
    origlangs: Vec<String>,
    domains: Vec<String>,
}

/// Normalise a segment: unescape the two entities this family uses, then collapse whitespace.
///
/// The collapse is measured, not guessed: `newstest2022` de-fr has `von  ihrem` with two spaces in
/// the XML and one in the processed `.src`, a segment containing a literal tab comes out with a
/// single space (`wmt24.en-de.src` holds no tab at all), and the segments that span lines in the XML
/// come out as one line. `split()` + `join(" ")` reproduces all three, and Rust and CPython agree on
/// what counts as whitespace for it (both follow the Unicode White_Space property, non-breaking
/// space included), which the byte-diff against the oracle checks.
fn normalise(text: &str) -> String {
    if !text.contains('&') {
        return text.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    // `&amp;` is replaced **last**, so a nested form keeps its inner entity literal. Measured on the
    // three nested forms the cached wmt24 XML contains: `A&amp;E;` becomes `A&E;`, `AT&amp;amp;T`
    // becomes `AT&amp;T`, and `Grey&#39;` (from `&amp;#39;`) stays `&#39;` - all three in the
    // oracle's own processed files. `&quot;`, `&apos;` and numeric forms never occur in the cached
    // wmt22/wmt24 XML, so they are left alone rather than guessed at.
    let text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&");
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse the XML into streams, in the order the segments appear.
fn parse(text: &str) -> Vec<Stream> {
    let mut streams: Vec<Stream> = Vec::new();
    let mut current: Option<usize> = None;
    // Documents carrying a `testsuite` attribute are the diagnostic suites, not the scored data:
    // wmt22 en-de has 17,618 of them and the oracle's processed files contain no `tsuite` id at all
    // (they are 2037 lines, the news part). Every non-testsuite doc in that file has a `domain`,
    // so the two candidate rules coincide there; this one is the explicit one.
    let mut skip_doc = false;
    let mut docid = String::new();
    let mut origlang = String::new();
    let mut domain = String::new();
    let mut rest = text;
    while let Some(at) = rest.find('<') {
        let tail = &rest[at..];
        if tail.starts_with("<doc ") || tail.starts_with("<doc>") {
            let Some(end) = tail.find('>') else { break };
            let tag = &tail[..end];
            docid = attr(tag, "id").unwrap_or_default();
            origlang = attr(tag, "origlang").unwrap_or_default();
            domain = attr(tag, "domain").unwrap_or_default();
            skip_doc = tag.contains("testsuite=\"");
            current = None;
            rest = &tail[end + 1..];
        } else if tail.starts_with("</doc>") {
            current = None;
            skip_doc = false;
            rest = &tail[6..];
        } else if tail.starts_with("<src") || tail.starts_with("<ref") || tail.starts_with("<hyp") {
            let Some(end) = tail.find('>') else { break };
            let tag = &tail[..end];
            let name = if tail.starts_with("<src") {
                "src".to_string()
            } else if tail.starts_with("<ref") {
                match attr(tag, "translator") {
                    Some(translator) => format!("ref:{translator}"),
                    None => "ref".to_string(),
                }
            } else {
                attr(tag, "system").unwrap_or_else(|| "hyp".to_string())
            };
            // A file repeats every stream once per `<doc>`, so streams of the same name are merged
            // in document order: one `src` for the whole corpus, not one per document.
            let index = match streams.iter().position(|stream| stream.name == name) {
                Some(index) => index,
                None => {
                    streams.push(Stream {
                        name,
                        segs: Vec::new(),
                        docids: Vec::new(),
                        origlangs: Vec::new(),
                        domains: Vec::new(),
                    });
                    streams.len() - 1
                }
            };
            current = Some(index);
            rest = &tail[end + 1..];
        } else if tail.starts_with("<seg ") || tail.starts_with("<seg>") {
            let Some(open_end) = tail.find('>') else { break };
            // A self-closing `<seg .../>` is an empty segment - 89 of them in wmt24 en-de. Without
            // this the search for `</seg>` runs on to the *next* segment's tag and swallows it, so
            // the stream came out one line short and with a literal `<seg id=...>` in the text.
            if tail[..open_end].ends_with('/') {
                if !skip_doc {
                    if let Some(stream) = current.and_then(|index| streams.get_mut(index)) {
                        stream.segs.push(String::new());
                        stream.docids.push(docid.clone());
                        stream.origlangs.push(origlang.clone());
                        stream.domains.push(domain.clone());
                    }
                }
                rest = &tail[open_end + 1..];
                continue;
            }
            let after_open = &tail[open_end + 1..];
            let Some(close) = after_open.find("</seg>") else { break };
            if skip_doc {
                rest = &after_open[close + 6..];
                continue;
            }
            if let Some(stream) = current.and_then(|index| streams.get_mut(index)) {
                stream.segs.push(normalise(&after_open[..close]));
                stream.docids.push(docid.clone());
                stream.origlangs.push(origlang.clone());
                stream.domains.push(domain.clone());
            }
            rest = &after_open[close + 6..];
        } else if tail.starts_with("</") {
            let Some(end) = tail.find('>') else { break };
            if tail.starts_with("</src") || tail.starts_with("</ref") || tail.starts_with("</hyp") {
                current = None;
            }
            rest = &tail[end + 1..];
        } else {
            rest = &tail[1..];
        }
    }
    streams
}

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

/// Rebuild `found`'s processed files for `pair` from the cached WMT XML, when the family applies.
pub fn rebuild(found: &Found, pair: &str) -> Result<bool, Fail> {
    let Some(digits) = found.name.split('/').next().and_then(|n| n.strip_prefix("wmt")) else {
        return Ok(false);
    };
    if digits.len() != 2 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return Ok(false);
    }
    let raw_dir = found.dir.join("raw");
    if !raw_dir.is_dir() {
        return Ok(false);
    }
    let wanted = format!("wmttest20{digits}.{pair}.all.xml");
    let mut files = Vec::new();
    walk(&raw_dir, &mut files);
    let Some(xml_path) = files
        .iter()
        .find(|path| path.file_name().map(|n| n == wanted.as_str()).unwrap_or(false))
    else {
        return Ok(false);
    };

    let text = std::fs::read_to_string(xml_path)
        .or_else(|e| fail(1, format!("{}: {e}", xml_path.display())))?;
    let streams = parse(&text);
    let Some(src) = streams.iter().find(|s| s.name == "src") else { return Ok(false) };
    if src.segs.is_empty() {
        return Ok(false);
    }

    // Only the fields the catalogue declares for this pair; a stream the catalogue does not list is
    // not written, and a declared field with no stream leaves the usual refusal to fire.
    let declared = found.fields_of(pair).cloned().unwrap_or_default();
    let writes: Vec<&Stream> = streams
        .iter()
        .filter(|stream| declared.iter().any(|d| d == &stream.name))
        .collect();
    if writes.is_empty() {
        return Ok(false);
    }
    for stream in &writes {
        let mut body = String::new();
        for seg in &stream.segs {
            body.push_str(seg);
            body.push('\n');
        }
        let path = dataset::field_path(found, pair, &stream.name);
        std::fs::write(&path, body).or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
    }
    // The annotations repeat per segment, taken from the `src` stream (identical inside a doc).
    for (field, values) in [
        ("docid", &src.docids),
        ("origlang", &src.origlangs),
        ("domain", &src.domains),
    ] {
        if !declared.iter().any(|d| d == field) {
            continue;
        }
        let mut column = String::new();
        for value in values {
            column.push_str(value);
            column.push('\n');
        }
        let path = dataset::field_path(found, pair, field);
        std::fs::write(&path, column).or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
    }
    Ok(true)
}
