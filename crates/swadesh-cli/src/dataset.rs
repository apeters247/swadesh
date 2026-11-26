//! The test-set data layer: resolve `-t NAME` to the files sacreBLEU's own cache holds.
//!
//! The layout is measured, not invented, and it is sacreBLEU's *processed* cache:
//!
//! ```text
//! <root>/<name>/<prefix>.<pair>.<field>      wmt16/wmt16.en-de.src
//!                                             wmt21/AB/wmt21_AB.en-de.ref-A
//! ```
//!
//! where a slash in the set name becomes a directory, an underscore in the file prefix
//! (`wmt21/AB` -> `wmt21_AB`), and a *reference* field's colon becomes a dash (`ref:A` ->
//! `.ref-A`). A field that is not a reference keeps its name verbatim (`Online-A`,
//! `NVIDIA-NeMo`), which is why the two transformations are separate: with one rule, `ref-A` and a
//! system called `Online-A` could not be told apart.
//!
//! Because the processed cache is exactly this layout, a directory of those files is a usable test
//! set with no download and no parser - which is what `$SWADESH_TEST_SETS` adds. See
//! `docs/spec/user-test-sets.md` and `PLAN.md` D-6.
//!
//! **Why the field order here is ours, not the oracle's.** sacreBLEU's own field list is not
//! reproducible: `-t wmt22 --list` prints `src, ref:A, ref:B` on one run and `src, ref:B, ref:A`
//! on the next (6/6 over 12 runs), the order following a hash-randomised mapping. The *default
//! reference* is stable, and it is the plain `ref` when the set has one, else the first `ref:*`
//! alphabetically - measured against wmt22 en-de (`ref:A`), wmt22 cs-en (`ref:B`), wmt24 en-de
//! (`ref:refA`) and wmt21/AB (`ref:A`). This module therefore orders fields deterministically and
//! reports the set, never the oracle's coin toss; the divergence is registered in
//! `docs/parity-report.md`.

use std::path::{Path, PathBuf};

use crate::datasets::DATASETS;
use crate::{fail, Fail};

/// A set that was found: its pairs and, for each, the fields in the order they are reported.
pub struct Found {
    /// The name to report (`wmt16`, `wmt21/AB`, or a directory's own name).
    pub name: String,
    /// The file-name prefix: the set name with `/` written as `_`.
    pub prefix: String,
    /// Where the set's files live.
    pub dir: PathBuf,
    /// `(langpair, [field, ...])`, in order.
    pub pairs: Vec<(String, Vec<String>)>,
}

/// `$SACREBLEU`, else `~/.sacrebleu`: where sacreBLEU keeps its cache.
pub fn cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("SACREBLEU") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join(".sacrebleu")
}

/// `$SWADESH_TEST_SETS`, colon-separated (swadesh's own extension).
pub fn user_roots() -> Vec<PathBuf> {
    std::env::var("SWADESH_TEST_SETS")
        .map(|v| v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default()
}

/// `ref:A` is stored as `ref-A`; every other field name is stored verbatim.
pub fn field_suffix(field: &str) -> String {
    match field.strip_prefix("ref:") {
        Some(rest) => format!("ref-{rest}"),
        None => field.to_string(),
    }
}

/// The inverse: `ref-A` is the field `ref:A`, anything else is itself.
pub fn field_name(suffix: &str) -> String {
    match suffix.strip_prefix("ref-") {
        Some(rest) => format!("ref:{rest}"),
        None => suffix.to_string(),
    }
}

/// `<dir>/<prefix>.<pair>.<suffix>`.
pub fn field_path(found: &Found, pair: &str, field: &str) -> PathBuf {
    found.dir.join(format!("{}.{}.{}", found.prefix, pair, field_suffix(field)))
}

/// The order a *discovered* (user) set's fields are reported in: the source, the reference and
/// its numbered variants (alphabetically), then the metadata in the order the oracle itself
/// prints it (`docid, genre, origlang` - stable over 12 runs on wmt16), then the rest
/// alphabetically. A built-in set keeps its declared order instead, minus the one part of it
/// the oracle does not reproduce (see `sort_refs_in_place`).
const METADATA: &[&str] = &["docid", "genre", "origlang"];

pub fn order_fields(names: Vec<String>) -> Vec<String> {
    fn key(f: &String) -> (u8, String) {
        if f == "src" {
            (0, String::new())
        } else if f == "ref" {
            (1, String::new())
        } else if f.starts_with("ref:") {
            (2, f.clone())
        } else if let Some(i) = METADATA.iter().position(|m| f == m) {
            (3, format!("{i}{f}"))
        } else {
            (4, f.clone())
        }
    }
    let mut names = names;
    names.sort_by_key(key);
    names.dedup();
    names
}

/// Sort only the `ref:*` subgroup, in place. The oracle's field listing is stable everywhere
/// except the relative order of numbered references, which flips run to run (`-t wmt22 --list`
/// printed `ref:A, ref:B` six times and `ref:B, ref:A` six times over twelve runs, while the
/// systems and metadata kept their order). Sorting just that subgroup is therefore the closest a
/// reproducible implementation can get, and it makes `--echo ref` - the stable default - the first
/// entry.
fn sort_refs_in_place(fields: &mut [String]) {
    let slots: Vec<usize> = (0..fields.len()).filter(|&i| fields[i].starts_with("ref")).collect();
    let mut refs: Vec<String> = slots.iter().map(|&i| fields[i].clone()).collect();
    refs.sort();
    for (slot, value) in slots.into_iter().zip(refs) {
        fields[slot] = value;
    }
}

/// Read a set's fields out of a directory: `<prefix>.<pair>.<suffix>` files.
fn discover(dir: &Path, prefix: &str) -> Found {
    let mut pairs: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let file = entry.file_name().to_string_lossy().into_owned();
            let Some(rest) = file.strip_prefix(&format!("{prefix}.")) else { continue };
            let Some((pair, suffix)) = rest.split_once('.') else { continue };
            pairs.entry(pair.to_string()).or_default().push(field_name(suffix));
        }
    }
    Found {
        name: prefix.to_string(),
        prefix: prefix.to_string(),
        dir: dir.to_path_buf(),
        pairs: pairs.into_iter().map(|(pair, fields)| (pair, order_fields(fields))).collect(),
    }
}

/// Look a set up. The catalogue wins, so every sacreBLEU name keeps its behaviour; then a
/// directory (given as a path, or under a user root, or under sacreBLEU's cache). `None` means "no
/// such set": `--list` reports that by printing nothing, and everything else as an unknown set.
pub fn lookup(name: &str) -> Option<Found> {
    if name.is_empty() {
        // An empty name is not a path and not a set; without this it would resolve to the root.
        return None;
    }
    if let Some(ds) = DATASETS.iter().find(|d| d.name == name) {
        return Some(Found {
            name: name.to_string(),
            prefix: name.replace('/', "_"),
            dir: cache_root().join(name),
            pairs: ds
                .langpairs
                .iter()
                .zip(ds.fields.iter())
                .map(|(pair, fields)| {
                    let mut names: Vec<String> = fields.iter().map(|s| s.to_string()).collect();
                    sort_refs_in_place(&mut names);
                    (pair.to_string(), names)
                })
                .collect(),
        });
    }
    let candidates: Vec<PathBuf> = if name.contains('/') {
        vec![PathBuf::from(name)]
    } else {
        user_roots()
            .into_iter()
            .chain(std::iter::once(cache_root()))
            .map(|root| root.join(name))
            .collect()
    };
    for dir in candidates {
        if dir.is_dir() {
            let base = dir.file_name().and_then(|s| s.to_str()).unwrap_or(name).to_string();
            return Some(discover(&dir, &base));
        }
    }
    None
}

impl Found {
    /// The fields of `pair`, if the set has that language pair.
    pub fn fields_of(&self, pair: &str) -> Option<&Vec<String>> {
        self.pairs.iter().find(|(p, _)| p == pair).map(|(_, f)| f)
    }

    /// The reference sacreBLEU scores against, and what `--echo ref` returns: the plain `ref` when
    /// the set has one, else the first `ref:*` alphabetically. Stable, unlike the oracle's field
    /// *list* order (see the module docs).
    pub fn default_reference(&self, pair: &str) -> Option<String> {
        let fields = self.fields_of(pair)?;
        fields
            .iter()
            .find(|f| f.as_str() == "ref")
            .or_else(|| fields.iter().find(|f| f.starts_with("ref")))
            .cloned()
    }
}

/// The unknown-test-set refusal, as measured from the oracle (`--list` prints nothing instead).
pub fn unknown_set(name: &str) -> Fail {
    eprintln!("sacreBLEU: Unknown test set '{name}'");
    eprintln!("sacreBLEU: Please run with --list to see the available test sets.");
    Fail { code: 1, msg: String::new() }
}

/// `sacreBLEU: No such language pair 'x'` plus the pairs that do exist, as measured.
pub fn unknown_pair(name: &str, pair: &str, available: &[String]) -> Fail {
    eprintln!("sacreBLEU: No such language pair '{pair}'");
    eprintln!("sacreBLEU: Available language pairs for '{name}' are:");
    for p in available {
        eprintln!("sacreBLEU:  > {p}");
    }
    Fail { code: 1, msg: String::new() }
}

/// `sacreBLEU: No such field x in test set S for language pair P.`, the fields that do exist, and -
/// when the pair has numbered references - the measured line that `ref` is still accepted and what
/// it returns.
pub fn unknown_field(
    name: &str,
    pair: &str,
    field: &str,
    available: &[String],
    default_ref: Option<&str>,
) -> Fail {
    eprintln!("sacreBLEU: No such field {field} in test set {name} for language pair {pair}.");
    eprintln!("sacreBLEU: available fields for {name}/{pair}: {}", available.join(", "));
    if let Some(default) = default_ref {
        if available.iter().any(|f| f.starts_with("ref:")) {
            eprintln!("sacreBLEU: 'ref' also allowed for backwards compatibility (will return {default})");
        }
    }
    Fail { code: 1, msg: String::new() }
}

/// The segments a set's `-ol`/`--subset` filters select, as indices into the unfiltered set.
///
/// Measured: `-ol en` keeps `origlang == en` (1500 of wmt16 en-de's 2999 segments) and `non-en`
/// keeps the rest; a value starting with `non-` therefore inverts. `--subset X` keeps
/// `domain == X` (wmt22 en-de: conversation 484, social 512, news 511, ecommerce 530, lower case).
/// A set without the annotation is an error in the oracle's own words; `--subset` has no fallback
/// to `genre` (wmt16 has `genre` and still refuses).
pub fn segment_filter(
    found: &Found,
    pair: &str,
    origlang: Option<&str>,
    subset: Option<&str>,
) -> Result<Vec<usize>, Fail> {
    let fields = found.fields_of(pair).ok_or_else(|| {
        Fail { code: 1, msg: format!("unknown language pair {pair} for test set {}", found.name) }
    })?;
    let origlangs = match origlang {
        None => None,
        Some(wanted) => {
            if !fields.iter().any(|f| f == "origlang") {
                eprintln!("Exception: No origlang annotation available for test set {}", found.name);
                return Err(Fail { code: 1, msg: String::new() });
            }
            let path = field_path(found, pair, "origlang");
            if !path.exists() {
                return Err(not_cached(found, pair, "origlang", &path));
            }
            Some(read_field(&path)?)
        }
    };
    let domains = match subset {
        None => None,
        Some(_) => {
            if !fields.iter().any(|f| f == "domain") {
                eprintln!("Exception: No subset annotation available for test set {}", found.name);
                return Err(Fail { code: 1, msg: String::new() });
            }
            let path = field_path(found, pair, "domain");
            if !path.exists() {
                return Err(not_cached(found, pair, "domain", &path));
            }
            Some(read_field(&path)?)
        }
    };
    let len = origlangs.as_ref().map(|l| l.len()).or_else(|| domains.as_ref().map(|l| l.len())).unwrap_or(0);
    let mut keep = Vec::new();
    for i in 0..len {
        if let (Some(langs), Some(wanted)) = (&origlangs, origlang) {
            let actual = langs.get(i).map(String::as_str).unwrap_or("");
            let ok = match wanted.strip_prefix("non-") {
                Some(rest) => actual != rest,
                None => actual == wanted,
            };
            if !ok {
                continue;
            }
        }
        if let (Some(doms), Some(wanted)) = (&domains, subset) {
            let actual = doms.get(i).map(String::as_str).unwrap_or("");
            if actual != wanted {
                continue;
            }
        }
        keep.push(i);
    }
    Ok(keep)
}

/// The oracle's "no sentence matched" line, with its formatting quirk reproduced: the subset value
/// is printed **twice** (`--subset zz` gives `subset=zzzz`, measured), because sacreBLEU
/// concatenates it instead of formatting it.
pub fn no_sentence(name: &str, origlang: Option<&str>, subset: Option<&str>) -> Fail {
    let mut parts = Vec::new();
    if let Some(lang) = origlang {
        parts.push(format!("origlang={lang}"));
    }
    if let Some(value) = subset {
        parts.push(format!("subset={value}{value}"));
    }
    eprintln!("sacreBLEU: Test set '{name}' contains no sentence with {}", parts.join(" "));
    Fail { code: 1, msg: String::new() }
}

/// The catalogue's citation for a built-in set. sacreBLEU 2.6.0 crashes on `--citation -t X`
/// (`TypeError: argument of type 'WMTXMLDataset' is not iterable`), so printing this is a
/// documented divergence rather than a reproduction of its traceback.
pub fn citation_of(name: &str) -> Option<String> {
    DATASETS.iter().find(|d| d.name == name).and_then(|d| d.citation).map(str::to_string)
}

/// The set is known but its files are not in the cache. Until the downloader lands (PLAN.md
/// WS0.3) this is a refusal with a clear reason, recorded as a temporary divergence in
/// `docs/parity-report.md`; sacreBLEU would fetch the set instead.
pub fn not_cached(found: &Found, pair: &str, field: &str, path: &Path) -> Fail {
    eprintln!(
        "swadesh: error: test set '{}' has no '{}' for language pair {} at {}",
        found.name,
        field,
        pair,
        path.display()
    );
    eprintln!(
        "swadesh: error: this build has no downloader yet, so fetch the set first (PLAN.md WS0.3)"
    );
    Fail { code: 1, msg: String::new() }
}

/// One field's lines, as the file has them: split on newlines, no other trimming (so a line's
/// trailing spaces survive, which the byte-diff against the oracle guards).
pub fn read_field(path: &Path) -> Result<Vec<String>, Fail> {
    let bytes = std::fs::read(path).or_else(|e| fail(1, format!("{}: {e}", path.display())))?;
    let text = String::from_utf8_lossy(&bytes);
    let mut lines: Vec<String> =
        text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l).to_string()).collect();
    if lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    Ok(lines)
}
