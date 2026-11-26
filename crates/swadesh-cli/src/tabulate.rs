//! The slice of `tabulate` that sacreBLEU's multi-system and paired reports use:
//! `fancy_grid` for `-f text` and `latex_booktabs` for `-f latex`.
//!
//! Geometry measured from sacreBLEU 2.6.0, not guessed (see docs/spec/cli-contract.md
//! and docs/lab-notebook.md for the measurements). Two rules are not obvious:
//!
//!   column content width = max(longest line of the heading + 2, longest line of any cell)
//!
//! and each line of a cell is drawn as one space, the aligned content, one space. A cell
//! may contain a newline: the text form expands it into extra grid lines (the paired
//! report's p-value), while the latex form replaces it with a space, as tabulate does.

/// Horizontal alignment of a column's cells.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
    Center,
}

impl Align {
    fn latex_char(self) -> char {
        match self {
            Align::Left => 'l',
            Align::Right => 'r',
            Align::Center => 'c',
        }
    }
}

/// The longest line of a cell (cells may be multi-line).
fn longest_line(cell: &str) -> usize {
    cell.split('\n').map(|line| line.chars().count()).max().unwrap_or(0)
}

/// Column content widths: `max(longest heading line + 2, longest cell line)`.
fn widths(header: &[String], rows: &[Vec<String>]) -> Vec<usize> {
    let mut w: Vec<usize> = header.iter().map(|h| longest_line(h) + 2).collect();
    for row in rows {
        for (j, cell) in row.iter().enumerate() {
            let len = longest_line(cell);
            if len > w[j] {
                w[j] = len;
            }
        }
    }
    w
}

/// `cell` centred / left / right in `width`, extra space on the right for a centre.
fn pad(cell: &str, width: usize, align: Align) -> String {
    let len = cell.chars().count();
    let fill = width.saturating_sub(len);
    match align {
        Align::Left => format!("{cell}{}", " ".repeat(fill)),
        Align::Right => format!("{}{cell}", " ".repeat(fill)),
        Align::Center => {
            let left = fill / 2;
            let right = fill - left;
            format!("{}{cell}{}", " ".repeat(left), " ".repeat(right))
        }
    }
}

/// A `fancy_grid` table, exactly as tabulate draws it (no wrapping).
pub fn fancy_grid(header: &[String], rows: &[Vec<String>], aligns: &[Align]) -> String {
    let w = widths(header, rows);
    let inner: Vec<usize> = w.iter().map(|x| x + 2).collect();

    let border = |left: char, mid: char, right: char, fill: char| {
        let mut s = String::new();
        s.push(left);
        for (j, width) in inner.iter().enumerate() {
            if j > 0 {
                s.push(mid);
            }
            s.extend(std::iter::repeat_n(fill, *width));
        }
        s.push(right);
        s
    };
    // One logical row may render as several grid lines (multi-line cells).
    let row = |cells: &[String]| {
        let split: Vec<Vec<&str>> = cells.iter().map(|c| c.split('\n').collect()).collect();
        let height = split.iter().map(|lines| lines.len()).max().unwrap_or(1);
        let mut s = String::new();
        for li in 0..height {
            s.push('\u{2502}');
            for (j, lines) in split.iter().enumerate() {
                if j > 0 {
                    s.push('\u{2502}');
                }
                s.push(' ');
                s.push_str(&pad(lines.get(li).copied().unwrap_or(""), w[j], aligns[j]));
                s.push(' ');
            }
            s.push('\u{2502}');
            s.push('\n');
        }
        s
    };

    let mut out = String::new();
    out.push_str(&border('\u{2552}', '\u{2564}', '\u{2555}', '\u{2550}'));
    out.push('\n');
    out.push_str(&row(header));
    out.push_str(&border('\u{255e}', '\u{256a}', '\u{2561}', '\u{2550}'));
    out.push('\n');
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            out.push_str(&border('\u{251c}', '\u{253c}', '\u{2524}', '\u{2500}'));
            out.push('\n');
        }
        out.push_str(&row(r));
    }
    out.push_str(&border('\u{2558}', '\u{2567}', '\u{255b}', '\u{2550}'));
    out.push('\n');
    out
}

/// The LaTeX escapes tabulate applies to a cell, measured one character at a time (a system name
/// containing all of them is in docs/lab-notebook.md).
fn latex_escape(cell: &str) -> String {
    let mut out = String::with_capacity(cell.len());
    for c in cell.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '_' => out.push_str("\\_"),
            '%' => out.push_str("\\%"),
            '&' => out.push_str("\\&"),
            '$' => out.push_str("\\$"),
            '#' => out.push_str("\\#"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\^{}"),
            '<' => out.push_str("\\ensuremath{<}"),
            '>' => out.push_str("\\ensuremath{>}"),
            _ => out.push(c),
        }
    }
    out
}

/// A `latex_booktabs` table: `tabular`, `toprule`, header, `midrule`, body, `bottomrule`.
///
/// A multi-line cell is joined with a space first and the result is LaTeX-escaped, both of which
/// are what tabulate's latex writer does; escaping comes before the widths are measured.
pub fn latex_table(header: &[String], rows: &[Vec<String>], aligns: &[Align]) -> String {
    let one_line = |cells: &[String]| -> Vec<String> {
        cells.iter().map(|c| c.replace('\n', " ")).collect()
    };
    let header = one_line(header);
    let rows: Vec<Vec<String>> = rows.iter().map(|r| one_line(r)).collect();
    let w = widths(&header, &rows);
    let spec: String = aligns.iter().map(|a| a.latex_char()).collect();
    // Padding comes from the raw length and the escaping is applied afterwards, so a cell with
    // special characters ends up wider than the column: that is what tabulate does, measured.
    let row = |cells: &[String]| {
        let mut s = String::new();
        for (j, cell) in cells.iter().enumerate() {
            if j > 0 {
                s.push('&');
            }
            s.push_str(&latex_escape(&format!(" {} ", pad(cell, w[j], aligns[j]))));
        }
        s.push_str("\\\\");
        s
    };
    let mut out = String::new();
    out.push_str(&format!("\\begin{{tabular}}{{{spec}}}\n"));
    out.push_str("\\toprule\n");
    out.push_str(&row(&header));
    out.push('\n');
    out.push_str("\\midrule\n");
    for r in &rows {
        out.push_str(&row(r));
        out.push('\n');
    }
    out.push_str("\\bottomrule\n");
    out.push_str("\\end{tabular}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn fancy_grid_two_systems_bleu() {
        let header = strings(&["System", "BLEU"]);
        let rows = vec![strings(&["System 1", "26.9"]), strings(&["System 2", "0.0"])];
        let got = fancy_grid(&header, &rows, &[Align::Right, Align::Center]);
        let want = "\
\u{2552}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2564}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2555}\n\
\u{2502}   System \u{2502}  BLEU  \u{2502}\n\
\u{255e}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2561}\n\
\u{2502} System 1 \u{2502}  26.9  \u{2502}\n\
\u{251c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}\n\
\u{2502} System 2 \u{2502}  0.0   \u{2502}\n\
\u{2558}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2567}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255b}\n";
        assert_eq!(got, want);
    }

    #[test]
    fn latex_two_systems_bleu() {
        let header = strings(&["System", "BLEU"]);
        let rows = vec![strings(&["System 1", "26.9"]), strings(&["System 2", "0.0"])];
        let got = latex_table(&header, &rows, &[Align::Right, Align::Center]);
        let want = "\\begin{tabular}{rc}\n\\toprule\n   System &  BLEU  \\\\\n\\midrule\n \
                    System 1 &  26.9  \\\\\n System 2 &  0.0   \\\\\n\\bottomrule\n\\end{tabular}\n";
        assert_eq!(got, want);
    }

    #[test]
    fn header_plus_two_keeps_a_two_space_margin() {
        // chrF2++ is 7 wide, so the metric column is 9 even though the scores are 4 wide.
        let header = strings(&["System", "chrF2++"]);
        let rows = vec![strings(&["System 1", "45.5"]), strings(&["System 2", "15.3"])];
        let got = fancy_grid(&header, &rows, &[Align::Right, Align::Center]);
        assert!(got.contains("\u{2502}  chrF2++  \u{2502}"), "{got}");
        assert!(got.contains("\u{2502}   45.5    \u{2502}"), "{got}");
    }

    #[test]
    fn a_multiline_cell_draws_extra_grid_lines_and_widens_the_column() {
        // The paired report: the p-value is a second line of the same cell, and because the
        // column must hold it, a short heading gets a wide column.
        let header = strings(&["System", "BLEU (mu)"]); // 9 wide -> min 11
        let rows = vec![
            strings(&["Baseline", "50.0 (51.0 +/- 1.9)"]), // 17/18
            strings(&["System 1", "58.1 (46.8 +/- 38.0)\n(p = 0.2941)"]),
        ];
        let got = fancy_grid(&header, &rows, &[Align::Right, Align::Center]);
        // The longest score line is 20 wide, so that is the column, and "(p = 0.2941)" (12) is
        // centred inside it -- on a line of its own, with no separator above it.
        assert!(got.contains("\u{2502}          \u{2502}     (p = 0.2941)     \u{2502}"), "{got}");
    }

    #[test]
    fn latex_escapes_the_measured_characters() {
        let header = strings(&["System", "BLEU (mu +- 95% CI)"]);
        let rows = vec![strings(&["a_b%c&d<$>#{}~^\\e", "1.0"])];
        let got = latex_table(&header, &rows, &[Align::Right, Align::Center]);
        assert!(
            got.contains("a\\_b\\%c\\&d\\ensuremath{<}\\$\\ensuremath{>}\\#\\{\\}\\textasciitilde{}\\^{}\\textbackslash{}e"),
            "{got}"
        );
        assert!(got.contains("95\\% CI"), "{got}");
    }

    #[test]
    fn latex_joins_a_multiline_cell_with_a_space() {
        let header = strings(&["System", "BLEU"]);
        let rows = vec![strings(&["System 1", "58.1\n(p = 0.2941)"])];
        let got = latex_table(&header, &rows, &[Align::Right, Align::Center]);
        assert!(got.contains(" & 58.1 (p = 0.2941) \\\\"), "{got}");
    }
}
