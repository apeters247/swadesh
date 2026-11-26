//! The slice of `tabulate` that sacreBLEU's multi-system and paired reports use:
//! `fancy_grid` for `-f text` and `latex_booktabs` for `-f latex`.
//!
//! Geometry measured from sacreBLEU 2.6.0, not guessed (see docs/lab-notebook.md
//! for the measurements). The one rule that is not obvious:
//!
//!   column content width = max(header.len() + 2, longest cell)
//!
//! and each cell is then drawn as one space, the aligned content, one space.

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

/// Column content widths: `max(header + 2, longest cell)`. Widths count chars, not
/// bytes, because tabulate does (`the number of colorful characters`).
fn widths(header: &[String], rows: &[Vec<String>]) -> Vec<usize> {
    let mut w: Vec<usize> = header.iter().map(|h| h.chars().count() + 2).collect();
    for row in rows {
        for (j, cell) in row.iter().enumerate() {
            let len = cell.chars().count();
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
    let row = |cells: &[String]| {
        let mut s = String::new();
        s.push('\u{2502}');
        for (j, cell) in cells.iter().enumerate() {
            if j > 0 {
                s.push('\u{2502}');
            }
            s.push(' ');
            s.push_str(&pad(cell, w[j], aligns[j]));
            s.push(' ');
        }
        s.push('\u{2502}');
        s
    };

    let mut out = String::new();
    out.push_str(&border('\u{2552}', '\u{2564}', '\u{2555}', '\u{2550}'));
    out.push('\n');
    out.push_str(&row(header));
    out.push('\n');
    out.push_str(&border('\u{255e}', '\u{256a}', '\u{2561}', '\u{2550}'));
    out.push('\n');
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            out.push_str(&border('\u{251c}', '\u{253c}', '\u{2524}', '\u{2500}'));
            out.push('\n');
        }
        out.push_str(&row(r));
        out.push('\n');
    }
    out.push_str(&border('\u{2558}', '\u{2567}', '\u{255b}', '\u{2550}'));
    out.push('\n');
    out
}

/// A `latex_booktabs` table: `tabular`, `toprule`, header, `midrule`, body, `bottomrule`.
pub fn latex_table(header: &[String], rows: &[Vec<String>], aligns: &[Align]) -> String {
    let w = widths(header, rows);
    let spec: String = aligns.iter().map(|a| a.latex_char()).collect();
    let row = |cells: &[String]| {
        let mut s = String::new();
        for (j, cell) in cells.iter().enumerate() {
            if j > 0 {
                s.push('&');
            }
            s.push(' ');
            s.push_str(&pad(cell, w[j], aligns[j]));
            s.push(' ');
        }
        s.push_str("\\\\");
        s
    };
    let mut out = String::new();
    out.push_str(&format!("\\begin{{tabular}}{{{spec}}}\n"));
    out.push_str("\\toprule\n");
    out.push_str(&row(header));
    out.push('\n');
    out.push_str("\\midrule\n");
    for r in rows {
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
}
