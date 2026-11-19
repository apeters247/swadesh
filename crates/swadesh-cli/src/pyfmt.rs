//! CPython float formatting, as used by sacreBLEU's output.
//!
//! * `fixed(x, p)` is `f"{x:.{p}f}"`: exact decimal expansion of the binary
//!   value, rounded half-to-even (Rust's `{:.p}` does the same).
//! * `repr(x)` is `repr(float)` / `json.dumps(float)`: the shortest digits that
//!   round-trip, in fixed notation for decimal exponents in [-4, 16) and in
//!   `d.ddde+XX` notation otherwise, always with a `.0` for integral values.

/// `f"{x:.{prec}f}"`.
pub fn fixed(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    format!("{:.*}", prec, x)
}

/// `json.dumps(x)` for a float (`float.__repr__`, with NaN/Infinity spelled as json does).
pub fn json_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    repr(x)
}

/// `repr(x)` for a finite float.
pub fn repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let (sign, digits, exp) = shortest_digits(x);
    let mut out = String::from(sign);
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = (exp + 1) as usize;
            if digits.len() <= int_len {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', int_len - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    }
    out
}

/// Shortest round-tripping decimal digits of `x` (no trailing zeros), with the
/// decimal exponent of the first digit and the sign.
///
/// Rust's `{:e}` also yields the shortest digits, but on an exact tie between
/// two of them it rounds up, where CPython's `repr` rounds half-to-even. The
/// correctly rounded form at the same length (`{:.*e}` rounds half-to-even) is
/// what CPython prints whenever it still round-trips, so prefer it.
fn shortest_digits(x: f64) -> (&'static str, String, i32) {
    let sign = if x < 0.0 { "-" } else { "" };
    let (short_digits, short_exp) = split_sci(&format!("{:e}", x.abs()));
    let n = short_digits.len();
    let alt_sci = format!("{:.*e}", n - 1, x.abs());
    let (alt_digits, alt_exp) = split_sci(&alt_sci);
    if alt_digits != short_digits && alt_sci.parse::<f64>().ok() == Some(x.abs()) {
        return (sign, alt_digits, alt_exp);
    }
    (sign, short_digits, short_exp)
}

/// Split LowerExp output "d.ddde[-]N" into (digits without trailing zeros, exponent).
fn split_sci(sci: &str) -> (String, i32) {
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp always has an 'e'");
    let exp: i32 = exp.parse().expect("LowerExp exponent is an integer");
    let mut digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    (digits, exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_ties_round_half_even() {
        assert_eq!(fixed(0.125, 2), "0.12");
        assert_eq!(fixed(0.375, 2), "0.38");
        assert_eq!(fixed(2.5, 0), "2");
        assert_eq!(fixed(3.5, 0), "4");
        assert_eq!(fixed(12.25, 1), "12.2");
        assert_eq!(fixed(12.75, 1), "12.8");
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn repr_matches_python() {
        assert_eq!(repr(25.4), "25.4");
        assert_eq!(repr(100.0), "100.0");
        assert_eq!(repr(0.0), "0.0");
        assert_eq!(repr(0.0001), "0.0001");
        assert_eq!(repr(0.00001), "1e-05");
        assert_eq!(repr(1e16), "1e+16");
        assert_eq!(repr(9999999999999998.0), "9999999999999998.0");
        assert_eq!(repr(1.5e-7), "1.5e-07");
        assert_eq!(repr(-0.5), "-0.5");
        // Exact tie between 17-digit candidates: CPython picks the even digit.
        assert_eq!(repr(1603999459147296.25), "1603999459147296.2");
    }
}
