use crate::core::{algorithm::Algorithm, individual::Value};

/// Callable type for the optional `func` update hook.
///
/// Mirrors `Column.__init__(func=...)` — a callable that receives the current
/// algorithm and returns a new column value.
pub type ColumnUpdateFn = Box<dyn Fn(&dyn Algorithm) -> Value>;

/// Concrete base-class column.
///
/// Mirrors `pymoo.util.display.column.Column`.
pub struct ColumnBase {
    pub name: String,
    pub width: usize,
    pub truncate: bool,
    pub value: Option<Value>,
    /// Optional update hook — mirrors `Column.__init__(func=None)`.
    pub func: Option<ColumnUpdateFn>,
}

impl ColumnBase {
    /// Mirrors `Column.__init__(name, width=13, func=None, truncate=True)`.
    pub fn new(
        name: impl Into<String>,
        width: Option<usize>,
        func: Option<ColumnUpdateFn>,
        truncate: Option<bool>,
    ) -> Self {
        Self {
            name: name.into(),
            width: width.unwrap_or(13),
            truncate: truncate.unwrap_or(true),
            value: None,
            func,
        }
    }
}

/// Interface for a single displayable column in the output table.
///
/// Mirrors `pymoo.util.display.column.Column` used as the base class for all
/// concrete column types.  `ColumnBase` provides the standard implementation;
/// custom columns override `update` (and optionally `text`).
pub trait Column {
    fn name(&self) -> &str;
    fn width(&self) -> usize;

    /// Update the column's stored value from the current algorithm state.
    /// Mirrors `Column.update(algorithm)`.
    fn update(&mut self, algorithm: &dyn Algorithm);

    /// Return the column value formatted and right-justified to `width` chars.
    /// Mirrors `Column.text()`.
    fn text(&self) -> String;

    /// Return the column name centred to `width` characters.
    /// Mirrors `Column.header()`.
    fn header(&self) -> String {
        center_str(self.name(), self.width())
    }

    fn set(&mut self, value: Value) -> ();
}

impl Column for ColumnBase {
    fn name(&self) -> &str {
        &self.name
    }

    fn width(&self) -> usize {
        self.width
    }

    /// Mirrors `Column.update(algorithm)`:
    /// `if self.func: self.value = self.func(algorithm)`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        if let Some(ref f) = self.func {
            self.value = Some(f(algorithm));
        }
    }

    fn text(&self) -> String {
        format_text(self.value.clone(), self.width, self.truncate)
    }

    /// Directly set the stored value.
    /// Mirrors `Column.set(value)`
    fn set(&mut self, value: Value) {
        self.value = Some(value);
    }
}

/// Format a floating-point number for a column of the given width.
///
/// - large (`>= 10`) or very small (`< 1e-5`) → exponential notation
/// - otherwise → fixed-point notation
///
/// Mirrors `column.number_to_text(number, width)`.
pub fn number_to_text(number: f64, width: usize) -> String {
    if number >= 10.0 || number * 1e5 < 1.0 {
        // mirrors: f"%.{width - 7}E" % number
        let precision = width.saturating_sub(7);
        format!("{:.prec$E}", number, prec = precision)
    } else {
        // mirrors: f"%.{width - 3}f" % number
        let precision = width.saturating_sub(3);
        format!("{:.prec$}", number, prec = precision)
    }
}

/// Format an arbitrary column value as right-justified text of exactly `width` chars.
///
/// Mirrors `column.format_text(value, width, truncate)`:
/// - `None`  → `"-"`
/// - `Float` → `number_to_text()`
/// - other   → `str(value)`
/// Then optionally truncate and right-justify.
pub fn format_text(value: Option<Value>, width: usize, truncate: bool) -> String {
    let mut text = match value {
        Some(Value::Float(f)) => number_to_text(f, width),
        Some(Value::Int(i)) => i.to_string(),
        Some(Value::Str(s)) => s,
        None => "-".to_string(),
    };

    // mirrors: if truncate and len(text) > width: text = text[:width]
    if truncate && text.len() > width {
        text.truncate(width);
    }

    // mirrors: text.rjust(width)
    format!("{:>width$}", text, width = width)
}

/// Centre `s` within a field of `width` characters (space-padded).
///
/// Mirrors Python's `str.center(width)`, used by `Column.header()` and
/// `Output.header()`.
pub fn center_str(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        return s.chars().take(width).collect();
    }
    let total = width - len;
    let left = total / 2;
    let right = total - left;
    format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
}
