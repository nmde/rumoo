use crate::core::{problem::Problem};

/// Displays the current generation number.
///
/// Mirrors `pymoo.util.display.output.NumberOfGenerations(Column)`.
pub struct NumberOfGenerations {
    pub name: String,
    pub width: usize,
    pub value: Option<usize>,
}

impl NumberOfGenerations {
    pub fn new(width: usize) -> Self {
        Self {
            name: "n_gen".to_string(),
            width,
            value: None,
        }
    }
}

impl Column for NumberOfGenerations {
    fn name(&self) -> &str {
        &self.name
    }

    fn width(&self) -> usize {
        self.width
    }

    /// Mirrors `NumberOfGenerations.update(algorithm)`:
    /// `self.value = algorithm.n_gen`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        self.value = Some(algorithm.n_gen());
    }

    fn text(&self) -> String {
        format_text(self.value.map(|v| v.to_string()), self.width, true)
    }
}

/// Displays the total number of objective function evaluations.
///
/// Mirrors `pymoo.util.display.output.NumberOfEvaluations(Column)`.
pub struct NumberOfEvaluations {
    pub name: String,
    pub width: usize,
    pub value: Option<usize>,
}

impl NumberOfEvaluations {
    pub fn new(width: usize) -> Self {
        Self {
            name: "n_eval".to_string(),
            width,
            value: None,
        }
    }
}

impl Column for NumberOfEvaluations {
    fn name(&self) -> &str {
        &self.name
    }

    fn width(&self) -> usize {
        self.width
    }

    /// Mirrors `NumberOfEvaluations.update(algorithm)`:
    /// `self.value = algorithm.evaluator.n_eval`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        self.value = Some(algorithm.evaluator().n_eval());
    }

    fn text(&self) -> String {
        format_text(self.value.map(|v| v.to_string()), self.width, true)
    }
}

/// Base output display: renders a table of columns after each generation.
///
/// Mirrors `pymoo.util.display.output.Output(Callback)`.
pub struct Output {
    pub columns: Vec<Box<dyn Column>>,
}

impl Output {
    /// Mirrors `Output.__init__()` — creates with the two default columns.
    pub fn new() -> Self {
        Self {
            columns: vec![
                Box::new(NumberOfGenerations::new(6)),
                Box::new(NumberOfEvaluations::new(8)),
            ],
        }
    }

    /// Update all columns from the current algorithm state.
    ///
    /// Mirrors `Output.update(algorithm)`:
    /// `[col.update(algorithm) for col in self.columns]`.
    pub fn update(&mut self, algorithm: &dyn Algorithm) {
        for col in self.columns.iter_mut() {
            col.update(algorithm);
        }
    }

    /// Render the column-name header row, optionally surrounded by `=` borders.
    ///
    /// Mirrors `Output.header(border=False)`.
    pub fn header(&self, border: Option<bool>) -> String {
        let border = border.unwrap_or(false);
        let parts: Vec<String> = self
            .columns
            .iter()
            .map(|col| center_str(col.name(), col.width()))
            .collect();
        let header = parts.join(" | ");

        if border {
            let line = "=".repeat(header.len());
            format!("{}\n{}\n{}", line, header, line)
        } else {
            header
        }
    }

    /// Render the current column values as a single line.
    ///
    /// Mirrors `Output.text()`:
    /// `regex.format(*[col.text() for col in self.columns])`.
    pub fn text(&self) -> String {
        let parts: Vec<String> = self.columns.iter().map(|col| col.text()).collect();
        parts.join(" | ")
    }
}

impl Callback for Output {
    fn update(&mut self, algorithm: &dyn Algorithm) {
        for col in self.columns.iter_mut() {
            col.update(algorithm);
        }
    }
}
