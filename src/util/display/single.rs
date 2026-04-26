use crate::{
    core::{algorithm::Algorithm, individual::Value},
    util::display::{
        column::{Column, format_text},
        output::{Output, OutputBase},
    },
};

// -------------------------------------------------------------------------------------------------
// MinimumConstraintViolation
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.display.single.MinimumConstraintViolation(Column)`.
pub struct MinimumConstraintViolation {
    pub name: String,
    pub width: usize,
    pub value: Option<f64>,
}

impl MinimumConstraintViolation {
    /// Mirrors `MinimumConstraintViolation.__init__()`.
    pub fn new() -> Self {
        Self {
            name: "cv_min".to_string(),
            width: 13,
            value: None,
        }
    }
}

impl Default for MinimumConstraintViolation {
    fn default() -> Self {
        Self::new()
    }
}

impl Column for MinimumConstraintViolation {
    fn name(&self) -> &str {
        &self.name
    }

    fn width(&self) -> usize {
        self.width
    }

    /// Mirrors `MinimumConstraintViolation.update(algorithm)`:
    /// `self.value = algorithm.opt.get("cv").min()`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        let cv = algorithm.opt().get_cv();
        self.value = cv.iter().copied().reduce(f64::min);
    }

    fn text(&self) -> String {
        format_text(self.value.map(Value::Float), self.width, true)
    }

    fn set(&mut self, value: Value) {
        if let Value::Float(f) = value {
            self.value = Some(f);
        } else {
            self.value = None;
        }
    }
}

// -------------------------------------------------------------------------------------------------
// AverageConstraintViolation
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.display.single.AverageConstraintViolation(Column)`.
pub struct AverageConstraintViolation {
    pub name: String,
    pub width: usize,
    pub value: Option<f64>,
}

impl AverageConstraintViolation {
    /// Mirrors `AverageConstraintViolation.__init__()`.
    pub fn new() -> Self {
        Self {
            name: "cv_avg".to_string(),
            width: 13,
            value: None,
        }
    }
}

impl Default for AverageConstraintViolation {
    fn default() -> Self {
        Self::new()
    }
}

impl Column for AverageConstraintViolation {
    fn name(&self) -> &str {
        &self.name
    }

    fn width(&self) -> usize {
        self.width
    }

    /// Mirrors `AverageConstraintViolation.update(algorithm)`:
    /// `self.value = algorithm.pop.get("cv").mean()`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        let cv = algorithm.pop().get_cv();
        if cv.is_empty() {
            self.value = None;
        } else {
            self.value = Some(cv.iter().sum::<f64>() / cv.len() as f64);
        }
    }

    fn text(&self) -> String {
        format_text(self.value.map(Value::Float), self.width, true)
    }

    fn set(&mut self, value: Value) {
        if let Value::Float(f) = value {
            self.value = Some(f);
        } else {
            self.value = None;
        }
    }
}

// -------------------------------------------------------------------------------------------------
// SingleObjectiveOutput
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.display.single.SingleObjectiveOutput(Output)`.
pub struct SingleObjectiveOutput {
    pub base: OutputBase,
    pub cv_min: MinimumConstraintViolation,
    pub cv_avg: AverageConstraintViolation,
    pub f_min: Option<f64>,
    pub f_avg: Option<f64>,
    pub f_gap: Option<f64>,
    pub best: Option<f64>,
    f_gap_in_columns: bool,
}

impl SingleObjectiveOutput {
    /// Mirrors `SingleObjectiveOutput.__init__()`.
    pub fn new() -> Self {
        Self {
            base: OutputBase::new(),
            cv_min: MinimumConstraintViolation::new(),
            cv_avg: AverageConstraintViolation::new(),
            f_min: None,
            f_avg: None,
            f_gap: None,
            best: None,
            f_gap_in_columns: false,
        }
    }

    /// Mirrors `SingleObjectiveOutput.initialize(algorithm)`.
    pub fn initialize(&mut self, algorithm: &dyn Algorithm) {
        let problem = algorithm.problem();

        if problem.has_constraints() {
            self.base
                .columns
                .push(Box::new(MinimumConstraintViolation::new()));
            self.base
                .columns
                .push(Box::new(AverageConstraintViolation::new()));
        }

        self.base.columns.push(Box::new(FAvgColumn::new()));
        self.base.columns.push(Box::new(FMinColumn::new()));

        let pf = problem.pareto_front();
        if let Some(ref pf_arr) = pf {
            self.best = pf_arr.iter().next().copied();
            self.base.columns.push(Box::new(FGapColumn::new()));
            self.f_gap_in_columns = true;
        }
    }

    /// Mirrors `SingleObjectiveOutput.update(algorithm)`.
    pub fn update(&mut self, algorithm: &dyn Algorithm) {
        self.base.update(algorithm);

        // Mirrors: f, cv, feas = algorithm.pop.get("f", "cv", "feas")
        let (f, feas) = algorithm.pop().get_f_feas();

        // Mirrors: if feas.sum() > 0: self.f_avg.set(f[feas].mean())
        //          else: self.f_avg.set(None)
        let feas_f: Vec<f64> = f
            .outer_iter()
            .zip(feas.iter())
            .filter_map(
                |(row, &is_feas)| {
                    if is_feas { row.get(0).copied() } else { None }
                },
            )
            .collect();

        if !feas_f.is_empty() {
            self.f_avg = Some(feas_f.iter().sum::<f64>() / feas_f.len() as f64);
        } else {
            self.f_avg = None;
        }

        // Mirrors: opt = algorithm.opt[0]; if opt.feas: self.f_min.set(opt.f) ...
        let opt = algorithm.opt();
        if let Some(ind) = opt.first() {
            if ind.feas() {
                self.f_min = ind.f().map(|v| v[0]);
                if let (Some(f_val), Some(best_val)) = (self.f_min, self.best) {
                    self.f_gap = Some(f_val - best_val);
                } else {
                    self.f_gap = None;
                }
            } else {
                self.f_min = None;
                self.f_gap = None;
            }
        }

        // Sync named values to base.columns for display
        self.set_column_value("f_avg", self.f_avg.map(Value::Float));
        self.set_column_value("f_min", self.f_min.map(Value::Float));
        if self.f_gap_in_columns {
            self.set_column_value("f_gap", self.f_gap.map(Value::Float));
        }
    }

    /// Find a column by name and set its value; passing `None` clears the display to "-".
    fn set_column_value(&mut self, name: &str, value: Option<Value>) {
        for col in &mut self.base.columns {
            if col.name() == name {
                match value {
                    Some(v) => col.set(v),
                    // Pass a non-Float sentinel so the column's set() clears to None
                    None => col.set(Value::Bool(false)),
                }
                return;
            }
        }
    }
}

impl Default for SingleObjectiveOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl Output for SingleObjectiveOutput {
    fn base(&self) -> OutputBase {
        OutputBase::new()
    }
}

// -------------------------------------------------------------------------------------------------
// Private display-only column wrappers for f_min / f_avg / f_gap
// -------------------------------------------------------------------------------------------------

struct FMinColumn {
    width: usize,
    value: Option<f64>,
}
impl FMinColumn {
    fn new() -> Self {
        Self {
            width: 13,
            value: None,
        }
    }
}
impl Column for FMinColumn {
    fn name(&self) -> &str {
        "f_min"
    }
    fn width(&self) -> usize {
        self.width
    }
    fn update(&mut self, _algorithm: &dyn Algorithm) {}
    fn text(&self) -> String {
        format_text(self.value.map(Value::Float), self.width, false)
    }
    fn set(&mut self, value: Value) {
        if let Value::Float(f) = value {
            self.value = Some(f);
        } else {
            self.value = None;
        }
    }
}

struct FAvgColumn {
    width: usize,
    value: Option<f64>,
}
impl FAvgColumn {
    fn new() -> Self {
        Self {
            width: 13,
            value: None,
        }
    }
}
impl Column for FAvgColumn {
    fn name(&self) -> &str {
        "f_avg"
    }
    fn width(&self) -> usize {
        self.width
    }
    fn update(&mut self, _algorithm: &dyn Algorithm) {}
    fn text(&self) -> String {
        format_text(self.value.map(Value::Float), self.width, false)
    }
    fn set(&mut self, value: Value) {
        if let Value::Float(f) = value {
            self.value = Some(f);
        } else {
            self.value = None;
        }
    }
}

struct FGapColumn {
    width: usize,
    value: Option<f64>,
}
impl FGapColumn {
    fn new() -> Self {
        Self {
            width: 13,
            value: None,
        }
    }
}
impl Column for FGapColumn {
    fn name(&self) -> &str {
        "f_gap"
    }
    fn width(&self) -> usize {
        self.width
    }
    fn update(&mut self, _algorithm: &dyn Algorithm) {}
    fn text(&self) -> String {
        format_text(self.value.map(Value::Float), self.width, false)
    }
    fn set(&mut self, value: Value) {
        if let Value::Float(f) = value {
            self.value = Some(f);
        } else {
            self.value = None;
        }
    }
}
