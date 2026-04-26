use ndarray::Array2;

use crate::{
    core::{algorithm::Algorithm, individual::Value},
    indicators::igd::IGD,
    termination::ftol::MultiObjectiveSpaceTermination,
    util::display::{
        column::{Column, format_text},
        output::{Output, OutputBase},
        single::{AverageConstraintViolation, MinimumConstraintViolation},
    },
};

// -------------------------------------------------------------------------------------------------
// NumberOfNondominatedSolutions
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.display.multi.NumberOfNondominatedSolutions(Column)`.
pub struct NumberOfNondominatedSolutions {
    pub name: String,
    pub width: usize,
    pub value: Option<usize>,
}

impl NumberOfNondominatedSolutions {
    /// Mirrors `NumberOfNondominatedSolutions.__init__(width=6)`.
    pub fn new() -> Self {
        Self {
            name: "n_nds".to_string(),
            width: 6,
            value: None,
        }
    }
}

impl Default for NumberOfNondominatedSolutions {
    fn default() -> Self {
        Self::new()
    }
}

impl Column for NumberOfNondominatedSolutions {
    fn name(&self) -> &str {
        &self.name
    }

    fn width(&self) -> usize {
        self.width
    }

    /// Mirrors `NumberOfNondominatedSolutions.update(algorithm)`:
    /// `self.value = len(algorithm.opt)`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        self.value = Some(algorithm.opt().len());
    }

    fn text(&self) -> String {
        format_text(self.value.map(|v| Value::Int(v as i64)), self.width, true)
    }

    fn set(&mut self, value: Value) {
        match value {
            Value::Int(i) => self.value = Some(i as usize),
            Value::Float(f) => self.value = Some(f as usize),
            _ => {}
        }
    }
}

// -------------------------------------------------------------------------------------------------
// MultiObjectiveOutput
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.display.multi.MultiObjectiveOutput(Output)`.
pub struct MultiObjectiveOutput {
    pub base: OutputBase,
    pub cv_min: MinimumConstraintViolation,
    pub cv_avg: AverageConstraintViolation,
    pub n_nds: NumberOfNondominatedSolutions,
    pub igd: Option<f64>,
    pub gd: Option<f64>,
    pub hv: Option<f64>,
    pub eps: Option<f64>,
    pub indicator: Option<String>,
    pub pf: Option<Array2<f64>>,
    pub indicator_no_pf: Option<MultiObjectiveSpaceTermination>,
    hv_in_columns: bool,
}

impl MultiObjectiveOutput {
    /// Mirrors `MultiObjectiveOutput.__init__()`.
    pub fn new() -> Self {
        Self {
            base: OutputBase::new(),
            cv_min: MinimumConstraintViolation::new(),
            cv_avg: AverageConstraintViolation::new(),
            n_nds: NumberOfNondominatedSolutions::new(),
            igd: None,
            gd: None,
            hv: None,
            eps: None,
            indicator: None,
            pf: None,
            indicator_no_pf: None,
            hv_in_columns: false,
        }
    }

    /// Mirrors `MultiObjectiveOutput.initialize(algorithm)`.
    pub fn initialize(&mut self, algorithm: &dyn Algorithm) {
        let problem = algorithm.problem();

        self.base
            .columns
            .push(Box::new(NumberOfNondominatedSolutions::new()));

        if problem.has_constraints() {
            self.base
                .columns
                .push(Box::new(MinimumConstraintViolation::new()));
            self.base
                .columns
                .push(Box::new(AverageConstraintViolation::new()));
        }

        self.pf = problem.pareto_front();
        if self.pf.is_some() {
            self.base.columns.push(Box::new(IgdColumn::new()));
            self.base.columns.push(Box::new(GdColumn::new()));

            if problem.n_obj() == 2 {
                self.base.columns.push(Box::new(HvColumn::new()));
                self.hv_in_columns = true;
            }
        } else {
            self.indicator_no_pf = Some(MultiObjectiveSpaceTermination::new(None, None, None));
            self.base.columns.push(Box::new(EpsColumn::new()));
            self.base.columns.push(Box::new(IndicatorColumn::new()));
        }
    }

    /// Mirrors `MultiObjectiveOutput.update(algorithm)`.
    pub fn update(&mut self, algorithm: &dyn Algorithm) {
        self.base.update(algorithm);

        // Reset all indicator values
        self.igd = None;
        self.gd = None;
        self.hv = None;
        self.eps = None;
        self.indicator = None;

        // Mirrors: F, feas = algorithm.opt.get("F", "feas"); F = F[feas]
        let (f_mat, feas) = algorithm.opt().get_f_feas();
        let f_feas: Array2<f64> = f_mat
            .outer_iter()
            .zip(feas.iter())
            .filter_map(|(row, &is_feas)| if is_feas { Some(row.to_owned()) } else { None })
            .collect();

        if f_feas.nrows() > 0 {
            let problem = algorithm.problem();

            // Mirrors: if hasattr(problem, "time"): self.pf = pareto_front_if_possible(problem)
            if problem.has_time() {
                self.pf = problem.pareto_front();
            }

            if let Some(ref pf) = self.pf {
                if !f_feas.is_empty() {
                    self.igd = IGD::new(pf.clone())
                        .ok()
                        .and_then(|igd| igd.do_calc(&f_feas).ok());
                    self.gd = Some(GD::new(pf, true).do_indicator(&f_feas));

                    if self.hv_in_columns {
                        self.hv = Some(Hypervolume::new(pf, true).do_indicator(&f_feas));
                    }
                }

                // Sync values to the columns in base for display
                self.set_column_value("igd", self.igd.map(Value::Float));
                self.set_column_value("gd", self.gd.map(Value::Float));
                if self.hv_in_columns {
                    self.set_column_value("hv", self.hv.map(Value::Float));
                }
            }

            if let Some(ref mut ind) = self.indicator_no_pf {
                ind.update(algorithm);

                if let Some(delta_ideal) = ind.delta_ideal() {
                    let (max_from, eps) = if delta_ideal > ind.tol() {
                        ("ideal", delta_ideal)
                    } else if ind.delta_nadir().unwrap_or(0.0) > ind.tol() {
                        ("nadir", ind.delta_nadir().unwrap_or(0.0))
                    } else {
                        ("f", ind.delta_f().unwrap_or(0.0))
                    };

                    self.eps = Some(eps);
                    self.indicator = Some(max_from.to_string());

                    self.set_column_value("eps", Some(Value::Float(eps)));
                    self.set_column_value("indicator", Some(Value::Str(max_from.to_string())));
                }
            }
        }
    }

    /// Find a column by name in `base.columns` and set its value.
    fn set_column_value(&mut self, name: &str, value: Option<Value>) {
        for col in &mut self.base.columns {
            if col.name() == name {
                if let Some(v) = value {
                    col.set(v);
                }
                return;
            }
        }
    }
}

impl Default for MultiObjectiveOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl Output for MultiObjectiveOutput {
    fn base(&self) -> OutputBase {
        OutputBase::new()
    }
}

// -------------------------------------------------------------------------------------------------
// Private display-only column wrappers for igd / gd / hv / eps / indicator
// -------------------------------------------------------------------------------------------------

struct IgdColumn {
    width: usize,
    value: Option<f64>,
}
impl IgdColumn {
    fn new() -> Self {
        Self {
            width: 8,
            value: None,
        }
    }
}
impl Column for IgdColumn {
    fn name(&self) -> &str {
        "igd"
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

struct GdColumn {
    width: usize,
    value: Option<f64>,
}
impl GdColumn {
    fn new() -> Self {
        Self {
            width: 8,
            value: None,
        }
    }
}
impl Column for GdColumn {
    fn name(&self) -> &str {
        "gd"
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

struct HvColumn {
    width: usize,
    value: Option<f64>,
}
impl HvColumn {
    fn new() -> Self {
        Self {
            width: 8,
            value: None,
        }
    }
}
impl Column for HvColumn {
    fn name(&self) -> &str {
        "hv"
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

struct EpsColumn {
    width: usize,
    value: Option<f64>,
}
impl EpsColumn {
    fn new() -> Self {
        Self {
            width: 8,
            value: None,
        }
    }
}
impl Column for EpsColumn {
    fn name(&self) -> &str {
        "eps"
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

struct IndicatorColumn {
    width: usize,
    value: Option<String>,
}
impl IndicatorColumn {
    fn new() -> Self {
        Self {
            width: 8,
            value: None,
        }
    }
}
impl Column for IndicatorColumn {
    fn name(&self) -> &str {
        "indicator"
    }
    fn width(&self) -> usize {
        self.width
    }
    fn update(&mut self, _algorithm: &dyn Algorithm) {}
    fn text(&self) -> String {
        format_text(self.value.clone().map(Value::Str), self.width, true)
    }
    fn set(&mut self, value: Value) {
        if let Value::Str(s) = value {
            self.value = Some(s);
        } else {
            self.value = None;
        }
    }
}
