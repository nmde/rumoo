use crate::core::{
    algorithm::Algorithm,
    individual::{IndividualField, Value},
    termination::{Termination, TerminationBase},
};

/// Mirrors `pymoo.termination.cv.ConstraintViolationTermination`.
pub struct ConstraintViolationTermination {
    pub inner: DeltaToleranceTermination,
    pub terminate_when_feasible: bool,
}

impl ConstraintViolationTermination {
    /// Mirrors `ConstraintViolationTermination.__init__(tol=1e-6, terminate_when_feasible=True)`.
    pub fn new(tol: Option<f64>, terminate_when_feasible: Option<bool>) -> Self {
        Self {
            inner: DeltaToleranceTermination::new(tol.unwrap_or(1e-6)),
            terminate_when_feasible: terminate_when_feasible.unwrap_or(true),
        }
    }

    /// Mirrors `ConstraintViolationTermination._delta(prev, current)`:
    /// `return max(0.0, prev - current)`.
    pub fn _delta(&self, prev: f64, current: f64) -> f64 {
        (prev - current).max(0.0)
    }

    /// Mirrors `ConstraintViolationTermination._data(algorithm)`:
    /// `return algorithm.opt.get("CV").min()`.
    pub fn _data(&self, algorithm: &dyn Algorithm) -> f64 {
        algorithm
            .base()
            .opt
            .as_ref()
            .and_then(|opt| match opt.get(&IndividualField::CV) {
                Value::FloatMatrix(cv) => cv.iter().cloned().reduce(f64::min),
                Value::FloatArray(cv) => cv.iter().cloned().reduce(f64::min),
                _ => None,
            })
            .unwrap_or(0.0)
    }
}

impl Termination for ConstraintViolationTermination {
    fn base(&self) -> &TerminationBase {
        self.inner.base()
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        self.inner.base_mut()
    }

    /// Mirrors `ConstraintViolationTermination._update`.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let has_constraints = algorithm
            .base()
            .problem
            .as_ref()
            .map_or(false, |p| p.has_constraints());

        if has_constraints {
            let feasible_found = algorithm
                .base()
                .opt
                .as_ref()
                .map_or(false, |opt| match opt.get(&IndividualField::Feas) {
                    Value::BoolArray(arr) => arr.iter().any(|&v| v),
                    _ => false,
                });

            if feasible_found {
                if self.terminate_when_feasible { 1.0 } else { 0.0 }
            } else {
                // Mirrors: return super()._update(algorithm)
                self.inner._update(algorithm)
            }
        } else {
            0.0
        }
    }
}

/// Mirrors `pymoo.termination.cv.UntilFeasibleTermination`.
pub struct UntilFeasibleTermination {
    pub base: TerminationBase,
    pub initial_cv: Option<f64>,
}

impl UntilFeasibleTermination {
    /// Mirrors `UntilFeasibleTermination.__init__()`.
    pub fn new() -> Self {
        Self {
            base: TerminationBase::new(),
            initial_cv: None,
        }
    }
}

impl Termination for UntilFeasibleTermination {
    fn base(&self) -> &TerminationBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.base
    }

    /// Mirrors `UntilFeasibleTermination._update`:
    /// tracks `initial_cv` on first call and returns `1 - cv / initial_cv`.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let cv = algorithm
            .base()
            .opt
            .as_ref()
            .and_then(|opt| match opt.get(&IndividualField::CV) {
                Value::FloatMatrix(cv) => cv.iter().cloned().reduce(f64::min),
                Value::FloatArray(cv) => cv.iter().cloned().reduce(f64::min),
                _ => None,
            })
            .unwrap_or(0.0);

        if self.initial_cv.is_none() {
            self.initial_cv = Some(if cv <= 0.0 { 1e-32 } else { cv });
        }

        1.0 - cv / self.initial_cv.unwrap()
    }
}
