use crate::core::{
    algorithm::Algorithm,
    individual::{IndividualField, Value},
    termination::{Termination, TerminationBase},
};

/// Mirrors `pymoo.termination.fmin.MinimumFunctionValueTermination`.
pub struct MinimumFunctionValueTermination {
    pub base: TerminationBase,
    pub fmin: f64,
}

impl MinimumFunctionValueTermination {
    /// Mirrors `MinimumFunctionValueTermination.__init__(fmin)`.
    pub fn new(fmin: f64) -> Self {
        Self {
            base: TerminationBase::new(),
            fmin,
        }
    }
}

impl Termination for MinimumFunctionValueTermination {
    fn base(&self) -> &TerminationBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.base
    }

    /// Mirrors `MinimumFunctionValueTermination._update`:
    /// `return self.fmin / opt.get("F").min()` if any feasible, else `0.0`.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let opt = match algorithm.base().opt.as_ref() {
            Some(o) => o,
            None => return 0.0,
        };

        // Mirrors: if not any(opt.get("feas")): return 0.0
        let any_feas = match opt.get(&IndividualField::Feas) {
            Value::BoolArray(arr) => arr.iter().any(|&v| v),
            _ => false,
        };
        if !any_feas {
            return 0.0;
        }

        // Mirrors: return self.fmin / opt.get("F").min()
        match opt.get(&IndividualField::F) {
            Value::FloatMatrix(f) => {
                let f_min = f.iter().cloned().fold(f64::INFINITY, f64::min);
                self.fmin / f_min
            }
            Value::FloatArray(f) => {
                let f_min = f.iter().cloned().fold(f64::INFINITY, f64::min);
                self.fmin / f_min
            }
            _ => 0.0,
        }
    }
}
