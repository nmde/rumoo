use crate::core::{
    algorithm::Algorithm,
    termination::{Termination, TerminationBase},
};

/// Mirrors `pymoo.termination.max_eval.MaximumFunctionCallTermination`.
pub struct MaximumFunctionCallTermination {
    pub base: TerminationBase,
    /// Maximum number of function evaluations. Defaults to `f64::INFINITY`.
    pub n_max_evals: f64,
}

impl MaximumFunctionCallTermination {
    /// Mirrors `MaximumFunctionCallTermination.__init__(n_max_evals=float("inf"))`.
    pub fn new(n_max_evals: Option<f64>) -> Self {
        Self {
            base: TerminationBase::new(),
            n_max_evals: n_max_evals.unwrap_or(f64::INFINITY),
        }
    }
}

impl Termination for MaximumFunctionCallTermination {
    fn base(&self) -> &TerminationBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.base
    }

    /// Mirrors `MaximumFunctionCallTermination._update`:
    /// `return algorithm.evaluator.n_eval / self.n_max_evals`.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        if self.n_max_evals.is_infinite() {
            return 0.0;
        }
        algorithm.base().evaluator.n_eval() as f64 / self.n_max_evals
    }
}
