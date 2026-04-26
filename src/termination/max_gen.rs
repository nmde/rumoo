use crate::core::{
    algorithm::Algorithm,
    termination::{Termination, TerminationBase},
};

/// Mirrors `pymoo.termination.max_gen.MaximumGenerationTermination`.
pub struct MaximumGenerationTermination {
    pub base: TerminationBase,
    /// Maximum number of generations. Defaults to `f64::INFINITY`.
    pub n_max_gen: f64,
}

impl MaximumGenerationTermination {
    /// Mirrors `MaximumGenerationTermination.__init__(n_max_gen=float("inf"))`.
    pub fn new(n_max_gen: Option<f64>) -> Self {
        Self {
            base: TerminationBase::new(),
            n_max_gen: n_max_gen.unwrap_or(f64::INFINITY),
        }
    }
}

impl Termination for MaximumGenerationTermination {
    fn base(&self) -> &TerminationBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.base
    }

    /// Mirrors `MaximumGenerationTermination._update`:
    /// `return algorithm.n_gen / self.n_max_gen`.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        if self.n_max_gen.is_infinite() {
            return 0.0;
        }
        algorithm.n_gen() as f64 / self.n_max_gen
    }
}
