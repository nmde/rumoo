use crate::{
    core::{
        algorithm::Algorithm,
        termination::{Termination, TerminationBase},
    },
    util::sliding_window::SlidingWindow,
};

/// Mirrors `pymoo.termination.robust.RobustTermination`.
///
/// Wraps an inner termination criterion and smooths its progress signal over a
/// sliding window of `period` generations, returning the minimum observed value.
pub struct RobustTermination {
    pub base: TerminationBase,
    pub termination: Box<dyn Termination>,
    pub history: SlidingWindow<f64>,
}

impl RobustTermination {
    /// Mirrors `RobustTermination.__init__(termination, period=30)`.
    pub fn new(termination: Box<dyn Termination>, period: usize) -> Self {
        Self {
            base: TerminationBase::new(),
            termination,
            history: SlidingWindow::new(period),
        }
    }
}

impl Termination for RobustTermination {
    fn base(&self) -> &TerminationBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.base
    }

    /// Mirrors `RobustTermination._update`:
    /// `perc = self.termination.update(algorithm); self.history.append(perc); return min(self.history)`.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let perc = self.termination.update(algorithm);
        self.history.push(perc);
        self.history.iter().cloned().fold(f64::INFINITY, f64::min)
    }
}
