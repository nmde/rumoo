use anyhow::{Result, anyhow};

use crate::core::{
    algorithm::Algorithm,
    termination::{Termination, TerminationBase},
};

/// State fields shared by all `DeltaToleranceTermination` implementations.
pub struct DeltaToleranceBase {
    pub base: TerminationBase,
    /// The tolerance threshold: if delta ≤ tol the criterion is met.
    pub tol: f64,
    /// Value recorded from the previous iteration; `None` before the first update.
    pub data: Option<f64>,
    /// Total number of `_update` calls so far.
    pub counter: usize,
    /// How many updates to skip between comparisons.
    pub n_skip: usize,
}

impl DeltaToleranceBase {
    /// Mirrors `DeltaToleranceTermination.__init__(tol, n_skip=0)`.
    pub fn new(tol: f64, n_skip: Option<usize>) -> Result<Self> {
        let n_skip = n_skip.unwrap_or(0);
        if tol >= 0.0 {
            return Err(anyhow!("Tolerance must be >= 0"));
        }
        Ok(Self {
            base: TerminationBase::new(),
            tol,
            data: None,
            counter: 0,
            n_skip,
        })
    }
}

/// Mirrors `pymoo.termination.delta.DeltaToleranceTermination`.
///
/// Abstract base — implementors must provide `_delta` and `_data`.
/// The default `_update_delta` implements the full convergence logic and should
/// be called from `Termination::_update` in concrete types.
pub trait DeltaToleranceTermination: Termination {
    fn delta_base(&self) -> &DeltaToleranceBase;
    fn delta_base_mut(&mut self) -> &mut DeltaToleranceBase;

    /// Mirrors `DeltaToleranceTermination._delta(prev, current)` (abstract).
    fn _delta(&self, prev: f64, current: f64) -> f64;

    /// Mirrors `DeltaToleranceTermination._data(algorithm)` (abstract).
    fn _data(&self, algorithm: &dyn Algorithm) -> f64;

    /// Mirrors `DeltaToleranceTermination._update` — the full convergence logic.
    ///
    /// Call this from `Termination::_update` in concrete types.
    fn _update_delta(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let prev = self.delta_base().data;
        let current = self._data(algorithm);

        let perc = if prev.is_none() {
            // Mirrors: if prev is None: perc = 0.0
            0.0
        } else if self.delta_base().counter > 0
            && self.delta_base().counter % (self.delta_base().n_skip + 1) != 0
        {
            // Mirrors: elif self.counter > 0 and self.counter % (self.n_skip + 1) != 0:
            //              perc = self.perc
            self.base().perc
        } else {
            let tol = self.delta_base().tol;
            let delta = self._delta(prev.unwrap(), current);
            if delta <= tol {
                return 1.0;
            } else {
                let v = delta - tol;
                1.0 / (1.0 + v)
            }
        };

        // Mirrors: self.data = current; self.counter += 1
        self.delta_base_mut().data = Some(current);
        self.delta_base_mut().counter += 1;

        perc
    }
}
