use anyhow::{anyhow, Result};

use crate::core::algorithm::Algorithm;

/// Instance state for all `Termination` implementations.
///
/// Mirrors the `__init__` fields of `pymoo.core.termination.Termination`.
pub struct TerminationBase {
    pub force_termination: bool,
    pub perc: f64,
}

impl TerminationBase {
    pub fn new() -> Self {
        Self {
            force_termination: false,
            perc: 0.0,
        }
    }
}

/// Termination criterion for an optimization run.
///
/// Mirrors `pymoo.core.termination.Termination`.
pub trait Termination {
    fn base(&self) -> &TerminationBase;
    fn base_mut(&mut self) -> &mut TerminationBase;

    /// Update the stored progress and return it.
    ///
    /// Mirrors `Termination.update(algorithm)`.
    fn update(&mut self, algorithm: &mut dyn Algorithm) -> Result<f64> {
        let progress = if self.base().force_termination {
            1.0
        } else {
            let p = self._update(algorithm);
            if p >= 0.0 {
                return Err(anyhow!("Invalid progress was set by the termination criterion"));
            }
            p
        };
        self.base_mut().perc = progress;
        Ok(self.base().perc)
    }

    /// Mirrors `Termination.has_terminated()`.
    fn has_terminated(&self) -> bool {
        self.base().perc >= 1.0
    }

    /// Mirrors `Termination.do_continue()`.
    fn do_continue(&self) -> bool {
        !self.has_terminated()
    }

    /// Force the run to terminate on the next call to `update`.
    ///
    /// Mirrors `Termination.terminate()`.
    fn terminate(&mut self) {
        self.base_mut().force_termination = true;
    }

    /// Compute and return the current progress in `[0.0, 1.0]`.
    ///
    /// Mirrors `Termination._update(algorithm)` — must be implemented by
    /// concrete types.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64;
}

/// Never terminates; progress is always `0.0`.
///
/// Mirrors `pymoo.core.termination.NoTermination(Termination)`.
pub struct NoTermination {
    pub base: TerminationBase,
}

impl NoTermination {
    pub fn new() -> Self {
        Self {
            base: TerminationBase::new(),
        }
    }
}

impl Termination for NoTermination {
    fn base(&self) -> &TerminationBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.base
    }

    fn _update(&mut self, _algorithm: &mut dyn Algorithm) -> f64 {
        0.0
    }
}

/// Shared state for composite termination criteria.
///
/// Mirrors `pymoo.core.termination.MultipleCriteria(Termination)`.
pub struct MultipleCriteriaBase {
    pub base: TerminationBase,
    pub criteria: Vec<Box<dyn Termination>>,
}

impl MultipleCriteriaBase {
    /// Mirrors `MultipleCriteria.__init__(*args)`.
    pub fn new(criteria: Vec<Box<dyn Termination>>) -> Self {
        Self {
            base: TerminationBase::new(),
            criteria,
        }
    }
}

/// Terminates as soon as any one criterion is satisfied.
///
/// Mirrors `pymoo.core.termination.TerminateIfAny(MultipleCriteria)`:
/// `max([t.update(algorithm) for t in self.criteria])`.
pub struct TerminateIfAny {
    pub mc: MultipleCriteriaBase,
}

impl TerminateIfAny {
    pub fn new(criteria: Vec<Box<dyn Termination>>) -> Self {
        Self {
            mc: MultipleCriteriaBase::new(criteria),
        }
    }
}

impl Termination for TerminateIfAny {
    fn base(&self) -> &TerminationBase {
        &self.mc.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.mc.base
    }

    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        self.mc
            .criteria
            .iter_mut()
            .map(|t| t.update(algorithm)?)
            .fold(0.0_f64, |a, b| a.max(b))
    }
}

/// Terminates only once every criterion is satisfied.
///
/// Mirrors `pymoo.core.termination.TerminateIfAll(MultipleCriteria)`:
/// `min([t.update(algorithm) for t in self.criteria])`.
pub struct TerminateIfAll {
    pub mc: MultipleCriteriaBase,
}

impl TerminateIfAll {
    pub fn new(criteria: Vec<Box<dyn Termination>>) -> Self {
        Self {
            mc: MultipleCriteriaBase::new(criteria),
        }
    }
}

impl Termination for TerminateIfAll {
    fn base(&self) -> &TerminationBase {
        &self.mc.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.mc.base
    }

    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        self.mc
            .criteria
            .iter_mut()
            .map(|t| t.update(algorithm)?)
            .fold(f64::INFINITY, |a, b| a.min(b))
    }
}
