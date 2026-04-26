use anyhow::Result;
use ndarray::Array2;

use crate::{
    core::{
        algorithm::Algorithm,
        individual::{IndividualField, Value},
        termination::{Termination, TerminationBase},
    },
    termination::delta::{DeltaToleranceBase, DeltaToleranceTermination},
};

/// Mirrors `pymoo.termination.xtol.DesignSpaceTermination`.
pub struct DesignSpaceTermination {
    pub delta: DeltaToleranceBase,
    /// Stores the design-variable matrix from the previous iteration.
    pub prev_x: Option<Array2<f64>>,
    pub n_skip: usize,
}

impl DesignSpaceTermination {
    /// Mirrors `DesignSpaceTermination.__init__(tol=0.005, n_skip=0)`.
    pub fn new(tol: Option<f64>, n_skip: Option<usize>) -> Result<Self> {
        let n_skip = n_skip.unwrap_or(0);
        let tol = tol.unwrap_or(0.005);
        Ok(Self {
            delta: DeltaToleranceBase::new(tol, Some(n_skip))?,
            prev_x: None,
            n_skip,
        })
    }

    /// Mirrors `DesignSpaceTermination._delta(prev, current)`:
    /// `return IGD(current.astype(float)).do(prev.astype(float))`.
    /// Falls back to `f64::INFINITY` on failure (mirrors the bare `except` clause).
    fn _delta_array(&self, prev: &Array2<f64>, current: &Array2<f64>) -> f64 {
        IGD::new(current).do_calc(prev).unwrap_or(f64::INFINITY)
    }

    /// Mirrors `DesignSpaceTermination._data(algorithm)`:
    /// returns the (optionally normalised) design-variable matrix.
    fn _data_array(&self, algorithm: &dyn Algorithm) -> Option<Array2<f64>> {
        let x = match algorithm.base().opt.as_ref()?.get(&IndividualField::X) {
            Value::FloatMatrix(x) => x,
            _ => return None,
        };

        // Mirrors: if X.dtype != object and problem.has_bounds(): X = normalize(X, xl=xl, xu=xu)
        let problem = algorithm.base().problem.as_ref()?;
        if problem.has_bounds() {
            Some(normalize(x, problem.xl(), problem.xu()))
        } else {
            Some(x)
        }
    }
}

impl Termination for DesignSpaceTermination {
    fn base(&self) -> &TerminationBase {
        &self.delta.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.delta.base
    }

    /// Mirrors `DeltaToleranceTermination._update` with array-typed `_data` / `_delta`.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let prev = self.prev_x.take();
        let current = self._data_array(algorithm);

        let perc = match (&prev, &current) {
            (None, _) => 0.0,
            (Some(_), _)
                if self.delta.counter > 0 && self.delta.counter % (self.n_skip + 1) != 0 =>
            {
                self.delta.base.perc
            }
            (Some(p), Some(c)) => {
                let delta = self._delta_array(p, c);
                if delta <= self.delta.tol {
                    self.prev_x = current;
                    self.delta.counter += 1;
                    return 1.0;
                } else {
                    let v = delta - self.delta.tol;
                    1.0 / (1.0 + v)
                }
            }
            (Some(_), None) => 0.0,
        };

        self.prev_x = current;
        self.delta.counter += 1;
        perc
    }
}

impl DeltaToleranceTermination for DesignSpaceTermination {
    fn delta_base(&self) -> &DeltaToleranceBase {
        &self.delta
    }

    fn delta_base_mut(&mut self) -> &mut DeltaToleranceBase {
        &mut self.delta
    }

    /// Not used — array-based logic is handled in `_update` directly.
    fn _delta(&self, prev: f64, current: f64) -> f64 {
        (prev - current).abs()
    }

    /// Not used — array-based logic is handled in `_update` directly.
    fn _data(&self, _algorithm: &dyn Algorithm) -> f64 {
        0.0
    }
}
