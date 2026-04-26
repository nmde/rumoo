use anyhow::Result;

use crate::indicators::distance_indicator::{DistanceIndicator, euclidean_distance};

pub struct GD {
    pub inner: DistanceIndicator,
}

impl GD {
    /// Mirrors `GD.__init__(pf, **kwargs)`:
    /// `super().__init__(pf, euclidean_distance, 0)`.
    pub fn new(pf: ndarray::Array2<f64>) -> Result<Self> {
        Ok(Self {
            inner: DistanceIndicator::new(pf, euclidean_distance, 0, None, None, None, None)?,
        })
    }

    /// Delegates to `DistanceIndicator.do`.
    pub fn do_calc(&self, f: &ndarray::Array2<f64>) -> Result<f64> {
        Ok(self.inner.do_calc(f))
    }
}
