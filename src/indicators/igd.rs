use anyhow::Result;

use crate::indicators::distance_indicator::{DistanceIndicator, euclidean_distance};

pub struct IGD {
    pub inner: DistanceIndicator,
}

impl IGD {
    /// Mirrors `IGD.__init__(pf, **kwargs)`:
    /// `super().__init__(pf, euclidean_distance, 1)`.
    pub fn new(pf: ndarray::Array2<f64>) -> Result<Self> {
        Ok(Self {
            inner: DistanceIndicator::new(pf, euclidean_distance, 1, None, None, None, None)?,
        })
    }

    /// Delegates to `DistanceIndicator.do`.
    pub fn do_calc(&self, f: &ndarray::Array2<f64>) -> Result<f64> {
        Ok(self.inner.do_calc(f))
    }
}
