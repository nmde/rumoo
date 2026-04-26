use ndarray::{Array1, Array2};

use crate::{
    core::{
        algorithm::Algorithm,
        individual::{IndividualField, Value},
        termination::{Termination, TerminationBase},
    },
    termination::delta::{DeltaToleranceBase, DeltaToleranceTermination},
    util::normalization::normalize,
};

/// Mirrors `pymoo.termination.ftol.calc_delta(a, b)`:
/// `np.max(np.abs(a - b))`.
pub fn calc_delta(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(f64::abs).fold(f64::NEG_INFINITY, f64::max)
}

/// Mirrors `pymoo.termination.ftol.calc_delta_norm(a, b, norm)`:
/// `np.max(np.abs((a - b) / norm))`.
pub fn calc_delta_norm(a: &Array1<f64>, b: &Array1<f64>, norm: &Array1<f64>) -> f64 {
    ((a - b) / norm)
        .mapv(f64::abs)
        .fold(f64::NEG_INFINITY, f64::max)
}

// ---------------------------------------------------------------------------
// SingleObjectiveSpaceTermination
// ---------------------------------------------------------------------------

/// Mirrors `pymoo.termination.ftol.SingleObjectiveSpaceTermination`.
pub struct SingleObjectiveSpaceTermination {
    pub delta: DeltaToleranceBase,
    pub only_feas: bool,
}

impl SingleObjectiveSpaceTermination {
    /// Mirrors `SingleObjectiveSpaceTermination.__init__(tol=1e-6, only_feas=True)`.
    pub fn new(tol: Option<f64>, only_feas: Option<bool>) -> Self {
        Self {
            delta: DeltaToleranceBase::new(tol.unwrap_or(1e-6), Some(0))?,
            only_feas: only_feas.unwrap_or(true),
        }
    }
}

impl Termination for SingleObjectiveSpaceTermination {
    fn base(&self) -> &TerminationBase {
        &self.delta.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.delta.base
    }

    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        self._update_delta(algorithm)
    }
}

impl DeltaToleranceTermination for SingleObjectiveSpaceTermination {
    fn delta_base(&self) -> &DeltaToleranceBase {
        &self.delta
    }

    fn delta_base_mut(&mut self) -> &mut DeltaToleranceBase {
        &mut self.delta
    }

    /// Mirrors `SingleObjectiveSpaceTermination._delta`:
    /// returns `f64::INFINITY` if either value is infinite, else `max(0, prev - current)`.
    fn _delta(&self, prev: f64, current: f64) -> f64 {
        if prev.is_infinite() || current.is_infinite() {
            f64::INFINITY
        } else {
            (prev - current).max(0.0)
        }
    }

    /// Mirrors `SingleObjectiveSpaceTermination._data`:
    /// minimum feasible F value, or `f64::INFINITY` if none.
    fn _data(&self, algorithm: &dyn Algorithm) -> f64 {
        let opt = match algorithm.base().opt.as_ref() {
            Some(o) => o,
            None => return f64::INFINITY,
        };

        let f_mat = match opt.get(&IndividualField::F) {
            Value::FloatMatrix(m) => m,
            _ => return f64::INFINITY,
        };

        if self.only_feas {
            let feas = match opt.get(&IndividualField::Feas) {
                Value::BoolArray(b) => b,
                _ => return f64::INFINITY,
            };
            let vals: Vec<f64> = f_mat
                .outer_iter()
                .zip(feas.iter())
                .filter(|(_, &ok)| ok)
                .map(|(row, _)| row[0])
                .collect();
            if vals.is_empty() {
                f64::INFINITY
            } else {
                vals.into_iter().fold(f64::INFINITY, f64::min)
            }
        } else {
            f_mat.iter().cloned().fold(f64::INFINITY, f64::min)
        }
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveSpaceTermination
// ---------------------------------------------------------------------------

/// Internal snapshot of the Pareto-front state for one iteration.
struct FrontData {
    ideal: Option<Array1<f64>>,
    nadir: Option<Array1<f64>>,
    f: Array2<f64>,
    feas: bool,
}

/// Mirrors `pymoo.termination.ftol.MultiObjectiveSpaceTermination`.
pub struct MultiObjectiveSpaceTermination {
    pub delta: DeltaToleranceBase,
    pub only_feas: bool,
    pub delta_ideal: Option<f64>,
    pub delta_nadir: Option<f64>,
    pub delta_f: Option<f64>,
    prev_data: Option<FrontData>,
}

impl MultiObjectiveSpaceTermination {
    /// Mirrors `MultiObjectiveSpaceTermination.__init__(tol=0.0025, only_feas=True)`.
    pub fn new(tol: Option<f64>, only_feas: Option<bool>, n_skip: Option<usize>) -> Self {
        Self {
            delta: DeltaToleranceBase::new(tol.unwrap_or(0.0025), Some(n_skip.unwrap_or(0)))?,
            only_feas: only_feas.unwrap_or(true),
            delta_ideal: None,
            delta_nadir: None,
            delta_f: None,
            prev_data: None,
        }
    }

    /// Mirrors `MultiObjectiveSpaceTermination._data(algorithm)`.
    fn _data_front(&self, algorithm: &dyn Algorithm) -> FrontData {
        let opt = match algorithm.base().opt.as_ref() {
            None => {
                return FrontData {
                    ideal: None,
                    nadir: None,
                    f: Array2::zeros((0, 0)),
                    feas: false,
                };
            }
            Some(o) => o,
        };

        let f_mat = match opt.get(&IndividualField::F) {
            Value::FloatMatrix(m) => m,
            _ => {
                return FrontData {
                    ideal: None,
                    nadir: None,
                    f: Array2::zeros((0, 0)),
                    feas: false,
                };
            }
        };

        let f_filtered: Array2<f64> = if self.only_feas {
            match opt.get(&IndividualField::Feas) {
                Value::BoolArray(mask) => {
                    let rows: Vec<_> = f_mat
                        .outer_iter()
                        .zip(mask.iter())
                        .filter(|(_, &ok)| ok)
                        .map(|(r, _)| r.to_owned())
                        .collect();
                    if rows.is_empty() {
                        Array2::zeros((0, f_mat.ncols()))
                    } else {
                        ndarray::stack(
                            ndarray::Axis(0),
                            &rows.iter().map(|r| r.view()).collect::<Vec<_>>(),
                        )
                        .unwrap_or_else(|_| Array2::zeros((0, f_mat.ncols())))
                    }
                }
                _ => f_mat.clone(),
            }
        } else {
            f_mat.clone()
        };

        if f_filtered.nrows() > 0 {
            let ideal = f_filtered.map_axis(ndarray::Axis(0), |col| {
                col.iter().cloned().fold(f64::INFINITY, f64::min)
            });
            let nadir = f_filtered.map_axis(ndarray::Axis(0), |col| {
                col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            });
            FrontData {
                ideal: Some(ideal),
                nadir: Some(nadir),
                f: f_filtered,
                feas: true,
            }
        } else {
            FrontData {
                ideal: None,
                nadir: None,
                f: f_filtered,
                feas: false,
            }
        }
    }

    /// Mirrors `MultiObjectiveSpaceTermination._delta(prev, current)`.
    fn _delta_front(&mut self, prev: &FrontData, current: &FrontData) -> f64 {
        if !prev.feas || !current.feas {
            return f64::INFINITY;
        }

        let c_ideal = current.ideal.as_ref().unwrap();
        let c_nadir = current.nadir.as_ref().unwrap();
        let p_ideal = prev.ideal.as_ref().unwrap();
        let p_nadir = prev.nadir.as_ref().unwrap();

        // Mirrors: norm = current["nadir"] - current["ideal"]; norm[norm < 1e-32] = 1.0
        let mut norm = c_nadir - c_ideal;
        norm.mapv_inplace(|v| if v < 1e-32 { 1.0 } else { v });

        let delta_ideal = calc_delta_norm(c_ideal, p_ideal, &norm);
        let delta_nadir = calc_delta_norm(c_nadir, p_nadir, &norm);

        let c_n = normalize(&current.f, Some(c_ideal), Some(c_nadir));
        let p_n = normalize(&prev.f, Some(c_ideal), Some(c_nadir));

        // Mirrors: delta_f = IGD(c_N).do(p_N)
        let delta_f = IGD::new(&c_n).do_calc(&p_n).unwrap_or(f64::INFINITY);

        self.delta_ideal = Some(delta_ideal);
        self.delta_nadir = Some(delta_nadir);
        self.delta_f = Some(delta_f);

        delta_ideal.max(delta_nadir).max(delta_f)
    }
}

impl Termination for MultiObjectiveSpaceTermination {
    fn base(&self) -> &TerminationBase {
        &self.delta.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.delta.base
    }

    /// Mirrors `DeltaToleranceTermination._update` with `FrontData`-typed snapshots.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let prev = self.prev_data.take();
        let current = self._data_front(algorithm);

        let perc = match &prev {
            None => 0.0,
            Some(_)
                if self.delta.counter > 0 && self.delta.counter % (self.delta.n_skip + 1) != 0 =>
            {
                self.delta.base.perc
            }
            Some(p) => {
                let delta = self._delta_front(p, &current);
                if delta <= self.delta.tol {
                    self.prev_data = Some(current);
                    self.delta.counter += 1;
                    return 1.0;
                } else {
                    let v = delta - self.delta.tol;
                    1.0 / (1.0 + v)
                }
            }
        };

        self.prev_data = Some(current);
        self.delta.counter += 1;
        perc
    }
}

impl DeltaToleranceTermination for MultiObjectiveSpaceTermination {
    fn delta_base(&self) -> &DeltaToleranceBase {
        &self.delta
    }

    fn delta_base_mut(&mut self) -> &mut DeltaToleranceBase {
        &mut self.delta
    }

    fn _delta(&self, prev: f64, current: f64) -> f64 {
        (prev - current).max(0.0)
    }

    fn _data(&self, _algorithm: &dyn Algorithm) -> f64 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveSpaceTerminationWithRenormalization
// ---------------------------------------------------------------------------

/// Mirrors `pymoo.termination.ftol.MultiObjectiveSpaceTerminationWithRenormalization`.
pub struct MultiObjectiveSpaceTerminationWithRenormalization {
    pub inner: MultiObjectiveSpaceTermination,
    pub n: usize,
    pub all_to_current: bool,
    pub sliding_window: bool,
    pub indicator: String,
}

impl MultiObjectiveSpaceTerminationWithRenormalization {
    /// Mirrors `__init__(n=30, all_to_current=False, sliding_window=True, indicator="igd")`.
    pub fn new(
        n: Option<usize>,
        all_to_current: Option<bool>,
        sliding_window: Option<bool>,
        indicator: Option<String>,
        tol: Option<f64>,
        only_feas: Option<bool>,
        n_skip: Option<usize>,
    ) -> Self {
        Self {
            inner: MultiObjectiveSpaceTermination::new(tol, only_feas, n_skip),
            n: n.unwrap_or(30),
            all_to_current: all_to_current.unwrap_or(false),
            sliding_window: sliding_window.unwrap_or(true),
            indicator: indicator.unwrap_or_else(|| "igd".to_string()),
        }
    }
}

impl Termination for MultiObjectiveSpaceTerminationWithRenormalization {
    fn base(&self) -> &TerminationBase {
        self.inner.base()
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        self.inner.base_mut()
    }

    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        self.inner._update(algorithm)
    }
}
