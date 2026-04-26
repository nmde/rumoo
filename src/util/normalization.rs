use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;

// -------------------------------------------------------------------------------------------------
// Normalization trait
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.normalization.Normalization` (abstract base).
pub trait Normalization {
    fn forward(&self, x: &Array2<f64>) -> Array2<f64>;
    fn backward(&self, x: &Array2<f64>) -> Array2<f64>;
}

// -------------------------------------------------------------------------------------------------
// NoNormalization
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.normalization.NoNormalization` — identity transform.
pub struct NoNormalization;

impl Normalization for NoNormalization {
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }

    fn backward(&self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }
}

// -------------------------------------------------------------------------------------------------
// ZeroToOneNormalization
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.normalization.ZeroToOneNormalization`.
///
/// Normalises each column independently, handling `NaN` bounds via column masks.
pub struct ZeroToOneNormalization {
    pub xl: Option<Array1<f64>>,
    pub xu: Option<Array1<f64>>,
    /// Columns where xl is known and xu is NaN.
    pub xl_only: Option<Array1<bool>>,
    /// Columns where xu is known and xl is NaN.
    pub xu_only: Option<Array1<bool>>,
    /// Columns where both xl and xu are NaN.
    pub both_nan: Option<Array1<bool>>,
    /// Columns where neither xl nor xu is NaN.
    pub neither_nan: Option<Array1<bool>>,
}

impl ZeroToOneNormalization {
    /// Mirrors `ZeroToOneNormalization.__init__(xl=None, xu=None)`.
    pub fn new(xl: Option<Array1<f64>>, xu: Option<Array1<f64>>) -> Result<Self> {
        if xl.is_none() && xu.is_none() {
            return Ok(Self {
                xl: None,
                xu: None,
                xl_only: None,
                xu_only: None,
                both_nan: None,
                neither_nan: None,
            });
        }

        let n = xl.as_ref().or(xu.as_ref()).map(|a| a.len()).unwrap_or(0);

        let mut xl = xl.unwrap_or_else(|| Array1::from_elem(n, f64::NAN));
        let mut xu = xu.unwrap_or_else(|| Array1::from_elem(n, f64::NAN));

        // Mirrors: xu[xl == xu] = np.nan
        for i in 0..n {
            if xl[i] == xu[i] {
                xu[i] = f64::NAN;
            }
        }

        let xl_nan: Array1<bool> = xl.mapv(f64::is_nan);
        let xu_nan: Array1<bool> = xu.mapv(f64::is_nan);

        let xl_only = Array1::from_shape_fn(n, |i| !xl_nan[i] && xu_nan[i]);
        let xu_only = Array1::from_shape_fn(n, |i| xl_nan[i] && !xu_nan[i]);
        let both_nan = Array1::from_shape_fn(n, |i| xl_nan[i] && xu_nan[i]);
        let neither_nan = Array1::from_shape_fn(n, |i| !xl_nan[i] && !xu_nan[i]);

        // Mirrors: assert np.all(np.logical_or(xu >= xl, any_nan))
        for i in 0..n {
            if !xl_nan[i] && !xu_nan[i] && xu[i] >= xl[i] {
                return Err(anyhow!("xl must be less or equal than xu."));
            }
        }

        Ok(Self {
            xl: Some(xl),
            xu: Some(xu),
            xl_only: Some(xl_only),
            xu_only: Some(xu_only),
            both_nan: Some(both_nan),
            neither_nan: Some(neither_nan),
        })
    }
}

impl Normalization for ZeroToOneNormalization {
    /// Mirrors `ZeroToOneNormalization.forward(X)`.
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        if self.xl.is_none() && self.xu.is_none() {
            return x.clone();
        }
        let xl = self.xl.as_ref().unwrap();
        let xu = self.xu.as_ref().unwrap();
        let neither_nan = self.neither_nan.as_ref().unwrap();
        let xl_only = self.xl_only.as_ref().unwrap();
        let xu_only = self.xu_only.as_ref().unwrap();

        let mut n = x.clone();
        let ncols = x.ncols();

        for j in 0..ncols {
            if neither_nan[j] {
                let mut col = n.column_mut(j);
                let range = xu[j] - xl[j];
                col.mapv_inplace(|v| (v - xl[j]) / range);
            } else if xl_only[j] {
                let mut col = n.column_mut(j);
                col.mapv_inplace(|v| v - xl[j]);
            } else if xu_only[j] {
                let mut col = n.column_mut(j);
                col.mapv_inplace(|v| 1.0 - (xu[j] - v));
            }
            // both_nan: leave unchanged
        }
        n
    }

    /// Mirrors `ZeroToOneNormalization.backward(N)`.
    fn backward(&self, x: &Array2<f64>) -> Array2<f64> {
        if self.xl.is_none() && self.xu.is_none() {
            return x.clone();
        }
        let xl = self.xl.as_ref().unwrap();
        let xu = self.xu.as_ref().unwrap();
        let neither_nan = self.neither_nan.as_ref().unwrap();
        let xl_only = self.xl_only.as_ref().unwrap();
        let xu_only = self.xu_only.as_ref().unwrap();

        let mut result = x.clone();
        let ncols = x.ncols();

        for j in 0..ncols {
            if neither_nan[j] {
                let mut col = result.column_mut(j);
                col.mapv_inplace(|v| xl[j] + v * (xu[j] - xl[j]));
            } else if xl_only[j] {
                let mut col = result.column_mut(j);
                col.mapv_inplace(|v| v + xl[j]);
            } else if xu_only[j] {
                let mut col = result.column_mut(j);
                col.mapv_inplace(|v| xu[j] - (1.0 - v));
            }
        }
        result
    }
}

// -------------------------------------------------------------------------------------------------
// SimpleZeroToOneNormalization
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.normalization.SimpleZeroToOneNormalization`.
pub struct SimpleZeroToOneNormalization {
    pub xl: Option<Array1<f64>>,
    pub xu: Option<Array1<f64>>,
    pub estimate_bounds: bool,
}

impl SimpleZeroToOneNormalization {
    /// Mirrors `SimpleZeroToOneNormalization.__init__(xl=None, xu=None, estimate_bounds=True)`.
    pub fn new(
        xl: Option<Array1<f64>>,
        xu: Option<Array1<f64>>,
        estimate_bounds: Option<bool>,
    ) -> Self {
        Self {
            xl,
            xu,
            estimate_bounds: estimate_bounds.unwrap_or(true),
        }
    }
}

impl Normalization for SimpleZeroToOneNormalization {
    /// Mirrors `SimpleZeroToOneNormalization.forward(X)`.
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let xl = self.xl.clone().unwrap_or_else(|| {
            if self.estimate_bounds {
                x.map_axis(Axis(0), |col| {
                    col.iter().cloned().fold(f64::INFINITY, f64::min)
                })
            } else {
                Array1::zeros(x.ncols())
            }
        });
        let xu = self.xu.clone().unwrap_or_else(|| {
            if self.estimate_bounds {
                x.map_axis(Axis(0), |col| {
                    col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                })
            } else {
                Array1::ones(x.ncols())
            }
        });

        // Mirrors: denom = xu - xl; denom += (denom == 0) * 1e-32
        let denom: Array1<f64> = (&xu - &xl).mapv(|v| if v == 0.0 { 1e-32 } else { v });

        (x - &xl) / &denom
    }

    /// Mirrors `SimpleZeroToOneNormalization.backward(X)`.
    fn backward(&self, x: &Array2<f64>) -> Array2<f64> {
        let xl = self.xl.as_ref().expect("xl required for backward");
        let xu = self.xu.as_ref().expect("xu required for backward");
        x * &(xu - xl) + xl
    }
}

// -------------------------------------------------------------------------------------------------
// Functional interface
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.normalization.normalize(X, xl=None, xu=None, return_bounds=False, estimate_bounds_if_none=True)`.
pub fn normalize(
    x: &Array2<f64>,
    xl: Option<&Array1<f64>>,
    xu: Option<&Array1<f64>>,
) -> Array2<f64> {
    let n = x.ncols();

    let xl = xl.cloned().or_else(|| {
        Some(x.map_axis(Axis(0), |col| {
            col.iter().cloned().fold(f64::INFINITY, f64::min)
        }))
    });
    let xu = xu.cloned().or_else(|| {
        Some(x.map_axis(Axis(0), |col| {
            col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }))
    });

    ZeroToOneNormalization::new(xl, xu).forward(x)
}

/// Mirrors `pymoo.util.normalization.denormalize(x, xl, xu)`.
pub fn denormalize(x: &Array2<f64>, xl: &Array1<f64>, xu: &Array1<f64>) -> Array2<f64> {
    ZeroToOneNormalization::new(Some(xl.clone()), Some(xu.clone())).backward(x)
}

/// Mirrors `pymoo.util.normalization.standardize(x, return_bounds=False)`.
///
/// Returns `(standardized, mean, std)`.
pub fn standardize(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mean = x.map_axis(Axis(0), |col| col.mean().unwrap_or(0.0));
    let std = x.map_axis(Axis(0), |col| {
        let m = col.mean().unwrap_or(0.0);
        let var = col.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / col.len() as f64;
        var.sqrt()
    });
    let val = (x - &mean) / &std;
    (val, mean, std)
}

/// Mirrors `pymoo.util.normalization.destandardize(x, mean, std)`.
pub fn destandardize(x: &Array2<f64>, mean: &Array1<f64>, std: &Array1<f64>) -> Array2<f64> {
    x * std + mean
}

// -------------------------------------------------------------------------------------------------
// PreNormalization
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.normalization.PreNormalization`.
pub struct PreNormalization {
    pub ideal: Option<Array1<f64>>,
    pub nadir: Option<Array1<f64>>,
    pub normalization: Box<dyn Normalization>,
}

impl PreNormalization {
    /// Mirrors `PreNormalization.__init__(zero_to_one=False, ideal=None, nadir=None)`.
    pub fn new(
        zero_to_one: bool,
        ideal: Option<Array1<f64>>,
        nadir: Option<Array1<f64>>,
    ) -> Result<Self> {
        if zero_to_one {
            if ideal.is_some() && nadir.is_some() {
                return Err(anyhow!("For normalization, provide either pf or bounds"));
            }
            let n_dim = ideal.as_ref().unwrap().len();
            let normalization = ZeroToOneNormalization::new(ideal.clone(), nadir.clone());
            Ok(Self {
                ideal: Some(Array1::zeros(n_dim)),
                nadir: Some(Array1::ones(n_dim)),
                normalization: Box::new(normalization),
            })
        } else {
            Ok(Self {
                ideal,
                nadir,
                normalization: Box::new(NoNormalization),
            })
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Objective-space helper functions
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.normalization.find_ideal(F, current=None)`.
pub fn find_ideal(f: &Array2<f64>, current: Option<&Array1<f64>>) -> Array1<f64> {
    let p = f.map_axis(Axis(0), |col| {
        col.iter().cloned().fold(f64::INFINITY, f64::min)
    });
    match current {
        None => p,
        Some(c) => Array1::from_shape_fn(p.len(), |i| p[i].min(c[i])),
    }
}

/// Mirrors `pymoo.util.normalization.get_extreme_points_c(F, ideal_point, extreme_points=None)`.
pub fn get_extreme_points_c(
    f: &Array2<f64>,
    ideal_point: &Array1<f64>,
    extreme_points: Option<&Array2<f64>>,
) -> Array2<f64> {
    let n_obj = f.ncols();

    // Mirrors: weights = np.eye(F.shape[1]); weights[weights == 0] = 1e6
    let mut weights = Array2::<f64>::eye(n_obj);
    weights.mapv_inplace(|v| if v == 0.0 { 1e6 } else { v });

    // Mirrors: _F = np.concatenate([extreme_points, _F], axis=0) if extreme_points is not None
    let combined: Array2<f64> = match extreme_points {
        Some(ep) => ndarray::concatenate(Axis(0), &[ep.view(), f.view()]).unwrap(),
        None => f.clone(),
    };

    // Mirrors: __F = _F - ideal_point; __F[__F < 1e-3] = 0
    let mut shifted = &combined - ideal_point;
    shifted.mapv_inplace(|v| if v < 1e-3 { 0.0 } else { v });

    // Mirrors: F_asf = np.max(__F * weights[:, None, :], axis=2)
    // weights[i]: weight vector i; F_asf[i, j] = max(__F[j, :] * weights[i, :])
    let n_ind = shifted.nrows();
    let mut f_asf = Array2::<f64>::zeros((n_obj, n_ind));
    for i in 0..n_obj {
        for j in 0..n_ind {
            f_asf[[i, j]] = (0..n_obj)
                .map(|k| shifted[[j, k]] * weights[[i, k]])
                .fold(f64::NEG_INFINITY, f64::max);
        }
    }

    // Mirrors: I = np.argmin(F_asf, axis=1); extreme_points = _F[I, :]
    let indices: Vec<usize> = (0..n_obj)
        .map(|i| {
            f_asf
                .row(i)
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect();

    let rows: Vec<_> = indices
        .iter()
        .map(|&idx| combined.row(idx).to_owned())
        .collect();
    ndarray::stack(Axis(0), &rows.iter().map(|r| r.view()).collect::<Vec<_>>())
        .unwrap_or_else(|_| Array2::zeros((0, n_obj)))
}

/// Mirrors `pymoo.util.normalization.get_nadir_point(...)`.
pub fn get_nadir_point(
    extreme_points: &Array2<f64>,
    ideal_point: &Array1<f64>,
    worst_point: &Array1<f64>,
    worst_of_front: &Array1<f64>,
    worst_of_population: &Array1<f64>,
) -> Array1<f64> {
    let n_obj = ideal_point.len();

    // Mirrors: M = extreme_points - ideal_point; b = np.ones(n); plane = np.linalg.solve(M, b)
    let m = extreme_points - ideal_point;
    let b = Array1::ones(n_obj);

    let mut nadir_point = match m.solve(&b) {
        Ok(plane) => {
            // Mirrors: intercepts = 1 / plane; nadir_point = ideal_point + intercepts
            let intercepts: Array1<f64> = plane.mapv(|v| 1.0 / v);

            // Mirrors: if not np.allclose(M @ plane, b) or np.any(intercepts <= 1e-6): raise LinAlgError
            let residual: Array1<f64> = m.dot(&plane) - &b;
            let close = residual.iter().all(|&v| v.abs() < 1e-8);
            let valid = intercepts.iter().all(|&v| v > 1e-6);

            if close && valid {
                let mut np = ideal_point + &intercepts;
                // Mirrors: b = nadir_point > worst_point; nadir_point[b] = worst_point[b]
                for i in 0..n_obj {
                    if np[i] > worst_point[i] {
                        np[i] = worst_point[i];
                    }
                }
                np
            } else {
                worst_of_front.clone()
            }
        }
        Err(_) => worst_of_front.clone(),
    };

    // Mirrors: b = nadir_point - ideal_point <= 1e-6; nadir_point[b] = worst_of_population[b]
    for i in 0..n_obj {
        if nadir_point[i] - ideal_point[i] <= 1e-6 {
            nadir_point[i] = worst_of_population[i];
        }
    }

    nadir_point
}

// -------------------------------------------------------------------------------------------------
// ObjectiveSpaceNormalization
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.util.normalization.ObjectiveSpaceNormalization`.
pub struct ObjectiveSpaceNormalization {
    pub _ideal: Option<Array1<f64>>,
    pub _infeas_ideal: Option<Array1<f64>>,
    pub _worst: Option<Array1<f64>>,
}

impl ObjectiveSpaceNormalization {
    /// Mirrors `ObjectiveSpaceNormalization.__init__()`.
    pub fn new() -> Self {
        Self {
            _ideal: None,
            _infeas_ideal: None,
            _worst: None,
        }
    }

    /// Mirrors `ObjectiveSpaceNormalization.update(pop)`.
    pub fn update(&mut self, f: &Array2<f64>, feas: &Array1<bool>) {
        // Mirrors: self._infeas_ideal = find_ideal(F, current=self._infeas_ideal)
        self._infeas_ideal = Some(find_ideal(f, self._infeas_ideal.as_ref()));

        // Mirrors: if np.any(feas): self._ideal = find_ideal(F[feas[:, 0]], self._ideal)
        if feas.iter().any(|&v| v) {
            let feas_rows: Vec<_> = f
                .outer_iter()
                .zip(feas.iter())
                .filter(|(_, &ok)| ok)
                .map(|(r, _)| r.to_owned())
                .collect();
            if !feas_rows.is_empty() {
                let f_feas = ndarray::stack(
                    Axis(0),
                    &feas_rows.iter().map(|r| r.view()).collect::<Vec<_>>(),
                )
                .unwrap();
                self._ideal = Some(find_ideal(&f_feas, self._ideal.as_ref()));
            }
        }
    }

    /// Mirrors `ObjectiveSpaceNormalization.ideal(only_feas=True)`.
    pub fn ideal(&self, only_feas: bool) -> Option<&Array1<f64>> {
        if only_feas {
            self._ideal.as_ref()
        } else {
            self._infeas_ideal.as_ref()
        }
    }
}
