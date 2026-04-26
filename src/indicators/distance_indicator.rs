use anyhow::Result;
use ndarray::{Array1, Array2, Axis};

use crate::util::normalization::PreNormalization;

/// Distance function type: takes two aligned 2-D row-sets and an optional
/// per-dimension normalisation vector, returns a 1-D distance array.
///
/// Mirrors the `func_dist` argument accepted by `vectorized_cdist`.
pub type DistFunc = fn(&Array2<f64>, &Array2<f64>, Option<&Array1<f64>>) -> Array1<f64>;

// -------------------------------------------------------------------------------------------------
// Distance functions
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.indicators.distance_indicator.euclidean_distance(a, b, norm=None)`:
/// `sqrt(sum(((a - b) / norm)^2, axis=1))`.
pub fn euclidean_distance(
    a: &Array2<f64>,
    b: &Array2<f64>,
    norm: Option<&Array1<f64>>,
) -> Array1<f64> {
    let diff = a - b;
    let scaled = match norm {
        Some(n) => diff / n,
        None => diff,
    };
    scaled.mapv(|v| v * v).sum_axis(Axis(1)).mapv(f64::sqrt)
}

/// Mirrors `pymoo.indicators.distance_indicator.modified_distance(z, a, norm=None)`:
/// clips negative deltas to 0 before computing Euclidean distance.
pub fn modified_distance(
    z: &Array2<f64>,
    a: &Array2<f64>,
    norm: Option<&Array1<f64>>,
) -> Array1<f64> {
    let mut d = a - z;
    d.mapv_inplace(|v| v.max(0.0));
    let d = match norm {
        Some(n) => d / n,
        None => d,
    };
    d.mapv(|v| v * v).sum_axis(Axis(1)).mapv(f64::sqrt)
}

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.indicators.distance_indicator.derive_ideal_and_nadir_from_pf`.
pub fn derive_ideal_and_nadir_from_pf(
    pf: &Array2<f64>,
    ideal: Option<Array1<f64>>,
    nadir: Option<Array1<f64>>,
) -> (Option<Array1<f64>>, Option<Array1<f64>>) {
    let ideal = ideal.or_else(|| {
        Some(pf.map_axis(Axis(0), |col| {
            col.iter().cloned().fold(f64::INFINITY, f64::min)
        }))
    });
    let nadir = nadir.or_else(|| {
        Some(pf.map_axis(Axis(0), |col| {
            col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }))
    });
    (ideal, nadir)
}

/// Mirrors `pymoo.util.misc.vectorized_cdist`:
/// returns a distance matrix D of shape `(n_a, n_b)` where `D[i, j] = dist(a[i], b[j])`.
///
/// Implements `np.repeat` / `np.tile` tiling manually to pass all pairs to `dist_func`
/// in a single vectorised call.
fn vectorized_cdist(
    a: &Array2<f64>,
    b: &Array2<f64>,
    dist_func: DistFunc,
    norm: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let n_a = a.nrows();
    let n_b = b.nrows();
    let n_cols = a.ncols();

    // Mirrors: u = np.repeat(A, n_b, axis=0)  — each row of A repeated n_b times
    let mut u = Array2::<f64>::zeros((n_a * n_b, n_cols));
    for i in 0..n_a {
        for k in 0..n_b {
            u.row_mut(i * n_b + k).assign(&a.row(i));
        }
    }

    // Mirrors: v = np.tile(B, (n_a, 1))  — B repeated n_a times
    let mut v = Array2::<f64>::zeros((n_a * n_b, n_cols));
    for i in 0..n_a {
        for j in 0..n_b {
            v.row_mut(i * n_b + j).assign(&b.row(j));
        }
    }

    let d = dist_func(&u, &v, norm);
    Ok(d.into_shape((n_a, n_b))?)
}

// -------------------------------------------------------------------------------------------------
// DistanceIndicator
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.indicators.distance_indicator.DistanceIndicator`.
pub struct DistanceIndicator {
    pub dist_func: DistFunc,
    /// Axis over which to take the minimum of the distance matrix.
    /// `axis=1` → IGD (min over PF for each objective-space point).
    pub axis: usize,
    pub norm_by_dist: bool,
    /// Pareto front, already normalised.
    pub pf: Array2<f64>,
    /// Pre-normalization (mirrors `Indicator`'s `normalization` attribute).
    pub pre_norm: PreNormalization,
}

impl DistanceIndicator {
    /// Mirrors `DistanceIndicator.__init__(pf, dist_func, axis, zero_to_one=False, ideal=None, nadir=None, norm_by_dist=False)`.
    pub fn new(
        pf: Array2<f64>,
        dist_func: DistFunc,
        axis: usize,
        zero_to_one: Option<bool>,
        ideal: Option<Array1<f64>>,
        nadir: Option<Array1<f64>>,
        norm_by_dist: Option<bool>,
    ) -> Result<Self> {
        let (ideal, nadir) = derive_ideal_and_nadir_from_pf(&pf, ideal, nadir);
        let pre_norm = PreNormalization::new(zero_to_one.unwrap_or(false), ideal, nadir)?;
        let pf_norm = pre_norm.normalization.forward(&pf);
        Ok(Self {
            dist_func,
            axis,
            norm_by_dist: norm_by_dist.unwrap_or(false),
            pf: pf_norm,
            pre_norm,
        })
    }

    /// Mirrors `DistanceIndicator._do(F)` / `Indicator.do(F)`.
    pub fn do_calc(&self, f: &Array2<f64>) -> f64 {
        let f_norm = self.pre_norm.normalization.forward(f);

        // Mirrors: norm = nadir - ideal if norm_by_dist else 1.0
        let norm_arr: Option<Array1<f64>> = if self.norm_by_dist {
            match (self.pre_norm.ideal.as_ref(), self.pre_norm.nadir.as_ref()) {
                (Some(ideal), Some(nadir)) => Some(nadir - ideal),
                _ => None,
            }
        } else {
            None
        };

        let d = vectorized_cdist(&self.pf, &f_norm, self.dist_func, norm_arr.as_ref());

        // Mirrors: np.mean(np.min(D, axis=self.axis))
        let min_d = d.map_axis(Axis(self.axis), |row| {
            row.iter().cloned().fold(f64::INFINITY, f64::min)
        });
        min_d.mean().unwrap_or(0.0)
    }
}
