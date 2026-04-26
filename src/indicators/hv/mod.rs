use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Axis};

use crate::{
    indicators::distance_indicator::derive_ideal_and_nadir_from_pf,
    util::normalization::PreNormalization,
};

// -------------------------------------------------------------------------------------------------
// Hypervolume
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.indicators.hv.Hypervolume(Indicator)`.
pub struct Hypervolume {
    pub nds: bool,
    pub ref_point: Array1<f64>,
    pub pre_norm: PreNormalization,
}

impl Hypervolume {
    /// Mirrors `Hypervolume.__init__(ref_point, pf, nds, norm_ref_point, ideal, nadir)`.
    pub fn new(
        ref_point: Option<Array1<f64>>,
        pf: Option<Array2<f64>>,
        nds: Option<bool>,
        norm_ref_point: Option<bool>,
        ideal: Option<Array1<f64>>,
        nadir: Option<Array1<f64>>,
    ) -> Result<Self> {
        let norm_ref_point = norm_ref_point.unwrap_or(true);

        let (ideal, nadir) = match &pf {
            Some(pf_arr) => derive_ideal_and_nadir_from_pf(pf_arr, ideal, nadir),
            None => (ideal, nadir),
        };

        let pre_norm = PreNormalization::new(true, ideal, nadir)?;

        let mut ref_pt = match ref_point {
            Some(rp) => rp,
            None => match &pf {
                Some(pf_arr) => {
                    // ref_point = pf.max(axis=0)
                    pf_arr.map_axis(Axis(0), |col| {
                        col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    })
                }
                None => {
                    return Err(anyhow!(
                        "For Hypervolume a reference point needs to be provided!"
                    ));
                }
            },
        };

        if norm_ref_point {
            // Normalize: reshape (n,) -> (1, n), forward, extract row 0
            let rp_2d = ref_pt.view().insert_axis(Axis(0)).to_owned();
            let rp_norm = pre_norm.normalization.forward(&rp_2d);
            ref_pt = rp_norm.row(0).to_owned();
        }

        Ok(Self {
            nds: nds.unwrap_or(true),
            ref_point: ref_pt,
            pre_norm,
        })
    }

    /// Mirrors `Hypervolume._do(F)` — normalises F then computes hypervolume.
    pub fn do_calc(&self, f: &Array2<f64>) -> f64 {
        let f_norm = self.pre_norm.normalization.forward(f);
        hypervolume(&f_norm, &self.ref_point)
    }
}

// -------------------------------------------------------------------------------------------------
// HV alias
// -------------------------------------------------------------------------------------------------

/// Mirrors `class HV(Hypervolume): pass`.
pub type HV = Hypervolume;

// -------------------------------------------------------------------------------------------------
// hvc_looped
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.indicators.hv.hvc_looped(ref_point, F, func)`.
///
/// Computes hypervolume contribution of each point in `F` using `func`.
pub fn hvc_looped<F>(ref_point: &Array1<f64>, f: &Array2<f64>, func: F) -> Array1<f64>
where
    F: Fn(&Array1<f64>, &Array2<f64>) -> f64,
{
    let hv = func(ref_point, f);
    let n = f.nrows();

    Array1::from_shape_fn(n, |k| {
        // v[k] = False — exclude row k
        let indices: Vec<usize> = (0..n).filter(|&i| i != k).collect();
        let f_without_k = f.select(Axis(0), &indices);
        let hv_k = func(ref_point, &f_without_k);
        hv - hv_k
    })
}
