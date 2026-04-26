use std::cmp::Ordering;

use ndarray::{Array1, Array2, Axis, s};

use crate::util::misc::find_duplicates;

pub enum CrowdingFunctionType {
    Cd,
    Pcd,
    Ce,
    Mnn,
    Twonn,
}

// -------------------------------------------------------------------------------------------------
// CrowdingFunction trait (mirrors CrowdingDiversity)
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.operators.survival.rank_and_crowding.metrics.CrowdingDiversity`.
pub trait CrowdingFunction {
    /// Mirrors `CrowdingDiversity.do(F, n_remove)`.
    fn do_crowd(&self, f: &Array2<f64>, n_remove: usize) -> Array1<f64>;
}

/// Function pointer type for crowding metric implementations.
pub type CrowdFn = fn(&Array2<f64>, usize) -> Array1<f64>;

// -------------------------------------------------------------------------------------------------
// FunctionalDiversity
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.operators.survival.rank_and_crowding.metrics.FunctionalDiversity`.
pub struct FunctionalDiversity {
    pub function: CrowdFn,
    pub filter_out_duplicates: bool,
}

impl FunctionalDiversity {
    pub fn new(function: CrowdFn, filter_out_duplicates: bool) -> Self {
        Self {
            function,
            filter_out_duplicates,
        }
    }
}

impl CrowdingFunction for FunctionalDiversity {
    /// Mirrors `FunctionalDiversity._do(F, **kwargs)`.
    fn do_crowd(&self, f: &Array2<f64>, n_remove: usize) -> Array1<f64> {
        let n_points = f.nrows();

        if n_points <= 2 {
            return Array1::from_elem(n_points, f64::INFINITY);
        }

        let is_unique: Vec<usize> = if self.filter_out_duplicates {
            let dups = find_duplicates(f, Some(1e-32_f64));
            (0..n_points).filter(|&i| !dups[i]).collect()
        } else {
            (0..n_points).collect()
        };

        let f_unique = f.select(Axis(0), &is_unique);
        let d_unique = (self.function)(&f_unique, n_remove);

        let mut d = Array1::<f64>::zeros(n_points);
        for (k, &i) in is_unique.iter().enumerate() {
            d[i] = d_unique[k];
        }

        d
    }
}

// -------------------------------------------------------------------------------------------------
// FuncionalDiversityMNN (note: original Python typo preserved)
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.operators.survival.rank_and_crowding.metrics.FuncionalDiversityMNN`.
pub struct FuncionalDiversityMNN {
    pub inner: FunctionalDiversity,
}

impl FuncionalDiversityMNN {
    pub fn new(function: CrowdFn, filter_out_duplicates: bool) -> Self {
        Self {
            inner: FunctionalDiversity::new(function, filter_out_duplicates),
        }
    }
}

impl CrowdingFunction for FuncionalDiversityMNN {
    /// Mirrors `FuncionalDiversityMNN._do(F, **kwargs)`.
    fn do_crowd(&self, f: &Array2<f64>, n_remove: usize) -> Array1<f64> {
        let (n_points, n_obj) = f.dim();
        if n_points <= n_obj {
            return Array1::from_elem(n_points, f64::INFINITY);
        }
        self.inner.do_crowd(f, n_remove)
    }
}

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.operators.survival.rank_and_crowding.metrics.get_crowding_function`.
pub fn get_crowding_function(label: &CrowdingFunctionType) -> Box<dyn CrowdingFunction> {
    match label {
        CrowdingFunctionType::Cd => {
            Box::new(FunctionalDiversity::new(calc_crowding_distance, false))
        }
        CrowdingFunctionType::Pcd => Box::new(FunctionalDiversity::new(calc_pcd, true)),
        CrowdingFunctionType::Ce => Box::new(FunctionalDiversity::new(calc_crowding_entropy, true)),
        CrowdingFunctionType::Mnn => Box::new(FuncionalDiversityMNN::new(calc_mnn_fast, true)),
        CrowdingFunctionType::Twonn => Box::new(FuncionalDiversityMNN::new(calc_2nn_fast, true)),
    }
}

// -------------------------------------------------------------------------------------------------
// calc_crowding_distance
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.operators.survival.rank_and_crowding.metrics.calc_crowding_distance`.
pub fn calc_crowding_distance(f: &Array2<f64>, _n_remove: usize) -> Array1<f64> {
    let (n_points, n_obj) = f.dim();

    // I = np.argsort(F, axis=0, kind='mergesort')
    let mut big_i = Array2::<usize>::zeros((n_points, n_obj));
    for j in 0..n_obj {
        let col = f.column(j);
        let mut indices: Vec<usize> = (0..n_points).collect();
        indices.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap_or(Ordering::Equal));
        for (i, &idx) in indices.iter().enumerate() {
            big_i[[i, j]] = idx;
        }
    }

    // F = F[I, np.arange(n_obj)]  — sort each column independently
    let mut sorted_f = Array2::<f64>::zeros((n_points, n_obj));
    for i in 0..n_points {
        for j in 0..n_obj {
            sorted_f[[i, j]] = f[[big_i[[i, j]], j]];
        }
    }

    // dist shape (n+1, n_obj): differences between consecutive sorted rows
    // with -inf and +inf boundaries.
    let mut dist = Array2::<f64>::zeros((n_points + 1, n_obj));
    for j in 0..n_obj {
        dist[[0, j]] = f64::INFINITY;
        for i in 1..n_points {
            dist[[i, j]] = sorted_f[[i, j]] - sorted_f[[i - 1, j]];
        }
        dist[[n_points, j]] = f64::INFINITY;
    }

    // norm = max - min per objective; 0 → NaN (so division gives NaN → replaced with 0)
    let mut norm = Array1::<f64>::zeros(n_obj);
    for j in 0..n_obj {
        let col = sorted_f.column(j);
        let max_v = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_v = col.fold(f64::INFINITY, |a, &b| a.min(b));
        let r = max_v - min_v;
        norm[j] = if r == 0.0 { f64::NAN } else { r };
    }

    // dist_to_last = dist[:-1] / norm;  dist_to_next = dist[1:] / norm
    let mut dist_to_last = dist.slice(s![..n_points, ..]).to_owned();
    let mut dist_to_next = dist.slice(s![1.., ..]).to_owned();
    for j in 0..n_obj {
        for i in 0..n_points {
            dist_to_last[[i, j]] /= norm[j];
            dist_to_next[[i, j]] /= norm[j];
        }
    }
    dist_to_last.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });
    dist_to_next.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });

    // J = np.argsort(I, axis=0)  — inverse permutation
    let mut big_j = Array2::<usize>::zeros((n_points, n_obj));
    for j in 0..n_obj {
        for i in 0..n_points {
            big_j[[big_i[[i, j]], j]] = i;
        }
    }

    // cd[k] = sum_j( dist_to_last[J[k,j], j] + dist_to_next[J[k,j], j] ) / n_obj
    let mut cd = Array1::<f64>::zeros(n_points);
    for k in 0..n_points {
        let mut sum = 0.0;
        for j in 0..n_obj {
            let rank = big_j[[k, j]];
            sum += dist_to_last[[rank, j]] + dist_to_next[[rank, j]];
        }
        cd[k] = sum / n_obj as f64;
    }

    cd
}

// -------------------------------------------------------------------------------------------------
// calc_crowding_entropy
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.operators.survival.rank_and_crowding.metrics.calc_crowding_entropy`.
pub fn calc_crowding_entropy(f: &Array2<f64>, _n_remove: usize) -> Array1<f64> {
    let (n_points, n_obj) = f.dim();

    let mut big_i = Array2::<usize>::zeros((n_points, n_obj));
    for j in 0..n_obj {
        let col = f.column(j);
        let mut indices: Vec<usize> = (0..n_points).collect();
        indices.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap_or(Ordering::Equal));
        for (i, &idx) in indices.iter().enumerate() {
            big_i[[i, j]] = idx;
        }
    }

    let mut sorted_f = Array2::<f64>::zeros((n_points, n_obj));
    for i in 0..n_points {
        for j in 0..n_obj {
            sorted_f[[i, j]] = f[[big_i[[i, j]], j]];
        }
    }

    let mut dist = Array2::<f64>::zeros((n_points + 1, n_obj));
    for j in 0..n_obj {
        dist[[0, j]] = f64::INFINITY;
        for i in 1..n_points {
            dist[[i, j]] = sorted_f[[i, j]] - sorted_f[[i - 1, j]];
        }
        dist[[n_points, j]] = f64::INFINITY;
    }

    let mut norm = Array1::<f64>::zeros(n_obj);
    for j in 0..n_obj {
        let col = sorted_f.column(j);
        let max_v = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_v = col.fold(f64::INFINITY, |a, &b| a.min(b));
        let r = max_v - min_v;
        norm[j] = if r == 0.0 { f64::NAN } else { r };
    }

    let mut dl = dist.slice(s![..n_points, ..]).to_owned();
    let mut du = dist.slice(s![1.., ..]).to_owned();
    dl.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });
    du.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });

    let cd = &dl + &du;

    // entropy shape (n_points, n_obj): inf at boundaries, -(pl*log2(pl)+pu*log2(pu)) interior
    let mut entropy = Array2::<f64>::zeros((n_points, n_obj));
    for j in 0..n_obj {
        entropy[[0, j]] = f64::INFINITY;
        entropy[[n_points - 1, j]] = f64::INFINITY;
    }
    if n_points > 2 {
        let cd_interior = cd.slice(s![1..n_points - 1, ..]).to_owned();
        let pl = dl.slice(s![1..n_points - 1, ..]).to_owned() / &cd_interior;
        let pu = du.slice(s![1..n_points - 1, ..]).to_owned() / &cd_interior;
        for i in 1..n_points - 1 {
            for j in 0..n_obj {
                let p_l = pl[[i - 1, j]];
                let p_u = pu[[i - 1, j]];
                entropy[[i, j]] = -(p_l * p_l.log2() + p_u * p_u.log2());
            }
        }
    }

    let mut big_j = Array2::<usize>::zeros((n_points, n_obj));
    for j in 0..n_obj {
        for i in 0..n_points {
            big_j[[big_i[[i, j]], j]] = i;
        }
    }

    // _cej[k] = sum_j( cd[J[k,j],j] * entropy[J[k,j],j] / norm[j] )  — NaN → 0
    let mut ce = Array1::<f64>::zeros(n_points);
    for k in 0..n_points {
        let mut sum = 0.0;
        for j in 0..n_obj {
            let rank = big_j[[k, j]];
            let v = cd[[rank, j]] * entropy[[rank, j]] / norm[j];
            sum += if v.is_nan() { 0.0 } else { v };
        }
        ce[k] = sum;
    }

    ce
}

// -------------------------------------------------------------------------------------------------
// calc_mnn_fast / calc_2nn_fast
// -------------------------------------------------------------------------------------------------

fn calc_mnn_fast_impl(f: &Array2<f64>, n_neighbors: usize) -> Array1<f64> {
    let (n_points, n_obj) = f.dim();

    // normalize: (F - min) / (max - min), with (max-min)==0 treated as 1
    let max_v = f.map_axis(Axis(0), |col| col.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    let min_v = f.map_axis(Axis(0), |col| col.fold(f64::INFINITY, |a, &b| a.min(b)));
    let norm = (&max_v - &min_v).mapv(|v| if v == 0.0 { 1.0 } else { v });
    let f_norm = (f - &min_v) / &norm;

    // pairwise squared euclidean distances — mirrors squareform(pdist(F, "sqeuclidean"))
    let mut dist_mat = Array2::<f64>::zeros((n_points, n_points));
    for i in 0..n_points {
        for j in 0..n_points {
            let d: f64 = f_norm
                .row(i)
                .iter()
                .zip(f_norm.row(j).iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            dist_mat[[i, j]] = d;
        }
    }

    // for each row: sort, skip self-distance (index 0), take n_neighbors smallest, product
    // mirrors: _D = np.partition(D, range(1, M+1), axis=1)[:, 1:M+1]; d = np.prod(_D, axis=1)
    let mut d = Array1::<f64>::ones(n_points);
    for i in 0..n_points {
        let mut row: Vec<f64> = dist_mat.row(i).to_vec();
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let take = n_neighbors.min(row.len().saturating_sub(1));
        d[i] = row[1..=take].iter().product();
    }

    // set extremes (argmin/argmax per objective) to inf
    for j in 0..n_obj {
        let col = f_norm.column(j);
        let amin = col
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let amax = col
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        d[amin] = f64::INFINITY;
        d[amax] = f64::INFINITY;
    }

    d
}

/// Mirrors `pymoo.operators.survival.rank_and_crowding.metrics.calc_mnn_fast`.
pub fn calc_mnn_fast(f: &Array2<f64>, _n_remove: usize) -> Array1<f64> {
    calc_mnn_fast_impl(f, f.ncols())
}

/// Mirrors `pymoo.operators.survival.rank_and_crowding.metrics.calc_2nn_fast`.
pub fn calc_2nn_fast(f: &Array2<f64>, _n_remove: usize) -> Array1<f64> {
    calc_mnn_fast_impl(f, 2)
}
