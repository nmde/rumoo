use std::cmp::Ordering;

use ndarray::{Array1, Array2, s};

// -------------------------------------------------------------------------------------------------
// calc_pcd
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.functions.pruning_cd.calc_pcd`.
pub fn calc_pcd(x: &Array2<f64>, n_remove: usize) -> Array1<f64> {
    let n = x.nrows();
    let m = x.ncols();

    // Mirrors: if n_remove <= (N - M): clamp to 0; else n_remove = N - M
    let n_remove = n_remove.min(n.saturating_sub(m));

    // extremes_min = np.argmin(X, axis=0)
    let extremes_min: Vec<usize> = (0..m)
        .map(|j| {
            x.column(j)
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();

    // extremes_max = np.argmax(X, axis=0)
    let extremes_max: Vec<usize> = (0..m)
        .map(|j| {
            x.column(j)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();

    let min_vals: Vec<f64> = (0..m)
        .map(|j| x.column(j).fold(f64::INFINITY, |acc, &v| acc.min(v)))
        .collect();
    let max_vals: Vec<f64> = (0..m)
        .map(|j| x.column(j).fold(f64::NEG_INFINITY, |acc, &v| acc.max(v)))
        .collect();

    // extremes = np.concatenate((extremes_min, extremes_max))
    let extremes: Vec<usize> = extremes_min
        .iter()
        .chain(extremes_max.iter())
        .copied()
        .collect();

    // X = (X - min_vals) / (max_vals - min_vals)
    let mut x_norm = x.clone();
    for j in 0..m {
        let range = max_vals[j] - min_vals[j];
        for i in 0..n {
            x_norm[[i, j]] = (x_norm[[i, j]] - min_vals[j]) / range;
        }
    }

    // H = np.arange(N)
    let mut h: Vec<usize> = (0..n).collect();

    // d = np.full(N, np.inf)
    let mut d = Array1::<f64>::from_elem(n, f64::INFINITY);

    // Compute initial crowding distances and assign d[H] = _d
    let cd = crowding_distances_for_active(&x_norm, &h, m);
    for (idx, &hi) in h.iter().enumerate() {
        d[hi] = cd[idx];
    }
    for &e in &extremes {
        d[e] = f64::INFINITY;
    }

    let mut n_removed = 0usize;

    // Mirrors: while n_removed < (n_remove - 1)
    while n_removed < n_remove.saturating_sub(1) {
        // _k = np.argmin(d[H]); k = H[_k]
        let (_, &k) = h
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| d[a].partial_cmp(&d[b]).unwrap_or(Ordering::Equal))
            .unwrap();

        // H = H[H != k]
        h.retain(|&v| v != k);
        n_removed += 1;

        // Recompute crowding distances for remaining active set
        let cd = crowding_distances_for_active(&x_norm, &h, m);
        for (idx, &hi) in h.iter().enumerate() {
            d[hi] = cd[idx];
        }
        for &e in &extremes {
            d[e] = f64::INFINITY;
        }
    }

    d
}

// -------------------------------------------------------------------------------------------------
// Helper: crowding distances for a subset of rows
// -------------------------------------------------------------------------------------------------

/// Computes crowding distances for the rows of `x_norm` indexed by `h`.
/// Mirrors the inline computation inside `calc_pcd`.
fn crowding_distances_for_active(x_norm: &Array2<f64>, h: &[usize], m: usize) -> Array1<f64> {
    let nh = h.len();
    if nh == 0 {
        return Array1::zeros(0);
    }

    // X[H] — extract active rows
    let x_h = Array2::from_shape_fn((nh, m), |(i, j)| x_norm[[h[i], j]]);

    // I = np.argsort(X[H], axis=0, kind='mergesort') — stable column-wise argsort
    let mut big_i = Array2::<usize>::zeros((nh, m));
    for j in 0..m {
        let mut order: Vec<usize> = (0..nh).collect();
        order.sort_by(|&a, &b| {
            x_h[[a, j]]
                .partial_cmp(&x_h[[b, j]])
                .unwrap_or(Ordering::Equal)
        });
        for (rank, &orig) in order.iter().enumerate() {
            big_i[[rank, j]] = orig;
        }
    }

    // _X = X[H][I, np.arange(M)] — sorted values per column
    let x_sorted = Array2::from_shape_fn((nh, m), |(i, j)| x_h[[big_i[[i, j]], j]]);

    // dist = vstack([_X, inf_row]) - vstack([-inf_row, _X])  — shape (nh+1, m)
    let mut dist = Array2::<f64>::zeros((nh + 1, m));
    for j in 0..m {
        dist[[0, j]] = x_sorted[[0, j]] - f64::NEG_INFINITY;
        for i in 1..nh {
            dist[[i, j]] = x_sorted[[i, j]] - x_sorted[[i - 1, j]];
        }
        dist[[nh, j]] = f64::INFINITY - x_sorted[[nh - 1, j]];
    }

    // dist_to_last = dist[:-1],  dist_to_next = dist[1:]
    let mut dist_to_last = dist.slice(s![..nh, ..]).to_owned();
    let mut dist_to_next = dist.slice(s![1.., ..]).to_owned();

    // Replace NaN with 0.0 (occurs when normalization gives 0/0 for constant columns)
    dist_to_last.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });
    dist_to_next.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });

    // J = np.argsort(I, axis=0) — inverse permutation: J[I[rank,j], j] = rank
    let mut big_j = Array2::<usize>::zeros((nh, m));
    for j in 0..m {
        for rank in 0..nh {
            big_j[[big_i[[rank, j]], j]] = rank;
        }
    }

    // _d[i] = sum_j( dist_to_last[J[i,j], j] + dist_to_next[J[i,j], j] )
    Array1::from_shape_fn(nh, |i| {
        (0..m)
            .map(|j| {
                let r = big_j[[i, j]];
                dist_to_last[[r, j]] + dist_to_next[[r, j]]
            })
            .sum::<f64>()
    })
}
