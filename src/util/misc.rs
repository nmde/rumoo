use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};

use anyhow::{Result, anyhow};
use ndarray::{Array, Array1, Array2, Array3, Axis, ShapeBuilder};
use rand::{rngs::StdRng, seq::SliceRandom};

use crate::{
    core::{
        individual::{IndividualField, Value},
        population::Population,
        problem::Problem,
        sampling::Sampling,
    },
    util::default_random_state,
};

/// All combinations of `items` taken `k` at a time.
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.len() < k {
        return vec![];
    }
    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
        for mut rest in combinations(&items[i + 1..], k - 1) {
            rest.insert(0, item.clone());
            result.push(rest);
        }
    }
    result
}

/// Apply parameter-less penalty to infeasible individuals.
///
/// Replaces objective values of infeasible rows (CV > 0) with `fmax + cv`.
/// When `inplace` is false a copy is made first.
/// Mirrors `pymoo.util.misc.parameter_less`.
pub fn parameter_less(f: &Array2<f64>, cv: &Array1<f64>, fmax: Option<f64>) -> Result<Array2<f64>> {
    if f.nrows() != cv.len() {
        return Err(anyhow!(
            "parameter_less: F and CV must have the same length"
        ));
    }

    let mut f_copy = f.clone();
    let fmax = fmax.unwrap_or_else(|| f.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // param_less = fmax + CV
    let param_less = cv.mapv(|v| fmax + v);

    // infeas = (CV > 0).flatten()
    for (i, &cv_i) in cv.iter().enumerate() {
        if cv_i > 0.0 {
            f_copy.row_mut(i).fill(param_less[i]);
        }
    }

    Ok(f_copy)
}

pub fn parameter_less_inplace(
    f: &mut Array2<f64>,
    cv: &Array1<f64>,
    f_max: Option<f64>,
) -> Result<()> {
    *f = parameter_less(f, cv, f_max)?;
    Ok(())
}

/// Clip the first row of `X` to the bounds `[xl, xu]`.
/// Mirrors `pymoo.util.misc.repair`.
pub fn repair(x: &Array2<f64>, xl: &Array1<f64>, xu: &Array1<f64>) -> Array2<f64> {
    let n_var = x.ncols();
    let mut local = x.clone();
    for j in 0..n_var {
        let v = local[[0, j]];
        if v > xu[j] {
            local[[0, j]] = xu[j];
        } else if v < xl[j] {
            local[[0, j]] = xl[j];
        }
    }
    local
}

/// Return the lexicographically unique rows of a 2-D array.
/// Mirrors `pymoo.util.misc.unique_rows`.
pub fn unique_rows(a: &Array2<f64>) -> Result<Array2<f64>> {
    let ncols = a.ncols();
    let mut rows: Vec<Vec<f64>> = a.outer_iter().map(|r| r.to_vec()).collect();

    // lexicographic sort
    rows.sort_by(|x, y| {
        x.iter()
            .zip(y.iter())
            .find_map(|(xi, yi)| xi.partial_cmp(yi).filter(|c| c.is_ne()))
            .unwrap_or(Ordering::Equal)
    });

    // deduplicate
    rows.dedup_by(|a, b| a == b);

    let n = rows.len();
    Ok(Array2::from_shape_vec(
        (n, ncols),
        rows.into_iter().flatten().collect(),
    )?)
}

pub fn parameter_less_constraints(
    f: &Array2<f64>,
    cv: &Array1<f64>,
    f_max: Option<f64>,
) -> Array2<f64> {
    let mut f_copy = f.clone();
    let f_max = f_max.unwrap_or_else(|| f.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    for (i, &cv_i) in cv.iter().enumerate() {
        if cv_i > 0.0 {
            f_copy.row_mut(i).fill(cv_i + f_max);
        }
    }
    f_copy
}

/// Generate `n` random permutations of `[0, l)` and concatenate them.
///
/// Equivalent to calling `random_state.permutation(l)` `n` times and
/// concatenating (the `concat=True` default).
/// Mirrors `pymoo.util.misc.random_permutations` (`@default_random_state`).
pub fn random_permutations(n: usize, l: usize, random_state: Option<&mut StdRng>) -> Array1<usize> {
    let mut local;
    let rng = random_state.unwrap_or(&mut default_random_state());

    let mut result = Vec::with_capacity(n * l);
    let mut perm: Vec<usize> = (0..l).collect();

    for _ in 0..n {
        perm.shuffle(rng);
        result.extend_from_slice(&perm);
    }

    Array1::from_vec(result)
}

/// Return groups of row indices that point to identical rows.
///
/// Each inner `Vec` is a group of indices `[i, j, ...]` where
/// `M[i] == M[j] == ...`.  Groups of size 1 are omitted.
/// Mirrors `pymoo.util.misc.get_duplicates`.
pub fn get_duplicates(m: &Array2<f64>) -> Vec<Vec<usize>> {
    let n = m.nrows();

    // I = np.lexsort(...)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        m.row(a)
            .iter()
            .zip(m.row(b).iter())
            .find_map(|(x, y)| x.partial_cmp(y).filter(|c| c.is_ne()))
            .unwrap_or(Ordering::Equal)
    });

    let rows_equal = |i: usize, j: usize| -> bool {
        m.row(indices[i])
            .iter()
            .zip(m.row(indices[j]).iter())
            .all(|(a, b)| a == b)
    };

    let mut result = Vec::new();
    let mut i = 0;

    while i + 1 < n {
        let mut group = Vec::new();
        while i + 1 < n && rows_equal(i, i + 1) {
            group.push(indices[i]);
            i += 1;
        }
        if !group.is_empty() {
            group.push(indices[i]);
            result.push(group);
        }
        i += 1;
    }

    result
}

// -----------------------------------------------
// Euclidean Distance
// -----------------------------------------------

/// Row-wise Euclidean distance: `sqrt(sum((a - b)^2, axis=1))`.
/// Mirrors `pymoo.util.misc.func_euclidean_distance`.
pub fn func_euclidean_distance(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    (a - b).mapv(|v| v * v).sum_axis(Axis(1)).mapv(f64::sqrt)
}

/// Returns a closure for normalised Euclidean distance: `sqrt(sum(((a-b)/(xu-xl))^2))`.
/// Mirrors `pymoo.util.misc.func_norm_euclidean_distance`.
pub fn func_norm_euclidean_distance(
    xl: Array1<f64>,
    xu: Array1<f64>,
) -> impl Fn(&Array2<f64>, &Array2<f64>) -> Array1<f64> {
    move |a, b| {
        let range = &xu - &xl;
        let diff = (a - b) / &range;
        diff.mapv(|v| v * v).sum_axis(Axis(1)).mapv(f64::sqrt)
    }
}

/// Normalised Euclidean pairwise distance matrix using explicit bounds.
/// Mirrors `pymoo.util.misc.norm_eucl_dist_by_bounds`.
pub fn norm_eucl_dist_by_bounds(
    a: &Array2<f64>,
    b: &Array2<f64>,
    xl: Array1<f64>,
    xu: Array1<f64>,
) -> Result<Array2<f64>> {
    vectorized_cdist(a, b, &func_norm_euclidean_distance(xl, xu), false)
}

/// Normalised Euclidean pairwise distance matrix using problem bounds.
/// Mirrors `pymoo.util.misc.norm_eucl_dist`.
pub fn norm_eucl_dist(
    problem: &dyn Problem,
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> Result<Array2<f64>> {
    let (xl, xu) = problem.bounds();
    norm_eucl_dist_by_bounds(a, b, xl, xu)
}

// -----------------------------------------------
// Manhattan Distance
// -----------------------------------------------

/// Row-wise Manhattan distance: `sum(|a - b|, axis=1)`.
/// Mirrors `pymoo.util.misc.func_manhatten_distance`.
pub fn func_manhatten_distance(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    (a - b).mapv(f64::abs).sum_axis(Axis(1))
}

/// Returns a closure for normalised Manhattan distance.
/// Mirrors `pymoo.util.misc.func_norm_manhatten_distance`.
pub fn func_norm_manhatten_distance(
    xl: Array1<f64>,
    xu: Array1<f64>,
) -> impl Fn(&Array2<f64>, &Array2<f64>) -> Array1<f64> {
    move |a, b| {
        let range = &xu - &xl;
        ((a - b) / &range).mapv(f64::abs).sum_axis(Axis(1))
    }
}

/// Normalised Manhattan pairwise distance matrix using explicit bounds.
/// Mirrors `pymoo.util.misc.norm_manhatten_dist_by_bounds`.
pub fn norm_manhatten_dist_by_bounds(
    a: &Array2<f64>,
    b: &Array2<f64>,
    xl: Array1<f64>,
    xu: Array1<f64>,
) -> Result<Array2<f64>> {
    vectorized_cdist(a, b, &func_norm_manhatten_distance(xl, xu), false)
}

/// Normalised Manhattan pairwise distance matrix using problem bounds.
/// Mirrors `pymoo.util.misc.norm_manhatten_dist`.
pub fn norm_manhatten_dist(
    problem: &dyn Problem,
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> Result<Array2<f64>> {
    let (xl, xu) = problem.bounds();
    norm_manhatten_dist_by_bounds(a, b, xl, xu)
}

// -----------------------------------------------
// Chebyshev Distance
// -----------------------------------------------

/// Row-wise Chebyshev (L∞) distance: `max(|a - b|, axis=1)`.
/// Mirrors `pymoo.util.misc.func_tchebychev_distance`.
pub fn func_tchebychev_distance(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    (a - b)
        .mapv(f64::abs)
        .fold_axis(Axis(1), f64::NEG_INFINITY, |acc, &v| acc.max(v))
}

/// Returns a closure for normalised Chebyshev distance.
/// Mirrors `pymoo.util.misc.func_norm_tchebychev_distance`.
pub fn func_norm_tchebychev_distance(
    xl: Array1<f64>,
    xu: Array1<f64>,
) -> impl Fn(&Array2<f64>, &Array2<f64>) -> Array1<f64> {
    move |a, b| {
        let range = &xu - &xl;
        ((a - b) / &range)
            .mapv(f64::abs)
            .fold_axis(Axis(1), f64::NEG_INFINITY, |acc, &v| acc.max(v))
    }
}

/// Normalised Chebyshev pairwise distance matrix using explicit bounds.
/// Mirrors `pymoo.util.misc.norm_tchebychev_dist_by_bounds`.
pub fn norm_tchebychev_dist_by_bounds(
    a: &Array2<f64>,
    b: &Array2<f64>,
    xl: Array1<f64>,
    xu: Array1<f64>,
) -> Result<Array2<f64>> {
    vectorized_cdist(a, b, &func_norm_tchebychev_distance(xl, xu), false)
}

/// Normalised Chebyshev pairwise distance matrix using problem bounds.
/// Mirrors `pymoo.util.misc.norm_tchebychev_dist`.
pub fn norm_tchebychev_dist(
    problem: &dyn Problem,
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> Result<Array2<f64>> {
    let (xl, xu) = problem.bounds();
    norm_tchebychev_dist_by_bounds(a, b, xl, xu)
}

// -----------------------------------------------
// Others
// -----------------------------------------------

/// Euclidean pairwise distance matrix (all pairs of rows).
/// Mirrors `pymoo.util.misc.cdist` (scipy wrapper in Python).
pub fn cdist(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    vectorized_cdist(a, b, &func_euclidean_distance, false)
}

/// Vectorised pairwise distance matrix via a row-distance function.
///
/// Builds the `(n_a × n_b)` cross-distance matrix by repeating / tiling
/// `A` and `B` into aligned flat arrays, calling `func_dist`, then reshaping.
/// Mirrors `pymoo.util.misc.vectorized_cdist`.
pub fn vectorized_cdist(
    a: &Array2<f64>,
    b: &Array2<f64>,
    func_dist: &dyn Fn(&Array2<f64>, &Array2<f64>) -> Array1<f64>,
    fill_diag_with_inf: bool,
) -> Result<Array2<f64>> {
    let na = a.nrows();
    let nb = b.nrows();
    let ncols = a.ncols();

    // u = np.repeat(A, B.shape[0], axis=0)  — each row of A repeated nb times
    let mut u = Array2::zeros((na * nb, ncols));
    for (i, row) in a.outer_iter().enumerate() {
        for j in 0..nb {
            u.row_mut(i * nb + j).assign(&row);
        }
    }

    // v = np.tile(B, (A.shape[0], 1))  — B repeated na times
    let mut v = Array2::zeros((na * nb, ncols));
    for i in 0..na {
        for (j, row) in b.outer_iter().enumerate() {
            v.row_mut(i * nb + j).assign(&row);
        }
    }

    let d = func_dist(&u, &v);
    let mut m = d.into_shape((na, nb))?;

    // np.fill_diagonal(M, np.inf)
    if fill_diag_with_inf {
        let diag_len = na.min(nb);
        for i in 0..diag_len {
            m[[i, i]] = f64::INFINITY;
        }
    }

    Ok(m)
}

/// Convert a decision-variable matrix to a problem's variable type.
/// Mirrors `pymoo.util.misc.covert_to_type` (note: original spelling preserved).
pub fn covert_to_type(problem: &dyn Problem, x: Array2<f64>) -> Array2<f64> {
    match problem.vtype() {
        "float" => x,
        "int" => x.mapv(|v| v.round()),
        "bool" => x.mapv(|v| if v < 0.5 { 0.0 } else { 1.0 }),
        _ => x,
    }
}

/// Return a boolean mask where `true` marks rows that are duplicates of an earlier row.
/// Mirrors `pymoo.util.misc.find_duplicates`.
pub fn find_duplicates(x: &Array2<f64>, epsilon: Option<f64>) -> Array1<bool> {
    let epsilon = epsilon.unwrap_or(1e-16);
    let d = cdist(x, x);

    let n = x.nrows();
    let mut is_duplicate = Array1::from_elem(n, false);

    // set upper triangle (including diagonal) to infinity, then check columns
    for i in 0..n {
        for j in 0..=i {
            // upper triangle indices
            if d[[i, j]] <= epsilon && i != j {
                is_duplicate[i] = true;
            }
        }
    }
    is_duplicate
}

pub enum ExtendAs {
    Row,
    Column,
}

/// Ensure a 1-D array is promoted to 2-D.
///
/// - `extend_as = "row"` (default) → shape `(1, n)`
/// - `extend_as = "column"` → shape `(n, 1)`
///
/// Returns `(array_2d, was_reshaped)`.
/// Mirrors `pymoo.util.misc.at_least_2d_array`.
pub fn at_least_2d_array(x: Array1<f64>, extend_as: Option<ExtendAs>) -> Result<Array2<f64>> {
    let extend_as = extend_as.unwrap_or(ExtendAs::Row);
    let n = x.len();
    let arr = if extend_as == ExtendAs::Row {
        x.into_shape((1, n))?
    } else {
        x.into_shape((n, 1))?
    };
    Ok(arr)
}

/// Convert a 2-D array to 1-D if one dimension has length 1.
/// Mirrors `pymoo.util.misc.to_1d_array_if_possible`.
pub fn to_1d_array_if_possible(x: Array2<f64>) -> Result<Array1<f64>> {
    let (r, c) = x.dim();
    if r == 1 {
        Ok(x.into_shape(c)?)
    } else if c == 1 {
        Ok(x.into_shape(r)?)
    } else {
        Err(anyhow!("to_1d_array_if_possible: neither dimension is 1"))
    }
}

/// Vertically stack a list of 2-D arrays (mirrors `np.vstack`).
/// Mirrors `pymoo.util.misc.stack`.
pub fn stack(arrays: &[Array2<f64>]) -> Result<Array2<f64>> {
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    Ok(ndarray::concatenate(Axis(0), &views)?)
}

/// Return all elements of `x` whose index is not in `exclude`.
/// Mirrors `pymoo.util.misc.all_except`.
pub fn all_except(x: &Array1<f64>, exclude: &[usize]) -> Array1<f64> {
    if exclude.is_empty() {
        return x.clone();
    }
    let excl: HashSet<usize> = exclude.iter().copied().collect();
    let kept: Vec<f64> = x
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if excl.contains(&i) { None } else { Some(v) })
        .collect();
    Array1::from_vec(kept)
}

/// All pairwise row combinations of two 1-D arrays stacked into columns.
/// Mirrors `pymoo.util.misc.all_combinations`.
pub fn all_combinations(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let na = a.len();
    let nb = b.len();
    let mut out = Array2::zeros((na * nb, 2));
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[[i * nb + j, 0]] = ai;
            out[[i * nb + j, 1]] = bj;
        }
    }
    out
}

/// Create a `Population` from a sampling strategy, an array, or an existing population.
/// Mirrors `pymoo.util.misc.pop_from_sampling`.
pub fn pop_from_sampling(
    problem: &dyn Problem,
    sampling: &dyn Sampling,
    n_initial_samples: usize,
    pop: Option<Population>,
) -> Option<Population> {
    let base = pop.unwrap_or_else(|| Population::empty(0));
    Some(sampling.sample(problem, n_initial_samples, Some(base)))
}

/// Evaluate any individuals in `pop` whose F has not been set.
/// Mirrors `pymoo.util.misc.evaluate_if_not_done_yet`.
pub fn evaluate_if_not_done_yet(
    evaluator: &dyn Evaluator,
    problem: &dyn Problem,
    pop: &mut Population,
    algorithm: Option<&dyn Algorithm>,
) {
    // I = np.where(pop.get("F") == None)[0]
    let unevaluated: Vec<usize> = (0..pop.len()).filter(|&i| pop[i].f.is_none()).collect();
    if !unevaluated.is_empty() {
        let sub = pop.select(&unevaluated);
        let evaluated = evaluator.process(problem, &sub, algorithm);
        for (dst, src) in unevaluated.iter().zip(evaluated.iter()) {
            pop[*dst] = src.clone();
        }
    }
}

/// Set `kwargs[key] = val` if `key` is not already present.
/// Mirrors `pymoo.util.misc.set_if_none`.
pub fn set_if_none<V: Clone>(kwargs: &mut HashMap<String, V>, key: &str, val: V) {
    kwargs.entry(key.to_string()).or_insert(val);
}

/// Set multiple key/value defaults at once.
/// Mirrors `pymoo.util.misc.set_if_none_from_tuples`.
pub fn set_if_none_from_tuples<V: Clone>(kwargs: &mut HashMap<String, V>, pairs: &[(&str, V)]) {
    for (key, val) in pairs {
        kwargs.entry(key.to_string()).or_insert_with(|| val.clone());
    }
}

/// Closest-point distances: for each row of `X`, the nearest other row's index and distance.
/// Mirrors `pymoo.util.misc.distance_of_closest_points_to_others`.
pub fn distance_of_closest_points_to_others(x: &Array2<f64>) -> (Array1<usize>, Array1<f64>) {
    let mut d = vectorized_cdist(x, x, &func_euclidean_distance, false);
    let n = d.nrows();
    for i in 0..n {
        d[[i, i]] = f64::INFINITY;
    }
    let argmin: Array1<usize> = (0..n)
        .map(|i| {
            d.row(i)
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(j, _)| j)
                .unwrap()
        })
        .collect();
    let min_dist: Array1<f64> = (0..n).map(|i| d.row(i)[argmin[i]]).collect();
    (argmin, min_dist)
}

/// Parse `"HH:MM:SS"` (or sub-formats) into total seconds.
/// Mirrors `pymoo.util.misc.time_to_int`.
pub fn time_to_int(t: &str) -> Result<u64> {
    let vals: Vec<u64> = t.split(':').filter_map(|s| s.parse()?).collect();
    let mut s = vals.last().copied().unwrap_or(0);
    if vals.len() > 1 {
        s += 60 * vals[vals.len() - 2];
    }
    if vals.len() > 2 {
        s += 3600 * vals[vals.len() - 3];
    }
    Ok(s)
}

/// Generate the powerset of a slice (all subsets, ordered by size).
/// Mirrors `pymoo.util.misc.powerset`.
pub fn powerset<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    let n = items.len();
    let mut result = Vec::new();
    for size in 0..=n {
        result.extend(combinations(items, size));
    }
    result
}

/// Return elements of `a` that also appear in `b` (preserving `a`'s order).
/// Mirrors `pymoo.util.misc.intersect`.
pub fn intersect<T: Eq + Hash + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let set: HashSet<_> = b.iter().collect();
    a.iter().filter(|e| set.contains(e)).cloned().collect()
}

/// True if any individual in `pop` is feasible (FEAS == true).
/// Mirrors `pymoo.util.misc.has_feasible`.
pub fn has_feasible(pop: &Population) -> bool {
    match pop.get(&IndividualField::Feas) {
        Value::BoolArray(feas) => feas.iter().any(|&f| f),
        _ => false,
        Value::Float(_) => todo!(),
        Value::Int(_) => todo!(),
        Value::Bool(_) => todo!(),
        Value::FloatArray(array_base) => todo!(),
        Value::IntArray(array_base) => todo!(),
        Value::FloatMatrix(array_base) => todo!(),
    }
}

/// Find unique values and all their original indices (sorted per group).
/// Mirrors `pymoo.util.misc.unique_and_all_indices`.
pub fn unique_and_all_indices(arr: &Array1<f64>) -> (Array1<f64>, Vec<Vec<usize>>) {
    // sort_indexes = np.argsort(arr)
    let mut sort_indexes: Vec<usize> = (0..arr.len()).collect();
    sort_indexes.sort_by(|&a, &b| arr[a].partial_cmp(&arr[b]).unwrap_or(Ordering::Equal));

    let mut vals: Vec<f64> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for &orig_idx in &sort_indexes {
        let v = arr[orig_idx];
        match vals.last() {
            Some(&last) if last == v => {
                groups.last_mut().unwrap().push(orig_idx);
            }
            _ => {
                vals.push(v);
                groups.push(vec![orig_idx]);
            }
        }
    }

    for group in &mut groups {
        group.sort_unstable();
    }

    (Array1::from_vec(vals), groups)
}

/// Retrieve multiple values from a map by key.
/// Mirrors `pymoo.util.misc.from_dict`.
pub fn from_dict<'a, V>(d: &'a HashMap<String, V>, keys: &[&str]) -> Vec<Option<&'a V>> {
    keys.iter().map(|k| d.get(*k)).collect()
}

/// Group list-of-dicts values by key, returning unique values in insertion order.
/// Mirrors `pymoo.util.misc.list_of_dicts_unique`.
pub fn list_of_dicts_unique<'a, V: Debug>(l: &'a [HashMap<String, V>], k: &str) -> Vec<&'a V> {
    let mut seen: Vec<String> = Vec::new();
    let mut result = Vec::new();
    for d in l {
        if let Some(v) = d.get(k) {
            let repr = format!("{:?}", v);
            if !seen.contains(&repr) {
                seen.push(repr);
                result.push(v);
            }
        }
    }
    result
}

/// Filter a list of dicts, keeping only entries where all `(key, val)` pairs match.
/// Mirrors `pymoo.util.misc.list_of_dicts_filter`.
pub fn list_of_dicts_filter<'a, V: PartialEq>(
    l: &'a [HashMap<String, V>],
    pairs: &[(&str, &V)],
) -> Vec<&'a HashMap<String, V>> {
    l.iter()
        .filter(|d| pairs.iter().all(|(k, v)| d.get(*k) == Some(v)))
        .collect()
}

/// Apply a binary `func` cumulatively to `a`, `b`, and all `rest`.
/// Mirrors `pymoo.util.misc.logical_op`.
pub fn logical_op<T: Clone>(func: impl Fn(&T, &T) -> T, a: &T, b: &T, rest: &[T]) -> T {
    let mut ret = func(a, b);
    for c in rest {
        ret = func(&ret, c);
    }
    ret
}

/// Replace NaN values in an array.
/// Mirrors `pymoo.util.misc.replace_nan_by`.
pub fn replace_nan_by<S: ShapeBuilder>(x: &Array<f64, S>, val: f64) -> Array1<f64> {
    x.mapv(|v| if v.is_nan() { val } else { v })
}

/// Replace NaN values in a 2-D array in-place.
pub fn replace_nan_by_inplace<S: ShapeBuilder>(x: &mut Array<f64, S>, val: f64) {
    x.mapv_inplace(|v| if v.is_nan() { val } else { v });
}

/// Set defaults in `kwargs`, optionally overwriting existing keys.
/// Mirrors `pymoo.util.misc.set_defaults`.
pub fn set_defaults<V: Clone>(
    dict: &mut HashMap<String, V>,
    defaults: &HashMap<String, V>,
    overwrite: Option<bool>,
    func_get: &Option<impl Fn(&V) -> V>,
) {
    let overwrite = overwrite.unwrap_or(false);
    for (k, v) in defaults {
        if overwrite || !dict.contains_key(k) {
            let val = match func_get {
                None => v.clone(),
                _ => (func_get.unwrap())(v),
            };
            dict.insert(k.clone(), val);
        }
    }
}

/// Filter a map by key prefix, optionally stripping the prefix.
/// Mirrors `pymoo.util.misc.filter_params`.
pub fn filter_params<V: Clone>(
    params: &HashMap<String, V>,
    prefix: &str,
    delete_prefix: Option<bool>,
) -> HashMap<String, V> {
    let delete_prefix = delete_prefix.unwrap_or(true);
    params
        .iter()
        .filter_map(|(k, v)| {
            if k.starts_with(prefix) {
                let new_key = if delete_prefix {
                    k[prefix.len()..].to_string()
                } else {
                    k.clone()
                };
                Some((new_key, v.clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Map each distinct value in `x` to all indices where it occurs.
/// Mirrors `pymoo.util.misc.where_is_what`.
pub fn where_is_what<T: Eq + Hash + Clone>(x: &[T]) -> HashMap<T, Vec<usize>> {
    let mut h: HashMap<T, Vec<usize>> = HashMap::new();
    for (k, e) in x.iter().enumerate() {
        h.entry(e.clone()).or_default().push(k);
    }
    h
}

/// Apply a crossover mask: swap elements between parents where the mask is `true`.
///
/// `x` has shape `(2, n_ind, n_var)`; `m` has shape `(n_ind, n_var)`.
/// Mirrors `pymoo.util.misc.crossover_mask`.
pub fn crossover_mask(x: Array3<f64>, m: &Array2<bool>) -> Array3<f64> {
    let mut local = x.clone();
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if m[[i, j]] {
                let tmp = local[[0, i, j]];
                local[[0, i, j]] = local[[1, i, j]];
                local[[1, i, j]] = tmp;
            }
        }
    }
    local
}

/// Ensure every row of a boolean matrix has at least one `true`.
/// Rows that are all-false get a random column set to `true`.
///
/// Mirrors `pymoo.util.misc.row_at_least_once_true` (including `@default_random_state`).
pub fn row_at_least_once_true(m: &Array2<bool>, random_state: Option<&mut StdRng>) -> Array2<bool> {
    let (_, n_cols) = m.dim();
    let mut local = m.clone();
    let rng = random_state.unwrap_or(&mut default_random_state());
    for k in 0..local.nrows() {
        if !local.row(k).iter().any(|&v| v) {
            // random_state.integers(d)
            let col = rng.gen_range(0..n_cols);
            local[[k, col]] = true;
        }
    }
    local
}
