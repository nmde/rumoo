use ndarray::{Array1, Array2, ArrayView1, Axis, concatenate, s};
use std::path::Path;
use std::io::{self, BufRead};
use std::fs::File;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Normal};

pub mod algorithm;
pub mod callback;
pub mod duplicate;
pub mod initialization;
pub mod evaluator;
pub mod crossover;
pub mod individual;
pub mod mating;
pub mod mutation;
pub mod operator;
pub mod population;
pub mod problem;
pub mod repair;
pub mod result;
pub mod sampling;
pub mod selection;
pub mod survival;
pub mod termination;

// ============================================================================
// Error types
// ============================================================================

#[derive(Debug)]
pub enum ReadDatasetsError {
    ReadInputFileEmpty,
    ReadInputWrongInitialDim,
    ErrorFopen,
    ErrorConversion,
    ErrorColumns,
}

impl std::fmt::Display for ReadDatasetsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::ReadInputFileEmpty => "READ_INPUT_FILE_EMPTY",
            Self::ReadInputWrongInitialDim => "READ_INPUT_WRONG_INITIAL_DIM",
            Self::ErrorFopen => "ERROR_FOPEN",
            Self::ErrorConversion => "ERROR_CONVERSION",
            Self::ErrorColumns => "ERROR_COLUMNS",
        };
        write!(f, "{}", s)
    }
}

impl std::error::Error for ReadDatasetsError {}

// ============================================================================
// Maximise specification (mirrors Python's `bool | Sequence[bool]`)
// ============================================================================

#[derive(Clone, Debug)]
pub enum MaximiseSpec {
    All(bool),
    PerObj(Array1<bool>),
}

impl Default for MaximiseSpec {
    fn default() -> Self {
        MaximiseSpec::All(false)
    }
}

impl From<bool> for MaximiseSpec {
    fn from(v: bool) -> Self {
        MaximiseSpec::All(v)
    }
}

impl From<Array1<bool>> for MaximiseSpec {
    fn from(arr: Array1<bool>) -> Self {
        MaximiseSpec::PerObj(arr)
    }
}

fn parse_maximise(spec: &MaximiseSpec, nobj: usize) -> Array1<bool> {
    match spec {
        MaximiseSpec::All(v) => Array1::from_elem(nobj, *v),
        MaximiseSpec::PerObj(arr) => {
            assert_eq!(arr.len(), nobj, "maximise length must equal nobj");
            arr.clone()
        }
    }
}

fn apply_maximise_transform(data: &mut Array2<f64>, maximise: &Array1<bool>) {
    for (j, &max) in maximise.iter().enumerate() {
        if max {
            data.column_mut(j).mapv_inplace(|v| -v);
        }
    }
}

fn all_positive(x: &Array2<f64>) -> bool {
    x.iter().all(|&v| v > 0.0)
}

fn unique_rows(data: &Array2<f64>) -> Array2<f64> {
    let ncols = data.ncols();
    let mut rows: Vec<Vec<u64>> = data
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|&v| v.to_bits()).collect())
        .collect();
    rows.sort();
    rows.dedup();
    let nrows = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().map(f64::from_bits).collect();
    Array2::from_shape_vec((nrows, ncols), flat).unwrap()
}

// ============================================================================
// Dataset I/O
// ============================================================================

/// Read an input dataset file. Lines are one point each; blank lines separate sets.
/// Returns an array whose last column is the set number (1-indexed).
pub fn read_datasets(filename: &Path) -> Result<Array2<f64>, anyhow::Error> {
    if !filename.exists() {
        anyhow::bail!("file '{}' not found", filename.display());
    }

    let file = File::open(filename)
        .map_err(|_| ReadDatasetsError::ErrorFopen)?;
    let reader = io::BufReader::new(file);

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut set_id = 1usize;
    let mut last_was_empty = false;
    let mut ncols: Option<usize> = None;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !last_was_empty && !rows.is_empty() {
                set_id += 1;
            }
            last_was_empty = true;
            continue;
        }
        last_was_empty = false;

        let values: Result<Vec<f64>, _> = trimmed
            .split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect();
        let mut row = values.map_err(|_| ReadDatasetsError::ErrorConversion)?;

        match ncols {
            Some(n) if row.len() != n => anyhow::bail!(ReadDatasetsError::ErrorColumns),
            None => {
                if row.is_empty() {
                    anyhow::bail!(ReadDatasetsError::ReadInputFileEmpty);
                }
                ncols = Some(row.len());
            }
            _ => {}
        }
        row.push(set_id as f64);
        rows.push(row);
    }

    if rows.is_empty() {
        anyhow::bail!(ReadDatasetsError::ReadInputFileEmpty);
    }

    let ncols = ncols.unwrap() + 1;
    let nrows = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((nrows, ncols), flat)?)
}

// ============================================================================
// IGD / IGD+ / Average Hausdorff distance
// ============================================================================

/// Inverted Generational Distance (IGD).
pub fn igd(data: &Array2<f64>, ref_set: &Array2<f64>, maximise: impl Into<MaximiseSpec>) -> f64 {
    assert_eq!(
        data.ncols(), ref_set.ncols(),
        "data and ref need to have the same number of columns ({} != {})",
        data.ncols(), ref_set.ncols()
    );
    let _maximise_arr = parse_maximise(&maximise.into(), data.ncols());
    todo!("IGD computation not yet implemented")
}

/// Modified IGD (IGD+).
pub fn igd_plus(data: &Array2<f64>, ref_set: &Array2<f64>, maximise: impl Into<MaximiseSpec>) -> f64 {
    assert_eq!(
        data.ncols(), ref_set.ncols(),
        "data and ref need to have the same number of columns ({} != {})",
        data.ncols(), ref_set.ncols()
    );
    let _maximise_arr = parse_maximise(&maximise.into(), data.ncols());
    todo!("IGD+ computation not yet implemented")
}

/// Average Hausdorff distance. `p` must be greater than 0.
pub fn avg_hausdorff_dist(
    data: &Array2<f64>,
    ref_set: &Array2<f64>,
    maximise: impl Into<MaximiseSpec>,
    p: f64,
) -> f64 {
    assert!(p > 0.0, "'p' must be larger than zero");
    assert_eq!(
        data.ncols(), ref_set.ncols(),
        "data and ref need to have the same number of columns ({} != {})",
        data.ncols(), ref_set.ncols()
    );
    let _maximise_arr = parse_maximise(&maximise.into(), data.ncols());
    todo!("avg_hausdorff_dist not yet implemented")
}

// ============================================================================
// Epsilon metrics
// ============================================================================

/// Additive epsilon metric.
pub fn epsilon_additive(
    data: &Array2<f64>,
    ref_set: &Array2<f64>,
    maximise: impl Into<MaximiseSpec>,
) -> f64 {
    assert_eq!(
        data.ncols(), ref_set.ncols(),
        "data and ref need to have the same number of columns ({} != {})",
        data.ncols(), ref_set.ncols()
    );
    let _maximise_arr = parse_maximise(&maximise.into(), data.ncols());
    todo!("epsilon_additive not yet implemented")
}

/// Multiplicative epsilon metric. All values in `data` and `ref_set` must be > 0.
pub fn epsilon_mult(
    data: &Array2<f64>,
    ref_set: &Array2<f64>,
    maximise: impl Into<MaximiseSpec>,
) -> f64 {
    assert!(all_positive(data), "All values must be larger than 0 in the input data");
    assert!(all_positive(ref_set), "All values must be larger than 0 in the reference set");
    assert_eq!(
        data.ncols(), ref_set.ncols(),
        "data and ref need to have the same number of columns ({} != {})",
        data.ncols(), ref_set.ncols()
    );
    let _maximise_arr = parse_maximise(&maximise.into(), data.ncols());
    todo!("epsilon_mult not yet implemented")
}

// ============================================================================
// Hypervolume
// ============================================================================

fn _hypervolume(data: &Array2<f64>, ref_point: &Array1<f64>) -> f64 {
    todo!("hypervolume computation not yet implemented")
}

/// Compute the hypervolume indicator with respect to `ref_point`.
pub fn hypervolume(
    data: &Array2<f64>,
    ref_point: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
) -> f64 {
    let nobj = data.ncols();
    assert!(nobj > 0, "input data must have at least 1 column");
    assert_eq!(ref_point.len(), nobj, "ref must be same length as a single point in data");

    let maximise_arr = parse_maximise(&maximise.into(), nobj);
    let mut data = data.to_owned();
    let mut ref_point = ref_point.to_owned();

    if maximise_arr.iter().any(|&v| v) {
        apply_maximise_transform(&mut data, &maximise_arr);
        for (j, &max) in maximise_arr.iter().enumerate() {
            if max {
                ref_point[j] = -ref_point[j];
            }
        }
    }
    _hypervolume(&data, &ref_point)
}

/// Object-oriented interface for the hypervolume indicator.
pub struct Hypervolume {
    ref_point: Array1<f64>,
    maximise: Array1<bool>,
    nobj: usize,
}

impl Hypervolume {
    pub fn new(ref_point: Array1<f64>, maximise: impl Into<MaximiseSpec>) -> Self {
        let n = ref_point.len();
        let maximise_arr = parse_maximise(&maximise.into(), n);
        let mut ref_point = ref_point;
        for (j, &max) in maximise_arr.iter().enumerate() {
            if max {
                ref_point[j] = -ref_point[j];
            }
        }
        Self { ref_point, maximise: maximise_arr, nobj: n }
    }

    pub fn compute(&self, data: &Array2<f64>) -> f64 {
        assert_eq!(
            data.ncols(), self.nobj,
            "data and ref need to have the same number of objectives ({} != {})",
            data.ncols(), self.nobj
        );
        let mut data = data.to_owned();
        if self.maximise.iter().any(|&v| v) {
            apply_maximise_transform(&mut data, &self.maximise);
        }
        _hypervolume(&data, &self.ref_point)
    }
}

/// Computes hypervolume of fronts relative to the hypervolume of a reference front.
/// Value is `1 - hyp(X) / hyp(R)`, where lower is better.
pub struct RelativeHypervolume {
    inner: Hypervolume,
    ref_set_hv: f64,
}

impl RelativeHypervolume {
    pub fn new(
        ref_point: Array1<f64>,
        ref_set: &Array2<f64>,
        maximise: impl Into<MaximiseSpec>,
    ) -> Self {
        let inner = Hypervolume::new(ref_point, maximise);
        let ref_set_hv = inner.compute(ref_set);
        assert!(ref_set_hv != 0.0, "hypervolume of 'ref_set' is zero");
        Self { inner, ref_set_hv }
    }

    /// Compute `1 - hypervolume(data, ref) / hypervolume(ref_set, ref)`.
    pub fn compute(&self, data: &Array2<f64>) -> f64 {
        1.0 - self.inner.compute(data) / self.ref_set_hv
    }
}

// ============================================================================
// Hypervolume contributions
// ============================================================================

/// Hypervolume contributions of each point. Dominated and duplicated points have zero contribution.
pub fn hv_contributions(
    x: &Array2<f64>,
    ref_point: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
    _ignore_dominated: bool,
) -> Array1<f64> {
    let nobj = x.ncols();
    assert_eq!(ref_point.len(), nobj);
    let maximise_arr = parse_maximise(&maximise.into(), nobj);
    let mut x = x.to_owned();
    let mut ref_point = ref_point.to_owned();
    if maximise_arr.iter().any(|&v| v) {
        apply_maximise_transform(&mut x, &maximise_arr);
        for (j, &max) in maximise_arr.iter().enumerate() {
            if max {
                ref_point[j] = -ref_point[j];
            }
        }
    }
    todo!("hv_contributions not yet implemented")
}

// ============================================================================
// Approximate hypervolume
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HvApproxMethod {
    RphiFwePlus,
    Dz2019Hw,
    Dz2019Mc,
}

/// Approximate the hypervolume indicator.
pub fn hv_approx(
    data: &Array2<f64>,
    ref_point: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
    nsamples: u32,
    seed: Option<u64>,
    method: HvApproxMethod,
) -> f64 {
    let nobj = data.ncols();
    assert_eq!(ref_point.len(), nobj);
    let _maximise_arr = parse_maximise(&maximise.into(), nobj);
    todo!("hv_approx not yet implemented")
}

// ============================================================================
// Nondominated set generation
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NdSetMethod {
    Simplex,
    ConcaveSphere,
    ConvexSphere,
    ConvexSimplex,
    InvertedSimplex,
    ConcaveSimplex,
}

impl std::str::FromStr for NdSetMethod {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "simplex" | "linear" | "L" => Ok(Self::Simplex),
            "concave-sphere" | "sphere" | "C" => Ok(Self::ConcaveSphere),
            "convex-sphere" | "X" => Ok(Self::ConvexSphere),
            "convex-simplex" => Ok(Self::ConvexSimplex),
            "inverted-simplex" | "inverted-linear" => Ok(Self::InvertedSimplex),
            "concave-simplex" => Ok(Self::ConcaveSimplex),
            _ => Err(format!("unknown method={}", s)),
        }
    }
}

/// Generate a random set of `n` mutually nondominated points of dimension `d`.
pub fn generate_ndset(
    n: usize,
    d: usize,
    method: NdSetMethod,
    seed: Option<u64>,
    integer: bool,
) -> Array2<f64> {
    let mut rng: rand::rngs::SmallRng = match seed {
        Some(s) => rand::rngs::SmallRng::seed_from_u64(s),
        None => rand::rngs::SmallRng::from_os_rng(),
    };

    let sample_simplex = |rng: &mut rand::rngs::SmallRng| -> Array2<f64> {
        let exp = Exp::new(1.0f64).unwrap();
        let data: Vec<f64> = (0..n * d).map(|_| exp.sample(rng)).collect();
        let mut arr = Array2::from_shape_vec((n, d), data).unwrap();
        for mut row in arr.rows_mut() {
            let sum: f64 = row.sum();
            row /= sum;
        }
        arr
    };

    let sample_sphere = |rng: &mut rand::rngs::SmallRng| -> Array2<f64> {
        let normal = Normal::new(0.0f64, 1.0f64).unwrap();
        let data: Vec<f64> = (0..n * d).map(|_| normal.sample(rng).abs()).collect();
        let mut arr = Array2::from_shape_vec((n, d), data).unwrap();
        for mut row in arr.rows_mut() {
            let norm = row.dot(&row).sqrt();
            row /= norm;
        }
        arr
    };

    loop {
        let x = match method {
            NdSetMethod::Simplex => sample_simplex(&mut rng),
            NdSetMethod::ConcaveSphere => sample_sphere(&mut rng),
            NdSetMethod::ConvexSphere => sample_sphere(&mut rng).mapv(|v| 1.0 - v),
            NdSetMethod::ConvexSimplex => sample_simplex(&mut rng).mapv(|v| v * v),
            NdSetMethod::InvertedSimplex => sample_simplex(&mut rng).mapv(|v| 1.0 - v),
            NdSetMethod::ConcaveSimplex => {
                let s = sample_simplex(&mut rng);
                s.mapv(|v| 1.0 - v * v)
            }
        };

        if any_dominated(&x, false, false) {
            continue;
        }

        if !integer {
            return x;
        }

        let mut x = x * 64.0;
        loop {
            let y = x.mapv(|v| v as f64);
            if !any_dominated(&y, false, false) {
                return y;
            }
            x *= 2.0;
        }
    }
}

// ============================================================================
// Dominance
// ============================================================================

fn point_dominates(a: ArrayView1<f64>, b: ArrayView1<f64>) -> bool {
    let mut strictly_better = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Identify dominated points. Returns a bool array where `true` means nondominated.
pub fn is_nondominated(
    data: &Array2<f64>,
    maximise: impl Into<MaximiseSpec>,
    keep_weakly: bool,
) -> Array1<bool> {
    let nrows = data.nrows();
    let nobj = data.ncols();

    if nrows == 0 {
        return Array1::from_elem(0, false);
    }

    let maximise_arr = parse_maximise(&maximise.into(), nobj);
    let mut data = data.to_owned();
    if maximise_arr.iter().any(|&v| v) {
        apply_maximise_transform(&mut data, &maximise_arr);
    }

    let mut nondom = Array1::from_elem(nrows, true);

    for i in 0..nrows {
        if !nondom[i] {
            continue;
        }
        for j in 0..nrows {
            if i == j || !nondom[j] {
                continue;
            }
            if point_dominates(data.row(j), data.row(i)) {
                nondom[i] = false;
                break;
            }
            if !keep_weakly && j < i && data.row(j) == data.row(i) {
                nondom[i] = false;
                break;
            }
        }
    }
    nondom
}

/// Returns `true` if `data` contains at least one (weakly-)dominated point.
pub fn any_dominated(
    data: &Array2<f64>,
    maximise: impl Into<MaximiseSpec>,
    keep_weakly: bool,
) -> bool {
    let nrows = data.nrows();
    assert!(nrows > 0, "no points in the input data");
    if nrows == 1 {
        return false;
    }
    if data.ncols() == 1 {
        return true;
    }

    if keep_weakly {
        let deduped = unique_rows(data);
        return !is_nondominated(&deduped, maximise, false).iter().all(|&v| v);
    }
    !is_nondominated(data, maximise, keep_weakly).iter().all(|&v| v)
}

/// Identify dominated points within each set.
pub fn is_nondominated_within_sets(
    data: &Array2<f64>,
    sets: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
    keep_weakly: bool,
) -> Array1<bool> {
    assert!(data.ncols() >= 2, "'data' must have at least 2 columns (2 objectives)");
    assert_eq!(data.nrows(), sets.len());

    let maximise = maximise.into();
    let mut unique_ids: Vec<f64> = sets.to_vec();
    unique_ids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_ids.dedup();

    let mut result = Array1::from_elem(data.nrows(), false);

    for &set_id in &unique_ids {
        let indices: Vec<usize> = sets
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s == set_id { Some(i) } else { None })
            .collect();
        let subset = data.select(Axis(0), &indices);
        let nondom = is_nondominated(&subset, maximise.clone(), keep_weakly);
        for (k, &orig_idx) in indices.iter().enumerate() {
            result[orig_idx] = nondom[k];
        }
    }
    result
}

/// Remove dominated points according to Pareto optimality.
pub fn filter_dominated(
    data: &Array2<f64>,
    maximise: impl Into<MaximiseSpec>,
    keep_weakly: bool,
) -> Array2<f64> {
    let nondom = is_nondominated(data, maximise, keep_weakly);
    let indices: Vec<usize> = nondom
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None })
        .collect();
    data.select(Axis(0), &indices)
}

/// Given a dataset where the last column is the set index, filter dominated points within each set.
pub fn filter_dominated_within_sets(
    data: &Array2<f64>,
    maximise: impl Into<MaximiseSpec>,
    keep_weakly: bool,
) -> Array2<f64> {
    assert!(
        data.ncols() >= 3,
        "'data' must have at least 3 columns (2 objectives + set column)"
    );
    let ncols = data.ncols();
    let objectives = data.slice(s![.., ..ncols - 1]).to_owned();
    let sets = data.column(ncols - 1).to_owned();

    let is_nondom = is_nondominated_within_sets(&objectives, &sets, maximise, keep_weakly);
    let indices: Vec<usize> = is_nondom
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None })
        .collect();
    data.select(Axis(0), &indices)
}

// ============================================================================
// Pareto ranking (nondominated sorting)
// ============================================================================

/// Rank points by Pareto dominance. Rank 0 = not dominated by any other point.
pub fn pareto_rank(data: &Array2<f64>, maximise: impl Into<MaximiseSpec>) -> Array1<i32> {
    let nrows = data.nrows();
    let nobj = data.ncols();
    let maximise_arr = parse_maximise(&maximise.into(), nobj);

    let mut data = data.to_owned();
    if maximise_arr.iter().any(|&v| v) {
        apply_maximise_transform(&mut data, &maximise_arr);
    }

    let mut ranks = Array1::from_elem(nrows, 0i32);
    let mut remaining: Vec<usize> = (0..nrows).collect();
    let mut current_rank = 0i32;

    while !remaining.is_empty() {
        let front: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&i| {
                !remaining
                    .iter()
                    .any(|&j| j != i && point_dominates(data.row(j), data.row(i)))
            })
            .collect();

        for &i in &front {
            ranks[i] = current_rank;
        }
        remaining.retain(|i| !front.contains(i));
        current_rank += 1;
    }
    ranks
}

// ============================================================================
// Normalisation
// ============================================================================

/// Compute per-column ideal (min or max depending on `maximise`).
pub fn get_ideal(x: &Array2<f64>, maximise: impl Into<MaximiseSpec>) -> Array1<f64> {
    let nobj = x.ncols();
    let maximise_arr = parse_maximise(&maximise.into(), nobj);
    let lower = x.map_axis(Axis(0), |col| col.iter().cloned().fold(f64::INFINITY, f64::min));
    let upper = x.map_axis(Axis(0), |col| col.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    Array1::from_shape_fn(nobj, |j| if maximise_arr[j] { upper[j] } else { lower[j] })
}

/// Normalise points per coordinate to `to_range`. Default `to_range` is `(0.0, 1.0)`.
/// `lower` / `upper`: per-column bounds; pass `None` to use the data's min/max.
pub fn normalise(
    data: &Array2<f64>,
    to_range: (f64, f64),
    lower: Option<Array1<f64>>,
    upper: Option<Array1<f64>>,
    maximise: impl Into<MaximiseSpec>,
) -> Array2<f64> {
    let npoints = data.nrows();
    let nobj = data.ncols();
    assert!(nobj >= 2, "'data' must have at least two columns");

    let (range_min, range_max) = to_range;
    let maximise_arr = parse_maximise(&maximise.into(), nobj);

    let lower = lower.unwrap_or_else(|| {
        data.map_axis(Axis(0), |col| col.iter().cloned().fold(f64::INFINITY, f64::min))
    });
    let upper = upper.unwrap_or_else(|| {
        data.map_axis(Axis(0), |col| col.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    });

    let mut result = data.to_owned();
    for j in 0..nobj {
        let lo = lower[j];
        let hi = upper[j];
        let span = hi - lo;
        let scale = range_max - range_min;
        for i in 0..npoints {
            result[[i, j]] = if span == 0.0 {
                range_min
            } else if maximise_arr[j] {
                range_max - (result[[i, j]] - lo) / span * scale
            } else {
                range_min + (result[[i, j]] - lo) / span * scale
            };
        }
    }
    result
}

// ============================================================================
// EAF (Empirical Attainment Function)
// ============================================================================

/// Exact computation of the Empirical Attainment Function (EAF).
/// Returns an array with the same columns as `data` plus a percentile column.
pub fn eaf(data: &Array2<f64>, sets: &Array1<f64>, percentiles: &[f64]) -> Array2<f64> {
    let ncols = data.ncols();
    assert!(ncols >= 2, "'data' must have at least 2 columns");
    assert!(
        ncols <= 3,
        "Only 2D or 3D datasets are currently supported for computing the EAF"
    );
    assert_eq!(
        sets.len(), data.nrows(),
        "'sets' must have the same length as the number of rows of 'data'"
    );
    todo!("eaf not yet implemented")
}

// ============================================================================
// Vorob'ev threshold and expectation
// ============================================================================

pub struct VorobResult {
    pub threshold: f64,
    pub ve: Array2<f64>,
    pub avg_hyp: f64,
}

/// Compute Vorob'ev threshold and expectation.
pub fn vorob_t(data: &Array2<f64>, sets: &Array1<f64>, ref_point: &Array1<f64>) -> VorobResult {
    let nobj = data.ncols();
    assert!(nobj >= 2, "'data' must have at least 2 columns");

    let hv_ind = Hypervolume::new(ref_point.to_owned(), false);

    let avg_hyp = {
        let hvs = apply_within_sets(data, sets, |g| hv_ind.compute(g));
        hvs.iter().sum::<f64>() / hvs.len() as f64
    };

    let mut prev_hyp = f64::INFINITY;
    let mut a = 0.0f64;
    let mut b = 100.0f64;
    let mut c = 0.0f64;
    let mut eaf_res: Array2<f64> = Array2::zeros((0, nobj));

    loop {
        c = (a + b) / 2.0;
        let full = eaf(data, sets, &[c]);
        eaf_res = full.slice(s![.., ..nobj]).to_owned();
        let tmp = hv_ind.compute(&eaf_res);
        if tmp > avg_hyp {
            a = c;
        } else {
            b = c;
        }
        let diff = prev_hyp - tmp;
        prev_hyp = tmp;
        if diff == 0.0 {
            break;
        }
    }

    VorobResult { threshold: c, ve: eaf_res, avg_hyp }
}

/// Compute Vorob'ev deviation.
pub fn vorob_dev(
    data: &Array2<f64>,
    sets: &Array1<f64>,
    ref_point: &Array1<f64>,
    ve: Option<&Array2<f64>>,
) -> f64 {
    let nobj = data.ncols();
    assert!(nobj >= 2, "'data' must have at least 2 columns");

    let hv_ind = Hypervolume::new(ref_point.to_owned(), false);

    let (ve_owned, h1) = match ve {
        Some(v) => {
            let h1 = apply_within_sets(data, sets, |g| hv_ind.compute(g))
                .iter()
                .sum::<f64>()
                / sets.iter().collect::<std::collections::HashSet<_>>().len() as f64;
            (v.to_owned(), h1)
        }
        None => {
            let res = vorob_t(data, sets, ref_point);
            (res.ve, res.avg_hyp)
        }
    };

    let h2 = hv_ind.compute(&ve_owned);
    let vd = 2.0
        * apply_within_sets(data, sets, |g| {
            let combined = concatenate![Axis(0), g.view(), ve_owned.view()];
            hv_ind.compute(&combined)
        })
        .iter()
        .sum::<f64>()
        / sets.iter().collect::<std::collections::HashSet<_>>().len() as f64;

    vd - h1 - h2
}

// ============================================================================
// EAF differences
// ============================================================================

/// Compute empirical attainment function (EAF) differences between two datasets.
/// Only supports 2 objectives.
pub fn eafdiff(
    x: &Array2<f64>,
    y: &Array2<f64>,
    intervals: Option<usize>,
    maximise: impl Into<MaximiseSpec>,
    rectangles: bool,
) -> Array2<f64> {
    assert_eq!(x.ncols(), y.ncols(), "'x' and 'y' must have the same number of columns");
    let nobj = x.ncols() - 1;
    assert_eq!(nobj, 2, "Only 2 objectives are currently supported");

    let maximise_arr = parse_maximise(&maximise.into(), nobj);

    let mut x = x.to_owned();
    let mut y = y.to_owned();

    // Sort each by set column
    let sort_by_set = |arr: &mut Array2<f64>| {
        let last = arr.ncols() - 1;
        let mut rows: Vec<Vec<f64>> = arr.rows().into_iter().map(|r| r.to_vec()).collect();
        rows.sort_by(|a, b| a[last].partial_cmp(&b[last]).unwrap());
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        *arr = Array2::from_shape_vec((arr.nrows(), arr.ncols()), flat).unwrap();
    };
    sort_by_set(&mut x);
    sort_by_set(&mut y);

    let count_sets = |arr: &Array2<f64>| -> usize {
        let last = arr.ncols() - 1;
        let mut ids: Vec<u64> = arr.column(last).iter().map(|&v| v.to_bits()).collect();
        ids.sort();
        ids.dedup();
        ids.len()
    };
    let nsets_x = count_sets(&x);
    let nsets_y = count_sets(&y);
    let nsets = nsets_x + nsets_y;
    let _intervals = intervals
        .map(|i| i.min(nsets / 2))
        .unwrap_or(nsets / 2);

    let mut data = {
        let x_obj = x.slice(s![.., ..nobj]).to_owned();
        let y_obj = y.slice(s![.., ..nobj]).to_owned();
        concatenate![Axis(0), x_obj.view(), y_obj.view()]
    };
    if maximise_arr.iter().any(|&v| v) {
        apply_maximise_transform(&mut data, &maximise_arr);
    }

    todo!("eafdiff C computation not yet implemented")
}

// ============================================================================
// Weighted hypervolume (rectangles-based)
// ============================================================================

/// Compute weighted hypervolume given a set of rectangles (2D only).
pub fn whv_rect(
    x: &Array2<f64>,
    rectangles: &Array2<f64>,
    ref_point: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
) -> f64 {
    let nobj = x.ncols();
    assert_eq!(nobj, 2, "Only 2D datasets are currently supported");
    assert_eq!(rectangles.ncols(), 5, "Invalid number of columns in 'rectangles' (should be 5)");
    let maximise_arr = parse_maximise(&maximise.into(), nobj);
    assert!(
        !maximise_arr.iter().any(|&v| v),
        "Only minimization is currently supported"
    );
    assert_eq!(ref_point.len(), nobj);
    todo!("whv_rect C computation not yet implemented")
}

/// Compute total weighted hypervolume: `hypervolume + scalefactor * |prod(ref - ideal)| * whv_rect`.
pub fn total_whv_rect(
    x: &Array2<f64>,
    rectangles: &Array2<f64>,
    ref_point: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
    ideal: Option<&Array1<f64>>,
    scalefactor: f64,
) -> f64 {
    let nobj = x.ncols();
    assert_eq!(nobj, 2, "Only 2D datasets are currently supported");
    assert_eq!(rectangles.ncols(), 5, "Invalid number of columns in 'rectangles' (should be 5)");
    assert!(scalefactor > 0.0 && scalefactor <= 1.0, "'scalefactor' must be within (0,1]");
    assert_eq!(ref_point.len(), nobj);

    let maximise_spec = maximise.into();
    let whv = whv_rect(x, rectangles, ref_point, maximise_spec.clone());
    let hv = hypervolume(x, ref_point, false);

    let ideal_owned: Array1<f64> = match ideal {
        Some(v) => {
            assert_eq!(v.len(), nobj);
            v.to_owned()
        }
        None => get_ideal(x, maximise_spec),
    };

    let beta = scalefactor * (ref_point - &ideal_owned).iter().product::<f64>().abs();
    hv + beta * whv
}

/// Identify the pair of datasets with the largest EAF differences (2D only).
pub fn largest_eafdiff(
    datasets: &[Array2<f64>],
    ref_point: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
    intervals: usize,
    ideal: Option<&Array1<f64>>,
) -> ((usize, usize), f64) {
    let n = datasets.len();
    assert!(n > 0, "Empty list");
    let nobj = datasets[0].ncols() - 1;
    assert_eq!(nobj, 2, "Only 2D datasets are currently supported");

    let maximise_spec = maximise.into();
    let maximise_arr = parse_maximise(&maximise_spec, nobj);

    let ideal_owned: Array1<f64> = match ideal {
        Some(v) => v.to_owned(),
        None => {
            let all_obj: Vec<Array2<f64>> = datasets
                .iter()
                .map(|z| z.slice(s![.., ..nobj]).to_owned())
                .collect();
            let views: Vec<_> = all_obj.iter().map(|a| a.view()).collect();
            let combined = concatenate(Axis(0), &views).unwrap();
            get_ideal(&combined, maximise_spec.clone())
        }
    };
    let ideal_row = ideal_owned.view().insert_axis(Axis(0)).to_owned();

    let mut best_value = 0.0f64;
    let mut best_pair = (0, 1);

    for a in 0..n - 1 {
        for b in (a + 1)..n {
            let diff = eafdiff(&datasets[a], &datasets[b], Some(intervals), maximise_arr.clone(), true);

            let a_rects: Array2<f64> = {
                let mask: Vec<usize> = (0..diff.nrows())
                    .filter(|&i| diff[[i, diff.ncols() - 1]] >= 1.0)
                    .collect();
                let mut r = diff.select(Axis(0), &mask);
                let last = r.ncols() - 1;
                r.column_mut(last).fill(1.0);
                r
            };
            let a_value = whv_rect(&ideal_row, &a_rects, ref_point, maximise_arr.clone());

            let b_rects: Array2<f64> = {
                let mask: Vec<usize> = (0..diff.nrows())
                    .filter(|&i| diff[[i, diff.ncols() - 1]] <= -1.0)
                    .collect();
                let mut r = diff.select(Axis(0), &mask);
                let last = r.ncols() - 1;
                r.column_mut(last).fill(1.0);
                r
            };
            let b_value = whv_rect(&ideal_row, &b_rects, ref_point, maximise_arr.clone());

            let value = a_value.min(b_value);
            if value > best_value {
                best_value = value;
                best_pair = (a, b);
            }
        }
    }
    (best_pair, best_value)
}

// ============================================================================
// Approximate weighted hypervolume (Monte Carlo, 2D only)
// ============================================================================

#[derive(Debug, Clone)]
pub enum WhvDist {
    Uniform,
    Point(Array1<f64>),
    Exponential(f64),
}

/// Approximation of the (weighted) hypervolume by Monte-Carlo sampling (2D only).
pub fn whv_hype(
    data: &Array2<f64>,
    ref_point: &Array1<f64>,
    ideal: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
    nsamples: u32,
    seed: Option<u64>,
    dist: WhvDist,
) -> f64 {
    let nobj = data.ncols();
    assert_eq!(nobj, 2, "Only 2D datasets are currently supported");
    assert_eq!(ref_point.len(), nobj);
    assert_eq!(ideal.len(), nobj);

    let maximise_arr = parse_maximise(&maximise.into(), nobj);
    let mut data = data.to_owned();
    let mut ref_point = ref_point.to_owned();
    let mut ideal = ideal.to_owned();

    if maximise_arr.iter().any(|&v| v) {
        apply_maximise_transform(&mut data, &maximise_arr);
        for (j, &max) in maximise_arr.iter().enumerate() {
            if max {
                ref_point[j] = -ref_point[j];
                ideal[j] = -ideal[j];
            }
        }
    }

    todo!("whv_hype C computation not yet implemented")
}

// ============================================================================
// Apply within sets
// ============================================================================

/// Split `x` by row according to `sets` and apply `func` to each sub-array, returning a scalar per set.
/// Results are ordered by unique set IDs (sorted).
pub fn apply_within_sets<F>(x: &Array2<f64>, sets: &Array1<f64>, func: F) -> Array1<f64>
where
    F: Fn(&Array2<f64>) -> f64,
{
    assert_eq!(x.nrows(), sets.len());

    let mut unique_ids: Vec<f64> = sets.to_vec();
    unique_ids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_ids.dedup();

    let results: Vec<f64> = unique_ids
        .iter()
        .map(|&set_id| {
            let indices: Vec<usize> = sets
                .iter()
                .enumerate()
                .filter_map(|(i, &s)| if s == set_id { Some(i) } else { None })
                .collect();
            let subset = x.select(Axis(0), &indices);
            func(&subset)
        })
        .collect();

    Array1::from_vec(results)
}

/// Split `x` by row according to `sets`, apply `func` returning a 2D array per group,
/// and reassemble in original row order when each result has the same number of rows as the input group.
pub fn apply_within_sets_rows<F>(x: &Array2<f64>, sets: &Array1<f64>, func: F) -> Array2<f64>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    assert_eq!(x.nrows(), sets.len());

    let mut unique_ids: Vec<f64> = sets.to_vec();
    unique_ids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_ids.dedup();

    let groups: Vec<Vec<usize>> = unique_ids
        .iter()
        .map(|&set_id| {
            sets.iter()
                .enumerate()
                .filter_map(|(i, &s)| if s == set_id { Some(i) } else { None })
                .collect()
        })
        .collect();

    let results: Vec<Array2<f64>> = groups.iter().map(|g| func(&x.select(Axis(0), g))).collect();

    let shorter = results.iter().zip(groups.iter()).any(|(r, g)| r.nrows() != g.len());

    let views: Vec<_> = results.iter().map(|r| r.view()).collect();
    let concatenated = concatenate(Axis(0), &views).unwrap();

    if shorter {
        return concatenated;
    }

    let all_indices: Vec<usize> = groups.into_iter().flatten().collect();
    let n = all_indices.len();
    let mut inverse = vec![0usize; n];
    for (new_pos, &orig_pos) in all_indices.iter().enumerate() {
        inverse[orig_pos] = new_pos;
    }
    concatenated.select(Axis(0), &inverse)
}

// ============================================================================
// R2 indicator
// ============================================================================

/// Exact R2 indicator (2D only).
pub fn r2_exact(
    data: &Array2<f64>,
    ref_point: &Array1<f64>,
    maximise: impl Into<MaximiseSpec>,
) -> f64 {
    let nobj = data.ncols();
    assert_eq!(nobj, 2, "Only 2D datasets are currently supported");
    assert_eq!(ref_point.len(), nobj);

    let maximise_arr = parse_maximise(&maximise.into(), nobj);
    let mut data = data.to_owned();
    let mut ref_point = ref_point.to_owned();
    if maximise_arr.iter().any(|&v| v) {
        apply_maximise_transform(&mut data, &maximise_arr);
        for (j, &max) in maximise_arr.iter().enumerate() {
            if max {
                ref_point[j] = -ref_point[j];
            }
        }
    }
    todo!("r2_exact C computation not yet implemented")
}
