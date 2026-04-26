use std::{
    collections::{HashSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

use ndarray::{Array1, Array2};

use crate::{
    core::{
        individual::{Individual, IndividualField, Value},
        population::Population,
    },
    util::misc::cdist,
};

// -------------------------------------------------------------------------------------------------
// default_attr
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.core.duplicate.default_attr`.
pub fn default_attr(pop: &Population) -> Array2<f64> {
    match pop.get(&IndividualField::X) {
        Value::FloatMatrix(m) => m,
        _ => Array2::zeros((pop.len(), 0)),
    }
}

// -------------------------------------------------------------------------------------------------
// DuplicateResult / DuplicateElimination trait
// -------------------------------------------------------------------------------------------------

/// Return type for `DuplicateElimination::do_elimination`.
///
/// Mirrors the two return paths of `DuplicateElimination.do`:
/// - `return_indices=False` → `Filtered`
/// - `return_indices=True`  → `WithIndices`
pub enum DuplicateResult {
    Filtered(Population),
    WithIndices {
        pop: Population,
        no_duplicate: Vec<usize>,
        is_duplicate: Vec<usize>,
    },
}

/// Mirrors `pymoo.core.duplicate.DuplicateElimination`.
pub trait DuplicateElimination {
    /// Mark individuals in `pop` that are duplicates.
    ///
    /// Mirrors `DuplicateElimination._do(pop, other, is_duplicate)`.
    fn _do(&self, pop: &Population, other: Option<&Population>, is_duplicate: &mut Array1<bool>);

    /// Filter duplicates from `pop`, optionally checking against extra populations.
    ///
    /// Mirrors `DuplicateElimination.do(pop, *args, return_indices, to_itself)`.
    fn do_elimination(
        &self,
        pop: &Population,
        others: &[&Population],
        return_indices: bool,
        to_itself: bool,
    ) -> DuplicateResult {
        let original_len = pop.len();

        if original_len == 0 {
            return if return_indices {
                DuplicateResult::WithIndices {
                    pop: Population::empty(0),
                    no_duplicate: vec![],
                    is_duplicate: vec![],
                }
            } else {
                DuplicateResult::Filtered(Population::empty(0))
            };
        }

        // surviving holds indices into the original pop that have not yet been eliminated.
        let mut surviving: Vec<usize> = (0..original_len).collect();

        // Mirrors: if to_itself: pop = pop[~self._do(pop, None, np.full(len(pop), False))]
        if to_itself {
            let sub = pop.select(&surviving);
            let mut is_dup = Array1::from_elem(surviving.len(), false);
            self._do(&sub, None, &mut is_dup);
            surviving = surviving
                .into_iter()
                .zip(is_dup.iter())
                .filter_map(|(idx, &dup)| if !dup { Some(idx) } else { None })
                .collect();
        }

        // Mirrors: for arg in args: pop = pop[~self._do(pop, arg, ...)]
        for &other in others {
            if surviving.is_empty() {
                break;
            }
            if other.is_empty() {
                continue;
            }
            let sub = pop.select(&surviving);
            let mut is_dup = Array1::from_elem(surviving.len(), false);
            self._do(&sub, Some(other), &mut is_dup);
            surviving = surviving
                .into_iter()
                .zip(is_dup.iter())
                .filter_map(|(idx, &dup)| if !dup { Some(idx) } else { None })
                .collect();
        }

        if return_indices {
            // Mirrors: H = set(pop); for i, ind in enumerate(original): ...
            let surviving_set: HashSet<usize> = surviving.iter().copied().collect();
            let no_dup: Vec<usize> = (0..original_len)
                .filter(|i| surviving_set.contains(i))
                .collect();
            let is_dup_idx: Vec<usize> = (0..original_len)
                .filter(|i| !surviving_set.contains(i))
                .collect();
            DuplicateResult::WithIndices {
                pop: pop.select(&surviving),
                no_duplicate: no_dup,
                is_duplicate: is_dup_idx,
            }
        } else {
            DuplicateResult::Filtered(pop.select(&surviving))
        }
    }
}

// -------------------------------------------------------------------------------------------------
// DefaultDuplicateElimination
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.core.duplicate.DefaultDuplicateElimination`.
pub struct DefaultDuplicateElimination {
    pub func: fn(&Population) -> Array2<f64>,
    pub epsilon: f64,
}

impl DefaultDuplicateElimination {
    pub fn new(func: Option<fn(&Population) -> Array2<f64>>, epsilon: Option<f64>) -> Self {
        Self {
            func: func.unwrap_or(default_attr),
            epsilon: epsilon.unwrap_or(1e-16),
        }
    }

    /// Mirrors `DefaultDuplicateElimination.calc_dist(pop, other)`.
    pub fn calc_dist(&self, pop: &Population, other: Option<&Population>) -> Array2<f64> {
        let x = (self.func)(pop);
        let n = x.nrows();

        if let Some(other_pop) = other {
            let x_other = (self.func)(other_pop);
            cdist(&x, &x_other).unwrap_or_else(|_| Array2::zeros((n, other_pop.len())))
        } else {
            // D = cdist(X, X); D[np.triu_indices(len(X))] = np.inf
            let mut d = cdist(&x, &x).unwrap_or_else(|_| Array2::zeros((n, n)));
            for i in 0..n {
                for j in i..n {
                    d[[i, j]] = f64::INFINITY;
                }
            }
            d
        }
    }
}

impl DuplicateElimination for DefaultDuplicateElimination {
    /// Mirrors `DefaultDuplicateElimination._do(pop, other, is_duplicate)`.
    fn _do(&self, pop: &Population, other: Option<&Population>, is_duplicate: &mut Array1<bool>) {
        let mut d = self.calc_dist(pop, other);
        // D[np.isnan(D)] = np.inf
        d.mapv_inplace(|v| if v.is_nan() { f64::INFINITY } else { v });
        // is_duplicate[np.any(D <= self.epsilon, axis=1)] = True
        for i in 0..d.nrows() {
            if d.row(i).iter().any(|&v| v <= self.epsilon) {
                is_duplicate[i] = true;
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// to_float / ElementwiseDuplicateElimination
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.core.duplicate.to_float`.
///
/// In Python this converts a bool to float: True → 0.0, False → 1.0.
/// In Rust the comparison function already returns `f64` directly.
pub fn to_float(val: bool) -> f64 {
    if val { 0.0 } else { 1.0 }
}

/// Mirrors `pymoo.core.duplicate.ElementwiseDuplicateElimination`.
pub struct ElementwiseDuplicateElimination {
    pub epsilon: f64,
    pub cmp_func: Box<dyn Fn(&Individual, &Individual) -> f64>,
}

impl ElementwiseDuplicateElimination {
    pub fn new(
        cmp_func: Option<Box<dyn Fn(&Individual, &Individual) -> f64>>,
        epsilon: Option<f64>,
    ) -> Self {
        Self {
            epsilon: epsilon.unwrap_or(1e-16),
            // default mirrors is_equal (abstract — returns not-equal by default)
            cmp_func: cmp_func
                .unwrap_or_else(|| Box::new(|_: &Individual, _: &Individual| f64::INFINITY)),
        }
    }
}

impl DuplicateElimination for ElementwiseDuplicateElimination {
    /// Mirrors `ElementwiseDuplicateElimination._do(pop, other, is_duplicate)`.
    fn _do(&self, pop: &Population, other: Option<&Population>, is_duplicate: &mut Array1<bool>) {
        if let Some(other_pop) = other {
            // Mirrors: for i in range(len(pop)): for j in range(len(other)): ...
            for i in 0..pop.len() {
                for j in 0..other_pop.len() {
                    let val = (self.cmp_func)(&pop[i], &other_pop[j]);
                    if val < self.epsilon {
                        is_duplicate[i] = true;
                        break;
                    }
                }
            }
        } else {
            // Mirrors: for i in range(len(pop)): for j in range(i+1, len(pop)): ...
            for i in 0..pop.len() {
                for j in (i + 1)..pop.len() {
                    let val = (self.cmp_func)(&pop[i], &pop[j]);
                    if val < self.epsilon {
                        is_duplicate[i] = true;
                        break;
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// to_hash / HashDuplicateElimination
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.core.duplicate.to_hash`.
///
/// Hashes the decision variable (X) values of an individual.
/// Falls back to hashing a string representation if X is unavailable.
pub fn to_hash(ind: &Individual) -> u64 {
    let mut hasher = DefaultHasher::new();
    match ind.data.get("X") {
        Some(Value::FloatArray(arr)) => {
            for &v in arr.iter() {
                v.to_bits().hash(&mut hasher);
            }
        }
        Some(Value::FloatMatrix(mat)) => {
            for &v in mat.iter() {
                v.to_bits().hash(&mut hasher);
            }
        }
        _ => {
            "no_x".hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Mirrors `pymoo.core.duplicate.HashDuplicateElimination`.
pub struct HashDuplicateElimination {
    pub func: Box<dyn Fn(&Individual) -> u64>,
}

impl HashDuplicateElimination {
    pub fn new(func: Option<Box<dyn Fn(&Individual) -> u64>>) -> Self {
        Self {
            func: func.unwrap_or_else(|| Box::new(to_hash)),
        }
    }
}

impl DuplicateElimination for HashDuplicateElimination {
    /// Mirrors `HashDuplicateElimination._do(pop, other, is_duplicate)`.
    fn _do(&self, pop: &Population, other: Option<&Population>, is_duplicate: &mut Array1<bool>) {
        let mut h_set: HashSet<u64> = HashSet::new();

        // Mirrors: if other is not None: for o in other: H.add(self.func(o))
        if let Some(other_pop) = other {
            for i in 0..other_pop.len() {
                h_set.insert((self.func)(&other_pop[i]));
            }
        }

        // Mirrors: for i, ind in enumerate(pop): h = self.func(ind); ...
        for i in 0..pop.len() {
            let h = (self.func)(&pop[i]);
            if h_set.contains(&h) {
                is_duplicate[i] = true;
            } else {
                h_set.insert(h);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// NoDuplicateElimination
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.core.duplicate.NoDuplicateElimination`.
pub struct NoDuplicateElimination;

impl DuplicateElimination for NoDuplicateElimination {
    fn _do(
        &self,
        _pop: &Population,
        _other: Option<&Population>,
        _is_duplicate: &mut Array1<bool>,
    ) {
        // unreachable — do_elimination is overridden to bypass _do entirely
    }

    /// Mirrors `NoDuplicateElimination.do(pop)` — returns `pop` unchanged.
    fn do_elimination(
        &self,
        pop: &Population,
        _others: &[&Population],
        return_indices: bool,
        _to_itself: bool,
    ) -> DuplicateResult {
        let n = pop.len();
        let all: Vec<usize> = (0..n).collect();
        if return_indices {
            DuplicateResult::WithIndices {
                pop: pop.select(&all),
                no_duplicate: all,
                is_duplicate: vec![],
            }
        } else {
            DuplicateResult::Filtered(pop.select(&(0..n).collect::<Vec<_>>()))
        }
    }
}
