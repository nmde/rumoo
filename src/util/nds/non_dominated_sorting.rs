use ndarray::{Array1, Array2};

use crate::util::{dominator::Dominator, nds::fast_non_dominated_sort::fast_non_dominated_sort};

enum NonDominatedSortingMethod {
    FastNonDominatedSort,
}

/// Mirrors `pymoo.util.nds.non_dominated_sorting.NonDominatedSorting`.
pub struct NonDominatedSorting {
    pub epsilon: Option<f64>,
    pub method: NonDominatedSortingMethod,
    pub dominator: Option<Dominator>,
}

impl NonDominatedSorting {
    /// Mirrors `NonDominatedSorting.__init__(epsilon=None, method="fast_non_dominated_sort")`.
    pub fn new(
        epsilon: Option<f64>,
        method: Option<NonDominatedSortingMethod>,
        dominator: Option<Dominator>,
    ) -> Self {
        Self {
            epsilon,
            method: method.unwrap_or(NonDominatedSortingMethod::FastNonDominatedSort),
            dominator,
        }
    }

    /// Mirrors `NonDominatedSorting.do(F, return_rank, only_non_dominated_front, n_stop_if_ranked, n_fronts)`.
    ///
    /// Returns `(fronts, rank)` — `rank` is `None` unless `return_rank` is `true`.
    /// `do` is a Rust keyword; this method is named `sort`.
    pub fn sort(
        &self,
        f: &Array2<f64>,
        return_rank: Option<bool>,
        only_non_dominated_front: Option<bool>,
        n_stop_if_ranked: Option<usize>,
        n_fronts: Option<usize>,
    ) -> (Vec<Vec<usize>>, Option<Array1<usize>>) {
        let n_stop_if_ranked = n_stop_if_ranked.unwrap_or(100_000_000);
        let return_rank = return_rank.unwrap_or(false);
        let only_non_dominated_front = only_non_dominated_front.unwrap_or(false);

        let n_fronts = if only_non_dominated_front {
            1
        } else {
            n_fronts.unwrap_or(100_000_000)
        };

        // Mirrors: if self.dominator is not None: use fast_non_dominated_sort with custom dominator
        //          else: func = load_function(self.method); fronts = func(F, ...)
        let raw_fronts: Vec<Vec<usize>> = if self.dominator.is_some() {
            fast_non_dominated_sort(f, self.dominator, None)
        } else {
            let epsilon = self.epsilon;
            match self.method {
                NonDominatedSortingMethod::FastNonDominatedSort => {
                    fast_non_dominated_sort(f, None, epsilon)
                }
                _ => fast_non_dominated_sort(f, None, epsilon),
            }
        };

        // Mirrors: collect fronts, stop when n_stop_if_ranked reached, limit to n_fronts
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut n_ranked = 0usize;
        for front in raw_fronts.into_iter().take(n_fronts) {
            n_ranked += front.len();
            fronts.push(front);
            if n_ranked >= n_stop_if_ranked {
                break;
            }
        }

        let rank = if return_rank {
            Some(rank_from_fronts(&fronts, f.nrows()))
        } else {
            None
        };

        (fronts, rank)
    }

    /// Convenience wrapper: mirrors `do(F, only_non_dominated_front=True)`.
    ///
    /// Returns the indices of the first (non-dominated) front.
    pub fn do_sort(&self, f: &Array2<f64>, only_non_dominated_front: bool) -> Vec<usize> {
        let (fronts, _) = self.sort(f, Some(false), Some(only_non_dominated_front), None, None);
        if only_non_dominated_front {
            fronts.into_iter().next().unwrap_or_default()
        } else {
            fronts.into_iter().flatten().collect()
        }
    }
}

/// Mirrors `pymoo.util.nds.non_dominated_sorting.rank_from_fronts(fronts, n)`.
///
/// Returns a rank array of length `n`; `rank[i]` is the front index of solution `i`.
/// Solutions not assigned to any front receive `usize::MAX` (mirrors `sys.maxsize`).
pub fn rank_from_fronts(fronts: &[Vec<usize>], n: usize) -> Array1<usize> {
    let mut rank = Array1::from_elem(n, usize::MAX);
    for (i, front) in fronts.iter().enumerate() {
        for &idx in front {
            rank[idx] = i;
        }
    }
    rank
}

/// Mirrors `pymoo.util.nds.non_dominated_sorting.find_non_dominated(F, _F=None)`.
///
/// Returns indices of rows in `F` that are not dominated.
/// If `_f` is `None`, delegates to the compiled `find_non_dominated` function.
/// If `_f` is provided, uses the domination matrix fallback.
pub fn find_non_dominated(f: &Array2<f64>, _f: Option<&Array2<f64>>) -> Vec<usize> {
    match _f {
        None => {
            // Mirrors: indices = func(F.astype(float)); return np.array(indices, dtype=int)
            find_non_dominated_fast(f)
        }
        Some(other) => {
            // Mirrors: M = Dominator.calc_domination_matrix(F, _F)
            //          I = np.where(np.all(M >= 0, axis=1))[0]
            let m = Dominator::calc_domination_matrix(f, Some(other), None);
            m.outer_iter()
                .enumerate()
                .filter_map(|(i, row)| {
                    if row.iter().all(|&v| v >= 0) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
}
