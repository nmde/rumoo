use ndarray::Array2;
use rand::rngs::StdRng;

use crate::core::{operator::Operator, population::Population, problem::Problem};

/// Return type of `Selection::do_selection` — either a population of individuals
/// or a raw index matrix.
///
/// Mirrors the two return paths in `Selection.do`:
/// - integer ndarray → `Indices`
/// - anything else  → `Individuals`
pub enum SelectionResult {
    /// Indices into the source population, shape `(n_select, n_parents)`.
    Indices(Array2<i64>),
    /// Population of selected individuals.
    Population(Population),
}

/// Abstract base for parent-selection strategies.
///
/// Mirrors `pymoo.core.selection.Selection`.
pub trait Selection: Operator {
    /// Select `n_select * n_parents` individuals from `pop`.
    ///
    /// If `to_pop` is `true` and `_do_selection` returns `SelectionResult::Indices`,
    /// the indices are used to look up the actual individuals and the result is
    /// converted to `SelectionResult::Individuals`.
    ///
    /// Mirrors `Selection.do(problem, pop, n_select, n_parents, to_pop=True,
    ///                       random_state=None)`.
    fn do_selection(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        n_select: usize,
        n_parents: usize,
        to_pop: Option<bool>,
        random_state: Option<&mut StdRng>,
    ) -> SelectionResult {
        let to_pop = to_pop.unwrap_or(true);
        let ret = self._do_selection(problem, pop, n_select, n_parents, None, random_state);

        // if some selections return indices they are used to create the individual list
        if to_pop {
            if let SelectionResult::Indices(ref idx) = ret {
                return SelectionResult::Population(pop.select_by_index_matrix(idx));
            }
        }

        ret
    }

    /// Abstract — subclasses must implement.
    ///
    /// Mirrors `Selection._do(problem, pop, n_select, n_parents, random_state)`.
    fn _do_selection(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        n_select: usize,
        n_parents: usize,
        to_pop: Option<bool>,
        random_state: Option<&mut StdRng>,
    ) -> SelectionResult;
}
