use anyhow::{Result, anyhow};
use ndarray::{Array2, s};
use rand::{rngs::StdRng, seq::IndexedRandom};

use crate::{
    core::{population::Population, problem::Problem},
    util::{default_random_state, misc::random_permutations},
};

/*
from pymoo.core.selection import Selection
from pymoo.util.misc import random_permutations
from pymoo.util import default_random_state
*/

/// The comparison function stored by `TournamentSelection`.
///
/// Mirrors the Python `func_comp` parameter:
/// `func_comp(pop, P, random_state=random_state, **kwargs)`
///
/// In Python the algorithm reference is forwarded via `**kwargs`. In Rust
/// that context must be captured by the caller when constructing the closure
/// (or passed through a trait-object context if more flexibility is needed).
pub type CompareFunc =
    Box<dyn Fn(&Population, &Array2<usize>, Option<&mut StdRng>) -> Result<Array2<i64>>>;

/// Simulate a tournament between individuals.
/// The selection pressure balances how greedy the genetic algorithm will be.
///
/// Mirrors `pymoo.operators.selection.tournament.TournamentSelection`.
pub struct TournamentSelection {
    /// Comparison function that decides the tournament winner.
    pub func_comp: CompareFunc,
    /// Number of individuals competing per tournament slot (default: 2 = binary).
    pub pressure: usize,
}

impl TournamentSelection {
    /// Create a new `TournamentSelection`.
    ///
    /// # Panics
    /// Mirrors Python's `raise Exception("Please provide the comparing function …")` —
    /// the compiler enforces this at the type level; no runtime panic needed.
    pub fn new(func_comp: Option<CompareFunc>, pressure: Option<usize>) -> Result<Self> {
        if func_comp.is_none() {
            return Err(anyhow!(
                "Please provide the comparing function for the tournament selection!"
            ));
        }
        Ok(Self {
            func_comp: Box::new(func_comp.unwrap()),
            pressure: pressure.unwrap_or(2),
        })
    }

    /// Run the tournament and return the selected parent indices.
    ///
    /// Mirrors `TournamentSelection._do(_, pop, n_select, n_parents, random_state)`.
    ///
    /// Returns an `Array2<i64>` of shape `(n_select, n_parents)`.
    pub fn _do(
        &self,
        pop: &Population,
        n_select: usize,
        n_parents: Option<usize>,
        mut random_state: Option<&mut StdRng>,
    ) -> Result<Array2<i64>> {
        let n_parents = n_parents.unwrap_or(1);

        // number of random individuals needed
        let n_random = n_select * n_parents * self.pressure;

        // number of permutations needed
        // n_perms = math.ceil(n_random / len(pop))
        let n_perms = (n_random as f64 / pop.len() as f64).ceil() as usize;

        // get random permutations and reshape them
        // P = random_permutations(n_perms, len(pop), random_state=random_state)[:n_random]
        let p_flat = random_permutations(n_perms, pop.len(), random_state.as_deref_mut());
        let p_flat = p_flat.slice(s![..n_random]).to_owned();

        // P = np.reshape(P, (n_select * n_parents, self.pressure))
        let p = p_flat.into_shape((n_select * n_parents, self.pressure))?;

        // compare using tournament function
        // S = self.func_comp(pop, P, random_state=random_state, **kwargs)
        let s = (self.func_comp)(pop, &p, random_state);

        // return np.reshape(S, (n_select, n_parents))
        Ok(s.into_shape((n_select, n_parents))?)
    }
}

impl Selection for TournamentSelection {
    fn do_selection(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        n_select: usize,
        n_parents: usize,
        to_pop: bool,
        random_state: Option<&mut StdRng>,
    ) -> SelectionResult {
        let indices = self.do_select(pop, n_select, n_parents, random_state);

        // if some selections return indices they are used to create the individual list
        // if to_pop and isinstance(ret, np.ndarray) and np.issubdtype(ret.dtype, np.integer):
        //     ret = pop[ret]
        if to_pop {
            SelectionResult::Individuals(pop.select_by_index_matrix(&indices))
        } else {
            SelectionResult::Indices(indices)
        }
    }
}

pub enum CompareMethod {
    LargerIsBetter,
    SmallerIsBetter,
}

/// Compare two tournament candidates by value and return the winner's index.
///
/// | `method`            | winner condition |
/// |---------------------|-----------------|
/// | `"larger_is_better"` | higher `a_val` / `b_val` wins |
/// | `"smaller_is_better"`| lower `a_val` / `b_val` wins |
///
/// Returns `Some(a)` or `Some(b)` on a clear winner; on a tie returns
/// `Some(random choice)` when `return_random_if_equal` is `true`, else `None`.
///
/// Mirrors the module-level `compare()` in `pymoo.operators.selection.tournament`,
/// including the `@default_random_state` decorator behaviour (random_state is
/// optional; `None` is accepted and treated as "no randomness available").
pub fn compare(
    a: usize,
    a_val: f64,
    b: usize,
    b_val: f64,
    method: &CompareMethod,
    return_random_if_equal: Option<bool>,
    random_state: Option<&mut StdRng>,
) -> Option<usize> {
    let return_random_if_equal = return_random_if_equal.unwrap_or(false);
    let random_state = random_state.unwrap_or(&mut default_random_state());
    match method {
        CompareMethod::LargerIsBetter => {
            if a_val > b_val {
                Some(a)
            } else if a_val < b_val {
                Some(b)
            } else if return_random_if_equal {
                Some(*[a, b].choose(random_state).unwrap())
            } else {
                None
            }
        }
        CompareMethod::SmallerIsBetter => {
            if a_val < b_val {
                Some(a)
            } else if a_val > b_val {
                Some(b)
            } else if return_random_if_equal {
                Some(*[a, b].choose(random_state).unwrap())
            } else {
                None
            }
        }
    }
}
