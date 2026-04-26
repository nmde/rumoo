use anyhow::{Result, anyhow};
use ndarray::{Array2, s};
use rand::{rngs::StdRng, seq::IndexedRandom};

use crate::{
    core::{
        operator::{Operator, OperatorBase},
        population::Population,
        problem::Problem,
        selection::{Selection, SelectionResult},
    },
    util::{default_random_state, misc::random_permutations},
};

/// The comparison function stored by `TournamentSelection`.
///
/// Mirrors the Python `func_comp` parameter:
/// `func_comp(pop, P, random_state=random_state, **kwargs)`
pub type CompareFunc =
    Box<dyn Fn(&Population, &Array2<usize>, Option<&mut StdRng>) -> Result<Array2<i64>>>;

/// Simulate a tournament between individuals.
///
/// Mirrors `pymoo.operators.selection.tournament.TournamentSelection`.
pub struct TournamentSelection {
    pub base: OperatorBase,
    pub func_comp: CompareFunc,
    /// Number of individuals competing per tournament slot (default: 2 = binary).
    pub pressure: usize,
}

impl TournamentSelection {
    /// Mirrors `TournamentSelection.__init__(func_comp, pressure=2)`.
    ///
    /// Returns `Err` if `func_comp` is `None`, mirroring Python's runtime exception.
    pub fn new(func_comp: Option<CompareFunc>, pressure: Option<usize>) -> Result<Self> {
        let func_comp = func_comp.ok_or_else(|| {
            anyhow!("Please provide the comparing function for the tournament selection!")
        })?;
        Ok(Self {
            base: OperatorBase::new(None, None, None),
            func_comp,
            pressure: pressure.unwrap_or(2),
        })
    }

    /// Run tournament selection and return selected parent indices.
    ///
    /// Mirrors `TournamentSelection._do(_, pop, n_select, n_parents, random_state)`.
    /// Renamed from `_do` to avoid a name collision with `Operator::_do`.
    ///
    /// Returns an `Array2<i64>` of shape `(n_select, n_parents)`.
    pub fn _do_tournament(
        &self,
        pop: &Population,
        n_select: usize,
        n_parents: Option<usize>,
        mut random_state: Option<&mut StdRng>,
    ) -> Result<Array2<i64>> {
        let n_parents = n_parents.unwrap_or(1);

        // Mirrors: n_random = n_select * n_parents * self.pressure
        let n_random = n_select * n_parents * self.pressure;

        // Mirrors: n_perms = math.ceil(n_random / len(pop))
        let n_perms = (n_random as f64 / pop.len() as f64).ceil() as usize;

        // Mirrors: P = random_permutations(n_perms, len(pop))[:n_random]
        let p_flat = random_permutations(n_perms, pop.len(), random_state.as_deref_mut());
        let p_flat = p_flat.slice(s![..n_random]).to_owned();

        // Mirrors: P = np.reshape(P, (n_select * n_parents, self.pressure))
        let p = p_flat.into_shape((n_select * n_parents, self.pressure))?;

        // Mirrors: S = self.func_comp(pop, P, random_state=random_state)
        let s = (self.func_comp)(pop, &p, random_state)?;

        // Mirrors: return np.reshape(S, (n_select, n_parents))
        Ok(s.into_shape((n_select, n_parents))?)
    }
}

impl Operator for TournamentSelection {
    fn base(&self) -> &OperatorBase {
        &self.base
    }

    fn _do(
        &self,
        _problem: &dyn Problem,
        _elem: &Population,
        _random_state: Option<&mut StdRng>,
    ) -> Population {
        unimplemented!(
            "TournamentSelection is invoked via Selection::do_selection, not Operator::do_op"
        )
    }
}

impl Selection for TournamentSelection {
    /// Mirrors `TournamentSelection._do(_, pop, n_select, n_parents, random_state)`.
    fn _do_selection(
        &self,
        _problem: &dyn Problem,
        pop: &Population,
        n_select: usize,
        n_parents: usize,
        _to_pop: Option<bool>,
        random_state: Option<&mut StdRng>,
    ) -> SelectionResult {
        let indices = self
            ._do_tournament(pop, n_select, Some(n_parents), random_state)
            .expect("tournament selection failed");
        SelectionResult::Indices(indices)
    }
}

// -------------------------------------------------------------------------------------------------
// compare() utility
// -------------------------------------------------------------------------------------------------

pub enum CompareMethod {
    LargerIsBetter,
    SmallerIsBetter,
}

/// Compare two tournament candidates by value and return the winner's index.
///
/// Mirrors the module-level `compare()` in `pymoo.operators.selection.tournament`.
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
    let mut fallback_rng = default_random_state();
    let rng = random_state.unwrap_or(&mut fallback_rng);

    match method {
        CompareMethod::LargerIsBetter => {
            if a_val > b_val {
                Some(a)
            } else if a_val < b_val {
                Some(b)
            } else if return_random_if_equal {
                Some(*[a, b].choose(rng).unwrap())
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
                Some(*[a, b].choose(rng).unwrap())
            } else {
                None
            }
        }
    }
}
