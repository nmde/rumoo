use ndarray::Array2;
use rand::rngs::StdRng;

use crate::{
    core::{
        individual::{IndividualField, Value},
        operator::Operator,
        population::Population,
        problem::Problem,
    },
    util::default_random_state,
};

/// Extra data fields specific to mutation operators.
///
/// Mirrors the `__init__` attributes added by `pymoo.core.mutation.Mutation`
/// on top of `Operator`.
pub struct MutationBase {
    /// Per-individual mutation probability (default 1.0).
    /// Mirrors `self.prob = Real(prob, bounds=(0.7, 1.0), strict=(0.0, 1.0))`.
    pub prob: f64,
    /// Per-variable mutation probability.
    /// Mirrors `self.prob_var = Real(prob_var, ...)` (None means use `min(0.5, 1/n_var)`).
    pub prob_var: Option<f64>,
}

impl MutationBase {
    pub fn new(prob: Option<f64>, prob_var: Option<f64>) -> Self {
        Self {
            prob: prob.unwrap_or(1.0),
            prob_var,
        }
    }
}

/// Abstract base for mutation operators.
///
/// Mirrors `pymoo.core.mutation.Mutation`.
pub trait Mutation: Operator {
    fn mutation_base(&self) -> &MutationBase;

    /// Apply mutation to a population and return the (possibly modified) result.
    ///
    /// `inplace` is kept for API fidelity with Python; in Rust the return value
    /// always carries the (potentially mutated) population and the caller
    /// reassigns. When `inplace` is `false` the input `pop` is cloned before
    /// mutation so the original is logically unchanged — mirrors `deepcopy(pop)`.
    ///
    /// Mirrors `Mutation.do(problem, pop, inplace=True, random_state=None)`.
    fn do_mutation(
        &self,
        problem: &dyn Problem,
        pop: &mut Population,
        inplace: Option<bool>,
        mut random_state: Option<&mut StdRng>,
    ) -> Population {
        let inplace = inplace.unwrap_or(true);
        let random_state = random_state.unwrap_or(&mut default_random_state());
        // if not inplace copy the population first (mirrors `pop = deepcopy(pop)`)
        let mut out = if inplace { pop } else { pop.clone() };

        let n_mut = out.len();

        // get the variables to be mutated: X shape (n_mut, n_var)
        let x = match out.get(&IndividualField::X) {
            Value::FloatMatrix(m) => m,
            _ => return out.clone(),
        };

        // retrieve the mutated variable matrix from the implementation
        let xp = self._do_mutation(problem, &x, Some(&mut random_state));

        // per-individual mutation mask
        // Mirrors: prob = get(self.prob, size=n_mut); mut = random_state.random(size=n_mut) <= prob
        let prob = self.mutation_base().prob;
        let mut mut_mask = vec![false; n_mut];
        for m in mut_mask.iter_mut() {
            *m = random_state.random::<f64>() <= prob;
        }

        // store the mutated X back for individuals where the mask is true
        // mirrors: pop[mut].set("X", Xp[mut])
        for (i, ind) in out.iter_mut().enumerate() {
            if mut_mask[i] {
                ind.x = Some(xp.row(i).to_owned());
            }
        }

        out.clone()
    }

    /// Apply the mutation to the decision variable matrix.
    ///
    /// Default implementation is the identity (no change), matching Python's
    /// `def _do(self, problem, X, ...) -> return X`.
    ///
    /// Subclasses override this to implement specific mutation strategies.
    /// Input and output shape: `(n_individuals, n_var)`.
    ///
    /// Mirrors `Mutation._do(problem, X, random_state)`.
    fn _do_mutation(
        &self,
        _problem: &dyn Problem,
        x: &Array2<f64>,
        _random_state: Option<&mut StdRng>,
    ) -> Array2<f64> {
        x.clone()
    }

    /// Return the per-variable mutation probability for a given problem.
    ///
    /// Falls back to `min(0.5, 1 / n_var)` when `prob_var` was not set.
    ///
    /// Mirrors `Mutation.get_prob_var(problem)`.
    fn get_prob_var(&self, problem: &dyn Problem) -> f64 {
        match self.mutation_base().prob_var {
            Some(p) => p,
            None => f64::min(0.5, 1.0 / problem.base().n_var as f64),
        }
    }
}
