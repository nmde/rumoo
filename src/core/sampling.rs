use ndarray::Array2;
use rand::rngs::StdRng;

use crate::core::{
    individual::{IndividualField, Value},
    operator::Operator,
    population::Population,
    problem::Problem,
};

/// Abstract base for all sampling strategies.
///
/// Mirrors `pymoo.core.sampling.Sampling`.
pub trait Sampling: Operator {
    /// Sample `n_samples` new points for the given `problem`.
    ///
    /// Mirrors `Sampling.do(problem, n_samples, random_state=None)`.
    fn do_sampling(
        &self,
        problem: &dyn Problem,
        n_samples: usize,
        random_state: Option<&mut StdRng>,
    ) -> Population {
        let val = self._do(problem, n_samples, random_state);
        Population::new_with_attrs(&[(&IndividualField::X, Value::FloatMatrix(val))])
    }

    /// Produce the raw sample matrix of shape `(n_samples, n_var)`.
    ///
    /// Mirrors the abstract `Sampling._do(problem, n_samples, random_state)`.
    fn _do(
        &self,
        problem: &dyn Problem,
        n_samples: usize,
        random_state: Option<&mut StdRng>,
    ) -> Array2<f64>;
}
