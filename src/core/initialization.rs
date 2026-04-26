use rand::rngs::StdRng;

use ndarray::Array2;

use crate::{
    core::{
        duplicate::{DuplicateElimination, DuplicateResult, NoDuplicateElimination},
        individual::{IndividualField, Value},
        population::Population,
        problem::Problem,
        repair::{NoRepair, Repair},
        sampling::Sampling,
    },
    util::default_random_state,
};

// -------------------------------------------------------------------------------------------------
// SamplingInput
// -------------------------------------------------------------------------------------------------

/// The source of the initial population.
///
/// Mirrors the three dispatch cases inside `Initialization.do`:
/// - `Pop`     — an already-constructed `Population` (passed through unchanged)
/// - `Matrix`  — a raw X matrix wrapped into a `Population` via `Population.new(X=...)`
/// - `Sampler` — a `Sampling` object that generates individuals
pub enum SamplingInput {
    Pop(Population),
    Matrix(Array2<f64>),
    Sampler(Box<dyn Sampling>),
}

// -------------------------------------------------------------------------------------------------
// Initialization
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.core.initialization.Initialization`.
pub struct Initialization {
    pub sampling: SamplingInput,
    pub repair: Box<dyn Repair>,
    pub eliminate_duplicates: Box<dyn DuplicateElimination>,
}

impl Initialization {
    /// Mirrors `Initialization.__init__(sampling, repair, eliminate_duplicates)`.
    pub fn new(
        sampling: SamplingInput,
        repair: Option<Box<dyn Repair>>,
        eliminate_duplicates: Option<Box<dyn DuplicateElimination>>,
    ) -> Self {
        Self {
            sampling,
            repair: repair.unwrap_or_else(|| Box::new(NoRepair::new())),
            eliminate_duplicates: eliminate_duplicates
                .unwrap_or_else(|| Box::new(NoDuplicateElimination)),
        }
    }

    /// Mirrors `Initialization.do(problem, n_samples, random_state)`.
    pub fn do_init(
        &self,
        problem: &dyn Problem,
        n_samples: usize,
        random_state: Option<&mut StdRng>,
    ) -> Population {
        let mut fallback = default_random_state();
        let rng = random_state.unwrap_or(&mut fallback);

        // provide a whole population object - (individuals might be already evaluated)
        let mut pop = match &self.sampling {
            SamplingInput::Pop(stored) => {
                // mirrors: if isinstance(self.sampling, Population): pop = self.sampling
                stored.select(&(0..stored.len()).collect::<Vec<_>>())
            }
            SamplingInput::Matrix(arr) => {
                // mirrors: isinstance(np.ndarray): pop = Population.new(X=at_least_2d_array(self.sampling))
                Population::new_with_attrs(&[(
                    &IndividualField::X,
                    Value::FloatMatrix(arr.clone()),
                )])
            }
            SamplingInput::Sampler(sampler) => {
                // mirrors: pop = self.sampling(problem, n_samples, random_state=random_state)
                sampler.do_sampling(problem, n_samples, Some(rng))
            }
        };

        // repair all solutions that are not already evaluated
        // not_eval_yet = [k for k in range(len(pop)) if len(pop[k].evaluated) == 0]
        let not_eval_yet: Vec<usize> = (0..pop.len())
            .filter(|&k| pop[k].evaluated.is_empty())
            .collect();

        if !not_eval_yet.is_empty() {
            // pop[not_eval_yet] = self.repair(problem, pop[not_eval_yet], ...)
            let sub = pop.select(&not_eval_yet);
            let repaired = self.repair.do_repair(problem, &sub);

            // write repaired X back into pop at the original indices
            for (i, &orig_idx) in not_eval_yet.iter().enumerate() {
                pop[orig_idx].x = repaired[i].x.clone();
            }
        }

        // filter duplicate in the population
        // pop = self.eliminate_duplicates.do(pop)
        match self
            .eliminate_duplicates
            .do_elimination(&pop, &[], false, true)
        {
            DuplicateResult::Filtered(p) => p,
            DuplicateResult::WithIndices { pop: p, .. } => p,
        }
    }
}
