use ndarray::Array2;
use rand::rngs::StdRng;

use crate::{
    core::{
        crossover::Crossover,
        mutation::Mutation,
        population::Population,
        problem::Problem,
        selection::{Selection, SelectionResult},
    },
    util::default_random_state,
};

/// Mirrors `pymoo.core.mating.Mating`.
pub struct Mating {
    pub selection: Box<dyn Selection>,
    pub crossover: Box<dyn Crossover>,
    pub mutation: Box<dyn Mutation>,
}

impl Mating {
    /// Mirrors `Mating.__init__(selection, crossover, mutation)`.
    pub fn new(
        selection: Box<dyn Selection>,
        crossover: Box<dyn Crossover>,
        mutation: Box<dyn Mutation>,
    ) -> Self {
        Self { selection, crossover, mutation }
    }

    /// Mirrors `Mating._do(problem, pop, n_offsprings, parents, random_state)`.
    pub fn do_mating(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        n_offsprings: usize,
        parents: Option<Array2<usize>>,
        random_state: Option<&mut StdRng>,
    ) -> Population {
        let mut fallback = default_random_state();
        let rng = random_state.unwrap_or(&mut fallback);

        // how many parents need to be select for the mating - depending on number of offsprings remaining
        let n_matings = (n_offsprings as f64
            / self.crossover.crossover_base().n_offsprings as f64)
            .ceil() as usize;

        // if the parents for the mating are not provided directly - usually selection will be used
        let parents = match parents {
            Some(p) => p,
            None => {
                let n_parents = self.crossover.crossover_base().n_parents;
                // select the parents for the mating - just an index array
                match self.selection.do_selection(
                    problem,
                    pop,
                    n_matings,
                    n_parents,
                    Some(false),
                    Some(&mut *rng),
                ) {
                    SelectionResult::Indices(idx) => idx.mapv(|v| v as usize),
                    // fallback: should not occur when to_pop=false
                    SelectionResult::Population(_) => Array2::zeros((n_matings, n_parents)),
                }
            }
        };

        // do the crossover using the parents index and the population - additional data provided if necessary
        let mut off = self
            .crossover
            .do_crossover(problem, pop, Some(&parents), Some(&mut *rng))
            .unwrap_or_else(|_| Population::empty(0));

        // do the mutation on the offsprings created through crossover
        off = self.mutation.do_mutation(problem, &mut off, None, Some(rng));

        off
    }
}
