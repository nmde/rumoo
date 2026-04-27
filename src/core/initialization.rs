/*
from pymoo.core.population import Population
from pymoo.util import default_random_state
from pymoo.util.misc import at_least_2d_array
*/

use crate::core::{
    duplicate::{
        DefaultDuplicateElimination, DuplicateElimination, EliminateDuplicates,
        NoDuplicateElimination,
    },
    repair::{NoRepair, Repair},
    sampling::Sampling,
};

pub struct Initialization {
    sampling: Option<Box<dyn Sampling>>,
    eliminate_duplicates: Box<dyn DuplicateElimination>,
    repair: Box<dyn Repair>,
}

impl Initialization {
    pub fn new(
        sampling: Option<Box<dyn Sampling>>,
        repair: Option<&dyn Repair>,
        eliminate_duplicates: EliminateDuplicates,
    ) -> Self {
        Self {
            sampling,
            eliminate_duplicates: match eliminate_duplicates {
                EliminateDuplicates::None => Box::new(NoDuplicateElimination::new()),
                EliminateDuplicates::Bool(v) => {
                    if v {
                        Box::new(DefaultDuplicateElimination::default())
                    } else {
                        Box::new(NoDuplicateElimination::new())
                    }
                }
                EliminateDuplicates::Eliminator(v) => v,
            },
            repair: if repair.is_some() { Box::new(*repair.unwrap()) } else { Box::new(NoRepair::new()) },
        }
    }
    /*
        @default_random_state
        def do(self, problem, n_samples, random_state=None, **kwargs):

            # provide a whole population object - (individuals might be already evaluated)
            if isinstance(self.sampling, Population):
                pop = self.sampling

            else:
                if isinstance(self.sampling, np.ndarray):
                    sampling = at_least_2d_array(self.sampling)
                    pop = Population.new(X=sampling)
                else:
                    pop = self.sampling(problem, n_samples, random_state=random_state, **kwargs)

            # repair all solutions that are not already evaluated
            not_eval_yet = [k for k in range(len(pop)) if len(pop[k].evaluated) == 0]
            if len(not_eval_yet) > 0:
                pop[not_eval_yet] = self.repair(problem, pop[not_eval_yet], random_state=random_state, **kwargs)

            # filter duplicate in the population
            pop = self.eliminate_duplicates.do(pop)

            return pop
    */
}
