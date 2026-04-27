use crate::core::{
    crossover::Crossover, duplicate::EliminateDuplicates, infill::InfillCriterion,
    mutation::Mutation, repair::Repair, selection::Selection,
};

pub struct Mating {
    infill: InfillCriterion,
    selection: Option<Box<dyn Selection>>,
    crossover: Option<Box<dyn Crossover>>,
    mutation: Option<Box<dyn Mutation>>,
}

impl Mating {
    pub fn new(
        selection: Option<Box<dyn Selection>>,
        crossover: Option<Box<dyn Crossover>>,
        mutation: Option<Box<dyn Mutation>>,
        repair: Option<&dyn Repair>,
        eliminate_duplicates: EliminateDuplicates,
        n_max_iterations: Option<usize>,
    ) -> Self {
        Self {
            infill: InfillCriterion::new(repair, eliminate_duplicates, n_max_iterations),
            selection,
            crossover,
            mutation,
        }
    }
    /*
        def _do(self, problem, pop, n_offsprings, parents=None, random_state=None, **kwargs):

            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

            # if the parents for the mating are not provided directly - usually selection will be used
            if parents is None:

                # select the parents for the mating - just an index array
                parents = self.selection(problem, pop, n_matings, n_parents=self.crossover.n_parents, random_state=random_state, **kwargs)

            # do the crossover using the parents index and the population - additional data provided if necessary
            off = self.crossover(problem, parents, random_state=random_state, **kwargs)

            # do the mutation on the offsprings created through crossover
            off = self.mutation(problem, off, random_state=random_state, **kwargs)

            return off
    */
}
