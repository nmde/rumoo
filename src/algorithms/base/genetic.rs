use crate::{
    core::{
        algorithm::{Algorithm, AlgorithmBase},
        crossover::Crossover,
        duplicate::{
            DefaultDuplicateElimination, DuplicateElimination, EliminateDuplicates,
            NoDuplicateElimination,
        },
        initialization::Initialization,
        mating::Mating,
        mutation::Mutation,
        population::Population,
        repair::{NoRepair, Repair},
        sampling::Sampling,
        selection::Selection,
        survival::Survival,
    },
    util::display::output::Output,
};

pub struct GeneticAlgorithm {
    base: AlgorithmBase,
    pop_size: Option<usize>,
    advance_after_initial_infill: bool,
    survival: Option<Box<dyn Survival>>,
    n_offsprings: Option<usize>,
    eliminate_duplicates: Box<dyn DuplicateElimination>,
    repair: Box<dyn Repair>,
    initialization: Initialization,
    mating: Mating,
    n_gen: Option<usize>,
    pub pop: Option<Population>,
    off: Option<Population>,
}

impl GeneticAlgorithm {
    pub fn new(
        pop_size: Option<usize>,
        sampling: Option<Box<dyn Sampling>>,
        selection: Option<Box<dyn Selection>>,
        crossover: Option<Box<dyn Crossover>>,
        mutation: Option<Box<dyn Mutation>>,
        survival: Option<Box<dyn Survival>>,
        n_offsprings: Option<usize>,
        eliminate_duplicates: EliminateDuplicates,
        repair: Option<Box<dyn Repair>>,
        mating: Option<Mating>,
        advance_after_initial_infill: Option<bool>,
        output: Option<Box<dyn Output>>,
    ) -> Self {
        let repair = repair.unwrap_or(Box::new(NoRepair::new()));
        let eliminate_duplicates: Box<dyn DuplicateElimination> = match eliminate_duplicates {
            EliminateDuplicates::None => Box::new(DefaultDuplicateElimination::default()),
            EliminateDuplicates::Bool(v) => {
                if v {
                    Box::new(DefaultDuplicateElimination::default())
                } else {
                    Box::new(NoDuplicateElimination::new())
                }
            }
            EliminateDuplicates::Eliminator(v) => v,
        };
        let initialization = Initialization::new(
            sampling,
            Some(repair.as_ref()),
            EliminateDuplicates::Eliminator(eliminate_duplicates.as_ref()),
        );
        let mating = mating.unwrap_or(Mating::new(
            selection,
            crossover,
            mutation,
            Some(repair.as_ref()),
            EliminateDuplicates::Eliminator(eliminate_duplicates.as_ref()),
            Some(100),
        ));
        Self {
            base: AlgorithmBase::new(None, output, None, None, None, None, None, None, None, None),
            // the population size used
            pop_size,
            // whether the algorithm should be advanced after initialization of not
            advance_after_initial_infill: advance_after_initial_infill.unwrap_or(false),
            // the survival for the genetic algorithm
            survival,
            // number of offsprings to generate through recombination
            // if the number of offspring is not set - equal to population size
            n_offsprings: if n_offsprings.is_some() {
                Some(n_offsprings.unwrap())
            } else {
                pop_size
            },
            // set the duplicate detection class - a boolean value chooses the default duplicate detection
            eliminate_duplicates,
            // simply set the no repair object if it is None
            repair,
            initialization,
            mating,
            // other run specific data updated whenever solve is called - to share them in all algorithms
            n_gen: None,
            pop: None,
            off: None,
        }
    }
    /*
        def _initialize_infill(self):
            pop = self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)
            return pop

        def _initialize_advance(self, infills=None, **kwargs):
            if self.advance_after_initial_infill:
                self.pop = self.survival.do(self.problem, infills, n_survive=len(infills),
                                            random_state=self.random_state, algorithm=self, **kwargs)

        def _infill(self):

            # do the mating using the current population
            off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self, random_state=self.random_state)

            # if the mating could not generate any new offspring (duplicate elimination might make that happen)
            if len(off) == 0:
                self.termination.force_termination = True
                return

            # if not the desired number of offspring could be created
            elif len(off) < self.n_offsprings:
                if self.verbose:
                    print("WARNING: Mating could not produce the required number of (unique) offsprings!")

            return off

        def _advance(self, infills=None, **kwargs):

            # the current population
            pop = self.pop

            # merge the offsprings with the current population
            if infills is not None:
                pop = Population.merge(self.pop, infills)

            # execute the survival to find the fittest solutions
            self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, random_state=self.random_state, **kwargs)
    */
}

impl Algorithm for GeneticAlgorithm {
    fn base(&self) -> &AlgorithmBase {
        &self.base
    }
}
