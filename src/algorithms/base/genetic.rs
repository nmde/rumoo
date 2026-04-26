use crate::core::{
    algorithm::{Algorithm, AlgorithmBase},
    crossover::Crossover,
    duplicate::DuplicateElimination,
    initialization::Initialization,
    mating::Mating,
    mutation::Mutation,
    population::Population,
    problem::Problem,
    repair::Repair,
    sampling::Sampling,
    selection::Selection,
    survival::Survival,
};

/// Mirrors `pymoo.algorithms.base.genetic.GeneticAlgorithm`.
pub struct GeneticAlgorithm {
    pub base: AlgorithmBase,

    /// Size of the maintained population.
    pub pop_size: Option<usize>,

    /// Whether to run `survival` immediately after the initial infill is evaluated.
    pub advance_after_initial_infill: bool,

    /// Survival strategy (rank-and-crowding, reference-direction, etc.).
    pub survival: Option<Box<dyn Survival>>,

    /// Number of offspring to produce each generation. Defaults to `pop_size`.
    pub n_offsprings: Option<usize>,

    pub eliminate_duplicates: Box<dyn DuplicateElimination>,
    pub repair: Box<dyn Repair>,

    pub initialization: Initialization,
    pub mating: Mating,
}

impl GeneticAlgorithm {
    /// Mirrors `GeneticAlgorithm.__init__(...)`.
    pub fn new(
        pop_size: Option<usize>,
        sampling: Option<Box<dyn Sampling>>,
        selection: Option<Box<dyn Selection>>,
        crossover: Option<Box<dyn Crossover>>,
        mutation: Option<Box<dyn Mutation>>,
        survival: Option<Box<dyn Survival>>,
        n_offsprings: Option<usize>,
        eliminate_duplicates: Option<Box<dyn DuplicateElimination>>,
        repair: Option<Box<dyn Repair>>,
        mating: Option<Mating>,
        advance_after_initial_infill: Option<bool>,
    ) -> Self {
        // Mirrors: if n_offsprings is None: n_offsprings = pop_size
        let n_offsprings = n_offsprings.or(pop_size);

        // Mirrors: isinstance(eliminate_duplicates, bool) dispatch
        let eliminate_duplicates =
            eliminate_duplicates.unwrap_or_else(|| Box::new(DefaultDuplicateElimination::new()));

        // Mirrors: repair = repair if repair is not None else NoRepair()
        let repair = repair.unwrap_or_else(|| Box::new(NoRepair::new()));

        let initialization =
            Initialization::new(sampling, repair.as_ref(), eliminate_duplicates.as_ref());

        // Mirrors: if mating is None: mating = Mating(selection, crossover, mutation, ...)
        let mating = mating.unwrap_or_else(|| {
            Mating::new(
                selection,
                crossover,
                mutation,
                repair.as_ref(),
                eliminate_duplicates.as_ref(),
                Some(100),
            )
        });

        Self {
            base: AlgorithmBase::new(None, None, None, None, None, None, None, None, None, None),
            pop_size,
            advance_after_initial_infill: advance_after_initial_infill.unwrap_or(false),
            survival,
            n_offsprings,
            eliminate_duplicates,
            repair,
            initialization,
            mating,
        }
    }
}

impl Algorithm for GeneticAlgorithm {
    fn base(&self) -> &AlgorithmBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut AlgorithmBase {
        &mut self.base
    }

    /// Mirrors `GeneticAlgorithm._initialize_infill`:
    /// `pop = self.initialization.do(self.problem, self.pop_size, ...)`.
    fn _initialize_infill(&mut self) -> Option<Population> {
        let pop_size = self.pop_size.unwrap_or(0);
        let prob_ptr = self
            .base
            .problem
            .as_ref()
            .map(|p| p.as_ref() as *const dyn Problem)?;
        let mut rng = self.base.random_state.take();
        let pop = self
            .initialization
            .do_init(unsafe { &*prob_ptr }, pop_size, rng.as_mut());
        self.base.random_state = rng;
        Some(pop)
    }

    /// Mirrors `GeneticAlgorithm._initialize_advance`:
    /// if `advance_after_initial_infill`, run survival on the initial population.
    fn _initialize_advance(&mut self, infills: Option<&Population>) {
        if !self.advance_after_initial_infill {
            return;
        }
        let infills = match infills {
            Some(p) => p,
            None => return,
        };
        let n = infills.len();
        let prob_ptr = match self
            .base
            .problem
            .as_ref()
            .map(|p| p.as_ref() as *const dyn Problem)
        {
            Some(p) => p,
            None => return,
        };
        let mut rng = self.base.random_state.take();
        if let Some(ref mut survival) = self.survival {
            let result = survival.do_survival(unsafe { &*prob_ptr }, infills, n, rng.as_mut());
            self.base.pop = Some(result);
        }
        self.base.random_state = rng;
    }

    /// Mirrors `GeneticAlgorithm._infill`:
    /// produce offspring via mating; force-terminate if mating yields nothing.
    fn _infill(&mut self) -> Option<Population> {
        let n_offsprings = self.n_offsprings.unwrap_or(0);
        let pop = self.base.pop.clone()?;
        let prob_ptr = self
            .base
            .problem
            .as_ref()
            .map(|p| p.as_ref() as *const dyn Problem)?;
        let mut rng = self.base.random_state.take();

        let off = self
            .mating
            .do_mating(unsafe { &*prob_ptr }, &pop, n_offsprings, rng.as_mut());
        self.base.random_state = rng;

        match off {
            // Mirrors: if len(off) == 0: self.termination.force_termination = True; return
            None => {
                if let Some(ref mut t) = self.base.termination {
                    t.base_mut().force_termination = true;
                }
                None
            }
            Some(ref o) if o.is_empty() => {
                if let Some(ref mut t) = self.base.termination {
                    t.base_mut().force_termination = true;
                }
                None
            }
            Some(o) => {
                // Mirrors: elif len(off) < self.n_offsprings: print("WARNING: ...")
                if o.len() < n_offsprings && self.base.verbose {
                    eprintln!(
                        "WARNING: Mating could not produce the required number of (unique) offsprings!"
                    );
                }
                Some(o)
            }
        }
    }

    /// Mirrors `GeneticAlgorithm._advance`:
    /// merge population with offspring, then run survival.
    fn _advance(&mut self, infills: Option<&Population>) -> Option<bool> {
        let pop = self
            .base
            .pop
            .clone()
            .unwrap_or_else(|| Population::empty(0));

        // Mirrors: if infills is not None: pop = Population.merge(self.pop, infills)
        let merged = match infills {
            Some(offs) => Population::merge(pop, *offs),
            None => pop,
        };

        let n_survive = self.pop_size.unwrap_or(0);
        let prob_ptr = match self
            .base
            .problem
            .as_ref()
            .map(|p| p.as_ref() as *const dyn Problem)
        {
            Some(p) => p,
            None => return None,
        };
        let mut rng = self.base.random_state.take();

        if let Some(ref mut survival) = self.survival {
            let result =
                survival.do_survival(unsafe { &*prob_ptr }, &merged, n_survive, rng.as_mut());
            self.base.pop = Some(result);
        }
        self.base.random_state = rng;

        None
    }
}
