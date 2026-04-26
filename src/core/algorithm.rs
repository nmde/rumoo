use std::{collections::HashMap, time::Instant};

use anyhow::{Result, anyhow};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    core::{
        callback::{Callback, CallbackCollection},
        evaluator::Evaluator,
        individual::{IndividualField, Value},
        population::Population,
        problem::Problem,
        result::AlgorithmResult,
        termination::Termination,
    },
    termination::default::{DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination},
    util::{
        display::{display::Display, output::Output},
        optimum::filter_optimum,
    },
};

/// Instance state shared by all `Algorithm` implementations.
///
/// Mirrors the instance attributes of `pymoo.core.algorithm.Algorithm.__init__`.
pub struct AlgorithmBase {
    pub problem: Option<Box<dyn Problem>>,
    pub termination: Option<Box<dyn Termination>>,
    pub output: Option<Box<dyn Output>>,
    pub archive: Option<Population>,
    pub display: Option<Box<dyn Display>>,
    pub callback: Box<dyn Callback>,
    pub return_least_infeasible: bool,
    pub save_history: bool,
    pub verbose: bool,
    pub seed: Option<u64>,
    pub random_state: Option<StdRng>,
    pub evaluator: Box<dyn Evaluator>,
    pub history: Vec<Box<dyn Algorithm>>,
    pub pop: Option<Population>,
    pub off: Option<Population>,
    pub opt: Option<Population>,
    pub n_iter: Option<usize>,
    pub data: HashMap<String, Value>,
    pub is_initialized: bool,
    pub start_time: Option<Instant>,
}

impl AlgorithmBase {
    /// Mirrors `Algorithm.__init__` defaults.
    pub fn new(
        termination: Option<Box<dyn Termination>>,
        output: Option<Box<dyn Output>>,
        display: Option<Box<dyn Display>>,
        callback: Option<Box<dyn Callback>>,
        archive: Option<Population>,
        return_least_infeasible: Option<bool>,
        save_history: Option<bool>,
        verbose: Option<bool>,
        seed: Option<u64>,
        evaluator: Option<Box<dyn Evaluator>>,
    ) -> Self {
        Self {
            problem: None,
            termination,
            output,
            archive,
            display,
            callback: callback.unwrap_or(Box::new(CallbackCollection::new(vec![]))),
            return_least_infeasible: return_least_infeasible.unwrap_or(false),
            save_history: save_history.unwrap_or(false),
            verbose: verbose.unwrap_or(false),
            seed,
            random_state: None,
            evaluator: evaluator.unwrap_or(Box::new(Evaluator::new())),
            history: Vec::new(),
            pop: None,
            off: None,
            opt: None,
            n_iter: None,
            data: HashMap::new(),
            is_initialized: false,
            start_time: None,
        }
    }
}

/// Base trait for all optimization algorithms.
///
/// Mirrors `pymoo.core.algorithm.Algorithm`.
pub trait Algorithm {
    fn base(&self) -> &AlgorithmBase;
    fn base_mut(&mut self) -> &mut AlgorithmBase;

    /// Mirrors `Algorithm.setup(problem, verbose, progress, **kwargs)`.
    fn setup(&mut self, problem: Box<dyn Problem>, verbose: Option<bool>, progress: Option<bool>) {
        let verbose = verbose.unwrap_or(false);
        let progress = progress.unwrap_or(false);

        self.base_mut().problem = Some(problem);

        let seed = self.base().seed;
        self.base_mut().random_state = Some(match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        });

        if self.base().termination.is_none() {
            let n_obj = self.base().problem.as_ref().map_or(0, |p| p.n_obj());
            self.base_mut().termination = Some(default_termination(n_obj));
        } else {
            let term = self.base_mut().termination.take();
            self.base_mut().termination = term;
        }

        if self.base().display.is_none() {
            let output = self.base_mut().output.take();
            self.base_mut().display = Some(Box::new(Display::new(output, verbose, progress)));
        }

        self._setup();
    }

    /// Mirrors `Algorithm.run()`.
    fn run(&mut self) -> AlgorithmResult {
        while self.has_next() {
            self.next()?;
        }
        self.result()
    }

    /// Mirrors `Algorithm.has_next()`.
    fn has_next(&self) -> bool {
        self.base()
            .termination
            .as_ref()
            .map_or(false, |t| !t.has_terminated())
    }

    /// Mirrors `Algorithm.finalize()`.
    fn finalize(&mut self) {
        let mut display = self.base_mut().display.take();
        if let Some(ref mut d) = display {
            d.finalize();
        }
        self.base_mut().display = display;
        self._finalize();
    }

    /// Mirrors `Algorithm.next()`.
    fn next(&mut self) -> Result<()> {
        let infills = self.infill()?;
        if let Some(mut infills) = infills {
            // Take the evaluator out so we can pass `&mut self` to eval() while
            // the problem pointer is held as a raw pointer to avoid a double-borrow.
            let mut evaluator = self.base_mut().evaluator.take();
            let prob = self
                .base()
                .problem
                .as_ref()
                .map(|p| p.as_ref() as *const dyn Problem);
            if let Some(prob_ptr) = prob {
                evaluator.eval(
                    unsafe { &*prob_ptr },
                    &mut infills,
                    self as &mut dyn Algorithm,
                );
            }
            self.base_mut().evaluator = evaluator;
            self.advance(Some(infills));
        } else {
            self.advance(None);
        }
        Ok(())
    }

    /// Mirrors `Algorithm._initialize()`.
    fn _initialize(&mut self) {
        self.base_mut().start_time = Some(Instant::now());
        self.base_mut().n_iter = Some(1);
        self.base_mut().pop = Some(Population::empty(0));
        self.base_mut().opt = None;
    }

    /// Mirrors `Algorithm.infill()`.
    fn infill(&mut self) -> Result<Option<Population>> {
        if self.base().problem.is_none() {
            return Err(anyhow!(
                "Please call `setup(problem)` before calling next()."
            ));
        }

        let mut infills = if !self.base().is_initialized {
            self._initialize();
            self._initialize_infill()
        } else {
            self._infill()
        };

        if let Some(ref mut pop) = infills {
            let n_iter = self.base().n_iter.unwrap_or(1);
            pop.set(
                &IndividualField::DataField("n_gen".to_string()),
                Value::Int(n_iter as i64),
            );
            pop.set(
                &IndividualField::DataField("n_iter".to_string()),
                Value::Int(n_iter as i64),
            );
        }

        Ok(infills)
    }

    /// Mirrors `Algorithm.advance(infills, **kwargs)`.
    fn advance(&mut self, infills: Option<Population>) -> Option<Population> {
        self.base_mut().off = infills.clone();

        if !self.base().is_initialized {
            self.base_mut().n_iter = Some(1);
            self.base_mut().pop = infills.clone();
            self._initialize_advance(infills.as_ref());
            self.base_mut().is_initialized = true;
            self._post_advance();
        } else {
            let should_advance = self._advance(infills.as_ref()).unwrap_or(true);
            if should_advance {
                self._post_advance();
            }
        }

        let terminated = self
            .base()
            .termination
            .as_ref()
            .map_or(false, |t| t.has_terminated());

        let ret = if terminated {
            self.finalize();
            self.result().opt
        } else {
            self.base().opt.clone()
        };

        // Mirrors: if self.archive is not None and infills is not None:
        //              self.archive = self.archive.add(infills)
        let off = self.base().off.clone();
        if let Some(ref inf) = off {
            if let Some(ref mut arch) = self.base_mut().archive {
                arch.add(inf);
            }
        }

        ret
    }

    /// Mirrors `Algorithm.result()`.
    fn result(&self) -> AlgorithmResult {
        let mut res = AlgorithmResult::new();

        res.start_time = self.base().start_time;
        res.end_time = Some(Instant::now());
        res.exec_time = res.end_time - res.start_time;

        res.pop = self.base().pop.clone();
        res.archive = self.base().archive.clone();
        res.data = self.base().data.clone();

        // Mirrors: opt = self.opt; handle None / empty / infeasible cases.
        let opt = match self.base().opt.clone() {
            None => None,
            Some(ref o) if o.is_empty() => None,
            Some(ref o) if !o.any_feasible() => {
                if self.base().return_least_infeasible && self.base().opt.is_some() {
                    self.base()
                        .opt
                        .as_ref()
                        .unwrap()
                        .map(|o| filter_optimum(o, Some(true)))
                } else {
                    None
                }
            }
            Some(o) => Some(o),
        };
        res.opt = opt;

        // Mirrors: X, F, CV, G, H = self.opt.get("X", "F", "CV", "G", "H")
        if let Some(ref opt_pop) = opt {
            res.X = opt_pop.get(&IndividualField::X);
            res.F = opt_pop.get(&IndividualField::F);
            res.CV = opt_pop.get(&IndividualField::CV);
            res.G = opt_pop.get(&IndividualField::G);
            res.H = opt_pop.get(&IndividualField::H);
            // Mirrors: if n_obj == 1 and len(X) == 1: X, F, … = X[0], F[0], …
            // Concrete result types should squeeze single-solution single-objective
            // arrays to 1-D as needed.
        }

        res.problem = self.base().problem;
        res.history = self.base().history;

        res
    }

    /// Mirrors `Algorithm.ask()`.
    fn ask(&mut self) -> Result<Option<Population>> {
        self.infill()
    }

    /// Mirrors `Algorithm.tell(*args, **kwargs)`.
    fn tell(&mut self, infills: Option<Population>) -> Option<Population> {
        self.advance(infills)
    }

    /// Mirrors `Algorithm._set_optimum()`.
    fn _set_optimum(&mut self) {
        let pop = self.base().pop.clone();
        if let Some(ref p) = pop {
            self.base_mut().opt = filter_optimum(p, Some(true));
        }
    }

    /// Mirrors `Algorithm._post_advance()`.
    fn _post_advance(&mut self) {
        self._set_optimum();

        // termination.update(self)
        let mut term = self.base_mut().termination.take();
        if let Some(ref mut t) = term {
            t.update(self as &mut dyn Algorithm);
        }
        self.base_mut().termination = term;

        // display(self)
        let mut display = self.base_mut().display.take();
        if let Some(ref mut d) = display {
            d.call(self as &dyn Algorithm);
        }
        self.base_mut().display = display;

        self.base_mut().callback.call(self as &mut dyn Algorithm);

        self.base_mut().n_iter += 1;
    }

    fn _setup(&mut self) {}
    fn _initialize_infill(&mut self) -> Option<Population> {
        None
    }
    fn _initialize_advance(&mut self, _infills: Option<&Population>) {}
    fn _infill(&mut self) -> Option<Population> {
        None
    }
    fn _advance(&mut self, _infills: Option<&Population>) -> Option<bool> {
        None
    }
    fn _finalize(&mut self) {}

    /// Mirrors `Algorithm.n_gen` property getter (alias for `n_iter`).
    fn n_gen(&self) -> usize {
        self.base().n_iter.unwrap_or(0)
    }

    /// Mirrors `Algorithm.n_gen` property setter.
    fn set_n_gen(&mut self, value: usize) {
        self.base_mut().n_iter = Some(value);
    }
}

/// State fields for loop-wise algorithms.
///
/// Mirrors `pymoo.core.algorithm.LoopwiseAlgorithm` instance attributes.
pub struct LoopwiseBase {
    pub base: AlgorithmBase,
    /// Mirrors `LoopwiseAlgorithm.state` — the infill population currently
    /// waiting to be evaluated, or `None` when no step is in progress.
    pub state: Option<Population>,
}

impl LoopwiseBase {
    pub fn new() -> Self {
        Self {
            base: AlgorithmBase::new(None, None, None, None, None, None, None, None, None, None),
            state: None,
        }
    }
}

/// Extension trait for generator-style algorithms.
///
/// Mirrors `pymoo.core.algorithm.LoopwiseAlgorithm(Algorithm)`.
///
/// Python uses coroutine generators (`yield` / `generator.send(value)`) to
/// interleave infill requests with evaluations.  In Rust that pattern is
/// replaced by an explicit `state` field and a `_next` method that either
/// populates `state` with the next infill request or sets it to `None` when
/// the loop is exhausted (mirrors `StopIteration`).
///
/// Concrete implementations should wire `Algorithm::_infill` and
/// `Algorithm::_advance` to the helpers below:
///
/// ```rust
/// fn _infill(&mut self) -> Option<Population> { self._infill_lw() }
/// fn _advance(&mut self, infills: Option<&Population>) -> Option<bool> {
///     self._advance_lw(infills.cloned())
/// }
/// ```
pub trait LoopwiseAlgorithm: Algorithm {
    fn lw_base(&self) -> &LoopwiseBase;
    fn lw_base_mut(&mut self) -> &mut LoopwiseBase;

    /// Mirrors `LoopwiseAlgorithm._next()` generator.
    ///
    /// Drive one step of the inner loop.  When `infills` is `None` the loop
    /// starts (or restarts); when it carries a `Population` the step continues
    /// after evaluation.  Implementors should set
    /// `self.lw_base_mut().state = Some(next_infills)` to yield a request, or
    /// `None` to signal completion.
    fn _next(&mut self, infills: Option<Population>);

    /// Helper for `Algorithm::_infill` — mirrors `LoopwiseAlgorithm._infill`.
    fn _infill_lw(&mut self) -> Option<Population> {
        if self.lw_base().state.is_none() {
            self._next(None);
        }
        self.lw_base().state.clone()
    }

    /// Helper for `Algorithm::_advance` — mirrors `LoopwiseAlgorithm._advance`.
    fn _advance_lw(&mut self, infills: Option<Population>) -> Option<bool> {
        self._next(infills);
        if self.lw_base().state.is_none() {
            Some(true) // loop exhausted — mirrors StopIteration returning True
        } else {
            Some(false)
        }
    }
}

/// Selects the default termination criterion based on the number of objectives.
///
/// Mirrors `pymoo.core.algorithm.default_termination(problem)`.
pub fn default_termination(n_obj: usize) -> Box<dyn Termination> {
    if n_obj > 1 {
        Box::new(DefaultMultiObjectiveTermination::new(
            None, None, None, None, None, None, None,
        ))
    } else {
        Box::new(DefaultSingleObjectiveTermination::new(
            None, None, None, None, None, None,
        ))
    }
}

/// A transparent algorithm wrapper that delegates all calls to an inner algorithm.
///
/// Mirrors `pymoo.core.algorithm.MetaAlgorithm(Meta)`.
pub struct MetaAlgorithm {
    pub inner: Box<dyn Algorithm>,
    pub extra: HashMap<String, Value>,
}

impl MetaAlgorithm {
    /// Mirrors `MetaAlgorithm.__init__(algorithm, copy=True, **kwargs)`.
    pub fn new(algorithm: Box<dyn Algorithm>) -> Self {
        Self {
            inner: algorithm,
            extra: HashMap::new(),
        }
    }
}

impl Algorithm for MetaAlgorithm {
    fn base(&self) -> &AlgorithmBase {
        self.inner.base()
    }

    fn base_mut(&mut self) -> &mut AlgorithmBase {
        self.inner.base_mut()
    }
}
