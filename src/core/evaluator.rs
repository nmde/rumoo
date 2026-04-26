use ndarray::Array1;

use crate::core::{
    algorithm::Algorithm,
    callback::Callback,
    individual::{IndividualField, Value},
    population::Population,
    problem::Problem,
};

/// Instance fields shared by all `Evaluator` implementations.
///
/// Mirrors the `__init__` attributes of `pymoo.core.evaluator.Evaluator`.
pub struct EvaluatorBase {
    pub evaluate_values_of: Vec<IndividualField>,
    pub skip_already_evaluated: bool,
    pub callback: Option<Box<dyn Callback>>,
    pub n_eval: usize,
}

impl EvaluatorBase {
    /// Mirrors `Evaluator.__init__(skip_already_evaluated, evaluate_values_of, callback)`.
    pub fn new(
        skip_already_evaluated: Option<bool>,
        evaluate_values_of: Option<Vec<IndividualField>>,
        callback: Option<Box<dyn Callback>>,
    ) -> Self {
        Self {
            evaluate_values_of: evaluate_values_of.unwrap_or_else(|| {
                vec![IndividualField::F, IndividualField::G, IndividualField::H]
            }),
            skip_already_evaluated: skip_already_evaluated.unwrap_or(true),
            callback,
            n_eval: 0,
        }
    }
}

/// Glues a `Problem` to a `Population`, tracking function-evaluation counts.
///
/// Mirrors `pymoo.core.evaluator.Evaluator`.
pub trait Evaluator {
    fn base(&self) -> &EvaluatorBase;
    fn base_mut(&mut self) -> &mut EvaluatorBase;

    /// Current total number of objective-function evaluations.
    ///
    /// Mirrors `Evaluator.n_eval`.
    fn n_eval(&self) -> usize {
        self.base().n_eval
    }

    /// Evaluate the population, optionally skipping already-evaluated individuals.
    ///
    /// Mirrors `Evaluator.eval(problem, pop, skip_already_evaluated, evaluate_values_of,
    /// count_evals, **kwargs)`.
    fn eval(
        &mut self,
        problem: &dyn Problem,
        pop: &mut Population,
        skip_already_evaluated: Option<bool>,
        evaluate_values_of: Option<&[IndividualField]>,
        count_evals: Option<bool>,
    ) -> &mut Population {
        let count_evals = count_evals.unwrap_or(true);
        let evaluate_values_of = evaluate_values_of.unwrap_or(self.base().evaluate_values_of.clone());
        let skip_already_evaluated = skip_already_evaluated.unwrap_or(self.base().skip_already_evaluated);

        // Mirrors: I = [i for i, ind in enumerate(pop)
        //               if not all(e in ind.evaluated for e in evaluate_values_of)]
        let indices: Vec<usize> = if skip_already_evaluated {
            (0..pop.len())
                .filter(|&i| {
                    !evaluate_values_of
                        .iter()
                        .all(|field| pop[i].evaluated.contains(field))
                })
                .collect()
        } else {
            (0..pop.len()).collect()
        };

        if !indices.is_empty() {
            // Mirrors: self._eval(problem, pop[I], evaluate_values_of)
            let mut sub_pop = pop.select(&indices);
            self._eval(problem, &mut sub_pop, &evaluate_values_of);
            // Write evaluated individuals back into the original population.
            for (sub_i, &orig_i) in indices.iter().enumerate() {
                pop[orig_i] = sub_pop[sub_i].clone();
            }
        }

        // Mirrors: if count_evals: self.n_eval += len(I)
        if count_evals {
            self.base_mut().n_eval += indices.len();
        }

        // Mirrors: if self.callback: self.callback(pop)
        if let Some(ref cb) = self.base().callback {
            cb(pop);
        }

        pop
    }

    /// Perform the actual problem evaluation and write outputs back into `pop`.
    ///
    /// Mirrors `Evaluator._eval(problem, pop, evaluate_values_of, **kwargs)`.
    fn _eval(
        &mut self,
        problem: &dyn Problem,
        pop: &mut Population,
        evaluate_values_of: &[IndividualField],
    ) {
        // Mirrors: X = pop.get("X")
        let x = pop.get(&IndividualField::X);

        // Mirrors: out = problem.evaluate(X, return_values_of=…, return_as_dictionary=True)
        let out = problem.evaluate(x, Some(evaluate_values_of.to_vec()), None)?;

        // Mirrors: for key, val in out.items(): if val is not None: pop.set(key, val)
        for (key, val) in out {
            if let Some(v) = val {
                pop.set(&key, v);
            }
        }

        // Mirrors: pop.apply(lambda ind: ind.evaluated.update(out.keys()))
        for ind in pop.iter_mut() {
            for field in evaluate_values_of {
                ind.evaluated.insert(field.clone());
            }
        }
    }
}

/// Default evaluator — uses all `Evaluator` trait defaults.
///
/// Mirrors `pymoo.core.evaluator.Evaluator` used directly (not subclassed).
pub struct DefaultEvaluator {
    pub base: EvaluatorBase,
}

impl DefaultEvaluator {
    pub fn new() -> Self {
        Self {
            base: EvaluatorBase::new(None, None, None),
        }
    }
}

impl Evaluator for DefaultEvaluator {
    fn base(&self) -> &EvaluatorBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut EvaluatorBase {
        &mut self.base
    }
}

/// Fills all unevaluated individuals with a constant value without calling
/// the problem's evaluation function.
///
/// Mirrors `pymoo.core.evaluator.VoidEvaluator(Evaluator)`.
pub struct VoidEvaluator {
    pub base: EvaluatorBase,
    /// The fill value; `None` means do nothing.
    ///
    /// Mirrors `VoidEvaluator.value` (default `np.inf`).
    pub value: Option<f64>,
}

impl VoidEvaluator {
    /// Mirrors `VoidEvaluator.__init__(value=np.inf, **kwargs)`.
    pub fn new(value: Option<f64>) -> Self {
        Self {
            base: EvaluatorBase::new(None, None, None),
            value: Some(value.unwrap_or(f64::INFINITY)),
        }
    }
}

impl Evaluator for VoidEvaluator {
    fn base(&self) -> &EvaluatorBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut EvaluatorBase {
        &mut self.base
    }

    /// Mirrors `VoidEvaluator.eval(problem, pop, **kwargs)`.
    fn eval(
        &mut self,
        problem: &dyn Problem,
        pop: &mut Population,
        _algorithm: &mut dyn Algorithm,
    ) {
        let val = match self.value {
            None => return,
            Some(v) => v,
        };

        let n_obj = problem.n_obj();
        let n_ieq = problem.n_ieq_constr();
        let n_eq = problem.n_eq_constr();

        for ind in pop.iter_mut() {
            // Mirrors: if len(individual.evaluated) == 0:
            if ind.evaluated.is_empty() {
                ind.f = Some(Array1::from_elem(n_obj, val));

                // Mirrors: np.full(n_ieq_constr, val) if n_ieq_constr > 0 else None
                ind.g = if n_ieq > 0 {
                    Some(Array1::from_elem(n_ieq, val))
                } else {
                    None
                };

                // Mirrors: np.full(n_eq_constr, val) if n_eq_constr else None
                ind.h = if n_eq > 0 {
                    Some(Array1::from_elem(n_eq, val))
                } else {
                    None
                };

                // Mirrors: individual.CV = [-np.inf]
                ind.set(
                    &IndividualField::CV,
                    Value::FloatArray(Array1::from_elem(1, f64::NEG_INFINITY)),
                );

                // Mirrors: individual.feas = [False]
                ind.set(
                    &IndividualField::Feas,
                    Value::BoolArray(Array1::from_elem(1, false)),
                );
            }
        }
    }
}
