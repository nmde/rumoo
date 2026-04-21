use std::cmp::Ordering;

use rand::rngs::StdRng;

use crate::{
    core::{
        individual::{IndividualField, Value},
        population::{Population, merge},
        problem::Problem,
    },
    util::default_random_state,
};

/// Return type of `Survival::do_survival`, chosen by the `return_indices` flag.
///
/// Mirrors the two return paths of `Survival.do`:
/// - `return_indices=False` → `Survivors`
/// - `return_indices=True`  → `Indices`
pub enum SurvivalResult {
    Survivors(Population),
    Indices(Vec<usize>),
}

/// Abstract base for survival selection strategies.
///
/// Mirrors `pymoo.core.survival.Survival`.
pub trait Survival {
    fn filter_infeasible(&self) -> bool {
        true
    }

    /// Select `n_survive` individuals to carry forward to the next generation.
    ///
    /// `off` is an optional offspring population forwarded as `*args` to
    /// `_do_survival` (used by `ToReplacement`).
    ///
    /// Mirrors `Survival.do(problem, pop, n_survive=None, random_state=None,
    ///                      return_indices=False)`.
    fn do_survival(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        off: Option<&Population>,
        n_survive: Option<usize>,
        mut random_state: Option<&mut StdRng>,
        return_indices: Option<bool>,
    ) -> SurvivalResult {
        let return_indices = return_indices.unwrap_or(true);
        if pop.is_empty() {
            return if return_indices {
                SurvivalResult::Indices(vec![])
            } else {
                SurvivalResult::Survivors(Population::empty(0))
            };
        }

        let n_survive = n_survive.unwrap_or_else(|| pop.len()).min(pop.len());

        // Tag every individual with its original index so we can recover indices
        // after survival selection regardless of cloning.
        // Mirrors the `H = {ind: k for k, ind in enumerate(pop)}` map used later.
        let mut tagged = pop.clone();
        for (k, ind) in tagged.iter_mut().enumerate() {
            ind.data
                .insert("__orig_idx__".to_string(), Value::Int(k as i64));
        }

        let survivors = if self.filter_infeasible() && problem.has_constraints() {
            // split feasible and infeasible solutions
            let (feas, infeas) = split_by_feasibility(&tagged, Some(true), Some(false));

            let feas_survivors = if feas.is_empty() {
                Population::empty(0)
            } else {
                let feas_pop = tagged.select(&feas);
                self._do_survival(
                    problem,
                    &feas_pop,
                    off,
                    Some(feas.len().min(n_survive)),
                    random_state.as_deref_mut(),
                )
            };

            // fill remaining slots with infeasible individuals (sorted by CV)
            let n_remaining = n_survive - feas_survivors.len();
            if n_remaining > 0 {
                let take = n_remaining.min(infeas.len());
                let infeas_pop = tagged.select(&infeas[..take]);
                merge(feas_survivors, infeas_pop)
            } else {
                feas_survivors
            }
        } else {
            self._do_survival(problem, &tagged, off, Some(n_survive), random_state)
        };

        if return_indices {
            // mirrors: return [H[survivor] for survivor in survivors]
            let indices: Vec<usize> = survivors
                .iter()
                .filter_map(|ind| match ind.data.get("__orig_idx__") {
                    Some(Value::Int(k)) => Some(*k as usize),
                    _ => None,
                })
                .collect();
            SurvivalResult::Indices(indices)
        } else {
            SurvivalResult::Survivors(survivors)
        }
    }

    /// Abstract — subclasses must implement.
    ///
    /// `off` carries any extra positional arguments forwarded from `do_survival`
    /// (e.g. the offspring population for `ToReplacement`).
    ///
    /// Mirrors `Survival._do(problem, pop, n_survive=None, random_state=None)`.
    fn _do_survival(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        off: Option<&Population>,
        n_survive: Option<usize>,
        random_state: Option<&mut StdRng>,
    ) -> Population;
}

/// Wraps a survival strategy and applies replacement: offspring replace parents
/// only when the offspring rank better under the wrapped survival.
///
/// Mirrors `pymoo.core.survival.ToReplacement`.
pub struct ToReplacement {
    pub survival: Box<dyn Survival>,
}

impl ToReplacement {
    pub fn new(survival: Box<dyn Survival>) -> Self {
        Self { survival }
    }
}

impl Survival for ToReplacement {
    fn filter_infeasible(&self) -> bool {
        false
    }

    /// Mirrors `ToReplacement._do(problem, pop, off, random_state)`.
    fn _do_survival(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        off: Option<&Population>,
        _n_survive: Option<usize>,
        mut random_state: Option<&mut StdRng>,
    ) -> Population {
        let random_state = random_state.unwrap_or(&mut default_random_state());

        let off = match off {
            Some(o) => o,
            None => return *pop,
        };

        let n = pop.len();
        let merged = merge(*pop, *off);
        let n_merged = merged.len();

        // rank all merged individuals by the wrapped survival strategy
        // mirrors: I = self.survival.do(problem, merged, n_survive=len(merged),
        //                               return_indices=True, ...)
        let result = self.survival.do_survival(
            problem,
            &merged,
            None,
            Some(n_merged),
            Some(random_state),
            Some(true),
        );

        let rank_vec = match result {
            SurvivalResult::Indices(v) => v,
            _ => return *pop,
        };

        // mirrors: merged.set("__rank__", I)
        // In Python shared references propagate the rank back to pop/off elements.
        // In Rust we access rank_vec directly: pop[k] ↔ rank_vec[k],
        //                                       off[k] ↔ rank_vec[n + k].

        // for k in range(len(pop)):
        //     if off[k].get("__rank__") < pop[k].get("__rank__"):
        //         pop[k] = off[k]
        let mut result_pop = pop.clone();
        for k in 0..n {
            let pop_rank = rank_vec.get(k).copied().unwrap_or(usize::MAX);
            let off_rank = rank_vec.get(n + k).copied().unwrap_or(usize::MAX);
            if off_rank < pop_rank {
                result_pop[k] = off[k].clone();
            }
        }

        *result_pop
    }
}

/// Split a population into feasible and infeasible index sets.
///
/// `sort_infeas_by_cv` — sort infeasible indices ascending by constraint violation.
/// `sort_feas_by_obj`  — sort feasible indices ascending by first objective value.
///
/// Mirrors `pymoo.core.survival.split_by_feasibility(
///     pop, sort_infeas_by_cv=True, sort_feas_by_obj=False, return_pop=False)`.
///
/// The `return_pop=True` variant is omitted; callers can call `pop.select(&feas)` themselves.
pub fn split_by_feasibility(
    pop: &Population,
    sort_infeas_by_cv: Option<bool>,
    sort_feas_by_obj: Option<bool>,
) -> (Vec<usize>, Vec<usize>) {
    let sort_infeas_by_cv = sort_infeas_by_cv.unwrap_or(true);
    let sort_feas_by_obj = sort_feas_by_obj.unwrap_or(false);
    // mirrors: F, CV, b = pop.get("F", "CV", "FEAS")
    let b = match pop.get(&IndividualField::Feas) {
        Value::BoolArray(arr) => arr,
        _ => return (vec![], (0..pop.len()).collect()),
    };

    let cv = match pop.get(&IndividualField::CV) {
        Value::FloatArray(arr) => arr,
        _ => return ((0..pop.len()).collect(), vec![]),
    };

    let f_val = pop.get(&IndividualField::F);

    // mirrors: feasible = np.where(b)[0]; infeasible = np.where(~b)[0]
    let mut feasible: Vec<usize> = b
        .iter()
        .enumerate()
        .filter_map(|(i, &feas)| if feas { Some(i) } else { None })
        .collect();

    let mut infeasible: Vec<usize> = b
        .iter()
        .enumerate()
        .filter_map(|(i, &feas)| if !feas { Some(i) } else { None })
        .collect();

    // mirrors: infeasible = infeasible[np.argsort(CV[infeasible, 0])]
    if sort_infeas_by_cv {
        infeasible.sort_by(|&a, &b| cv[a].partial_cmp(&cv[b]).unwrap_or(Ordering::Equal));
    }

    // mirrors: feasible = feasible[np.argsort(F[feasible, 0])]
    if sort_feas_by_obj {
        if let Value::FloatMatrix(ref f_mat) = f_val {
            feasible.sort_by(|&a, &b| {
                f_mat[[a, 0]]
                    .partial_cmp(&f_mat[[b, 0]])
                    .unwrap_or(Ordering::Equal)
            });
        }
    }

    (feasible, infeasible)
}
