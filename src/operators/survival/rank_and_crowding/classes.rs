use std::cmp::Ordering;

use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;

use crate::{
    core::{
        individual::{IndividualField, Value},
        population::Population,
        problem::Problem,
        survival::{Survival, split_by_feasibility},
    },
    util::nds::non_dominated_sorting::NonDominatedSorting,
};
use super::metrics::{CrowdingFunction, get_crowding_function};

// -------------------------------------------------------------------------------------------------
// RankAndCrowding
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.operators.survival.rank_and_crowding.classes.RankAndCrowding`.
pub struct RankAndCrowding {
    pub nds: NonDominatedSorting,
    pub crowding_func: Box<dyn CrowdingFunction>,
}

impl RankAndCrowding {
    /// Mirrors `RankAndCrowding.__init__(nds=None, crowding_func="cd")`.
    pub fn new(nds: Option<NonDominatedSorting>, crowding_func: Option<&str>) -> Self {
        Self {
            nds: nds.unwrap_or_else(NonDominatedSorting::new),
            crowding_func: get_crowding_function(crowding_func.unwrap_or("cd")),
        }
    }
}

impl Survival for RankAndCrowding {
    fn filter_infeasible(&self) -> bool {
        true
    }

    /// Mirrors `RankAndCrowding._do(problem, pop, n_survive, random_state)`.
    fn _do_survival(
        &self,
        _problem: &dyn Problem,
        pop: &Population,
        _off: Option<&Population>,
        n_survive: Option<usize>,
        random_state: Option<&mut StdRng>,
    ) -> Population {
        let n_survive = n_survive.unwrap_or_else(|| pop.len());

        // Mirrors: F = pop.get("F").astype(float)
        let f = match pop.get(&IndividualField::F) {
            Value::FloatMatrix(m) => m,
            _ => return pop.select(&(0..pop.len()).collect::<Vec<_>>()),
        };

        let mut survivors: Vec<usize> = Vec::new();

        // Mirrors: fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        let (fronts, _) = self
            .nds
            .sort(&f, Some(false), Some(false), Some(n_survive), None);

        // Select all to get an owned mutable copy for setting rank/crowding.
        let mut pop_mut = pop.select(&(0..pop.len()).collect::<Vec<_>>());

        for (k, front) in fronts.iter().enumerate() {
            // Mirrors: I = np.arange(len(front))
            let mut i_sel: Vec<usize> = (0..front.len()).collect();

            // Mirrors: F[front, :] — sub-matrix of rows at front indices
            let front_rows: Vec<_> = front.iter().map(|&idx| f.row(idx)).collect();
            let f_front = ndarray::stack(
                Axis(0),
                &front_rows.iter().map(|r| r.view()).collect::<Vec<_>>(),
            )
            .unwrap_or_else(|_| Array2::zeros((0, f.ncols())));

            let crowding_of_front: Array1<f64>;

            if survivors.len() + front.len() > n_survive {
                // Mirrors: n_remove = len(survivors) + len(front) - n_survive
                let n_remove = survivors.len() + front.len() - n_survive;

                // Mirrors: crowding_of_front = self.crowding_func.do(F[front, :], n_remove=n_remove)
                crowding_of_front = self.crowding_func.do_crowd(&f_front, n_remove);

                // Mirrors: I = randomized_argsort(crowding_of_front, order='descending', ...)
                //          I = I[:-n_remove]
                let mut order: Vec<usize> = (0..crowding_of_front.len()).collect();
                order.sort_by(|&a, &b| {
                    crowding_of_front[b]
                        .partial_cmp(&crowding_of_front[a])
                        .unwrap_or(Ordering::Equal)
                });
                randomized_argsort(&mut order, &crowding_of_front, random_state.as_deref_mut());
                i_sel = order[..order.len() - n_remove].to_vec();
            } else {
                // Mirrors: crowding_of_front = self.crowding_func.do(F[front, :], n_remove=0)
                crowding_of_front = self.crowding_func.do_crowd(&f_front, 0);
            }

            // Mirrors: for j, i in enumerate(front): pop[i].set("rank", k); pop[i].set("crowding", ...)
            for (j, &i) in front.iter().enumerate() {
                pop_mut[i].set(
                    &IndividualField::DataField("rank".to_string()),
                    Value::Int(k as i64),
                );
                if j < crowding_of_front.len() {
                    pop_mut[i].set(
                        &IndividualField::DataField("crowding".to_string()),
                        Value::Float(crowding_of_front[j]),
                    );
                }
            }

            // Mirrors: survivors.extend(front[I])
            for &sel in &i_sel {
                survivors.push(front[sel]);
            }
        }

        // Mirrors: return pop[survivors]
        pop_mut.select(&survivors)
    }
}

// -------------------------------------------------------------------------------------------------
// ConstrRankAndCrowding
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.operators.survival.rank_and_crowding.classes.ConstrRankAndCrowding`.
pub struct ConstrRankAndCrowding {
    pub nds: NonDominatedSorting,
    pub ranking: RankAndCrowding,
}

impl ConstrRankAndCrowding {
    /// Mirrors `ConstrRankAndCrowding.__init__(nds=None, crowding_func="cd")`.
    pub fn new(nds: Option<NonDominatedSorting>, crowding_func: Option<&str>) -> Self {
        let nds = nds.unwrap_or_else(NonDominatedSorting::new);
        let ranking = RankAndCrowding::new(Some(nds.clone()), crowding_func);
        Self { nds, ranking }
    }
}

impl Survival for ConstrRankAndCrowding {
    fn filter_infeasible(&self) -> bool {
        false
    }

    /// Mirrors `ConstrRankAndCrowding._do(problem, pop, n_survive)`.
    fn _do_survival(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        _off: Option<&Population>,
        n_survive: Option<usize>,
        random_state: Option<&mut StdRng>,
    ) -> Population {
        let n_survive = n_survive.unwrap_or_else(|| pop.len()).min(pop.len());

        if problem.n_constr() == 0 {
            // Mirrors: survivors = self.ranking.do(problem, pop, n_survive=n_survive)
            return self
                .ranking
                ._do_survival(problem, pop, None, Some(n_survive), random_state);
        }

        // Mirrors: feas, infeas = split_by_feasibility(pop, sort_infeas_by_cv=True, sort_feas_by_obj=False)
        let (feas, infeas) = split_by_feasibility(pop, Some(true), Some(false));

        let mut survivors = if feas.is_empty() {
            // Mirrors: if n_feas == 0: survivors = Population()
            Population::empty(0)
        } else {
            // Mirrors: survivors = self.ranking.do(problem, pop[feas], n_survive=min(len(feas), n_survive))
            let feas_pop = pop.select(&feas);
            self.ranking._do_survival(
                problem,
                &feas_pop,
                None,
                Some(feas.len().min(n_survive)),
                random_state,
            )
        };

        // Mirrors: n_remaining = n_survive - len(survivors)
        let n_remaining = n_survive.saturating_sub(survivors.len());

        if n_remaining > 0 && !infeas.is_empty() {
            let infeas_pop = pop.select(&infeas);

            // Mirrors: G = pop[infeas].get("G"); G = np.maximum(G, 0)
            //          H = pop[infeas].get("H"); H = np.absolute(H)
            //          C = np.column_stack((G, H))
            let g = match infeas_pop.get(&IndividualField::G) {
                Value::FloatMatrix(m) => m.mapv(|v| v.max(0.0)),
                _ => Array2::zeros((infeas_pop.len(), 0)),
            };
            let h = match infeas_pop.get(&IndividualField::H) {
                Value::FloatMatrix(m) => m.mapv(f64::abs),
                _ => Array2::zeros((infeas_pop.len(), 0)),
            };
            let c = ndarray::concatenate(Axis(1), &[g.view(), h.view()])
                .unwrap_or_else(|_| Array2::zeros((infeas_pop.len(), 1)));

            // Mirrors: infeas_fronts = self.nds.do(C, n_stop_if_ranked=n_remaining)
            let (infeas_fronts, _) =
                self.nds
                    .sort(&c, Some(false), Some(false), Some(n_remaining), None);

            let mut infeas_pop_mut = infeas_pop;

            for (k, front) in infeas_fronts.iter().enumerate() {
                // Mirrors: pop[infeas][front].set("cv_rank", k)
                for &i in front {
                    infeas_pop_mut[i].set(
                        &IndividualField::DataField("cv_rank".to_string()),
                        Value::Int(k as i64),
                    );
                }

                let i_sel: Vec<usize> = if survivors.len() + front.len() > n_survive {
                    // Mirrors: CV = pop[infeas][front].get("CV").flatten()
                    //          I = randomized_argsort(CV, order='ascending')
                    //          I = I[:(n_survive - len(survivors))]
                    let cv = match infeas_pop_mut.select(front).get(&IndividualField::CV) {
                        Value::FloatArray(arr) => arr,
                        Value::FloatMatrix(m) => m.column(0).to_owned(),
                        _ => Array1::zeros(front.len()),
                    };
                    let mut order: Vec<usize> = (0..front.len()).collect();
                    order.sort_by(|&a, &b| {
                        cv[a]
                            .partial_cmp(&cv[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let take = n_survive - survivors.len();
                    order[..take.min(order.len())].to_vec()
                } else {
                    // Mirrors: I = np.arange(len(front))
                    (0..front.len()).collect()
                };

                // Mirrors: survivors = Population.merge(survivors, pop[infeas][front[I]])
                let selected_infeas_indices: Vec<usize> = i_sel.iter().map(|&j| front[j]).collect();
                let to_add = infeas_pop_mut.select(&selected_infeas_indices);
                survivors = Population::merge(survivors, to_add);

                if survivors.len() >= n_survive {
                    break;
                }
            }
        }

        survivors
    }
}
