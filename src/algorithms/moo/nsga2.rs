use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;

use crate::{
    core::{individual::IndividualField, population::Population},
    operators::selection::tournament::{CompareMethod, TournamentSelection, compare},
    util::misc::has_feasible,
};

/*
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
*/

pub fn binary_tournament(
    pop: &Population,
    p: &Array2<usize>,
    algorithm: &NSGA2,
) -> Result<Array2<i64>> {
    let n_tournaments = p.nrows();
    let n_parents = p.ncols();

    if n_parents != 2 {
        return Err(anyhow!("Only implemented for binary tournament!"));
    }

    let tournament_type = &algorithm.tournament_type;
    let mut s: Array1<f64> = Array1::from_elem(n_tournaments, f64::NAN);

    for i in 0..n_tournaments {
        let (a, b) = (p[[i, 0]], p[[i, 1]]);
        let (a_cv, a_f) = (pop[a].cv[0], &pop[a].f);
        let (b_cv, b_f) = (pop[b].cv[0], &pop[b].f);
        let (rank_a, cd_a) = pop[a].get_tuple(
            &IndividualField::DataField("rank".to_string()),
            &IndividualField::DataField("crowding".to_string()),
        );
        let (rank_b, cd_b) = pop[b].get_tuple(
            &IndividualField::DataField("rank".to_string()),
            &IndividualField::DataField("crowding".to_string()),
        );

        // if at least one solution is infeasible
        if a_cv > 0.0 || b_cv > 0.0 {
            s[i] = compare(
                a,
                a_cv,
                b,
                b_cv,
                &CompareMethod::SmallerIsBetter,
                Some(true),
                Some(&algorithm.random_state),
            )
            .map_or(f64::NAN, |w| w as f64);
        }
        // both solutions are feasible
        else {
            if tournament_type == "comp_by_dom_and_crowding" {
                let rel = Dominator::get_relation(a_f, b_f, None, None);
                if rel == 1 {
                    s[i] = a as f64;
                } else if rel == -1 {
                    s[i] = b as f64;
                }
            } else if tournament_type == "comp_by_rank_and_crowding" {
                s[i] = compare(
                    a,
                    rank_a,
                    b,
                    rank_b,
                    &CompareMethod::SmallerIsBetter,
                    Some(false),
                    None,
                )
                .map_or(f64::NAN, |w| w as f64);
            } else {
                return Err(anyhow!("Unknown tournament type: {tournament_type}"));
            }

            // if rank or domination relation didn't make a decision compare by crowding
            if s[i].is_nan() {
                s[i] = compare(
                    a,
                    cd_a,
                    b,
                    cd_b,
                    "larger_is_better",
                    true,
                    Some(&algorithm.random_state),
                )
                .map_or(f64::NAN, |w| w as f64);
            }
        }
    }

    // s[:, None].astype(int)
    Ok(s.mapv(|x| x as i64).insert_axis(ndarray::Axis(1)))
}

pub struct NSGA2 {
    pub algorithm: GeneticAlgorithm,
    pub tournament_type: String,
    pub random_state: StdRng,
}

impl NSGA2 {
    pub fn new(
        pop_size: usize,
        sampling: Box<dyn Sampling>,
        selection: Option<Box<dyn Selection>>,
        crossover: Box<dyn Crossover>,
        mutation: Box<dyn Mutation>,
        survival: Option<Box<dyn Survival>>,
        output: Option<Box<dyn Output>>,
        advance_after_initial_infill: bool,
    ) -> Self {
        let selection = selection.unwrap_or(Box::new(TournamentSelection::new(
            Some(Box::new(binary_tournament)),
            None,
        )));
        let survival = survival.unwrap_or_else(|| RankAndCrowding::new(None, "cd"));
        let output = output.unwrap_or_else(MultiObjectiveOutput::new);

        let mut algorithm = GeneticAlgorithm::new(
            pop_size,
            sampling,
            selection,
            crossover,
            mutation,
            survival,
            output,
            advance_after_initial_infill,
        );

        algorithm.termination = Box::new(DefaultMultiObjectiveTermination::new());

        Self {
            algorithm,
            tournament_type: "comp_by_dom_and_crowding".to_string(),
            random_state: rand::SeedableRng::from_entropy(),
        }
    }

    pub fn set_optimum(&mut self) {
        if !has_feasible(&self.inner.pop) {
            // self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
            let cv: Array1<f64> = self.inner.pop.get("CV");
            let min_idx = cv
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            self.inner.opt = self.inner.pop.select(&[min_idx]);
        } else {
            // self.opt = self.pop[self.pop.get("rank") == 0]
            let ranks: Array1<i64> = self.inner.pop.get("rank");
            let front_0: Vec<usize> = ranks
                .iter()
                .enumerate()
                .filter_map(|(i, &r)| if r == 0 { Some(i) } else { None })
                .collect();
            self.inner.opt = self.inner.pop.select(&front_0);
        }
    }
}
