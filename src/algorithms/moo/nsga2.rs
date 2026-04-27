use crate::{
    algorithms::base::genetic::GeneticAlgorithm,
    core::{
        crossover::Crossover, duplicate::EliminateDuplicates, mutation::Mutation,
        sampling::Sampling, selection::Selection, survival::Survival,
    },
    util::display::output::Output,
};

/*
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True, random_state=algorithm.random_state)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True, random_state=algorithm.random_state)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(RankAndCrowding):

    def __init__(self, nds=None, crowding_func="cd"):
        super().__init__(nds, crowding_func)
*/
// =========================================================================================================
// Implementation
// =========================================================================================================

enum TournamentType {
    CompByDomAndCrowding,
}

struct NSGA2 {
    algorithm: GeneticAlgorithm,
    tournament_type: TournamentType,
}

impl NSGA2 {
    pub fn new(
        pop_size: Option<usize>,
        sampling: Option<Box<dyn Sampling>>,
        selection: Option<Box<dyn Selection>>,
        crossover: Option<Box<dyn Crossover>>,
        mutation: Option<Box<dyn Mutation>>,
        survival: Option<Box<dyn Survival>>,
        output: Option<Box<dyn Output>>,
        advance_after_initial_infill: Option<bool>,
    ) -> Self {
        let algorithm = GeneticAlgorithm::new(
            Some(pop_size.unwrap_or(100)),
            Some(sampling.unwrap_or(FloatRandomSampling::new())),
            Some(selection.unwrap_or(TournamentSelection::new(func_comp = binary_tournament))),
            Some(crossover.unwrap_or(SBX::new(eta = 15, prob = 0.9))),
            Some(mutation.unwrap_or(PM::new(eta = 20))),
            Some(survival.unwrap_or(RankAndCrowding::new())),
            None,
            EliminateDuplicates::None,
            None,
            None,
            Some(advance_after_initial_infill.unwrap_or(true)),
            Some(output.unwrap_or(MultiObjectiveOutput::new())),
        );
        algorithm.termination = MultiObjectiveTermination::default();

        Self {
            algorithm,
            tournament_type: TournamentType::CompByDomAndCrowding,
        }
    }

    fn _set_optimum(&self) {
        if !has_feasible(self.pop) {
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]];
        } else {
            self.opt = self.pop[self.pop.get("rank") == 0];
        }
    }
}
