use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, s};
use rand::{rngs::StdRng, seq::SliceRandom};

use crate::core::{
    individual::{IndividualField, Value},
    operator::Operator,
    population::Population,
    problem::Problem,
};

/// Extra data fields specific to crossover operators.
///
/// Mirrors the `__init__` attributes added by `pymoo.core.crossover.Crossover`
/// on top of `Operator`.
pub struct CrossoverBase {
    pub n_parents: usize,
    pub n_offsprings: usize,
    /// Probability that crossover is applied to a mating (default 0.9).
    /// Mirrors `self.prob = Real(prob, bounds=(0.5, 1.0), strict=(0.0, 1.0))`.
    pub prob: f64,
}

impl CrossoverBase {
    pub fn new(n_parents: usize, n_offsprings: usize, prob: Option<f64>) -> Self {
        Self {
            n_parents,
            n_offsprings,
            prob: prob.unwrap_or(0.9),
        }
    }
}

/// Abstract base for recombination (crossover) operators.
///
/// Mirrors `pymoo.core.crossover.Crossover`.
pub trait Crossover: Operator {
    fn crossover_base(&self) -> &CrossoverBase;

    /// Apply the crossover to a population.
    ///
    /// `parents` — optional index matrix of shape `(n_matings, n_parents)` that
    /// selects which individuals from `pop` participate in each mating.
    /// When `None`, pop is assumed to be pre-grouped as consecutive blocks of
    /// `n_parents` individuals per mating.
    ///
    /// Mirrors `Crossover.do(problem, pop, parents=None, random_state=None)`.
    fn do_crossover(
        &self,
        problem: &dyn Problem,
        pop: &Population,
        parents: Option<&Array2<usize>>,
        mut random_state: Option<&mut StdRng>,
    ) -> Result<Population> {
        let n_parents = self.crossover_base().n_parents;
        let n_offsprings = self.crossover_base().n_offsprings;
        let prob = self.crossover_base().prob;
        let n_var = problem.base().n_var as usize;

        // n_matings: number of crossover operations to perform
        let n_matings = match parents {
            Some(p) => p.nrows(),
            None => pop.len() / n_parents,
        };

        // Collect X from all parents into a 3-D array of shape (n_parents, n_matings, n_var).
        // Mirrors: X = np.swapaxes([[parent.get("X") for parent in mating] for mating in pop], 0, 1)
        let mut x = Array3::<f64>::zeros((n_parents, n_matings, n_var));
        for mating in 0..n_matings {
            for slot in 0..n_parents {
                let ind_idx = match parents {
                    Some(p) => p[[mating, slot]],
                    None => mating * n_parents + slot,
                };
                if let Some(ref xi) = pop[ind_idx].x {
                    x.slice_mut(s![slot, mating, ..]).assign(xi);
                }
            }
        }

        // Output offspring array: shape (n_offsprings, n_matings, n_var).
        let mut xp = Array3::<f64>::zeros((n_offsprings, n_matings, n_var));

        // Per-mating crossover mask.
        // Mirrors: prob = get(self.prob, size=n_matings); cross = random_state.random(n_matings) < prob
        let mut cross = vec![false; n_matings];
        if let Some(ref mut rng) = random_state {
            for c in cross.iter_mut() {
                *c = rng.random::<f64>() < prob;
            }
        }

        let any_cross = cross.iter().any(|&c| c);

        // Run _do_crossover and assign results for matings where crossover fires.
        if any_cross {
            let q = self._do_crossover(problem, &x, random_state.as_deref_mut());
            if q.shape() != &[n_offsprings, n_matings, n_var] {
                return Err(anyhow!("Shape is incorrect of crossover impl."));
            }
            for (k, &c) in cross.iter().enumerate() {
                if c {
                    xp.slice_mut(s![.., k, ..]).assign(&q.slice(s![.., k, ..]));
                }
            }
        }

        // For matings where NO crossover fires: copy parent X directly into offspring slots.
        // Mirrors the `for k in np.flatnonzero(~cross):` block.
        for (k, &c) in cross.iter().enumerate() {
            if !c {
                // build the mapping from offspring slot → parent slot
                let parent_slots: Vec<usize> = if n_offsprings < n_parents {
                    // sample without replacement
                    // mirrors: s = random_state.choice(np.arange(n_parents), size=n_offsprings, replace=False)
                    let mut indices: Vec<usize> = (0..n_parents).collect();
                    if let Some(ref mut rng) = random_state {
                        indices.shuffle(rng);
                    }
                    indices.truncate(n_offsprings);
                    indices
                } else if n_offsprings == n_parents {
                    // mirrors: s = np.arange(n_parents)
                    (0..n_parents).collect()
                } else {
                    // extend with repeated permutations until we have enough
                    // mirrors: while len(s) < n_offsprings: s.extend(random_state.permutation(n_parents))
                    let mut slots: Vec<usize> = Vec::new();
                    while slots.len() < n_offsprings {
                        let mut perm: Vec<usize> = (0..n_parents).collect();
                        if let Some(ref mut rng) = random_state {
                            perm.shuffle(rng);
                        }
                        slots.extend_from_slice(&perm);
                    }
                    slots.truncate(n_offsprings);
                    slots
                };

                // mirrors: Xp[:, k] = np.copy(X[s, k])
                for (offspring_slot, &parent_slot) in parent_slots.iter().enumerate() {
                    xp.slice_mut(s![offspring_slot, k, ..]).assign(&x.slice(s![
                        parent_slot,
                        k,
                        ..
                    ]));
                }
            }
        }

        // Flatten (n_offsprings, n_matings, n_var) → (n_offsprings * n_matings, n_var).
        // Mirrors: Xp = Xp.reshape(-1, X.shape[-1])
        let n_total = n_offsprings * n_matings;
        let xp_2d = xp.into_shape((n_total, n_var))?;

        // Mirrors: off = Population.new("X", Xp)
        Ok(Population::new_with_attrs(&[(
            &IndividualField::X,
            Value::FloatMatrix(xp_2d),
        )]))
    }

    /// Abstract — subclasses must implement.
    ///
    /// Receives `x` of shape `(n_parents, n_matings, n_var)`.
    /// Must return offspring of shape `(n_offsprings, n_matings, n_var)`.
    ///
    /// Mirrors `Crossover._do(problem, X, random_state)`.
    fn _do_crossover(
        &self,
        problem: &dyn Problem,
        x: &Array3<f64>,
        random_state: Option<&mut StdRng>,
    ) -> Array3<f64>;
}
