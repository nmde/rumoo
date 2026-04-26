use ndarray::Array2;

use crate::util::dominator::Dominator;

/// Mirrors `pymoo.util.nds.fast_non_dominated_sort.fast_non_dominated_sort`.
///
/// Performs fast non-dominated sorting on the objective matrix `f`.
/// Returns a list of fronts, where each front is a list of solution indices.
///
/// `dominator` — optional custom dominator instance; uses `Dominator::calc_domination_matrix`
/// by default.  `epsilon` is forwarded to the domination calculation when provided.
pub fn fast_non_dominated_sort(
    f: &Array2<f64>,
    dominator: Option<Dominator>,
    epsilon: Option<f64>,
) -> Vec<Vec<usize>> {
    let m: Array2<i32> = match dominator {
        Some(d) => d.calc_domination_matrix(Some(f), epsilon),
        None => Dominator::calc_domination_matrix(f, Some(f), epsilon),
    };

    let n = m.nrows();
    let mut fronts: Vec<Vec<usize>> = Vec::new();

    if n == 0 {
        return fronts;
    }

    let mut n_ranked = 0usize;
    let mut ranked = vec![false; n];

    // Mirrors: is_dominating = [[] for _ in range(n)]
    let mut is_dominating: Vec<Vec<usize>> = vec![Vec::new(); n];
    // Mirrors: n_dominated = np.zeros(n)
    let mut n_dominated = vec![0i32; n];

    let mut current_front: Vec<usize> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let rel = m[[i, j]];
            if rel == 1 {
                is_dominating[i].push(j);
                n_dominated[j] += 1;
            } else if rel == -1 {
                is_dominating[j].push(i);
                n_dominated[i] += 1;
            }
        }

        if n_dominated[i] == 0 {
            current_front.push(i);
            ranked[i] = true;
            n_ranked += 1;
        }
    }

    fronts.push(current_front.clone());

    // Mirrors: while n_ranked < n
    while n_ranked < n {
        let mut next_front: Vec<usize> = Vec::new();

        for &i in &current_front {
            for &j in &is_dominating[i] {
                n_dominated[j] -= 1;
                if n_dominated[j] == 0 {
                    next_front.push(j);
                    ranked[j] = true;
                    n_ranked += 1;
                }
            }
        }

        fronts.push(next_front.clone());
        current_front = next_front;
    }

    fronts
}
