use crate::core::{individual::IndividualField, individual::Value, population::Population};

/// Return the optimal subset of `pop`.
///
/// For feasible solutions: single-objective → individual with minimum F;
/// multi-objective → non-dominated front.
/// If no feasible solution exists and `least_infeasible` is `true`, returns
/// the individual with the smallest constraint violation instead.
///
/// Mirrors `pymoo.util.optimum.filter_optimum(pop, least_infeasible=False)`.
pub fn filter_optimum(pop: &Population, least_infeasible: Option<bool>) -> Option<Population> {
    let least_infeasible = least_infeasible.unwrap_or(false);
    // Mirrors: if pop is None or len(pop) == 0: return None
    if pop.is_empty() {
        return None;
    }

    // Mirrors: ret = pop[pop.get("feas")]
    let ret = match pop.get(&IndividualField::Feas) {
        Value::BoolArray(mask) => pop.select_where(&mask),
        _ => Population::empty(0),
    };

    if !ret.is_empty() {
        // Mirrors: F = ret.get("F")
        match ret.get(&IndividualField::F) {
            Value::FloatMatrix(f) => {
                let n_obj = f.ncols();

                if n_obj > 1 {
                    // Mirrors: I = NonDominatedSorting().do(F, only_non_dominated_front=True)
                    //          ret = ret[I]
                    let i = NonDominatedSorting::new().do_sort(&f, true);
                    Some(ret.select(&i))
                } else {
                    // Mirrors: ret = ret[np.argmin(F[:, 0])]
                    let best = argmin_f64(f.column(0).iter().copied());
                    Some(ret.select(&[best]))
                }
            }
            _ => Some(ret),
        }
    } else {
        // Mirrors: if least_infeasible: ret = pop[np.argmin(pop.get("CV"))]
        if least_infeasible {
            match pop.get(&IndividualField::CV) {
                Value::FloatArray(cv) => {
                    let best = argmin_f64(cv.iter().copied());
                    Some(pop.select(&[best]))
                }
                _ => None,
            }
        } else {
            None
        }
    }
}

/// Return the index of the minimum value in `iter`, or `0` if empty.
///
/// Mirrors `np.argmin(…)` for 1-D float sequences.
fn argmin_f64(iter: impl Iterator<Item = f64>) -> usize {
    iter.enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
