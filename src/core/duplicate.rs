use std::{collections::HashSet, f64::INFINITY};

use anyhow::Result;
use ndarray::{Array1, Array2};

use crate::{core::population::Population, util::misc::cdist};

pub enum EliminateDuplicates {
    None,
    Bool(bool),
    Eliminator(Box<dyn DuplicateElimination>),
}

struct DuplicateEliminationBase {
    func: fn(&Population) -> Array2<f64>,
}

impl DuplicateEliminationBase {
    pub fn new(func: Option<fn(&Population) -> Array2<f64>>) -> Self {
        Self {
            func: func.unwrap_or(|pop| pop.x()),
        }
    }
}

pub trait DuplicateElimination {
    fn base(&self) -> &DuplicateEliminationBase;

    fn do_elimination(
        &self,
        pop: &Population,
        others: &Vec<Population>,
        to_itself: Option<bool>,
    ) -> Result<(Population, Vec<usize>, Vec<usize>)> {
        let to_itself = to_itself.unwrap_or(true);
        let original = pop;

        let mut surviving = (0..pop.len()).collect::<Vec<usize>>();
        if to_itself {
            let mut is_dup = Array1::from_elem(surviving.len(), false);
            self._do(&pop.select(&surviving), None, &mut is_dup)?;
            surviving = surviving
                .into_iter()
                .zip(is_dup.iter())
                .filter_map(|(idx, dup)| if !dup { Some(idx) } else { None })
                .collect();
        }

        for other in others {
            if surviving.is_empty() {
                break;
            }
            if other.is_empty() {
                continue;
            }
            let mut is_dup = Array1::from_elem(surviving.len(), false);
            self._do(&pop.select(&surviving), Some(other), &mut is_dup)?;
            surviving = surviving
                .into_iter()
                .zip(is_dup.iter())
                .filter_map(|(idx, dup)| if !dup { Some(idx) } else { None })
                .collect();
        }

        let h = surviving.clone().into_iter().collect::<HashSet<usize>>();

        Ok((
            pop.select(&surviving),
            (0..original.len()).filter(|i| h.contains(i)).collect(),
            (0..original.len()).filter(|i| !h.contains(i)).collect(),
        ))
    }

    fn _do(
        &self,
        _pop: &Population,
        _other: Option<&Population>,
        _is_duplicate: &mut Array1<bool>,
    ) -> Result<()> {
        Ok(())
    }
}

pub struct DefaultDuplicateElimination {
    base: DuplicateEliminationBase,
    epsilon: f64,
}

impl DefaultDuplicateElimination {
    pub fn new(epsilon: Option<f64>, func: Option<fn(&Population) -> Array2<f64>>) -> Self {
        Self {
            base: DuplicateEliminationBase::new(func),
            epsilon: epsilon.unwrap_or(1e-16),
        }
    }

    fn calc_dist(&self, pop: &Population, other: Option<&Population>) -> Result<Array2<f64>> {
        let x = (self.base.func)(pop);

        if other.is_none() {
            let mut d = cdist(&x, &x)?;
            for i in 0..x.nrows() {
                for j in i..x.nrows() {
                    d[[i, j]] = INFINITY;
                }
            }
            return Ok(d);
        }
        cdist(&x, &(self.base.func)(other.unwrap()))
    }
}

impl DuplicateElimination for DefaultDuplicateElimination {
    fn base(&self) -> &DuplicateEliminationBase {
        &self.base
    }

    fn _do(
        &self,
        pop: &Population,
        other: Option<&Population>,
        is_duplicate: &mut Array1<bool>,
    ) -> Result<()> {
        let mut d = self.calc_dist(pop, other)?;
        d.mapv_inplace(|v| if v.is_nan() { INFINITY } else { v });

        for i in 0..d.nrows() {
            if d.row(i).iter().any(|v| v <= &self.epsilon) {
                is_duplicate[i] = true;
            }
        }

        Ok(())
    }
}

impl Default for DefaultDuplicateElimination {
    fn default() -> Self {
        Self::new(None, None)
    }
}

/*
def to_float(val):
    if isinstance(val, bool) or isinstance(val, np.bool_):
        return 0.0 if val else 1.0
    else:
        return val


class ElementwiseDuplicateElimination(DefaultDuplicateElimination):

    def __init__(self, cmp_func=None, **kwargs) -> None:
        super().__init__(**kwargs)

        if cmp_func is None:
            cmp_func = self.is_equal

        self.cmp_func = cmp_func

    def is_equal(self, a, b):
        pass

    def _do(self, pop, other, is_duplicate):

        if other is None:
            for i in range(len(pop)):
                for j in range(i + 1, len(pop)):
                    val = to_float(self.cmp_func(pop[i], pop[j]))
                    if val < self.epsilon:
                        is_duplicate[i] = True
                        break
        else:
            for i in range(len(pop)):
                for j in range(len(other)):
                    val = to_float(self.cmp_func(pop[i], other[j]))
                    if val < self.epsilon:
                        is_duplicate[i] = True
                        break

        return is_duplicate


def to_hash(x):
    try:
        h = hash(x)
    except:
        try:
            h = hash(str(x))
        except:
            raise Exception("Hash could not be calculated. Please use another duplicate elimination.")

    return h


class HashDuplicateElimination(DuplicateElimination):

    def __init__(self, func=to_hash) -> None:
        super().__init__()
        self.func = func

    def _do(self, pop, other, is_duplicate):
        H = set()

        if other is not None:
            for o in other:
                val = self.func(o)
                H.add(self.func(val))

        for i, ind in enumerate(pop):
            val = self.func(ind)
            h = self.func(val)

            if h in H:
                is_duplicate[i] = True
            else:
                H.add(h)

        return is_duplicate
*/

pub struct NoDuplicateElimination {
    base: DuplicateEliminationBase,
}

impl NoDuplicateElimination {
    pub fn new() -> Self {
        Self {
            base: DuplicateEliminationBase::new(None),
        }
    }
}

impl DuplicateElimination for NoDuplicateElimination {
    fn base(&self) -> &DuplicateEliminationBase {
        &self.base
    }

    fn do_elimination(
        &self,
        pop: &Population,
        _others: &Vec<Population>,
        _to_itself: Option<bool>,
    ) -> Result<(Population, Vec<usize>, Vec<usize>)> {
        Ok((pop.clone(), vec![], vec![]))
    }
}
