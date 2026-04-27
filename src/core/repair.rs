use ndarray::{Array1, Array2};

use crate::core::{operator::Operator, population::Population, problem::Problem};

pub struct RepairBase {}

impl RepairBase {
    pub fn new() -> Self {
        Self {}
    }
}

impl Operator for RepairBase {}

pub trait Repair {
    fn base(&self) -> &RepairBase;

    fn do_repair(&self, problem: &Box<dyn Problem>, pop: &Population) -> Population {
        let xp = self._do(problem, &pop.x());

        let mut result = pop.select(&(0..pop.len()).collect::<Vec<usize>>());
        result.set_each_x(
            &(0..result.len())
                .map(|i| xp.row(i).to_owned())
                .collect::<Vec<Array1<f64>>>(),
        );
        result
    }

    fn _do(&self, _problem: &Box<dyn Problem>, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }
}

pub struct NoRepair {
    base: RepairBase,
}

impl NoRepair {
    pub fn new() -> Self {
        Self {
            base: RepairBase::new(),
        }
    }
}

impl Repair for NoRepair {
    fn base(&self) -> &RepairBase {
        &self.base
    }
}
