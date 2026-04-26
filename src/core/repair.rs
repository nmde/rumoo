use ndarray::Array2;

use crate::core::{
    individual::{IndividualField, Value},
    population::Population,
    problem::Problem,
};

// -------------------------------------------------------------------------------------------------
// RepairBase
// -------------------------------------------------------------------------------------------------

/// Shared state for `Repair` implementors.
///
/// Mirrors the `Operator.__init__` attributes used by `Repair`
/// (`vtype` is the only field relevant to the repair workflow).
pub struct RepairBase {
    pub vtype: Option<Value>,
}

impl RepairBase {
    pub fn new(vtype: Option<Value>) -> Self {
        Self { vtype }
    }
}

// -------------------------------------------------------------------------------------------------
// Repair trait
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.core.repair.Repair`.
pub trait Repair {
    fn base(&self) -> &RepairBase;

    /// Mirrors `Repair._do(problem, X)` — transforms the decision-variable matrix.
    ///
    /// Default implementation is the identity (no repair), matching the Python base class.
    fn _do(&self, _problem: &dyn Problem, x: Array2<f64>) -> Array2<f64> {
        x
    }

    /// Mirrors `Repair.do(problem, pop)`.
    ///
    /// Extracts the X matrix from `pop`, optionally casts it via `vtype`,
    /// calls `_do`, then writes the repaired values back as per-individual X.
    fn do_repair(&self, problem: &dyn Problem, pop: &Population) -> Population {
        // X = np.array([ind.X for ind in pop])
        let x = match pop.get(&IndividualField::X) {
            Value::FloatMatrix(m) => m,
            _ => return pop.select(&(0..pop.len()).collect::<Vec<_>>()),
        };

        // if self.vtype is not None: X = X.astype(self.vtype)
        let x = if let Some(vt) = &self.base().vtype {
            match vt {
                Value::Int(_) => x.mapv(|v| v.trunc()),
                Value::Bool(_) => x.mapv(|v| if v != 0.0 { 1.0 } else { 0.0 }),
                _ => x,
            }
        } else {
            x
        };

        // Xp = self._do(problem, X, **kwargs)
        let xp = self._do(problem, x);

        // pop.set("X", Xp)
        let mut result = pop.select(&(0..pop.len()).collect::<Vec<_>>());
        let rows: Vec<Value> = (0..result.len())
            .map(|i| Value::FloatArray(xp.row(i).to_owned()))
            .collect();
        let _ = result.set_each(&IndividualField::X, rows);
        result
    }
}

// -------------------------------------------------------------------------------------------------
// NoRepair
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.core.repair.NoRepair`.
///
/// Uses all default `Repair` implementations — `_do` is the identity and
/// `do_repair` writes the X values back unchanged.
pub struct NoRepair {
    pub base: RepairBase,
}

impl NoRepair {
    pub fn new() -> Self {
        Self { base: RepairBase::new(None) }
    }
}

impl Default for NoRepair {
    fn default() -> Self {
        Self::new()
    }
}

impl Repair for NoRepair {
    fn base(&self) -> &RepairBase {
        &self.base
    }
}
