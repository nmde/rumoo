use ndarray::Array2;
use rand::rngs::StdRng;

use crate::core::{individual::Value, population::Population, problem::Problem};

/// Shared data fields for all operators.
///
/// Mirrors the `__init__` attributes of `pymoo.core.operator.Operator`.
pub struct OperatorBase {
    pub name: String,
    pub vtype: Option<Value>,
    pub repair: Option<Box<dyn Operator>>,
}

impl OperatorBase {
    pub fn new(
        name: Option<String>,
        vtype: Option<Value>,
        repair: Option<Box<dyn Operator>>,
    ) -> Self {
        Self {
            name: name.unwrap_or_default(),
            vtype,
            repair,
        }
    }
}

/// Output of `Operator::call` — either a population or a raw decision-variable matrix.
///
/// Mirrors the two return paths of `Operator.__call__` controlled by `to_numpy`.
pub enum OperatorOutput {
    Pop(Population),
    Matrix(Array2<f64>),
}

/// Abstract base for all evolutionary operators (crossover, mutation, sampling, etc.).
///
/// Mirrors `pymoo.core.operator.Operator`.
pub trait Operator {
    fn base(&self) -> &OperatorBase;

    /// Mirrors `Operator.do`, forwarding to `_do` with an injected random state.
    fn do_op(
        &self,
        problem: &dyn Problem,
        elem: &Population,
        random_state: Option<&mut StdRng>,
    ) -> Population {
        self._do(problem, elem, random_state)
    }

    /// Abstract — subclasses must implement.
    ///
    /// Mirrors `Operator._do(problem, elem, random_state)`.
    fn _do(
        &self,
        problem: &dyn Problem,
        elem: &Population,
        random_state: Option<&mut StdRng>,
    ) -> Population;

    /// Apply the operator, optionally cast `X` to `vtype`, apply repair, and
    /// return either a `Population` or a raw matrix of decision variables.
    ///
    /// Mirrors `Operator.__call__(problem, elem, to_numpy=False)`.
    fn call(
        &self,
        problem: &dyn Problem,
        elem: &Population,
        to_numpy: Option<bool>,
        random_state: Option<&mut StdRng>,
    ) -> OperatorOutput {
        let to_numpy = to_numpy.unwrap_or(false);
        let mut out = self.do_op(problem, elem, random_state);

        if let Some(vtype) = &self.base().vtype {
            for ind in out.iter_mut() {
                if let Some(ref x) = ind.x {
                    ind.x = Some(match vtype {
                        // mirrors ind.X.astype(int) — truncates toward zero
                        Value::Int(_) => x.mapv(|v| v.trunc()),
                        Value::Bool(_) => x.mapv(|v| if v != 0.0 { 1.0 } else { 0.0 }),
                        Value::Float(_) => x.clone(),
                        _ => x.clone(),
                    });
                }
            }
        }

        if let Some(repair) = &self.base().repair {
            out = repair.do_op(problem, &out, None);
        }

        if to_numpy {
            let n = out.len();
            if n == 0 {
                return OperatorOutput::Matrix(Array2::zeros((0, 0)));
            }
            let m = out[0].x.as_ref().map(|x| x.len()).unwrap_or(0);
            let mut mat = Array2::zeros((n, m));
            for (i, ind) in out.iter().enumerate() {
                if let Some(ref x) = ind.x {
                    mat.row_mut(i).assign(x);
                }
            }
            return OperatorOutput::Matrix(mat);
        }

        OperatorOutput::Pop(out)
    }
}
