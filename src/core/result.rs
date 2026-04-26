use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use crate::core::{
    algorithm::Algorithm, individual::Value, population::Population, problem::Problem,
};

/// The result of an optimization run.
///
/// Mirrors `pymoo.core.result.Result`.
pub struct AlgorithmResult {
    /// The optimal solution(s) found.
    pub opt: Option<Population>,

    pub success: Option<bool>,
    pub message: Option<String>,

    /// The problem that was solved.
    pub problem: Option<Box<dyn Problem>>,

    /// The archive maintained during the run.
    pub archive: Option<Population>,

    /// The true Pareto front of the problem, if known.
    pub pf: Option<Population>,

    /// The algorithm that produced this result.
    pub algorithm: Option<Box<dyn Algorithm>>,

    /// The final population.
    pub pop: Option<Population>,

    // Convenience copies of the optimal values.
    /// Mirrors `Result.X`.
    pub x: Option<Value>,
    /// Mirrors `Result.F`.
    pub f: Option<Value>,
    /// Mirrors `Result.CV`.
    pub cv: Option<Value>,
    /// Mirrors `Result.G`.
    pub g: Option<Value>,
    /// Mirrors `Result.H`.
    pub h: Option<Value>,

    /// Mirrors `Result.start_time`.
    pub start_time: Option<Instant>,
    /// Mirrors `Result.end_time`.
    pub end_time: Option<Instant>,
    /// Wall-clock duration of the run.
    ///
    /// Mirrors `Result.exec_time = end_time - start_time`.
    pub exec_time: Option<Duration>,

    /// Snapshots of the algorithm state saved each generation (when enabled).
    ///
    /// Mirrors `Result.history`.
    pub history: Vec<Box<dyn Algorithm>>,

    /// Arbitrary data carried by the algorithm at the end of the run.
    ///
    /// Mirrors `Result.data`.
    pub data: HashMap<String, Value>,
}

impl AlgorithmResult {
    /// Mirrors `Result.__init__()`.
    pub fn new() -> Self {
        Self {
            opt: None,
            success: None,
            message: None,
            problem: None,
            archive: None,
            pf: None,
            algorithm: None,
            pop: None,
            x: None,
            f: None,
            cv: None,
            g: None,
            h: None,
            start_time: None,
            end_time: None,
            exec_time: None,
            history: Vec::new(),
            data: HashMap::new(),
        }
    }

    /// Scalar constraint violation of the (first) optimal solution.
    ///
    /// Mirrors `Result.cv` property: `return self.CV[0]`.
    pub fn cv(&self) -> Option<f64> {
        match &self.cv {
            Some(Value::Float(f)) => Some(*f),
            Some(Value::FloatArray(arr)) if !arr.is_empty() => Some(arr[0]),
            _ => None,
        }
    }

    /// Objective value(s) of the (first) optimal solution.
    ///
    /// Mirrors `Result.f` property: `return self.F[0]`.
    pub fn f(&self) -> Option<Value> {
        match &self.f {
            Some(Value::Float(f)) => Some(Value::Float(*f)),
            Some(Value::FloatArray(arr)) if !arr.is_empty() => Some(Value::Float(arr[0])),
            Some(Value::FloatMatrix(m)) if m.nrows() > 0 => {
                Some(Value::FloatArray(m.row(0).to_owned()))
            }
            _ => None,
        }
    }

    /// Whether the optimal solution is feasible (`cv <= 0`).
    ///
    /// Mirrors `Result.feas` property: `return self.cv <= 0`.
    pub fn feas(&self) -> bool {
        self.cv().map_or(false, |cv| cv <= 0.0)
    }
}
