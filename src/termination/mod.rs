pub mod default;
pub mod fmin;
pub mod max_eval;
pub mod max_gen;
pub mod max_time;
pub mod robust;

use anyhow::{Result, anyhow};

use crate::{
    core::termination::Termination,
    termination::{
        default::{DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination},
        fmin::MinimumFunctionValueTermination,
        max_eval::MaximumFunctionCallTermination,
        max_gen::MaximumGenerationTermination,
        max_time::TimeBasedTermination,
    },
};

/// Mirrors `pymoo.termination.get_termination(name, *args, **kwargs)`.
///
/// The Python version accepts variadic positional and keyword arguments; the
/// Rust version collapses these into a single optional numeric `value` that
/// is forwarded to the chosen termination constructor.
pub fn get_termination(name: &str, value: Option<f64>) -> Result<Box<dyn Termination>> {
    match name {
        "n_eval" | "n_evals" => Ok(Box::new(MaximumFunctionCallTermination::new(value))),
        "n_gen" | "n_iter" => Ok(Box::new(MaximumGenerationTermination::new(value))),
        "fmin" => {
            if value.is_none() {
                return Err(anyhow!(
                    "Must specify a time value for minimum function value termination"
                ));
            }
            Ok(Box::new(MinimumFunctionValueTermination::new(
                value.unwrap(),
            )))
        }
        "time" => {
            if value.is_none() {
                return Err(anyhow!(
                    "Must specify a time value for time-based termination"
                ));
            }
            Ok(Box::new(TimeBasedTermination::new(value.unwrap())))
        }
        "soo" => Ok(Box::new(DefaultSingleObjectiveTermination::new(
            None, None, None, None, None, None,
        ))),
        "moo" => Ok(Box::new(DefaultMultiObjectiveTermination::new(
            None, None, None, None, None, None, None,
        ))),
        _ => Err(anyhow!("Termination not found.")),
    }
}
