use crate::{
    core::{
        algorithm::Algorithm,
        termination::{Termination, TerminationBase},
    },
    util::misc::time_to_int,
};

/// Mirrors `pymoo.termination.max_time.TimeBasedTermination`.
pub struct TimeBasedTermination {
    pub base: TerminationBase,
    pub max_time: f64,
}

impl TimeBasedTermination {
    /// Mirrors `TimeBasedTermination.__init__` for numeric `max_time` (seconds).
    pub fn new(max_time: f64) -> Self {
        Self {
            base: TerminationBase::new(),
            max_time,
        }
    }

    /// Mirrors `TimeBasedTermination.__init__` for string `max_time`.
    ///
    /// Parses the string via `time_to_int` (e.g. `"00:30:00"` → seconds).
    pub fn from_str(max_time: &str) -> Self {
        Self {
            base: TerminationBase::new(),
            max_time: time_to_int(max_time)? as f64,
        }
    }
}

impl Termination for TimeBasedTermination {
    fn base(&self) -> &TerminationBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.base
    }

    /// Mirrors `TimeBasedTermination._update`:
    /// `elapsed = time.time() - algorithm.start_time; return elapsed / self.max_time`.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let elapsed = algorithm
            .base()
            .start_time
            .map_or(0.0, |t| t.elapsed().as_secs_f64());
        elapsed / self.max_time
    }
}
