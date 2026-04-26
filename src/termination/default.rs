use crate::{
    core::{
        algorithm::Algorithm,
        termination::{Termination, TerminationBase},
    },
    termination::{
        max_eval::MaximumFunctionCallTermination, max_gen::MaximumGenerationTermination,
        robust::RobustTermination,
    },
};

/// Mirrors `pymoo.termination.default.DefaultTermination`.
pub struct DefaultTermination {
    pub base: TerminationBase,
    pub x: Box<dyn Termination>,
    pub cv: Box<dyn Termination>,
    pub f: Box<dyn Termination>,
    pub max_gen: Box<dyn Termination>,
    pub max_evals: Box<dyn Termination>,
}

impl DefaultTermination {
    /// Mirrors `DefaultTermination.__init__(x, cv, f, n_max_gen=1000, n_max_evals=100000)`.
    pub fn new(
        x: Box<dyn Termination>,
        cv: Box<dyn Termination>,
        f: Box<dyn Termination>,
        n_max_gen: Option<f64>,
        n_max_evals: Option<f64>,
    ) -> Self {
        let n_max_gen = n_max_gen.unwrap_or(1000.0);
        let n_max_evals = n_max_evals.unwrap_or(100000.0);
        Self {
            base: TerminationBase::new(),
            x,
            cv,
            f,
            max_gen: Box::new(MaximumGenerationTermination::new(Some(n_max_gen))),
            max_evals: Box::new(MaximumFunctionCallTermination::new(Some(n_max_evals))),
        }
    }
}

impl Termination for DefaultTermination {
    fn base(&self) -> &TerminationBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        &mut self.base
    }

    /// Mirrors `DefaultTermination._update`: runs all criteria and returns the max progress.
    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        let p = [
            self.x.update(algorithm),
            self.cv.update(algorithm),
            self.f.update(algorithm),
            self.max_gen.update(algorithm),
            self.max_evals.update(algorithm),
        ];
        p.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Mirrors `pymoo.termination.default.DefaultSingleObjectiveTermination`.
pub struct DefaultSingleObjectiveTermination {
    pub inner: DefaultTermination,
}

impl DefaultSingleObjectiveTermination {
    /// Mirrors `DefaultSingleObjectiveTermination.__init__(xtol, cvtol, ftol, period, **kwargs)`.
    pub fn new(
        xtol: Option<f64>,
        cvtol: Option<f64>,
        ftol: Option<f64>,
        period: Option<usize>,
        n_max_gen: Option<f64>,
        n_max_evals: Option<f64>,
    ) -> Self {
        let xtol = xtol.unwrap_or(1e-8);
        let cvtol = cvtol.unwrap_or(1e-8);
        let ftol = ftol.unwrap_or(1e-6);
        let period = period.unwrap_or(30);

        let x = Box::new(RobustTermination::new(
            Box::new(DesignSpaceTermination::new(xtol, None)),
            period,
        ));
        let cv = Box::new(RobustTermination::new(
            Box::new(ConstraintViolationTermination::new(cvtol, Some(false))),
            period,
        ));
        let f = Box::new(RobustTermination::new(
            Box::new(SingleObjectiveSpaceTermination::new(ftol, Some(true))),
            period,
        ));

        Self {
            inner: DefaultTermination::new(x, cv, f, n_max_gen, n_max_evals),
        }
    }
}

impl Termination for DefaultSingleObjectiveTermination {
    fn base(&self) -> &TerminationBase {
        self.inner.base()
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        self.inner.base_mut()
    }

    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        self.inner._update(algorithm)
    }
}

/// Mirrors `pymoo.termination.default.DefaultMultiObjectiveTermination`.
pub struct DefaultMultiObjectiveTermination {
    pub inner: DefaultTermination,
}

impl DefaultMultiObjectiveTermination {
    /// Mirrors `DefaultMultiObjectiveTermination.__init__(xtol, cvtol, ftol, n_skip, period, **kwargs)`.
    pub fn new(
        xtol: Option<f64>,
        cvtol: Option<f64>,
        ftol: Option<f64>,
        n_skip: Option<usize>,
        period: Option<usize>,
        n_max_gen: Option<f64>,
        n_max_evals: Option<f64>,
    ) -> Self {
        let xtol = xtol.unwrap_or(0.0005);
        let cvtol = cvtol.unwrap_or(1e-8);
        let ftol = ftol.unwrap_or(0.005);
        let n_skip = n_skip.unwrap_or(5);
        let period = period.unwrap_or(50);

        let x = Box::new(RobustTermination::new(
            Box::new(DesignSpaceTermination::new(xtol, Some(n_skip))),
            period,
        ));
        let cv = Box::new(RobustTermination::new(
            Box::new(ConstraintViolationTermination::new(cvtol, Some(false))),
            period,
        ));
        let f = Box::new(RobustTermination::new(
            Box::new(MultiObjectiveSpaceTermination::new(
                ftol,
                Some(true),
                Some(n_skip),
            )),
            period,
        ));

        Self {
            inner: DefaultTermination::new(x, cv, f, n_max_gen, n_max_evals),
        }
    }
}

impl Termination for DefaultMultiObjectiveTermination {
    fn base(&self) -> &TerminationBase {
        self.inner.base()
    }

    fn base_mut(&mut self) -> &mut TerminationBase {
        self.inner.base_mut()
    }

    fn _update(&mut self, algorithm: &mut dyn Algorithm) -> f64 {
        self.inner._update(algorithm)
    }
}
