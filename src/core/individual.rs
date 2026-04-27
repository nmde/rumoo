use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

/*
Module containing infrastructure for representing individuals in
population-based optimization algorithms.
*/
use ndarray::Array1;

#[derive(Clone)]
struct CvConstraintConfig {
    scale: Option<f64>,
    eps: Option<f64>,
    pow: Option<f64>,
    func: Option<fn(&Array1<f64>) -> f64>,
}

#[derive(Clone)]
struct IndividualConfig {
    cache: bool,
    cv_eps: f64,
    cv_ieq: CvConstraintConfig,
    cv_eq: CvConstraintConfig,
}

impl Default for IndividualConfig {
    /// Get default constraint violation configuration settings.
    ///
    /// Returns
    /// -------
    /// out : dict
    ///     A dictionary of default constraint violation settings.
    fn default() -> Self {
        Self {
            cache: true,
            cv_eps: 0.0,
            cv_ieq: CvConstraintConfig {
                scale: None,
                eps: Some(0.0),
                pow: None,
                func: Some(|a| a.sum()),
            },
            cv_eq: CvConstraintConfig {
                scale: None,
                eps: Some(1e-4),
                pow: None,
                func: Some(|a| a.sum()),
            },
        }
    }
}

#[derive(Clone)]
pub struct Individual {
    pub x: Array1<f64>,
    f: Array1<f64>,
    g: Array1<f64>,
    h: Array1<f64>,
    df: Array1<f64>,
    dg: Array1<f64>,
    dh: Array1<f64>,
    ddf: Array1<f64>,
    ddg: Array1<f64>,
    ddh: Array1<f64>,
    cv_cache: RefCell<Option<Array1<f64>>>,
    evaluated: Option<HashSet<String>>,
    data: Option<HashMap<String, f64>>,
    config: IndividualConfig,
}

/// Base class for representing an individual in a population-based
/// optimization algorithm.
impl Individual {
    /*
        # function: function to generate default configuration settings
        default_config = default_config
    */

    /// Constructor for the ``Invididual`` class.
    ///
    /// Parameters
    /// ----------
    /// config : dict, None
    ///     A dictionary of configuration metadata.
    ///     If ``None``, use a class-dependent default configuration.
    /// kwargs : Any
    ///     Additional keyword arguments containing data which is to be stored
    ///     in the ``Individual``.
    pub fn new(config: Option<IndividualConfig>, data: Option<HashMap<String, f64>>) -> Self {
        let empty = Array1::zeros(0);
        let mut this = Self {
            // set decision variable vector to None
            x: empty.clone(),
            // set values objective(s), inequality constraint(s), equality
            // contstraint(s) to None
            f: empty.clone(),
            g: empty.clone(),
            h: empty.clone(),
            // set first derivatives of objective(s), inequality constraint(s),
            // equality contstraint(s) to None
            df: empty.clone(),
            dg: empty.clone(),
            dh: empty.clone(),
            // set second derivatives of objective(s), inequality constraint(s),
            // equality contstraint(s) to None
            ddf: empty.clone(),
            ddg: empty.clone(),
            ddh: empty,
            // set constraint violation value to None
            cv_cache: RefCell::new(None),
            evaluated: None,
            // a local storage for data
            data,
            // the config for this individual
            config: config.unwrap_or(IndividualConfig::default()),
        };

        // initialize all the local variables
        this.reset(None);
        this
    }

    /// Reset the value of objective(s), inequality constraint(s), equality
    /// constraint(s), their first and second derivatives, the constraint
    /// violation, and the metadata to empty values.
    ///
    /// Parameters
    /// ----------
    /// data : bool
    ///     Whether to reset metadata associated with the ``Individiual``.
    fn reset(&mut self, data: Option<bool>) {
        let empty = Array1::zeros(0);
        // design variables
        self.x = empty.clone();

        // objectives and constraint values
        self.f = empty.clone();
        self.g = empty.clone();
        self.h = empty.clone();

        // first order derivation
        self.df = empty.clone();
        self.dg = empty.clone();
        self.dh = empty.clone();

        // second order derivation
        self.ddf = empty.clone();
        self.ddg = empty.clone();
        self.ddh = empty;

        // if the constraint violation value to be used
        *self.cv_cache.borrow_mut() = None;

        if data.unwrap_or(true) {
            self.data = None;
        }

        // a set storing what has been evaluated
        self.evaluated = None;
    }
    /*
        def has(
                self,
                key: str,
            ) -> bool:
            """
        Determine whether an individual has a provided key or not.

        Parameters
        ----------
        key : str
            The key for which to test.
        
        Returns
        -------
        out : bool
            Whether the ``Individual`` has the provided key.
        """
            return hasattr(self.__class__, key) or key in self.data

        @property
        def F(self) -> np.ndarray:
            """
        Get the objective function vector for an individual.

        Returns
        -------
        out : np.ndarray
            The objective function vector for the individual.
        """
            return self._F

        @F.setter
        def F(self, value: np.ndarray) -> None:
            """
        Set the objective function vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The objective function vector for the individual.
        """
            self._F = value

        @property
        def G(self) -> np.ndarray:
            """
        Get the inequality constraint vector for an individual.

        Returns
        -------
        out : np.ndarray
            The inequality constraint vector for the individual.
        """
            return self._G

        @G.setter
        def G(self, value: np.ndarray) -> None:
            """
        Set the inequality constraint vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The inequality constraint vector for the individual.
        """
            self._G = value

        @property
        def H(self) -> np.ndarray:
            """
        Get the equality constraint vector for an individual.

        Returns
        -------
        out : np.ndarray
            The equality constraint vector for the individual.
        """
            return self._H

        @H.setter
        def H(self, value: np.ndarray) -> None:
            """
        Get the equality constraint vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The equality constraint vector for the individual.
        """
            self._H = value
    */

    /// Get the constraint violation vector for an individual by either reading
    /// it from the cache or calculating it.
    ///
    /// Returns
    /// -------
    /// out : np.ndarray
    ///     The constraint violation vector for an individual.
    fn cv(&self) -> Array1<f64> {
        if self.config.cache && self.cv_cache.borrow().is_some() {
            self.cv_cache.borrow().clone().unwrap()
        } else {
            let val = calc_cv(&self.g, &self.h, Some(&self.config));
            let cv = Array1::from_elem(1, val);
            *self.cv_cache.borrow_mut() = Some(cv.clone());
            cv
        }
    }

    /*
        @CV.setter
        def CV(self, value: np.ndarray) -> None:
            """
        Set the constraint violation vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The constraint violation vector for the individual.
        """
            self._CV = value
    */

    /// Get whether an individual is feasible for each constraint.
    ///
    /// Returns
    /// -------
    /// out : np.ndarray
    ///     An array containing whether each constraint is feasible for an
    ///     individual.
    pub fn feas(&self) -> Array1<bool> {
        self.cv().mapv(|v| v <= self.config.cv_eps)
    }

    /*
        # -------------------------------------------------------
        # Gradients
        # -------------------------------------------------------

        @property
        def dF(self) -> np.ndarray:
            """
        Get the objective function vector first derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The objective function vector first derivatives for the individual.
        """
            return self._dF

        @dF.setter
        def dF(self, value: np.ndarray) -> None:
            """
        Set the objective function vector first derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The objective function vector first derivatives for the individual.
        """
            self._dF = value

        @property
        def dG(self) -> np.ndarray:
            """
        Get the inequality constraint(s) first derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The inequality constraint(s) first derivatives for the individual.
        """
            return self._dG

        @dG.setter
        def dG(self, value: np.ndarray) -> None:
            """
        Set the inequality constraint(s) first derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The inequality constraint(s) first derivatives for the individual.
        """
            self._dG = value

        @property
        def dH(self) -> np.ndarray:
            """
        Get the equality constraint(s) first derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The equality constraint(s) first derivatives for the individual.
        """
            return self._dH

        @dH.setter
        def dH(self, value: np.ndarray) -> None:
            """
        Set the equality constraint(s) first derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The equality constraint(s) first derivatives for the individual.
        """
            self._dH = value

        # -------------------------------------------------------
        # Hessians
        # -------------------------------------------------------

        @property
        def ddF(self) -> np.ndarray:
            """
        Get the objective function vector second derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The objective function vector second derivatives for the individual.
        """
            return self._ddF

        @ddF.setter
        def ddF(self, value: np.ndarray) -> None:
            """
        Set the objective function vector second derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The objective function vector second derivatives for the individual.
        """
            self._ddF = value

        @property
        def ddG(self) -> np.ndarray:
            """
        Get the inequality constraint(s) second derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The inequality constraint(s) second derivatives for the individual.
        """
            return self._ddG

        @ddG.setter
        def ddG(self, value: np.ndarray) -> None:
            """
        Set the inequality constraint(s) second derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The inequality constraint(s) second derivatives for the individual.
        """
            self._ddG = value

        @property
        def ddH(self) -> np.ndarray:
            """
        Get the equality constraint(s) second derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The equality constraint(s) second derivatives for the individual.
        """
            return self._ddH

        @ddH.setter
        def ddH(self, value: np.ndarray) -> None:
            """
        Set the equality constraint(s) second derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The equality constraint(s) second derivatives for the individual.
        """
            self._ddH = value

        # -------------------------------------------------------
        # Convenience (value instead of array)
        # -------------------------------------------------------

        @property
        def x(self) -> np.ndarray:
            """
        Convenience property. Get the decision vector for an individual.

        Returns
        -------
        out : np.ndarray
            The decision variable for the individual.
        """
            return self.X

        @property
        def f(self) -> float:
            """
        Convenience property. Get the first objective function value for an individual.

        Returns
        -------
        out : float
            The first objective function value for the individual.
        """
            return self.F[0]

        @property
        def cv(self) -> Union[float,None]:
            """
        Convenience property. Get the first constraint violation value for an 
        individual by either reading it from the cache or calculating it.

        Returns
        -------
        out : float, None
            The constraint violation vector for an individual.
        """
            if self.CV is None:
                return None
            else:
                return self.CV[0]

        @property
        def feas(self) -> bool:
            """
        Convenience property. Get whether an individual is feasible for the 
        first constraint.

        Returns
        -------
        out : bool
            Whether an individual is feasible for the first constraint.
        """
            return self.FEAS[0]

        # -------------------------------------------------------
        # Deprecated
        # -------------------------------------------------------

        @property
        def feasible(self) -> np.ndarray:
            """
        Deprecated. Get whether an individual is feasible for each constraint.

        Returns
        -------
        out : np.ndarray
            An array containing whether each constraint is feasible for an 
            individual.
        """
            warn(
                "The ``feasible`` property for ``pymoo.core.individual.Individual`` is deprecated",
                DeprecationWarning,
                stacklevel = 2,
            )
            return self.FEAS

        # -------------------------------------------------------
        # Other Functions
        # -------------------------------------------------------

        def set_by_dict(
                self,
                **kwargs: Any
            ) -> None:
            """
        Set an individual's data or metadata using values in a dictionary.

        Parameters
        ----------
        kwargs : Any
            Keyword arguments defining the data to set.
        """
            for k, v in kwargs.items():
                self.set(k, v)

        def set(
                self,
                key: str,
                value: object,
            ) -> 'Individual':
            """
        Set an individual's data or metadata based on a key and value.

        Parameters
        ----------
        key : str
            Key of the data for which to set.
        value : object
            Value of the data for which to set.
        
        Returns
        -------
        out : Individual
            A reference to the ``Individual`` for which values were set.
        """
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.data[key] = value
            return self

        def get(
                self,
                *keys: str,
            ) -> Union[tuple,object]:
            """
        Get the values for one or more keys for an individual.

        Parameters
        ----------
        keys : str
            Keys for which to get values.

        Returns
        -------
        out : tuple, object
            If more than one key provided, return a ``tuple`` of retrieved values.
            If a single key provided, return the retrieved value.
        """
            ret = []

            for key in keys:
                if hasattr(self, key):
                    v = getattr(self, key)
                elif key in self.data:
                    v = self.data[key]
                else:
                    v = None

                ret.append(v)

            if len(ret) == 1:
                return ret[0]
            else:
                return tuple(ret)

        def duplicate(
                self,
                key: str,
                new_key: str,
            ) -> None:
            """
        Duplicate a key to a new key.

        Parameters
        ----------
        key : str
            Name of the key to duplicated.
        new_key : str
            Name of the key to which to duplicate the original key.
        """
            self.set(new_key, self.get(key))

        def new(self) -> 'Individual':
            """
        Create a new instance of this class.

        Returns
        -------
        out : Individual
            A new instance of an ``Individual``.
        """
            return self.__class__()

        def copy(
                self,
                other: Optional['Individual'] = None,
                deep: bool = True,
            ) -> 'Individual':
            """
        Copy an individual.

        Parameters
        ----------
        other : Individual, None
            The individual to copy. If ``None``, assumed to be self.
        deep : bool
            Whether to deep copy the individual.
        
        Returns
        -------
        out : Individual
            A copy of the individual.
        """
            obj = self.new()

            # if not provided just copy yourself
            if other is None:
                other = self

            # the data the new object needs to have
            D = other.__dict__

            # if it should be a deep copy do it
            if deep:
                D = copy.deepcopy(D)

            for k, v in D.items():
                obj.__dict__[k] = v

            return obj
    */
}

/// Calculate the constraint violation(s) for a set of inequality constraint(s),
/// equality constraint(s), and a scoring configuration.
///
/// Parameters
/// ----------
/// G : np.ndarray, None
///     A vector of inequality constraint(s).
/// H : np.ndarray, None
///     A vector of equality constraint(s).
/// config : dict, None
///     A dictionary of constraint violation scoring configuration settings.
///
/// Returns
/// -------
/// out : np.ndarray
///     An array of constraint violations for each objective.
fn calc_cv(g: &Array1<f64>, h: &Array1<f64>, config: Option<&IndividualConfig>) -> f64 {
    let def_config = IndividualConfig::default();
    let config = config.unwrap_or(&def_config);

    constr_to_cv(
        &g,
        config.cv_ieq.eps,
        config.cv_ieq.scale,
        config.cv_ieq.pow,
        None,
    ) + constr_to_cv(
        &h.mapv(f64::abs),
        config.cv_eq.eps,
        config.cv_eq.scale,
        config.cv_eq.pow,
        None,
    )
}

/// Convert a constraint to a constraint violation.
///
/// c : np.ndarray
///     An array of constraint violations.
/// eps : float
///     Error tolerance bounds.
/// scale : float, None
///     The scale to apply to a constraint violation.
///     If ``None``, no scale alteration is applied.
/// pow : float, None
///     A power to apply to a constraint violation.
///     If ``None``, no power alteration is applied.
/// func : function
///     A function to convert multiple constraint violations into a single score.
fn constr_to_cv(
    c: &Array1<f64>,
    eps: Option<f64>,
    scale: Option<f64>,
    pow: Option<f64>,
    func: Option<fn(&Array1<f64>) -> f64>,
) -> f64 {
    let eps = eps.unwrap_or(0.0);
    if c.is_empty() {
        return 0.0;
    }

    // subtract eps to allow some violation and then zero out all values less than zero
    let mut c = c.mapv(|v| (v - eps).max(0.0));

    // apply init_simplex_scale if necessary
    if scale.is_some() {
        c = c / scale.unwrap();
    }

    // if a pow factor has been provided
    if pow.is_some() {
        c = c.powf(pow.unwrap());
    }

    match func {
        None => c.mean().unwrap_or(0.0),
        Some(f) => f(&c),
    }
}
