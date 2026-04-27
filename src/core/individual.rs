use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use anyhow::anyhow;
use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub enum Value {
    Str(String),
    Float(f64),
    Int(i64),
    Bool(bool),
    FloatArray(Array1<f64>),
    IntArray(Array1<i64>),
    BoolArray(Array1<bool>),
    FloatMatrix(Array2<f64>),
}

/// Per-constraint-type violation scoring settings.
/// Mirrors the `cv_ieq` / `cv_eq` sub-dicts inside `default_config()`.
pub struct CvConstraintConfig {
    pub scale: Option<f64>,
    pub eps: Option<f64>,
    pub pow: Option<f64>,
    /// Aggregation over the constraint vector (Python: `func=np.sum`).
    pub func: Option<fn(&Array1<f64>) -> f64>,
}

/// Top-level constraint violation configuration.
/// Mirrors the dict returned by `default_config()`.
#[derive(Clone)]
pub struct CvConfig {
    pub cache: bool,
    pub cv_eps: Option<f64>,
    pub cv_ieq: CvConstraintConfig,
    pub cv_eq: CvConstraintConfig,
}

impl Default for CvConstraintConfig {
    fn default() -> Self {
        Self {
            scale: None,
            eps: Some(0.0),
            pow: None,
            func: Some(|a| a.sum()),
        }
    }
}

/// Mirrors `default_config()`:
/// `cv_ieq` uses eps=0.0, `cv_eq` uses eps=1e-4, both aggregate with sum.
impl Default for CvConfig {
    fn default() -> Self {
        Self {
            cache: true,
            cv_eps: Some(0.0),
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

/// Base class for representing an individual in a population-based
/// optimization algorithm.
///
/// Mirrors `pymoo.core.individual.Individual`.
pub struct Individual {
    // decision variables  (Python: _X / X)
    pub x: Option<Array1<f64>>,

    // objectives (Python: _F / F)
    pub f: Option<Array1<f64>>,
    // inequality constraints — G ≤ 0 is satisfied (Python: _G / G)
    pub g: Option<Array1<f64>>,
    // equality constraints (Python: _H / H)
    pub h: Option<Array1<f64>>,

    // first-order derivatives (Python: _dF / dF, _dG / dG, _dH / dH)
    pub df: Option<Array1<f64>>,
    pub dg: Option<Array1<f64>>,
    pub dh: Option<Array1<f64>>,

    // second-order derivatives (Python: _ddF / ddF, _ddG / ddG, _ddH / ddH)
    pub ddf: Option<Array1<f64>>,
    pub ddg: Option<Array1<f64>>,
    pub ddh: Option<Array1<f64>>,

    // cached constraint violation — None until first access (Python: _CV / CV)
    cv_cache: RefCell<Option<Array1<f64>>>,

    // attributes that have been evaluated (Python: evaluated: set())
    pub evaluated: HashSet<IndividualField>,

    // arbitrary per-individual storage (rank, crowding, etc.) (Python: data: {})
    pub data: HashMap<String, Value>,

    pub config: CvConfig,
}

impl Default for Individual {
    fn default() -> Self {
        Self::new(None)
    }
}

pub enum IndividualField {
    X,
    F,
    G,
    H,
    CV,
    DF,
    DG,
    DH,
    DDF,
    DDG,
    DDH,
    Feas,
    DataField(String),
}

impl Individual {
    pub fn new(config: Option<CvConfig>) -> Self {
        let empty: Array1<f64> = Array1::zeros(0);
        let mut ind = Self {
            x: None,
            f: None,
            g: None,
            h: None,
            df: None,
            dg: None,
            dh: None,
            ddf: None,
            ddg: None,
            ddh: None,
            cv_cache: RefCell::new(None),
            evaluated: HashSet::new(),
            data: HashMap::new(),
            config: config.unwrap_or_default(),
        };
        // mirrors self.reset() call in __init__
        ind.reset(Some(true));
        ind
    }

    /// Reset all fields to empty. Mirrors `Individual.reset(data=True)`.
    pub fn reset(&mut self, reset_data: Option<bool>) {
        let reset_data = reset_data.unwrap_or(true);
        let empty: Array1<f64> = Array1::zeros(0);

        // design variables
        self.x = None;

        // objectives and constraint values
        self.f = None;
        self.g = None;
        self.h = None;

        // first-order derivatives
        self.df = None;
        self.dg = None;
        self.dh = None;

        // second-order derivatives
        self.ddf = None;
        self.ddg = None;
        self.ddh = None;

        // invalidate the constraint violation cache
        *self.cv_cache.borrow_mut() = None;

        if reset_data {
            self.data = HashMap::new();
        }

        self.evaluated = HashSet::new();
    }

    /// True if the individual has the given attribute key.
    /// Mirrors `Individual.has(key)`.
    pub fn has(&self, key: &str) -> bool {
        matches!(
            key.to_lowercase().as_str(),
            "df" | "dg" | "dh" | "ddf" | "ddg" | "ddh" | "x" | "f" | "g" | "h" | "cv" | "feas"
        ) || self.data.contains_key(key)
    }

    // -------------------------------------------------------
    // Values
    // -------------------------------------------------------

    /// Constraint violation vector of length 1.
    /// Lazily computed from G and H and cached until G or H change.
    /// Mirrors `Individual.CV` (the property, not the setter).
    pub fn cv(&self) -> Array1<f64> {
        if self.config.cache {
            if let Some(ref cached) = *self.cv_cache.borrow() {
                return cached.clone();
            }
        }
        let val = calc_cv(self.g, self.h, Some(&self.config));
        let cv = Array1::from_elem(1, val);
        *self.cv_cache.borrow_mut() = Some(cv.clone());
        cv
    }

    /// Per-constraint feasibility. Mirrors `Individual.FEAS`.
    pub fn feas(&self) -> Array1<bool> {
        let eps = self.config.cv_eps;
        self.cv().mapv(|v| v <= eps.unwrap_or(0.0))
    }

    // -------------------------------------------------------
    // Other Functions
    // -------------------------------------------------------

    /// Create a fresh default instance of this type.
    /// Set multiple attributes from a dict-like slice.
    /// Mirrors `Individual.set_by_dict(**kwargs)`.
    pub fn set_by_dict(&mut self, attrs: &[(IndividualField, Value)]) {
        for (key, value) in attrs {
            self.set(key, value.clone());
        }
    }

    /// Set a named attribute by key.
    ///
    /// Known fields (X, F, G, H, CV, and the derivative fields) are stored
    /// directly; anything else goes into `self.data`.
    /// Setting F, G, or H invalidates the CV cache.
    /// Mirrors `Individual.set(key, value)`.
    pub fn set(&mut self, key: &IndividualField, value: Value) -> Result<&mut Self> {
        match key {
            IndividualField::X => {
                if let Value::FloatArray(arr) = value {
                    self.x = Some(arr);
                }
            }
            IndividualField::F => {
                if let Value::FloatArray(arr) = value {
                    self.f = Some(arr);
                    *self.cv_cache.borrow_mut() = None;
                }
            }
            IndividualField::G => {
                if let Value::FloatArray(arr) = value {
                    self.g = Some(arr);
                    *self.cv_cache.borrow_mut() = None;
                }
            }
            IndividualField::H => {
                if let Value::FloatArray(arr) = value {
                    self.h = Some(arr);
                    *self.cv_cache.borrow_mut() = None;
                }
            }
            IndividualField::CV => {
                if let Value::FloatArray(arr) = value {
                    *self.cv_cache.borrow_mut() = Some(arr);
                }
            }
            IndividualField::DF => {
                if let Value::FloatArray(arr) = value {
                    self.df = Some(arr);
                }
            }
            IndividualField::DG => {
                if let Value::FloatArray(arr) = value {
                    self.dg = Some(arr);
                }
            }
            IndividualField::DH => {
                if let Value::FloatArray(arr) = value {
                    self.dh = Some(arr);
                }
            }
            IndividualField::DDF => {
                if let Value::FloatArray(arr) = value {
                    self.ddf = Some(arr);
                }
            }
            IndividualField::DDG => {
                if let Value::FloatArray(arr) = value {
                    self.ddg = Some(arr);
                }
            }
            IndividualField::DDH => {
                if let Value::FloatArray(arr) = value {
                    self.ddh = Some(arr);
                }
            }
            IndividualField::DataField(key) => {
                self.data.insert(key.to_string(), value);
            }
            _ => {
                return Err(anyhow!("Given field cannot be set."));
            }
        }
        Ok(self)
    }

    /// Get a single named attribute.
    /// Returns `None` if the key is absent.
    /// Mirrors `Individual.get(key)` when called with one argument.
    pub fn get(&self, key: &IndividualField) -> Option<Value> {
        match key {
            IndividualField::X => {
                if self.x.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.x.unwrap()))
                }
            }
            IndividualField::F => {
                if self.f.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.f.unwrap()))
                }
            }
            IndividualField::G => {
                if self.g.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.g.unwrap()))
                }
            }
            IndividualField::H => {
                if self.h.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.h.unwrap()))
                }
            }
            IndividualField::CV => Some(Value::FloatArray(self.cv())),
            IndividualField::FEAS => Some(Value::BoolArray(self.feas())),
            IndividualField::DF => {
                if self.df.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.df.unwrap()))
                }
            }
            IndividualField::DG => {
                if self.dg.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.dg.unwrap()))
                }
            }
            IndividualField::DH => {
                if self.dh.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.dh.unwrap()))
                }
            }
            IndividualField::DDF => {
                if self.ddf.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.ddf.unwrap()))
                }
            }
            IndividualField::DDG => {
                if self.ddg.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.ddg.unwrap()))
                }
            }
            IndividualField::DDH => {
                if self.ddh.is_none() {
                    None
                } else {
                    Some(Value::FloatArray(self.ddh.unwrap()))
                }
            }
            IndividualField::DataField(key) => self.data.get(key).cloned(),
        }
    }

    /// Get two named attributes as a tuple.
    ///
    /// Mirrors Python's variadic `ind.get("rank", "crowding")` when called
    /// with two arguments.
    pub fn get_tuple(
        &self,
        key1: &IndividualField,
        key2: &IndividualField,
    ) -> (Option<Value>, Option<Value>) {
        (self.get_attr(key1), self.get(key2))
    }

    /// Copy `key` to `new_key`. Mirrors `Individual.duplicate(key, new_key)`.
    pub fn duplicate(&mut self, key: &IndividualField, new_key: &IndividualField) {
        if let Some(val) = self.get(key) {
            self.set(new_key, val);
        }
    }

    /// Deep-copy this individual (or `other` if provided).
    /// Mirrors `Individual.copy(other=None, deep=True)`.
    pub fn copy(&self, other: Option<&Individual>) -> Individual {
        let source = other.unwrap_or(self);
        source.clone()
    }
}

// RefCell is not Clone by default in a straightforward way for derived Clone,
// but we can implement Clone manually to deep-copy the cache.
impl Clone for Individual {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            f: self.f.clone(),
            g: self.g.clone(),
            h: self.h.clone(),
            df: self.df.clone(),
            dg: self.dg.clone(),
            dh: self.dh.clone(),
            ddf: self.ddf.clone(),
            ddg: self.ddg.clone(),
            ddh: self.ddh.clone(),
            cv_cache: RefCell::new(self.cv_cache.borrow().clone()),
            evaluated: self.evaluated.clone(),
            data: self.data.clone(),
            config: self.config.clone(),
        }
    }
}

/// Calculate the constraint violation scalar for one individual.
///
/// Mirrors `pymoo.core.individual.calc_cv(G, H, config)`.
/// Returns `ieq_cv + eq_cv` (summed across all constraints).
pub fn calc_cv(g: Option<Array1<f64>>, h: Option<Array1<f64>>, config: Option<&CvConfig>) -> f64 {
    let default = CvConfig::default();
    let config = config.unwrap_or(&default);

    let empty: Array1<f64> = Array1::zeros(0);
    let g = g.unwrap_or(empty);
    let h = h.unwrap_or(empty);

    let ieq_cv = if g.is_empty() {
        0.0
    } else {
        constr_to_cv(
            &g,
            config.cv_ieq.eps,
            config.cv_ieq.scale,
            config.cv_ieq.pow,
            config.cv_ieq.func,
        )
    };

    // equality constraints: use |H|
    let h_abs: Array1<f64> = h.mapv(f64::abs);
    let eq_cv = if h_abs.is_empty() {
        0.0
    } else {
        constr_to_cv(
            &h_abs,
            config.cv_eq.eps,
            config.cv_eq.scale,
            config.cv_eq.pow,
            config.cv_eq.func,
        )
    };

    ieq_cv + eq_cv
}

/// Convert a constraint vector into a scalar constraint violation score.
///
/// Steps (mirrors `pymoo.core.individual.constr_to_cv`):
///   1. `c = max(0, c - eps)`  — allow tolerance, clip negatives
///   2. divide by `scale` if given
///   3. raise to `pow` if given
///   4. apply `func` to reduce to a scalar
pub fn constr_to_cv(
    c: &Array1<f64>,
    eps: Option<f64>,
    scale: Option<f64>,
    pow: Option<f64>,
    func: Option<fn(&Array1<f64>) -> f64>,
) -> f64 {
    if c.is_empty() {
        return 0.0;
    }
    let eps = eps.unwrap_or(0.0);

    // subtract eps to allow some violation and then zero out all values less than zero
    let mut c: Array1<f64> = c.mapv(|v| (v - eps).max(0.0));

    // apply scale if necessary
    if let Some(s) = scale {
        c.mapv_inplace(|v| v / s);
    }

    // if a pow factor has been provided
    if let Some(p) = pow {
        c.mapv_inplace(|v| v.powf(p));
    }

    if func.is_some() {
        return func(&c);
    }
    c.mean().unwrap_or(0.0)
}
