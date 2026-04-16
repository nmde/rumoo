use std::ops::{Index, IndexMut};

use ndarray::{Array1, Array2};

use crate::core::individual::{Individual, IndividualField, Value, calc_cv as calc_cv_individual};

/// A collection of `Individual`s.
/// Mirrors `pymoo.core.population.Population` (a numpy array of individuals).
pub struct Population {
    individuals: Vec<Individual>,
}

impl Population {
    pub fn new(individuals: Vec<Individual>) -> Self {
        Self { individuals }
    }

    /// True if all individuals have the given key.
    pub fn has(&self, key: &str) -> bool {
        self.individuals.iter().all(|ind| ind.has(key))
    }

    /// Map a function over individuals, collecting results.
    pub fn collect<T, F: Fn(&Individual) -> T>(&self, func: F) -> Vec<T> {
        self.individuals.iter().map(func).collect()
    }

    pub fn apply<F: Fn(&Individual)>(&self, func: F) {
        self.individuals.iter().for_each(func);
    }

    /// Set a scalar attribute on every individual in the population.
    /// Mirrors `pop.set("rank", k)` where k is a scalar.
    pub fn set(&mut self, key: &IndividualField, value: Value) {
        for ind in &mut self.individuals {
            ind.set(key, value.clone());
        }
    }

    /// Set a per-individual attribute from a `Vec<Value>` whose length equals the population size.
    /// Mirrors `pop.set("crowding", crowding_array)`.
    pub fn set_each(&mut self, key: &IndividualField, values: Vec<Value>) {
        assert_eq!(
            values.len(),
            self.individuals.len(),
            "Population::set_each: values length must match population size"
        );
        for (ind, val) in self.individuals.iter_mut().zip(values) {
            ind.set(key, val);
        }
    }

    /// Collect a named attribute from all individuals and stack into an array.
    /// Mirrors `pop.get("F")` → `Array2<f64>`, `pop.get("CV")` → `Array1<f64>`, etc.
    pub fn get(&self, key: &IndividualField) -> Value {
        if self.individuals.is_empty() {
            return Value::FloatMatrix(Array2::zeros((0, 0)));
        }
        match key {
            IndividualField::F => {
                let n = self.individuals.len();
                let m = self.individuals[0].f.len();
                let mut out = Array2::zeros((n, m));
                for (i, ind) in self.individuals.iter().enumerate() {
                    out.row_mut(i).assign(&ind.f);
                }
                Value::FloatMatrix(out)
            }
            IndividualField::G => {
                let n = self.individuals.len();
                let m = self.individuals[0].g.len();
                let mut out = Array2::zeros((n, m));
                for (i, ind) in self.individuals.iter().enumerate() {
                    out.row_mut(i).assign(&ind.g);
                }
                Value::FloatMatrix(out)
            }
            IndividualField::H => {
                let n = self.individuals.len();
                let m = self.individuals[0].h.len();
                let mut out = Array2::zeros((n, m));
                for (i, ind) in self.individuals.iter().enumerate() {
                    out.row_mut(i).assign(&ind.h);
                }
                Value::FloatMatrix(out)
            }
            IndividualField::X => {
                let n = self.individuals.len();
                let m = self.individuals[0].x.len();
                let mut out = Array2::zeros((n, m));
                for (i, ind) in self.individuals.iter().enumerate() {
                    out.row_mut(i).assign(&ind.x);
                }
                Value::FloatMatrix(out)
            }
            IndividualField::CV => {
                let cv: Vec<f64> = self.individuals.iter()
                    .map(|ind| ind.cv()[0])
                    .collect();
                Value::FloatArray(Array1::from_vec(cv))
            }
            IndividualField::Feas => {
                let feas: Vec<bool> = self.individuals.iter()
                    .map(|ind| ind.feas()[0])
                    .collect();
                Value::BoolArray(Array1::from_vec(feas))
            }
            _ => {
                // Generic: collect scalar values from the data store.
                // Try float first, then int.
                let raw: Vec<Option<Value>> = self.individuals.iter()
                    .map(|ind| ind.data.get(key).cloned())
                    .collect();

                let as_floats: Option<Vec<f64>> = raw.iter().map(|v| match v {
                    Some(Value::Float(f)) => Some(*f),
                    Some(Value::Int(i)) => Some(*i as f64),
                    _ => None,
                }).collect();

                if let Some(fs) = as_floats {
                    return Value::FloatArray(Array1::from_vec(fs));
                }

                let as_ints: Option<Vec<i64>> = raw.iter().map(|v| match v {
                    Some(Value::Int(i)) => Some(*i),
                    _ => None,
                }).collect();

                if let Some(is) = as_ints {
                    return Value::IntArray(Array1::from_vec(is));
                }

                Value::FloatArray(Array1::zeros(self.individuals.len()))
            }
        }
    }

    /// Collect multiple named attributes in one call.
    /// Mirrors Python's `pop.get("G", "H")` returning a tuple.
    pub fn get_many(&self, keys: &[&IndividualField]) -> Vec<Value> {
        keys.iter().map(|k| self.get(k)).collect()
    }

    pub fn merge(a: Population, b: Population) -> Population {
        merge(a, b)
    }

    pub fn empty(size: usize) -> Self {
        Self {
            individuals: (0..size).map(|_| Individual::new()).collect(),
        }
    }

    /// Mirrors `Population.new("X", x_matrix, ...)`.
    /// Accepts interleaved `(key, Value)` pairs; matrix values are split per-row.
    pub fn new_with_attrs(attrs: &[(&IndividualField, Value)]) -> Self {
        if attrs.is_empty() {
            return Self::empty(0);
        }
        let size = match &attrs[0].1 {
            Value::FloatMatrix(m) => m.nrows(),
            Value::FloatArray(a) => a.len(),
            Value::IntArray(a) => a.len(),
            Value::BoolArray(a) => a.len(),
            _ => 1,
        };
        let mut pop = Self::empty(size);
        for (key, value) in attrs {
            match value {
                Value::FloatMatrix(m) => {
                    for (i, ind) in pop.individuals.iter_mut().enumerate() {
                        ind.set(key, Value::FloatArray(m.row(i).to_owned()));
                    }
                }
                other => pop.set(key, other.clone()),
            }
        }
        pop
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<Individual> {
        self.individuals.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Individual> {
        self.individuals.iter_mut()
    }

    /// Fancy-index into the population: `pop.select(&[0, 2, 5])`.
    /// Mirrors `pop[[0, 2, 5]]` in numpy.
    pub fn select(&self, indices: &[usize]) -> Self {
        Self {
            individuals: indices.iter()
                .map(|&i| self.individuals[i].clone())
                .collect(),
        }
    }

    /// Boolean-mask selection.
    /// Mirrors `pop[mask]` in numpy.
    pub fn select_where(&self, mask: &Array1<bool>) -> Self {
        assert_eq!(mask.len(), self.individuals.len());
        Self {
            individuals: self.individuals.iter()
                .zip(mask.iter())
                .filter_map(|(ind, &keep)| if keep { Some(ind.clone()) } else { None })
                .collect(),
        }
    }
}

impl Index<usize> for Population {
    type Output = Individual;
    fn index(&self, i: usize) -> &Individual {
        &self.individuals[i]
    }
}

impl IndexMut<usize> for Population {
    fn index_mut(&mut self, i: usize) -> &mut Individual {
        &mut self.individuals[i]
    }
}

pub enum PopulationInput {
    Pop(Population),
    Matrix(Array2<f64>),
    Individual(Individual),
}

/// Convert an array or individual to a `Population`.
/// Mirrors `pymoo.core.population.pop_from_array_or_individual`.
pub fn pop_from_array_or_individual(
    input: PopulationInput,
    base: Option<Population>,
) -> Option<Population> {
    let base = base.unwrap_or_else(|| Population::empty(0));
    match input {
        PopulationInput::Pop(p) => Some(p),
        PopulationInput::Matrix(m) => {
            Some(base.new_with_attrs(&[("X", Value::FloatMatrix(m))]))
        }
        PopulationInput::Individual(ind) => {
            let mut pop = Population::empty(1);
            pop[0] = ind;
            Some(pop)
        }
    }
}

/// Concatenate two populations.
/// Mirrors `pymoo.core.population.merge`.
pub fn merge(a: Population, b: Population) -> Population {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let mut individuals = a.individuals;
    individuals.extend(b.individuals);
    Population { individuals }
}

pub struct CvConstraintConfig {
    pub scale: Option<f64>,
    pub eps: f64,
    pub pow: Option<f64>,
}

pub struct CvConfig {
    pub cache: bool,
    pub cv_eps: f64,
    pub cv_ieq: CvConstraintConfig,
    pub cv_eq: CvConstraintConfig,
}

impl Default for CvConfig {
    fn default() -> Self {
        Self {
            cache: true,
            cv_eps: 0.0,
            cv_ieq: CvConstraintConfig { scale: None, eps: 0.0, pow: None },
            cv_eq: CvConstraintConfig { scale: None, eps: 1e-4, pow: None },
        }
    }
}

/// Compute the constraint violation array for a whole population.
/// Mirrors the module-level `calc_cv(pop, config)` in population.py.
pub fn calc_cv(pop: &Population, config: Option<&CvConfig>) -> Array1<f64> {
    let default_config = CvConfig::default();
    let config = config.unwrap_or(&default_config);

    let g_val = pop.get(&IndividualField::G);
    let h_val = pop.get(&IndividualField::H);

    let g_mat = match &g_val {
        Value::FloatMatrix(m) => Some(m),
        _ => None,
    };
    let h_mat = match &h_val {
        Value::FloatMatrix(m) => Some(m),
        _ => None,
    };

    let n = pop.len();
    Array1::from_shape_fn(n, |i| {
        let g_row = g_mat.map(|m| m.row(i).to_owned());
        let h_row = h_mat.map(|m| m.row(i).to_owned());
        calc_cv_individual(g_row.as_ref(), h_row.as_ref(), config)
    })
}
