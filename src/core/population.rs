use std::{iter::Enumerate, slice::Iter};

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};

use crate::core::individual::Individual;

enum IndividualOrMore {
    Single(Individual),
    Multiple(Vec<Individual>),
}

#[derive(Clone)]
pub struct Population {
    individuals: Vec<Individual>,
}

impl Population {
    pub fn new(individuals: Option<IndividualOrMore>) -> Self {
        Self {
            individuals: if individuals.is_some() {
                match individuals.unwrap() {
                    IndividualOrMore::Single(i) => vec![i],
                    IndividualOrMore::Multiple(i) => i,
                }
            } else {
                vec![]
            },
        }
    }
    /*
            def has(self, key):
                return all([ind.has(key) for ind in self])

            def collect(self, func, to_numpy=True):
                val = []
                for i in range(len(self)):
                    val.append(func(self[i]))
                if to_numpy:
                    val = np.array(val)
                return val

            def apply(self, func):
                self.collect(func, to_numpy=False)

            pub fn get(&self, *args, to_numpy=True, **kwargs):

                val = {}
                for c in args:
                    val[c] = []

                # for each individual
                for i in range(len(self)):

                    # for each argument
                    for c in args:
                        val[c].append(self[i].get(c, **kwargs))

                # convert the results to a list
                res = [val[c] for c in args]

                # to numpy array if desired - default true
                if to_numpy:
                    res = [np.array(e) for e in res]

                # return as tuple or single value
                if len(args) == 1:
                    return res[0]
                else:
                    return tuple(res)
    */

    pub fn x(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.individuals.len(), self.individuals[0].x.iter().len()));
        for (i, ind) in self.individuals.iter().enumerate() {
            out.row_mut(i).assign(&ind.x);
        }
        out
    }

    pub fn set_x(&mut self, value: Array1<f64>) {
        for ind in &mut self.individuals {
            ind.x = value.clone();
        }
    }

    pub fn set_each_x(&mut self, values: &Vec<Array1<f64>>) -> Result<&Self> {
        if values.len() != self.len() {
            return Err(anyhow!(
                "Population Set Attribute Error: Number of values and population size do not match!"
            ));
        }

        for (ind, val) in self.individuals.iter_mut().zip(values) {
            ind.x = val.clone();
        }

        Ok(self)
    }

    pub fn feas(&self) -> Array1<bool> {
        Array1::from_vec(self.individuals.iter().map(|ind| ind.feas()[0]).collect())
    }

    /*
        @classmethod
        def merge(cls, a, b, *args):

            # do the regular merge between first and second element
            m = merge(a, b)

            # process the list of others and merge as well
            others = list(args)
            while len(others) > 0:
                m = merge(m, others.pop(0))

            return m

        @classmethod
        def create(cls, *args):
            return Population.__new__(cls, args)

        @classmethod
        def empty(cls, size=0):
            individuals = [Individual() for _ in range(size)]
            return Population.__new__(cls, individuals)

        @classmethod
        def new(cls, *args, **kwargs):
            kwargs = interleaving_args(*args, kwargs=kwargs)

            if len(kwargs) > 0:
                sizes = np.unique(np.array([len(v) for _, v in kwargs.items()]))
                if len(sizes) == 1:
                    size = sizes[0]
                else:
                    raise Exception(f"Population.new needs to be called with same-sized inputs, but the sizes are {sizes}")
            else:
                size = 0

            pop = Population.empty(size)
            pop.set(**kwargs)

            return pop
    */

    // Mirrors for array functionality
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    pub fn select(&self, indices: &[usize]) -> Self {
        Self::new(Some(IndividualOrMore::Multiple(
            indices
                .iter()
                .map(|i| self.individuals[*i].clone())
                .collect(),
        )))
    }

    pub fn enumerate(&self) -> Enumerate<Iter<'_, Individual>> {
        self.individuals.iter().enumerate()
    }
}

/*
def pop_from_array_or_individual(array, pop=None):
    # the population type can be different - (different type of individuals)
    if pop is None:
        pop = Population.empty()

    # provide a whole population object - (individuals might be already evaluated)
    if isinstance(array, Population):
        pop = array
    elif isinstance(array, np.ndarray):
        pop = pop.new("X", np.atleast_2d(array))
    elif isinstance(array, Individual):
        pop = Population.empty(1)
        pop[0] = array
    else:
        return None

    return pop


def merge(a, b):
    if a is None:
        return b
    elif b is None:
        return a

    a, b = pop_from_array_or_individual(a), pop_from_array_or_individual(b)

    if len(a) == 0:
        return b
    elif len(b) == 0:
        return a
    else:
        obj = np.concatenate([a, b]).view(Population)
        return obj


def interleaving_args(*args, kwargs=None):
    if len(args) % 2 != 0:
        raise Exception(f"Even number of arguments are required but {len(args)} arguments were provided.")

    if kwargs is None:
        kwargs = {}

    for i in range(int(len(args) / 2)):
        key, values = args[i * 2], args[i * 2 + 1]
        kwargs[key] = values
    return kwargs


def calc_cv(pop, config=None):

    if config is None:
        config = Individual.default_config()

    G, H = pop.get("G", "H")

    from pymoo.core.individual import calc_cv as func
    CV = np.array([func(g, h, config) for g, h in zip(G, H)])

    return CV
*/
