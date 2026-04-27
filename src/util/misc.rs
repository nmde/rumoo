use std::f64::INFINITY;

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};

use crate::core::population::Population;

/*
from collections import OrderedDict
from itertools import combinations

import numpy as np

from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
from pymoo.util import default_random_state


def parameter_less(F, CV, fmax=None, inplace=False):
    assert len(F) == len(CV)

    if not inplace:
        F = np.copy(F)

    if fmax is None:
        fmax = np.max(F)

    param_less = fmax + CV

    infeas = (CV > 0).flatten()
    F[infeas] = param_less[infeas]

    return F


def swap(M, a, b):
    tmp = M[a]
    M[a] = M[b]
    M[b] = tmp


# repairs a numpy array to be in bounds
def repair(X, xl, xu):
    larger_than_xu = X[0, :] > xu
    X[0, larger_than_xu] = xu[larger_than_xu]

    smaller_than_xl = X[0, :] < xl
    X[0, smaller_than_xl] = xl[smaller_than_xl]

    return X


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def parameter_less_constraints(F, CV, F_max=None):
    if F_max is None:
        F_max = np.max(F)
    has_constraint_violation = CV > 0
    F[has_constraint_violation] = CV[has_constraint_violation] + F_max
    return F


@default_random_state
def random_permutations(n, l, concat=True, random_state=None):
    P = []
    for i in range(n):
        P.append(random_state.permutation(l))
    if concat:
        P = np.concatenate(P)
    return P


def get_duplicates(M):
    res = []
    I = np.lexsort([M[:, i] for i in reversed(range(0, M.shape[1]))])
    S = M[I, :]

    i = 0

    while i < S.shape[0] - 1:
        l = []
        while np.all(S[i, :] == S[i + 1, :]):
            l.append(I[i])
            i += 1
        if len(l) > 0:
            l.append(I[i])
            res.append(l)
        i += 1

    return res
*/

// -----------------------------------------------
// Euclidean Distance
// -----------------------------------------------

fn func_euclidean_distance(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    (a - b).mapv(|v| v * v).sum_axis(Axis(1)).mapv(f64::sqrt)
}

/*
def func_norm_euclidean_distance(xl, xu):
    return lambda a, b: np.sqrt((((a - b) / (xu - xl)) ** 2).sum(axis=1))


def norm_eucl_dist_by_bounds(A, B, xl, xu, **kwargs):
    return vectorized_cdist(A, B, func_dist=func_norm_euclidean_distance(xl, xu), **kwargs)


def norm_eucl_dist(problem, A, B, **kwargs):
    return norm_eucl_dist_by_bounds(A, B, *problem.bounds(), **kwargs)


# -----------------------------------------------
# Manhatten Distance
# -----------------------------------------------

def func_manhatten_distance(a, b):
    return np.abs(a - b).sum(axis=1)


def func_norm_manhatten_distance(xl, xu):
    return lambda a, b: np.abs((a - b) / (xu - xl)).sum(axis=1)


def norm_manhatten_dist_by_bounds(A, B, xl, xu, **kwargs):
    return vectorized_cdist(A, B, func_dist=func_norm_manhatten_distance(xl, xu), **kwargs)


def norm_manhatten_dist(problem, A, B, **kwargs):
    return norm_manhatten_dist_by_bounds(A, B, *problem.bounds(), **kwargs)


# -----------------------------------------------
# Tchebychev Distance
# -----------------------------------------------


def func_tchebychev_distance(a, b):
    return np.abs(a - b).max(axis=1)


def func_norm_tchebychev_distance(xl, xu):
    return lambda a, b: np.abs((a - b) / (xu - xl)).max(axis=1)


def norm_tchebychev_dist_by_bounds(A, B, xl, xu, **kwargs):
    return vectorized_cdist(A, B, func_dist=func_norm_tchebychev_distance(xl, xu), **kwargs)


def norm_tchebychev_dist(problem, A, B, **kwargs):
    return norm_tchebychev_dist_by_bounds(A, B, *problem.bounds(), **kwargs)
*/

// -----------------------------------------------
// Others
// -----------------------------------------------

pub fn cdist(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    vectorized_cdist(a, b, None, Some(false))
}

fn vectorized_cdist(
    a: &Array2<f64>,
    b: &Array2<f64>,
    func_dist: Option<&dyn Fn(&Array2<f64>, &Array2<f64>) -> Result<Array1<f64>>>,
    fill_diag_with_inf: Option<bool>,
) -> Result<Array2<f64>> {
    let na = a.nrows();
    let nb = b.nrows();

    let mut u = Array2::zeros((na * nb, a.ncols()));
    for (i, row) in a.outer_iter().enumerate() {
        for j in 0..b.nrows() {
            u.row_mut(i * nb + j).assign(&row);
        }
    }

    let mut v = Array2::zeros((na * nb, a.ncols()));
    for i in 0..na {
        for (j, row) in b.outer_iter().enumerate() {
            v.row_mut(i * nb + j).assign(&row);
        }
    }

    let tmp = match func_dist {
        None => func_euclidean_distance(&u, &v),
        Some(f) => f(&u, &v)?,
    };
    let mut m = tmp.to_shape((na, nb))?;

    if fill_diag_with_inf.unwrap_or(false) {
        for i in 0..na.min(nb) {
            m[[i, i]] = INFINITY;
        }
    }

    Ok(m.to_owned())
}

/*
def covert_to_type(problem, X):
    if problem.vtype == float:
        return X.astype(np.double)
    elif problem.vtype == int:
        return np.round(X).astype(int)
    elif problem.vtype == bool:
        return X < (problem.xu - problem.xl) / 2


def find_duplicates(X, epsilon=1e-16):
    # calculate the distance matrix from each point to another
    D = cdist(X, X)

    # set the diagonal to infinity
    D[np.triu_indices(len(X))] = np.inf

    # set as duplicate if a point is really close to this one
    is_duplicate = np.any(D <= epsilon, axis=1)

    return is_duplicate


def at_least_2d(*args, **kwargs):
    ret = tuple([at_least_2d_array(arg, **kwargs) for arg in args])
    if len(ret) == 1:
        ret = ret[0]
    return ret
*/

#[derive(PartialEq)]
enum ExtendAs {
    Row,
    Column,
}

fn at_least_2d_array(x: Array1<f64>, extend_as: Option<ExtendAs>) -> Result<Array2<f64>> {
    let extend_as = extend_as.unwrap_or(ExtendAs::Row);

    Ok((if extend_as == ExtendAs::Row {
        x.to_shape((1, x.len()))?
    } else {
        x.to_shape((x.len(), 1))?
    })
    .to_owned())
}

/*
def to_1d_array_if_possible(x):
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if x.ndim == 2:
        if x.shape[0] == 1 or x.shape[1] == 1:
            x = x.flatten()

    return x


def stack(*args, flatten=True):
    if not flatten:
        ps = np.concatenate([e[None, ...] for e in args])
    else:
        ps = np.vstack(args)
    return ps


def all_except(x, *args):
    if len(args) == 0:
        return x
    else:
        H = set(args) if len(args) > 5 else args
        I = [k for k in range(len(x)) if k not in H]
        return x[I]


def all_combinations(A, B):
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, A.shape[0])
    return np.column_stack([u, v])


def pop_from_sampling(problem, sampling, n_initial_samples, pop=None):
    # the population type can be different - (different type of individuals)
    if pop is None:
        pop = Population()

    # provide a whole population object - (individuals might be already evaluated)
    if isinstance(sampling, Population):
        pop = sampling

    else:
        # if just an X array create a pop
        if isinstance(sampling, np.ndarray):
            pop = pop.new("X", sampling)

        elif isinstance(sampling, Sampling):
            # use the sampling
            pop = sampling.do(problem, n_initial_samples, pop=pop)

        else:
            return None

    return pop


def evaluate_if_not_done_yet(evaluator, problem, pop, algorithm=None):
    I = np.where(pop.get("F") == None)[0]
    if len(I) > 0:
        pop[I] = evaluator.process(problem, pop[I], algorithm=algorithm)


def set_if_none(kwargs, str, val):
    if str not in kwargs:
        kwargs[str] = val


def set_if_none_from_tuples(kwargs, *args):
    for key, val in args:
        if key not in kwargs:
            kwargs[key] = val



def distance_of_closest_points_to_others(X):
    D = vectorized_cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return D.argmin(axis=1), D.min(axis=1)


def time_to_int(t):
    vals = [int(e) for e in t.split(":")][::-1]
    s = vals[0]
    if len(vals) > 1:
        s += 60 * vals[1]
    if len(vals) > 2:
        s += 3600 * vals[2]
    return s


def powerset(iterable):
    for n in range(len(iterable) + 1):
        yield from combinations(iterable, n)


def intersect(a, b):
    H = set()
    for entry in b:
        H.add(entry)

    ret = []
    for entry in a:
        if entry in H:
            ret.append(entry)

    return ret
*/

pub fn has_feasible(pop: &Population) -> bool {
    pop.feas().iter().any(|f| *f)
}

/*
def to_numpy(a):
    return np.array(a)


def termination_from_tuple(termination):
    from pymoo.core.termination import Termination

    # get the termination if provided as a tuple - create an object
    if termination is not None and not isinstance(termination, Termination):
        from pymoo.termination import get_termination
        if isinstance(termination, str):
            termination = get_termination(termination)
        else:
            termination = get_termination(*termination)

    return termination


def unique_and_all_indices(arr):
    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, inverse, counts = np.unique(arr,
                                                     return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes


def from_dict(D, *keys):
    return [D.get(k) for k in keys]


def list_of_dicts_unique(l, k):
    return list(OrderedDict([(e[k], None) for e in l]).keys())


def list_of_dicts_filter(l, *pairs):
    return [e for e in l if all(e[k] == v for (k, v) in pairs)]


def logical_op(func, a, b, *args):
    ret = func(a, b)
    for c in args:
        ret = func(ret, c)
    return ret


def replace_nan_by(x, val, inplace=False):
    is_nan = np.isnan(x)
    if np.sum(is_nan) > 0:
        if not inplace:
            x = x.copy()
        x[is_nan] = val
    return x


def set_defaults(kwargs, defaults, overwrite=False, func_get=lambda x: x):
    for k, v in defaults.items():
        if overwrite or k not in kwargs:
            kwargs[k] = func_get(v)


def filter_params(params, prefix, delete_prefix=True):
    ret = {}
    for k, v in params.items():
        if k.startswith(prefix):
            if delete_prefix:
                k = k[len(prefix):]
            ret[k] = v
    return ret


def where_is_what(x):
    H = {}
    for k, e in enumerate(x):
        if e not in H:
            H[e] = []
        H[e].append(k)
    return H


def crossover_mask(X, M):
    # convert input to output by flatting along the first axis
    _X = np.copy(X)
    _X[0][M] = X[1][M]
    _X[1][M] = X[0][M]
    return _X


@default_random_state
def row_at_least_once_true(M, random_state=None):
    _, d = M.shape
    for k in np.where(~np.any(M, axis=1))[0]:
        M[k, random_state.integers(d)] = True
    return M
    */
