use ndarray::{Array1, Array2};

use crate::core::individual::Individual;

/// Compare two individuals by dominance, accounting for constraint violation.
///
/// Mirrors the module-level `get_relation(ind_a, ind_b)`.
pub fn get_relation(ind_a: &Individual, ind_b: &Individual) -> i32 {
    let (Some(ref a), Some(ref b)) = (&ind_a.f, &ind_b.f) else {
        return 0;
    };
    let cva = ind_a.cv().first().copied();
    let cvb = ind_b.cv().first().copied();
    Dominator::get_relation(*a, *b, cva, cvb)
}

/// Pareto-dominance utilities.
///
/// Mirrors `pymoo.util.dominator.Dominator` (static-method-only class).
pub struct Dominator;

impl Dominator {
    /// Determine the dominance relation between two objective vectors `a` and `b`.
    ///
    /// Returns:
    ///  `1`  — `a` dominates `b`
    /// `-1`  — `b` dominates `a`
    ///  `0`  — neither dominates (indifferent)
    ///
    /// When constraint violations `cva`/`cvb` are provided, a strictly lower CV
    /// takes priority over the objective comparison.
    ///
    /// Mirrors `Dominator.get_relation(a, b, cva=None, cvb=None)`.
    pub fn get_relation(a: Array1<f64>, b: Array1<f64>, cva: Option<f64>, cvb: Option<f64>) -> i32 {
        if let (Some(cva), Some(cvb)) = (cva, cvb) {
            if cva < cvb {
                return 1;
            } else if cvb < cva {
                return -1;
            }
        }

        let mut val = 0i32;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            if ai < bi {
                // indifferent because once better and once worse
                if val == -1 {
                    return 0;
                }
                val = 1;
            } else if bi < ai {
                // indifferent because once better and once worse
                if val == 1 {
                    return 0;
                }
                val = -1;
            }
        }
        val
    }

    /// Build the full n×n domination matrix using an explicit double loop.
    ///
    /// `G` is the constraint matrix (n × n_constr); CV is computed as the
    /// row-sum of positive constraint violations: `CV[i] = Σ max(0, G[i,j])`.
    ///
    /// `M[i, j] =  1` means row i dominates row j.
    /// `M[i, j] = -1` means row j dominates row i.
    /// `M[i, j] =  0` means neither dominates.
    ///
    /// Mirrors `Dominator.calc_domination_matrix_loop(F, G)`.
    pub fn calc_domination_matrix_loop(f: &Array2<f64>, g: &Array2<f64>) -> Array2<i32> {
        let n = f.nrows();

        // CV = np.sum(G * (G > 0).astype(float), axis=1)
        let cv: Array1<f64> = g
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v.max(0.0)).sum())
            .collect();

        let mut m = Array2::<i32>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let rel = Dominator::get_relation(
                    f.row(i).into_owned(),
                    f.row(j).into_owned(),
                    Some(cv[i]),
                    Some(cv[j]),
                );
                m[[i, j]] = rel;
                m[[j, i]] = -rel;
            }
        }
        m
    }

    /// Build the n×m domination matrix using a vectorised broadcast strategy.
    ///
    /// Each cell `M[i, j]` gives the dominance relation of `F[i]` over `_F[j]`.
    /// When `_F` is `None`, it defaults to `F` (producing the square self-domination
    /// matrix).  `epsilon` adds a tolerance to the `<` / `>` comparisons.
    ///
    /// Mirrors `Dominator.calc_domination_matrix(F, _F=None, epsilon=0.0)`.
    pub fn calc_domination_matrix(
        f: &Array2<f64>,
        _f: Option<&Array2<f64>>,
        epsilon: Option<f64>,
    ) -> Array2<i32> {
        let epsilon = epsilon.unwrap_or(0.0);
        let _f = _f.unwrap_or(f);

        let n = f.nrows();
        let m = _f.nrows();
        let n_obj = f.ncols();

        // L = np.repeat(F, m, axis=0)   → shape (n*m, n_obj)
        // L[i*m + j, :] = F[i, :] for all j in 0..m
        let mut l = Array2::<f64>::zeros((n * m, n_obj));
        for i in 0..n {
            for j in 0..m {
                l.row_mut(i * m + j).assign(&f.row(i));
            }
        }

        // R = np.tile(_F, (n, 1))       → shape (n*m, n_obj)
        // R[i*m + j, :] = _F[j, :] for all i in 0..n
        let mut r = Array2::<f64>::zeros((n * m, n_obj));
        for i in 0..n {
            for j in 0..m {
                r.row_mut(i * m + j).assign(&_f.row(j));
            }
        }

        // smaller[k] = any(L[k] + epsilon < R[k])
        // larger[k]  = any(L[k] > R[k] + epsilon)
        let mut smaller_flat = Array1::<bool>::from_elem(n * m, false);
        let mut larger_flat = Array1::<bool>::from_elem(n * m, false);

        for k in 0..(n * m) {
            smaller_flat[k] = l
                .row(k)
                .iter()
                .zip(r.row(k).iter())
                .any(|(&li, &ri)| li + epsilon < ri);
            larger_flat[k] = l
                .row(k)
                .iter()
                .zip(r.row(k).iter())
                .any(|(&li, &ri)| li > ri + epsilon);
        }

        // reshape flat bool arrays to (n, m)
        let smaller = smaller_flat.into_shape((n, m))?;
        let larger = larger_flat.into_shape((n, m))?;

        // M = (smaller & !larger) * 1 + (larger & !smaller) * -1
        Array2::from_shape_fn((n, m), |(i, j)| {
            if smaller[[i, j]] && !larger[[i, j]] {
                1i32
            } else if larger[[i, j]] && !smaller[[i, j]] {
                -1i32
            } else {
                0i32
            }
        })
    }
}
