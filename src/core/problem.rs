use std::{any::type_name, cell::RefCell, cmp::Ordering, collections::HashMap, fmt};

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, ArrayD, Axis, concatenate};

use crate::core::individual::IndividualField;

/// Mirrors `LoopedElementwiseEvaluation`.
pub struct LoopedElementwiseEvaluation;

pub trait ElementwiseFn {
    fn call(&self, x: Array1<f64>) -> HashMap<String, Array1<f64>>;
}

pub trait ElementwiseRunner {
    fn call(&self, f: &dyn ElementwiseFn, x: &Array2<f64>) -> Vec<HashMap<String, Array1<f64>>>;
}

impl ElementwiseRunner for LoopedElementwiseEvaluation {
    fn call(&self, f: &dyn ElementwiseFn, x: &Array2<f64>) -> Vec<HashMap<String, Array1<f64>>> {
        x.rows()
            .into_iter()
            .map(|row| f.call(row.to_owned()))
            .collect()
    }
}

/// Mirrors `ElementwiseEvaluationFunction`.
pub struct ElementwiseEvaluationFunction<'a> {
    pub problem: &'a (dyn Problem + 'a),
}

impl<'a> ElementwiseFn for ElementwiseEvaluationFunction<'a> {
    fn call(&self, x: Array1<f64>) -> HashMap<String, Array1<f64>> {
        let mut out = HashMap::<String, Array1<f64>>::new();
        self.problem._evaluate_single(x, &mut out);
        out
    }
}

/// Return type of `Problem::evaluate`.
pub enum EvalResult {
    Single(ArrayD<f64>),
    Multiple(Vec<ArrayD<f64>>),
    Dict(HashMap<String, ArrayD<f64>>),
}

/// Mirrors Python's `xl`/`xu` accepting either a scalar or an array.
pub enum BoundsSpec {
    Scalar(f64),
    Array(Array1<f64>),
}

impl BoundsSpec {
    pub fn to_array(self, n: usize) -> Array1<f64> {
        match self {
            BoundsSpec::Scalar(v) => Array1::from_elem(n, v),
            BoundsSpec::Array(a) => a,
        }
    }
}

/// Variable definition for structured (`vars`) problems.
/// Mirrors `.lb` / `.ub` attributes accessed via `hasattr(var, "lb")`.
pub struct VarDef {
    pub lb: Option<f64>,
    pub ub: Option<f64>,
}

pub struct ProblemBase {
    pub n_var: i64,
    pub n_obj: usize,
    pub n_ieq_constr: usize,
    pub n_eq_constr: usize,

    pub xl: Option<Array1<f64>>,
    pub xu: Option<Array1<f64>>,

    pub vtype: Option<String>,
    pub vars: Option<HashMap<String, VarDef>>,

    pub elementwise: bool,
    pub elementwise_runner: Box<dyn ElementwiseRunner>,

    pub requires_kwargs: bool,
    pub strict: bool,
    pub replace_nan_values_by: Option<f64>,
    pub exclude_from_serialization: Option<Vec<String>>,
    pub callback: Option<Box<dyn Fn(&Array2<f64>, &HashMap<String, ArrayD<f64>>)>>,

    pub data: HashMap<String, String>,

    // @Cache fields — outer None == not yet computed
    nadir_point_cache: RefCell<Option<Option<Array1<f64>>>>,
    ideal_point_cache: RefCell<Option<Option<Array1<f64>>>>,
    pareto_front_cache: RefCell<Option<Option<Array2<f64>>>>,
    pareto_set_cache: RefCell<Option<Option<Array2<f64>>>>,
}

impl ProblemBase {
    pub fn new(
        n_var: Option<i64>,
        n_obj: Option<usize>,
        n_ieq_constr: Option<usize>,
        n_eq_constr: Option<usize>,
        xl: Option<BoundsSpec>,
        xu: Option<BoundsSpec>,
        vtype: Option<String>,
        vars: Option<HashMap<String, VarDef>>,
        elementwise: bool,
        elementwise_runner: Box<dyn ElementwiseRunner>,
        requires_kwargs: bool,
        strict: bool,
        replace_nan_values_by: Option<f64>,
        exclude_from_serialization: Option<Vec<String>>,
        callback: Option<Box<dyn Fn(&Array2<f64>, &HashMap<String, ArrayD<f64>>)>>,
        data: HashMap<String, String>,
        // mirrors: max(n_ieq_constr, kwargs["n_constr"]) if "n_constr" in kwargs
        n_constr_compat: Option<usize>,
    ) -> Self {
        let n_var = n_var.unwrap_or(-1);
        let n_obj = n_obj.unwrap_or(1);
        let n_ieq_constr = n_ieq_constr.unwrap_or(0);
        let n_eq_constr = n_eq_constr.unwrap_or(0);

        let n_ieq_constr = match n_constr_compat {
            Some(nc) => n_ieq_constr.max(nc),
            None => n_ieq_constr,
        };

        let (n_var, xl, xu, vars) = if let Some(ref v) = vars {
            let nv = v.len() as i64;
            let xl_out = xl.map(|b| b.to_array(nv as usize)).or_else(|| {
                let lbs: Vec<f64> = v
                    .values()
                    .map(|d| d.lb.unwrap_or(f64::NEG_INFINITY))
                    .collect();
                Some(Array1::from_vec(lbs))
            });
            let xu_out = xu.map(|b| b.to_array(nv as usize)).or_else(|| {
                let ubs: Vec<f64> = v.values().map(|d| d.ub.unwrap_or(f64::INFINITY)).collect();
                Some(Array1::from_vec(ubs))
            });
            (nv, xl_out, xu_out, vars)
        } else {
            let xl_out = if n_var > 0 {
                xl.map(|b| b.to_array(n_var as usize))
            } else {
                None
            };
            let xu_out = if n_var > 0 {
                xu.map(|b| b.to_array(n_var as usize))
            } else {
                None
            };
            (n_var, xl_out, xu_out, None)
        };

        Self {
            n_var,
            n_obj,
            n_ieq_constr,
            n_eq_constr,
            xl,
            xu,
            vtype,
            vars,
            elementwise,
            elementwise_runner,
            requires_kwargs,
            strict,
            replace_nan_values_by,
            exclude_from_serialization,
            callback,
            data,
            nadir_point_cache: RefCell::new(None),
            ideal_point_cache: RefCell::new(None),
            pareto_front_cache: RefCell::new(None),
            pareto_set_cache: RefCell::new(None),
        }
    }
}

pub trait Problem {
    fn base(&self) -> &ProblemBase;

    fn evaluate(
        &self,
        x: Array2<f64>,
        return_values_of: Option<Vec<IndividualField>>,
        return_as_dictionary: Option<bool>,
    ) -> Result<EvalResult> {
        let return_as_dictionary = return_as_dictionary.unwrap_or(false);
        let base = self.base();

        let return_values_of = return_values_of.unwrap_or_else(|| {
            let mut v = vec![IndividualField::F];
            if base.n_ieq_constr > 0 {
                v.push(IndividualField::G);
            }
            if base.n_eq_constr > 0 {
                v.push(IndividualField::H);
            }
            v
        });

        if base.n_var > 0 && x.ncols() != base.n_nvar as usize {
            return Err(anyhow!(
                "Input dimension {0} is not equal to n_var {1}!",
                x.ncols(),
                base.n_var
            ));
        }

        let _out = self.do_eval(x.clone(), &return_values_of)?;

        let mut out = HashMap::<String, ArrayD<f64>>::new();
        for (k, mut v) in _out {
            if let Some(replace) = base.replace_nan_values_by {
                v.mapv_inplace(|val| if val.is_nan() { replace } else { val });
            }
            out.insert(k, v);
        }

        if let Some(ref cb) = base.callback {
            cb(&x, &out);
        }

        if return_as_dictionary {
            return Ok(EvalResult::Dict(out));
        }

        if return_values_of.len() == 1 {
            let key = &return_values_of[0];
            let val = out.remove(key).unwrap_or_else(|| {
                Array2::from_elem((x.nrows(), base.n_obj), f64::INFINITY).into_dyn()
            });
            Ok(EvalResult::Single(val))
        } else {
            let vals: Vec<ArrayD<f64>> = return_values_of
                .iter()
                .map(|k| {
                    out.remove(k).unwrap_or_else(|| {
                        Array2::from_elem((x.nrows(), 1), f64::INFINITY).into_dyn()
                    })
                })
                .collect();
            Ok(EvalResult::Multiple(vals))
        }
    }

    fn do_eval(
        &self,
        x: Array2<f64>,
        return_values_of: &[IndividualField],
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let base = self.base();
        let mut out: HashMap<String, Option<ArrayD<f64>>> =
            return_values_of.iter().map(|k| (k.clone(), None)).collect();

        if base.elementwise {
            self._evaluate_elementwise_impl(x.clone(), &mut out);
        } else {
            let mut out_tmp = HashMap::<String, ArrayD<f64>>::new();
            self._evaluate_vectorized(x.clone(), &mut out_tmp);
            for (k, v) in out_tmp {
                out.insert(k, Some(v));
            }
        }

        let n = x.nrows();
        self._format_dict(out, n, return_values_of)
    }

    fn _evaluate_vectorized(&self, x: Array2<f64>, out: &mut HashMap<String, ArrayD<f64>>) {
        self._evaluate(x, out);
    }

    fn _evaluate_elementwise(
        &self,
        x: Array2<f64>,
        out: &mut HashMap<String, Option<ArrayD<f64>>>,
    ) -> Result<()> {
        // Mirrors LoopedElementwiseEvaluation: call _evaluate_single for each row.
        // &Self cannot be coerced to &dyn Problem in a struct literal, so we inline
        // the runner loop here rather than going through ElementwiseEvaluationFunction.
        for row in x.rows() {
            let mut elem = HashMap::<String, Array1<f64>>::new();
            self._evaluate_single(row.to_owned(), &mut elem);
            for (k, v) in elem {
                let row_dyn = v.insert_axis(Axis(0)).into_dyn().to_owned();
                let entry = out.entry(k).or_insert(None);
                match entry {
                    None => *entry = Some(row_dyn),
                    Some(acc) => {
                        *acc = concatenate(Axis(0), &[acc.view(), row_dyn.view()])?;
                    }
                }
            }
        }
        Ok(())
    }

    fn _format_dict(
        &self,
        out: HashMap<String, Option<ArrayD<f64>>>,
        n: usize,
        return_values_of: &[IndividualField],
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let base = self.base();
        let shapes = default_shape(base, n);
        let mut ret = HashMap::<String, ArrayD<f64>>::new();

        for (name, v_opt) in &out {
            if let Some(v) = v_opt {
                if let Some(shape) = shapes.get(name.as_str()) {
                    let reshaped = v.clone().into_shape(shape.as_slice()).map_err(|_| {
                        anyhow!(
                            "Problem Error: {} can not be set, expected shape {:?} but provided {:?}",
                            name,
                            shape,
                            v.shape()
                        )
                    })?;
                    ret.insert(name.clone(), reshaped);
                } else {
                    ret.insert(name.clone(), v.clone());
                }
            }
        }

        for name in return_values_of {
            if !ret.contains_key(name) {
                let shape = shapes
                    .get(name.as_str())
                    .cloned()
                    .unwrap_or_else(|| vec![n]);
                ret.insert(name.to_string().clone(), ArrayD::from_elem(shape, f64::INFINITY));
            }
        }

        Ok(ret)
    }

    fn nadir_point(&self) -> Option<Array1<f64>> {
        let cache = &self.base().nadir_point_cache;
        if cache.borrow().is_none() {
            let result = self
                .pareto_front()
                .map(|pf| pf.fold_axis(Axis(0), f64::NEG_INFINITY, |&acc, &x| acc.max(x)));
            *cache.borrow_mut() = Some(result);
        }
        cache.borrow().as_ref().unwrap().clone()
    }

    fn ideal_point(&self) -> Option<Array1<f64>> {
        let cache = &self.base().ideal_point_cache;
        if cache.borrow().is_none() {
            let result = self
                .pareto_front()
                .map(|pf| pf.fold_axis(Axis(0), f64::INFINITY, |&acc, &x| acc.min(x)));
            *cache.borrow_mut() = Some(result);
        }
        cache.borrow().as_ref().unwrap().clone()
    }

    fn pareto_front(&self) -> Option<Array2<f64>> {
        let cache = &self.base().pareto_front_cache;
        if cache.borrow().is_none() {
            let mut pf = self._calc_pareto_front();
            // at_least_2d_array(pf, extend_as='r') — already Array2, no-op
            // if 2-objective front, sort by first column
            if let Some(ref p) = pf {
                if p.ncols() == 2 {
                    let mut rows: Vec<Array1<f64>> =
                        p.rows().into_iter().map(|r| r.to_owned()).collect();
                    rows.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(Ordering::Equal));
                    let n = rows.len();
                    if n > 0 {
                        let flat: Vec<f64> = rows.into_iter().flat_map(|r| r.to_vec()).collect();
                        pf = Array2::from_shape_vec((n, 2), flat).ok();
                    }
                }
            }
            *cache.borrow_mut() = Some(pf);
        }
        cache.borrow().as_ref().unwrap().clone()
    }

    fn pareto_set(&self) -> Option<Array2<f64>> {
        let cache = &self.base().pareto_set_cache;
        if cache.borrow().is_none() {
            let ps = self._calc_pareto_set();
            // at_least_2d_array(ps, extend_as='r') — already Array2
            *cache.borrow_mut() = Some(ps);
        }
        cache.borrow().as_ref().unwrap().clone()
    }

    fn n_constr(&self) -> usize {
        self.base().n_ieq_constr + self.base().n_eq_constr
    }

    /// Vectorized evaluation — x shape: (N, n_var).
    /// Mirrors the abstract `Problem._evaluate`.
    fn _evaluate(&self, x: Array2<f64>, out: &mut HashMap<String, ArrayD<f64>>);

    fn has_bounds(&self) -> bool {
        self.base().xl.is_some() && self.base().xu.is_some()
    }

    fn has_constraints(&self) -> bool {
        self.n_constr() > 0
    }

    fn bounds(&self) -> (Option<&Array1<f64>>, Option<&Array1<f64>>) {
        (self.base().xl.as_ref(), self.base().xu.as_ref())
    }

    fn name(&self) -> &'static str {
        type_name::<Self>()
    }

    fn _calc_pareto_front(&self) -> Option<Array2<f64>> {
        None
    }

    fn _calc_pareto_set(&self) -> Option<Array2<f64>> {
        None
    }
}

impl fmt::Display for dyn Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let base = self.base();
        write!(
            f,
            "# name: {}\n# n_var: {}\n# n_obj: {}\n# n_ieq_constr: {}\n# n_eq_constr: {}\n",
            self.name(),
            base.n_var,
            base.n_obj,
            base.n_ieq_constr,
            base.n_eq_constr,
        )
    }
}

/// Mirrors `ElementwiseProblem` — sets `elementwise = true` by default.
pub struct ElementwiseProblem {
    pub base: ProblemBase,
}

impl ElementwiseProblem {
    pub fn new(mut base: ProblemBase) -> Self {
        base.elementwise = true;
        Self { base }
    }
}

/// Mirrors `default_shape(problem, n)`.
/// Returns expected output shapes keyed by output name.
/// dF / dG / dH are 3-D: (n, n_obj/constr, n_var).
pub fn default_shape(base: &ProblemBase, n: usize) -> HashMap<&'static str, Vec<usize>> {
    let n_var = base.n_var.max(0) as usize;
    let mut s = HashMap::new();
    s.insert("F", vec![n, base.n_obj]);
    s.insert("G", vec![n, base.n_ieq_constr]);
    s.insert("H", vec![n, base.n_eq_constr]);
    s.insert("dF", vec![n, base.n_obj, n_var]);
    s.insert("dG", vec![n, base.n_ieq_constr, n_var]);
    s.insert("dH", vec![n, base.n_eq_constr, n_var]);
    s
}
