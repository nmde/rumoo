use std::cmp::Ordering;

use ndarray::{Array1, Array2, Axis};

use crate::util::dominator::Dominator;

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

fn lex_argsort(f: &Array2<f64>) -> Vec<usize> {
    let n = f.nrows();
    let m = f.ncols();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        for k in 0..m {
            match f[[a, k]].partial_cmp(&f[[b, k]]) {
                Some(Ordering::Equal) | None => continue,
                Some(ord) => return ord,
            }
        }
        Ordering::Equal
    });
    indices
}

/// Mirrors `np.argsort(F[:, :0:-1], axis=1) + 1` from `tree_based_non_dominated_sort`.
fn compute_obj_seq(f: &Array2<f64>) -> Array2<usize> {
    let (n, m) = (f.nrows(), f.ncols());
    let mut obj_seq = Array2::<usize>::zeros((n, m - 1));
    for p in 0..n {
        let mut order: Vec<usize> = (0..m - 1).collect();
        // F[:, :0:-1] = columns [M-1, M-2, ..., 1]; slice index j → F column (M-1-j)
        order.sort_by(|&a, &b| {
            let col_a = m - 1 - a;
            let col_b = m - 1 - b;
            f[[p, col_a]]
                .partial_cmp(&f[[p, col_b]])
                .unwrap_or(Ordering::Equal)
        });
        for (j, &val) in order.iter().enumerate() {
            obj_seq[[p, j]] = val + 1;
        }
    }
    obj_seq
}

// -------------------------------------------------------------------------------------------------
// Tree arena for tree_based_non_dominated_sort
// -------------------------------------------------------------------------------------------------

struct TreeNode {
    key: usize,
    children: Vec<Option<usize>>,
}

struct TreeArena {
    nodes: Vec<TreeNode>,
}

impl TreeArena {
    fn new() -> Self {
        Self { nodes: vec![] }
    }

    fn alloc(&mut self, key: usize, num_branch: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode {
            key,
            children: vec![None; num_branch],
        });
        idx
    }

    fn traversal(&self, root: usize) -> Vec<usize> {
        let mut result = vec![];
        self.traversal_rec(root, &mut result);
        result
    }

    fn traversal_rec(&self, node_idx: usize, result: &mut Vec<usize>) {
        result.push(node_idx);
        for &child_opt in &self.nodes[node_idx].children {
            if let Some(child_idx) = child_opt {
                self.traversal_rec(child_idx, result);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// fast_non_dominated_sort
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.functions.non_dominated_sorting.fast_non_dominated_sort`.
pub fn fast_non_dominated_sort(f: &Array2<f64>, native_biobj_sorting: bool) -> Vec<Vec<usize>> {
    if f.is_empty() {
        return vec![];
    }

    let n_points = f.nrows();
    let n_objectives = f.ncols();

    if n_points <= 1 {
        return if n_points == 1 { vec![vec![0]] } else { vec![] };
    }

    if native_biobj_sorting && n_objectives == 2 {
        return fast_biobjective_nondominated_sort(f);
    }

    let m = Dominator::calc_domination_matrix(f, None, None);
    let n = m.nrows();
    if n == 0 {
        return vec![];
    }

    let mut n_ranked = 0usize;
    let mut is_dominating: Vec<Vec<usize>> = (0..n).map(|_| vec![]).collect();
    let mut n_dominated = vec![0i32; n];
    let mut current_front: Vec<usize> = vec![];

    for i in 0..n {
        for j in (i + 1)..n {
            let rel = m[[i, j]];
            if rel == 1 {
                is_dominating[i].push(j);
                n_dominated[j] += 1;
            } else if rel == -1 {
                is_dominating[j].push(i);
                n_dominated[i] += 1;
            }
        }
        if n_dominated[i] == 0 {
            current_front.push(i);
            n_ranked += 1;
        }
    }

    let mut fronts = vec![current_front.clone()];

    while n_ranked < n {
        let mut next_front: Vec<usize> = vec![];
        for &i in &current_front {
            for &j in &is_dominating[i] {
                n_dominated[j] -= 1;
                if n_dominated[j] == 0 {
                    next_front.push(j);
                    n_ranked += 1;
                }
            }
        }
        fronts.push(next_front.clone());
        current_front = next_front;
    }

    fronts
}

// -------------------------------------------------------------------------------------------------
// _fast_biobjective_nondominated_sort
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.functions.non_dominated_sorting._fast_biobjective_nondominated_sort`.
fn fast_biobjective_nondominated_sort(f: &Array2<f64>) -> Vec<Vec<usize>> {
    let n_points = f.nrows();
    if n_points == 0 {
        return vec![];
    }

    let mut sorted_indices: Vec<usize> = (0..n_points).collect();
    sorted_indices.sort_by(|&a, &b| f[[a, 0]].partial_cmp(&f[[b, 0]]).unwrap_or(Ordering::Equal));

    let mut fronts = vec![];
    let mut assigned = vec![false; n_points];
    let mut n_assigned = 0usize;

    while n_assigned < n_points {
        let mut current_front = vec![];
        let mut current_indices: Vec<usize> = vec![];
        let mut min_second_obj = f64::INFINITY;

        for i in 0..n_points {
            if assigned[i] {
                continue;
            }

            let orig_idx = sorted_indices[i];
            let is_dominated = !current_indices.is_empty() && f[[orig_idx, 1]] >= min_second_obj;

            if !is_dominated {
                current_front.push(orig_idx);
                current_indices.push(i);
                assigned[i] = true;
                n_assigned += 1;
                let v2 = f[[orig_idx, 1]];
                if v2 < min_second_obj {
                    min_second_obj = v2;
                }
            }
        }

        if !current_front.is_empty() {
            fronts.push(current_front);
        } else {
            break;
        }
    }

    fronts
}

// -------------------------------------------------------------------------------------------------
// find_non_dominated
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.functions.non_dominated_sorting.find_non_dominated`.
pub fn find_non_dominated(f: &Array2<f64>, epsilon: f64) -> Vec<usize> {
    let n_points = f.nrows();
    let mut non_dominated = vec![];

    for i in 0..n_points {
        let mut is_dominated = false;

        'outer: for j in 0..n_points {
            if i == j {
                continue;
            }

            let mut dominates = true;
            let mut at_least_one_better = false;

            for k in 0..f.ncols() {
                if f[[j, k]] + epsilon < f[[i, k]] {
                    at_least_one_better = true;
                } else if f[[j, k]] > f[[i, k]] + epsilon {
                    dominates = false;
                    break;
                }
            }

            if dominates && at_least_one_better {
                is_dominated = true;
                break 'outer;
            }
        }

        if !is_dominated {
            non_dominated.push(i);
        }
    }

    non_dominated
}

// -------------------------------------------------------------------------------------------------
// efficient_non_dominated_sort
// -------------------------------------------------------------------------------------------------

pub enum NonDominatedSortStrategy {
    Sequential,
    Binary,
}

/// Mirrors `pymoo.functions.non_dominated_sorting.efficient_non_dominated_sort`.
pub fn efficient_non_dominated_sort(
    f: &Array2<f64>,
    strategy: &NonDominatedSortStrategy,
) -> Vec<Vec<usize>> {
    let n_points = f.nrows();
    let big_i = lex_argsort(f);
    let f_sorted = f.select(Axis(0), &big_i);

    let mut fronts: Vec<Vec<usize>> = vec![];

    for i in 0..n_points {
        let k = if strategy == NonDominatedSortStrategy::Sequential {
            sequential_search(&f_sorted, i, &fronts)
        } else {
            binary_search(&f_sorted, i, &fronts)
        };

        if k >= fronts.len() {
            fronts.push(vec![]);
        }
        fronts[k].push(i);
    }

    fronts
        .iter()
        .map(|front| front.iter().map(|&j| big_i[j]).collect())
        .collect()
}

/// Mirrors `pymoo.functions.non_dominated_sorting.sequential_search`.
pub fn sequential_search(f: &Array2<f64>, i: usize, fronts: &[Vec<usize>]) -> usize {
    let num_found_fronts = fronts.len();
    if num_found_fronts == 0 {
        return 0;
    }

    let current = f.row(i).to_owned();
    let mut k = 0usize;

    loop {
        let fk_indices = &fronts[k];
        let non_dominated = fk_indices.iter().rev().all(|&j| {
            Dominator::get_relation(current.clone(), f.row(j).to_owned(), None, None) != -1
        });

        if non_dominated {
            return k;
        } else {
            k += 1;
            if k >= num_found_fronts {
                return num_found_fronts;
            }
        }
    }
}

/// Mirrors `pymoo.functions.non_dominated_sorting.binary_search`.
pub fn binary_search(f: &Array2<f64>, i: usize, fronts: &[Vec<usize>]) -> usize {
    let num_found_fronts = fronts.len();
    if num_found_fronts == 0 {
        return 0;
    }

    let mut k_min = 0usize;
    let mut k_max = num_found_fronts;
    let mut k = ((k_max + k_min) as f64 / 2.0 + 0.5).floor() as usize;
    let current = f.row(i).to_owned();

    loop {
        let fk_indices = &fronts[k - 1];
        let non_dominated = fk_indices.iter().rev().all(|&j| {
            Dominator::get_relation(current.clone(), f.row(j).to_owned(), None, None) != -1
        });

        if non_dominated {
            if k == k_min + 1 {
                return k - 1;
            } else {
                k_max = k;
                k = ((k_max + k_min) as f64 / 2.0 + 0.5).floor() as usize;
            }
        } else {
            k_min = k;
            if k_max == k_min + 1 && k_max < num_found_fronts {
                return k_max - 1;
            } else if k_min == num_found_fronts {
                return num_found_fronts;
            } else {
                k = ((k_max + k_min) as f64 / 2.0 + 0.5).floor() as usize;
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// tree_based_non_dominated_sort
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.functions.non_dominated_sorting.tree_based_non_dominated_sort`.
pub fn tree_based_non_dominated_sort(f: &Array2<f64>) -> Vec<Vec<usize>> {
    let n_points = f.nrows();

    let indices = lex_argsort(f);
    let f_sorted = f.select(Axis(0), &indices);

    let obj_seq = compute_obj_seq(&f_sorted);

    let mut arena = TreeArena::new();
    let mut forest: Vec<Option<usize>> = vec![];
    let mut left = vec![true; n_points];
    let mut k = 0usize;

    while left.iter().any(|&b| b) {
        forest.push(None);
        for p in 0..n_points {
            if left[p] {
                update_tree(
                    &f_sorted,
                    p,
                    &mut forest,
                    k,
                    &mut left,
                    &obj_seq,
                    &mut arena,
                );
            }
        }
        k += 1;
    }

    let mut fronts: Vec<Vec<usize>> = (0..k).map(|_| vec![]).collect();
    for (ki, root_opt) in forest.iter().enumerate() {
        if let Some(root) = root_opt {
            for node_idx in arena.traversal(*root) {
                fronts[ki].push(indices[arena.nodes[node_idx].key]);
            }
        }
    }
    fronts
}

/// Mirrors `pymoo.functions.non_dominated_sorting.update_tree`.
fn update_tree(
    f: &Array2<f64>,
    p: usize,
    forest: &mut Vec<Option<usize>>,
    k: usize,
    left: &mut Vec<bool>,
    obj_seq: &Array2<usize>,
    arena: &mut TreeArena,
) {
    let big_m = f.ncols();
    if forest[k].is_none() {
        let new_root = arena.alloc(p, big_m - 1);
        forest[k] = Some(new_root);
        left[p] = false;
    } else if check_tree(f, p, forest[k], obj_seq, true, arena) {
        left[p] = false;
    }
}

/// Mirrors `pymoo.functions.non_dominated_sorting.check_tree`.
fn check_tree(
    f: &Array2<f64>,
    p: usize,
    node_opt: Option<usize>,
    obj_seq: &Array2<usize>,
    add_pos: bool,
    arena: &mut TreeArena,
) -> bool {
    let node_idx = match node_opt {
        None => return true,
        Some(idx) => idx,
    };

    let big_m = f.ncols();
    let tree_key = arena.nodes[node_idx].key;

    let mut m = 0usize;
    while m < big_m - 1 {
        let col = obj_seq[[tree_key, m]];
        if f[[p, col]] < f[[tree_key, col]] {
            break;
        }
        m += 1;
    }

    if m == big_m - 1 {
        return false;
    }

    for i in 0..=m {
        let child_idx = arena.nodes[node_idx].children[i];
        if !check_tree(f, p, child_idx, obj_seq, i == m && add_pos, arena) {
            return false;
        }
    }

    if arena.nodes[node_idx].children[m].is_none() && add_pos {
        let new_child = arena.alloc(p, big_m - 1);
        arena.nodes[node_idx].children[m] = Some(new_child);
    }

    true
}

// -------------------------------------------------------------------------------------------------
// construct_comp_matrix / construct_domination_matrix
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.functions.non_dominated_sorting.construct_comp_matrix`.
pub fn construct_comp_matrix(vec: &Array1<f64>, sorted_idx: &[usize]) -> Array2<i32> {
    let n = vec.len();
    let mut c = Array2::<i32>::zeros((n, n));

    let first = sorted_idx[0];
    c.row_mut(first).fill(1);

    for i in 1..n {
        let si = sorted_idx[i];
        let si_prev = sorted_idx[i - 1];
        if vec[si] == vec[si_prev] {
            let prev_row = c.row(si_prev).to_owned();
            c.row_mut(si).assign(&prev_row);
        } else {
            for j in i..n {
                c[[si, sorted_idx[j]]] = 1;
            }
        }
    }

    c
}

/// Mirrors `pymoo.functions.non_dominated_sorting.construct_domination_matrix`.
pub fn construct_domination_matrix(f_scores: &Array2<f64>) -> Array2<i32> {
    let (n, n_obj) = (f_scores.nrows(), f_scores.ncols());
    let mut d = Array2::<i32>::zeros((n, n));

    for j in 0..n_obj {
        let col = f_scores.column(j).to_owned();
        let mut sorted_idx: Vec<usize> = (0..n).collect();
        sorted_idx.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap_or(Ordering::Equal));
        let c = construct_comp_matrix(&col, &sorted_idx);
        d += &c;
    }

    // zero out mutual complete domination (equal points)
    let n_obj_i32 = n_obj as i32;
    let mut mask = vec![false; n * n];
    for i in 0..n {
        for j in 0..n {
            if d[[i, j]] == n_obj_i32 && d[[j, i]] == n_obj_i32 {
                mask[i * n + j] = true;
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            if mask[i * n + j] {
                d[[i, j]] = 0;
            }
        }
    }

    d
}

// -------------------------------------------------------------------------------------------------
// dda_ns / dda_ens / dominance_degree_non_dominated_sort
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.functions.non_dominated_sorting.dda_ns`.
pub fn dda_ns(f_scores: &Array2<f64>) -> Vec<Vec<usize>> {
    let (n, n_obj) = (f_scores.nrows(), f_scores.ncols());
    let n_obj_i32 = n_obj as i32;
    let mut d_mx = construct_domination_matrix(f_scores);

    let mut fronts = vec![];
    let mut count = 0usize;

    while count < n {
        // column-wise max of d_mx
        let max_d: Vec<i32> = (0..n)
            .map(|j| (0..n).map(|i| d_mx[[i, j]]).fold(i32::MIN, i32::max))
            .collect();

        let front: Vec<usize> = (0..n)
            .filter(|&i| max_d[i] >= 0 && max_d[i] < n_obj_i32)
            .collect();

        count += front.len();
        for &i in &front {
            d_mx.row_mut(i).fill(-1);
        }
        for &i in &front {
            d_mx.column_mut(i).fill(-1);
        }
        fronts.push(front);
    }

    fronts
}

/// Mirrors `pymoo.functions.non_dominated_sorting.dda_ens`.
pub fn dda_ens(f_scores: &Array2<f64>) -> Vec<Vec<usize>> {
    let n_obj_i32 = f_scores.ncols() as i32;
    let d_mx = construct_domination_matrix(f_scores);

    let order = lex_argsort(f_scores);
    let mut fronts: Vec<Vec<usize>> = vec![];

    for &s in &order {
        let mut isinserted = false;
        for fk in &mut fronts {
            if !fk.iter().any(|&k| d_mx[[k, s]] == n_obj_i32) {
                fk.push(s);
                isinserted = true;
                break;
            }
        }
        if !isinserted {
            fronts.push(vec![s]);
        }
    }

    fronts
}

pub enum DominatedSortStrategy {
    Efficient,
    Fast,
}

/// Mirrors `pymoo.functions.non_dominated_sorting.dominance_degree_non_dominated_sort`.
pub fn dominance_degree_non_dominated_sort(
    f_scores: &Array2<f64>,
    strategy: &DominatedSortStrategy,
) -> Vec<Vec<usize>> {
    match strategy {
        DominatedSortStrategy::Efficient => dda_ens(f_scores),
        DominatedSortStrategy::Fast => dda_ns(f_scores),
    }
}

// -------------------------------------------------------------------------------------------------
// fast_best_order_sort
// -------------------------------------------------------------------------------------------------

/// Mirrors `pymoo.functions.non_dominated_sorting.fast_best_order_sort`.
pub fn fast_best_order_sort(_f: &Array2<f64>) -> Vec<Vec<usize>> {
    unimplemented!("fast_best_order_sort is only available in compiled (Cython) version")
}
