use std::cmp::Ordering;

use ndarray::{Array1, s};
use rand::{rngs::StdRng, seq::SliceRandom};

use crate::util::default_random_state;

enum RandomSortingMethod {
    Numpy,
    Quicksort,
}

enum SortingOrder {
    Ascending,
    Descending,
}

/// Mirrors `pymoo.util.randomized_argsort.randomized_argsort`.
///
/// Returns indices that sort `a` (ascending by default) with randomized tie-breaking.
/// `method`: `"numpy"` (default) or `"quicksort"`.
/// `order`:  `"ascending"` (default) or `"descending"`.
pub fn randomized_argsort(
    a: &Array1<f64>,
    method: Option<&RandomSortingMethod>,
    order: Option<&SortingOrder>,
    random_state: Option<&mut StdRng>,
) -> Array1<usize> {
    let method = method.unwrap_or(&RandomSortingMethod::Numpy);
    let order = order.unwrap_or(&SortingOrder::Ascending);

    let mut fallback = default_random_state();
    let rng = random_state.unwrap_or(&mut fallback);

    let result = match method {
        RandomSortingMethod::Numpy => {
            let n = a.len();

            // P = random_state.permutation(len(A))
            let mut p: Vec<usize> = (0..n).collect();
            p.shuffle(rng);

            // I = np.argsort(A[P], kind='quicksort')
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&x, &y| a[p[x]].partial_cmp(&a[p[y]]).unwrap_or(Ordering::Equal));

            // I = P[I]
            let mapped: Vec<usize> = idx.iter().map(|&i| p[i]).collect();
            Array1::from(mapped)
        }
        RandomSortingMethod::Quicksort => quicksort(a, Some(rng)),
    };

    match order {
        SortingOrder::Ascending => result,
        SortingOrder::Descending => result.slice(s![..;-1]).to_owned(),
    }
}

/// Mirrors `pymoo.util.randomized_argsort.quicksort`.
pub fn quicksort(a: &Array1<f64>, random_state: Option<&mut StdRng>) -> Array1<usize> {
    let n = a.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let mut fallback = default_random_state();
    let rng = random_state.unwrap_or(&mut fallback);
    if n > 0 {
        _quicksort(a, &mut idx, 0, n - 1, rng);
    }
    Array1::from(idx)
}

/// Mirrors `pymoo.util.randomized_argsort._quicksort`.
fn _quicksort(a: &Array1<f64>, idx: &mut Vec<usize>, left: usize, right: usize, rng: &mut StdRng) {
    if left < right {
        // index = random_state.integers(left, right + 1)
        let index = rng.gen_range(left..=right);

        // swap(I, right, index)
        idx.swap(right, index);

        let pivot = a[idx[right]];

        // i = left - 1  (use isize to represent -1 initial value)
        let mut i: isize = left as isize - 1;

        for j in left..right {
            if a[idx[j]] <= pivot {
                i += 1;
                // swap(I, i, j)
                idx.swap(i as usize, j);
            }
        }

        let partition_idx = (i + 1) as usize;
        // swap(I, right, index)
        idx.swap(right, partition_idx);

        // _quicksort(A, I, left, index - 1, ...)
        if partition_idx > 0 {
            _quicksort(a, idx, left, partition_idx - 1, rng);
        }
        // _quicksort(A, I, index + 1, right, ...)
        _quicksort(a, idx, partition_idx + 1, right, rng);
    }
}
