use std::collections::VecDeque;

use ndarray::Array1;

/// Mirrors `pymoo.util.sliding_window.SlidingWindow`.
///
/// A fixed-capacity FIFO window: once full, the oldest entry is dropped on each push.
/// Backed by `VecDeque` so that front-removal is O(1) rather than O(n).
pub struct SlidingWindow<T> {
    pub size: Option<usize>,
    data: VecDeque<T>,
}

impl<T> SlidingWindow<T> {
    /// Mirrors `SlidingWindow.__init__(size)` — bounded window.
    pub fn new(size: usize) -> Self {
        Self {
            size: Some(size),
            data: VecDeque::with_capacity(size),
        }
    }

    /// Mirrors `SlidingWindow.__init__(size=None)` — unbounded window.
    pub fn unbounded() -> Self {
        Self {
            size: None,
            data: VecDeque::new(),
        }
    }

    /// Mirrors `SlidingWindow.append(entry)`.
    ///
    /// Appends the entry and evicts from the front while `len > size`.
    pub fn push(&mut self, entry: T) {
        self.data.push_back(entry);
        if let Some(size) = self.size {
            while self.data.len() > size {
                self.data.pop_front();
            }
        }
    }

    /// Mirrors `SlidingWindow.is_full()`.
    pub fn is_full(&self) -> bool {
        self.size.map_or(false, |s| self.data.len() == s)
    }

    /// Mirrors `SlidingWindow.clear()`.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
}

impl<T: Clone> SlidingWindow<T> {
    /// Mirrors `SlidingWindow.to_numpy()` — converts the window contents to a 1-D array.
    pub fn to_array(&self) -> Array1<T>
    where
        T: Clone + Default,
    {
        Array1::from_vec(self.data.iter().cloned().collect())
    }
}
