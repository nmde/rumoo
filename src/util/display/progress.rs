use indicatif::{ProgressBar as InnerBar, ProgressStyle};

/// A terminal progress bar that enforces non-decreasing progress.
///
/// Wraps `indicatif::ProgressBar` the same way the Python implementation wraps
/// `alive_progress.alive_bar`.
///
/// Mirrors `pymoo.util.display.progress.ProgressBar`.
pub struct ProgressBar {
    pub non_decreasing: bool,
    _max: f64,
    bar: Option<InnerBar>,
}

impl ProgressBar {
    /// Mirrors `ProgressBar.__init__(start=True, non_decreasing=True)`.
    pub fn new() -> Self {
        let mut pb = Self {
            non_decreasing: true,
            _max: 0.0,
            bar: None,
        };
        pb.start();
        pb
    }

    /// Create without auto-starting; call `start()` explicitly.
    ///
    /// Mirrors `ProgressBar.__init__(start=False, ...)`.
    pub fn new_deferred(non_decreasing: bool) -> Self {
        Self {
            non_decreasing,
            _max: 0.0,
            bar: None,
        }
    }

    /// Update the displayed progress.
    ///
    /// When `non_decreasing` is set, the value is clamped to the running
    /// maximum so the bar never moves backwards.
    ///
    /// Mirrors `ProgressBar.set(value)`:
    /// ```python
    /// prec = 100
    /// value = math.floor(value * prec) / prec
    /// self.obj(value)
    /// ```
    pub fn set(&mut self, value: f64) {
        let value = if self.non_decreasing {
            self._max = self._max.max(value);
            self._max
        } else {
            value
        };

        // Mirrors: prec = 100; value = math.floor(value * prec) / prec
        // Then scale to indicatif's integer position range (0..=100).
        let position = (value * 100.0).floor() as u64;

        if let Some(ref bar) = self.bar {
            bar.set_position(position);
        }
    }

    /// Create and display the progress bar.
    ///
    /// Mirrors `ProgressBar.start()`.
    pub fn start(&mut self) {
        if self.bar.is_none() {
            let bar = InnerBar::new(100);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{bar:40.cyan/blue}] {percent:>3}%")
                    .unwrap_or_else(|_| ProgressStyle::default_bar())
                    .progress_chars("=>-"),
            );
            self.bar = Some(bar);
        }
    }

    /// Finish and hide the progress bar.
    ///
    /// Mirrors `ProgressBar.close()`.
    pub fn close(&mut self) {
        // Take ownership so finish() is called at most once.
        if let Some(bar) = self.bar.take() {
            bar.finish();
        }
    }
}

/// Mirrors `ProgressBar.__exit__` — close on drop.
impl Drop for ProgressBar {
    fn drop(&mut self) {
        self.close();
    }
}
