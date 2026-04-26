use crate::{
    core::{
        algorithm::Algorithm,
        callback::{Callback, CallbackBase},
    },
    util::display::{output::Output, progress::ProgressBar},
};

/// Renders per-generation console output and/or a progress bar.
///
/// Extends `Callback` so it participates in the algorithm's callback lifecycle
/// (`call` → `initialize` → `notify` → `update`).
///
/// Mirrors `pymoo.util.display.display.Display(Callback)`.
pub trait Display: Callback {
    /// Called after the run ends; close the progress bar if present.
    ///
    /// Mirrors `Display.finalize()`.
    fn finalize(&mut self) {}
}

/// Concrete display: prints a column table and/or updates a progress bar.
pub struct DefaultDisplay {
    pub base: CallbackBase,
    pub output: Option<Box<dyn Output>>,
    pub verbose: bool,
    pub progress: Option<ProgressBar>,
}

impl DefaultDisplay {
    /// Mirrors `Display.__init__(output=None, progress=False, verbose=False)`.
    pub fn new(output: Option<Box<dyn Output>>, verbose: bool, progress: bool) -> Self {
        Self {
            base: CallbackBase::new(),
            output,
            verbose,
            progress: if progress {
                Some(ProgressBar::new())
            } else {
                None
            },
        }
    }
}

impl Callback for DefaultDisplay {
    fn base(&self) -> &CallbackBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut CallbackBase {
        &mut self.base
    }

    /// Mirrors `Display.update(algorithm)`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        if self.verbose {
            if let Some(ref mut output) = self.output {
                // Mirrors: header = not output.is_initialized
                // (checked before output(algorithm) sets it to True)
                let header = !output.base().is_initialized;

                // Mirrors: output(algorithm)  →  Callback.__call__(algorithm)
                output.call(algorithm);

                let mut text = String::new();
                if header {
                    // Mirrors: text += output.header(border=True) + '\n'
                    text += &output.header(Some(true));
                    text += "\n";
                }
                // Mirrors: text += output.text()
                text += &output.text();

                println!("{}", text);
            }
        }

        if let Some(ref mut pb) = self.progress {
            // Mirrors: perc = algorithm.termination.perc
            let perc = algorithm
                .base()
                .termination
                .as_ref()
                .map_or(0.0, |t| t.base().perc);
            // Mirrors: progress.set(perc)
            pb.set(perc);
        }
    }
}

impl Display for DefaultDisplay {
    /// Mirrors `Display.finalize()`.
    fn finalize(&mut self) {
        // Mirrors: if self.progress: self.progress.close()
        if let Some(ref mut pb) = self.progress {
            pb.close();
        }
    }
}
