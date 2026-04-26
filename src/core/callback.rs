use std::collections::HashMap;

use crate::core::{algorithm::Algorithm, individual::Value};

/// Instance data for a `Callback`.
///
/// Mirrors the `self.data` and `self.is_initialized` fields of
/// `pymoo.core.callback.Callback.__init__`.
pub struct CallbackBase {
    pub data: HashMap<String, Value>,
    pub is_initialized: bool,
}

impl CallbackBase {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            is_initialized: false,
        }
    }
}

/// Lifecycle hook invoked after each generation.
///
/// Mirrors `pymoo.core.callback.Callback`.
pub trait Callback {
    fn base(&self) -> &CallbackBase;
    fn base_mut(&mut self) -> &mut CallbackBase;

    /// Mirrors `Callback.initialize(algorithm)` — called once before the first
    /// `notify`/`update`.
    fn initialize(&mut self, _algorithm: &dyn Algorithm) {}

    /// Mirrors `Callback.notify(algorithm)`.
    fn notify(&mut self, _algorithm: &dyn Algorithm) {}

    /// Mirrors `Callback.update(algorithm)` → delegates to `_update`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        self._update(algorithm);
    }

    /// Mirrors `Callback._update(algorithm)` — override in concrete types.
    fn _update(&mut self, _algorithm: &dyn Algorithm) {}

    /// Mirrors `Callback.__call__(algorithm)`:
    /// initializes once, then calls `notify` and `update`.
    fn call(&mut self, algorithm: &dyn Algorithm) {
        if !self.base().is_initialized {
            self.initialize(algorithm);
            self.base_mut().is_initialized = true;
        }
        self.notify(algorithm);
        self.update(algorithm);
    }
}

/// A ordered collection of callbacks, each forwarded the same algorithm state.
///
/// Mirrors `pymoo.core.callback.CallbackCollection(Callback)`.
pub struct CallbackCollection {
    pub base: CallbackBase,
    pub callbacks: Vec<Box<dyn Callback>>,
}

impl CallbackCollection {
    /// Mirrors `CallbackCollection.__init__(*args)`.
    pub fn new(callbacks: Vec<Box<dyn Callback>>) -> Self {
        Self {
            base: CallbackBase::new(),
            callbacks,
        }
    }
}

impl Callback for CallbackCollection {
    fn base(&self) -> &CallbackBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut CallbackBase {
        &mut self.base
    }

    /// Mirrors `CallbackCollection.update(algorithm)`:
    /// `[callback.update(algorithm) for callback in self.callbacks]`.
    fn update(&mut self, algorithm: &dyn Algorithm) {
        for callback in self.callbacks.iter_mut() {
            callback.update(algorithm);
        }
    }
}
