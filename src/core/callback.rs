use std::collections::HashMap;

struct CallbackBase {
    data: HashMap<String, f64>,
    is_initialized: bool,
}

impl CallbackBase {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            is_initialized: false,
        }
    }
/*
def initialize(self, algorithm):
        pass

    def notify(self, algorithm):
        pass

    def update(self, algorithm):
        return self._update(algorithm)

    def _update(self, algorithm):
        pass

    def __call__(self, algorithm):

        if not self.is_initialized:
            self.initialize(algorithm)
            self.is_initialized = True

        self.notify(algorithm)
        self.update(algorithm)
*/
}

pub trait Callback {
    fn base(&self) -> &CallbackBase;
}

pub struct DefaultCallback {
    base: CallbackBase,
}

impl DefaultCallback {
    pub fn new() -> Self {
        Self {
            base: CallbackBase::new(),
        }
    }
}

impl Callback for DefaultCallback {
    fn base(&self) -> &CallbackBase {
        &self.base
    }
}

/*
class CallbackCollection(Callback):

    def __init__(self, *args) -> None:
        super().__init__()
        self.callbacks = args

    def update(self, algorithm):
        [callback.update(algorithm) for callback in self.callbacks]
*/
