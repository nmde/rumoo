use std::{collections::HashMap, env::current_exe, path::PathBuf};

use anyhow::Result;

/// The configuration of this package in general providing the place
/// for declaring global variables.
struct Config {
    // the root directory where the package is located at
    warnings: HashMap<String, bool>,
    // whether a warning should be printed if compiled modules are not available
    show_compile_hint: bool,
    // whether when import a file the doc should be parsed - only activate when creating doc files
    parse_custom_docs: bool,
}

impl Config {
    pub fn new() -> Self {
        Self {
            warnings: HashMap::from([("not_compiled".to_string(), true)]),
            show_compile_hint: true,
            parse_custom_docs: false,
        }
    }

    /// a method defining the endpoint to load data remotely - default from GitHub repo
    pub fn data(&self) -> &str {
        "https://raw.githubusercontent.com/anyoptimization/pymoo-data/main/"
    }

    pub fn root() -> Result<Option<PathBuf>> {
        let curr = current_exe()?;
        let dirname = curr.parent();
        if dirname.is_some() {
            return Ok(Some(dirname.unwrap().to_path_buf()));
        }
        return Ok(None);
    }
}

// returns the directory to be used for imports
pub fn get_rumoo() -> Result<Option<PathBuf>> {
    Ok(Config::root()?)
}
