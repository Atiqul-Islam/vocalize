//! BDD acceptance-test runner (spec-forge flow).
//!
//! Gherkin features live in `test/features/` at the repository root; step
//! definitions live in `tests/bdd/steps/`. Run with:
//!
//! ```sh
//! cargo test -p vocalize-core --test bdd
//! ```
//!
//! Scenario filtering (the spec-test skill's per-feature scope) is done by
//! pointing the runner at a single feature file via the `BDD_FEATURE` env
//! var, e.g. `BDD_FEATURE=test/features/foo.feature`.

// bdd.rs is this test binary's crate root, so a bare `mod steps;` would
// resolve to tests/steps.rs; the path attribute keeps everything under
// tests/bdd/ (subdirectories of tests/ are not auto-built as targets).
#[path = "bdd/steps/mod.rs"]
mod steps;

use std::path::{Path, PathBuf};

use cucumber::World as _;

/// Shared state threaded through every scenario.
///
/// Step modules extend behavior, not this struct's invariants: keep fields
/// optional and reset-friendly (a fresh `World` is built per scenario).
#[derive(Debug, Default, cucumber::World)]
pub struct VocalizeWorld {
    /// Path of the most recent synthesis output, if a step produced one.
    pub last_output: Option<PathBuf>,
    /// Error message from the most recent fallible step, if it failed.
    pub last_error: Option<String>,
}

fn features_root() -> PathBuf {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    match std::env::var("BDD_FEATURE") {
        Ok(one) => repo_root.join(one),
        Err(_) => repo_root.join("test/features"),
    }
}

#[tokio::main]
async fn main() {
    VocalizeWorld::cucumber()
        .fail_on_skipped()
        .run_and_exit(features_root())
        .await;
}
