//! Steps for test/features/bdd-harness-smoke.feature — proves the cucumber
//! wiring compiles, discovers features, and executes steps. Not tied to a
//! spec; library behavior scenarios come from the spec-forge flow.

use cucumber::{given, then};

use crate::VocalizeWorld;

#[given("the BDD harness is wired")]
fn harness_wired(_world: &mut VocalizeWorld) {}

#[then("it executes steps")]
fn executes_steps(_world: &mut VocalizeWorld) {}
