# Source: infrastructure smoke check (no spec) — proves the cucumber harness
# wiring end-to-end. Real feature files carry a "# Source: test/specs/<slug>.md"
# line instead.

Feature: BDD harness smoke check
  The cucumber runner discovers features in test/features/ and executes
  steps defined in crates/vocalize-core/tests/bdd/steps/.

  Scenario: The harness runs a scenario
    Given the BDD harness is wired
    Then it executes steps
