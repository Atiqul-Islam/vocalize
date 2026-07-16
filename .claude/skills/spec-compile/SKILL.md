---
name: spec-compile
description: Compile a plain English spec to Gherkin feature file and step definitions
context: full
allowed-tools: Bash(npx *)
---

# Spec Compile Skill

Compile a plain English spec from `test/specs/` into a Gherkin `.feature` file and any new step definitions needed.

## Usage

```
/spec-compile <feature-name>
/spec-compile               (lists available specs to choose from)
```

**Examples:**
```
/spec-compile chart-filtering
/spec-compile session-management
```

The spec name is provided via `$ARGUMENTS`.

## Workflow

### Phase 1: Locate the Spec

1. **Parse `$ARGUMENTS`** to get the spec name.
   - If no arguments provided, list all specs in `test/specs/` using Glob and present them with AskUserQuestion for the user to choose.
   - Normalize the name: strip `.md` extension if provided, try `test/specs/<name>.md`.

2. **Read the spec file** from `test/specs/<name>.md`.
   - If not found, report error: "No spec found at `test/specs/<name>.md`. Run `/spec-create <name>` first."

### Phase 2: Inventory Existing Steps

3. **Read all existing step definition files** in `test/steps/` using Glob(`test/steps/**/*.{ts,py}`).
   - Parse each file to extract existing step patterns (Given/When/Then strings).
   - Build an inventory of reusable steps. This prevents duplicate step definitions.

### Phase 2.5: Validate Spec Structure

3b. **Check for non-UI-observable lines in Expected Behavior.**
   - Read each bullet under "Expected Behavior" and check whether it describes something observable in the UI.
   - Lines about storage backends, hashing algorithms, database tables, performance constraints, or other implementation details are NOT UI-observable.
   - If any non-UI-observable lines are found in Expected Behavior:
     - **Report the violation** to the user: "The following Expected Behavior lines are not UI-observable and should be moved to an 'Implementation Requirements' section: ..."
     - **Do NOT proceed** with compilation until the user confirms or updates the spec.
   - Also verify that every Expected Behavior bullet has a corresponding Acceptance Criterion. Flag any orphaned behaviors.
   - Implementation Requirements (if present) are noted but do NOT generate Gherkin scenarios — they map to `pytest` unit tests.

### Phase 3: Generate Feature File

4. **Read existing feature file** (if it exists) from `test/features/<name>.feature`.
   - If it exists, note the current scenarios, step choices, and assertion patterns.
   - These represent **tested, working choices** — especially assertion patterns for LLM output (e.g., regex `matching` instead of exact `containing`).

5. **Generate the `.feature` file** from the spec:

```gherkin
# Source: test/specs/<name>.md

Feature: <Feature Title from spec>
  <Description from spec>

  Scenario: <Derived from first acceptance criterion>
    Given <precondition>
    When <action>
    Then <expected outcome>

  Scenario: <Derived from second acceptance criterion>
    Given <precondition>
    When <action>
    Then <expected outcome>
```

**Generation rules:**
- Add `# Source: test/specs/<name>.md` as the first line for traceability
- Feature name and description come directly from the spec
- Each acceptance criterion maps to one or more Gherkin scenarios
- Reuse existing step definitions wherever possible (match by pattern)
- Use parameterized steps with `{string}`, `{int}`, etc. for flexibility
- Include edge case and error scenarios from the spec
- Keep scenarios focused — one behavior per scenario
- Use `And` for additional steps within a Given/When/Then block
- **If an existing `.feature` file was found**: preserve its scenario structure, step phrasing, and assertion patterns unless the spec explicitly contradicts them. Manual refinements (e.g., regex matching for LLM output, specific test data choices) represent hard-won fixes and should be carried forward. Only add/remove/restructure scenarios to reflect changes in the spec's acceptance criteria.

6. **Write the feature file** to `test/features/<name>.feature`.

### Phase 4: Generate New Step Definitions

7. **Identify steps that don't exist** in the current inventory.

8. **For each new step**, generate a step definition using the project's BDD framework:
   - **playwright-bdd (TypeScript)**: Generate `.ts` files with `createBdd()` pattern
   - **pytest-bdd (Python)**: Generate `.py` files with `@given`, `@when`, `@then` decorators
   - **Detect which framework** by checking existing step files in `test/steps/`

**Step definition rules:**
- New steps for a feature go in `test/steps/<feature-name>.steps.{ts,py}`
- Common/reusable steps go in `test/steps/common/`
- Every new step MUST throw a "not implemented" error by default — this ensures the RED step works
- Include a `// TODO: Implement` or `# TODO: Implement` comment describing what the step should do

**Step definition resilience rules:**
- **Use explicit timeouts on click/action steps** — never rely on the global test timeout
- **Prefer the locator API** — for better auto-retry on detached elements
- **Wait for readiness, not just existence** — verify the element is interactive before acting
- **Assert DOM state with waits, not snapshots** — use polling/retry patterns for async UIs

9. **Write the step definition file(s)**.

10. **Validate with bddgen.** If the project uses playwright-bdd, run `npx bddgen` from the project root with the appropriate config flag to verify the feature file parses and all step references resolve.

If using pytest-bdd, skip this step (pytest discovers features directly).

If bddgen reports errors (missing steps, syntax issues), fix them before proceeding.

### Phase 4.5: Generate Unit Test Stubs

11. **Check the spec for an `### Implementation Requirements` section.**
    - If absent, skip this phase entirely.

12. **Check for an existing unit test file** at `test/unit/test_<feature_name>.py`.
    - Convert the feature name from kebab-case to snake_case (e.g., `user-login` → `test_user_login.py`).
    - If the file exists, read it and identify which test methods are already implemented (i.e., do NOT contain `pytest.fail`). Preserve these — only add stubs for NEW requirements.

13. **Generate the unit test stub file** with:

```python
# Source: test/specs/<name>.md — Implementation Requirements

import pytest


class Test<FeatureName>:
    """Unit tests for <Feature Title> implementation requirements."""

    def test_<requirement_slug>(self):
        """<Original requirement text>"""
        # TODO: Implement
        pytest.fail("Not implemented: <requirement text>")

    def test_<requirement_slug>(self):
        """<Original requirement text>"""
        # TODO: Implement
        pytest.fail("Not implemented: <requirement text>")
```

**Generation rules:**
- Class name: Convert feature name to PascalCase (e.g., `user-login` → `TestUserLogin`)
- Method name: Convert requirement to a short snake_case slug (e.g., "Passwords are stored with bcrypt hashing" → `test_passwords_stored_with_bcrypt`)
- Each stub uses `pytest.fail("Not implemented: <requirement text>")` to ensure RED step works
- Include docstring with the original requirement text for traceability
- Add `# Source:` traceability comment as the first line

14. **Write the unit test file** to `test/unit/test_<feature_name>.py`.

### Phase 5: Present Results

15. **Show the user what was generated:**
    - The `.feature` file content
    - Any new step definition files
    - Which existing steps were reused
    - The unit test stub file (if generated)
    - Count: "Generated N scenarios from M acceptance criteria, reusing X existing steps, created Y new steps"
    - If unit test stubs were generated: "Generated Z unit test stubs from Implementation Requirements"

16. **Report next step**: "Feature compiled. Next: run `/spec-test <name>` to verify RED (both BDD tests and unit test stubs should fail since they are not implemented)."

## WORKFLOW COMPLETE

After presenting results, this skill workflow is FINISHED. Return to normal conversation mode.
