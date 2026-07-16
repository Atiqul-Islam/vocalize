---
name: spec-crap
description: Compute the CRAP (Change Risk Analyzer and Predictor) score per function, flag functions above threshold
context: full
allowed-tools: Bash(pytest *), Bash(radon *), Bash(python *), Bash(ls *), Read, Glob
---

# Spec CRAP Skill

Compute the **CRAP** score for every production function and flag high-risk functions before `/code-review`.

CRAP combines cyclomatic complexity with unit-test coverage into a single change-risk score per function:

```
CRAP(f) = CC(f)² × (1 − cov(f)/100)³ + CC(f)
```

- At 100% coverage, CRAP collapses to CC.
- At 0% coverage, CRAP ≈ CC² + CC.

This skill runs as the first half of step 8 REVIEW. It produces the report; `/code-review` consumes it.

## Usage

```
/spec-crap              Full report over all production code
/spec-crap --changed    Limit to files changed vs. the merge base
```

The argument is provided via `$ARGUMENTS`.

## Thresholds

Three tiers, layered from Uncle Bob's own repos:

| Tier | CRAP | Meaning |
|------|------|---------|
| alert | > 30 | Function is crappy by the conventional Savoia/Evans line |
| **fail** | **> 8** | Must be addressed before commit (matches `crap4java`) |
| target | ≤ 4 | Goal during refactor (matches SwarmForge reviewer) |

Exit code 2 is returned when any function has CRAP > 8.

Complexity counters differ per language tool (radon counts boolean operators; lizard counts match arms and `&&`/`||`), so scores are not comparable across languages — but the thresholds are kept identical on purpose. Fix the code, not the calibration.

## Workflow

### Phase 0: Detect Language Paths

Detect which paths apply — a repo can have both; run every path that applies:

- **Rust path** applies if a cargo workspace exists (this repo: `crates/Cargo.toml`, production code in `crates/*/src/`).
- **Python path** applies if a `src/` directory with Python files exists.

If a path's layout is absent, skip that path. If NO path applies, print `"CRAP check skipped — no production source layout found"` and exit 0.

### Phase 1: Parse Arguments

1. **Parse `$ARGUMENTS`**:
   - `--changed` → scope to files changed vs. `master` (fallback to `main` if `master` is missing).
   - Any other value or empty → full production scope.

## Python Path

### Phase 2: Preflight

2. **Check tools are installed** — run `radon --version` and `python -c "import coverage"`. If either is missing, print:

   ```
   CRAP check (Python) skipped — <tool> not installed.
   Install: pip install radon coverage pytest-cov
   ```

   End this path. CRAP is advisory when tooling is absent.

3. **Check src/ exists.** If not, print `"CRAP check (Python) skipped — no src/ directory"` and end this path.

### Phase 3: Produce Coverage JSON

4. **Reuse existing coverage data if fresh.** If `test-results/coverage.json` exists and is less than 10 minutes old, skip the coverage run.

5. **Otherwise run coverage**:

   ```bash
   pytest --cov=src --cov-report=json:test-results/coverage.json test/unit/ -q
   ```

   If `test/unit/` has no tests, pytest will exit 5. Treat that as "0% coverage across all functions" — still run Phase 4 so complexity alone is surfaced.

### Phase 4: Produce Cyclomatic Complexity JSON

6. **Run radon over the scope from Phase 1**:

   ```bash
   radon cc -s --json <scope> > test-results/radon-cc.json
   ```

   Where `<scope>` is `src/` for a full report, or the space-separated list of changed files for `--changed`.

### Phase 5: Compute

7. **Invoke the analyzer**:

   ```bash
   python3 test/tools/crap.py
   ```

## Rust Path

### Phase 2: Preflight

2. **Check tools are installed** — run `cargo llvm-cov --version` and `uvx lizard --version`. If either is missing, print:

   ```
   CRAP check (Rust) skipped — <tool> not installed.
   Install: rustup component add llvm-tools-preview && cargo install cargo-llvm-cov
            (lizard runs via uvx; install uv if missing)
   ```

   End this path. CRAP is advisory when tooling is absent.

3. **Check the workspace exists.** If `crates/Cargo.toml` (or a root `Cargo.toml`) is missing, print `"CRAP check (Rust) skipped — no cargo workspace"` and end this path.

### Phase 3: Produce Coverage JSON

4. **Reuse existing coverage data if fresh.** If `test-results/rust-coverage.json` exists and is less than 10 minutes old, skip the coverage run.

5. **Otherwise run coverage** over the test targets that currently compile (this repo today: vocalize-core's lib tests; vocalize-rust's in-source tests are stale and under repair):

   ```bash
   mkdir -p test-results
   cd crates && cargo llvm-cov -p vocalize-core --lib --json \
       --output-path ../test-results/rust-coverage.json && cd ..
   ```

   Add further green targets to the same invocation as they come online (e.g. `--test bdd`, `--test spec_<slug>`) — coverage should reflect every passing test layer. A target has "come online" when `cargo check -p vocalize-core --test <name>` (from `crates/`) exits 0. If the test run itself fails, report ERROR and stop this path — a broken build is not a coverage number.

### Phase 4: Produce Cyclomatic Complexity CSV

6. **Run lizard over the scope from Phase 1**:

   ```bash
   uvx lizard -l rust --csv crates/vocalize-core/src/ crates/vocalize-rust/src/ \
       > test-results/lizard-cc.csv
   ```

   For `--changed`, pass the changed production `.rs` files (skip anything under `tests/`) instead of the directories.

   Regenerate the CSV unconditionally — lizard is cheap; the 10-minute reuse window applies only to the coverage JSON.

### Phase 5: Compute

7. **Invoke the analyzer**:

   ```bash
   python3 test/tools/crap_rust.py
   ```

   It joins llvm-cov's per-function region coverage with lizard's per-function CC by file + line-span overlap (no demangler needed; closures aggregate into their enclosing function), computes CRAP, and prints a table sorted descending by score plus a summary.

   Rust calibration notes (also documented in the tool header):
   - Coverage metric is llvm **region coverage** — the same number `cargo llvm-cov report` shows.
   - Functions inside inline `#[cfg(test)] mod` blocks are excluded from the report.
   - Feature-gated code that is compiled out (e.g. the off-by-default `ggml` feature) still appears — at 0% coverage — because lizard reads source text. Read full reports with that in mind; `--changed` scope is the honest gate for feature work.

## Both Paths

### Phase 6: Propagate and Report Results

8. **Propagate the analyzer's exit code**. Exit 2 means at least one function has CRAP > 8 and the workflow should stop here. If both paths ran, the combined exit is 2 if either failed.

9. **Present results** to the user:

   **If all functions pass (exit 0):**
   ```
   CRAP Report: PASSED
   - Functions analyzed: N
   - Max CRAP: X (function: <name> in <file>)
   - Above target (>4): K
   - Above alert (>30): 0
   - Above fail (>8): 0

   Ready for /code-review.
   ```

   **If any function fails (exit 2):**
   ```
   CRAP Report: FAILED
   - Functions above fail line (>8): M
   - Functions above alert line (>30): N

   Top offenders (up to 10, highest CRAP first):
     <file>:<name> — CC=<cc>, cov=<cov>%, CRAP=<score>
     ...

   Fix by lowering cyclomatic complexity or raising unit-test coverage for each offender, then re-run /spec-crap.
   ```

## Rules

- NEVER lower a threshold to make the report pass. Fix the code.
- NEVER delete or weaken unit tests to simplify coverage attribution.
- Coverage and complexity data must be from the current state of the source — never carry forward stale JSON/CSV across implementation changes. The 10-minute reuse window exists only for back-to-back runs.

## WORKFLOW COMPLETE

After reporting results, this skill workflow is FINISHED. Return to normal conversation mode. If exit code was 0, the next step is `/code-review`.
