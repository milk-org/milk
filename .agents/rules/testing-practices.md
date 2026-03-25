---
description: Ensure tests are run after code changes
  and regression tests are added for bug fixes.
---

# Testing Practices

## After Code Changes

After modifying any C source file, you MUST run
the test suite in addition to compiling:

```bash
cd ~/src/milk/_build
ctest --output-on-failure
```

## After CLI Changes

After modifying any file in `src/cli/CLIcore/`,
also run the CLI robustness test suite:

```bash
bash tests/cli/run_cli_robustness_tests.sh \
  --verbose
```

See the [`cli-test-writer`](../skills/cli-test-writer/SKILL.md)
skill for writing new tests.

## Regression Tests

When fixing a bug, **always** add a regression
test that reproduces the original failure:

1. Write a test case that triggers the bug.
2. Verify the test fails before the fix and
   passes after.
3. For CLI bugs, add to
   `tests/cli/cli_robustness_tests.milk`.
4. For computation bugs, add a ctest case.

## What Counts as Passing

- **ctest**: all tests must pass (exit code 0).
- **CLI robustness**: no `CRASH`, `HANG`, or
  unexpected `FAIL` results.
- `MISSING_ERROR` results should be investigated
  — the error path may need a proper error message.
