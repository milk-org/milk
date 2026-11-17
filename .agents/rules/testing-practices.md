---
description: Ensure tests are run after code changes
  and regression tests are added for bug fixes.
---

# Testing Practices

The test suite is pytest under `python/tests/`, run against an install as
[tooling.md §4](../../docs/developer/tooling.md#4-testing-through-a-standard-python-suite-pytest-and-nox)
says. Run it after modifying any source file, in addition to compiling. The
[`cli-test-writer`](../skills/cli-test-writer/SKILL.md) skill covers writing CLI tests.

## Regression Tests

When fixing a bug, **always** add a regression test under `python/tests/` that reproduces the
original failure, and check that it fails before the fix and passes after.
