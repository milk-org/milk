---
description: Info on building tests.
---

# Testing Practices

## Writing tests

Follow the principles:



1. **Admission.** A test exists only if it names the product behavior it protects: a contract someone relies on, a bug that actually happened, or a parser of bytes the project did not produce.
A test may come before its code only when that code is being written in the same piece of work: the code will exist, not may exist someday.

2. **It can fail.** Breaking the protected behavior must fail the test. Break it once and watch.

3. **Cadence and levels.** While working on a change, run the targeted tests for the code it affects.

4. **Test before committing** Before any commit that changes code, run the test suite on the build. If the build options cause tests to be ignored relating to sections that are reasonably expected to have changed (e.g. work done in cli, cuda code but not currently compiled), notify the user. Benchmarks are never tests.

5. **Requirements fail loudly.** A missing tool or missing data fails the run instead of skipping it; only hardware the machine lacks or a disabled build option may skip, and the skip names it.

6. **Proportion.** A change adds tests in proportion to the behavior it changes, and its summary states the tests added and removed and the core suite's time before and after.

7. **Pruning.** Existing suites meet the same admission rule retroactively, through a ledger the owner approves.

8. **References.** A numerical test takes its reference from the test itself. Bit parity between two implementations is a validation tool, not a unit test. Fixtures are okay when their purpose is transparent; fixture factories are only to be used when the needed abstraction is unavoidable.


## Before committing

After compiling after C changes, run the ctest and pytest suites
(see [tooling.md §4](../../docs/developer/tooling.md#4-testing-through-a-standard-python-suite-pytest-and-nox)).
Benchmarks carry the ctest label `benchmark` and are excluded:

```bash
cd ${MILK_ROOT}/_build
ctest -LE benchmark --output-on-failure
cd ${MILK_ROOT}/python
pytest
```

## Regression Tests

When fixing a bug, **always** add a regression
test that reproduces the original failure:

1. Write a test case that triggers the bug.
2. Verify the test fails before the fix and
   passes after.
3. For computation bugs, add a ctest case.

## What Counts as Passing

- **ctest**: all tests must pass (exit code 0).
- **pytest**: all non-GPU tests must pass. If a GPU is available and the performed work
  is GPU-related, all GPU tests must also not be skipped and pass
