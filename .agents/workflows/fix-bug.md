---
description: Investigate, fix, and verify a bug
---

# Fix a Bug

Use this workflow when investigating and fixing a
reported bug. It ensures reproduction, testing,
and documentation.

## 1. Reproduce the Bug

Confirm the bug exists with a minimal reproducer, with the install on `PATH`
([tooling.md §6.1](../../docs/developer/tooling.md#61-install-and-use)):

```bash
echo "trigger_command" | milk-cli 2>&1
echo "Exit code: $?"
```

For standalone executable bugs:

```bash
milk-fpsexec-<name> <args> 2>&1
```

Document the exact reproduction steps.

## 2. Identify the Component

Determine which subsystem is affected:

| Symptom           | Component     | Key Files                   |
| ----------------- | ------------- | --------------------------- |
| CLI crash/error   | CLI           | `src/cli/` (`CLIcore/`, `libmilkscript/`) |
| Stream corruption | ImageStreamIO | `src/engine/ImageStreamIO/` |
| FPS sync issue    | libfps        | `src/engine/libfps/`        |
| Standalone crash  | fpsexec unit  | Module source               |
| Build failure     | CMake         | `CMakeLists.txt` files      |

## 3. Diagnose

- For **CLI bugs**: use the `debug-cli-behavior`
  skill.
- For **build failures**: use the
  `diagnose-build-failure` skill.
- For **computation errors**: add temporary debug
  output guarded by `if (VERBOSE > 0)`.
- For **crashes**: get a backtrace as
  [tooling.md §7](../../docs/developer/tooling.md#7-debugging-a-crashing-fpsexec-gdb-and-valgrind)
  and [debugging.md §3](../../docs/operations/debugging.md#3-gdb-debugging) show.

## 4. Fix the Bug

Apply the minimal fix. Follow coding style rules.

## 5. Add Regression Test

**Always** add a test that would have caught the
bug: a pytest case under `python/tests/`
([tooling.md §4.2](../../docs/developer/tooling.md#42-pytest)). For CLI bugs, the
`cli-test-writer` skill says where; in `cli_robustness_test.py`, use `TESTLIST_EXPECT_ERROR`
when error detection was missing and `TESTLIST_EXPECT_OK` when valid input was rejected.

## 6. Verify

1. Run the [`/compile-test`](compile-test.md) workflow (build, local install, tests), including
   the regression test of step 5.

2. Verify the original reproducer no longer
   triggers the bug.

## 7. Documentation

- If the bug was significant, add an entry to
  `docs/whatsnew.md`.
- Update any documentation that was incorrect
  due to the bug.
