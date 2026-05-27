---
description: Investigate, fix, and verify a bug
---

# Fix a Bug

Use this workflow when investigating and fixing a
reported bug. It ensures reproduction, testing,
and documentation.

## 1. Reproduce the Bug

Confirm the bug exists with a minimal reproducer:

```bash
source ~/src/milk/local/bin/milk-setup.bash
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
| CLI crash/error   | CLIcore       | `src/cli/CLIcore/`          |
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
- For **crashes**: get a backtrace:
  ```bash
  echo "command" | gdb -batch \
    -ex run -ex bt --args milk-cli 2>&1
  ```

## 4. Fix the Bug

Apply the minimal fix. Follow coding style rules.

## 5. Add Regression Test

**Always** add a test that would have caught the
bug:

- For CLI bugs: add to
  `tests/cli/cli_robustness_tests.milk`
- For computation bugs: add a ctest case
- Use `#EXPECT:ERR` for bugs where error detection
  was missing, `#EXPECT:OK` for bugs where valid
  input was rejected.

## 6. Verify

// turbo-all

1. Compile:

```bash
cd /home/oguyon/src/milk/_build && \
cmake --build . -- -j$(nproc)
```

2. Run tests:

```bash
cd /home/oguyon/src/milk/_build && \
ctest --output-on-failure
```

3. Run CLI tests (if CLI-related):

```bash
cd /home/oguyon/src/milk && \
bash tests/cli/run_cli_robustness_tests.sh \
  --verbose
```

4. Verify the original reproducer no longer
   triggers the bug.

## 7. Documentation

- If the bug was significant, add an entry to
  `docs/whatsnew.md`.
- Update any documentation that was incorrect
  due to the bug.
