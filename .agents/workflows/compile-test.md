---
description: Compile and test milk after source code changes
---

# Compile and Test milk

Run this workflow after **any** source code change in the milk (or cacao) tree.

## Steps

// turbo-all

1. Run the incremental build from the existing `_build` directory:

```bash
cd /home/oguyon/src/milk/_build && cmake --build . -- -j$(nproc)
```

2. Install so standalone executables and libraries
   are available:

```bash
cd /home/oguyon/src/milk/_build && make install
```

3. If the build **fails**, read the compiler errors, fix them, and re-run
   step 1 until the build succeeds.

4. After a successful build, run the test suite:

```bash
cd /home/oguyon/src/milk/_build && ctest --output-on-failure
```

5. If any tests fail, investigate and fix.

6. Run pre-commit formatting checks on changed files:

```bash
cd /home/oguyon/src/milk && pre-commit run clang-format --all-files
```

If clang-format reports failures, the files have been
auto-fixed in place. Re-run the command to verify
they now pass. If the fix changed any source files,
rebuild (step 1) to confirm compilation still
succeeds.

7. Report the result to the user:
   - Number of compiler warnings (if any)
   - Test results (pass/fail count)
   - Formatting check result (pass/fail)
   - If a new standalone executable was added,
     verify it appears in `milk-fpsexec-list` output:
     ```bash
     milk-fpsexec-list | grep <new-exe-name>
     ```
