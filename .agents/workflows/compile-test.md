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

2. If the build **fails**, read the compiler errors, fix them, and re-run
   step 1 until the build succeeds.

3. After a successful build, run the test suite:

```bash
cd /home/oguyon/src/milk/_build && ctest --output-on-failure
```

4. If any tests fail, investigate and fix.

5. Report the result to the user:
   - Number of compiler warnings (if any)
   - Test results (pass/fail count)
   - If a new standalone executable was added,
     verify it appears in `milk-fpsexec-list` output:
     ```bash
     milk-fpsexec-list | grep <new-exe-name>
     ```
