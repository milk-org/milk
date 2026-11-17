---
description: Compile and test milk after source code changes
---

# Compile and Test milk

Run this workflow after **any** source code change in the milk (or cacao) tree.

## Steps

// turbo-all

1. From the repository root, run the incremental build in the existing `_build` directory:

```bash
cmake --build _build -- -j$(nproc)
```

2. Install so standalone executables and libraries are available, under `_build/_install` and
   never the system prefix
   ([tooling.md §6](../../docs/developer/tooling.md#6-dev-installs-without-sudo)):

```bash
cmake --install _build --prefix "$PWD/_build/_install"
```

3. If the build **fails**, read the compiler errors, fix them, and re-run
   step 1 until the build succeeds.

4. After a successful build, run the tests (the whole suite, or the files covering the change) as
   [tooling.md §4.2](../../docs/developer/tooling.md#42-pytest) shows, with the install of step 2
   first on `PATH` (otherwise the tests run an older install):

```bash
export MILK_INSTALLDIR="$PWD/_build/_install" PATH="$PWD/_build/_install/bin:$PATH"
cd python && pytest -q --color=no --tb=short
```

5. If any tests fail, investigate and fix.

6. Run pre-commit formatting checks on changed files:

```bash
pre-commit run clang-format --files <changed files>
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
