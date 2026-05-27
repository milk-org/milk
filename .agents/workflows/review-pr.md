---
description: Review a pull request for coding standards compliance
---

# Review a Pull Request

Use this workflow to systematically review a PR
for compliance with project conventions.

## 1. Read the PR

Get the PR diff and file list:

```bash
git diff framework-dev..HEAD --stat
git diff framework-dev..HEAD
```

## 2. Coding Style Check

For each modified `.c` and `.h` file, verify:

- [ ] Lines ≤ 100 characters
- [ ] Linux kernel C coding style
- [ ] Kernel-Doc comments above functions
- [ ] Explicit `#include` for every header used
- [ ] Code blocks `{ }` minimize variable scope
- [ ] Multi-line function prototypes (one arg/line)

## 3. Architecture Check

- [ ] No new cross-module dependencies violating
      `docs/dependency_graph.md`
- [ ] Standalone executables link `_compute`
      variants only (never `CLIcore`)
- [ ] Dual-mode files use `#ifdef MILK_NO_CLI`
      pattern
- [ ] New modules follow
      `milk_module_example` patterns

## 4. Performance Check

For compute-function changes:

- [ ] `MILK_HOT` on hot functions
- [ ] `restrict` on pixel pointer parameters
- [ ] Float math uses `f` variants (`sqrtf`)
- [ ] No `malloc`/`free` in per-frame loops
- [ ] No `printf` in hot paths
- [ ] `else if` chains for datatype dispatch
- [ ] BLAS for matrix operations

## 5. Documentation Check

- [ ] Module README updated if files changed
- [ ] Kernel-Doc on new/modified functions
- [ ] `docs/programmers_guide.md` updated if
      architecture changed
- [ ] FPS_APP_INFO `.description` is descriptive
- [ ] Help sources consistent (run
      `/audit-help-consistency` if help changed)

## 6. Build and Test

// turbo-all

```bash
cd /home/oguyon/src/milk/_build && \
cmake --build . -- -j$(nproc) && \
make install && \
ctest --output-on-failure
```

## 7. Post Review

Summarize findings:

- Number of issues found by category
- Suggested changes
- Overall assessment (approve / request changes)
