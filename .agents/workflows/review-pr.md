---
description: Review a pull request for coding standards compliance
---

# Review a Pull Request

Use this workflow to systematically review a PR
for compliance with project conventions.

## 1. Read the PR

Get the PR diff and file list (`TARGET`: the PR's target branch, `framework-dev` unless the PR
targets `dev`):

```bash
TARGET=framework-dev
git diff ${TARGET}..HEAD --stat
git diff ${TARGET}..HEAD
```

## 2. Coding Style Check

Check each modified `.c` and `.h` file against the
[coding standards](../../docs/developer/coding_standards.md) and the `code-style-guide` rule.

## 3. Architecture Check

- [ ] No new cross-module dependencies violating
      `docs/arch/dependency_graph.md`
- [ ] Runnable sources include `milk_runnable.h`, not
      `CLIcore.h`; CLI exposure only via
      `MILK_MODULE_RUNNABLES`
- [ ] New modules follow
      `milk_module_example` patterns

## 4. Performance Check

For compute-function changes, check the `performance-practices` rule.

## 5. Documentation Check

- [ ] Module README updated if files changed
- [ ] Kernel-Doc on new/modified functions
- [ ] `docs/arch/programmers_guide.md` updated if
      architecture changed
- [ ] Runnable `.app.description` is descriptive
- [ ] Help sources consistent (run
      `/audit-help-consistency` if help changed)

## 6. Build and Test

Run the [`/compile-test`](compile-test.md) workflow.

## 7. Post Review

Summarize findings:

- Number of issues found by category
- Suggested changes
- Overall assessment (approve / request changes)
