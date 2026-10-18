---
name: cli-test-writer
description: Add regression tests for milk-cli, milk-script
  and milk.bash (python/tests/cli)
---

# CLI Test Writer

Use after adding a CLI feature or fixing a CLI bug.

## Read First

- [docs/guide/scripting.md](../../../docs/guide/scripting.md) — the behaviour the tests check.

## Where Tests Go

| File (`python/tests/cli/`)    | Covers                                                            |
| ----------------------------- | ----------------------------------------------------------------- |
| `syntax_redesign_test.py`     | the language: keywords, arguments, catalog, `--modernize` goldens |
| `syntax_enhancements_test.py` | Phase C additions (S1-S7): binding, `@defer` scopes, calculator, events, `@args`, undo |
| `bash_engine_review_test.py`  | the bash engine: translator, server, coprocess, regressions       |
| `bash_integration_test.py`    | longer scenarios in plain bash after `source milk.bash`           |
| `cli_robustness_test.py`      | command lists that must run, fail, or print a given text          |

## Rules

- Use the helpers of `bash_engine_review_test.py` (`milk_cli_script`, `milk_script`, `_run_bash`,
  `value`) and `milk_bash` in `syntax_redesign_test.py`: each gives the test its own HOME and SHM
  directory.
- A test for behaviour not built yet is `xfail(strict=True)` and names its work package (`@wp`,
  `@bug`); remove the marker when it passes.
- Every removed spelling gets a `MODERNIZE_CASES` golden (old → new).
- Run the suites with an isolated home:
  `HOME=$(mktemp -d) python3 -m pytest -q python/tests/cli/<file>`.
