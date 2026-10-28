---
name: cli-test-writer
description: Add regression tests for milk-cli, milk-script
  and milk.bash (python/tests/cli)
---

# CLI Test Writer

Use after adding a CLI feature or fixing a CLI bug.

## Read First

- [docs/guide/scripting.md](../../../docs/guide/scripting.md) — top-level language behavior,
  module commands, entities, events, and exit codes.
- [docs/guide/calculator.md](../../../docs/guide/calculator.md) — `@calc` expressions, functions,
  and `milk-fpsexec-milk-calc` / `milk-fpsexec-milk-script` units.
- [docs/guide/sessions.md](../../../docs/guide/sessions.md) — `@session`, `source milk.bash`,
  concurrency, and state snapshots.

## Where Tests Go

| File (`python/tests/cli/`)    | Covers                                                            |
| ----------------------------- | ----------------------------------------------------------------- |
| `syntax_redesign_test.py`     | the language: keywords, arguments, catalog, legacy error checks   |
| `syntax_enhancements_test.py` | Phase C additions (S1-S7): binding, `@defer` scopes, calculator, events, `@args`, undo |
| `bash_engine_review_test.py`  | the bash engine: translator, server, coprocess, regressions       |
| `bash_integration_test.py`    | longer scenarios in plain bash after `source milk.bash`           |
| `units_sessions_test.py`      | `@session` lifecycle, `milk-fpsexec-milk-calc`, `milk-fpsexec-milk-script` |
| `cli_robustness_test.py`      | command lists that must run, fail, or print a given text          |

## Rules

- Use the helpers of `bash_engine_review_test.py` (`milk_cli_script`, `milk_script`, `_run_bash`,
  `value`) and `milk_bash` in `syntax_redesign_test.py`: each gives the test its own HOME and SHM
  directory.
- A test for behaviour not built yet is `xfail(strict=True)` and names its work package (`@wp`,
  `@bug`); remove the marker when it passes.
- Run the suites with an isolated home:
  `HOME=$(mktemp -d) python3 -m pytest -q python/tests/cli/<file>`.
