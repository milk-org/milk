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
| `milk_syntax_test.py`         | the language: keywords, arguments, catalog, legacy error checks   |
| `bash_translation_test.py`    | how milk lines become bash: heredocs, quoting, line numbers       |
| `milk_cli_test.py`            | the milk-cli program: exit, traps, piped input, the session log   |
| `completion_test.py`          | tab completion in bash and at the milk-cli prompt                 |
| `calc_script_language_test.py` | calculator language rules: values, names, conditions, errors     |
| `calc_execution_test.py`      | how calculator programs run: fusion, reductions, replanning       |
| `removed_forms_test.py`       | removed spellings and features fail with a hint                   |
| `bash_integration_test.py`    | longer scenarios in plain bash after `source milk.bash`           |
| `units_sessions_test.py`      | `@session` lifecycle, `milk-fpsexec-milk-calc`, `milk-fpsexec-milk-script` |
| `cli_robustness_test.py`      | command lists that must run, fail, or print a given text          |

## Rules

- Use the helpers in `conftest.py` (`milk_bash`, `milk_cli_script`, `milk_script`, `_run_bash`,
  `value`): each gives the test its own HOME and SHM directory. `CAM` there makes an 8x8 `cam`,
  and `UNIT_HELPERS` adds the shell functions that unit tests use.
- A test for behaviour not built yet is `xfail(strict=True)` and names its work package (`@wp`,
  `@bug`); remove the marker when it passes.
- Run the suites with an isolated home:
  `HOME=$(mktemp -d) python3 -m pytest -q python/tests/cli/<file>`.
