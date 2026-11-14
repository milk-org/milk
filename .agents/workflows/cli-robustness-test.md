---
description: Run the CLI robustness test suite
---

# CLI Robustness Test Suite

Run this workflow to verify that `milk-cli` correctly
handles valid commands, invalid commands, missing arguments,
and scripting constructs — without crashing, hanging, or
silently failing.

The suite is `python/tests/cli/cli_robustness_test.py`: run it with pytest as
[tooling.md §4.2](../../docs/developer/tooling.md#42-pytest) shows. Adding cases (its
`TESTLIST_EXPECT_OK`, `TESTLIST_EXPECT_ERROR` and `TESTLIST_EXPECT_GREP_*` lists) and the other
CLI test files: the [`cli-test-writer`](../skills/cli-test-writer/SKILL.md) skill.
