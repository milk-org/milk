---
description: Run the CLI robustness test suite
---

# CLI Robustness Test Suite

Run this workflow to verify that `milk-cli` correctly
handles valid commands, invalid commands, missing arguments,
and scripting constructs — without crashing, hanging, or
silently failing.

## Steps

// turbo-all

1. Run the test suite from the repository root:

```bash
cd /home/oguyon/src/milk && bash tests/cli/run_cli_robustness_tests.sh --verbose
```

2. Review the summary output. Categorized results:
   - **PASS** — command worked as expected
   - **FAIL** — expected success but got error exit
   - **MISSING_ERROR** — expected error message but
     none was printed
   - **CRASH** — process killed by signal (segfault etc.)
   - **HANG** — command timed out

3. If any failures, examine the detailed report at:
   `tests/cli/cli_test_report.txt`

4. Fix issues in the CLI code, then re-run step 1.

5. When adding new CLI features or changing syntax,
   update `tests/cli/cli_robustness_tests.milk` with
   matching test cases.

## Test File Format

Each test block in `cli_robustness_tests.milk` has:

```
#DESC: Short description
#EXPECT:OK       (or #EXPECT:ERR)
command_line_1
command_line_2   (for multi-line constructs)
```

## Adding New Tests

1. Add a `#DESC:` + `#EXPECT:` annotation pair
2. Follow with the command(s) to test
3. Use `#EXPECT:ERR` for commands that should print
   an error message
4. For multi-line constructs (if/while/for/function),
   include all lines in the block
