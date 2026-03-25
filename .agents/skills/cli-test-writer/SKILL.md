---
name: cli-test-writer
description: Author new test cases for the milk-cli
  robustness test suite with correct format and
  coverage analysis
---

# CLI Test Writer

This skill guides the agent through writing new
test cases for the `milk-cli` robustness test
suite, ensuring correct format, comprehensive
coverage, and integration with the test runner.

## When to Use

- After adding new CLI features or syntax
- After fixing a CLI bug (add regression test)
- During coverage gap analysis
- User asks to expand test coverage

## Test Suite Location

- **Test file**:
  `tests/cli/cli_robustness_tests.milk`
- **Test runner**:
  `tests/cli/run_cli_robustness_tests.sh`
- **Report output**:
  `tests/cli/cli_test_report.txt`

## Test Block Format

Each test block consists of:

```
#DESC: Short description of what is tested
#EXPECT:OK
command_line
```

Or for error-checking tests:

```
#DESC: Short description of expected error
#EXPECT:ERR
command_that_should_fail
```

### Rules

1. `#DESC:` and `#EXPECT:` must be on separate
   lines, in that order
2. `#EXPECT:` is either `OK` or `ERR`
3. One or more command lines follow
4. A blank line or the next `#DESC:` ends the
   block
5. Multi-line constructs (if/while/for/function)
   include all lines in one block

## Test Categories

Organize tests by category. Current categories
in the test file:

| Category | Tests for |
|----------|-----------|
| Arithmetic | `a=1+2`, `a=3.14*2`, operators |
| Variables | Assignment, types, substitution |
| Strings | Quoting, escaping, concatenation |
| Control flow | `if`/`elif`/`else`, `while`, `for` |
| Functions | `function` blocks, recursion |
| Error handling | Invalid syntax, missing args |
| Shell bypass | System commands via CLI |
| Built-in commands | `listim`, `m?`, `help`, etc. |
| Stream operations | Image creation, slicing |
| Math functions | `sin()`, `cos()`, `sqrt()`, etc. |
| Aliases | `alias`, `unalias` |

## Writing Good Tests

### OK test — verify feature works

```
#DESC: Integer addition
#EXPECT:OK
a=2+3

#DESC: Nested parentheses in expression
#EXPECT:OK
a=(2+3)*(4-1)
```

### ERR test — verify error is caught

```
#DESC: Division by zero produces error
#EXPECT:ERR
a=1/0

#DESC: Unknown function produces error
#EXPECT:ERR
a=bogus(42)
```

### Semantics of OK vs ERR

- **`#EXPECT:OK`** — the command must:
  - Exit with code 0
  - NOT print any message containing "ERROR"
    to stderr

- **`#EXPECT:ERR`** — the command must:
  - Print a message containing "ERROR" to stderr
  - (exit code may be 0 or non-zero)

### Multi-line constructs

```
#DESC: If-else block
#EXPECT:OK
a=5
if [ $a -gt 3 ]
b=1
else
b=0
fi

#DESC: For loop with range
#EXPECT:OK
for i in 1 2 3
x=$i
done
```

## Coverage Gap Analysis

To find gaps, cross-reference CLI source files
against existing tests:

### Method 1 — Feature-driven

1. List all features from `CLIcore_help.c` and
   `CLIcore_script_builtin.c`
2. For each feature, search the test file for
   a matching `#DESC:` block
3. Missing features = gaps

### Method 2 — Code-path-driven

1. Identify error-path code in CLI source:
   ```bash
   grep -n "PRINT_ERROR\|FUNC_RETURN_FAILURE" \
     src/cli/CLIcore/**/*.c
   ```
2. For each error path, check if there's an
   `#EXPECT:ERR` test that triggers it

### Method 3 — Count by category

```bash
grep '#DESC:' tests/cli/cli_robustness_tests.milk \
  | wc -l
# Total test count

grep '#EXPECT:OK' tests/cli/cli_robustness_tests.milk \
  | wc -l
# Success tests

grep '#EXPECT:ERR' tests/cli/cli_robustness_tests.milk \
  | wc -l
# Error tests
```

A healthy ratio is roughly 60% OK / 40% ERR.

## Running Tests

```bash
cd ~/src/milk
bash tests/cli/run_cli_robustness_tests.sh \
  --verbose
```

### Interpreting results

| Status | Meaning |
|--------|---------|
| `PASS` | Behaved as expected |
| `FAIL` | Expected OK but got error exit |
| `MISSING_ERROR` | Expected ERR but no error msg |
| `CRASH` | Process killed by signal |
| `HANG` | Command timed out |

### Debugging failures

For MISSING_ERROR:
1. Run the command manually with piped input
2. Check if the error path is actually reached
3. The error message might not contain "ERROR"
   (fix the test or the error message)

For CRASH:
1. Use the `debug-cli-behavior` skill
2. Check the exit report logs

## Adding Tests After Bug Fixes

When fixing a CLI bug, always add a regression
test:

1. Write a test that triggers the original bug
2. Mark it `#EXPECT:OK` or `#EXPECT:ERR` based
   on correct behavior
3. Verify the test PASSes with the fix
4. Add a comment noting the fix context:
   ```
   #DESC: Regression: var assign shows value
   #EXPECT:OK
   e=$(echo hello)
   ```
