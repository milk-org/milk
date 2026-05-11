---
description: CLI commands must print a red error message
  and colored help on argument errors.
---

# CLI Argument Error + Help Display

When a CLI command fails due to missing arguments or
wrong argument types, the error handler must:

1. **Print the error message with "ERROR" in red**
   using ANSI escape codes (bold red `\033[1;31m`).
2. **Print the full command help** by calling
   `help_command()` — this produces the same
   colored output as the `-h` / `?` help display.

## Where This Applies

- `CLI_checkarg_array()` in
  `CLIcore_checkargs.c` — handles both
  missing-argument and wrong-type failures.
- Both paths (missing mandatory arg and
  accumulated type errors) must emit the red
  error line **and** the help output.

## Error Message Format

Use a consistent format for argument errors:

```c
printf("\n\033[1;31mERROR\033[0m %s\n",
       error_description);
```

## Implementation Pattern

```c
/* Missing mandatory argument */
printf("\n\033[1;31mERROR\033[0m "
       "Missing mandatory argument %d "
       "(%s: %s)\n",
       CLIarg, tag, descr);
help_command(data.cmd[data.cmdindex].key);
return RETURN_CLICHECKARGARRAY_FAILURE;

/* Wrong argument type(s) after all args checked */
if (nberr > 0)
{
    printf("\n\033[1;31mERROR\033[0m "
           "%d argument(s) have wrong type\n",
           nberr);
    help_command(data.cmd[data.cmdindex].key);
    return RETURN_CLICHECKARGARRAY_FAILURE;
}
```

## What NOT to Do

- Do not print help for runtime errors that are
  unrelated to argument parsing (e.g., file not
  found, stream unavailable). Those are handled
  by `PRINT_ERROR()` macros.
- Do not duplicate the help output — call
  `help_command()` once per error path.
