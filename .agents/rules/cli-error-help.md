---
description: Argument errors of module commands, units and @args are usage errors (exit status 2)
  reported in one line, the way usage_error() does.
---

# CLI Argument Errors

The argument rules and their errors are described for users in
[Scripting — Arguments](../../docs/guide/scripting.md#arguments): a missing, invalid or extra value,
or a parameter given twice, is a usage error, the command does not run, and it exits with status 2.

- Module commands report through `usage_error()` in
  `src/cli/libmilkscript/core/CLIcore_checkargs.c`: `milk: <command>: <what is wrong>` on stderr,
  then a one-line usage; `<command> -h` shows the full help.
- Words bind to parameters in `milk_args_bind()` and values are read by `milk_argval_parse()`
  (`src/engine/libfps/params/fps_args.c`), for module commands, unit command lines
  (`fps_cli_sync.c`) and `@args` (`milk_client_args.c`) alike: change the rules there, once.
- A keyword's usage error prints its lexicon synopsis: `milk_lexicon_usage("<keyword>", stderr)`.
