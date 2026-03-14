---
description: Update docs/scripts.md and script help
  when shell scripts change.
---

When a task adds, renames, removes, or changes the
behaviour of any `milk-*` or `cacao-*` shell script,
you MUST:

## Trigger Conditions

1. A new script file is created.
2. An existing script is renamed or deleted.
3. A script's options, arguments, or behaviour
   change significantly.

## Required Actions

1. **Update `docs/scripts.md`** — add, rename, or
   remove the script's entry. Include a one-line
   description and the install path.
2. **Ensure `--help` support** — every script must
   print a usage summary when invoked with `--help`
   or `-h`. If the modified script lacks this,
   add a `usage()` function.
3. **Check CMake install** — verify the script is
   listed in an `install(PROGRAMS ...)` directive
   in the appropriate `CMakeLists.txt`.
4. Summarize documentation updates in the task
   summary.
