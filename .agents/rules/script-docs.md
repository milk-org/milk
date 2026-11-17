---
description: Update docs/guide/scripts.md and script help
  when shell scripts change.
---

When a task adds, renames, removes, or changes the
behaviour of any `milk-*` shell script (cacao's
scripts are documented in cacao),
you MUST:

## Trigger Conditions

1. A new script file is created.
2. An existing script is renamed or deleted.
3. A script's options, arguments, or behaviour
   change significantly.

## Required Actions

1. **Update `docs/guide/scripts.md`** — add, rename, or
   remove the script's row in the table of its source
   folder.
2. **Ensure `--help` support** (`script-naming.md`): if
   the modified script lacks it, add a `usage()` function.
3. **Check CMake install** — verify the script is
   listed in an `install(PROGRAMS ...)` directive
   in the appropriate `CMakeLists.txt`.
4. Summarize documentation updates in the task
   summary.
