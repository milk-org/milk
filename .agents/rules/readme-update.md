---
description: Update module README when source files
  are added or removed.
---

When a task adds, removes, or renames a source file
(`.c` or `.h`) within a module directory, you MUST
update that module's `README.md`.

## Trigger Conditions

1. A new `.c` or `.h` file is added to a module.
2. An existing source file is renamed or deleted.
3. A source file's purpose changes significantly
   (warrants a description update).
4. An `EXPORT_FPSEXEC` line is added, removed or
   renamed (the fpsexec table).

## Required Actions

1. Update the module's `README.md` (e.g.,
   `src/coremods/COREMOD_arith/README.md`), whose
   contents [programmers_guide.md §7](../../docs/arch/programmers_guide.md#7-c-source-file-conventions)
   lists: add, remove or rename the rows of the source
   file and fpsexec tables.
2. Summarize documentation updates in the task
   summary.
