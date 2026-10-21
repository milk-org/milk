---
description: Keep all help and documentation sources
  in sync when any one of them changes.
---

When a task modifies help-related content, you MUST
check every sibling source that covers the same topic
and update them if they have become inconsistent.

## Documentation Source Map

The project's help content lives in these locations:

| #   | Type                 | Path(s)                                             | Topic                         |
| --- | -------------------- | --------------------------------------------------- | ----------------------------- |
| 1   | Help executable (C)  | `src/cli/CLIcore/milk-cli-help.c`, `milk-help.c`    | CLI usage, startup, piping    |
| 2   | Help executable (C)  | `src/engine/libfps/tools/milk-fps-help.c`                 | FPS concepts & management     |
| 3   | Help executable (C)  | `src/engine/libfps/tools/milk-fpsexec-help.c`             | fpsexec standalone usage      |
| 4   | Help executable (C)  | `src/engine/libprocessinfo/milk-procinfo-help.c`    | Processinfo & real-time       |
| 5   | Help executable (C)  | `src/cli/CLIcore/milk-cli-help-synchro.c`           | Synchronization overview      |
| 6   | In-code help (C)     | `src/cli/CLIcore/CLIcore/CLIcore_help_*.c`          | Interactive `?`/`help`/`cmd?` |
| 7   | Kernel-Doc / Doxygen | `@brief`/`@file` comments in `.c`/`.h` files        | Per-function API docs         |
| 8   | Runnable descriptors | `.app.description` of each `MILK_RUNNABLE`          | One-line summaries (`-h1`)    |
| 9   | Markdown docs        | `docs/guide/*.md` (fps.md, streams.md, procinfo.md) | User/developer reference      |
| 10  | Module READMEs       | `src/*/README.md`                                   | Per-module overviews          |
| 11  | Programmer's Guide   | `docs/arch/programmers_guide.md`                    | Architecture overview         |
| 12  | Examples / tutorials | `src/milk_module_example/examples/`                 | Getting-started walkthroughs  |

## Cross-Reference Groups

Sources that cover the **same topic** must stay
consistent with each other. The main groups are:

1. **CLI usage & options**
   Rows 1, 6, 9 (`docs/quickstart/CLI_Overview.md`,
   `docs/guide/CLIcore.md`)

2. **FPS concepts & management**
   Rows 2, 3, 9 (`docs/guide/fps.md`)

3. **Processinfo & real-time**
   Rows 4, 5, 9 (`docs/guide/procinfo.md`)

4. **Streams / synchronization**
   Rows 5, 9 (`docs/guide/streams.md`)

5. **fpsexec executables**
   Rows 3, 8, 9 (`docs/guide/fps.md` §6), 11

6. **Per-function / per-module**
   Rows 7, 8, 10, 12

## Trigger Conditions

You MUST run the cross-reference check if ANY of these
occur during a task:

1. You modify a **help executable** (rows 1–5).
2. You modify `CLIcore_help.c` (row 6).
3. You add, rename, or change the behaviour of a
   **CLI command**, an **FPS parameter**, or a
   **fpsexec executable**.
4. You change a **markdown doc** in `docs/` that
   describes commands, options, or workflows.
5. You modify the `.app.description` of a
   `MILK_RUNNABLE` (row 8)
6. You add or modify a **module README** (row 10).

## Required Action

1. Identify which cross-reference group(s) the change
   belongs to.
2. Read each sibling source in that group using
   `view_file` or `grep_search`.
3. If any sibling is now inconsistent (outdated
   command name, missing new option, wrong example,
   etc.), update it in the same task.
4. If a sibling cannot be updated (e.g., auto-generated
   content), note it in your task summary and suggest
   a follow-up.
5. Summarize all documentation updates in your task
   summary.
