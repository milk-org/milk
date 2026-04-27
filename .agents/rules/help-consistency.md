---
description: Keep all help and documentation sources
  in sync when any one of them changes.
---

When a task modifies help-related content, you MUST
check every sibling source that covers the same topic
and update them if they have become inconsistent.

## Documentation Source Map

The project's help content lives in these locations:

| # | Type                   | Path(s)                                              | Topic                         |
|---|------------------------|------------------------------------------------------|-------------------------------|
| 1 | Help executable (C)    | `src/cli/CLIcore/milk-cli-help.c`, `milk-help.c`     | CLI usage, startup, piping    |
| 2 | Help executable (C)    | `src/engine/libfps/milk-fps-help.c`                  | FPS concepts & management     |
| 3 | Help executable (C)    | `src/engine/libfps/milk-fpsexec-help.c`              | fpsexec standalone usage      |
| 4 | Help executable (C)    | `src/engine/libprocessinfo/milk-procinfo-help.c`     | Processinfo & real-time       |
| 5 | Help executable (C)    | `src/cli/CLIcore/milk-cli-help-synchro.c`            | Synchronization overview      |
| 6 | Static help text       | `src/cli/CLIcore/doc/help.txt`                       | CLI options, syntax, FITS I/O |
| 7 | In-code help (C)       | `src/cli/CLIcore/CLIcore/CLIcore_help_*.c`           | Interactive `?`/`help`/`cmd?` |
| 8 | Kernel-Doc / Doxygen   | `@brief`/`@file` comments in `.c`/`.h` files         | Per-function API docs         |
| 9 | FPS_APP_INFO           | `.description` field in each fpsexec source           | One-line summaries (`-h1`)    |
|10 | Markdown docs          | `docs/*.md` (fps.md, streams.md, procinfo.md, cli/)  | User/developer reference      |
|11 | Module READMEs         | `src/*/README.md`                                    | Per-module overviews          |
|12 | Programmer's Guide     | `docs/programmers_guide.md`                          | Architecture overview         |
|13 | Examples / tutorials   | `src/milk_module_example/examples/`                  | Getting-started walkthroughs  |

## Cross-Reference Groups

Sources that cover the **same topic** must stay
consistent with each other. The main groups are:

1. **CLI usage & options**
   Rows 1, 6, 7, 10 (`docs/cli/CLI_Overview.md`,
   `docs/cli/CLIcore.md`)

2. **FPS concepts & management**
   Rows 2, 3, 10 (`docs/fps.md`,
   `docs/FPS_Standalone_CMD_Modes.md`)

3. **Processinfo & real-time**
   Rows 4, 5, 10 (`docs/procinfo.md`)

4. **Streams / synchronization**
   Rows 5, 10 (`docs/streams.md`)

5. **fpsexec executables**
   Rows 3, 9, 10 (`docs/FPS_Standalone_CMD_Modes.md`),
   12

6. **Per-function / per-module**
   Rows 8, 9, 11, 13

## Trigger Conditions

You MUST run the cross-reference check if ANY of these
occur during a task:

1. You modify a **help executable** (rows 1–5).
2. You modify `doc/help.txt` (row 6) or
   `CLIcore_help.c` (row 7).
3. You add, rename, or change the behaviour of a
   **CLI command**, an **FPS parameter**, or a
   **fpsexec executable**.
4. You change a **markdown doc** in `docs/` that
   describes commands, options, or workflows.
5. You modify an `FPS_APP_INFO` `.description` field
   (row 9).
6. You add or modify a **module README** (row 11).

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
