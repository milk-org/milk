# CLIcore

Interactive Command Line Interface (REPL) and syntax-aware UI layer for the
**milk** framework.

## Role

`CLIcore` provides:

- The interactive `milk>` CLI prompt (Readline REPL) and the session server startup
  (`milk-cli --server`).
- Module and command registration (`MILK_MODULE_RUNNABLES`, `milk_register_runnable`).
- Context-aware tab completion, interactive Zsh-style grid menu selection, and
  inline ghost-text suggestions.
- Real-time Tree-sitter and lexical syntax highlighting, delimiter matching,
  inline syntax diagnostics, and code formatting.
- Interactive TUI help browser (`milk-help`) and CLI utilities.

## Source Organization

- `CLIcore.c` — Main entry point (`runCLI()`), CLI flag parsing, session server
  startup, and Readline event loop.
- `CLIcore_cmd_registry.c` — Command registration table (`MILK_RUNNABLE` /
  `milk_register_runnable`).
- `CLIcore/CLIcore_UI_completion.c` — Readline line acceptance, multiline
  execution.
- `CLIcore/CLIcore_UI_completion_tab.c` — Tab completion and the hint line's
  suggestion (`CLI_generator()`, `CLI_completion()`): the word's context picks
  what to ask `milk_complete()`, the completer bash uses too.
- `CLIcore/CLIcore_UI_completion_menu.c` — Interactive Zsh-style grid menu for
  navigating multiple tab-completion matches (`cli_menu_display_matches()`).
- `CLIcore/CLIcore_UI_highlight.c` — Readline redisplay hook that draws the
  tree-sitter highlighting (`@config synhl on|off`).
- `CLIcore/CLIcore_UI_hintarea.c` — Inline ghost-text suggestions and reserved
  bottom-line syntax/diagnostic hint bar.
- `CLIcore/CLIcore_UI_history*.c` — Persistent command history, search, and
  display.
- `CLIcore/CLIcore_UI_builtins*.c` — Interactive UI built-in commands and FIFO
  control.
- `CLIcore/CLIcore_UI_bookmark.c` — Directory bookmark management.
- `CLIcore/CLIcore_UI_prompt.c` — Prompt string formatting (`PS1`).
- `CLIcore/CLIcore_help_tui.c` — Interactive Ncurses/ANSI help browser.
- `treesitter/cli_treesitter.h` — Public API for syntax highlighting, completion
  context, diagnostics, and formatting.
- `treesitter/cli_treesitter_internal.h` — Shared internal state across
  `cli_treesitter_*.c` translation units.
- `treesitter/cli_treesitter.c` — Tree-sitter parser lifecycle, `@` sigil input
  normalization, AST syntax highlighting, and AST bracket/keyword matching.
- `treesitter/cli_treesitter_lexical.c` — Lexical fallback scanner, entity token
  classifier (`stream.*`, `local.*`, `fps.*`, `proc.*`), and lexical completion
  mode detection.
- `treesitter/cli_treesitter_diag.c` — Real-time syntax diagnostics, incomplete
  multiline detection, and AST-based completion context analysis.
- `treesitter/cli_treesitter_format.c` — Auto-indentation depth calculation,
  script code formatting, and foldable block outline extraction.

## Related Components

- **libmilkscript** (`src/cli/libmilkscript/`): Scripting, `milk-cli`
  session server (`server/`), IPC client (`ipc/`) and Bash coprocess bridge
  (`bash/`). The shared-memory property accessor is `src/prop/` (`libmilkprop`)
  and the calculator `src/calc/` (`libmilkcalc`).
- **libfps** (`src/engine/libfps/`): Function Parameter Structure core.
- **libprocessinfo** (`src/engine/libprocessinfo/`): Process monitoring.

## Dependencies

- `milkscript`, `milkcomputeunit`, `milkfps`, `milkprocessinfo`, `milkdata`, `milkcommon`, and the
  `COREMOD_*` libraries (`src/cli/CLIcore/CMakeLists.txt` has the list)
- System: `readline`, `ncurses`, `tree-sitter` (optional)
