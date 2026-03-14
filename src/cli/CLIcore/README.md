# CLIcore

Command Line Interface (CLI) for the **milk** package.

Uses a hand-written recursive-descent (Pratt) parser
for expression evaluation and command dispatch.

## Role

CLIcore provides:
- The interactive `milk` CLI prompt (REPL)
- Module/command registration (`CLIADDCMD`)
- Argument parsing and type-checked function dispatch
- Startup scripts and configuration loading

## Related Components

The following frameworks have been factored out into their own libraries:
- **libfps** (`src/engine/libfps/`): Function Parameter Structure core
- **libprocessinfo** (`src/engine/libprocessinfo/`): Process monitoring
- **libmilkTUI** (`src/cli/libmilkTUI/`): Terminal UI widgets

## Dependencies

- `milkfps`, `milkprocessinfo`, `ImageStreamIO`, `libmilkdata`
- System: `readline`, `ncurses`
