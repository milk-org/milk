# libmilkscript

Scripting engine and interactive CLI interpreter for
the milk shell. Handles command parsing, argument
validation, script execution, FPS parameter expansion,
and the built-in calculator.

## Purpose

This library powers the `milk>` interactive prompt and
batch script execution. It processes user input through
a pipeline of expansion, interception, validation, and
dispatch stages before invoking registered CLI commands.

## Major Subsystems

### Command Execution Pipeline

- `CLIcore_UI_execute.c` — Top-level `CLI_execute_line()`
- `CLIcore_UI_execute_redir.c` — I/O redirection handling
- `CLIcore_UI_execute_preproc.c` — Preprocessor passes
- `CLIcore_UI_execute_debug.c` — Debug/trace support

### Argument Checking

- `CLIcore_checkargs.c` — Type validation for CLI args
- `CLIcore_checkargs_fps.c` — FPS-aware argument binding

### Script Interpreter

- `CLIcore_script.c` — Script runner entry point
- `CLIcore_script_flow.c` — Control flow (if/while/for)
- `CLIcore_script_case.c` — Case/switch dispatch
- `CLIcore_script_func.c` — User-defined functions
- `CLIcore_script_var.c` — Variable management
- `CLIcore_script_traps.c` — Signal/error traps

### Script Interception Layer

- `CLIcore_script_intercept.c` — Intercept dispatcher
- `CLIcore_script_intercept_process.c` — Process control
- `CLIcore_script_intercept_execution.c` — Exec control
- `CLIcore_script_intercept_environment.c` — Env vars
- `CLIcore_script_intercept_flow.c` — Flow interception
- `CLIcore_script_intercept_env_trap.c` — Env traps
- `CLIcore_script_intercept_exec_waitany.c` — Wait-any
- `CLIcore_script_intercept_utils.c` — Shared helpers

### Expansion Layer

- `CLIcore_script_expand_fps.c` — FPS parameter expansion
- `CLIcore_script_expand_arith.c` — Arithmetic expansion
- `CLIcore_script_expand_env.c` — Environment expansion
- `CLIcore_script_expand_test.c` — Test expression eval

### Built-in Calculator

- `cli_calc_tokenizer.c` — Lexer
- `cli_calc_parser.c` — Parser
- `cli_calc_eval.c` — Evaluator
- `cli_calc_functions.c` — Math function library
- `cli_calc_binops.c` — Binary operator dispatch

### Shell Built-ins

- `CLIcore_script_cmd_builtins.c` — cd, echo, etc.
- `CLIcore_script_cmd_io.c` — I/O commands
- `CLIcore_script_cmd_inspect.c` — Inspect commands
- `CLIcore_script_cmd_defer.c` — Deferred execution

### Help System

- `CLIcore_help.c` — General help display
- `CLIcore_help_command.c` — Per-command help
- `CLIcore_help_topics.c` — Topic-based help pages

### Module Management

- `CLIcore_modules.c` — Dynamic module loading

### Standalone Support

- `milkscript_main.c` — Standalone `milkscript` entry
- `milkscript_api.c` — Embedding API
- `milkscript_stubs.c` — Stub implementations
- `standalone_dependencies.c` — Standalone link helpers

## Build Tier

CLI tier — requires `USE_CLI=ON`.
