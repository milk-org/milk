---
name: debug-cli-behavior
description: Systematic investigation of milk-cli
  runtime issues (crashes, display bugs, missing
  errors, unexpected behavior)
---

# Debug CLI Behavior

This skill provides a systematic methodology for
investigating `milk-cli` runtime issues. It covers
safe execution patterns, common root causes, and
the internal architecture that most bugs trace to.

## When to Use

- `milk-cli` crashes (segfault, abort)
- Commands produce wrong output or no output
- Error messages are missing or incorrect
- Display is corrupted (prompt, readline, TUI)
- Module commands appear empty or duplicated
- Shell bypass behaves unexpectedly

## Safe Execution Patterns

> [!CAUTION]
> Never run `milk-cli` interactively from the
> agent — it blocks as a REPL. Always use piped
> input.

### Basic command test

```bash
echo "command args" | milk-cli 2>&1
echo "Exit code: $?"
```

### Multi-line test

```bash
milk-cli <<'EOF' 2>&1
command1
command2
EOF
echo "Exit code: $?"
```

### Capture both stdout and stderr separately

```bash
echo "command" | milk-cli >out.txt 2>err.txt
echo "Exit: $?"
cat out.txt
cat err.txt
```

### Test with the robustness suite

```bash
cd python && python3 -m pytest -q tests/cli/cli_robustness_test.py
```

## Crash Investigation

### Step 1 — Reproduce

Run the crashing command with piped input and
capture the exit code:

```bash
echo "crashing_command" | milk-cli 2>&1
echo "Signal: $?"
```

Exit codes > 128 indicate signal death:

- 139 = SIGSEGV (segfault)
- 134 = SIGABRT (assertion / abort)
- 136 = SIGFPE (division by zero)

### Step 2 — Check exit reports

`milk-cli` writes crash reports to the current
directory:

```bash
ls exitreport-SIG*.log
cat exitreport-SIGSEGV.*.log
```

These contain a backtrace from the signal handler.

### Step 3 — GDB backtrace

For detailed debugging:

```bash
echo "crashing_command" | gdb -batch \
  -ex run -ex bt -ex quit \
  --args milk-cli 2>&1
```

### Step 4 — Identify the source

Common crash locations and their causes:

| Crash location                 | Likely cause                                         |
| ------------------------------ | ---------------------------------------------------- |
| `CLIcore_checkargs.c`          | `nbarg` mismatch, missing `FPFLAG_PRIMARY_CLI_INPUT` |
| `RegisterCLIcmd`               | `MILK_RUNNABLE` with NULL `.cli_func` / `.farg`      |
| `mc_eval.c`                    | Expression evaluation in `src/calc/`                 |
| `mc_lex.c` / `mc_parse.c`      | Lexer / Pratt parser in `src/calc/`                  |
| `CLIcore_modules.c`            | Module load order, `dlopen` failure                  |
| `image_ID()` / `variable_ID()` | Invalid image index, corrupted `data.image[]` array  |

## Command Registration Architecture

Understanding how commands are registered is key
to diagnosing "empty command" and "command not
found" issues.

### Registration flow

```
1. CLIcore.c: main()
     → load_module_shared()        [for each .so]
       → dlopen(sofile)
       → dlsym("__milk_module_info")
       → newstyle_milk_module_registration()
         → for each MILK_RUNNABLE of the module registry
           → milk_runnable_prepare() + RegisterCLIcmd()
             → populates data.cmd[i], data.cmd[i].runnable
```

### Key data structures

- `data.cmd[]` — array of all registered commands
  (type `CMD`)
- `data.module[]` — array of loaded modules
  (type `MODULE`)
- `data.moduleindex` — **global** index of the
  module currently being loaded (source of race
  conditions)
- `MILK_RUNNABLE` — per-command descriptor (see
  `src/engine/libcomputeunit/milk_runnable.h`); its
  `rt->cmddata` holds the command's CLI keyword, args, etc.

### Common issues

1. **Empty / missing commands**: the runnable is not
   in the array given to `MILK_MODULE_RUNNABLES`, its
   source file was dropped by an unmet
   `MILK_CMAKE_MANDATE_*` (the entry is then a NULL
   weak declaration, skipped silently), or `.app.cmdkey`
   is empty.

2. **Wrong module metadata**: `data.moduleindex`
   is a global that gets overwritten when modules
   load each other transitively (via library
   dependencies). Fix: look up the correct module
   slot by matching `sofilename` rather than
   relying on `data.moduleindex`.

3. **Duplicate commands**: two modules register
   the same `cmdkey`. Check with:
   ```bash
   echo "m?" | milk-cli 2>&1 | sort | uniq -d
   ```

## CLI Pipeline Architecture

A line at the milk-cli prompt is bash. milk-cli translates its `@{…}` and runs it in a bash
co-process that sourced `milk.bash`, so from there it runs as in plain bash:

```
CLIcore_UI_execute.c → milk_bash_exec.c (translation) → milk_bash_coproc.c (bash co-process)
  @keyword    → libmilkbash builtin or milk.bash function (milk_lexicon.c says which)
  module.cmd  → shell function → milkcall → milk_client_ops.c → session server:
                milk_server_dispatch.c → CLIcore_checkargs.c (arguments: milk_args_bind)
  @calc text  → session server: milk_server_eval.c → src/calc/mc_eval.c
```

### Debugging each stage

| Stage           | How to trace                                                        |
| --------------- | ------------------------------------------------------------------- |
| Translation     | the same line in plain bash after `source milk.bash`                |
| Keyword         | `type -t @get` (builtin or function), `@get --help`                 |
| Module command  | `<cmd> --no-run ...` prints the resolved call; usage errors exit 2 |
| Server request  | `@log -n 5` shows the session's last requests, status and timing   |

## Display / Prompt Issues

### Prompt corruption

The `milk-cli` prompt is set internally. If the
user's `PS1` environment variable leaks in, the
prompt becomes corrupted. Check:

- `CLIcore_UI_prompt.c` — prompt construction
- `unsetenv("PS1")` should happen early in init

### Readline echo issues

If characters are not echoed during input:

- Check `rl_redisplay()` calls in completion and
  hint callbacks
- Check for `tcsetattr()` calls that might alter
  terminal settings
- Verify `rl_bind_key()` overrides are correct

## Shared Memory Cleanup

Stale SHM files from crashed processes can cause unexpected behavior: list and remove them as
[debugging.md §4.2](../../../docs/operations/debugging.md#42-shared-memory-leaks) says.

## Source File Reference

Key files for CLI debugging:

| File                                 | Role                                        |
| ------------------------------------ | ------------------------------------------- |
| `src/cli/CLIcore/CLIcore.c`                          | Main entry, module loading                  |
| `libmilkscript/core/CLIcore_modules.c`               | `load_module_shared()`, module registration |
| `libmilkscript/core/CLIcore_UI_execute.c`            | Prompt line execution                       |
| `libmilkscript/bash/milk_bash_exec.c`                | `@{…}` translation, bash co-process         |
| `libmilkscript/server/milk_server_dispatch.c`        | Command dispatch on the session server      |
| `libmilkscript/core/CLIcore_checkargs.c`             | Argument binding and validation             |
| `src/calc/mc_eval.c`                 | Arithmetic expression entry point           |
| `src/calc/mc_lex.c`                  | Expression tokenizer                        |
| `src/calc/mc_parse.c`                | Pratt parser                                |
| `src/calc/mc_scalar.c`               | Built-in scalar math & string functions     |
