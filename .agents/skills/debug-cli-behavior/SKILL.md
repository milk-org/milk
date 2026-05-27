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
source ~/src/milk/local/bin/milk-setup.bash
echo "command args" | milk-cli 2>&1
echo "Exit code: $?"
```

### Multi-line test

```bash
source ~/src/milk/local/bin/milk-setup.bash
milk-cli <<'EOF' 2>&1
command1
command2
EOF
echo "Exit code: $?"
```

### Capture both stdout and stderr separately

```bash
source ~/src/milk/local/bin/milk-setup.bash
echo "command" | milk-cli >out.txt 2>err.txt
echo "Exit: $?"
cat out.txt
cat err.txt
```

### Test with the robustness suite

```bash
cd ~/src/milk
bash tests/cli/run_cli_robustness_tests.sh \
  --verbose 2>&1
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
| `RegisterCLIcmd`               | NULL function pointer, uninitialized `CLIcmddata`    |
| `cli_calc_eval.c`              | Stack underflow in expression evaluator              |
| `cli_calc_tokenizer.c`         | Buffer overflow on long expressions                  |
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
       → dlsym("initModule")
       → initModule()
         → CLIADDCMD_module__func()
           → RegisterCLIcmd()
             → populates data.cmd[i]
```

### Key data structures

- `data.cmd[]` — array of all registered commands
  (type `CMD`)
- `data.module[]` — array of loaded modules
  (type `MODULE`)
- `data.moduleindex` — **global** index of the
  module currently being loaded (source of race
  conditions)
- `CLIcmddata` — per-function static struct
  containing the command's CLI keyword, args, etc.

### Common issues

1. **Empty commands**: `CLIcmddata` initialized
   by `__attribute__((constructor))` before
   `initModule` runs. If the constructor runs in
   a different load order than expected, the
   static `CLIcmddata` may not be populated when
   `RegisterCLIcmd` reads it.

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

When `milk-cli` receives input, it flows through:

```
readline → intercept check
  → variable expansion (CLIcore_script_expand.c)
  → special-form check (if/while/for/function)
  → calc expression check (cli_calc_parser.c)
  → CLI command lookup (CLIcore_UI_execute.c)
  → shell bypass (system())
```

### Debugging each stage

| Stage              | How to trace                                         |
| ------------------ | ---------------------------------------------------- |
| Intercept          | Check `CLIcore_script_intercept.c` for early returns |
| Variable expansion | Print `line` before/after `expand_variables()`       |
| Calc expression    | Check `cli_try_calc()` return value                  |
| Command lookup     | Check `find_cmd()` return value                      |
| Shell bypass       | Check `[shell bypass]` message on stderr             |

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

Stale SHM files from crashed processes can cause
unexpected behavior:

```bash
# List all milk shared memory
ls /dev/shm/*.im.shm /dev/shm/fps.* \
  /dev/shm/proc.* 2>/dev/null

# Clean specific entries
rm /dev/shm/fps.<name>.shm
rm /dev/shm/<stream>.im.shm

# Kill orphaned tmux sessions
tmux ls 2>/dev/null
tmux kill-session -t <name>
```

## Source File Reference

Key files for CLI debugging:

| File                                 | Role                                        |
| ------------------------------------ | ------------------------------------------- |
| `CLIcore.c`                          | Main entry, module loading                  |
| `CLIcore/CLIcore_modules.c`          | `load_module_shared()`, module registration |
| `CLIcore/CLIcore_UI_execute.c`       | Command dispatch                            |
| `CLIcore/CLIcore_script.c`           | Script interpreter main loop                |
| `CLIcore/CLIcore_script_intercept.c` | Early interception (comments, blank lines)  |
| `CLIcore/CLIcore_script_var.c`       | Variable assignment and lookup              |
| `CLIcore/CLIcore_script_expand.c`    | Variable and command substitution           |
| `CLIcore/CLIcore_checkargs.c`        | Argument validation                         |
| `cli_calc_parser.c`                  | Arithmetic expression entry point           |
| `cli_calc_tokenizer.c`               | Expression tokenizer                        |
| `cli_calc_eval.c`                    | Expression evaluator (stack machine)        |
| `cli_calc_functions.c`               | Built-in math functions                     |
