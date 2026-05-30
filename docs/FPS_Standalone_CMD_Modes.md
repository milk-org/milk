# FPS Standalone and CMD Modes

## Overview

The Function Parameter Structure (FPS) in the `milk` package
provides a unified way to manage parameters for processes
and CLI tools. Developers expose FPS-driven functionalities
using two primary modes:

1. **CMD/CLI Mode:** Registered commands inside the
   `milk-cli` environment.
2. **Standalone Mode:** Self-contained executables invoked
   directly from the terminal (`milk-fpsexec-*`).

These modes unify function parameter management so developers
write core logic once, bind arguments to FPS parameters,
and expose them universally.

See also: [FPS](fps.md) ·
[CLI Reference](cli/CLIcore.md) ·
[Programmer's Guide](programmers_guide.md) ·
[Developer Tutorial](developer/tutorial.md)

---

## 1. CMD / CLI Mode (Integrated)

### Implementation

CMD mode allows a function to be executed through the
interactive `milk-cli` shell. It is implemented by
registering a custom wrapper function with the CLI
framework using `RegisterCLIcmd()`.

In this mode, arguments are captured via the CLI parser
(stored in `data.cmdargtoken`) rather than `argv`.

### Key Characteristics

- **Local Memory FPS:** CLI implementations maintain an
  entirely local `FPS` (with
  `.SMfd = -1`) preventing the need for shared memory
  allocations if the command executes synchronously in a
  single shot.
- **Data Binding:** A binding structure (e.g.,
  `FPS_CLI_BINDING`) maps native C variables to their
  respective FPS representations.
- **Argument Processing:** When the CLI command executes,
  `FPS_process_CLI_and_sync()` extracts the typed arguments
  from the CLI tokens, updates the internal FPS values,
  and synchronizes them to the statically or dynamically
  allocated C variables.

**Example Implementation (`examplefunc_fps_cli_poc.c`):**

```c
static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static errno_t CLIfunction(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}
```

---

## 2. Standalone Mode

### Implementation

Standalone mode allows executing an FPS-driven module
directly as an independent binary (e.g.,
`milk-fpsexec-mymodule`). It handles native OS arguments
(`argc`, `argv`) while hooking into the underlying FPS
metadata architecture.

This mode uses the `FPS_MAIN_STANDALONE_V2` macro provided
in `fps.h`.

### Command Line Parsing

Standalone executables support two layers of argument
parsing:

<details markdown="1">
<summary><b>A. Standard FPS Process Control Commands</b></summary>

Before mapping to business logic, the executable looks
for built-in flags and process control commands. This
establishes the lifecycle of the shared memory FPS
segment.

- **Options** (placed before the command):
  - `-h`, `--help`: View detailed parameter bindings
    and command help.
  - `-h1`, `--help-oneline`: Print one-line description
    and exit.
  - `-h2`, `--help-description`: Print verbose description
    only (no color) and exit.
  - `-hm`, `--help-mono`: Full help, forced monochrome.
  - `-tmux`: Automatically create a `tmux` session and
    dispatch commands isolated from the main terminal.
  - `-n`, `--name <fpsname>`: Override the default FPS
    shared memory name.
  - `-procinfo`: Enable processinfo entries at fpsinit.
  - `-loops`: Infinite loop with stream semaphore
    trigger.
  - `-loopd <sec>`: Infinite loop with delay trigger
    (seconds).
  - `-k`, `--keywords`: Pass keywords for `fpsinit`
    (parsed but currently reserved — not actively used).
  - `-d`, `--description`: Pass description for
    `fpsinit` (parsed but currently reserved — not
    actively used).
- **Commands** (placed last on the command line):
  - `fpsinit`: Create the FPS shared memory segment.
  - `confstart`, `confstep`, `confstop`: Manage the
    configuration loop.
  - `runstart`, `runstop`: Manage the execution loop.
  - `run`: Alias for `runstart`.
  - `set [args]`: Set positional arguments in FPS
    (use `.` to skip a position).
  - `fps`: Print content of the FPS.
  - `fpslist`: List all FPS instances matching this
    executable.
  - `exec [args]`: Auto-init + set args + run
    (one-shot).

</details>

<details markdown="1">
<summary><b>B. Direct Execution (Positional Arguments)</b></summary>

If the user passes positional arguments that do not
match the built-in control commands (e.g.,
`./milk-fpsexec-mymod 2.5 100`), the standalone
operates as a streamlined one-shot execution tool:

1. **FPS Handshake:** Attempts to connect to the shared
   memory FPS (if running). If not found, provisions a
   temporary local FPS memory space inline.
2. **Positional Parsing:** Maps positional `argv`
   elements against defined `FPS_CLI_BINDING`
   configurations, converting types as needed.
3. **Synchronization & Execution:** Bound C pointers are
   updated synchronously, and the core logic executes
   exactly as it would inside the `milk-cli` environment.

</details>

### Using `FPS_MAIN_STANDALONE_V2`

For all applications, use the V2 macro:

```c
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function
)
```

This macro automatically orchestrates standard commands
(`fpsinit`, `confstart`, etc.), integrates `-tmux`
dispatch, and registers with `processinfo` for
heartbeat monitoring.

If you have a custom configuration check function, use:

```c
FPS_MAIN_STANDALONE_V2_CONFCHECK(
    FPS_app_info,
    FPS_PARAMS,
    compute_function,
    customCONFcheck
)
```

---

← [Documentation Index](index.md)
