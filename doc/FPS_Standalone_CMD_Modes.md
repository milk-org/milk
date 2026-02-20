# FPS Standalone and CMD Modes

## Overview
The Function Parameter Structure (FPS) in the `milk` package provides a unified way to manage parameters for processes and CLI tools. 
Recent enhancements allow developers to seamlessly expose FPS-driven functionalities using two primary modes:
1. **CMD/CLI Mode:** Registered commands inside the `milk` environment.
2. **Standalone Mode:** Self-contained executables invoked directly from the terminal.

These modes aim to unify the function parameter management such that developers write the core logic once, abstract the arguments into bindings, and expose them universally.

---

## 1. CMD / CLI Mode (Integrated)

### Implementation
CMD mode allows a function to be executed through the interactive `milk` shell. It is implemented by registering a custom wrapper function with the CLI framework using `RegisterCLIcmd()`.

In this mode, arguments are captured via the CLI parser (stored in `data.cmdargtoken`) rather than `argv`.

### Key Characteristics
- **Local Memory FPS:** Usually, CLI implementations maintain an entirely local `FUNCTION_PARAMETER_STRUCT` (with `.SMfd = -1`) preventing the need for shared memory allocations if the command executes synchronously in a single shot.
- **Data Binding:** A binding structure (e.g., `FPS_CLI_BINDING`) maps native C variables to their respective FPS representations.
- **Argument Processing:** When the CLI command executes, `FPS_process_CLI_and_sync()` extracts the typed arguments from the CLI tokens, updates the internal FPS values, and synchronizes them to the statically or dynamically allocated C variables.

**Example Implementation (`examplefunc_fps_cli_poc.c`):**
```c
static FPS_CLI_BINDING my_bindings[] = {
    { .fpskeyword = "gain", .ptr = &param_gain, .type = FPTYPE_FLOAT64, .cli_index = 0 },
    // ...
};

static errno_t example_fps_cli_wrapper(void) {
    FUNCTION_PARAMETER_STRUCT fps = {0};
    // ... Initialize local FPS ...
    
    // Process CLI arguments mapped by CLIcore
    FPS_process_CLI_and_sync(&fps, my_bindings, 2);
    
    // Execute logic
    example_fps_computation();
    return RETURN_SUCCESS;
}
```

---

## 2. Standalone Mode

### Implementation
Standalone mode allows executing an FPS-driven module directly as an independent binary (e.g., `./fpsclitest`). It handles native operating system arguments (`argc`, `argv`) while still hooking into the underlying FPS metadata architecture.

This mode relies on either:
- The standard `FPS_MAIN_STANDALONE` macro provided in `fps.h`.
- A fully bespoke `main()` overriding standard behavior.

### Command Line Parsing
The standalone executables inherently support two layers of argument parsing:

#### A. Standard FPS Process Control Commands
Before mapping to business logic, the executable looks for built-in flags and process control commands. This establishes the lifecycle of the shared memory FPS segment.

- **Options:**
  - `-h`, `--help`: View detailed parameter bindings and command help.
  - `-tmux`: Automatically create a `tmux` session and dispatch commands isolated from the main terminal.
  - `-n`, `--name <fpsname>`: Override the default FPS shared memory name.
  - `-k`, `-d`: Pass keywords and descriptions for `fpsinit`.
- **Commands:**
  - `fpsinit`: Create the FPS shared memory segment.
  - `confstart`, `confstep`, `confstop`: Manage the configuration loop.
  - `runstart`, `runstop`: Manage the execution loop.

#### B. Direct Execution Fallback (Positional Arguments)
If the user passes positional arguments that do not match the built-in control commands (e.g., `./fpsclitest 2.5 100`), the standalone implementation operates as a streamlined one-shot execution tool:

1. **FPS Handshake:** It attempts to connect to the shared memory FPS (if running). If not found, it provisions a temporary local FPS memory space inline.
2. **Positional Parsing:** It maps positional `argv` elements strictly against defined `FPS_CLI_BINDING` configurations. String arguments are converted (e.g., `atof`, `atol`) to their matching numerical types.
3. **Synchronization & Execution:** Bound C pointers are updated synchronically, and the core implementation logic functions exactly as it would inside the `milk` CLI environment.

### Leveraging the `FPS_MAIN_STANDALONE` Macro
For most applications, the `FPS_MAIN_STANDALONE(DEFAULT_FPS_NAME, FUNC_PREFIX, HELPTEXT, PARAMS_MACRO)` boilerplate simplifies standalone integrations. It automatically orchestrates standard commands (`fpsinit`, `confstart`, etc.) and seamlessly integrates `-tmux` dispatch.
