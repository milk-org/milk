# Programmer's Guide to `milk`

Welcome to `milk`. This document serves as an overview of its core architecture and programming model. If you are reading this while setting up a new module, debugging, or wanting to write a custom module, this guide will orient you on the core concepts.

## 1. Core Architecture
`milk` is structured around decoupled, high-performance computing components. Instead of monolithic structures, it relies on small modular units ("compute units") talking to each other via standard inter-process communication mechanisms.

The architecture orbits around two primary concepts:

1. **ImageStreamIO (Streams):**
   - The primary data layer. Shared memory images/data cubes are passed around with near-zero copy overhead. Stream metadata holds dimensions, data format, keywords, and synchronization semaphores that trigger downstream processes.

2. **Function Processing System (FPS):**
   - The control and parameter layer. FPS manages configuration parameters, state, and commands for compute units. FPS instances reside in shared memory (`/dev/shm/fps.*`), allowing for real-time adjustments via the CLI, GUI, or other automated processes without restarting the compute module itself.

## 2. Process Management

`milk` isolates its execution environments utilizing `tmux` and its own framework:

- **Isolated Execution:** When an FPS script is launched via a standalone program (e.g., `milk-fps-deploy` or via the `milk-fpsexec-<name>` executables), `milk` places these instances inside dedicated `tmux` sessions. This ensures that failures in one component do not drag down the whole system, while maintaining accessibility for debugging standard error/output.
- **Processinfo (`procinfo`):** Every FPS instance tracks its heartbeat, state (idle, computing, waiting), loops per second, and error conditions in the system. The `milk-procinfo-list` command depends on these heartbeat counters properly updating.

## 3. Writing a Compute Unit

When building a new compute task (a new module), `milk` enforces a standardized format, particularly the "V2" structure. Standard standalone units are found in the `src/` directory.

### The 8-Section Layout
Most compute modules built for `fpsexec` (the execution framework) follow this C-language structure:

1. **`FPS_APP_INFO`:** Registration of the FPS application metadata (name, command keyword, and a mandatory one-line description).
2. **Local parameters:** Definition of standard C variables.
3. **`FPS_PARAMS` (X-macro):** A mapping table that binds standard C variables (from section 2) to their corresponding shared memory parameters.
4. **Compute Function (`fpsexec()`):** The pure calculation core where data pointers are obtained and manipulated.
5. **`CLIcmddata`:** Standard scoping configuration for the `milk` CLI registry.
6. **Command parsing/registration:** Often empty for standard implementations, but available for custom commands.
7. **Module Initialization:** Used to load parameters if needed when compiled as a dynamic plugin.
8. **Standalone `main()`:** Utilizing standard macro `FPS_MAIN_STANDALONE_V2` which automatically initializes FPS, syncs data, starts `processinfo` heartbeats, parsing `-h1` and `-tmux` flags seamlessly.

## 4. Directory Map
Understanding where things exist guarantees you are looking in the right place:

- `src/ImageStreamIO`: The standalone core for shared-memory data (the stream).
- `src/libfps`: The core library enabling the Function Processing System.
- `src/milk_module_example`: Examples on how to structure a correct compute unit.
- `src/fpsCTRL`: The CLI/TUI interfaces (`milk-fpsCTRL`).
- `docs/`: System documentation.

---

*(This guide is automatically updated by your coding agent based on `.agents/rules/maintain-programmers-guide.md`)*
