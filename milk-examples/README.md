# Milk Examples

This directory contains examples demonstrating the usage of the Milk libraries. The examples are structured to follow the dependency chain of the core libraries.

## Library Dependency Chain

The core libraries in Milk follow a strict hierarchical dependency chain. It is important to understand this structure when developing modules or applications.

1.  **ImageStreamIO** (Base Library)
    - **Description:** The foundational library for shared memory image streams. It handles low-level POSIX shared memory, semaphores, and data types.
    - **Dependencies:** `cfitsio`, `m` (Math), `CUDA` (Optional).
    - **Used by:** `libprocessinfo`, `libfps`.

2.  **libprocessinfo** (`milkprocessinfo`)
    - **Description:** Provides tools for process monitoring, control, and synchronization. It defines the `PROCESSINFO` structure and manages process lists in shared memory.
    - **Dependencies:** `ImageStreamIO` (It uses image streams for trigger mechanisms), `rt` (Real-time), `ncurses` (TUI), `pthread`.
    - **Used by:** `libfps`.

3.  **libfps** (`milkfps`)
    - **Description:** The Function Parameter Structure library. It provides a framework for managing function parameters, configuration, and runtime control via a standardized interface (CLI and TUI).
    - **Dependencies:** `libprocessinfo` (For process integration), `ImageStreamIO` (For stream parameters), `ncurses`.

## Example Progression

The examples in this directory mirror this hierarchy:

- **`example01_ImageStreamIO/`**: Demonstrates how to create, write to, and read from Image Streams using the base `ImageStreamIO` library.
- **`example02_processinfo/`**: Builds upon ImageStreamIO to show how to register a process, monitor its status, and handle process triggers.
- **`example03_fps/`**: The most advanced level, demonstrating how to wrap a function with `libfps` to gain automatic parameter management, configuration/run loops, and CLI integration.
