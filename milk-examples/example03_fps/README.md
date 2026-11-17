# Example 03: ImageStreamIO + ProcessInfo + FPS

This example demonstrates the complete integration of the Milk core library stack:

1.  **ImageStreamIO:** Low-latency shared memory data streaming.
2.  **ProcessInfo:** Real-time process monitoring, signal handling, and loop control.
3.  **FPS (Function Parameter Structure):** Hierarchical parameter management with dynamic updates via shared memory.

## Architectural Overview

The example consists of two primary components:

- **Writer:** Generates a dynamic 2D sine/cosine pattern and posts it to a shared memory stream.
- **Processor:** Reads the input stream, extracts a Region of Interest (ROI), and writes it to an output stream.

### Key Concepts

#### Hierarchical Dependency

The Milk ecosystem follows a strict hierarchy:

- `ImageStreamIO` provides the transport layer.
- `ProcessInfo` wraps a compute loop, providing heartbeat, timing, and control hooks.
- `FPS` adds a configuration layer on top of `ProcessInfo`, allowing parameters to be viewed and modified externally (via CLI or TUI) while the process is running.

#### Dual-Mode Implementation

The example is built in two modes:

1.  **Standalone Executable:** A self-contained binary that manages its own shared memory segments and loops.
2.  **CLI Module:** A shared object (`.so`) loadable into the `milk` shell. The module exports its `MILK_RUNNABLE` descriptors (`MILK_MODULE_RUNNABLES`), the CLI core registers them after `dlopen()` and handles the boilerplate, while the module provides the compute logic.

#### Shared Parameter Management (X-Macros)

To avoid duplicating parameter names, descriptions, and types between the FPS initialization and the CLI argument definitions, this example uses the **X-Macro** technique.
Parameters are defined once in a header file (e.g., `PROCESSOR_PARAMS` in `processor.h`) and then expanded into:

- Global pointer declarations.
- `function_parameter_add_entry` calls for FPS initialization.
- `CLICMDARGDEF` array entries for the Milk CLI.

---

## Directory Structure

- `processor.h / .c`: Core logic for the ROI extraction processor.
- `writer.h / .c`: Core logic for the pattern generator writer.
- `processor_module.c`: `MILK_RUNNABLE` descriptor of the processor CLI command.
- `writer_module.c`: `MILK_RUNNABLE` descriptor of the writer CLI command.
- `example03fps_module.c`: Module registry (`MILK_MODULE_RUNNABLES`) of the `example03fps` shared object.
- `CMakeLists.txt`: Multi-target build configuration.

---

## Compilation

```bash
mkdir -p build
cd build
cmake ..
make
```

This produces:

- `milk-example-03-writer`: Standalone writer binary.
- `milk-example-03-processor`: Standalone processor binary.
- `example03fps.so`: Combined Milk CLI module.

---

## Usage Patterns

### 1. Standalone Mode (High Performance)

Initialize the FPS shared memory segments:

```bash
./milk-example-03-writer    fpsinit -d "Writer Init"
./milk-example-03-processor fpsinit -d "Processor Init"
```

Start the loops in the background (each command returns once its loop runs; `milk-fps-log NAME`
shows a unit's output, `runstop` and `confstop` stop it):

```bash
./milk-example-03-writer    runstart
./milk-example-03-writer    confstart
./milk-example-03-processor runstart
./milk-example-03-processor confstart
```

### 2. Milk CLI Mode (Integrated Control)

Start the `milk` shell and load the combined module:

```bash
milk
milk > soload ./build/example03fps.so
```

Launch the commands (these run as managed threads within the CLI):

```bash
milk > ex03.writer03
milk > ex03.processor03
```

You can pass arguments directly at launch:

```bash
milk > ex03.processor03 .roi_size 100 .off_x 10
```

### 3. Monitoring and Control (TUI)

Use the `milk-fpsCTRL` tool to monitor both processes simultaneously:

```bash
milk-fpsCTRL
```

- **Navigation:** Use arrow keys to explore the hierarchical parameter tree.
- **Help:** Press **Left Arrow** at the root level to view the detailed multiline description for the selected process.
- **Modification:** Press **Enter** or **Space** on a parameter to modify its value in real-time.

---

## Detailed Logic Description

### Writer Computation

The writer uses `processinfo->loopcnt` as a temporal phase. For each pixel $(x, y)$, it calculates:
$$val = 0.5 \cdot \sin((x + cnt) \cdot freq_x) + 0.5 \cdot \cos((y + cnt) \cdot freq_y)$$
The result is written to the output stream, and `ImageStreamIO_UpdateIm` is called to increment the stream counter and post semaphores.

### Processor Computation

The processor waits on the input stream's semaphore. Once triggered, it copies a sub-region defined by `roi_size` and `off_x` into the output stream.
The **Configuration Loop** (`FPSCONF`) runs in parallel, validating that the requested ROI fits within the physical dimensions of the input stream, automatically clamping values if necessary.
