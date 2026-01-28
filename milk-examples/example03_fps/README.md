# Example 03: ImageStreamIO + ProcessInfo + FPS

This example demonstrates how to integrate `ImageStreamIO` (streaming data), `libprocessinfo` (process monitoring), and `libfps` (function parameter structure) into a single application. It provides both a standalone executable and a shared object module loadable into the `milk` Command Line Interface (CLI).

## Overview

The `processor03` application reads an input stream (default: `stream03`), processes a region of interest (ROI), and writes the result to an output stream (default: `stream03_proc`). It supports dynamic parameter updates via the Function Parameter Structure (FPS).

Key Features:
- **FPS Integration:** Parameters like input/output names, ROI size, and offsets are managed via shared memory.
- **Process Monitoring:** Uses `libprocessinfo` to register the process, handle signals, and monitor performance.
- **Dual Build:** Can be run as a standalone binary or loaded as a module in `milk`.

## Compilation

```bash
mkdir -p build
cd build
cmake ..
make
```

This will produce:
- `milk-example-03-writer`: A helper program to generate a test input stream.
- `milk-example-03-processor`: The standalone processor executable.
- `processor03.so`: The shared object module for the milk CLI.

---

## Standalone Usage

### 1. Start the Data Source
First, run the writer to create and populate the input stream (`stream03`):
```bash
./milk-example-03-writer
```

### 2. Initialize FPS
Initialize the FPS shared memory structure. You can optionally set keywords (`-k`) and a description (`-d`).
```bash
./milk-example-03-processor fpsinit -k "test,example" -d "Example 03 Processor"
```

### 3. Start Processing
Start the processing loop. The `-tmux` option is recommended to launch separate tmux windows for the run loop and configuration loop.
```bash
./milk-example-03-processor runstart -tmux
```
Alternatively, run directly in the current terminal (blocking):
```bash
./milk-example-03-processor runstart
```

### 4. Start Configuration Loop
The configuration loop validates parameters in the background (e.g., ensuring the ROI fits within the image).
```bash
./milk-example-03-processor confstart -tmux
```

### 5. Control Parameters
You can control the processor using `milk-fpsCTRL` (TUI) or by modifying the FPS entries directly.
```bash
milk-fpsCTRL -n processor03
```

### 6. Stop Processes
To stop the loops:
```bash
./milk-example-03-processor runstop
./milk-example-03-processor confstop
```

### Custom Options
- Specify a custom FPS name: `-n myproc`
- Specify initial parameters (during `fpsinit`): Not supported via CLI args in standalone (uses FPS defaults), but can be modified via FPS after init.

---

## Milk CLI Usage (Shared Object)

You can load the compiled module into the `milk` CLI to run the processor as a native command.

### 1. Start Milk
Run the milk shell:
```bash
milk
```

### 2. Load the Module
Load the compiled shared object. Adjust the path if necessary.
```bash
milk > mload ./build/processor03.so
```

### 3. Run the Command
The module registers the `processor03` command.
```bash
milk > processor03
```
This will start the FPS configuration and run loops managed by the CLI's internal FPS framework.

### 4. Access Parameters
In the milk CLI, you can pass arguments directly:
```bash
milk > processor03 .in_name "stream03" .roi_size 100
```
Or use the FPS interface within milk to modify parameters dynamically.

---

## Directory Structure

- `processor.c`: Source code for both standalone and module implementations.
- `writer.c`: Source for the test stream generator.
- `CMakeLists.txt`: Build configuration.
