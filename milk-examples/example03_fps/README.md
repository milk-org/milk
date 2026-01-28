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
- `milk-example-03-writer`: The standalone writer executable.
- `milk-example-03-processor`: The standalone processor executable.
- `writer03.so`: The shared object module for the milk CLI.
- `processor03.so`: The shared object module for the milk CLI.

---

## Standalone Usage

### 1. Initialize FPS
Initialize the FPS shared memory structures for both writer and processor.
```bash
./milk-example-03-writer    fpsinit -d "Example 03 Writer"
./milk-example-03-processor fpsinit -d "Example 03 Processor"
```

### 2. Start Processes
Start the run and configuration loops. Using `-tmux` is highly recommended.
```bash
./milk-example-03-writer    runstart -tmux
./milk-example-03-writer    confstart -tmux
./milk-example-03-processor runstart -tmux
./milk-example-03-processor confstart -tmux
```

### 3. Control Parameters
You can control both processes using `milk-fpsCTRL` (TUI).
```bash
milk-fpsCTRL
```
Use `Left/Right` arrows to navigate the hierarchical parameters and `h` for detailed help.

### 4. Stop Processes
```bash
./milk-example-03-writer    runstop
./milk-example-03-writer    confstop
./milk-example-03-processor runstop
./milk-example-03-processor confstop
```

---

## Milk CLI Usage (Shared Object)

You can load the compiled modules into the `milk` CLI to run them as native commands.

### 1. Load the Modules
```bash
milk > soload ./build/writer03.so
milk > soload ./build/processor03.so
```

### 2. Run the Commands
```bash
milk > writer03
milk > processor03
```
This will start the FPS loops within the CLI. You can also pass parameters:
```bash
milk > writer03 .width 100 .height 100
milk > processor03 .in_name "stream03" .roi_size 50
```

---

## Directory Structure

- `processor.c`: Source code for both standalone and module implementations.
- `writer.c`: Source for the test stream generator.
- `CMakeLists.txt`: Build configuration.
