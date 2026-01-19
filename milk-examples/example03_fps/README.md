# Example 03: ImageStreamIO + ProcessInfo + FPS

This example demonstrates how to use `ImageStreamIO`, `libprocessinfo`, and `libfps` together.

## Compilation

```bash
mkdir build
cd build
cmake ..
make
```

## Running

1. Start the writer:
   ```bash
   ./milk-example-03-writer
   ```

2. Initialize the FPS (one-time setup):
   ```bash
   ./milk-example-03-processor fpsinit
   ```

3. Start the processor (directly or via tmux):
   ```bash
   ./milk-example-03-processor run
   ```
   OR
   ```bash
   ./milk-example-03-processor run -tmux
   ```

The processor will use FPS for its configuration parameters and can be controlled via `libfps` or the `milk-fpsCTRL` script.