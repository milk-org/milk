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
   ./writer
   ```

2. Initialize the FPS (one-time setup):
   ```bash
   ./processor conf
   ```
   (Wait until it exits or stop it once it created the shared memory)

3. Start the processor:
   ```bash
   ./processor run
   ```

The processor will use FPS for its configuration parameters and can be controlled via `libfps` or the `milk-fpsCTRL` script.
