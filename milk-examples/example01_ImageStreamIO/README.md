# Example 01: ImageStreamIO

This example demonstrates how to use `ImageStreamIO` standalone to create a stream writer and a processor.

## Compilation

```bash
mkdir build
cd build
cmake ..
make
```

## Running

1. Start the writer in one terminal:

   ```bash
   ./milk-example-01-writer
   ```

2. Start the processor in another terminal:
   ```bash
   ./milk-example-01-processor
   ```

The processor will wait for new frames from the writer, crop 4 ROIs, sum them, and post the result to `stream01_proc`.
