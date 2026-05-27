# Example 02: ImageStreamIO + ProcessInfo

This example demonstrates how to use `ImageStreamIO` and `libprocessinfo` together to create a processor that can be monitored and controlled externally.

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
   ./milk-example-02-writer
   ```

2. Start the processor:
   ```bash
   ./milk-example-02-processor
   ```

The processor creates a ProcessInfo structure in shared memory. You can see it in `/milk/shm/proc.proc_ex02.*.shm`.
