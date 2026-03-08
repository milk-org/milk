# Shared Memory Streams (`ImageStreamIO`)

`milk` is built around a low-latency, zero-copy architecture designed for high-performance pipelines. This core feature is powered by `ImageStreamIO` which allocates streams (n-dimensional tensors, typically images or data cubes) directly in the Linux tmpfs (`/dev/shm/`).

## Core Concepts

Unlike file-system-based intermediate data passing, `ImageStreamIO` provides direct memory pointers to running compute units. Processes can read from or write to the same stream with microsecond latencies.

## Metadata and Semaphores

Every stream contains more than just pixel values. It includes a comprehensive metadata header:
1. **Dimensionality:** Size and shape axes (1D arrays to 3D cubes).
2. **Data Types:** Support for signed/unsigned integers and floating point architectures up to 64-bit precision.
3. **Keywords:** An embedded dictionary of FITS-style keywords to propagate state (e.g., exposure parameters or telemetry data).
4. **Semaphores:** Posix semaphores are bound natively to the streams. When a compute unit finishes writing its frame to a stream, it naturally posts a semaphore. Downstream processes that are blocking (waiting) on that stream immediately wake up, ensuring perfectly synchronized cascading pipelines.

## Stream Modifiers

When interacting with streams on the CLI or within `milk` algorithms, standard modifiers are supported directly inside the stream string. E.g. passing `myImage@L:` to a module.

- `@S:` (Shared): Expected to reside in `/dev/shm/`. (This is the default if no modifier is given).
- `@L:` (Local): Allocated completely in private local process memory. Generally used for internal buffering that doesn't need to be visible externally to the rest of the pipeline.
- `@F:` (File / FITS): Bypasses shared memory allocation to directly read a physical file on the disk layout.

*When writing modules using updated `fpsexec` patterns, passing a non-existent or disallowed modifier automatically alerts the user and securely aborts the module spin-up to prevent silent failures.*

## Introspection
Tools like `milk-streamCTRL` provide real-time introspection into active streams, displaying frame arrival rates, recent values, and the current state of semaphores without disrupting operations.
