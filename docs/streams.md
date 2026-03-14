# Shared Memory Streams (`ImageStreamIO`)

`milk` is built around a low-latency, zero-copy architecture
designed for high-performance pipelines. This core feature is
powered by `ImageStreamIO` which allocates streams
(n-dimensional tensors, typically images or data cubes)
directly in the Linux tmpfs (`/dev/shm/`).

See also: [FPS](fps.md) ·
[Process Info](procinfo.md) ·
[CLI Reference](cli/CLIcore.md) ·
[Scripts Reference](scripts.md)

## 1. Core Concepts

Unlike file-system-based intermediate data passing,
`ImageStreamIO` provides direct memory pointers to running
compute units. Processes can read from or write to the same
stream with microsecond latencies.

```mermaid
sequenceDiagram
    participant Writer as Compute Unit A (Writer)
    participant SHM as /dev/shm (ImageStreamIO)
    participant Reader as Compute Unit B (Reader)

    Note over Writer, Reader: Zero-copy shared memory architecture
    Writer->>SHM: imgid_mkimage("stream1")
    Reader->>SHM: imgid_connect("stream1")
    Reader->>Reader: Wait for Semaphore
    
    loop Frame processing
        Writer->>Writer: Compute new frame
        Writer->>SHM: Write direct memory pointer
        Writer->>SHM: Post Semaphore
        SHM-->>Reader: Wakeup Signal
        Reader->>SHM: Read direct memory pointer
    end
```

## 2. Metadata and Semaphores

Every stream contains more than just pixel values. It
includes a comprehensive metadata header:

1. **Dimensionality:** Size and shape axes (1D arrays to
   3D cubes).
2. **Data Types:** Support for signed/unsigned integers
   and floating point up to 64-bit precision.
3. **Keywords:** An embedded dictionary of FITS-style
   keywords to propagate state (e.g., exposure parameters
   or telemetry data).
4. **Semaphores:** POSIX semaphores are bound natively to
   streams. When a compute unit finishes writing a frame,
   it posts a semaphore. Downstream processes blocking on
   that stream immediately wake up, ensuring synchronized
   cascading pipelines.

## 3. Stream Modifiers

When interacting with streams on the CLI or within `milk`
algorithms, standard modifiers are supported directly
inside the stream string. E.g. passing `myImage@L:` to a
module.

- `@S:` (Shared): Expected to reside in `/dev/shm/`.
  This is the default if no modifier is given.
- `@L:` (Local): Allocated in private local process
  memory. Used for internal buffering that doesn't need
  to be visible externally.
- `@F:` (File / FITS): Bypasses shared memory to directly
  read a physical file on disk.

*When writing modules using `fpsexec` patterns, passing
a non-existent or disallowed modifier automatically alerts
the user and aborts the module spin-up to prevent silent
failures.*

## 4. Introspection

Tools like `milk-streamCTRL` provide real-time
introspection into active streams, displaying frame
arrival rates, recent values, and the current state of
semaphores without disrupting operations.

## 5. C API (`IMGID`)

`milk` uses the `IMGID` structure to hold references to
images and streams. This structure is one level above
`ImageStreamIO`, and is local to the `milk` process. It is
the preferred way to pass images and streams as function
arguments.

### 5.1. Creating an `IMGID`

Creating a blank `IMGID` (this does not allocate memory yet):

```c
static inline IMGID imgid_make()
```

Creating an `IMGID` with a name:

```c
static inline IMGID imgid_make_from_name(CONST_WORD name)
```

*(Special characters like `s>tf32>im1` can also be used to automatically set type and location properties).*

### 5.2. Connecting to a stream

If you expect the stream to already exist, you can connect to it:

```c
IMGID img1 = imgid_make();
// imgid_connect returns quickly. Check if img1.ID != -1
imgid_connect("streamname1", &img1, 0);
if(img1.ID == -1) {
	// handle failure securely
}
```

### 5.3. Creating an image in shared memory

To create a new stream:

```c
IMGID img = imgid_make_from_name("im1");
img.naxis = 2;
img.size[0] = 128;
img.size[1] = 128;
img.shared = 1; // 1 for shared memory stream, 0 for local memory

// Allocate memory and initialize stream headers
imgid_mkimage(&img);

// When done parsing or exiting
imgid_free(&img);
```

### 5.4. Creating or connecting with format checks

Often, you want to ensure the stream has specific dimensions and types before using it, or create it if it doesn't match:

```c
IMGID img1 = imgid_make();
img1.naxis = 2;
img1.size[0] = 128;
img1.size[1] = 128;

// IMGID_CONNECT_CHECK_CREATE will create it if the format is wrong or it doesn't exist
// IMGID_CONNECT_CHECK_FAIL will fail the connection if the format is wrong
imgid_connect("streamname1", &img1, IMGID_CONNECT_CHECK_CREATE);
```

For convenience, specialized typed functions are available:

```c
// Force connection/creation of a 2D float32 array
imgid_connect_create_2Df32("streamname1", &img1, xsize, ysize);
```

> [!TIP]
> Functions should prefer passing parameters using `IMGID` pointers and accessing pixels through `img.im->array.F` or similar data type unions based on `img.im->md[0].datatype`.

---
← [Documentation Index](index.md)
