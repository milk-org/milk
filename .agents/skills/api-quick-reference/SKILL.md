---
name: api-quick-reference
description: Deep reference for milk APIs including IMGID, datatype dispatch, processinfo macros, magic context variables, FPS parameter types, and stream write protocols.
---

# milk API Quick Reference

This skill provides a condensed "cheat sheet" of critical APIs used in milk compute units.

## 1. IMGID API Reference
`IMGID` is the structure used to represent shared memory image streams.
Always use `mdt->` (metadata template) when configuring a new stream, never direct struct fields.

```c
IMGID img = imgid_make_from_name("mystream");

// Configure metadata BEFORE create/connect
img.mdt->naxis = 2;
img.mdt->size[0] = 128;
img.mdt->size[1] = 128;
img.mdt->datatype = _DATATYPE_FLOAT;
img.mdt->shared = 1;

// To create a new stream:
imcreateIMGID(&img);

// OR to resolve (connect to) an existing stream:
resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

// OR to connect with flags (creates if doesn't exist):
imgid_connect(&img, IMGID_CONNECT_CHECK_CREATE);

// Accessing pixel data (always via im->array):
float *ptr = img.im->array.F;

// Free the struct at the end
imgid_free(&img);
```

## 2. Datatype Dispatch Table
When a function supports multiple datatypes, use an `else if` chain matching `_DATATYPE_*` to the correct union member of `im->array`.

| Type Constant | Union Member | C Type | Size Macro |
|---|---|---|---|
| `_DATATYPE_UINT8` | `.UI8` | `uint8_t` | `SIZEOF_DATATYPE_UINT8` |
| `_DATATYPE_INT8` | `.I8` | `int8_t` | `SIZEOF_DATATYPE_INT8` |
| `_DATATYPE_UINT16` | `.UI16` | `uint16_t` | `SIZEOF_DATATYPE_UINT16` |
| `_DATATYPE_INT16` | `.I16` | `int16_t` | `SIZEOF_DATATYPE_INT16` |
| `_DATATYPE_UINT32` | `.UI32` | `uint32_t` | `SIZEOF_DATATYPE_UINT32` |
| `_DATATYPE_INT32` | `.I32` | `int32_t` | `SIZEOF_DATATYPE_INT32` |
| `_DATATYPE_UINT64` | `.UI64` | `uint64_t` | `SIZEOF_DATATYPE_UINT64` |
| `_DATATYPE_INT64` | `.I64` | `int64_t` | `SIZEOF_DATATYPE_INT64` |
| `_DATATYPE_FLOAT` | `.F` | `float` | `SIZEOF_DATATYPE_FLOAT` |
| `_DATATYPE_DOUBLE` | `.D` | `double` | `SIZEOF_DATATYPE_DOUBLE` |
| `_DATATYPE_COMPLEX_FLOAT` | `.CF` | `complex float`| `SIZEOF_DATATYPE_COMPLEX_FLOAT` |
| `_DATATYPE_COMPLEX_DOUBLE` | `.CD` | `complex double`| `SIZEOF_DATATYPE_COMPLEX_DOUBLE` |

For untyped generic copies, use `.raw` with `__builtin_memcpy`.

## 3. processinfo Macro Reference
These macros wrap your compute loop to provide timing, semaphore triggering, and tmux status.

- `INSERT_STD_PROCINFO_COMPUTEFUNC_INIT`: Initializes the `processinfo` structure based on CLI options. Must be called before the loop.
- `INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART`: Begins the `while(processloopOK)` loop and handles semaphore waiting.
- `INSERT_STD_PROCINFO_COMPUTEFUNC_START`: Convenience macro combining `INIT` and `LOOPSTART`.
- `INSERT_STD_PROCINFO_COMPUTEFUNC_END`: Closes the loop and cleans up.

A standard continuous-loop compute unit looks like:
```c
INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

// (Optional) custom setup here

INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
{
    // Do computation here...

    // (Optional) If you wrote to outimg based on inimg's trigger:
    processinfo_update_output_stream(processinfo, outimg.im, inimg.im);
}
INSERT_STD_PROCINFO_COMPUTEFUNC_END
```

## 4. Magic Context Variables
These global variables are defined by the framework and are always available in compute functions.

| Variable | Type | Defined By | Description |
|---|---|---|---|
| `dcfpsptr` | `FPS*` | `milkdata_macros.h` | Pointer to the current FPS instance. |
| `dcfpsname` | `char[]` | `milkdata_macros.h` | Name of the current FPS instance. |
| `dcimg` | `IMAGE*` | `milkdata_macros.h` | The global image array (used in `resolveIMGID`). |
| `dcnimg` | `long` | `milkdata_macros.h` | Size of the `dcimg` array. |
| `processinfo`| `PROCESSINFO*`| `fps_procinfo_macros.h` | Struct tracking process timing, loop limits, and triggers. Injected by `INSERT_STD_PROCINFO_COMPUTEFUNC_INIT`. |
| `processloopOK`| `int` | `fps_procinfo_macros.h` | Loop condition variable. Injected by `INIT`, checked in `LOOPSTART`. |

## 5. Stream Write Protocol
When writing data to a shared memory stream (`IMAGE`), you must use the semaphore protocol so readers know when the data is ready.

```c
// 1. Acquire write lock (increments md->cnt0)
SHMIM_WRITE_ACQUIRE(outimg.im->md);

// 2. Modify data
for(int i = 0; i < N; i++) outimg.im->array.F[i] = ...;

// 3. Release write lock (increments md->cnt1)
SHMIM_WRITE_RELEASE(outimg.im->md);

// 4. Update timing and post semaphores
// If inside a processinfo loop triggered by an input stream:
processinfo_update_output_stream(processinfo, outimg.im, inimg.im);

// Or if not using processinfo triggers (e.g. one-shot or generator):
ImageStreamIO_UpdateIm(outimg.im);
```

## 6. FPS Parameter Quick-Reference
Common `FPTYPE_*` mapped to C types:
- `FPTYPE_INT32` / `FPTYPE_UINT32` → `int32_t` / `uint32_t`
- `FPTYPE_INT64` / `FPTYPE_UINT64` → `int64_t` / `uint64_t`
- `FPTYPE_FLOAT32` / `FPTYPE_FLOAT64` → `float` / `double`
- `FPTYPE_ONOFF` → `int32_t` (0 = OFF, nonzero = ON)
- `FPTYPE_STREAMNAME`, `FPTYPE_FILENAME`, `FPTYPE_STRING` → `char[]` buffer
- `FPTYPE_TIMESPEC` → `struct timespec`
- `FPTYPE_PID` → `pid_t`

X-Macro Binding Column Order (Strict):
`X(keyword, pointer, type, is_primary, flags, description)`
