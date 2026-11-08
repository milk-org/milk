---
name: imagestream-internals
description: Deep reference for ImageStreamIO shared
  memory streams, semaphore protocol, and data layout
---

# ImageStreamIO Internals

This skill provides deep context on the
`ImageStreamIO` shared memory layer that underlies
all stream data passing in milk. Essential for
writing correct stream readers/writers and
debugging synchronization issues.

## When to Use

- Writing a new stream processing loop
- Debugging missed frames or busy-waits
- Understanding the SHM file layout
- Troubleshooting semaphore issues
- Working with circular buffer (cube) mode

## SHM File Layout

Each stream creates a file at
`$MILK_SHM_DIR/<name>.im.shm` (default
`/milk/shm`) containing:

```
┌─────────────────────────────┐
│ IMAGE_METADATA (md)         │
│   - name, naxis, size[]     │
│   - datatype, shared flag   │
│   - write counter (cnt0)    │
│   - timing metadata         │
│   - semaphore config        │
│   - NBproctrace count       │
├─────────────────────────────┤
│ Pixel Data (array)          │
│   - Typed union:            │
│     .UI8, .SI8, .UI16, etc. │
│     .F (float), .D (double) │
│     .raw (void*)            │
├─────────────────────────────┤
│ Keywords (kw[])             │
│   - Named metadata values   │
├─────────────────────────────┤
│ Semaphore File Data         │
│   - sem_t per semaphore     │
├─────────────────────────────┤
│ semReadPID[]                │
│ semWritePID[]               │
│ semctrl[], semstatus[]      │
├─────────────────────────────┤
│ streamproctrace[]           │
│   - STREAM_PROC_TRACE ×     │
│     NBproctrace (default 10)│
│   - Process ancestry chain  │
├─────────────────────────────┤
│ (optional) Circular Buffer  │
│   atimearray, writetimearray│
│   cntarray, flagarray       │
└─────────────────────────────┘
```

## IMAGE Struct

The `IMAGE` struct is the C handle for a stream:

```c
typedef struct {
    IMAGE_METADATA *md;    // metadata
    union {
        uint8_t  *UI8;
        int8_t   *SI8;
        uint16_t *UI16;
        int16_t  *SI16;
        uint32_t *UI32;
        int32_t  *SI32;
        uint64_t *UI64;
        int64_t  *SI64;
        float    *F;
        double   *D;
        void     *raw;
    } array;
    sem_t **semptr;                   // semaphore array
    IMAGE_KEYWORD *kw;                // keywords
    STREAM_PROC_TRACE *streamproctrace; // process ancestry
} IMAGE;
```

Access pixel data through the typed union:

```c
float *data = img->array.F;
data[pixel_index] = value;
```

## IMGID vs IMAGE

| Struct  | Purpose                                                                   |
| ------- | ------------------------------------------------------------------------- |
| `IMGID` | High-level handle with name, size, modifiers. Use in function interfaces. |
| `IMAGE` | Low-level SHM-mapped struct. Access via `imgid.im`.                       |

```c
IMGID imgid = imgid_make_from_name("mystream");
resolveIMGID(&imgid, ERRMODE_WARN, dcimg, dcnimg);

IMAGE *img = imgid.im;         // low-level
float *data = img->array.F;    // pixel access
```

## Semaphore Protocol

### Overview

Streams use POSIX named semaphores for
reader-writer synchronization:

- **Writer** posts after updating pixel data
- **Readers** wait on their assigned semaphore
- Each reader gets a unique semaphore index
  (0 to `IMAGE_NB_SEMAPHORE - 1`, typically 10)

### Writer and Reader Sides

In a compute unit the writer sets `md->write = 1`, writes the pixels and posts with
`processinfo_update_output_stream()`; the reader waits with `processinfo_waitoninputstream()`:
[procinfo.md §7](../../../docs/guide/procinfo.md#7-compute-function-macros). Underneath:
`ImageStreamIO_UpdateIm()` (cnt0++, posts all readers), and `ImageStreamIO_semtimedwait()` on the
index from `ImageStreamIO_getsemwaitindex()`.

### Common Semaphore Issues

See also [debugging.md §4.1](../../../docs/operations/debugging.md#41-semaphore-deadlocks).

| Symptom                      | Cause                     | Fix                                                  |
| ---------------------------- | ------------------------- | ---------------------------------------------------- |
| Reader never wakes           | Writer not posting        | Post with `processinfo_update_output_stream()`       |
| Reader busy-spins            | Sem leaked (value > 0)    | Flush with `ImageStreamIO_semflush(img, -1)`         |
| Multiple readers miss frames | All on same semindex      | Each reader needs unique index                       |
| Frames arrive too fast       | Reader slower than writer | Check loop timing, skip frames                       |

## Circular Buffer (Cube) Mode

When `naxis = 3`, the stream acts as a circular
buffer:

```
Frame 0: array.F[0 .. size[0]*size[1]-1]
Frame 1: array.F[size[0]*size[1] .. 2*frame-1]
...
Frame N: array.F[N*framesize .. (N+1)*frame-1]
```

The current write position is tracked in
`md->cnt1` (modulo `size[2]`).

### Reading the Latest Frame

```c
long framesize = img->md->size[0]
               * img->md->size[1];
long slice = img->md->cnt1;
float *frame = img->array.F
             + slice * framesize;
```

## Stream Creation

### Using IMGID (Recommended)

`imgid_create_registered()`, `imcreateIMGID()` or `imgid_mkimage()` (only the first two put the
image in milk's image table): [streams.md §5](../../../docs/guide/streams.md#5-c-api-imgid).

### Using ImageStreamIO Directly

```c
uint32_t imsize[2] = {128, 128};
ImageStreamIO_createIm(
    &img, "mystream", 2, imsize,
    _DATATYPE_FLOAT, 1, 0, 0);
```

## Key Functions

| Function                          | Purpose                    |
| --------------------------------- | -------------------------- |
| `ImageStreamIO_createIm()`        | Create a new SHM stream    |
| `ImageStreamIO_openIm()`          | Connect to existing stream |
| `ImageStreamIO_closeIm()`         | Disconnect (don't destroy) |
| `ImageStreamIO_destroyIm()`       | Delete SHM file            |
| `ImageStreamIO_sempost()`         | Post semaphore(s)          |
| `ImageStreamIO_semwait()`         | Wait on semaphore          |
| `ImageStreamIO_getsemwaitindex()` | Get unique sem index       |
| `ImageStreamIO_semflush()`        | Flush semaphore(s)         |
| `ImageStreamIO_UpdateIm()`        | cnt0++, write=0, sem post  |

## Stream Process Trace (Ancestry)

The `STREAM_PROC_TRACE` struct, its propagation by `processinfo_update_output_stream()` (pass
the triggering input `IMAGE *`, or `NULL` for a generator) and the milk-CTRL TRACE page:
[streams.md §6](../../../docs/guide/streams.md#6-stream-process-trace-ancestry). Values of its
`triggermode` and `triggerstatus` fields (`processtools_trigger.h`):

### Trigger Modes

| Constant                            | Value | Description |
| ----------------------------------- | ----- | ----------- |
| `PROCESSINFO_TRIGGERMODE_IMMEDIATE` | 0     | No wait     |
| `PROCESSINFO_TRIGGERMODE_CNT0`      | 1     | Poll cnt0   |
| `PROCESSINFO_TRIGGERMODE_CNT1`      | 2     | Poll cnt1   |
| `PROCESSINFO_TRIGGERMODE_SEMAPHORE` | 3     | sem_wait    |
| `PROCESSINFO_TRIGGERMODE_DELAY`     | 4     | Fixed delay |
| `PROCESSINFO_TRIGGERMODE_CNT2`      | 6     | Poll cnt2   |

### Trigger Statuses

| Constant                             | Value | Meaning             |
| ------------------------------------ | ----- | ------------------- |
| `PROCESSINFO_TRIGGERSTATUS_WAITING`  | 1     | Waiting for trigger |
| `PROCESSINFO_TRIGGERSTATUS_RECEIVED` | 2     | Trigger received    |
| `PROCESSINFO_TRIGGERSTATUS_TIMEDOUT` | 3     | Trigger timed out   |

## Source Files

| File                                       | Role                    |
| ------------------------------------------ | ----------------------- |
| `src/engine/ImageStreamIO/ImageStreamIO.c` | Core SHM operations     |
| `src/engine/ImageStreamIO/ImageStreamIO.h` | Public API              |
| `src/engine/ImageStreamIO/ImageStruct.h`   | IMAGE struct definition |
| `src/engine/libfps/IMGID.h`                | IMGID struct, imgid_*   |

## Debugging Tips

Listing and inspecting streams (milk-streamCTRL's INFO, SEMS and TRACE pages) and removing stale
ones: [debugging.md §1.3](../../../docs/operations/debugging.md#13-milk-streamctrl),
[§4.1](../../../docs/operations/debugging.md#41-semaphore-deadlocks) and
[§4.2](../../../docs/operations/debugging.md#42-shared-memory-leaks).
