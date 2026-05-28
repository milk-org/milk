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
`/dev/shm/<name>.im.shm` containing:

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
resolveIMGID(&imgid, ERRMODE_WARN);

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

### Writer Side

```c
// Write new frame data
memcpy(img->array.F, new_data, nbytes);

// Update metadata
img->md->cnt0++;  // frame counter

// Wake ALL readers (semindex = -1)
ImageStreamIO_sempost(img, -1);
```

### Reader Side

```c
// Wait for new frame (blocks until writer
// posts)
ImageStreamIO_semwait(img, semindex);

// Read the data
float *data = img->array.F;
process(data);
```

### Semaphore Assignment

Use `ImageStreamIO_getsemwaitindex()` to get a
unique semaphore index for your reader:

```c
int semindex = ImageStreamIO_getsemwaitindex(
    img, 0);
```

### Common Semaphore Issues

| Symptom                      | Cause                     | Fix                                                  |
| ---------------------------- | ------------------------- | ---------------------------------------------------- |
| Reader never wakes           | Writer not posting        | Add `ImageStreamIO_sempost(img, -1)`                 |
| Reader busy-spins            | Sem leaked (value > 0)    | Reset with `sem_init()` or `ImageStreamIO_seminit()` |
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

```c
IMGID img = imgid_make_from_name("mystream");
img.mdt->naxis = 2;
img.mdt->size[0] = 128;
img.mdt->size[1] = 128;
img.mdt->datatype = _DATATYPE_FLOAT;
img.mdt->shared = 1;
imgid_mkimage(&img);
```

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
| `ImageStreamIO_seminit()`         | Reset semaphore            |
| `ImageStreamIO_UpdateIm()`        | cnt0++, write=0, sem post  |

## Stream Process Trace (Ancestry)

Every stream carries an array of
`STREAM_PROC_TRACE` entries that records the
chain of processes that produced the data.
This is the mechanism that lets `streamCTRL`
and `milk-CTRL` display which process wrote
a stream and which upstream stream triggered
that process.

### STREAM_PROC_TRACE Struct

Defined in `ImageStruct.h`:

```c
typedef struct {
    int             triggermode;     // PROCESSINFO_TRIGGERMODE_*
    pid_t           procwrite_PID;   // PID of writing process
    ino_t           trigger_inode;   // inode of trigger stream
    struct timespec ts_procstart;    // when process was triggered
    struct timespec ts_streamupdate; // when stream was written
    int             trigsemindex;    // trigger semaphore index
    int             triggerstatus;   // PROCESSINFO_TRIGGERSTATUS_*
    uint64_t        cnt0;            // trigger stream cnt0
} STREAM_PROC_TRACE;
```

### How Ancestry Propagates

When a compute unit calls
`processinfo_update_output_stream(processinfo,
output_image, input_image)`, the function:

1. Writes `streamproctrace[0]` on the output
   with the current process's PID, trigger
   mode, trigger inode, and timestamps.
2. Copies `input_image->streamproctrace[0..N-2]`
   into `output_image->streamproctrace[1..N-1]`,
   shifting the input's ancestry down by one
   slot.
3. Calls `ImageStreamIO_UpdateIm()` which
   increments `cnt0`, clears `write`, and posts
   all semaphores.

This creates a rolling history of up to
`IMAGE_NB_PROCTRACE` (default 10) entries,
where index 0 is the most recent writer and
index N-1 is the oldest ancestor.

```
Stream A (camera)         Stream B (processed)      Stream C (output)
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ [0] camera PID  │──────▶│ [0] procB PID   │──────▶│ [0] procC PID   │
│ [1] (empty)     │       │ [1] camera PID  │       │ [1] procB PID   │
│ [2] (empty)     │       │ [2] (empty)     │       │ [2] camera PID  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

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

### For Compute Unit Authors

Ancestry propagation is **automatic** if you
call `processinfo_update_output_stream()`
correctly. No extra code is needed. The two
requirements are:

1. Pass the correct `input_image` pointer so
   the function can copy ancestry from it.
2. For generators (no input), pass `NULL` —
   only `streamproctrace[0]` is written.

### Viewing Ancestry

Use `streamCTRL` (press `t` on a selected
stream) to see the full process trace chain,
or view it in `milk-CTRL` overview.

## Source Files

| File                                       | Role                    |
| ------------------------------------------ | ----------------------- |
| `src/engine/ImageStreamIO/ImageStreamIO.c` | Core SHM operations     |
| `src/engine/ImageStreamIO/ImageStreamIO.h` | Public API              |
| `src/engine/ImageStreamIO/ImageStruct.h`   | IMAGE struct definition |
| `src/engine/libmilkdata/imgid.c`           | IMGID creation/parsing  |
| `src/engine/libmilkdata/imgid.h`           | IMGID struct definition |

## Debugging Tips

### List All Streams

```bash
echo "listim" | milk-cli 2>/dev/null
# or
ls /dev/shm/*.im.shm 2>/dev/null
```

### Inspect Stream Metadata

```bash
echo "iminfo mystream" | milk-cli 2>/dev/null
echo "imseminfo mystream" | milk-cli 2>/dev/null
```

### Monitor Stream Activity

Use `streamCTRL` or `milk-streamCTRL` to see
real-time frame rates and semaphore states.

### Clean Up Stale Streams

```bash
rm /dev/shm/mystream.im.shm
```
