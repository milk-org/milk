---
name: stream-modifier-guide
description: Deep reference for IMGID stream parsing,
  modifiers (@S:, @L:, @F:), and slice syntax
---

# Stream Modifier Guide

This skill provides deep context on how `milk-cli`
parses stream references, applies modifiers, and
handles slice syntax. Essential for debugging
stream-related CLI issues or extending the
modifier system.

## When to Use

- Debugging stream creation/connection failures
- Adding new stream modifiers
- Fixing modifier parsing bugs
- Implementing or debugging slice syntax `im[a:b]`
- Understanding how streams flow through the CLI
  pipeline

## IMGID Overview

`IMGID` is the unified handle for images and
streams throughout milk. It combines a name, size
metadata, shared-memory flag, and optional
modifiers.

```c
typedef struct {
    char name[200];    // Stream name
    long naxis;        // Number of axes
    uint32_t size[3];  // Axis sizes
    int shared;        // 1 = shared memory
    IMAGE *im;         // Pointer to image data
    // ... modifier fields ...
} IMGID;
```

## Stream Modifier Syntax

Modifiers are appended to stream names using the
`@` prefix:

| Modifier | Syntax | Meaning |
|----------|--------|---------|
| Semaphore | `stream@S:N` | Use semaphore index N |
| Slice | `stream@L:N` | Use 3D slice index N |
| FITS file | `stream@F:path` | Load from FITS file |

### Examples

```
# Connect to stream using semaphore 3
imA@S:3

# Access slice 5 of a 3D cube
cube@L:5

# Load from FITS file
data@F:/path/to/file.fits
```

### Modifier chaining

Multiple modifiers can be chained:
```
stream@S:2@L:5
```

## Parsing Pipeline

Stream references are parsed in this order:

```
User input string
  → imgid_make_from_name()      [IMGID creation]
    → parse modifiers (@S:, @L:, @F:)
    → strip modifiers from name
  → imgid_resolve()             [SHM connection]
    → connect to /dev/shm/<name>.im.shm
    → apply semaphore selection
    → apply slice selection
```

### Key source files

| File | Role |
|------|------|
| `src/engine/libmilkdata/imgid.c` | IMGID creation and modifier parsing |
| `src/engine/libmilkdata/imgid.h` | IMGID struct definition |
| `src/engine/ImageStreamIO/ImageStreamIO.c` | Low-level SHM stream operations |
| `src/cli/CLIcore/cli_calc_parser.c` | CLI expression parser (handles stream refs) |
| `src/cli/CLIcore/cli_calc_tokenizer.c` | Tokenizer (handles bracket syntax) |

## Modifier Parsing Details

### `@S:` — Semaphore selection

```c
// In imgid.c, parse_imgid_modifiers():
// Extracts integer N from "@S:N"
// Sets imgid.semindex = N
// Semaphore N is used for wait/post operations
```

Valid range: 0 to `IMAGE_NB_SEMAPHORE - 1`
(typically 0–9).

### `@L:` — Slice selection

```c
// Extracts integer N from "@L:N"
// When the stream is 3D (naxis=3), selects
// the 2D slice at index N along axis 2
// Effective result: a 2D view of shape
//   [size[0], size[1]] at offset N*size[0]*size[1]
```

### `@F:` — FITS file source

```c
// Extracts path string from "@F:/path/to/file"
// Loads the FITS file into the IMGID
// Used for offline processing / testing
```

## Bracket Slice Syntax

The bracket syntax `im[x0:x1,y0:y1]` provides
NumPy-style slicing in the CLI:

```
# Create a 100x100 subregion starting at (10,20)
im[10:109,20:119]
```

### Tokenizer handling

The CLI tokenizer must recognize brackets as part
of a stream expression, not as shell glob
characters. The restricted symbol checker in the
CLI was modified to allow `[` and `]` to pass
through to the expression evaluator.

### Parse flow

```
"im[10:19,20:29]"
  → tokenizer splits into:
    - stream name: "im"
    - slice spec: [10:19,20:29]
  → evaluator creates a temporary image with
    the sliced data
  → temporary is cleaned up after expression
    evaluation
```

### Current limitations

- Slicing creates a **copy**, not a view
- Only 2D slices from 2D images are fully
  supported in the CLI calc pipeline
- Step syntax (`im[::2]`) is not supported
- Negative indices are not supported

## Creating Streams Programmatically

### From C code

```c
IMGID img = imgid_make_from_name("mystream");
img.mdt->naxis = 2;
img.mdt->size[0] = 128;
img.mdt->size[1] = 128;
img.mdt->shared = 1;       // shared memory stream
img.type = _DATATYPE_FLOAT;
imgid_mkimage(&img);

// Write data
float *data = img.im->array.F;
data[0] = 42.0f;

// Post semaphores to notify readers
ImageStreamIO_sempost(img.im, -1);
```

### From CLI

```
# Create 128x128 float stream
mk2Dim mystream 128 128

# Assign via expression
mystream=0.5*otherstream
```

## Debugging Stream Issues

### Stream not found

```bash
# Check if stream exists in SHM
ls /dev/shm/mystream.im.shm

# List all streams
echo "listim" | milk-cli 2>/dev/null
```

### Wrong data type

```bash
# Check stream info
echo "iminfo mystream" | milk-cli 2>/dev/null
```

### Semaphore issues

```bash
# Check semaphore state
echo "imseminfo mystream" | milk-cli 2>/dev/null
```

### Modifier not applied

1. Set `VERBOSE=1` environment variable
2. Check that modifier string is parsed before
   `imgid_resolve()` is called
3. Verify the modifier field in the IMGID struct
   is populated
