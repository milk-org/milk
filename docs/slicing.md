---
tags:
  - streams
  - slicing
  - IMGID
---

# Stream Slicing

Stream slicing provides **FITSIO-like extended file syntax**
for extracting, flipping, striding, and binning sub-regions of
streams directly from their name string. This eliminates
the need for separate crop/flip commands in both interactive
CLI expressions and FPS stream-processing loops.

## Syntax

Append `[axis0, axis1, axis2]` to any stream name.
Each axis specification follows this grammar:

| Form               | Meaning                      |
| ------------------ | ---------------------------- |
| `*`                | Full axis, forward           |
| `-*`               | Full axis, reversed (flip)   |
| `N`                | Single index N               |
| `start:end`        | Inclusive range [start, end] |
| `start:end::step`  | Range with stride            |
| `start:end::stepb` | Range with binning (average) |
| _(empty)_          | Same as `*`                  |

**Indexing is 0-based, inclusive on both ends.**

### Examples

```text
im[0:19,10:29]       # 2D crop: x=0..19, y=10..29
im[*,-*]             # Flip Y axis
im[0:99::2,*]        # X stride: every other column
im[0:99::4b,*]       # X binning: average groups of 4
cube[*,*,-*]         # 3D: flip along Z axis
im[-20:-1,0:9]       # Negative index: last 20 columns
```

## Architecture

### Parser (`imgid_slice.h` / `imgid_slice.c`)

The parser lives in `src/engine/libfps/` (engine tier,
no CLI dependency). Access functions are in
`src/coremods/COREMOD_memory/stream_slice.h`.
Key parser functions:

| Function                    | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| `imgid_slice_split_name()`  | Split `im[spec]` → bare name `im` + spec        |
| `imgid_slice_parse()`       | Parse spec string → `IMGID_SLICE` descriptor    |
| `imgid_slice_output_size()` | Compute output dimensions from source + slice   |
| `imgid_slice_format()`      | Format descriptor back to display string        |
| `imgid_slice_shmname()`     | Generate deterministic SHM name for shared mode |

### IMGID Integration

The `IMGID` struct (in `IMGID.h`) was extended with:

```c
IMGID_SLICE slice;         // parsed from name[...]
IMAGE      *slice_im;      // materialized buffer
uint64_t    slice_last_cnt0; // source frame counter
int         slice_shared;  // 1 if @S: requested
```

**`imgid_make_from_name()`** automatically parses
brackets. The bare name (without brackets) is stored
in `img.name`, and the slice descriptor in `img.slice`.

### Access Functions (`stream_slice.h`)

Two inline functions provide the unified access API:

```c
IMAGE *imgid_get_image(IMGID *img);
void   imgid_put_image(IMGID *img);
```

**`imgid_get_image()`** returns `img->im` for non-sliced
streams (zero overhead — one predicted-away branch).
For sliced streams, it materializes the slice on first
call and re-materializes when the source `cnt0` changes.

**`imgid_put_image()`** is a no-op for non-sliced
streams. For sliced streams, it writes the local buffer
back into the source at the correct offsets.

## Usage

### CLI Expressions

In CLI arithmetic expressions, sliced image names are
preserved as single tokens. The bracket contents are
kept together with the image name:

```text
milk> mk2Dim im 100 100   # create 100×100 image
milk> out = im[10:29,10:29] + 1.0
```

### FPS Compute Units

In stream-processing code, use the access functions:

```c
IMGID imgin = imgid_make_from_name("wfs[0:63,0:63]");
resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);

/* Read sliced region */
IMAGE *view = imgid_get_image(&imgin);
float *data = view->array.F;

/* Modify and write back */
data[0] = 42.0f;
imgid_put_image(&imgin);
```

### Shared Memory Exposure

By default, materialized slices are **local-only**
buffers (not visible to other processes). To expose
a sliced view as shared memory, use the `@S:` prefix:

```text
@S:im[0:63,0:63]
```

This creates a shared memory stream with a deterministic
name derived from the source name and slice specification
(e.g., `im__s_0_63_0_63`).

### Output (LHS) Slicing

Slicing can also be used on the left-hand side of
assignments to write into a sub-region:

```text
milk> im[10:19,10:19] = 0.0
```

This zeroes a 10×10 region inside the stream `im`.

## Performance

- **Non-sliced streams**: Zero overhead. The
  `imgid_get_image()` check compiles to a single
  predicted-away branch.
- **Simple 2D crops**: Fast path using `memcpy`
  per row.
- **Strided/flipped**: General element-by-element
  copy with computed source/dest strides.
- **Buffer reuse**: The materialized buffer is
  allocated once and reused across frames.

## See Also

- [Streams](streams.md) — stream modifiers (`@S:`,
  `@L:`, `@F:`)
- [FPS](fps.md) — Function Parameter System
