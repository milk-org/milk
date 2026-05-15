---
description: Use const on pointer parameters that
  are read-only, with milk-specific exceptions.
---

# Const-Correctness Policy

Use `const` to document intent and let the
compiler catch accidental mutations. This rule
applies to **new code and code being actively
refactored** — do not bulk-migrate existing
functions.

## Required: String Parameters

All `char *` parameters that are not modified
by the function MUST be `const char *`.

```c
/* WRONG — allows accidental mutation */
IMGID stream_connect(char *name);
errno_t fps_save2disk(
    FUNCTION_PARAMETER_STRUCT *fps,
    char *dirname
);

/* RIGHT */
IMGID stream_connect(const char *name);
errno_t fps_save2disk(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *dirname
);
```

## Required: Input Pixel-Data Pointers

Pixel-data pointers in compute functions that
are read-only MUST be `const restrict`:

```c
/* WRONG — no const on read-only data */
static void compute(
    float *restrict in,
    float *restrict out,
    uint64_t nelement
);

/* RIGHT */
static void compute(
    const float *restrict in,
    float       *restrict out,
    uint64_t nelement
);
```

This applies to all typed array pointers
(`float *`, `double *`, `uint16_t *`, etc.)
used for input data in compute-heavy functions.

## Encouraged: Public API Input Pointers

Input-only struct pointers in public API
declarations (`.h` files) SHOULD be `const`:

```c
/* In .h — encouraged */
errno_t processinfo_print_summary(
    const PROCESSINFO *pinfo
);

errno_t image_export_fits(
    const IMAGE *img,
    const char  *filename
);
```

Migrate opportunistically — when a function
signature is already being changed for another
reason, add `const` to input pointers at the
same time.

## Not Required: IMGID Parameters

`IMGID` is passed by value and contains a
pointer to shared memory. Marking it `const`
only prevents reassigning the struct fields,
not the underlying pixel data. The protection
is too weak to justify the annotation noise:

```c
/* Acceptable — const adds little value */
void func(IMGID img);

/* Also acceptable but not required */
void func(const IMGID img);
```

## Not Required: FPS Pointers

`FUNCTION_PARAMETER_STRUCT *fps` points to
shared memory that other processes may modify
at any time. Using `const` is misleading
because the data is inherently mutable.
It is acceptable to use `const` when the
function genuinely does not write to FPS, but
it is not required:

```c
/* Acceptable — fps is SHM, always mutable */
errno_t fps_print_status(
    FUNCTION_PARAMETER_STRUCT *fps
);
```

## Local Variables

Using `const` on local pointer variables is
encouraged when it clarifies intent, but not
required:

```c
/* Encouraged — documents read-only access */
const IMAGE_METADATA *md = img.im->md;
const float *restrict pix = img.im->array.F;

/* Also acceptable */
IMAGE_METADATA *md = img.im->md;
```

## Transition Strategy

- **New code:** follow this rule strictly.
- **Existing code:** add `const` when editing
  a function for other reasons. Do not create
  const-only PRs that touch many files.
- **Headers:** when updating a `.h` declaration,
  update the matching `.c` definition to match.
