---
trigger: always_on
---

# Function Parameter Alignment

[Coding Standards](../../docs/developer/coding_standards.md#function-signatures)
asks for one parameter per line, with the names
column-aligned. How to align them:

## Rule

1. The alignment column is determined by the longest
   `type [*]` prefix among all parameters.
2. Use exactly **one space** between the last type
   token and the `*` or name when there is no need
   for padding (i.e., it is already the longest).
3. Qualifiers like `const` and `restrict` are part
   of the type for alignment purposes.

## Examples

**BAD** — random padding:

```c
int functionparameter_GetFileName(
    FPS *fps,
    FPS_PARAM        *fparam,
    char                      *outfname,
    char                      *tagname)
```

**GOOD** — column-aligned:

```c
int functionparameter_GetFileName(
    FPS       *fps,
    FPS_PARAM *fparam,
    char      *outfname,
    char      *tagname)
```

**GOOD** — no pointers, names align:

```c
static errno_t compute_stream(
    const float *restrict in,
    float       *restrict out,
    uint64_t              nelement)
```

**GOOD** — single short type, no padding needed:

```c
static int parse_one_axis(
    const char  *spec,
    IMGID_SLICE *s,
    int          axis)
```

## Scope

Applies to all `.c` and `.h` files in `src/`.
Does not apply to single-parameter functions
(no alignment needed) or to macro arguments.
