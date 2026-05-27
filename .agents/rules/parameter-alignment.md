---
trigger: always_on
---

# Function Parameter Alignment

Multi-line function prototypes and definitions must
use **column-aligned** parameter names for
readability. Pad each type (including any `*` prefix
on the name) so that all parameter names start at the
same column.

## Rule

1. Each parameter on its own line, indented 4 spaces.
2. Pad the type with spaces so all parameter names
   begin at the same column within that prototype.
3. The alignment column is determined by the longest
   `type [*]` prefix among all parameters.
4. Use exactly **one space** between the last type
   token and the `*` or name when there is no need
   for padding (i.e., it is already the longest).
5. Qualifiers like `const` and `restrict` are part
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
