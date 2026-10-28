---
tags: [cli, fps, api, performance, guide]
---

# Bringing Your Own Compiled C Code (`milk_calc_ext.h` & `milk-cc`)

`milk` provides four progressive tiers for building real-time stream compute units — from pure
`.calc` expressions with zero C code (Tier 0) to custom SIMD pixel/stencil/image kernels fused
into `.calc` (Tier 1), stateful multi-port operators (Tier 2), and standalone `MILK_RUNNABLE`
binaries compiled with a single command via `milk-cc` (Tier 3).

```mermaid
flowchart LR
    CAM["wfscam"] --> PREP["Tier 0 / Tier 1:\ncalibrate + clean"]
    PREP --> SLOPES["Tier 2:\nslopes sx, sy, flux"]
    SLOPES --> CTRL["Tier 3:\nmodal controller"]
    CTRL --> DM["dmcmd"]
```

---

## Overview of the Four Tiers

| Tier | What You Write | What `milk` Handles | Entry Point / Header |
| :--- | :--- | :--- | :--- |
| **0: Pure `.calc`** | A `.calc` program or `-e` expression | Compiles fused SIMD bytecode, runs on each trigger frame, manages FPS & `processinfo`, posts outputs | `milk-fpsexec-milk-calc` / `@calc` |
| **1a: 1D SIMD Block Kernel** | A 1D per-pixel C function (`MILK_CALC_REG_BLK` / `MILK_CALC_REG_BLK2`) + parameter table | Fuses directly into the OpenMP L1 cache tile loop (`FUSE_BLK = 1024`), auto-exposes parameters in FPS | `#include "milk_calc_ext.h"` (`-l file.c`) |
| **1b: 2D Stencil Kernel** | A $(2R+1)\times(2R+1)$ row-strip C function (`MILK_CALC_REG_STENCIL`, $R \in \{1,2,3\}$) | Pre-clamps image borders and feeds $2R+1$ row pointers of length $N + 2R$ per tile | `#include "milk_calc_ext.h"` (`-l file.c`) |
| **1c: Whole-Image Kernel** | A 1-input / 1-output whole-image C function (`MILK_CALC_REG_IMG`) | Runs as an in-expression pre-stage with zero intermediate copy on bare `float32` streams | `#include "milk_calc_ext.h"` (`-l file.c`) |
| **2: Stateful Multi-Port Operator** | `init` / `exec` / `free` C callbacks with named `IN`/`OUT` image & scalar ports (`MILK_CALC_OP_DECLARE`) | Allocates/resizes output streams (`milk_op_set_out_shape`), re-runs `init` on shape/CONF changes, exposes scalar outputs in FPS | `#include "milk_calc_ext.h"` (`-l file.c`) |
| **3: Standalone `MILK_RUNNABLE`** | A complete `MILK_RUNNABLE` `.c` file with custom loop/trigger logic | `milk-cc` generates the `main()` stub and links `<prefix>-fpsexec-<id>` without CMake boilerplate | `#include "milk_runnable.h"` (`milk-cc unit.c`) |

---

## Tier 0: Pure `.calc` (No C Code)

**Use when:** the pipeline can be expressed with built-in `.calc` operators, slices, reductions,
and control flow (`if`, `for`, `skip`, `hist`, `prev`).

### `wfsprep.calc`

```calc
# Runs on each frame of wfscam; dark and flat are shared streams
if imax(wfscam) > satlevel {
    nsat = nsat + 1                      # declared parameter: written back to the FPS
    skip                                 # saturated frame: no output this tick
}
local raw = wfscam - dark                # process-private scratch, no /dev/shm stream
local med = medfilt3(raw)
clean = where(abs(raw - med) > hotth, med, raw) * flat      # hot-pixel replacement
mean, rms, lo, hi = istats(clean)        # one pass, four FPS outputs
```

### Run and Tune

```bash
milk-fpsexec-milk-calc -n wfsprep -t wfscam \
    -p "satlevel:float=60000 hotth:float=200 nsat:int=0 mean:float=0 rms:float=0 \
        lo:float=0 hi:float=0" \
    -f wfsprep.calc fpsinit
milk-fpsexec-milk-calc -n wfsprep runstart   # in the background

@set f.wfsprep.hotth=150                   # applied on the next frame
@get f.wfsprep.rms
```

---

## Tier 1: Custom Pixel, Stencil, and Whole-Image Kernels (`milk_calc_ext.h`)

**Use when:** you want to embed custom C math inside `.calc` expressions without giving up loop
fusion or declarative `.calc` composition:

- **Tier 1a (`MILK_CALC_REG_BLK`, `MILK_CALC_REG_BLK2`)**: 1-input or 2-input 1D SIMD block kernels
  fused directly into the L1 cache tile loop (`FUSE_BLK = 1024`).
- **Tier 1b (`MILK_CALC_REG_STENCIL`)**: 2D $(2R+1)\times(2R+1)$ neighborhood kernels
  ($R \in \{1, 2, 3\}$). The engine pre-clamps row and column boundaries into thread-local strip
  buffers and passes `const float *const *rows` (`2*R + 1` row pointers, each of length
  `len + 2*R`). For horizontal output index `i` (`0 <= i < len`), the center pixel is
  `rows[R][i + R]` and its `(dy, dx)` neighbor (`-R <= dy, dx <= R`) is
  `rows[R + dy][i + R + dx]`.
- **Tier 1c (`MILK_CALC_REG_IMG`)**: Single-function whole-image kernels (`IMAGE *out`,
  `const IMAGE *in`, `const struct <name>_par *P`) for operations that need 2D random access or
  column/row passes across the full image while still composing inside `.calc` expressions.

### Example: `wfskernels.c` (Tiers 1a, 1b, and 1c in One File)

```c
#include <math.h>
#include <stdint.h>
#include "milk_calc_ext.h"

/* ---- Tier 1a: 1D per-pixel detector non-linearity with saturation knee ---- */
#define NLCORR_PARAMS(X)                                                                 \
    X(".c2",   c2,   FPTYPE_FLOAT64, -1.2e-6, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,    \
      "Quadratic non-linearity coefficient")                                             \
    X(".knee", knee, FPTYPE_FLOAT64, 52000.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,    \
      "ADU ceiling above which pixel is clamped")

MILK_CALC_PARAMS(nlcorr, NLCORR_PARAMS); /* defines struct nlcorr_par { double c2, knee; } */

static void nlcorr_f(
    float *restrict          out,
    const float *restrict    in,
    const struct nlcorr_par *P,
    int                      len)
{
    const float c2   = (float) P->c2;
    const float knee = (float) P->knee;
    #pragma omp simd
    for (int i = 0; i < len; i++)
    {
        float x = (in[i] > knee) ? knee : in[i];
        out[i]  = x * (1.0f + c2 * x);
    }
}

/* ---- Tier 1b: 3x3 neighborhood despiker (R = 1: rows[0..2] of length len + 2) ---- */
#define DESPIKE_PARAMS(X)                                                                \
    X(".th", th, FPTYPE_FLOAT64, 50.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,           \
      "Replace center pixel when |center - mean8| > th")

MILK_CALC_PARAMS(despike, DESPIKE_PARAMS);

static void despike3x3_f(
    float *restrict           out,
    const float *const       *rows,
    const struct despike_par *P,
    int                       len)
{
    const float  th = (float) P->th;
    const float *a  = rows[0];
    const float *b  = rows[1];
    const float *c  = rows[2];

    #pragma omp simd
    for (int i = 0; i < len; i++)
    {
        float m8 = 0.125f * (a[i] + a[i + 1] + a[i + 2] +
                             b[i]            + b[i + 2] +
                             c[i] + c[i + 1] + c[i + 2]);
        float v  = b[i + 1];
        out[i]   = (fabsf(v - m8) > th) ? m8 : v;
    }
}

/* ---- Tier 1c: Whole-image per-column mean subtraction ---- */
#define COLDEMEAN_PARAMS(X)                                                              \
    X(".scale", scale, FPTYPE_FLOAT64, 1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,      \
      "Post-demean scale factor")

MILK_CALC_PARAMS(col_demean, COLDEMEAN_PARAMS);

static int col_demean_img(
    IMAGE                       *out,
    const IMAGE                 *in,
    const struct col_demean_par *P)
{
    uint32_t     nx  = in->md[0].size[0];
    uint32_t     ny  = (in->md[0].naxis > 1) ? in->md[0].size[1] : 1;
    const float *src = in->array.F;
    float       *dst = out->array.F;
    const float  sc  = (float) P->scale;

    for (uint32_t x = 0; x < nx; x++)
    {
        double sum = 0.0;
        for (uint32_t y = 0; y < ny; y++)
        {
            sum += src[(size_t) y * nx + x];
        }
        float mean = (float) (sum / (double) ny);
        for (uint32_t y = 0; y < ny; y++)
        {
            dst[(size_t) y * nx + x] = (src[(size_t) y * nx + x] - mean) * sc;
        }
    }
    return 0;
}

MILK_CALC_EXT_EXPORT(
    MILK_CALC_REG_BLK("nlcorr", nlcorr_f, nlcorr, /*cost=*/2),
    MILK_CALC_REG_STENCIL("despike3x3", despike3x3_f, despike, /*radius=*/1, /*cost=*/4),
    MILK_CALC_REG_IMG("col_demean", col_demean_img, col_demean));
```

### Using `wfskernels.c` in `@calc` or `milk-fpsexec-milk-calc`

No manual compiler invocation is needed — pass `-l wfskernels.c` directly:

```bash
# One-shot in @calc:
@calc -l wfskernels.c "clean = despike3x3(col_demean(nlcorr(wfscam - dark))) * flat"

# Persistent compute unit:
milk-fpsexec-milk-calc -n wfsprep -t wfscam -l wfskernels.c \
    -e "clean = despike3x3(col_demean(nlcorr(wfscam - dark))) * flat" fpsinit
milk-fpsexec-milk-calc -n wfsprep runstart
```

On first load, `milk` compiles `wfskernels.c` into
`${TMPDIR:-/tmp}/milk-calc-cache-<uid>/ext_<fnv1a64>.so` with
`-O3 -march=native -ffast-math -fopenmp-simd -fPIC -shared` and reuses the cached `.so` on subsequent runs
until the source file content changes.

---

## FPS Parameter Binding Rules

A parameter table holds at most 16 numeric fields (`FPTYPE_FLOAT64/32`, `INT64/32`, `UINT64/32`,
`ONOFF`) in a struct of at most 256 bytes; a library whose table breaks this, or declares a string
field, is refused at load with the offending parameter named.

When a C kernel or operator declares parameters via `MILK_CALC_PARAMS(name, PARAM_MACRO)`,
`milk-fpsexec-milk-calc` inspects every call site in the `.calc` program (`-e`, `-f`, `-i`, `-I`)
and applies three rules:

1. **Unbound parameters are automatically exposed in FPS**:
   - Calling `nlcorr(raw)` exposes `.nlcorr.c2` and `.nlcorr.knee` in the unit's FPS with their C
     defaults, types, flags, and descriptions.
   - Live updates via `@set f.wfsprep.nlcorr.knee = 48000` take effect on the next frame without
     rebuilding the fused program.
2. **Instance labels (`func:label(...)`) disambiguate multiple calls**:
   - Calling `despike3x3:coarse(raw)` and `despike3x3:fine(a)` exposes two independent FPS
     parameter groups: `.coarse.th` and `.fine.th`.
3. **Call-site bound arguments are hidden from FPS**:
   - Calling `despike3x3:fine(a, th = 3.0 * rms)` binds `th` to the `.calc` expression `3.0 * rms`
     and omits `.fine.th` from the FPS so external users cannot accidentally fight the internal
     formula.

---

## Tier 2: Stateful Multi-Port Operators (`MILK_CALC_OP_DECLARE`)

**Use when:** your algorithm requires:

- Multiple input streams and/or multiple output streams (`sx, sy, flux = shslopes(...)`).
- Output dimensions that differ from the input image dimensions (`milk_op_set_out_shape`).
- Persistent state (`void *state`) allocated once in `init()`, reused on every frame in `exec()`,
  and released in `free()`.
- Automatic re-initialization when an input stream's shape changes or when a configuration-only
  parameter (declared without `FPFLAG_WRITERUN`) is modified.

### Example: `shslopes.c`

```c
#include <stdlib.h>
#include "milk_calc_ext.h"

#define SH_PARAMS(X)                                                                     \
    X(".nsub",   nsub,   FPTYPE_INT64,   2,   FPFLAG_DEFAULT_INPUT,                      \
      "Sub-aperture grid size (CONF-only: changing it re-runs sh_init)")                 \
    X(".thresh", thresh, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,    \
      "Pixel threshold before summing (run-writable)")

#define SH_PORTS(X)                                                                      \
    X(IN,  frame, MILK_PORT_IMAGE,  MILK_PORT_TRIGGER,                "WFS image")       \
    X(IN,  flat,  MILK_PORT_IMAGE,  MILK_PORT_LATEST | MILK_PORT_OPT, "Optional flat")   \
    X(OUT, sx,    MILK_PORT_IMAGE,  MILK_POST_EVERY,                  "X slopes")        \
    X(OUT, sy,    MILK_PORT_IMAGE,  MILK_POST_EVERY,                  "Y slopes")        \
    X(OUT, flux,  MILK_PORT_SCALAR, 0,                                "Total WFS flux")

MILK_CALC_OP_DECLARE(shs, SH_PARAMS, SH_PORTS);

struct sh_state
{
    int nsub;
    int sub_pix;
};

static int sh_init(
    milk_op_ctx          *ctx,
    const struct shs_par *P,
    void                **state)
{
    const IMAGE *frame = MILK_IN(ctx, shs_frame);
    int          nsub  = (P->nsub > 0) ? (int) P->nsub : 2;
    uint32_t     nx    = frame->md[0].size[0];

    struct sh_state *st = calloc(1, sizeof(*st));
    if (st == NULL)
    {
        return -1;
    }
    st->nsub    = nsub;
    st->sub_pix = (int) (nx / (uint32_t) nsub);

    /* Output slope maps have shape nsub x nsub instead of frame's nx x ny */
    milk_op_set_out_shape(ctx, shs_sx, (uint32_t) nsub, (uint32_t) nsub);
    milk_op_set_out_shape(ctx, shs_sy, (uint32_t) nsub, (uint32_t) nsub);
    *state = st;
    return 0;
}

static int sh_exec(
    milk_op_ctx          *ctx,
    const struct shs_par *P,
    void                 *state)
{
    struct sh_state *st    = (struct sh_state *) state;
    const IMAGE     *frame = MILK_IN(ctx, shs_frame);
    IMAGE           *sx    = MILK_OUT(ctx, shs_sx);
    IMAGE           *sy    = MILK_OUT(ctx, shs_sy);

    const float *in  = frame->array.F;
    uint32_t     nx  = frame->md[0].size[0];
    int          ns  = st->nsub;
    int          sp  = st->sub_pix;
    float        th  = (float) P->thresh;
    double       tot = 0.0;

    for (int j = 0; j < ns; j++)
    {
        for (int i = 0; i < ns; i++)
        {
            double s = 0.0;
            for (int dy = 0; dy < sp; dy++)
            {
                for (int dx = 0; dx < sp; dx++)
                {
                    float v = in[(j * sp + dy) * (int) nx + (i * sp + dx)] - th;
                    if (v > 0.0f)
                    {
                        s += v;
                    }
                }
            }
            sx->array.F[j * ns + i] = (float) s;
            sy->array.F[j * ns + i] = (float) (s * 2.0);
            tot += s;
        }
    }
    MILK_OUT_SCALAR(ctx, shs_flux) = tot;
    return 0;
}

static void sh_free(void *state)
{
    free(state);
}

MILK_CALC_EXT_EXPORT(
    MILK_CALC_REG_OP("shslopes", shs, sh_init, sh_exec, sh_free));
```

### Running `shslopes.c`

```bash
milk-fpsexec-milk-calc -n u_sh -t wfs_in -l shslopes.c \
    -e "wfs_sx, wfs_sy, total_flux = shslopes(wfs_in)" exec

# Update runtime parameter and inspect scalar output port in FPS:
@set f.u_sh.shslopes.thresh = 1.0
@get f.u_sh.total_flux
```

Because `flux` is declared as `MILK_PORT_SCALAR`, assigning it to `total_flux` on the LHS
automatically creates the output parameter `.total_flux` (`FPFLAG_DEFAULT_OUTPUT`) in `u_sh`'s FPS.

---

## Tier 3: Standalone `MILK_RUNNABLE` Units via `milk-cc`

**Use when:** your unit needs full control over its event loop, multiple asynchronous semaphores,
custom `confcheck` validation, or hardware/network threads.

Write a standard single-file `MILK_RUNNABLE` compute unit with `EXPORT_FPSEXEC(prefix, "id", sym)`
and compile it directly with `milk-cc`:

```c
#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

static char   in_sname[FUNCTION_PARAMETER_STRMAXLEN]  = "cscale_in";
static char   out_sname[FUNCTION_PARAMETER_STRMAXLEN] = "cscale_out";
static double factor                                  = 3.0;

#define CSCALE_PARAMS(X)                                                                 \
    X(".in_sname",  in_sname,  FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT_STREAM,        \
      "Input stream")                                                                    \
    X(".out_sname", out_sname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,               \
      "Output stream")                                                                   \
    X(".factor",    &factor,   FPTYPE_FLOAT64,    1,                                     \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Scale factor")

MILK_PARAMS_DEFINE(cscale, CSCALE_PARAMS);

static errno_t compute_cscale(const MILK_RUNNABLE *self)
{
    IMGID imgin = imgid_make_from_name(in_sname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }
    IMGID imgout = imgid_make_from_name(out_sname);
    imgid_copy(&imgin, &imgout);
    imgout.mdt->shared = 1;
    imcreateIMGID(&imgout);

    uint64_t nelem = imgin.md->nelement;
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        const float f    = (float) factor;
        imgout.md->write = 1;
        for (uint64_t i = 0; i < nelem; i++)
        {
            imgout.im->array.F[i] = imgin.im->array.F[i] * f;
        }
        processinfo_update_output_stream(processinfo, imgout.im, imgin.im);
    }
    MILK_PROCINFO_END(self)
    imgid_free(&imgin);
    imgid_free(&imgout);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cscale = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = {
        .fps_name         = "cscale",
        .cmdkey           = "cscale",
        .description      = "Scale a stream by a factor",
        .description_long = "Standalone MILK_RUNNABLE compiled via milk-cc.",
    },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cscale),
    .compute = compute_cscale,
};

EXPORT_FPSEXEC(milk, "custom-scale", runnable_cscale);
```

### Build and Run with `milk-cc`

```bash
# Auto-detects EXPORT_FPSEXEC(milk, "custom-scale", runnable_cscale)
# and builds ./milk-fpsexec-custom-scale:
milk-cc custom_scale.c

# Run in the foreground (exec), or as a unit in the background (fpsinit, runstart) like any
# built-in fpsexec:
./milk-fpsexec-custom-scale -n cs1 exec cscale_in cscale_out 2.5
```

`milk-cc` also compiles Tier 1 / Tier 2 `.c` files (those using `MILK_CALC_EXT_EXPORT`) into
shared libraries (`<stem>.so`) when you prefer pre-compiling `.so` plugins ahead of time:

```bash
milk-cc wfskernels.c -o wfskernels.so
milk-fpsexec-milk-calc -n wfsprep -l ./wfskernels.so -f wfsprep.calc exec

# Inspect the pre-compiled bytecode engines and any loaded C extension kernels/params:
@calc -l wfskernels.c --list
milk-fpsexec-milk-calc -l ./wfskernels.so --list
```

---

## `milk_calc_ext.h` Quick Reference

| Macro / Function | Purpose |
| :--- | :--- |
| `MILK_CALC_PARAMS(name, PARAM_MACRO)` | Define `struct <name>_par` (and `struct <name>`) plus FPS metadata from `X(".kw", field, fptype, defval, flags, desc)` |
| `MILK_CALC_NOPARAMS` | Sentinel for stateless kernels with no parameters |
| `MILK_CALC_REG_BLK(name, fn, par, cost)` | Register a 1-input 1D SIMD block kernel: `void fn(float *restrict out, const float *restrict in, const struct par_par *P, int len)` |
| `MILK_CALC_REG_BLK2(name, fn, par, cost)` | Register a 2-input 1D SIMD block kernel: `void fn(float *restrict out, const float *restrict in0, const float *restrict in1, const struct par_par *P, int len)` |
| `MILK_CALC_REG_STENCIL(name, fn, par, R, cost)` | Register a 2D $(2R+1)\times(2R+1)$ stencil ($R \in \{1,2,3\}$): `void fn(float *restrict out, const float *const *rows, const struct par_par *P, int len)` |
| `MILK_CALC_REG_IMG(name, fn, par)` | Register a 1-input / 1-output whole-image kernel: `int fn(IMAGE *out, const IMAGE *in, const struct par_par *P)` |
| `MILK_CALC_OP_DECLARE(name, PARAMS, PORTS)` | Declare a Tier 2 operator's parameter struct and `IN`/`OUT` port table |
| `MILK_CALC_REG_OP(name, decl, init, exec, free)` | Register a Tier 2 stateful operator |
| `MILK_IN(ctx, port)` / `MILK_IN_SCALAR(ctx, port)` | Read an input `const IMAGE *` or `double` port inside `init` / `exec` |
| `MILK_OUT(ctx, port)` / `MILK_OUT_SCALAR(ctx, port)` | Access an output `IMAGE *` or assignable `double` lvalue inside `exec` |
| `milk_op_set_out_shape(ctx, port, nx, ny)` | Set 2D output port dimensions inside `init` (`milk_op_set_out_shape3d` for 3D) |
| `MILK_CALC_EXT_EXPORT(...)` | Export one or more `MILK_CALC_REG_*` entries from the translation unit |

---

← [Scripting](scripting.md) · [Programmer's Guide](../arch/programmers_guide.md) · [Documentation Index](../index.md)
