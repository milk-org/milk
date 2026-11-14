// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_ext_builtin.c
 * @brief Built-in `milk_calc_ext` spatial kernels, temporal `streamdelay` operator, and stream helpers.
 *
 * Purpose:     offer four COREMOD_arith spatial kernels and the COREMOD_memory temporal delay /
 *              filtering engine as calculator functions through the same interface as user
 *              extensions (milk_calc_ext.h), and hold the helpers that turn a name into an image
 *              for the resolver and the run-time steps.
 * Does:        registers `bblur1pix(im[, amp, niter])` (iterated 5-point cross blur),
 *              `rotate2d(im[, angle])` (bilinear rotation about the center), `circsym(im[, cx, cy])`
 *              and `circasym(im[, cx, cy])` (azimuthally symmetric part and its residual) as
 *              MILK_EXT_IMG kernels, and `streamdelay(im[, weights], delay=..., unit=...)` as a
 *              stateful MILK_EXT_OP operator when libmilkcalc loads (MILK_CALC_MODULE_EXPORT
 *              constructor). mc_resolve_or_autoconnect_image() finds an
 *              image by name (`@S:`/`@L:` prefix allowed), reattaches a loaded stream whose
 *              /dev/shm file was recreated (new inode), and attaches a /dev/shm stream not yet
 *              loaded; mc_parse_ident_streamname() replaces a FPTYPE_STREAMNAME variable by the
 *              stream it names and strips the prefix; mc_set_expr_from_im() copies the type and
 *              shape of an image to an AST node.
 * Used by:     the kernels: front/mc_resolve.c (lookup by name), ext/mc_ext_exec.c (run); the
 *              helpers: front/mc_resolve.c, plan/mc_plan.c, plan/mc_history.c, run/,
 *              ext/mc_ext_exec.c, linalg/. Declared in front/mc_ast.h.
 * Uses:        milk_calc_ext.h, COREMOD_arith image_spatial.h, ImageStreamIO,
 *              fps_streamname_parse.h (libfps), COREMOD_memory (image_ID.h, variable_ID.h),
 *              milkdata.h (milk_table_bump()).
 * See also:    ext/mc_ext_registry.c (the table the kernels join); resolve_arg_image() in
 *              src/cli/libmilkscript/core/milk_cmd_args.c (the same lookup, for module-command
 *              arguments); docs/guide/calc_c_extensions.md,
 *              "Tier 1: Custom Pixel, Stencil, and Whole-Image Kernels (`milk_calc_ext.h`)".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libfps/fps_streamname_parse.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/stream_delay.h"
#include "COREMOD_memory/variable_ID.h"
#include "front/mc_ast.h"
#include "milk_calc_ext.h"

/* ============================================================================================== */
/* Built-In `MILK_CALC_REG_IMG` Spatial Kernels                                                   */
/* ============================================================================================== */

#define BBLUR_EXT_PARAMS(X)                                                                    \
    X(".amp", amp, FPTYPE_FLOAT64, 0.1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Side weight") \
    X(".niter", niter, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Blur passes")
MILK_CALC_PARAMS(bblur1pix, BBLUR_EXT_PARAMS);

/**
 * bblur1pix_ext_img - Execute iterative 1-pixel 5-point cross blur on @in into @out
 * @out: Destination float32 image (pre-allocated to match @in)
 * @in:  Source float32 image
 * @P:   Packed parameter struct (`amp` neighbor weight, `niter` pass count)
 *
 * Return: 0 on success, non-zero on error
 */
static int bblur1pix_ext_img(IMAGE *out, const IMAGE *in, const struct bblur1pix_par *P)
{
    int    niter = (P != NULL && P->niter > 0) ? (int) P->niter : 1;
    double amp   = (P != NULL) ? P->amp : 0.1;
    return arith_image_bblur1pix_im(in, out, amp, niter);
}

#define ROT2D_EXT_PARAMS(X) \
    X(".angle", angle, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Angle (rad)")
MILK_CALC_PARAMS(rotate2d, ROT2D_EXT_PARAMS);

/**
 * rotate2d_ext_img - Rotate 2D image @in by `P->angle` radians around its center into @out
 * @out: Destination float32 or float64 image
 * @in:  Source 2D image
 * @P:   Packed parameter struct (`angle` in radians)
 *
 * Return: 0 on success, non-zero on error
 */
static int rotate2d_ext_img(IMAGE *out, const IMAGE *in, const struct rotate2d_par *P)
{
    return arith_image_rotate2d_im(in, out, (P != NULL) ? P->angle : 0.0);
}

#define CIRCSYM_EXT_PARAMS(X)                                                              \
    X(".cx", cx, FPTYPE_FLOAT64, -1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "X center") \
    X(".cy", cy, FPTYPE_FLOAT64, -1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Y center")
MILK_CALC_PARAMS(circsym, CIRCSYM_EXT_PARAMS);

/**
 * circsym_ext_img - Compute the azimuthally symmetric component of @in around `(cx, cy)`
 * @out: Destination float32 image
 * @in:  Source 2D float32 image
 * @P:   Packed parameter struct (`cx`, `cy`; negative values default to `0.5 * size`)
 *
 * Return: 0 on success, non-zero on error
 */
static int circsym_ext_img(IMAGE *out, const IMAGE *in, const struct circsym_par *P)
{
    double cx = (P != NULL && P->cx >= 0.0) ? P->cx : 0.5 * (double) in->md[0].size[0];
    double cy = (P != NULL && P->cy >= 0.0)
                    ? P->cy
                    : 0.5 * (double) ((in->md[0].naxis > 1) ? in->md[0].size[1] : 1);
    return arith_image_circsym_im(in, out, cx, cy, 0);
}

/**
 * circasym_ext_img - Compute the azimuthally asymmetric residual of @in around `(cx, cy)`
 * @out: Destination float32 image (`in - circsym(in, cx, cy)`)
 * @in:  Source 2D float32 image
 * @P:   Packed parameter struct (`cx`, `cy`; negative values default to `0.5 * size`)
 *
 * Return: 0 on success, non-zero on error
 */
static int circasym_ext_img(IMAGE *out, const IMAGE *in, const struct circsym_par *P)
{
    double cx = (P != NULL && P->cx >= 0.0) ? P->cx : 0.5 * (double) in->md[0].size[0];
    double cy = (P != NULL && P->cy >= 0.0)
                    ? P->cy
                    : 0.5 * (double) ((in->md[0].naxis > 1) ? in->md[0].size[1] : 1);
    return arith_image_circsym_im(in, out, cx, cy, 1);
}

/* ============================================================================================== */
/* Built-In `MILK_CALC_REG_OP` Temporal Stream Delay / Filter Operator                            */
/* ============================================================================================== */

#define SDELAY_EXT_PARAMS(X)                                                          \
    X(".delay", delay, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,   \
      "Delay (seconds if unit=0, frames if unit=1)")                                  \
    X(".unit", unit, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,         \
      "Delay/window unit (0=seconds, 1=frames)")                                      \
    X(".interp", interp, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,     \
      "Fractional interpolation (0=discrete, 1=linear)")                              \
    X(".avemode", avemode, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,   \
      "Window filter (0=off, 1=boxcar, 2=linear, 3=exp, 4=gauss)")                    \
    X(".window", window, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, \
      "Window width or FIR tap step (0=1 frame)")                                     \
    X(".buffsize", buffsize, FPTYPE_UINT64, 1000, FPFLAG_DEFAULT_INPUT,               \
      "Ring buffer depth in frames")

#define SDELAY_EXT_PORTS(X)                                                       \
    X(IN, frame, MILK_PORT_IMAGE, 0, "Input image/stream")                        \
    X(IN, weights, MILK_PORT_IMAGE, MILK_PORT_OPT, "Optional 1D FIR tap weights") \
    X(OUT, out, MILK_PORT_IMAGE, 0, "Delayed/filtered output image")

MILK_CALC_OP_DECLARE(streamdelay, SDELAY_EXT_PARAMS, SDELAY_EXT_PORTS);

/**
 * sdelay_ext_init - Allocate per-call-site `MILK_STREAM_DELAY_STATE` ring buffer
 * @ctx:   Operator context holding input `frame` geometry
 * @P:     Packed parameter struct (`buffsize` ring capacity)
 * @state: Output opaque state pointer
 *
 * Return: 0 on success, -1 on allocation failure
 */
static int sdelay_ext_init(milk_op_ctx *ctx, const struct streamdelay_par *P, void **state)
{
    const IMAGE *frame = MILK_IN(ctx, frame);
    if (frame == NULL)
    {
        return -1;
    }
    MILK_STREAM_DELAY_STATE *st = calloc(1, sizeof(*st));
    if (st == NULL)
    {
        return -1;
    }
    uint64_t cap = (P != NULL && P->buffsize > 0ULL) ? P->buffsize : 1000ULL;
    if (milk_stream_delay_init(st, frame, cap) != RETURN_SUCCESS)
    {
        free(st);
        return -1;
    }
    *state = st;
    return 0;
}

/**
 * sdelay_ext_exec - Push current `frame` and evaluate delayed/filtered output `out`
 * @ctx:   Operator context (`frame`, optional `weights`, destination `out`)
 * @P:     Packed live parameter struct
 * @state: Persistent `MILK_STREAM_DELAY_STATE` allocated by `sdelay_ext_init()`
 *
 * Return: 0 on success, -1 on execution failure
 */
static int sdelay_ext_exec(milk_op_ctx *ctx, const struct streamdelay_par *P, void *state)
{
    MILK_STREAM_DELAY_STATE *st      = (MILK_STREAM_DELAY_STATE *) state;
    const IMAGE             *frame   = MILK_IN(ctx, frame);
    const IMAGE             *weights = MILK_IN(ctx, weights);
    IMAGE                   *out     = MILK_OUT(ctx, out);
    if (st == NULL || frame == NULL || out == NULL)
    {
        return -1;
    }

    MILK_STREAM_DELAY_CFG cfg = {
        .delay      = (P != NULL) ? P->delay : 0.0,
        .window     = (P != NULL) ? P->window : 0.0,
        .unit_mode  = (P != NULL && P->unit != 0) ? MILK_SDELAY_UNIT_FRAME : MILK_SDELAY_UNIT_TIME,
        .interp     = (P != NULL && P->interp != 0) ? 1 : 0,
        .ave_mode   = (P != NULL) ? (int) P->avemode : 0,
        .naive_mode = 0,
        .force_push = 1,
        .allow_skip = 0,
    };

    int updated = 0;
    return (milk_stream_delay_step(st, frame, weights, &cfg, out, &updated) == RETURN_SUCCESS) ? 0
                                                                                               : -1;
}

/**
 * sdelay_ext_free - Release `MILK_STREAM_DELAY_STATE` ring-buffer memory
 * @state: Opaque pointer allocated by `sdelay_ext_init()`
 */
static void sdelay_ext_free(void *state)
{
    if (state != NULL)
    {
        milk_stream_delay_free((MILK_STREAM_DELAY_STATE *) state);
        free(state);
    }
}

MILK_CALC_MODULE_EXPORT(
    milkcalc_builtin,
    MILK_CALC_REG_IMG(bblur1pix, bblur1pix_ext_img, bblur1pix),
    MILK_CALC_REG_IMG(rotate2d, rotate2d_ext_img, rotate2d),
    MILK_CALC_REG_IMG(circsym, circsym_ext_img, circsym),
    MILK_CALC_REG_IMG(circasym, circasym_ext_img, circsym),
    MILK_CALC_REG_OP(streamdelay, streamdelay, sdelay_ext_init, sdelay_ext_exec, sdelay_ext_free));

/* ============================================================================================== */
/* Stream and Identifier Resolution Helpers                                                       */
/* ============================================================================================== */

/**
 * mc_resolve_or_autoconnect_image - Resolve an image by name or attach to `/dev/shm/<bare>.im.shm`
 * @name: Image identifier (optionally prefixed with `@S:` or `@L:`)
 *
 * Resolution steps:
 *   1. Parses any `@S:` or `@L:` prefix via `fps_streamname_parse()`.
 *   2. Looks up the bare name in `milk_data.image[]`. If found and backed by shared memory,
 *      checks via `stat()` whether the on-disk `.im.shm` inode changed (indicating an external
 *      process recreated the stream); if so, closes the stale mapping, attaches the fresh
 *      shared-memory image in place, and bumps `milk_table_generation()`.
 *   3. If not in `milk_data.image[]` and `@L:` was not forced, checks `/dev/shm/<bare>.im.shm` and
 *      auto-attaches the shared-memory stream into the next free `milk_data.image[]` slot.
 *
 * Return: Valid `imageID` (`>= 0`) on success, or -1 if not found
 */
long mc_resolve_or_autoconnect_image(const char *name)
{
    if (milk_data.image == NULL || milk_data.NB_MAX_IMAGE <= 0)
    {
        return -1;
    }
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(name);
    if (sp.error)
    {
        return -1;
    }
    const char *bare = (sp.name != NULL && sp.name[0] != '\0') ? sp.name : name;
    imageID     id   = image_ID(bare, milk_data.image, milk_data.NB_MAX_IMAGE);
    char        shmpath[512];
    struct stat st;

    if (id != -1)
    {
        IMAGE *im = &milk_data.image[id];
        if (im->used && im->md != NULL && im->md[0].shared == 1 &&
            ImageStreamIO_filename(shmpath, sizeof(shmpath), im->name) == IMAGESTREAMIO_SUCCESS &&
            stat(shmpath, &st) == 0 && st.st_ino != im->md[0].inode)
        {
            IMAGE fresh = { 0 };
            if (ImageStreamIO_read_sharedmem_image_toIMAGE(im->name, &fresh) ==
                IMAGESTREAMIO_SUCCESS)
            {
                ImageStreamIO_closeIm(im);
                memcpy(im, &fresh, sizeof(IMAGE));
                im->used = 1;
                milk_table_bump();
            }
        }
        return id;
    }
    if (sp.loc == 'L' ||
        ImageStreamIO_filename(shmpath, sizeof(shmpath), bare) != IMAGESTREAMIO_SUCCESS ||
        access(shmpath, R_OK) != 0)
    {
        return -1;
    }
    IMAGE tmp_im = { 0 };
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(bare, &tmp_im) == IMAGESTREAMIO_SUCCESS)
    {
        id = next_avail_image_ID(-1);
        if (id != -1)
        {
            memcpy(&milk_data.image[id], &tmp_im, sizeof(IMAGE));
            milk_data.image[id].used = 1;
            milk_table_bump();
        }
        else
        {
            ImageStreamIO_closeIm(&tmp_im);
        }
    }
    return id;
}

/**
 * mc_parse_ident_streamname - Expand `FPTYPE_STREAMNAME` variable aliases and parse `@S:`/`@L:`
 * @name: In/out buffer of size `MC_TOK_STR_MAX` holding the raw identifier (updated if aliased)
 * @bare: Output buffer of size `MC_IDENT_MAX` receiving the bare stream/variable name
 * @loc:  Optional output character receiving `'S'`, `'L'`, or `'\0'`
 *
 * When @name matches a bound `FPTYPE_STREAMNAME` variable (`type == 3`) in `milk_data.variable[]`,
 * replaces @name with the target stream name stored in that variable before stripping any `@S:` or
 * `@L:` storage prefix.
 *
 * Return: void
 */
void mc_parse_ident_streamname(char *name, char *bare, char *loc)
{
    long svid = variable_ID(name);
    if (svid != -1 && milk_data.variable[svid].type == 3 &&
        milk_data.variable[svid].value.s[0] != '\0')
    {
        snprintf(name, MC_TOK_STR_MAX, "%s", milk_data.variable[svid].value.s);
    }
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(name);
    if (loc != NULL)
    {
        *loc = sp.loc;
    }
    snprintf(bare, MC_IDENT_MAX, "%s",
             (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : name);
}

/**
 * mc_set_expr_from_im - Populate result metadata on AST node @e from an `IMAGE *`
 * @e:      AST expression node to annotate
 * @im_ptr: Pointer to `const IMAGE` providing datatype, axis count, and dimensions
 *
 * Return: void
 */
void mc_set_expr_from_im(mc_expr_t *e, const void *im_ptr)
{
    const IMAGE *im = (const IMAGE *) im_ptr;
    e->res_kind     = MC_RES_IMAGE;
    e->res_dt       = im->md[0].datatype;
    e->res_naxis    = im->md[0].naxis;
    e->res_size[0]  = (im->md[0].naxis > 0) ? im->md[0].size[0] : 1;
    e->res_size[1]  = (im->md[0].naxis > 1) ? im->md[0].size[1] : 1;
    e->res_size[2]  = (im->md[0].naxis > 2) ? im->md[0].size[2] : 1;
    e->needs_image  = 1;
}
