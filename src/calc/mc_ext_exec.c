// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_ext_exec.c
 * @brief Resolution, parameter packing, and runtime execution of `milk_calc_ext` kernels/operators.
 *
 * Bridges user-defined and built-in C extensions (`milk_calc_ext.h`) into the `libmilkcalc`
 * resolver (`mc_resolve.c`) and constructor executor (`mc_exec.c`):
 *   - **Call-site validation (`mc_ext_resolve_call`)**: Validates positional and keyword arguments
 *     against the extension's `milk_calc_param_set_t` and `milk_calc_op_ports_t` tables, enforces
 *     port arity and multi-LHS unpacking counts, and infers output `res_kind`, `res_dt`, and
 *     dimensions when static.
 *   - **Parameter struct packing (`pack_ext_params`)**: Assembles the 256-byte binary parameter
 *     buffer `pbuf` (`MILK_CALC_PARAM_BYTES`) for each invocation by binding keyword arguments,
 *     trailing positional scalar arguments, or live `<label>.<param>` FPS/calculator variables
 *     (falling back to `<ext_name>.<param>` or creating `<label>.<param>` with the default value).
 *   - **Tier 1 execution (`mc_exec_ext_ctor`)**:
 *       - `MILK_EXT_IMG`: Whole-image callback `img_fn(out, in, pbuf)`.
 *       - `MILK_EXT_STENCIL`: Parallel 2D row-strip execution via `arith_team_run()`, clamping
 *         `2*R + 1` input rows with horizontal border replication into a per-thread pad buffer.
 *       - `MILK_EXT_BLK1` / `MILK_EXT_BLK2`: Parallel 1D SIMD block execution over flat float32
 *         arrays across `arith_team_run()`.
 *   - **Tier 2 stateful operator execution (`exec_ext_op`, `mc_exec_multi_linalg_or_ext`)**:
 *     Caches operator state (`op_cache_state_t`) on `cs->ext_state`, re-running `op_init()` only
 *     when input dimensions or non-`FPFLAG_WRITERUN` parameters change, and binds multi-output
 *     image and scalar ports.
 */

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libfps/fps.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/arith_team.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/create_variable.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/variable_ID.h"
#include "linalg/mc_linalg.h"
#include "milk_calc_ext.h"

int mc_exec_multi_svd_or_eigh(mcalc_prog_t    *prog,
                              const mc_stmt_t *s,
                              int              in_unit,
                              char            *err,
                              size_t           errsz);

/* ============================================================================================== */
/* Diagnostics and Call-Site Resolution                                                           */
/* ============================================================================================== */

/**
 * ext_err - Format a diagnostic message into @err if not already populated
 * @err:   Destination error buffer (may be NULL)
 * @errsz: Capacity of @err in bytes
 * @fmt:   printf-style format string
 *
 * Return: Always -1 for convenient tail-return error propagation
 */
__attribute__((format(printf, 3, 4))) static int ext_err(char       *err,
                                                         size_t      errsz,
                                                         const char *fmt,
                                                         ...)
{
    if (err != NULL && errsz > 0 && err[0] == '\0')
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errsz, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/**
 * strip_dot - Skip a leading `'.'` on a parameter name specification
 * @s: Parameter or keyword string (may be NULL)
 *
 * Extension parameter specs may be declared with a leading dot (e.g. `".sigma"`); this helper
 * normalizes both `".sigma"` and `"sigma"` to `"sigma"`.
 *
 * Return: Pointer to the normalized parameter name (never NULL)
 */
static const char *strip_dot(const char *s)
{
    return (s != NULL && s[0] == '.') ? (s + 1) : (s != NULL ? s : "");
}

/**
 * mc_ext_resolve_call - Validate and type-check an extension call AST node during resolution
 * @e:     `MC_EXPR_CALL` node with `e->u.call.ext_id >= 0`
 * @nlhs:  Number of LHS targets unpacking the call result (`1` for normal expressions)
 * @err:   Error message buffer
 * @errsz: Capacity of @err in bytes
 *
 * Checks that all keyword arguments match either a declared parameter in `ext->get_params()` or a
 * named input port in `ext->get_ports()`, verifies required positional input counts and types,
 * validates @nlhs against `pts->nout`, and populates `e->res_kind`, `e->res_dt`, and `e->res_size`.
 *
 * Return: 0 on success, -1 on validation error
 */
int mc_ext_resolve_call(mc_expr_t *e, int nlhs, char *err, size_t errsz)
{
    const milk_calc_ext_entry_t *ext = milk_calc_ext_get(e->u.call.ext_id);
    if (ext == NULL)
    {
        return ext_err(err, errsz, "unknown extension '%s'", e->u.call.name);
    }
    const milk_calc_param_set_t *pset = ext->get_params ? ext->get_params() : NULL;
    const milk_calc_op_ports_t  *pts  = ext->get_ports ? ext->get_ports() : NULL;

    for (int i = 0; i < e->u.call.nargs; i++)
    {
        if (e->u.call.kw_names[i][0] == '\0')
        {
            continue;
        }
        const char *kw    = strip_dot(e->u.call.kw_names[i]);
        int         found = 0;
        for (int p = 0; pset != NULL && p < pset->nparams; p++)
        {
            if (strcmp(strip_dot(pset->specs[p].name), kw) == 0)
            {
                found = 1;
                break;
            }
        }
        for (int p = 0; !found && pts != NULL && p < pts->nin; p++)
        {
            if (milk_calc_port_match_(ext->name, pts->in_ports[p].name, kw))
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            return ext_err(err, errsz, "%s: unknown keyword argument '%s'", ext->name, kw);
        }
    }

    if (ext->kind == MILK_EXT_OP && pts != NULL && nlhs > pts->nout)
    {
        return ext_err(err, errsz, "%s: too many LHS targets (%d > %d)", ext->name, nlhs,
                       pts->nout);
    }
    if (ext->kind != MILK_EXT_OP && nlhs > 1)
    {
        return ext_err(err, errsz, "%s: single-output kernel cannot unpack into %d targets",
                       ext->name, nlhs);
    }

    int in_arg[MILK_CALC_MAX_PORTS];
    int par_arg[MILK_CALC_MAX_PARAMS];
    int nin =
        mc_ext_bind_args(e->u.call.ext_id, e->u.call.nargs, (const char (*)[64]) e->u.call.kw_names,
                         in_arg, par_arg, err, errsz);
    if (nin < 0)
    {
        return -1;
    }
    for (int p = 0; p < nin; p++)
    {
        int is_op = (ext->kind == MILK_EXT_OP && pts != NULL);
        if (in_arg[p] < 0)
        {
            if (is_op && (pts->in_ports[p].flags & MILK_PORT_OPT))
            {
                continue;
            }
            return is_op ? ext_err(err, errsz, "%s: missing required input port '%s'", ext->name,
                                   pts->in_ports[p].name)
                         : ext_err(err, errsz, "%s: expected %d image argument(s)", ext->name, nin);
        }
        if (e->u.call.args[in_arg[p]]->res_kind != MC_RES_IMAGE &&
            !(is_op && pts->in_ports[p].kind == MILK_PORT_SCALAR))
        {
            return ext_err(err, errsz, "%s: argument %d must be an image", ext->name,
                           in_arg[p] + 1);
        }
    }
    for (int k = 0; pset != NULL && k < pset->nparams; k++)
    {
        if (par_arg[k] >= 0 && e->u.call.args[par_arg[k]]->res_kind == MC_RES_IMAGE)
        {
            return ext_err(err, errsz, "%s: parameter %s takes a number, not an image", ext->name,
                           strip_dot(pset->specs[k].name));
        }
    }

    e->needs_image       = 1;
    e->res_kind          = MC_RES_IMAGE;
    const mc_expr_t *im0 = (ext->kind != MILK_EXT_OP) ? e->u.call.args[in_arg[0]] : NULL;
    e->res_dt =
        (im0 != NULL && strcmp(ext->name, "rotate2d") == 0 && im0->res_dt == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    if (im0 != NULL && im0->res_naxis > 0)
    {
        e->res_naxis = im0->res_naxis;
        memcpy(e->res_size, im0->res_size, sizeof(e->res_size));
    }
    return 0;
}

/* ============================================================================================== */
/* Binary Parameter Struct Packing and Float32 Input Conversion                                   */
/* ============================================================================================== */

/**
 * write_param_field - Store a numeric value @val into field @sp of packed parameter buffer @pbuf
 * @pbuf: Destination 256-byte parameter struct buffer
 * @sp:   Parameter specification (`fptype`, `offset`, `size`)
 * @val:  Evaluated double-precision parameter value
 *
 * Converts @val to the C type matching `sp->fptype` (`int64_t`, `uint64_t`, `int32_t`, `uint32_t`,
 * `float`, or `double`) and copies it to `pbuf + sp->offset`.
 *
 * Return: void
 */
static void write_param_field(uint8_t *pbuf, const milk_calc_param_spec_t *sp, double val)
{
    if (sp->offset + sp->size > MILK_CALC_PARAM_BYTES)
    {
        return;
    }
    void *dst = pbuf + sp->offset;
    switch (sp->fptype)
    {
    case FPTYPE_INT64:
    {
        int64_t v = (int64_t) llround(val);
        memcpy(dst, &v, sizeof(v));
        break;
    }
    case FPTYPE_UINT64:
    {
        uint64_t v = (val <= 0.0) ? 0 : (uint64_t) llround(val);
        memcpy(dst, &v, sizeof(v));
        break;
    }
    case FPTYPE_INT32:
    case FPTYPE_ONOFF:
    {
        int32_t v = (int32_t) llround(val);
        memcpy(dst, &v, sizeof(v));
        break;
    }
    case FPTYPE_UINT32:
    {
        uint32_t v = (val <= 0.0) ? 0 : (uint32_t) llround(val);
        memcpy(dst, &v, sizeof(v));
        break;
    }
    case FPTYPE_FLOAT32:
    {
        float v = (float) val;
        memcpy(dst, &v, sizeof(v));
        break;
    }
    default:
        memcpy(dst, &val, sizeof(val));
        break;
    }
}

/**
 * lookup_calc_var_d - Read a numeric variable from `milk_data.variable[]` as double
 * @name: Variable identifier to look up (e.g. `"myop.gain"`)
 * @out:  Output pointer receiving the numeric value if found
 *
 * Return: 1 if @name exists and holds a float or integer value, 0 otherwise
 */
static int lookup_calc_var_d(const char *name, double *out)
{
    long vid = variable_ID(name);
    if (vid < 0 || !milk_data.variable[vid].used)
    {
        return 0;
    }
    *out = (milk_data.variable[vid].type == 1)   ? (double) milk_data.variable[vid].value.l
           : (milk_data.variable[vid].type == 0) ? milk_data.variable[vid].value.f
                                                 : 0.0;
    return milk_data.variable[vid].type == 0 || milk_data.variable[vid].type == 1;
}

/**
 * pack_ext_params - Populate the binary parameter struct buffer @pbuf for an extension call
 * @ext:     Registered extension entry
 * @rhs:     Extension `MC_EXPR_CALL` AST node
 * @par_arg: The argument that binds each parameter, or -1 (mc_ext_bind_args())
 * @pbuf:    Output buffer of size `MILK_CALC_PARAM_BYTES`
 * @err:     Error message buffer
 * @errsz:   Capacity of @err in bytes
 *
 * A parameter takes the value of its argument; without one, the live variable `<label>.<pname>`
 * (or `<ext->name>.<pname>` if `<label>` is an instance tag from `ext:label(...)`), created with
 * `sp->def_num` if absent.
 *
 * Return: 0 on success, -1 if evaluating a scalar argument expression fails
 */
static int pack_ext_params(const milk_calc_ext_entry_t *ext,
                           const mc_expr_t             *rhs,
                           const int                   *par_arg,
                           uint8_t                     *pbuf,
                           char                        *err,
                           size_t                       errsz)
{
    memset(pbuf, 0, MILK_CALC_PARAM_BYTES);
    const milk_calc_param_set_t *pset = ext->get_params ? ext->get_params() : NULL;
    if (pset == NULL || pset->nparams <= 0)
    {
        return 0;
    }

    const char *colon = strchr(rhs->u.call.name, ':');
    const char *label = (colon != NULL && colon[1] != '\0') ? (colon + 1) : ext->name;

    for (int p = 0; p < pset->nparams; p++)
    {
        const milk_calc_param_spec_t *sp    = &pset->specs[p];
        const char                   *pname = strip_dot(sp->name);
        double                        val   = sp->def_num;

        if (par_arg[p] >= 0)
        {
            if (mc_eval_scalar_d(rhs->u.call.args[par_arg[p]], &val, err, errsz) != 0)
            {
                return -1;
            }
        }
        else
        {
            char vname[128];
            char fbname[128];
            snprintf(vname, sizeof(vname), "%s.%s", label, pname);
            if (!lookup_calc_var_d(vname, &val))
            {
                int found_fb = 0;
                if (strcmp(label, ext->name) != 0)
                {
                    snprintf(fbname, sizeof(fbname), "%s.%s", ext->name, pname);
                    found_fb = lookup_calc_var_d(fbname, &val);
                }
                if (!found_fb)
                {
                    create_variable_ID(vname, sp->def_num);
                }
            }
        }
        write_param_field(pbuf, sp, val);
    }
    return 0;
}

/**
 * ensure_f32_im - Return a `_DATATYPE_FLOAT` `imageID` for @src_id, converting if necessary
 * @src_id:  Input `imageID` in `milk_data.image[]`
 * @out_tmp: Output flag set to 1 if a temporary float32 image was allocated, 0 if @src_id is used
 *
 * Tier 1 and Tier 2 extension callbacks operate on `float` buffers by default; non-float inputs
 * are converted into a pooled temporary `_tmpcalcN` image.
 *
 * Return: Float32 `imageID` on success, or -1 on allocation failure
 */
static imageID ensure_f32_im(imageID src_id, int *out_tmp)
{
    const IMAGE *src = &milk_data.image[src_id];
    if (src->md[0].datatype == _DATATYPE_FLOAT)
    {
        *out_tmp = 0;
        return src_id;
    }
    imageID tid = mc_tmp_image(src->md[0].naxis, src->md[0].size, _DATATYPE_FLOAT);
    if (tid >= 0)
    {
        IMAGE   *dst = &milk_data.image[tid];
        uint8_t  sdt = src->md[0].datatype;
        uint64_t nel = src->md[0].nelement;
        for (uint64_t i = 0; i < nel; i++)
        {
            dst->array.F[i] = (float) arith_read_im_pix(src, sdt, i);
        }
        *out_tmp = 1;
    }
    return tid;
}

/* ============================================================================================== */
/* Parallel Worker Team Callbacks for Tier 1 Block and Stencil Kernels                            */
/* ============================================================================================== */

/**
 * struct ext_team_job_t - Shared context for parallel `MILK_EXT_BLK*` and `MILK_EXT_STENCIL` jobs
 * @ext:  Extension entry being executed
 * @in0:  Primary float32 input pixel array
 * @in1:  Secondary float32 input pixel array (`MILK_EXT_BLK2` only)
 * @out:  Destination float32 output pixel array
 * @pbuf: Packed binary parameter struct buffer
 * @nx:   Image width (`size[0]`, used by stencil kernels)
 * @ny:   Image height (`size[1]`, used by stencil kernels)
 */
typedef struct
{
    const milk_calc_ext_entry_t *ext;
    const float                 *in0;
    const float                 *in1;
    float                       *out;
    const uint8_t               *pbuf;
    uint32_t                     nx, ny;
} ext_team_job_t;

/**
 * blk_team_cb - Worker callback invoking a 1D block kernel over element range `[lo, hi)`
 * @ctx: Pointer to `ext_team_job_t`
 * @tid: Worker thread index (unused)
 * @nth: Total number of worker threads (unused)
 * @lo:  Starting linear pixel index (inclusive)
 * @hi:  Ending linear pixel index (exclusive)
 *
 * Return: void
 */
static void blk_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) tid;
    (void) nth;
    const ext_team_job_t *j = (const ext_team_job_t *) ctx;
    if (hi <= lo)
    {
        return;
    }
    if (j->ext->kind == MILK_EXT_BLK1)
    {
        j->ext->blk1_fn(j->out + lo, j->in0 + lo, j->pbuf, (int) (hi - lo));
    }
    else
    {
        j->ext->blk2_fn(j->out + lo, j->in0 + lo, j->in1 + lo, j->pbuf, (int) (hi - lo));
    }
}

/**
 * stencil_team_cb - Worker callback invoking a 2D stencil kernel over image rows `[lo, hi)`
 * @ctx: Pointer to `ext_team_job_t`
 * @tid: Worker thread index (unused)
 * @nth: Total number of worker threads (unused)
 * @lo:  Starting image row `y` (inclusive)
 * @hi:  Ending image row `y` (exclusive)
 *
 * Allocates a thread-local padded strip of `2*R + 1` rows of width `nx + 2*R`, clamps vertical
 * (`cy`) and horizontal (`[0..R-1]` and `[R+nx..nx+2*R-1]`) borders by edge replication, and
 * passes `rows[0..2*R]` to `ext->stencil_fn()`.
 *
 * Return: void
 */
static void stencil_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) tid;
    (void) nth;
    const ext_team_job_t *j   = (const ext_team_job_t *) ctx;
    int                   R   = j->ext->radius;
    int                   nr  = 2 * R + 1;
    uint32_t              nx  = j->nx;
    uint32_t              ny  = j->ny;
    uint32_t              pw  = nx + 2 * (uint32_t) R;
    float                *pad = malloc(sizeof(float) * (size_t) nr * (size_t) pw);
    const float          *rows[15];
    if (pad == NULL)
    {
        return;
    }
    for (int k = 0; k < nr; k++)
    {
        rows[k] = pad + (size_t) k * pw;
    }
    for (uint64_t y = lo; y < hi; y++)
    {
        for (int dy = -R; dy <= R; dy++)
        {
            int          sy  = (int) y + dy;
            uint32_t     cy  = (sy < 0) ? 0 : ((uint32_t) sy >= ny) ? (ny - 1) : (uint32_t) sy;
            const float *src = j->in0 + (uint64_t) cy * nx;
            float       *dst = pad + (size_t) (dy + R) * pw;
            for (int l = 0; l < R; l++)
            {
                dst[l] = src[0];
            }
            memcpy(dst + R, src, sizeof(float) * nx);
            for (int r = 0; r < R; r++)
            {
                dst[(uint32_t) R + nx + (uint32_t) r] = src[nx - 1];
            }
        }
        j->ext->stencil_fn(j->out + y * nx, rows, j->pbuf, (int) nx);
    }
    free(pad);
}

/* ============================================================================================== */
/* Tier 2 Stateful Operator Caching and Execution                                                 */
/* ============================================================================================== */

/**
 * struct op_cache_state_t - Cached state and geometry signature for a Tier 2 `MILK_EXT_OP` step
 * @ext:         Registered operator entry
 * @op_state:    Opaque state pointer allocated by `ext->op_init()`
 * @state_valid: 1 if `op_init()` has succeeded and `op_state` is live
 * @last_params: Snapshot of packed parameter bytes at last `op_init()`
 * @last_in_sz:  Per-input-port `[naxis, size[0], size[1], size[2]]` at last `op_init()`
 * @out_naxis:   Cached output port axis counts set by `op_init()`
 * @out_size:    Cached output port dimensions set by `op_init()`
 * @out_dtype:   Cached output port datatypes set by `op_init()`
 */
typedef struct
{
    const milk_calc_ext_entry_t *ext;
    void                        *op_state;
    int                          state_valid;
    uint8_t                      last_params[MILK_CALC_PARAM_BYTES];
    uint32_t                     last_in_sz[MILK_CALC_MAX_PORTS][4];
    uint8_t                      out_naxis[MILK_CALC_MAX_PORTS];
    uint32_t                     out_size[MILK_CALC_MAX_PORTS][3];
    uint8_t                      out_dtype[MILK_CALC_MAX_PORTS];
} op_cache_state_t;

/**
 * free_op_cache_state - Destructor callback attached to `cs->ext_state_free`
 * @ptr: Pointer to `op_cache_state_t`
 *
 * Invokes `ext->op_free(st->op_state)` if initialized and frees the wrapper struct.
 *
 * Return: void
 */
static void free_op_cache_state(void *ptr)
{
    op_cache_state_t *st = (op_cache_state_t *) ptr;
    if (st == NULL)
    {
        return;
    }
    if (st->state_valid && st->ext != NULL && st->ext->op_free != NULL)
    {
        st->ext->op_free(st->op_state);
    }
    free(st);
}

/**
 * exec_ext_op - Execute a Tier 2 multi-port stateful operator (`MILK_EXT_OP`)
 * @prog:      Enclosing planned program (may be NULL for one-shot scalar evaluations)
 * @cs:        Constructor step holding persistent `ext_state` cache (may be NULL)
 * @ext:       Registered operator descriptor
 * @rhs:       Operator `MC_EXPR_CALL` AST node
 * @multi_s:   Enclosing `MC_STMT_MULTI_ASSIGN` statement when unpacking multiple outputs, or NULL
 * @avoid_id:  Output `imageID` to avoid input aliasing with, or -1
 * @in_unit:   1 when running inside a `milk.calc` unit
 * @out_val:   Optional pointer receiving the last scalar output port value
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Binds all input image and scalar ports (by keyword or position), packs parameters, checks if
 * `op_init()` must be called (on first run, input shape change, or init-time parameter change),
 * prepares all output image ports (allocating temporary images for any unbound trailing ports),
 * calls `ext->op_exec()`, and posts/stores all output ports.
 *
 * Return: 0 on success, -1 on error
 */
static int exec_ext_op(mcalc_prog_t                *prog,
                       mc_ctor_step_t              *cs,
                       const milk_calc_ext_entry_t *ext,
                       const mc_expr_t             *rhs,
                       const mc_stmt_t             *multi_s,
                       imageID                      avoid_id,
                       int                          in_unit,
                       mc_val_t                    *out_val,
                       char                        *err,
                       size_t                       errsz)
{
    const milk_calc_op_ports_t  *pts  = ext->get_ports ? ext->get_ports() : NULL;
    const milk_calc_param_set_t *pset = ext->get_params ? ext->get_params() : NULL;
    if (pts == NULL)
    {
        return ext_err(err, errsz, "%s: missing operator port table", ext->name);
    }

    milk_calc_op_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.op_name = ext->name;
    ctx.ports   = pts;

    imageID in_ids[MILK_CALC_MAX_PORTS];
    int     in_tmp[MILK_CALC_MAX_PORTS];
    int     f32_tmp[MILK_CALC_MAX_PORTS];
    imageID f32_ids[MILK_CALC_MAX_PORTS];
    int     in_arg[MILK_CALC_MAX_PORTS];
    int     par_arg[MILK_CALC_MAX_PARAMS];
    int     rc = 0;

    for (int p = 0; p < MILK_CALC_MAX_PORTS; p++)
    {
        in_ids[p]  = -1;
        f32_ids[p] = -1;
        in_tmp[p]  = 0;
        f32_tmp[p] = 0;
    }
    if (mc_ext_bind_args(rhs->u.call.ext_id, rhs->u.call.nargs,
                         (const char (*)[64]) rhs->u.call.kw_names, in_arg, par_arg, err,
                         errsz) < 0)
    {
        return -1;
    }

    for (int p = 0; p < pts->nin; p++)
    {
        const mc_expr_t *arg = (in_arg[p] >= 0) ? rhs->u.call.args[in_arg[p]] : NULL;
        if (arg == NULL)
        {
            if (pts->in_ports[p].flags & MILK_PORT_OPT)
            {
                continue;
            }
            rc = ext_err(err, errsz, "%s: missing required input port '%s'", ext->name,
                         pts->in_ports[p].name);
            goto done;
        }
        if (pts->in_ports[p].kind == MILK_PORT_SCALAR)
        {
            if (mc_eval_scalar_d(arg, &ctx.in_scalar[p], err, errsz) != 0)
            {
                rc = -1;
                goto done;
            }
        }
        else
        {
            if (mc_resolve_or_eval_im(arg, avoid_id, in_unit, &in_ids[p], &in_tmp[p], err, errsz) !=
                0)
            {
                rc = -1;
                goto done;
            }
            f32_ids[p] = ensure_f32_im(in_ids[p], &f32_tmp[p]);
            if (f32_ids[p] < 0)
            {
                rc = ext_err(err, errsz, "%s: failed to convert input port '%s'", ext->name,
                             pts->in_ports[p].name);
                goto done;
            }
            ctx.in_img[p] = &milk_data.image[f32_ids[p]];
        }
    }

    uint8_t pbuf[MILK_CALC_PARAM_BYTES];
    if (pack_ext_params(ext, rhs, par_arg, pbuf, err, errsz) != 0)
    {
        rc = -1;
        goto done;
    }

    op_cache_state_t  local_st = { .ext = ext };
    op_cache_state_t *st       = &local_st;
    if (cs != NULL)
    {
        if (cs->ext_state == NULL)
        {
            cs->ext_state      = calloc(1, sizeof(op_cache_state_t));
            cs->ext_state_free = free_op_cache_state;
        }
        if (cs->ext_state != NULL)
        {
            st      = (op_cache_state_t *) cs->ext_state;
            st->ext = ext;
        }
    }

    /* Check whether input shapes or init-time parameters changed since last `op_init()`. */
    int need_reinit = !st->state_valid;
    for (int p = 0; !need_reinit && p < pts->nin; p++)
    {
        const IMAGE *im     = ctx.in_img[p];
        uint32_t     cur[4] = {
            im ? im->md[0].naxis : 0,
            im ? im->md[0].size[0] : 0,
            (im && im->md[0].naxis > 1) ? im->md[0].size[1] : 0,
            (im && im->md[0].naxis > 2) ? im->md[0].size[2] : 0,
        };
        if (memcmp(cur, st->last_in_sz[p], sizeof(cur)) != 0)
        {
            need_reinit = 1;
        }
    }
    for (int p = 0; !need_reinit && pset != NULL && p < pset->nparams; p++)
    {
        const milk_calc_param_spec_t *sp = &pset->specs[p];
        if (!(sp->fpflags & FPFLAG_WRITERUN) &&
            memcmp(pbuf + sp->offset, st->last_params + sp->offset, sp->size) != 0)
        {
            need_reinit = 1;
        }
    }

    if (need_reinit)
    {
        if (st->state_valid && ext->op_free != NULL)
        {
            ext->op_free(st->op_state);
            st->op_state    = NULL;
            st->state_valid = 0;
        }
        const IMAGE *im0 = ctx.in_img[0];
        for (int p = 0; p < pts->nout; p++)
        {
            ctx.out_naxis[p]   = im0 ? im0->md[0].naxis : 2;
            ctx.out_size[p][0] = im0 ? im0->md[0].size[0] : 1;
            ctx.out_size[p][1] = (im0 && im0->md[0].naxis > 1) ? im0->md[0].size[1] : 1;
            ctx.out_size[p][2] = (im0 && im0->md[0].naxis > 2) ? im0->md[0].size[2] : 1;
            ctx.out_dtype[p]   = _DATATYPE_FLOAT;
        }
        if (ext->op_init != NULL && ext->op_init(&ctx, pbuf, &st->op_state) != 0)
        {
            rc = ext_err(err, errsz, "user operator '%s' init failed", ext->name);
            goto done;
        }
        st->state_valid = 1;
        memcpy(st->last_params, pbuf, sizeof(pbuf));
        for (int p = 0; p < pts->nin; p++)
        {
            const IMAGE *im      = ctx.in_img[p];
            st->last_in_sz[p][0] = im ? im->md[0].naxis : 0;
            st->last_in_sz[p][1] = im ? im->md[0].size[0] : 0;
            st->last_in_sz[p][2] = (im && im->md[0].naxis > 1) ? im->md[0].size[1] : 0;
            st->last_in_sz[p][3] = (im && im->md[0].naxis > 2) ? im->md[0].size[2] : 0;
        }
        memcpy(st->out_naxis, ctx.out_naxis, sizeof(st->out_naxis));
        memcpy(st->out_size, ctx.out_size, sizeof(st->out_size));
        memcpy(st->out_dtype, ctx.out_dtype, sizeof(st->out_dtype));
    }
    else
    {
        memcpy(ctx.out_naxis, st->out_naxis, sizeof(ctx.out_naxis));
        memcpy(ctx.out_size, st->out_size, sizeof(st->out_size));
        memcpy(ctx.out_dtype, st->out_dtype, sizeof(st->out_dtype));
    }

    int     nlhs = (multi_s != NULL) ? multi_s->u.multi.nlhs : 1;
    imageID out_ids[MILK_CALC_MAX_PORTS];
    int     out_is_tmp[MILK_CALC_MAX_PORTS];
    for (int p = 0; p < pts->nout; p++)
    {
        out_ids[p]    = -1;
        out_is_tmp[p] = 0;
        if (pts->out_ports[p].kind != MILK_PORT_IMAGE)
        {
            continue;
        }
        uint8_t dt = ctx.out_dtype[p] ? ctx.out_dtype[p] : _DATATYPE_FLOAT;
        if (p < nlhs)
        {
            mc_slot_t  tmp_slot = { 0 };
            mc_slot_t *sl       = NULL;
            if (multi_s == NULL && cs != NULL && prog != NULL)
            {
                sl = &prog->slots[cs->slot_idx];
            }
            else if (multi_s != NULL)
            {
                const mc_expr_t *lhs = multi_s->u.multi.lhs[p];
                for (int k = 0; prog != NULL && k < prog->nslots; k++)
                {
                    if (strcmp(prog->slots[k].bare, lhs->u.ident.bare) == 0)
                    {
                        sl = &prog->slots[k];
                        break;
                    }
                }
                if (sl == NULL)
                {
                    snprintf(tmp_slot.name, sizeof(tmp_slot.name), "%s", lhs->u.ident.name);
                    snprintf(tmp_slot.bare, sizeof(tmp_slot.bare), "%s", lhs->u.ident.bare);
                    tmp_slot.loc   = lhs->u.ident.loc;
                    tmp_slot.im_id = mc_resolve_or_autoconnect_image(lhs->u.ident.name);
                    sl             = &tmp_slot;
                }
            }
            out_ids[p] =
                mc_out_prepare(sl, ctx.out_naxis[p], ctx.out_size[p], dt, in_unit, err, errsz);
        }
        else
        {
            out_ids[p]    = mc_tmp_image(ctx.out_naxis[p], ctx.out_size[p], dt);
            out_is_tmp[p] = (out_ids[p] >= 0);
        }
        if (out_ids[p] < 0)
        {
            rc = -1;
            goto done_out;
        }
        ctx.out_img[p] = &milk_data.image[out_ids[p]];
    }

    if (ext->op_exec != NULL && ext->op_exec(&ctx, pbuf, st->op_state) != 0)
    {
        rc = ext_err(err, errsz, "user operator '%s' exec failed", ext->name);
        goto done_out;
    }

    for (int p = 0; p < pts->nout; p++)
    {
        if (pts->out_ports[p].kind == MILK_PORT_IMAGE)
        {
            if (!out_is_tmp[p] && out_ids[p] >= 0)
            {
                mc_out_post(out_ids[p]);
            }
        }
        else if (multi_s != NULL && p < nlhs)
        {
            mc_expr_t *lhs = multi_s->u.multi.lhs[p];
            mc_val_t   sv  = { .type = MC_VAL_DOUBLE, .dval = ctx.out_scalar[p] };
            mc_store_scalar_var(lhs->u.ident.bare[0] != '\0' ? lhs->u.ident.bare
                                                             : lhs->u.ident.name,
                                &lhs->u.ident.var_id, &sv);
            if (out_val != NULL)
            {
                *out_val = sv;
            }
        }
    }

done_out:
    for (int p = 0; p < pts->nout; p++)
    {
        if (out_is_tmp[p] && out_ids[p] >= 0)
        {
            mc_tmp_release(out_ids[p]);
        }
    }
    if (st == &local_st && local_st.state_valid && ext->op_free != NULL)
    {
        ext->op_free(local_st.op_state);
    }

done:
    for (int p = 0; p < MILK_CALC_MAX_PORTS; p++)
    {
        if (f32_tmp[p] && f32_ids[p] >= 0)
        {
            mc_tmp_release(f32_ids[p]);
        }
        if (in_tmp[p] && in_ids[p] >= 0)
        {
            mc_tmp_release(in_ids[p]);
        }
    }
    return rc;
}

/* ============================================================================================== */
/* Public Constructor and Multi-Assign Extension Entry Points                                     */
/* ============================================================================================== */

/**
 * mc_exec_ext_ctor - Execute a single-output C extension constructor step (`MC_STEP_CTOR`)
 * @prog:      Enclosing planned program
 * @cs:        Constructor step descriptor
 * @avoid_id:  Output `imageID` to avoid aliasing with when evaluating inputs, or -1
 * @in_unit:   1 when running inside a `milk.calc` unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Delegates Tier 2 operators (`MILK_EXT_OP`) to `exec_ext_op()` and executes Tier 1 kernels
 * (`MILK_EXT_IMG`, `MILK_EXT_STENCIL`, `MILK_EXT_BLK1`, `MILK_EXT_BLK2`).
 *
 * Return: 0 on success, -1 on error
 */
int mc_exec_ext_ctor(mcalc_prog_t   *prog,
                     mc_ctor_step_t *cs,
                     imageID         avoid_id,
                     int             in_unit,
                     char           *err,
                     size_t          errsz)
{
    mc_slot_t                   *slot = &prog->slots[cs->slot_idx];
    const mc_expr_t             *rhs  = cs->rhs_expr;
    const milk_calc_ext_entry_t *ext  = milk_calc_ext_get(rhs->u.call.ext_id);
    if (ext == NULL)
    {
        return ext_err(err, errsz, "unknown extension '%s'", rhs->u.call.name);
    }
    if (ext->kind == MILK_EXT_OP)
    {
        return exec_ext_op(prog, cs, ext, rhs, NULL, avoid_id, in_unit, NULL, err, errsz);
    }

    int in_arg[MILK_CALC_MAX_PORTS];
    int par_arg[MILK_CALC_MAX_PARAMS];
    int n_img =
        mc_ext_bind_args(rhs->u.call.ext_id, rhs->u.call.nargs,
                         (const char (*)[64]) rhs->u.call.kw_names, in_arg, par_arg, err, errsz);
    imageID id0   = -1;
    imageID id1   = -1;
    imageID f0    = -1;
    imageID f1    = -1;
    int     tmp0  = 0;
    int     tmp1  = 0;
    int     ftmp0 = 0;
    int     ftmp1 = 0;
    int     rc    = 0;

    if (n_img < 0 ||
        mc_resolve_or_eval_im(rhs->u.call.args[in_arg[0]], avoid_id, in_unit, &id0, &tmp0, err,
                              errsz) != 0 ||
        (n_img == 2 && mc_resolve_or_eval_im(rhs->u.call.args[in_arg[1]], avoid_id, in_unit, &id1,
                                             &tmp1, err, errsz) != 0))
    {
        if (tmp0)
        {
            mc_tmp_release(id0);
        }
        return -1;
    }

    uint8_t pbuf[MILK_CALC_PARAM_BYTES];
    if (pack_ext_params(ext, rhs, par_arg, pbuf, err, errsz) != 0)
    {
        rc = -1;
        goto done;
    }

    const IMAGE *im0 = &milk_data.image[id0];
    if (n_img == 2 && milk_data.image[id1].md[0].nelement != im0->md[0].nelement)
    {
        rc = ext_err(err, errsz, "%s: input image element count mismatch", ext->name);
        goto done;
    }
    if (ext->kind == MILK_EXT_STENCIL && im0->md[0].naxis != 2)
    {
        rc = ext_err(err, errsz, "%s: stencil kernel requires a 2D image", ext->name);
        goto done;
    }

    int keep_f64 = (ext->kind == MILK_EXT_IMG && strcmp(ext->name, "rotate2d") == 0 &&
                    im0->md[0].datatype == _DATATYPE_DOUBLE);
    f0           = keep_f64 ? id0 : ensure_f32_im(id0, &ftmp0);
    f1           = (n_img == 2) ? ensure_f32_im(id1, &ftmp1) : -1;
    if (f0 < 0 || (n_img == 2 && f1 < 0))
    {
        rc = ext_err(err, errsz, "%s: failed to prepare float32 input buffer", ext->name);
        goto done;
    }

    uint32_t sz[3] = {
        im0->md[0].size[0],
        (im0->md[0].naxis > 1) ? im0->md[0].size[1] : 1,
        (im0->md[0].naxis > 2) ? im0->md[0].size[2] : 1,
    };
    uint8_t out_dt = keep_f64 ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    imageID oid    = mc_out_prepare(slot, im0->md[0].naxis, sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        rc = -1;
        goto done;
    }
    IMAGE *out = &milk_data.image[oid];

    if (ext->kind == MILK_EXT_IMG)
    {
        if (ext->img_fn(out, &milk_data.image[f0], pbuf) != 0)
        {
            rc = ext_err(err, errsz, "user image kernel '%s' failed", ext->name);
            goto done;
        }
    }
    else if (ext->kind == MILK_EXT_STENCIL)
    {
        ext_team_job_t job = {
            .ext  = ext,
            .in0  = milk_data.image[f0].array.F,
            .out  = out->array.F,
            .pbuf = pbuf,
            .nx   = sz[0],
            .ny   = sz[1],
        };
        int nth = milk_omp_team_size(
            im0->md[0].nelement * (uint64_t) (ext->cost > 0 ? ext->cost : 4), 262144);
        if (nth <= 1)
        {
            stencil_team_cb(&job, 0, 1, 0, sz[1]);
        }
        else
        {
            arith_team_run(nth, sz[1], stencil_team_cb, &job);
        }
    }
    else
    {
        ext_team_job_t job = {
            .ext  = ext,
            .in0  = milk_data.image[f0].array.F,
            .in1  = (n_img == 2) ? milk_data.image[f1].array.F : NULL,
            .out  = out->array.F,
            .pbuf = pbuf,
        };
        uint64_t nel = im0->md[0].nelement;
        int      nth = milk_omp_team_size(nel * (uint64_t) (ext->cost > 0 ? ext->cost : 2), 262144);
        if (nth <= 1)
        {
            blk_team_cb(&job, 0, 1, 0, nel);
        }
        else
        {
            arith_team_run(nth, nel, blk_team_cb, &job);
        }
    }
    mc_out_post(oid);

done:
    if (ftmp0 && f0 >= 0)
    {
        mc_tmp_release(f0);
    }
    if (ftmp1 && f1 >= 0)
    {
        mc_tmp_release(f1);
    }
    if (tmp0 && id0 >= 0)
    {
        mc_tmp_release(id0);
    }
    if (tmp1 && id1 >= 0)
    {
        mc_tmp_release(id1);
    }
    return rc;
}

/**
 * mc_exec_multi_linalg_or_ext - Execute a multi-LHS operator extension or `svd`/`eigh` statement
 * @prog:    Enclosing planned program (may be NULL)
 * @cs:      Constructor step holding cached operator state (may be NULL)
 * @s:       `MC_STMT_MULTI_ASSIGN` statement
 * @in_unit: 1 when running inside a `milk.calc` unit
 * @out:     Optional output pointer receiving the last scalar output value
 * @err:     Error message buffer
 * @errsz:   Capacity of @err in bytes
 *
 * Routes multi-LHS Tier 2 operator calls (`a, b = my_op(...)`) to `exec_ext_op()` and multi-LHS
 * matrix decompositions (`U, S, Vt = svd(A)` or `w, V = eigh(A)`) to `mc_exec_multi_svd_or_eigh()`.
 *
 * Return: 0 on success, -1 on error
 */
int mc_exec_multi_linalg_or_ext(mcalc_prog_t    *prog,
                                mc_ctor_step_t  *cs,
                                const mc_stmt_t *s,
                                int              in_unit,
                                mc_val_t        *out,
                                char            *err,
                                size_t           errsz)
{
    const mc_expr_t *rhs = s->u.multi.rhs[0];
    if (rhs->kind == MC_EXPR_CALL && rhs->u.call.ext_id >= 0)
    {
        const milk_calc_ext_entry_t *ext = milk_calc_ext_get(rhs->u.call.ext_id);
        return (ext != NULL && ext->kind == MILK_EXT_OP)
                   ? exec_ext_op(prog, cs, ext, rhs, s, -1, in_unit, out, err, errsz)
                   : ext_err(err, errsz, "unknown multi-output extension '%s'", rhs->u.call.name);
    }
    if (rhs->kind == MC_EXPR_CALL && rhs->u.call.fn != NULL &&
        (strcmp(rhs->u.call.fn->name, "svd") == 0 || strcmp(rhs->u.call.fn->name, "eigh") == 0))
    {
        return mc_exec_multi_svd_or_eigh(prog, s, in_unit, err, errsz);
    }
    return ext_err(err, errsz, "'%s' has one result: assign it to one name",
                   (rhs->kind == MC_EXPR_CALL) ? rhs->u.call.name : "the expression");
}
