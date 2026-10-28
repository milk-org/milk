// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_ops_axis.c
 * @brief Bytecode compilers for math functions, complex parts, coordinates, and reductions.
 *
 * Role & Goal:
 *   Compiles built-in 1/2/3-argument math functions, `where()` conditionals, complex
 *   component extraction (`FI_CPART`), coordinate/PRNG generators (`FI_COORD`), and
 *   scalar or axis reductions (`FI_RED`, `AX_KIND_AXIS`).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "cli_calc_fuse_internal.h"

/**
 * cli_calc_fuse_c_cplx_part - Fuse `real(z)`, `imag(z)`, `abs(z)`, or `arg(z)` on complex leaf `z`
 * @p:     Target fused program
 * @which: Component selector (`ARITH_CPLX_*`)
 * @out:   Output value descriptor
 *
 * Return: 0 on success, -1 if argument is not a complex leaf image.
 */
int cli_calc_fuse_c_cplx_part(fprog *p, int which, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    if (cur()->type != TOK_IMAGE || strchr(cur()->sval, '[') != NULL)
    {
        return -1;
    }
    imageID id = image_ID(cur()->sval, data.core.image, data.core.NB_MAX_IMAGE);
    if (id == -1 || !arith_is_complex_dt(data.core.image[id].md[0].datatype))
    {
        return -1;
    }
    adv();
    if (cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    int k = cli_calc_fuse_add_complex_image(p, &data.core.image[id]);
    if (k < 0)
    {
        return -1;
    }
    p->out_dt = 0;
    out->vec  = 1;
    out->s1   = 0;
    return emit(p, (fins) { .k = FI_CPART, .img = k, .cpos = which }, 1);
}

/**
 * cli_calc_fuse_c_func - Compile a 1-, 2-, or 3-argument math or cast function call into @p
 * @p:   Target fused program
 * @ft:  Function token
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_func(fprog *p, const cli_token *ft, fval *out)
{
    if (ft->type == TOK_FUNC_D_D && ft->fnctptr == (double (*)()) fabs)
    {
        int pos = cli_calc_tok_pos();
        if (cli_calc_fuse_c_cplx_part(p, ARITH_CPLX_ABS, out) == 0)
        {
            return 0;
        }
        cli_calc_tok_seek(pos);
    }
    fval a[3]  = { { 0, 0, 0.0, 0 }, { 0, 0, 0.0, 0 }, { 0, 0, 0.0, 0 } };
    int  nargs = (ft->type == TOK_FUNC_D_D) ? 1 : (ft->type == TOK_FUNC_DD_D) ? 2 : 3;
    if (cli_calc_fuse_c_args(p, nargs, a) != 0)
    {
        return -1;
    }
    out->vec = a[0].vec || (nargs > 1 && a[1].vec) || (nargs > 2 && a[2].vec);
    if (!out->vec)
    {
        if (a[0].s1 == 0 && (nargs < 2 || a[1].s1 == 0) && (nargs < 3 || a[2].s1 == 0))
        {
            out->s1 = 0;
            out->c =
                (nargs == 1) ? ((double (*)(double)) ft->fnctptr)(a[0].c)
                : (nargs == 2)
                    ? ((double (*)(double, double)) ft->fnctptr)(a[0].c, a[1].c)
                    : ((double (*)(double, double, double)) ft->fnctptr)(a[0].c, a[1].c, a[2].c);
            return 0;
        }
        return emit_scalar(p,
                           (fsins) { .k  = (nargs == 1)   ? FI_FN1
                                           : (nargs == 2) ? FI_FN2
                                                          : FI_FN3,
                                     .fn = ft->fnctptr,
                                     .s  = { a[0].s1, a[1].s1, a[2].s1 },
                                     .c  = { a[0].c, a[1].c, a[2].c } },
                           out);
    }
    if (ft->type == TOK_FUNC_D_D)
    {
        p->out_dt = cli_calc_cast_datatype(ft->fnctptr);
        if (p->out_dt == _DATATYPE_DOUBLE)
        {
            p->out_double = 1;
        }
        return emit(
            p,
            (fins) { .k = FI_FN1, .fn = ft->fnctptr, .fn1 = cli_calc_fuse_fn1_kind(ft->fnctptr) },
            0);
    }
    p->out_dt = 0;
    if (ft->type == TOK_FUNC_DDD_D)
    {
        if (a[1].vec || a[2].vec)
        {
            return -1;
        }
        return emit(p,
                    (fins) { .k  = FI_FN3,
                             .fn = ft->fnctptr,
                             .s1 = a[1].s1,
                             .c  = a[1].c,
                             .s2 = a[2].s1,
                             .c2 = a[2].c },
                    0);
    }
    double (*fn)() = ft->fnctptr;
    int both       = a[0].vec && a[1].vec;
    int cmut       = (fn == (double (*)()) fmin || fn == (double (*)()) fmax ||
                      fn == (double (*)()) fmod || fn == (double (*)()) hypot);
    if ((both && !cmut && fn != (double (*)()) pow && fn != (double (*)()) atan2) ||
        (!a[0].vec && !cmut && fn != (double (*)()) atan2))
    {
        return -1;
    }
    fins in = { .k    = FI_FN2,
                .fn   = fn,
                .cpos = both       ? 0
                        : a[0].vec ? 2
                                   : 1,
                .s1   = a[0].vec ? a[1].s1 : a[0].s1,
                .c    = a[0].vec ? a[1].c : a[0].c };
    return emit(p, in, both ? -1 : 0);
}

/**
 * cli_calc_fuse_c_select - Emit per-pixel conditional `cond ? t : f` instruction (`FI_SEL`)
 * @p:   Target fused program
 * @t:   True-branch operand descriptor
 * @f:   False-branch operand descriptor
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on stack overflow.
 */
int cli_calc_fuse_c_select(fprog *p, fval t, fval f, fval *out)
{
    p->out_dt = 0;
    out->vec  = 1;
    out->s1   = 0;
    return emit(p,
                (fins) { .k    = FI_SEL,
                         .cpos = t.vec | (f.vec << 1),
                         .s1   = t.s1,
                         .c    = t.c,
                         .s2   = f.s1,
                         .c2   = f.c },
                -(t.vec + f.vec));
}

/**
 * cli_calc_fuse_c_coord - Compile `ix/iy/ir/itheta(im[, cx, cy])` or `rand/randn(im[, seed])`
 * @p:   Target fused program
 * @ft:  Coordinate/PRNG function token
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_coord(fprog *p, const cli_token *ft, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    cli_token *t            = adv();
    fval       c[2];
    int        kind = cli_calc_coord_kind(ft);

    if (t->type != TOK_IMAGE || strchr(t->sval, '[') != NULL)
    {
        return -1;
    }
    imageID id = image_ID(t->sval, data.core.image, data.core.NB_MAX_IMAGE);
    int     k  = (id == -1) ? -1 : cli_calc_fuse_add_image(p, &data.core.image[id]);
    if (k < 0)
    {
        return -1;
    }
    if (kind == CALC_COORD_RAND || kind == CALC_COORD_RANDN)
    {
        c[0] = (fval) { 0, 0, 0.0, 0 };
        c[1] = (fval) { 0, 0, 0.0, 0 };
        if (cur()->type == TOK_COMMA)
        {
            adv();
            if (cli_calc_fuse_c_expr(p, 0, &c[0]) != 0 || c[0].vec || cur()->type != TOK_RPAREN)
            {
                return -1;
            }
            adv();
            c[1].c = 1.0;
        }
        else if (cur()->type == TOK_RPAREN)
        {
            adv();
        }
        else
        {
            return -1;
        }
        p->out_dt = 0;
        out->vec  = 1;
        out->s1   = 0;
        return emit(
            p,
            (fins) {
                .k = FI_COORD, .img = k, .cpos = kind, .s1 = c[0].s1, .c = c[0].c, .c2 = c[1].c },
            1);
    }
    const IMAGE *im = p->img[k];
    c[0]            = (fval) { 0, 0, (double) (im->md[0].size[0] / 2), 0 };
    c[1] = (fval) { 0, 0, (double) ((im->md[0].naxis > 1) ? im->md[0].size[1] / 2 : 0), 0 };
    if (cur()->type == TOK_COMMA)
    {
        adv();
        if (cli_calc_fuse_c_args(p, 2, c) != 0 || c[0].vec || c[1].vec)
        {
            return -1;
        }
    }
    else if (cur()->type == TOK_RPAREN)
    {
        adv();
    }
    else
    {
        return -1;
    }
    p->out_dt = 0;
    out->vec  = 1;
    out->s1   = 0;
    return emit(p,
                (fins) { .k    = FI_COORD,
                         .img  = k,
                         .cpos = kind,
                         .s1   = c[0].s1,
                         .c    = c[0].c,
                         .s2   = c[1].s1,
                         .c2   = c[1].c },
                1);
}

/**
 * cli_calc_fuse_axis_result_dt - Determine output datatype for an axis or pre-stage reduction
 * @sub: Compiled sub-program
 *
 * Return: `_DATATYPE_FLOAT` or `_DATATYPE_DOUBLE`.
 */
uint8_t cli_calc_fuse_axis_result_dt(const fprog *sub)
{
    if (sub->nops == 0 && sub->nimg == 1)
    {
        uint8_t dt = sub->img[0]->md[0].datatype;
        return (dt == _DATATYPE_FLOAT || dt == _DATATYPE_HALF || dt == _DATATYPE_UINT8 ||
                dt == _DATATYPE_INT8 || dt == _DATATYPE_UINT16 || dt == _DATATYPE_INT16)
                   ? _DATATYPE_FLOAT
                   : _DATATYPE_DOUBLE;
    }
    return sub->out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
}
#define axis_result_dt cli_calc_fuse_axis_result_dt

/**
 * cli_calc_fuse_axred_setup - Configure pre-stage slot @i of @p as an internal image of shape @osz
 * @p:    Parent fused program
 * @i:    Pre-stage slot index
 * @onax: Output axis count
 * @osz:  Output 3D dimensions
 * @dt:   Output datatype
 *
 * Return: 0 on success, -1 on allocation failure.
 */
int cli_calc_fuse_axred_setup(fprog *p, int i, int onax, const uint32_t osz[3], uint8_t dt)
{
    IMAGE_METADATA *md    = &p->ax_md[i];
    IMAGE          *im    = &p->ax_im[i];
    uint64_t        nel   = (uint64_t) osz[0] * osz[1] * osz[2];
    size_t          bytes = (size_t) nel * (size_t) ImageStreamIO_typesize(dt);

    if (bytes > p->ax_cap[i])
    {
        void *nb = realloc(p->ax_data[i], bytes);
        if (nb == NULL)
        {
            return -1;
        }
        p->ax_data[i] = nb;
        p->ax_cap[i]  = bytes;
    }
    memset(md, 0, sizeof(*md));
    memset(im, 0, sizeof(*im));
    md->naxis    = (uint8_t) onax;
    md->size[0]  = osz[0];
    md->size[1]  = osz[1];
    md->size[2]  = osz[2];
    md->nelement = nel;
    md->datatype = dt;
    snprintf(md->name, sizeof(md->name), "_axred%d", i);
    snprintf(im->name, sizeof(im->name), "_axred%d", i);
    im->md        = md;
    im->used      = 1;
    im->array.raw = p->ax_data[i];
    return 0;
}
#define axred_setup cli_calc_fuse_axred_setup

/** c_axis_reduce - Register @sub (ownership transferred) as an axis-@axis reduction in @p. */
static int c_axis_reduce(fprog *p, fprog *sub, int fop, int axis, fval *out)
{
    int      slot   = p->nax;
    uint32_t osz[3] = { 1, 1, 1 };
    int      onax   = 0;

    if (slot >= FUSE_MAXAX || axis < 0 || axis >= sub->inaxis)
    {
        cli_calc_fuse_reject((slot >= FUSE_MAXAX) ? "more than 4 pre-stages (FUSE_MAXAX)"
                                                  : "reduction axis out of range");
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    for (int a = 0; a < sub->inaxis; a++)
    {
        if (a != axis)
        {
            osz[onax++] = sub->ishape[a];
        }
    }
    if (onax == 0)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    if (p->ax_p[slot] != NULL)
    {
        cli_calc_fuse_free_fprog_children(p->ax_p[slot]);
        free(p->ax_p[slot]);
    }
    p->ax_p[slot]    = sub;
    p->ax_p2[slot]   = NULL;
    p->ax_kind[slot] = AX_KIND_AXIS;
    p->ax_axis[slot] = axis;
    p->ax_op[slot]   = (reduce_op) fop;
    if (axred_setup(p, slot, onax, osz, axis_result_dt(sub)) != 0)
    {
        p->ax_p[slot] = NULL;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->nax = slot + 1;
    int k  = cli_calc_fuse_add_axis_operand(p, &p->ax_im[slot]);
    if (k < 0)
    {
        p->nax        = slot;
        p->ax_p[slot] = NULL;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->out_dt = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = FI_IMG, .img = k }, 1);
}

/** c_reduce_register_slot - Deduplicate or allocate a scalar-tape reduction slot in @p. */
static int c_reduce_register_slot(fprog           *p,
                                  const cli_token *ft,
                                  int              fop,
                                  fprog           *scratch,
                                  fprog           *scratch2,
                                  int              arg1_tok0,
                                  int              arg1_tok1,
                                  fval            *out)
{
    int slot = -1;
    if (scratch2 == NULL && arg1_tok0 >= 0)
    {
        for (int r = 0; r < p->nred; r++)
        {
            if (p->red_p2[r] == NULL && p->red_tok0[r] >= 0 &&
                cli_calc_tok_range_eq(p->red_tok0[r], p->red_tok1[r], arg1_tok0, arg1_tok1))
            {
                slot         = r;
                p->red_op[r] = arith_reduce_op_promote(p->red_op[r], (reduce_op) fop);
                cli_calc_fuse_free_fprog_children(scratch);
                free(scratch);
                break;
            }
        }
    }
    if (slot < 0)
    {
        if (p->nred >= FUSE_MAXRED)
        {
            cli_calc_fuse_reject("more than 4 reductions (FUSE_MAXRED)");
            cli_calc_fuse_free_fprog_children(scratch);
            free(scratch);
            if (scratch2 != NULL)
            {
                cli_calc_fuse_free_fprog_children(scratch2);
                free(scratch2);
            }
            return -1;
        }
        slot              = p->nred++;
        p->red_p1[slot]   = scratch;
        p->red_p2[slot]   = scratch2;
        p->red_op[slot]   = (reduce_op) fop;
        p->red_tok0[slot] = arg1_tok0;
        p->red_tok1[slot] = arg1_tok1;
        if (scratch2 != NULL || p->has_red == 0)
        {
            p->has_red = (scratch2 != NULL) ? 2 : 1;
        }
    }
    return emit_scalar(p,
                       (fsins) { .k   = FI_RED,
                                 .op  = (cli_token_type) fop,
                                 .vid = (ft->type == TOK_FUNC_IM_L),
                                 .c   = { (double) slot, 0.0, 0.0 } },
                       out);
}

/**
 * cli_calc_fuse_c_reduce - Compile an axis reduction or a scalar-tape reduction call
 * @p:   Target fused program
 * @ft:  Reduction function token
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_reduce(fprog *p, const cli_token *ft, fval *out)
{
    int fop = cli_calc_fused_reduce_op(ft->fnctptr);
    if (fop < 0)
    {
        return -1;
    }
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    fval v1;
    fval v2;

    fprog *scratch = (fprog *) calloc(1, sizeof(fprog));
    if (scratch == NULL)
    {
        return -1;
    }
    int udepth       = cli_calc_fuse_udef_depth();
    int arg1_tok0    = (udepth == 0) ? cli_calc_tok_pos() : -1;
    int saved_anchor = cli_calc_fuse_anchor_save();
    int rc1          = cli_calc_fuse_compile(scratch, 0, &v1);
    cli_calc_fuse_anchor_restore(saved_anchor);
    int arg1_tok1 = (udepth == 0) ? cli_calc_tok_pos() : -1;
    if (rc1 != 0 || !v1.vec)
    {
        cli_calc_fuse_free_fprog_children(scratch);
        free(scratch);
        return -1;
    }
    if (fop == (int) RED_TRACE)
    {
        if (cur()->type != TOK_RPAREN || cli_calc_fuse_restrict_diag(scratch, 0) != 0)
        {
            cli_calc_fuse_free_fprog_children(scratch);
            free(scratch);
            return -1;
        }
        adv();
        return c_reduce_register_slot(p, ft, RED_SUM, scratch, NULL, -1, -1, out);
    }
    if (cur()->type == TOK_COMMA && fop != RED_DOT && ft->type == TOK_FUNC_IM_D)
    {
        int    pos2   = cli_calc_tok_pos();
        double ax_val = 0.0;
        adv();
        if (cli_calc_eval_const_scalar(&ax_val) == 0)
        {
            if (!axis_op_ok(fop) || cur()->type != TOK_RPAREN ||
                cli_calc_fuse_finalize_views(scratch) != 0)
            {
                cli_calc_fuse_free_fprog_children(scratch);
                free(scratch);
                return -1;
            }
            adv();
            return c_axis_reduce(p, scratch, fop, (int) ax_val, out);
        }
        cli_calc_tok_seek(pos2);
    }

    fprog *scratch2 = NULL;
    if (fop == RED_DOT || cur()->type == TOK_COMMA)
    {
        if (cur()->type != TOK_COMMA || (fop != RED_DOT && !axis_op_ok(fop)))
        {
            cli_calc_fuse_free_fprog_children(scratch);
            free(scratch);
            return -1;
        }
        adv();
        scratch2 = (fprog *) calloc(1, sizeof(fprog));
        if (scratch2 == NULL)
        {
            cli_calc_fuse_free_fprog_children(scratch);
            free(scratch);
            return -1;
        }
        (void) cli_calc_fuse_anchor_save();
        int rc2 = cli_calc_fuse_compile(scratch2, 0, &v2);
        cli_calc_fuse_anchor_restore(saved_anchor);
        if (rc2 != 0 || !v2.vec || !cli_calc_fuse_reduce_shapes_match(scratch, scratch2))
        {
            cli_calc_fuse_free_fprog_children(scratch);
            free(scratch);
            cli_calc_fuse_free_fprog_children(scratch2);
            free(scratch2);
            return -1;
        }
    }
    else if (cli_calc_fuse_finalize_views(scratch) != 0)
    {
        cli_calc_fuse_free_fprog_children(scratch);
        free(scratch);
        return -1;
    }
    if (cur()->type != TOK_RPAREN)
    {
        cli_calc_fuse_free_fprog_children(scratch);
        free(scratch);
        if (scratch2 != NULL)
        {
            cli_calc_fuse_free_fprog_children(scratch2);
            free(scratch2);
        }
        return -1;
    }
    adv();
    return c_reduce_register_slot(p, ft, fop, scratch, scratch2, arg1_tok0, arg1_tok1, out);
}

/** centroid_accum_f - Accumulate thresholded centroid moments on a `float` block. */
static void centroid_accum_f(const float *stk0,
                             uint64_t     off,
                             int          len,
                             uint64_t     nx,
                             uint64_t     ny,
                             double       thresh,
                             double      *sum_i,
                             double      *sum_xi,
                             double      *sum_yi)
{
    uint64_t x_ = off % nx, r_ = off / nx, y_ = r_ % ny;
    int      J = 0;
    while (J < len)
    {
        int          N = (int) ((nx - x_ < (uint64_t) (len - J)) ? nx - x_ : (uint64_t) (len - J));
        double       row_i  = 0.0;
        double       row_xi = 0.0;
        double       x0     = (double) x_;
        const float *s      = stk0 + J;
        _Pragma("omp simd reduction(+ : row_i, row_xi)") for (int i = 0; i < N; i++)
        {
            double v  = (double) s[i] - thresh;
            double ip = (v > 0.0) ? v : 0.0;
            row_i += ip;
            row_xi += (x0 + (double) i) * ip;
        }
        *sum_i += row_i;
        *sum_xi += row_xi;
        *sum_yi += (double) y_ * row_i;
        J += N;
        x_ = 0;
        if (++y_ == ny)
        {
            y_ = 0;
        }
    }
}

/** centroid_accum_d - Accumulate thresholded centroid moments on a `double` block. */
static void centroid_accum_d(const double *stk0,
                             uint64_t      off,
                             int           len,
                             uint64_t      nx,
                             uint64_t      ny,
                             double        thresh,
                             double       *sum_i,
                             double       *sum_xi,
                             double       *sum_yi)
{
    uint64_t x_ = off % nx, r_ = off / nx, y_ = r_ % ny;
    int      J = 0;
    while (J < len)
    {
        int           N = (int) ((nx - x_ < (uint64_t) (len - J)) ? nx - x_ : (uint64_t) (len - J));
        double        row_i  = 0.0;
        double        row_xi = 0.0;
        double        x0     = (double) x_;
        const double *s      = stk0 + J;
        _Pragma("omp simd reduction(+ : row_i, row_xi)") for (int i = 0; i < N; i++)
        {
            double v  = s[i] - thresh;
            double ip = (v > 0.0) ? v : 0.0;
            row_i += ip;
            row_xi += (x0 + (double) i) * ip;
        }
        *sum_i += row_i;
        *sum_xi += row_xi;
        *sum_yi += (double) y_ * row_i;
        J += N;
        x_ = 0;
        if (++y_ == ny)
        {
            y_ = 0;
        }
    }
}

/**
 * cli_calc_fuse_run_centroid - Single-pass flux-weighted centroid reduction of @sub above @thresh
 * @sub:      Compiled fused sub-program
 * @thresh:   Flux threshold (`ip = max(0, im - thresh)`)
 * @out_cent: Output 3-element array receiving `(cx, cy, flux)`
 *
 * Return: 0 on success, -1 on error.
 */
typedef struct
{
    const fprog *sub;
    uint64_t     nx;
    uint64_t     ny;
    double       thresh;
    double       sums[3];
} centroid_job_t;

/** CENTROID_JOB - team job of one element type: flux moments of blocks [lo, hi), merged under the lock. */
#define CENTROID_JOB(NAME, FT, EVAL, ACCUM)                                       \
    static void NAME(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)       \
    {                                                                             \
        centroid_job_t     *j = (centroid_job_t *) ctx;                           \
        MILK_ALIGNED(64) FT stk[FUSE_MAXSTK][FUSE_BLK];                           \
        double              si = 0.0, sx = 0.0, sy = 0.0;                         \
        const uint64_t      nel = j->sub->inel;                                   \
        (void) tid;                                                               \
        (void) nth;                                                               \
        for (uint64_t b = lo; b < hi; b++)                                        \
        {                                                                         \
            uint64_t off = b * FUSE_BLK;                                          \
            int      len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK); \
            EVAL(j->sub, off, len, stk);                                          \
            ACCUM(stk[0], off, len, j->nx, j->ny, j->thresh, &si, &sx, &sy);      \
        }                                                                         \
        cli_calc_team_lock();                                                     \
        j->sums[0] += si;                                                         \
        j->sums[1] += sx;                                                         \
        j->sums[2] += sy;                                                         \
        cli_calc_team_unlock();                                                   \
    }

CENTROID_JOB(centroid_job_f, float, cli_calc_fuse_eval_block_f, centroid_accum_f)
CENTROID_JOB(centroid_job_d, double, cli_calc_fuse_eval_block_d, centroid_accum_d)
#undef CENTROID_JOB

int cli_calc_fuse_run_centroid(fprog *sub, double thresh, double out_cent[3])
{
    double sval[FUSE_MAXINS];
    if (cli_calc_fuse_eval_scalars(sub, sval) != 0 ||
        cli_calc_fuse_refresh_views(sub, sval, NULL) != 0 ||
        cli_calc_fuse_finalize_views(sub) != 0 || cli_calc_fuse_run_axes(sub) != 0)
    {
        return -1;
    }
    const uint64_t nel  = sub->inel;
    const uint64_t nx   = sub->ishape[0];
    const uint64_t ny   = sub->ishape[1];
    const uint64_t nblk = (nel + FUSE_BLK - 1) / FUSE_BLK;
    int            nth  = milk_omp_team_size(fprog_effective_nel(sub), FUSE_OMP_MIN);
    if ((uint64_t) nth > nblk)
    {
        nth = (int) nblk;
    }
    centroid_job_t job = { sub, nx, ny, thresh, { 0.0, 0.0, 0.0 } };
    cli_calc_team_run(nth, nblk,
                      cli_calc_fuse_all_float_prog(sub) ? centroid_job_f : centroid_job_d, &job);
    double sum_i  = job.sums[0];
    double sum_xi = job.sums[1];
    double sum_yi = job.sums[2];

    if (sum_i > 0.0)
    {
        out_cent[0] = sum_xi / sum_i;
        out_cent[1] = sum_yi / sum_i;
        out_cent[2] = sum_i;
    }
    else
    {
        out_cent[0] = NAN;
        out_cent[1] = NAN;
        out_cent[2] = 0.0;
    }
    return 0;
}
