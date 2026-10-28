// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_exec.c
 * @brief Cache-resident block kernels, scalar-tape runtime, and the block dispatcher on the team.
 *
 * Role & Goal:
 *   Evaluates scalar tapes (`sins[]`), instantiates double and float32 vector block kernels
 *   via `cli_calc_fuse_kernels.h`, dispatches OpenMP parallel block loops in
 *   `cli_calc_fuse_run_program()`, and manages immediate/deferred SHM output stream posting.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_fuse_internal.h"
#include "libmilkdata/milk_type_dispatch.h"
#include "milk_shm_prop.h"

/**
 * cli_calc_fuse_load_block - Convert a contiguous block of image pixels to `double`
 * @im:  Source image
 * @off: Starting pixel index
 * @len: Number of pixels to load
 * @dst: Output buffer of length @len
 *
 * Return: void
 */
void cli_calc_fuse_load_block(const IMAGE *im, uint64_t off, int len, double *restrict dst)
{
    arith_load_block_d_inline(im, off, len, dst);
}

/**
 * cli_calc_fuse_store_block - Write a block of evaluated `double` pixels or a constant into @im
 * @im:  Destination image
 * @off: Starting pixel index
 * @len: Number of pixels to write
 * @v:   Input vector buffer (ignored when @isc is non-zero)
 * @isc: Non-zero to fill with constant @c instead of reading @v
 * @c:   Constant fill value when @isc is non-zero
 *
 * Return: void
 */
void cli_calc_fuse_store_block(IMAGE   *im,
                               uint64_t off,
                               int      len,
                               const double *restrict v,
                               int    isc,
                               double c)
{
    arith_store_block_d_inline(im, off, len, v, isc, c);
}

static int eval_scalar_red_or_cent(fprog       *p,
                                   const fsins *s,
                                   double      *sval,
                                   int          i,
                                   fred_acc    *red_acc,
                                   int         *red_done,
                                   double       cent_vals[3],
                                   int         *cent_done)
{
    if (s->k == FI_RED)
    {
        int r = (int) s->c[0];
        if (r < 0 || r >= p->nred || p->red_p1[r] == NULL)
        {
            return -1;
        }
        if (!red_done[r])
        {
            if (cli_calc_fuse_run_reduce(p->red_p1[r], p->red_p2[r], p->red_op[r], &red_acc[r]) !=
                0)
            {
                return -1;
            }
            red_done[r] = 1;
        }
        sval[i] = fuse_finish(&red_acc[r], (reduce_op) s->op, s->vid);
        return 0;
    }
    if (p->red_p1[0] == NULL || (int) s->op < 0 || (int) s->op > 2)
    {
        return -1;
    }
    if (!*cent_done)
    {
        double th = (p->cent_th_s1 > 0) ? sval[p->cent_th_s1 - 1] : p->cent_th_c;
        if (cli_calc_fuse_run_centroid(p->red_p1[0], th, cent_vals) != 0)
        {
            return -1;
        }
        *cent_done = 1;
    }
    sval[i] = cent_vals[(int) s->op];
    return 0;
}

/** scalar_operand() - operand @k of @s: its double value, and its exact long value when it is long. */
static inline double scalar_operand(const fprog  *p,
                                    const fsins  *s,
                                    int           k,
                                    const double *sval,
                                    int          *is_long,
                                    int64_t      *lval)
{
    if (s->s[k] > 0)
    {
        int i    = s->s[k] - 1;
        *is_long = p->slong[i];
        *lval    = *is_long ? p->slval[i] : (int64_t) sval[i];
        return sval[i];
    }
    *is_long = s->cl[k];
    *lval    = (int64_t) s->c[k];
    return s->c[k];
}

/** eval_one_scalar - Evaluate a single scalar tape instruction @s at index @i of @p. */
static int eval_one_scalar(fprog    *p,
                           int       i,
                           double   *sval,
                           fred_acc *red_acc,
                           int      *red_done,
                           double    cent_vals[3],
                           int      *cent_done)
{
    const fsins *s = &p->sins[i];
    int          al, bl, dl;
    int64_t      la, lb, ld;
    double       a       = scalar_operand(p, s, 0, sval, &al, &la);
    double       b       = scalar_operand(p, s, 1, sval, &bl, &lb);
    double       d       = scalar_operand(p, s, 2, sval, &dl, &ld);
    int          is_long = 0;
    int64_t      lres    = 0;
    switch (s->k)
    {
    case FI_SVAR:
    {
        const VARIABLE *v = &data.core.variable[s->vid];
        is_long           = (v->type == 1);
        lres              = v->value.l;
        sval[i]           = is_long ? (double) v->value.l : v->value.f;
        break;
    }
    case FI_PROP:
    {
        long lval = 0;
        if (s->vid < 0 || s->vid >= p->nprop ||
            milk_shm_prop_get_num(p->prop_path[s->vid], &sval[i], &lval, &is_long) != 0)
        {
            parse_errmsg("Failed to read shared-memory property");
            return -1;
        }
        if (is_long)
        {
            sval[i] = (double) lval;
            lres    = lval;
        }
        break;
    }
    case FI_NEG:
        is_long = al;
        lres    = -la;
        sval[i] = is_long ? (double) lres : -a;
        break;
    case FI_NOT:
        is_long = 1;
        lres    = (a == 0.0);
        sval[i] = (double) lres;
        break;
    case FI_BNOT:
        is_long = 1;
        lres    = ~(al ? la : (int64_t) a);
        sval[i] = (double) lres;
        break;
    case FI_SEL:
        is_long = (a != 0.0) ? bl : dl;
        lres    = (a != 0.0) ? lb : ld;
        sval[i] = (a != 0.0) ? b : d;
        break;
    case FI_BIN:
        if (!s->guarded && (s->op == TOK_OP_SLASH || s->op == TOK_OP_MOD) && b == 0.0)
        {
            parse_errmsg((s->op == TOK_OP_SLASH) ? "Division by zero" : "Modulo by zero");
            return -1;
        }
        if (al && bl && s->op != TOK_OP_SLASH)
        {
            /* two longs: exactly as the text engine, with its overflow fallback to double */
            val_t       r;
            const char *err = NULL;
            if (cli_calc_long_binop(s->op, (long) la, (long) lb, &r, &err))
            {
                is_long = (r.type == VAL_LONG);
                lres    = r.lval;
                sval[i] = is_long ? (double) r.lval : r.dval;
                break;
            }
        }
        sval[i] = scalar_bin(s->op, a, b);
        is_long = bin_is_long(s->op, al, bl);
        lres    = (int64_t) sval[i];
        break;
    case FI_RED:
    case FI_CENT:
        if (eval_scalar_red_or_cent(p, s, sval, i, red_acc, red_done, cent_vals, cent_done) != 0)
        {
            return -1;
        }
        is_long = (s->k == FI_RED) && (s->vid != 0);
        lres    = (int64_t) sval[i];
        break;
    case FI_FN1:
        sval[i] = ((double (*)(double)) s->fn)(a);
        break;
    case FI_FN2:
        sval[i] = ((double (*)(double, double)) s->fn)(a, b);
        break;
    default:
        sval[i] = ((double (*)(double, double, double)) s->fn)(a, b, d);
        break;
    } /* switch (s->k) */
    p->slong[i] = (uint8_t) is_long;
    p->slval[i] = lres;
    return 0;
}

/**
 * cli_calc_fuse_eval_scalars - Evaluate per-frame scalar variable sub-expressions of @p
 * @p:        Compiled fused program
 * @out_sval: Optional output buffer receiving evaluated scalar tape values (`FUSE_MAXINS`)
 *
 * Return: 0 on success, -1 on runtime error.
 */
int cli_calc_fuse_eval_scalars(fprog *p, double *out_sval)
{
    for (int i = 0; i < p->n; i++)
    {
        if (p->ins[i].k == FI_COORD &&
            (p->ins[i].cpos == CALC_COORD_RAND || p->ins[i].cpos == CALC_COORD_RANDN) &&
            p->ins[i].s1 == 0 && p->ins[i].c2 == 0.0)
        {
            p->ins[i].c = (double) cli_calc_next_rng_seed();
        }
    }
    if (p->ns == 0)
    {
        for (int u = 0; u < p->nuser; u++)
        {
            fuse_user_call_t            *uc = &p->ucall[u];
            const milk_calc_ext_entry_t *e  = cli_calc_ext_get(uc->ext_idx);
            const milk_calc_param_set_t *pset =
                (e != NULL && e->get_params != NULL) ? e->get_params() : NULL;
            cli_calc_ext_pack_params(pset, uc->pbuf, uc->pvals);
        }
        return 0;
    }
    double   local_sval[FUSE_MAXINS];
    double  *sval = (out_sval != NULL) ? out_sval : local_sval;
    fred_acc red_acc[FUSE_MAXRED];
    int      red_done[FUSE_MAXRED] = { 0 };
    double   cent_vals[3]          = { 0.0, 0.0, 0.0 };
    int      cent_done             = 0;
    for (int i = 0; i < p->ns; i++)
    {
        if (eval_one_scalar(p, i, sval, red_acc, red_done, cent_vals, &cent_done) != 0)
        {
            return -1;
        }
    }
    for (int a = 0; a < p->nax; a++)
    {
        if (p->ax_s[a][0] > 0)
        {
            p->ax_c[a][0] = sval[p->ax_s[a][0] - 1];
        }
        if (p->ax_s[a][1] > 0)
        {
            p->ax_c[a][1] = sval[p->ax_s[a][1] - 1];
        }
    }
    for (int i = 0; i < p->n; i++)
    {
        if (p->ins[i].s1 > 0)
        {
            p->ins[i].c = sval[p->ins[i].s1 - 1];
        }
        if (p->ins[i].s2 > 0)
        {
            p->ins[i].c2 = sval[p->ins[i].s2 - 1];
        }
        if ((p->ins[i].k == FI_BIN || p->ins[i].k == FI_IMGC) && !p->ins[i].guarded &&
            (p->ins[i].op == TOK_OP_SLASH || p->ins[i].op == TOK_OP_MOD) && p->ins[i].cpos == 2 &&
            p->ins[i].c == 0.0)
        {
            parse_errmsg((p->ins[i].op == TOK_OP_SLASH) ? "Division by zero" : "Modulo by zero");
            return -1;
        }
    }
    for (int u = 0; u < p->nuser; u++)
    {
        fuse_user_call_t *uc = &p->ucall[u];
        for (int j = 0; j < uc->nparams; j++)
        {
            if (uc->ps1[j] > 0)
            {
                uc->pvals[j] = sval[uc->ps1[j] - 1];
            }
        }
        const milk_calc_ext_entry_t *e = cli_calc_ext_get(uc->ext_idx);
        const milk_calc_param_set_t *pset =
            (e != NULL && e->get_params != NULL) ? e->get_params() : NULL;
        cli_calc_ext_pack_params(pset, uc->pbuf, uc->pvals);
    }
    return 0;
}

static void store_block_d(IMAGE *im, uint64_t off, int len, const double *restrict v)
{
    cli_calc_fuse_store_block(im, off, len, v, 0, 0.0);
}

/** load_block_f() - float32 block of @im: a copy for FLOAT, a converting loop for other types. */
static void load_block_f(const IMAGE *im, uint64_t off, int len, float *restrict dst)
{
    arith_load_block_f_inline(im, off, len, dst);
}

/** store_block_f() - store a float32 block into @im, converting to its element type when not FLOAT. */
static void store_block_f(IMAGE *im, uint64_t off, int len, const float *restrict v)
{
    arith_store_block_f_inline(im, off, len, v);
}

#define LOAD_SEG(DT, d, src, N, st)                           \
    do                                                        \
    {                                                         \
        if ((st) == 1)                                        \
        {                                                     \
            _Pragma("omp simd") for (int i = 0; i < (N); i++) \
            {                                                 \
                (d)[i] = (DT) (src)[i];                       \
            }                                                 \
        }                                                     \
        else if ((st) == 0)                                   \
        {                                                     \
            const DT c_ = (DT) (src)[0];                      \
            _Pragma("omp simd") for (int i = 0; i < (N); i++) \
            {                                                 \
                (d)[i] = c_;                                  \
            }                                                 \
        }                                                     \
        else                                                  \
        {                                                     \
            for (int i = 0; i < (N); i++)                     \
            {                                                 \
                (d)[i] = (DT) (src)[(int64_t) i * (st)];      \
            }                                                 \
        }                                                     \
    } while (0)

#define STORE_SEG(OT, o, s, N, st)                            \
    do                                                        \
    {                                                         \
        if ((st) == 1)                                        \
        {                                                     \
            _Pragma("omp simd") for (int i = 0; i < (N); i++) \
            {                                                 \
                (o)[i] = (OT) (s)[i];                         \
            }                                                 \
        }                                                     \
        else                                                  \
        {                                                     \
            for (int i = 0; i < (N); i++)                     \
            {                                                 \
                (o)[(int64_t) i * (st)] = (OT) (s)[i];        \
            }                                                 \
        }                                                     \
    } while (0)

static void loadv_block(const fprog *p, int k, uint64_t off, int len, double *restrict dst)
{
    const IMAGE  *im = p->img[k];
    const fview  *v  = &p->view[k];
    const int64_t st = v->stride[0];
    uint8_t       dt = im->md[0].datatype;
#define LOADV_BODY(MBR) LOAD_SEG(double, dst + J, im->array.MBR + S, N, st);
    VIEW_WALK(p, v, off, len, { MILK_FOR_EACH_DATATYPE(dt, LOADV_BODY, LOADV_BODY(D)) });
#undef LOADV_BODY
}

static void storev_block(const fprog *p,
                         IMAGE       *out,
                         uint64_t     off,
                         int          len,
                         const double *restrict v)
{
    const fview  *dv = &p->dstv;
    const int64_t st = dv->stride[0];
    uint8_t       dt = out->md[0].datatype;
#define STOREV_BODY(MBR) STORE_SEG(__typeof__(out->array.MBR[0]), out->array.MBR + S, v + J, N, st);
    VIEW_WALK(p, dv, off, len, { MILK_FOR_EACH_DATATYPE(dt, STOREV_BODY, STOREV_BODY(D)) });
#undef STOREV_BODY
}

static void loadv_block_f(const fprog *p, int k, uint64_t off, int len, float *restrict dst)
{
    const IMAGE  *im = p->img[k];
    const fview  *v  = &p->view[k];
    const int64_t st = v->stride[0];
    uint8_t       dt = im->md[0].datatype;
#define LOADVF_BODY(MBR) LOAD_SEG(float, dst + J, im->array.MBR + S, N, st);
    VIEW_WALK(p, v, off, len, { MILK_FOR_EACH_DATATYPE(dt, LOADVF_BODY, LOADVF_BODY(D)) });
#undef LOADVF_BODY
}

static void storev_block_f(const fprog *p,
                           IMAGE       *out,
                           uint64_t     off,
                           int          len,
                           const float *restrict v)
{
    const fview  *dv = &p->dstv;
    const int64_t st = dv->stride[0];
    uint8_t       dt = out->md[0].datatype;
#define STOREVF_BODY(MBR) \
    STORE_SEG(__typeof__(out->array.MBR[0]), out->array.MBR + S, v + J, N, st);
    VIEW_WALK(p, dv, off, len, { MILK_FOR_EACH_DATATYPE(dt, STOREVF_BODY, STOREVF_BODY(D)) });
#undef STOREVF_BODY
}

#undef LOAD_SEG
#undef STORE_SEG

#define coord_block_d arith_coord_block_d_inline
#define coord_block_f arith_coord_block_f_inline
#define cplx_part_block_d arith_cplx_part_im_block_d_inline
#define cplx_part_block_f arith_cplx_part_im_block_f_inline
#define elem_pow_const_d arith_elem_pow_const_d_inline
#define elem_pow_const_f arith_elem_pow_const_f_inline
#define elem_clamp_d arith_elem_clamp_d_inline
#define elem_clamp_f arith_elem_clamp_f_inline
#define elem_sel_d arith_elem_sel_d_inline
#define elem_sel_f arith_elem_sel_f_inline

/* block kernels in double (any datatype; results converted on store) */
#define FT double
#define FK_DT _DATATYPE_DOUBLE
#define FK(name) name##_d
#define FM(fn) fn
#define FK_LOAD cli_calc_fuse_load_block
#define FK_STORE store_block_d
#define FK_LOADV loadv_block
#define FK_STOREV storev_block
#define FK_SHIFT cli_calc_fuse_shift_block_d
#define FK_LUT cli_calc_fuse_lut_block_d
#define FK_MEDFILT cli_calc_fuse_medfilt_block_d
#define FK_GATHER cli_calc_fuse_gather_block_d
#define FK_INTERP2 cli_calc_fuse_interp2_block_d
#define FK_USER_BLK cli_calc_fuse_user_blk_d
#define FK_USER_STENCIL cli_calc_fuse_user_stencil_d
#define FK_MATMUL cli_calc_fuse_matmul_block_d
#define FK_EYE cli_calc_fuse_eye_block_d
#include "cli_calc_fuse_kernels.h"
#undef FT
#undef FK_DT
#undef FK
#undef FM
#undef FK_LOAD
#undef FK_STORE
#undef FK_LOADV
#undef FK_STOREV
#undef FK_SHIFT
#undef FK_LUT
#undef FK_MEDFILT
#undef FK_GATHER
#undef FK_INTERP2
#undef FK_USER_BLK
#undef FK_USER_STENCIL
#undef FK_MATMUL
#undef FK_EYE

/* block kernels in float32 (images FLOAT or 8/16-bit integers: twice the SIMD width, float libm) */
#define FT float
#define FK_DT _DATATYPE_FLOAT
#define FK(name) name##_f
#define FM(fn) fn##f
#define FK_LOAD load_block_f
#define FK_STORE store_block_f
#define FK_LOADV loadv_block_f
#define FK_STOREV storev_block_f
#define FK_SHIFT cli_calc_fuse_shift_block_f
#define FK_LUT cli_calc_fuse_lut_block_f
#define FK_MEDFILT cli_calc_fuse_medfilt_block_f
#define FK_GATHER cli_calc_fuse_gather_block_f
#define FK_INTERP2 cli_calc_fuse_interp2_block_f
#define FK_USER_BLK cli_calc_fuse_user_blk_f
#define FK_USER_STENCIL cli_calc_fuse_user_stencil_f
#define FK_MATMUL cli_calc_fuse_matmul_block_f
#define FK_EYE cli_calc_fuse_eye_block_f
#include "cli_calc_fuse_kernels.h"
#undef FT
#undef FK_DT
#undef FK
#undef FM
#undef FK_LOAD
#undef FK_STORE
#undef FK_LOADV
#undef FK_STOREV
#undef FK_SHIFT
#undef FK_LUT
#undef FK_MEDFILT
#undef FK_GATHER
#undef FK_INTERP2
#undef FK_USER_BLK
#undef FK_USER_STENCIL
#undef FK_MATMUL
#undef FK_EYE

/**
 * cli_calc_fuse_eval_block_d - Evaluate program @p on double stack for pixels [@off, @off + @len)
 * @p:   Compiled fused program
 * @off: Starting pixel index
 * @len: Number of pixels (`<= FUSE_BLK`)
 * @stk: Per-thread double vector stack
 *
 * Return: void
 */
void cli_calc_fuse_eval_block_d(const fprog *p, uint64_t off, int len, double (*stk)[FUSE_BLK])
{
    eval_block_d(p, off, len, stk);
}

/**
 * cli_calc_fuse_eval_block_f - Evaluate program @p on float stack for pixels [@off, @off + @len)
 * @p:   Compiled fused program
 * @off: Starting pixel index
 * @len: Number of pixels (`<= FUSE_BLK`)
 * @stk: Per-thread float vector stack
 *
 * Return: void
 */
void cli_calc_fuse_eval_block_f(const fprog *p, uint64_t off, int len, float (*stk)[FUSE_BLK])
{
    eval_block_f(p, off, len, stk);
}

/**
 * cli_calc_fuse_all_float_prog - Return 1 if @p can execute entirely in single-precision float
 * @p: Compiled fused program
 *
 * Return: 1 if all inputs and constants fit float32 execution, 0 otherwise.
 */
/** float_exact_dt() - 1 if every value of datatype @dt is exact in float32: FLOAT and the 8/16-bit ints. */
static int float_exact_dt(uint8_t dt)
{
    return dt == _DATATYPE_FLOAT || dt == _DATATYPE_UINT8 || dt == _DATATYPE_INT8 ||
           dt == _DATATYPE_UINT16 || dt == _DATATYPE_INT16;
}

int cli_calc_fuse_all_float_prog(const fprog *p)
{
    if (p->out_double || (p->out_dt != 0 && !float_exact_dt(p->out_dt)))
    {
        return 0;
    }
    for (int i = 0; i < p->nimg; i++)
    {
        uint8_t dt = p->img[i]->md[0].datatype;
        if (!float_exact_dt(dt) && dt != _DATATYPE_COMPLEX_FLOAT)
        {
            return 0;
        }
    }
    for (int i = 0; i < p->n; i++)
    {
        const fins *in = &p->ins[i];
        if (in->k == FI_BIN &&
            ((in->op >= TOK_OP_BAND && in->op <= TOK_OP_BXOR) || in->op == TOK_OP_LSHIFT ||
             in->op == TOK_OP_RSHIFT) &&
            in->cpos != 0 && fabs(in->c) > 16777216.0)
        {
            return 0;
        }
        /* the float stencil, table and gather kernels read array.F directly */
        if ((in->k == FI_SHIFT || in->k == FI_MEDFILT || in->k == FI_USER_STENCIL ||
             in->k == FI_LUT || in->k == FI_GATHER || in->k == FI_INTERP2) &&
            p->img[in->img]->md[0].datatype != _DATATYPE_FLOAT)
        {
            return 0;
        }
        if (in->k == FI_MATMUL &&
            (p->img[p->matmul[in->cpos].imgA]->md[0].datatype != _DATATYPE_FLOAT ||
             p->img[p->matmul[in->cpos].imgB]->md[0].datatype != _DATATYPE_FLOAT))
        {
            return 0;
        }
    }
    return 1;
}

/**
 * cli_calc_fuse_all_float - Return 1 if @p and all output targets (@out and `extra_dst`) are float32
 * @p:   Compiled fused program
 * @out: Primary destination image
 *
 * Return: 1 if single-precision execution is valid, 0 otherwise.
 */
int cli_calc_fuse_all_float(const fprog *p, const IMAGE *out)
{
    if (!float_exact_dt(out->md[0].datatype) || !cli_calc_fuse_all_float_prog(p))
    {
        return 0;
    }
    for (int d = 0; d < p->nextra_dst; d++)
    {
        if (p->extra_dst[d] == NULL || !float_exact_dt(p->extra_dst[d]->md[0].datatype))
        {
            return 0;
        }
    }
    return 1;
}

typedef struct
{
    const fprog *p;
    IMAGE       *out;
    uint64_t     nel;
    uint64_t     blk_sz;
    uint64_t     nbx;
    uint64_t     nx;
    uint64_t     ny;
} run_job_t;

/** run_job_f() - team job: blocks [lo, hi) of a program on the float32 stack. */
static void run_job_f(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const run_job_t       *j   = (const run_job_t *) ctx;
    uint64_t               bsz = j->blk_sz;
    MILK_ALIGNED(64) float stk[FUSE_MAXSTK][FUSE_BLK];
    (void) tid;
    (void) nth;
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t off = b * bsz;
        int      len = (int) ((j->nel - off < bsz) ? j->nel - off : bsz);
        run_block_f(j->p, j->out, off, len, stk);
    }
}

/** run_job_d() - team job: blocks [lo, hi) of a program on the double stack. */
static void run_job_d(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const run_job_t        *j   = (const run_job_t *) ctx;
    uint64_t                bsz = j->blk_sz;
    MILK_ALIGNED(64) double stk[FUSE_MAXSTK][FUSE_BLK];
    (void) tid;
    (void) nth;
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t off = b * bsz;
        int      len = (int) ((j->nel - off < bsz) ? j->nel - off : bsz);
        run_block_d(j->p, j->out, off, len, stk);
    }
}

/** run_tile_job_f() - team job: 2D 64x16 cache-tiled blocks [lo, hi) on the float32 stack. */
static void run_tile_job_f(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const run_job_t       *j  = (const run_job_t *) ctx;
    uint64_t               nx = j->nx;
    uint64_t               ny = j->ny;
    MILK_ALIGNED(64) float stk[FUSE_MAXSTK][FUSE_BLK];
    (void) tid;
    (void) nth;
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t bx   = b % j->nbx;
        uint64_t by   = b / j->nbx;
        uint64_t x0   = bx * 64;
        uint64_t y0   = by * 16;
        uint64_t xlen = (nx - x0 < 64) ? (nx - x0) : 64;
        uint64_t y1   = (ny - y0 < 16) ? ny : (y0 + 16);
        for (uint64_t y = y0; y < y1; y++)
        {
            run_block_f(j->p, j->out, y * nx + x0, (int) xlen, stk);
        }
    }
}

/** run_tile_job_d() - team job: 2D 64x16 cache-tiled blocks [lo, hi) on the double stack. */
static void run_tile_job_d(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const run_job_t        *j  = (const run_job_t *) ctx;
    uint64_t                nx = j->nx;
    uint64_t                ny = j->ny;
    MILK_ALIGNED(64) double stk[FUSE_MAXSTK][FUSE_BLK];
    (void) tid;
    (void) nth;
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t bx   = b % j->nbx;
        uint64_t by   = b / j->nbx;
        uint64_t x0   = bx * 64;
        uint64_t y0   = by * 16;
        uint64_t xlen = (nx - x0 < 64) ? (nx - x0) : 64;
        uint64_t y1   = (ny - y0 < 16) ? ny : (y0 + 16);
        for (uint64_t y = y0; y < y1; y++)
        {
            run_block_d(j->p, j->out, y * nx + x0, (int) xlen, stk);
        }
    }
}

/**
 * cli_calc_fuse_run_program - Execute a compiled fused program across all blocks of @out
 * @p:   Compiled fused program
 * @out: Destination image
 *
 * Return: 0 on success, -1 on runtime error.
 */
int cli_calc_fuse_run_program(fprog *p, IMAGE *out)
{
    double sval[FUSE_MAXINS];
    if (cli_calc_fuse_eval_scalars(p, sval) != 0 ||
        cli_calc_fuse_refresh_views(p, sval, out) != 0 || cli_calc_fuse_finalize_views(p) != 0 ||
        cli_calc_fuse_run_axes(p) != 0)
    {
        return -1;
    }

    /* Snapshot @out if it aliases a shifted, sliced, or lookup-table input operand */
    IMAGE snap_im;
    int   need_snap = 0;
    for (int i = 0; i < p->nimg; i++)
    {
        if (p->img[i] == out &&
            (p->view[i].is_shift || p->view[i].is_view || p->view[i].is_lut || p->dstv.is_view))
        {
            need_snap = 1;
            break;
        }
    }
    if (need_snap)
    {
        size_t bytes =
            (size_t) out->md[0].nelement * (size_t) ImageStreamIO_typesize(out->md[0].datatype);
        if (bytes > p->alias_cap)
        {
            void *nb = realloc(p->alias_buf, bytes);
            if (nb == NULL)
            {
                return -1;
            }
            p->alias_buf = nb;
            p->alias_cap = bytes;
        }
        memcpy(p->alias_buf, out->array.raw, bytes);
        snap_im           = *out;
        snap_im.array.raw = p->alias_buf;
        for (int i = 0; i < p->nimg; i++)
        {
            if (p->img[i] == out)
            {
                p->img[i] = &snap_im;
            }
        }
    }

    uint64_t nel    = p->inel;
    uint64_t nblk   = 0;
    uint64_t blk_sz = FUSE_BLK;
    int      nth    = 1;
    fprog_choose_grid(nel, fprog_effective_nel(p), &nth, &nblk, &blk_sz);
    if (p->nmatmul > 0)
    {
        arith_blas_set_threads(1);
        if (cli_calc_fuse_prep_matmuls(p, nth) != 0)
        {
            return -1;
        }
    }

    int use_tile = 0;
    if (p->inaxis == 2 && p->ishape[0] >= 128 && p->ishape[1] >= 16 && p->nmatmul == 0)
    {
        for (int i = 0; i < p->nimg; i++)
        {
            if (p->view[i].is_view && (p->view[i].stride[0] >= 64 || p->view[i].stride[0] <= -64))
            {
                use_tile = 1;
                break;
            }
        }
    }

    int is_flt = cli_calc_fuse_all_float(p, out);
    if (use_tile)
    {
        uint64_t nbx  = (p->ishape[0] + 63) / 64;
        uint64_t nby  = (p->ishape[1] + 15) / 16;
        uint64_t ntil = nbx * nby;
        if ((uint64_t) nth > ntil)
        {
            nth = (int) ((ntil > 0) ? ntil : 1);
        }
        run_job_t job   = { p, out, nel, FUSE_BLK, nbx, p->ishape[0], p->ishape[1] };
        uint64_t  chunk = (ntil > (uint64_t) nth * 8) ? (ntil / ((uint64_t) nth * 8)) : 1;
        cli_calc_team_run_chunks(nth, ntil, chunk, is_flt ? run_tile_job_f : run_tile_job_d, &job);
    }
    else
    {
        run_job_t job = { p, out, nel, blk_sz, 0, 0, 0 };
        cli_calc_team_run(nth, nblk, is_flt ? run_job_f : run_job_d, &job);
    }

    if (need_snap)
    {
        for (int i = 0; i < p->nimg; i++)
        {
            if (p->img[i] == &snap_im)
            {
                p->img[i] = out;
            }
        }
    }
    return 0;
}
