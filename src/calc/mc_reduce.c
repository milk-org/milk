// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/arith_team.h"
#include "COREMOD_arith/image_coord.h"
#include "COREMOD_arith/image_stats.h"
#include "mc_plan.h"

#define MC_RED_BLK 1024

typedef struct
{
    const mc_expr_t *expr;
    mcalc_prog_t    *prog;
} sub_cache_entry_t;

static sub_cache_entry_t g_sub_cache[16];
static int               g_in_cache_clear = 0;

void mc_reduce_cache_clear(void)
{
    if (g_in_cache_clear)
    {
        return;
    }
    g_in_cache_clear = 1;
    for (int i = 0; i < 16; i++)
    {
        if (g_sub_cache[i].prog != NULL)
        {
            (void) (g_sub_cache[i].prog->ast_head != NULL &&
                    (g_sub_cache[i].prog->ast_head->u.expr = NULL, 1));
            mcalc_prog_free(g_sub_cache[i].prog);
            g_sub_cache[i].prog = NULL;
        }
        g_sub_cache[i].expr = NULL;
    }
    g_in_cache_clear = 0;
}

void mc_reduce_cache_evict_expr(const mc_expr_t *e)
{
    if (e == NULL || g_in_cache_clear)
    {
        return;
    }
    g_in_cache_clear = 1;
    for (int i = 0; i < 16; i++)
    {
        if (g_sub_cache[i].expr == e && g_sub_cache[i].prog != NULL)
        {
            (void) (g_sub_cache[i].prog->ast_head != NULL &&
                    (g_sub_cache[i].prog->ast_head->u.expr = NULL, 1));
            mcalc_prog_free(g_sub_cache[i].prog);
            g_sub_cache[i].prog = NULL, g_sub_cache[i].expr = NULL;
        }
    }
    g_in_cache_clear = 0;
}

__attribute__((format(printf, 3, 4))) static int red_err(char       *err,
                                                         size_t      errsz,
                                                         const char *fmt,
                                                         ...)
{
    if (err != NULL && errsz > 0 && err[0] == '\0')
    {
        va_list ap;
        va_start(ap, fmt), vsnprintf(err, errsz, fmt, ap), va_end(ap);
    }
    return -1;
}

static inline int is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}

int mc_map_reduce_op(const char *name)
{
    return (strcmp(name, "itot") == 0)                                ? ARITH_RED_SUM
           : (strcmp(name, "imean") == 0)                             ? ARITH_RED_MEAN
           : (strcmp(name, "imin") == 0)                              ? ARITH_RED_MIN
           : (strcmp(name, "imax") == 0)                              ? ARITH_RED_MAX
           : (strcmp(name, "istd") == 0 || strcmp(name, "irms") == 0) ? ARITH_RED_STD
           : (strcmp(name, "ivar") == 0)                              ? ARITH_RED_VAR
           : (strcmp(name, "norm") == 0 || strcmp(name, "nrm2") == 0) ? ARITH_RED_NORM
           : (strcmp(name, "asum") == 0)                              ? ARITH_RED_ASUM
           : (strcmp(name, "icount") == 0)                            ? ARITH_RED_COUNT
           : (strcmp(name, "dot") == 0)                               ? ARITH_RED_DOT
                                                                      : ARITH_RED_ALL;
}

static const mc_expr_t *unwrap_def_arg(const mc_expr_t *e, const mc_expr_t *call_site)
{
    return (e != NULL && e->kind == MC_EXPR_IDENT && e->u.ident.def_param_idx >= 0 &&
            call_site != NULL && call_site->kind == MC_EXPR_CALL &&
            e->u.ident.def_param_idx < call_site->u.call.nargs)
               ? call_site->u.call.args[e->u.ident.def_param_idx]
               : e;
}

int mc_eval_scalar_d(const mc_expr_t *e,
                     const mc_expr_t *call_site,
                     double          *out,
                     char            *err,
                     size_t           errsz)
{
    mc_val_t v;
    if (mc_scalar_eval_expr(unwrap_def_arg(e, call_site), &v, err, errsz) != 0 ||
        v.type == MC_VAL_STRING)
    {
        return (v.type == MC_VAL_STRING) ? red_err(err, errsz, "expected numeric argument") : -1;
    }
    *out = (v.type == MC_VAL_LONG) ? (double) v.lval : v.dval;
    return 0;
}

static imageID clone_to_tmp(imageID src_id)
{
    const IMAGE *src =
        (src_id >= 0 && milk_data.image[src_id].used) ? &milk_data.image[src_id] : NULL;
    imageID tid = src ? mc_tmp_image(src->md[0].naxis, src->md[0].size, src->md[0].datatype) : -1;
    (void) (tid >= 0 && memcpy(milk_data.image[tid].array.raw, src->array.raw,
                               (size_t) src->md[0].nelement *
                                   (size_t) ImageStreamIO_typesize(src->md[0].datatype)));
    return tid;
}

static mcalc_prog_t *get_or_build_sub_prog(const mc_expr_t *expr, char *err, size_t errsz)
{
    for (int i = 0; i < 16; i++)
    {
        if (g_sub_cache[i].expr == expr && g_sub_cache[i].prog != NULL)
        {
            return g_sub_cache[i].prog;
        }
    }
    mcalc_prog_t *p = calloc(1, sizeof(*p));
    mc_stmt_t    *s = calloc(1, sizeof(*s));
    p->stmt_to_step = malloc(sizeof(int));
    if (!p || !s || !p->stmt_to_step)
    {
        free(s), mcalc_prog_free(p);
        return NULL;
    }
    s->kind = MC_STMT_EXPR, s->u.expr = (mc_expr_t *) expr, s->needs_image = 1;
    p->ast_head = s, p->nstmts = 1, p->stmt_to_step[0] = -1, p->last_im_id = -1;
    if (mc_plan_build(p, 0, err, errsz) != 0)
    {
        s->u.expr = NULL, mcalc_prog_free(p);
        return NULL;
    }
    int slot = 15;
    for (int i = 0; i < 16; i++)
    {
        if (g_sub_cache[i].prog == NULL)
        {
            slot = i;
            break;
        }
    }
    if (g_sub_cache[slot].prog != NULL)
    {
        mcalc_prog_t *old      = g_sub_cache[slot].prog;
        g_sub_cache[slot].prog = NULL, g_sub_cache[slot].expr = NULL;
        (void) (old->ast_head != NULL && (old->ast_head->u.expr = NULL, 1));
        mcalc_prog_free(old);
    }
    g_sub_cache[slot].expr = expr, g_sub_cache[slot].prog = p;
    return p;
}

void mc_release_red_src(mc_red_src_t *s)
{
    (void) (s->tmp_id >= 0 && (mc_tmp_release(s->tmp_id), 1));
    s->tmp_id = -1;
}

static int fill_red_src_im(mc_red_src_t *out, imageID id, char *err, size_t errsz)
{
    out->im        = &milk_data.image[id];
    out->dt        = out->im->md[0].datatype;
    out->naxis     = out->im->md[0].naxis;
    out->ishape[0] = out->im->md[0].size[0];
    out->ishape[1] = (out->naxis > 1) ? out->im->md[0].size[1] : 1;
    out->ishape[2] = (out->naxis > 2) ? out->im->md[0].size[2] : 1;
    out->nel       = out->im->md[0].nelement;
    return is_complex_dt(out->dt)
               ? (mc_release_red_src(out),
                  red_err(err, errsz,
                          "Function of real pixels applied to a complex image; use real(), "
                          "imag() or abs() first"))
               : 0;
}

int mc_prepare_red_src(const mc_expr_t *e,
                       const mc_expr_t *call_site,
                       imageID          avoid_id,
                       int              allow_sub_pass,
                       mc_red_src_t    *out,
                       char            *err,
                       size_t           errsz)
{
    out->im = NULL, out->pass = NULL, out->tmp_id = -1;
    e = unwrap_def_arg(e, call_site);
    if (e == NULL)
    {
        return red_err(err, errsz, "null image expression");
    }
    if (e->kind == MC_EXPR_IDENT)
    {
        imageID id = e->u.ident.im_id;
        if (id < 0 || id >= milk_data.NB_MAX_IMAGE || !milk_data.image[id].used ||
            strcmp(milk_data.image[id].name, e->u.ident.bare) != 0)
        {
            id = mc_resolve_or_autoconnect_image(e->u.ident.name);
        }
        if (id < 0)
        {
            return red_err(err, errsz, "unknown image '%s'", e->u.ident.name);
        }
        if (id == avoid_id && (id = out->tmp_id = clone_to_tmp(id)) < 0)
        {
            return red_err(err, errsz, "failed to clone aliased image");
        }
        return fill_red_src_im(out, id, err, errsz);
    }

    mcalc_prog_t *sub = get_or_build_sub_prog(e, err, errsz);
    if (sub == NULL)
    {
        return -1;
    }
    if (allow_sub_pass && sub->nsteps == 1 && sub->steps[0].kind == MC_STEP_PASS &&
        sub->steps[0].pass.is_1d_contig && sub->steps[0].pass.nouts == 1 &&
        !sub->steps[0].pass.is_complex && sub->steps[0].pass.nops <= 32)
    {
        mc_pass_t *pa     = &sub->steps[0].pass;
        int        can_l1 = (avoid_id < 0);
        uint64_t   nel    = 0;
        uint8_t    nax    = 0;
        uint32_t   ish[3] = { 1, 1, 1 };
        out->out_op = pa->outs[0].op_idx, out->dt = pa->outs[0].out_dt;
        memset(out->ops, 0, (size_t) pa->nops * sizeof(mc_operand_t));
        can_l1 &= (out->dt == _DATATYPE_FLOAT || out->dt == _DATATYPE_DOUBLE);
        for (int i = 0; can_l1 && i < pa->nops; i++)
        {
            const mc_op_desc_t *d = &pa->ops[i];
            if (d->is_out)
            {
                continue;
            }
            if (d->is_scalar)
            {
                out->ops[i].scalar = (d->slice_expr != NULL && d->scalar_expr == NULL)
                                         ? (double) arith_next_rng_seed()
                                         : 0.0;
                if ((d->scalar_expr != NULL &&
                     mc_eval_scalar_d(d->scalar_expr, NULL, &out->ops[i].scalar, err, errsz) !=
                         0) ||
                    (d->check_div_kind != 0 && out->ops[i].scalar == 0.0))
                {
                    return (d->check_div_kind != 0 && out->ops[i].scalar == 0.0)
                               ? red_err(err, errsz,
                                         (d->check_div_kind == 1) ? "Division by zero"
                                                                  : "Modulo by zero")
                               : -1;
                }
                continue;
            }
            if (d->has_slice || d->is_expr_im || d->is_non_pointwise)
            {
                can_l1 = 0;
                break;
            }
            imageID id = sub->slots[d->slot_idx].im_id;
            if (id < 0 || !milk_data.image[id].used)
            {
                id = mc_resolve_or_autoconnect_image(sub->slots[d->slot_idx].name);
            }
            if (id < 0 || (nel != 0 && milk_data.image[id].md[0].nelement != nel))
            {
                can_l1 = 0;
                break;
            }
            sub->slots[d->slot_idx].im_id = id;
            const IMAGE *im               = &milk_data.image[id];
            nel = im->md[0].nelement, nax = im->md[0].naxis;
            ish[0] = im->md[0].size[0], ish[1] = (nax > 1) ? im->md[0].size[1] : 1;
            ish[2] = (nax > 2) ? im->md[0].size[2] : 1, out->ops[i].base = im->array.raw;
        }
        if (can_l1 && nel > 0)
        {
            out->pass = pa, out->naxis = nax, out->nel = nel;
            memcpy(out->ishape, ish, sizeof(ish));
            return 0;
        }
    }

    if (mcalc_prog_run(sub, err, errsz) != 0 || sub->last_im_id < 0)
    {
        return -1;
    }
    out->tmp_id = clone_to_tmp(sub->last_im_id);
    mc_tmp_release(sub->last_im_id);
    return (out->tmp_id < 0) ? red_err(err, errsz, "failed to allocate temporary image")
                             : fill_red_src_im(out, out->tmp_id, err, errsz);
}

static inline void eval_src_block_f(const mc_red_src_t *s,
                                    uint64_t            off,
                                    int                 len,
                                    float *restrict dst)
{
    if (s->pass != NULL)
    {
        mc_operand_t lops[32];
        memcpy(lops, s->ops, (size_t) s->pass->nops * sizeof(mc_operand_t));
        lops[s->out_op].base = (void *) ((uintptr_t) dst - (uintptr_t) off * sizeof(float));
        s->pass->fn(lops, off, off + (uint64_t) len);
        return;
    }
    if (s->dt == _DATATYPE_FLOAT)
    {
        memcpy(dst, s->im->array.F + off, (size_t) len * sizeof(float));
        return;
    }
    for (int i = 0; i < len; i++)
    {
        dst[i] = (float) arith_read_im_pix(s->im, s->dt, off + (uint64_t) i);
    }
}

void mc_eval_src_block_d(const mc_red_src_t *s, uint64_t off, int len, double *restrict dst)
{
    if (s->pass != NULL)
    {
        if (s->dt == _DATATYPE_DOUBLE)
        {
            mc_operand_t lops[32];
            memcpy(lops, s->ops, (size_t) s->pass->nops * sizeof(mc_operand_t));
            lops[s->out_op].base = (void *) ((uintptr_t) dst - (uintptr_t) off * sizeof(double));
            s->pass->fn(lops, off, off + (uint64_t) len);
            return;
        }
        MILK_ALIGNED(64) float bf[MC_RED_BLK];
        eval_src_block_f(s, off, len, bf);
        for (int i = 0; i < len; i++)
        {
            dst[i] = (double) bf[i];
        }
        return;
    }
    if (s->dt == _DATATYPE_DOUBLE)
    {
        memcpy(dst, s->im->array.D + off, (size_t) len * sizeof(double));
        return;
    }
    for (int i = 0; i < len; i++)
    {
        dst[i] = arith_read_im_pix(s->im, s->dt, off + (uint64_t) i);
    }
}

/* Sterbenz-shifted 128-float SIMD block accumulation + Chan double merge */
static inline void stat_block_sterbenz_f(arith_reduce_op op,
                                         const float *restrict v,
                                         const float *restrict m,
                                         int               len,
                                         arith_stat_acc_t *acc)
{
    if (op == ARITH_RED_DOT)
    {
        float s = 0.0f;
#pragma omp simd reduction(+ : s)
        for (int j = 0; j < len; j++)
        {
            s += v[j] * m[j];
        }
        acc->sum += (double) s, acc->n += (uint64_t) len;
        return;
    }
    if (m != NULL && len == MC_RED_BLK &&
        (op == ARITH_RED_SUM || op == ARITH_RED_MEAN || op == ARITH_RED_COUNT))
    {
        float bsum = 0.0f, bcnt = 0.0f;
#pragma omp simd reduction(+ : bsum, bcnt)
        for (int j = 0; j < len; j++)
        {
            float mj = (m[j] != 0.0f) ? 1.0f : 0.0f;
            bsum += v[j] * mj, bcnt += mj;
        }
        if (LIKELY(arith_isfinite_f(bsum)))
        {
            acc->n += (uint64_t) bcnt, acc->sum += (double) bsum;
            acc->mean = (acc->n > 0) ? (acc->sum / (double) acc->n) : NAN;
            return;
        }
    }
    if (m != NULL || (op != ARITH_RED_STD && op != ARITH_RED_VAR && op != ARITH_RED_ALL))
    {
        arith_stat_block_f_inline(op, v, m, len, acc);
        return;
    }
    for (int sub = 0; sub < len; sub += 128)
    {
        int          slen = (len - sub < 128) ? (len - sub) : 128;
        const float *sv   = v + sub;
        float        ref = sv[0], ds = 0.0f, dm2 = 0.0f, bmin = ref, bmax = ref;
#pragma omp simd reduction(+ : ds, dm2) reduction(min : bmin) reduction(max : bmax)
        for (int j = 0; j < slen; j++)
        {
            float x = sv[j], d = x - ref;
            ds += d, dm2 += d * d, bmin = (x < bmin) ? x : bmin, bmax = (x > bmax) ? x : bmax;
        }
        if (UNLIKELY(!arith_isfinite_f(ref) || !arith_isfinite_f(ds) || !arith_isfinite_f(dm2)))
        {
            arith_stat_block_f_inline(op, sv, NULL, slen, acc);
            continue;
        }
        double           dn = (double) slen, dds = (double) ds, sb = dn * (double) ref + dds;
        double           bm = (double) ref + dds / dn, m2 = (double) dm2 - (dds * dds) / dn;
        arith_stat_acc_t blk = { (uint64_t) slen, sb,           bm, (m2 > 0.0) ? m2 : 0.0,
                                 (double) bmin,   (double) bmax };
        arith_stat_merge_inline(acc, &blk);
    }
}

typedef struct
{
    const mc_red_src_t *s1;
    const mc_red_src_t *s2;
    arith_reduce_op     op;
    uint64_t            nel;
    int                 use_flt;
    arith_stat_acc_t    acc;
} stat_job_t;

static void stat_job_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    stat_job_t      *j   = (stat_job_t *) ctx;
    arith_stat_acc_t loc = { 0, 0.0, 0.0, 0.0, NAN, NAN };
    if (j->use_flt)
    {
        MILK_ALIGNED(64) float b1[MC_RED_BLK], b2[MC_RED_BLK];
        const float           *p1 =
            (j->s1->im != NULL && j->s1->dt == _DATATYPE_FLOAT) ? j->s1->im->array.F : NULL;
        const float *p2 = (j->s2 != NULL && j->s2->im != NULL && j->s2->dt == _DATATYPE_FLOAT)
                              ? j->s2->im->array.F
                              : NULL;
        for (uint64_t b = lo; b < hi; b++)
        {
            uint64_t     off = b * MC_RED_BLK;
            int          len = (int) ((j->nel - off < MC_RED_BLK) ? (j->nel - off) : MC_RED_BLK);
            const float *v1  = p1 ? (p1 + off) : (eval_src_block_f(j->s1, off, len, b1), b1);
            const float *v2  = !j->s2 ? NULL
                               : p2   ? (p2 + off)
                                      : (eval_src_block_f(j->s2, off, len, b2), b2);
            stat_block_sterbenz_f(j->op, v1, v2, len, &loc);
        }
    }
    else
    {
        MILK_ALIGNED(64) double b1[MC_RED_BLK], b2[MC_RED_BLK];
        const double           *p1 =
            (j->s1->im != NULL && j->s1->dt == _DATATYPE_DOUBLE) ? j->s1->im->array.D : NULL;
        const double *p2 = (j->s2 != NULL && j->s2->im != NULL && j->s2->dt == _DATATYPE_DOUBLE)
                               ? j->s2->im->array.D
                               : NULL;
        for (uint64_t b = lo; b < hi; b++)
        {
            uint64_t      off = b * MC_RED_BLK;
            int           len = (int) ((j->nel - off < MC_RED_BLK) ? (j->nel - off) : MC_RED_BLK);
            const double *v1  = p1 ? (p1 + off) : (mc_eval_src_block_d(j->s1, off, len, b1), b1);
            const double *v2  = !j->s2 ? NULL
                                : p2   ? (p2 + off)
                                       : (mc_eval_src_block_d(j->s2, off, len, b2), b2);
            arith_stat_block_d_inline(j->op, v1, v2, len, &loc);
        }
    }
    if (nth > 1)
    {
        arith_team_lock(), arith_stat_merge_inline(&j->acc, &loc), arith_team_unlock();
    }
    else
    {
        j->acc = loc;
    }
}

static int run_stat_reduce(const mc_red_src_t *s1,
                           const mc_red_src_t *s2,
                           arith_reduce_op     op,
                           arith_stat_acc_t   *out_acc,
                           char               *err,
                           size_t              errsz)
{
    if (s2 != NULL && (s1->naxis != s2->naxis || memcmp(s1->ishape, s2->ishape, 12) != 0))
    {
        return red_err(err, errsz,
                       (op == ARITH_RED_DOT) ? "dot: images must have the same dimensions"
                                             : "The mask must have the image's size");
    }
    uint64_t nblk = (s1->nel + MC_RED_BLK - 1) / MC_RED_BLK;
    int      nth  = milk_omp_team_size(s1->nel, 262144);
    nth           = ((uint64_t) nth > nblk) ? (int) nblk : nth;
    if (nth <= 1 && s1->im != NULL && s1->dt == _DATATYPE_FLOAT &&
        (!s2 || (s2->im != NULL && s2->dt == _DATATYPE_FLOAT)))
    {
        arith_stat_acc_t loc = { 0, 0.0, 0.0, 0.0, NAN, NAN };
        const float     *p1 = s1->im->array.F, *p2 = s2 ? s2->im->array.F : NULL;
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * MC_RED_BLK;
            int      len = (int) ((s1->nel - off < MC_RED_BLK) ? (s1->nel - off) : MC_RED_BLK);
            stat_block_sterbenz_f(op, p1 + off, p2 ? (p2 + off) : NULL, len, &loc);
        }
        *out_acc = loc;
        return 0;
    }
    int d1 = (s1->dt == _DATATYPE_FLOAT || s1->dt == _DATATYPE_UINT8 || s1->dt == _DATATYPE_INT8 ||
              s1->dt == _DATATYPE_UINT16 || s1->dt == _DATATYPE_INT16);
    int d2 = (s2 == NULL || s2->dt == _DATATYPE_FLOAT || s2->dt == _DATATYPE_UINT8 ||
              s2->dt == _DATATYPE_INT8 || s2->dt == _DATATYPE_UINT16 || s2->dt == _DATATYPE_INT16);
    stat_job_t job = { s1, s2, op, s1->nel, d1 && d2, { 0, 0.0, 0.0, 0.0, NAN, NAN } };
    arith_team_run(nth, nblk, stat_job_cb, &job);
    *out_acc = job.acc;
    return 0;
}

int mc_run_stat_max(const mc_red_src_t *s, double *out_max, char *err, size_t errsz)
{
    arith_stat_acc_t acc;
    return (run_stat_reduce(s, NULL, ARITH_RED_MAX, &acc, err, errsz) != 0)
               ? -1
               : ((*out_max = arith_stat_finish_inline(&acc, ARITH_RED_MAX, 0)), 0);
}

typedef struct
{
    const mc_red_src_t *s1;
    const mc_red_src_t *s2;
    int                 mode; /* 0 = argmax, 1 = argmin, 2 = iamax */
    uint64_t            nel;
    double              best_v;
    int64_t             best_i;
} argext_job_t;

static void argext_job_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    argext_job_t           *j = (argext_job_t *) ctx;
    MILK_ALIGNED(64) double b1[MC_RED_BLK], b2[MC_RED_BLK];
    double                  bv = 0.0;
    int64_t                 bi = -1;
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t off = b * MC_RED_BLK;
        int      len = (int) ((j->nel - off < MC_RED_BLK) ? (j->nel - off) : MC_RED_BLK);
        mc_eval_src_block_d(j->s1, off, len, b1);
        (void) (j->s2 != NULL && (mc_eval_src_block_d(j->s2, off, len, b2), 1));
        for (int q = 0; q < len; q++)
        {
            double raw = b1[q];
            if (arith_isnan_d(raw) || (j->s2 != NULL && (b2[q] == 0.0 || arith_isnan_d(b2[q]))))
            {
                continue;
            }
            double v = (j->mode == 2) ? fabs(raw) : (j->mode == 1) ? -raw : raw;
            if (bi < 0 || v > bv)
            {
                bv = v;
                bi = (int64_t) (off + (uint64_t) q);
            }
        }
    }
    arith_team_lock();
    if (bi >= 0 && (j->best_i < 0 || bv > j->best_v || (bv == j->best_v && bi < j->best_i)))
    {
        j->best_v = bv;
        j->best_i = bi;
    }
    arith_team_unlock();
}

static int run_argext_reduce(const mc_red_src_t *s1,
                             const mc_red_src_t *s2,
                             int                 mode,
                             int64_t            *out_idx,
                             char               *err,
                             size_t              errsz)
{
    if (s2 != NULL && (s1->naxis != s2->naxis || memcmp(s1->ishape, s2->ishape, 12) != 0))
    {
        return red_err(err, errsz, "The mask must have the image's size");
    }
    uint64_t nblk    = (s1->nel + MC_RED_BLK - 1) / MC_RED_BLK;
    int      nth     = milk_omp_team_size(s1->nel, 262144);
    nth              = ((uint64_t) nth > nblk) ? (int) nblk : nth;
    argext_job_t job = { s1, s2, mode, s1->nel, 0.0, -1 };
    arith_team_run(nth, nblk, argext_job_cb, &job);
    *out_idx = job.best_i;
    return 0;
}

int mc_exec_reduce_scalar(const mc_expr_t *call_expr, mc_val_t *out, char *err, size_t errsz)
{
    const char  *fn    = call_expr->u.call.fn->name;
    int          nargs = call_expr->u.call.nargs, rc = 0;
    mc_red_src_t s1, s2;
    s1.tmp_id = s2.tmp_id = -1;

    if (strcmp(fn, "imedian") == 0 || strcmp(fn, "perc") == 0 || strcmp(fn, "trace") == 0)
    {
        if (mc_prepare_red_src(call_expr->u.call.args[0], NULL, -1, 0, &s1, err, errsz) != 0)
        {
            return -1;
        }
        double res = 0.0, frac = 0.5;
        if (strcmp(fn, "trace") == 0)
        {
            uint64_t M = s1.ishape[0], N = (s1.naxis >= 2) ? s1.ishape[1] : 1;
            res = arith_image_reduce_lane(s1.im, s1.dt, 0, (M < N) ? M : N, M + 1, ARITH_RED_SUM);
        }
        else if (strcmp(fn, "perc") == 0 &&
                 mc_eval_scalar_d(call_expr->u.call.args[1], NULL, &frac, err, errsz) != 0)
        {
            rc = -1;
        }
        else if ((strcmp(fn, "imedian") == 0 && nargs == 2) ||
                 (strcmp(fn, "perc") == 0 && nargs == 3))
        {
            const mc_expr_t *mk_e = call_expr->u.call.args[(strcmp(fn, "perc") == 0) ? 2 : 1];
            if (mc_prepare_red_src(mk_e, NULL, -1, 0, &s2, err, errsz) != 0 ||
                s1.naxis != s2.naxis || memcmp(s1.ishape, s2.ishape, 12) != 0)
            {
                rc = (err[0] == '\0') ? red_err(err, errsz, "The mask must have the image's size")
                                      : -1;
            }
            else
            {
                res = arith_image_reduce_masked_im(s1.im, s2.im, ARITH_RED_PERC, frac);
            }
        }
        else
        {
            res = arith_image_percentile(s1.im->name, frac);
        }
        mc_release_red_src(&s1), mc_release_red_src(&s2);
        out->type = MC_VAL_DOUBLE, out->dval = res;
        return rc;
    }

    if (mc_prepare_red_src(call_expr->u.call.args[0], NULL, -1, 1, &s1, err, errsz) != 0)
    {
        return -1;
    }
    int has_im2 = (nargs >= 2 && call_expr->u.call.args[1]->res_kind == MC_RES_IMAGE);
    if (has_im2 && mc_prepare_red_src(call_expr->u.call.args[1], NULL, -1, 1, &s2, err, errsz) != 0)
    {
        mc_release_red_src(&s1);
        return -1;
    }
    if (!has_im2 && nargs >= 2)
    {
        double axd = 0.0;
        if (mc_eval_scalar_d(call_expr->u.call.args[1], NULL, &axd, err, errsz) != 0 ||
            llround(axd) != 0)
        {
            mc_release_red_src(&s1);
            return (err[0] == '\0') ? red_err(err, errsz, "Axis out of range") : -1;
        }
    }

    if (strcmp(fn, "iargmax") == 0 || strcmp(fn, "iargmin") == 0 || strcmp(fn, "iamax") == 0)
    {
        int64_t idx  = -1;
        int     mode = (strcmp(fn, "iargmin") == 0) ? 1 : (strcmp(fn, "iamax") == 0) ? 2 : 0;
        rc           = run_argext_reduce(&s1, has_im2 ? &s2 : NULL, mode, &idx, err, errsz);
        mc_release_red_src(&s1), mc_release_red_src(&s2);
        out->type = MC_VAL_LONG, out->lval = (long) idx;
        return rc;
    }

    arith_stat_acc_t acc;
    arith_reduce_op  op      = (arith_reduce_op) mc_map_reduce_op(fn);
    int              is_long = (call_expr->u.call.fn->kind == MC_FN_REDUCE_L);
    rc                       = run_stat_reduce(&s1, has_im2 ? &s2 : NULL, op, &acc, err, errsz);
    mc_release_red_src(&s1), mc_release_red_src(&s2);
    double r  = (rc == 0) ? arith_stat_finish_inline(&acc, op, is_long) : 0.0;
    out->type = is_long ? MC_VAL_LONG : MC_VAL_DOUBLE, out->lval = (long) r, out->dval = r;
    return rc;
}

int mc_exec_reduce_pair(const mc_expr_t *c0,
                        const mc_expr_t *c1,
                        double          *o0,
                        double          *o1,
                        char            *err,
                        size_t           errsz)
{
    arith_reduce_op op0  = (arith_reduce_op) mc_map_reduce_op(c0->u.call.fn->name);
    arith_reduce_op op1  = (arith_reduce_op) mc_map_reduce_op(c1->u.call.fn->name);
    int             comb = arith_reduce_op_promote(op0, op1);
    if (comb < 0)
    {
        return -1;
    }
    mc_red_src_t     s1;
    arith_stat_acc_t acc;
    s1.tmp_id = -1;
    if (mc_prepare_red_src(c0->u.call.args[0], NULL, -1, 1, &s1, err, errsz) != 0)
    {
        return -1;
    }
    int rc = run_stat_reduce(&s1, NULL, (arith_reduce_op) comb, &acc, err, errsz);
    mc_release_red_src(&s1);
    if (rc == 0)
    {
        *o0 = arith_stat_finish_inline(&acc, op0, 0);
        *o1 = arith_stat_finish_inline(&acc, op1, 0);
    }
    return rc;
}

typedef struct
{
    const mc_red_src_t *s1;
    uint64_t            nx;
    uint64_t            ny;
    double              thresh;
    double              sums[3];
} cent_job_t;

static void cent_job_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    cent_job_t             *j = (cent_job_t *) ctx;
    MILK_ALIGNED(64) double blk[MC_RED_BLK];
    MILK_ALIGNED(64) float  blkf[MC_RED_BLK];
    double                  si = 0.0, sx = 0.0, sy = 0.0, th = j->thresh;
    uint64_t                nx = j->nx, ny = j->ny, nel = j->s1->nel;
    int                     is_flt = (j->s1->dt == _DATATYPE_FLOAT);
    const float            *im_f   = (is_flt && j->s1->im != NULL) ? j->s1->im->array.F : NULL;
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t     off = b * MC_RED_BLK;
        int          len = (int) ((nel - off < MC_RED_BLK) ? (nel - off) : MC_RED_BLK);
        const float *pf =
            is_flt ? (im_f ? (im_f + off) : (eval_src_block_f(j->s1, off, len, blkf), blkf)) : NULL;
        (void) (!is_flt && (mc_eval_src_block_d(j->s1, off, len, blk), 1));
        uint64_t x_ = off % nx, y_ = (off / nx) % ny;
        int      J = 0;
        while (J < len)
        {
            int N = (int) ((nx - x_ < (uint64_t) (len - J)) ? nx - x_ : (uint64_t) (len - J));
            if (is_flt && th == (double) (float) th)
            {
                float        ri = 0.0f, rxi = 0.0f, x0 = (float) x_, thf = (float) th;
                const float *s = pf + J;
#pragma omp simd reduction(+ : ri, rxi)
                for (int i = 0; i < N; i++)
                {
                    float v  = s[i] - thf;
                    float ip = (v > 0.0f) ? v : 0.0f;
                    ri += ip;
                    rxi += (x0 + (float) i) * ip;
                }
                si += (double) ri, sx += (double) rxi, sy += (double) y_ * (double) ri;
            }
            else
            {
                double        ri = 0.0, rxi = 0.0, x0 = (double) x_;
                const double *s = blk + J;
#pragma omp simd reduction(+ : ri, rxi)
                for (int i = 0; i < N; i++)
                {
                    double v  = (is_flt ? (double) pf[J + i] : s[i]) - th;
                    double ip = (v > 0.0) ? v : 0.0;
                    ri += ip;
                    rxi += (x0 + (double) i) * ip;
                }
                si += ri, sx += rxi, sy += (double) y_ * ri;
            }
            J += N, x_ = 0;
            if (++y_ == ny)
            {
                y_ = 0;
            }
        }
    }
    arith_team_lock();
    j->sums[0] += si, j->sums[1] += sx, j->sums[2] += sy;
    arith_team_unlock();
}

int mc_exec_multi_reduce(const mc_stmt_t *s, mc_val_t *out, char *err, size_t errsz)
{
    const mc_expr_t *call = s->u.multi.rhs[0];
    const char      *fn   = call->u.call.fn->name;
    int              nlhs = s->u.multi.nlhs, nargs = call->u.call.nargs, rc = 0;
    mc_red_src_t     s1, s2;
    mc_val_t         vals[4];
    s1.tmp_id = s2.tmp_id = -1;

    if (mc_prepare_red_src(call->u.call.args[0], NULL, -1, 1, &s1, err, errsz) != 0)
    {
        return -1;
    }
    if (strcmp(fn, "istats") == 0)
    {
        if (nargs == 2 &&
            mc_prepare_red_src(call->u.call.args[1], NULL, -1, 1, &s2, err, errsz) != 0)
        {
            mc_release_red_src(&s1);
            return -1;
        }
        arith_stat_acc_t      acc;
        const arith_reduce_op ops[4] = { ARITH_RED_MEAN, ARITH_RED_STD, ARITH_RED_MIN,
                                         ARITH_RED_MAX };
        rc = run_stat_reduce(&s1, (nargs == 2) ? &s2 : NULL, ARITH_RED_ALL, &acc, err, errsz);
        for (int j = 0; rc == 0 && j < nlhs; j++)
        {
            vals[j].type = MC_VAL_DOUBLE, vals[j].dval = arith_stat_finish_inline(&acc, ops[j], 0);
        }
    }
    else if (strcmp(fn, "icentroid") == 0)
    {
        double th = 0.0;
        if (nargs == 2 && mc_eval_scalar_d(call->u.call.args[1], NULL, &th, err, errsz) != 0)
        {
            mc_release_red_src(&s1);
            return -1;
        }
        uint64_t   nblk = (s1.nel + MC_RED_BLK - 1) / MC_RED_BLK;
        int        nth  = milk_omp_team_size(s1.nel, 262144);
        cent_job_t job  = { &s1, s1.ishape[0], s1.ishape[1], th, { 0.0, 0.0, 0.0 } };
        arith_team_run(((uint64_t) nth > nblk) ? (int) nblk : nth, nblk, cent_job_cb, &job);
        double si    = job.sums[0];
        double cv[3] = { (si > 0.0) ? (job.sums[1] / si) : NAN,
                         (si > 0.0) ? (job.sums[2] / si) : NAN, (si > 0.0) ? si : 0.0 };
        for (int j = 0; j < nlhs; j++)
        {
            vals[j].type = MC_VAL_DOUBLE, vals[j].dval = cv[j];
        }
    }
    else
    {
        if (nlhs > (int) s1.naxis ||
            (nargs == 2 &&
             mc_prepare_red_src(call->u.call.args[1], NULL, -1, 1, &s2, err, errsz) != 0))
        {
            mc_release_red_src(&s1);
            return (nlhs > (int) s1.naxis)
                       ? red_err(err, errsz, "%s: cannot unpack %d coordinates from %uD image", fn,
                                 nlhs, (unsigned) s1.naxis)
                       : -1;
        }
        int64_t flat = -1;
        rc = run_argext_reduce(&s1, (nargs == 2) ? &s2 : NULL, (strcmp(fn, "iargmin") == 0) ? 1 : 0,
                               &flat, err, errsz);
        int64_t nx = (int64_t) s1.ishape[0], ny = (int64_t) s1.ishape[1];
        int64_t coords[3] = { (flat < 0) ? -1 : (flat % nx), (flat < 0) ? -1 : ((flat / nx) % ny),
                              (flat < 0) ? -1 : (flat / (nx * ny)) };
        for (int j = 0; j < nlhs; j++)
        {
            vals[j].type = MC_VAL_LONG, vals[j].lval = (long) coords[j];
        }
    }
    mc_release_red_src(&s1), mc_release_red_src(&s2);
    for (int j = 0; rc == 0 && j < nlhs; j++)
    {
        mc_expr_t *lhs = s->u.multi.lhs[j];
        mc_store_scalar_var(lhs->u.ident.bare, &lhs->u.ident.var_id, &vals[j]);
        out->type = vals[j].type, out->lval = vals[j].lval, out->dval = vals[j].dval;
    }
    return rc;
}
