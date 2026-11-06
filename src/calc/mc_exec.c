// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/arith_team.h"
#include "COREMOD_arith/image_coord.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/image_ID.h"
#include "mc_plan.h"

static uint64_t g_torn_count = 0;

uint64_t mcalc_torn_count(void)
{
    return __atomic_load_n(&g_torn_count, __ATOMIC_RELAXED);
}

__attribute__((format(printf, 3, 4))) static int exec_err(char       *err,
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

static inline int is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}

static void write_im_pix(IMAGE *im, uint8_t dt, uint64_t idx, double v)
{
    int64_t  si = llround(v);
    uint64_t ui = (v <= 0.0) ? 0 : (uint64_t) (v + 0.5);
#define CLAMP_S(lo, hi) ((si < (lo)) ? (lo) : (si > (hi)) ? (hi) : si)
    (void) ((dt == _DATATYPE_FLOAT && ((im->array.F[idx] = (float) v), 1)) ||
            (dt == _DATATYPE_DOUBLE && ((im->array.D[idx] = v), 1)) ||
            (dt == _DATATYPE_UINT8 &&
             ((im->array.UI8[idx] = (ui > 255) ? 255 : (uint8_t) ui), 1)) ||
            (dt == _DATATYPE_INT8 && ((im->array.SI8[idx] = (int8_t) CLAMP_S(-128, 127)), 1)) ||
            (dt == _DATATYPE_UINT16 &&
             ((im->array.UI16[idx] = (ui > 65535) ? 65535 : (uint16_t) ui), 1)) ||
            (dt == _DATATYPE_INT16 &&
             ((im->array.SI16[idx] = (int16_t) CLAMP_S(-32768, 32767)), 1)) ||
            (dt == _DATATYPE_UINT32 &&
             ((im->array.UI32[idx] = (ui > 4294967295ULL) ? 4294967295U : (uint32_t) ui), 1)) ||
            (dt == _DATATYPE_INT32 &&
             ((im->array.SI32[idx] = (int32_t) CLAMP_S(-2147483648LL, 2147483647)), 1)) ||
            (dt == _DATATYPE_UINT64 &&
             ((im->array.UI64[idx] = (v <= 0.0) ? 0 : (uint64_t) v), 1)) ||
            (dt == _DATATYPE_INT64 && ((im->array.SI64[idx] = (int64_t) v), 1)));
#undef CLAMP_S
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

static imageID eval_expr_to_tmp_im(const mc_expr_t *expr,
                                   int              in_unit,
                                   int              unsqueeze_red,
                                   char            *err,
                                   size_t           errsz)
{
    mc_stmt_t fake_stmt = { .kind = MC_STMT_EXPR, .u.expr = (mc_expr_t *) expr, .needs_image = 1 };
    int       step_map = -1, rc = 0;
    mcalc_prog_t p      = { .ast_head = &fake_stmt, .nstmts = 1, .stmt_to_step = &step_map };
    imageID      res_id = -1;

    if (mc_plan_build(&p, 0, err, errsz) == 0)
    {
        for (int s = 0; s < p.nsteps && rc == 0; s++)
        {
            mc_val_t v;
            (void) (unsqueeze_red && p.steps[s].kind == MC_STEP_CTOR &&
                    (p.steps[s].ctor.is_anon = 2));
            rc = (p.steps[s].kind == MC_STEP_SCALAR)
                     ? mc_scalar_exec_stmt(p.steps[s].scalar_stmt, &v, err, errsz)
                 : (p.steps[s].kind == MC_STEP_CTOR)
                     ? mc_exec_ctor(&p, &p.steps[s].ctor, in_unit, err, errsz)
                     : mc_exec_pass(&p, &p.steps[s].pass, in_unit, err, errsz);
        }
        const mc_step_t *last = (rc == 0 && p.nsteps > 0) ? &p.steps[p.nsteps - 1] : NULL;
        int              sidx = !last                          ? -1
                                : (last->kind == MC_STEP_CTOR) ? last->ctor.slot_idx
                                : (last->kind == MC_STEP_PASS && last->pass.nouts > 0)
                                    ? last->pass.outs[last->pass.nouts - 1].slot_idx
                                    : -1;
        if (sidx >= 0 && p.slots[sidx].im_id >= 0)
        {
            res_id = clone_to_tmp(p.slots[sidx].im_id);
            mc_tmp_release(p.slots[sidx].im_id);
        }
    }
    for (int s = 0; s < p.nsteps; s++)
    {
        (void) (p.steps[s].kind == MC_STEP_CTOR && (mc_ctor_step_free(&p.steps[s].ctor), 1));
        (void) (p.steps[s].kind == MC_STEP_PASS &&
                (free(p.steps[s].pass.ops), free(p.steps[s].pass.outs), free(p.steps[s].pass.c_src),
                 1));
    }
    free(p.steps);
    free(p.slots);
    return res_id;
}

static const mc_expr_t *unwrap_def_arg(const mc_expr_t *e, const mc_expr_t *call_site)
{
    return (e != NULL && e->kind == MC_EXPR_IDENT && e->u.ident.def_param_idx >= 0 &&
            call_site != NULL && call_site->kind == MC_EXPR_CALL &&
            e->u.ident.def_param_idx < call_site->u.call.nargs)
               ? call_site->u.call.args[e->u.ident.def_param_idx]
               : e;
}

static int resolve_or_eval_im(const mc_expr_t *e,
                              const mc_expr_t *call_site,
                              imageID          avoid_id,
                              int              in_unit,
                              imageID         *out_id,
                              int             *is_tmp,
                              char            *err,
                              size_t           errsz)
{
    e       = unwrap_def_arg(e, call_site);
    *out_id = -1;
    *is_tmp = 0;
    if (e == NULL)
    {
        return exec_err(err, errsz, "null image expression");
    }
    if (e->kind != MC_EXPR_IDENT)
    {
        *out_id = eval_expr_to_tmp_im(e, in_unit, 0, err, errsz);
        *is_tmp = (*out_id >= 0);
        return (*out_id >= 0) ? 0 : -1;
    }
    imageID id = e->u.ident.im_id;
    if (id < 0 || id >= milk_data.NB_MAX_IMAGE || !milk_data.image[id].used ||
        strcmp(milk_data.image[id].name, e->u.ident.bare) != 0)
    {
        id = mc_resolve_or_autoconnect_image(e->u.ident.name);
    }
    *out_id = (id < 0) ? -1 : (id == avoid_id) ? clone_to_tmp(id) : id;
    *is_tmp = (id >= 0 && id == avoid_id && *out_id >= 0);
    return (id < 0)         ? exec_err(err, errsz, "unknown image '%s'", e->u.ident.name)
           : (*out_id >= 0) ? 0
                            : exec_err(err, errsz, "failed to clone aliased image");
}

static int eval_scalar_d(const mc_expr_t *e,
                         const mc_expr_t *call_site,
                         double          *out,
                         char            *err,
                         size_t           errsz)
{
    mc_val_t v;
    if (mc_scalar_eval_expr(unwrap_def_arg(e, call_site), &v, err, errsz) != 0 ||
        v.type == MC_VAL_STRING)
    {
        return (v.type == MC_VAL_STRING) ? exec_err(err, errsz, "expected numeric argument") : -1;
    }
    *out = (v.type == MC_VAL_LONG) ? (double) v.lval : v.dval;
    return 0;
}

static uint8_t select_ctor_out_dt(const mc_slot_t *slot, uint8_t pref_dt)
{
    return (slot->existed_at_plan && slot->im_id >= 0 && milk_data.image[slot->im_id].used &&
            milk_data.image[slot->im_id].md[0].shared == 1)
               ? milk_data.image[slot->im_id].md[0].datatype
               : (pref_dt ? pref_dt : _DATATYPE_FLOAT);
}

int mc_exec_ctor(mcalc_prog_t *prog, mc_ctor_step_t *cs, int in_unit, char *err, size_t errsz)
{
    mc_slot_t       *slot = &prog->slots[cs->slot_idx];
    const mc_expr_t *rhs  = cs->rhs_expr;
    const mc_expr_t *call_site =
        (rhs->kind == MC_EXPR_CALL && rhs->u.call.udef_id >= 0 && rhs->u.call.udef_body != NULL)
            ? rhs
            : NULL;
    rhs              = call_site ? rhs->u.call.udef_body : rhs;
    imageID avoid_id = (slot->im_id >= 0 && milk_data.image[slot->im_id].used &&
                        strcmp(milk_data.image[slot->im_id].name, slot->bare) == 0)
                           ? slot->im_id
                           : image_ID(slot->bare, milk_data.image, milk_data.NB_MAX_IMAGE);

    if (rhs->kind == MC_EXPR_CALL && rhs->u.call.fn != NULL &&
        (rhs->u.call.fn->kind == MC_FN_REDUCE_D || rhs->u.call.fn->kind == MC_FN_STENCIL_LIB))
    {
        return mc_exec_prestage(prog, cs, avoid_id, call_site, in_unit, err, errsz);
    }

    if (rhs->kind == MC_EXPR_VEC_LIT ||
        (rhs->kind == MC_EXPR_CALL && rhs->u.call.fn != NULL &&
         strcmp(rhs->u.call.fn->name, "vec") == 0 &&
         (rhs->u.call.nargs != 1 ||
          unwrap_def_arg(rhs->u.call.args[0], call_site)->res_kind != MC_RES_IMAGE)))
    {
        int n = (rhs->kind == MC_EXPR_VEC_LIT) ? rhs->u.vec_lit.nelem : rhs->u.call.nargs;
        mc_expr_t *const *elems =
            (rhs->kind == MC_EXPR_VEC_LIT) ? rhs->u.vec_lit.elems : rhs->u.call.args;
        double vals[MC_MAX_ARGS];
        for (int i = 0; i < n; i++)
        {
            if (eval_scalar_d(elems[i], call_site, &vals[i], err, errsz) != 0)
            {
                return -1;
            }
        }
        uint32_t sz[3]  = { (uint32_t) n, 1, 1 };
        uint8_t  out_dt = select_ctor_out_dt(slot, _DATATYPE_FLOAT);
        imageID  oid    = mc_out_prepare(slot, 1, sz, out_dt, in_unit, err, errsz);
        for (int i = 0; oid >= 0 && i < n; i++)
        {
            write_im_pix(&milk_data.image[oid], out_dt, (uint64_t) i, vals[i]);
        }
        return (oid < 0) ? -1 : (mc_out_post(oid), 0);
    }
    const char *fn    = (rhs->u.call.fn != NULL) ? rhs->u.call.fn->name : "";
    int         nargs = rhs->u.call.nargs;
    if (strcmp(fn, "linspace") == 0 || strcmp(fn, "zeros") == 0 || strcmp(fn, "eye") == 0)
    {
        double v[3] = { 0.0, 0.0, 0.0 };
        for (int i = 0; i < nargs; i++)
        {
            if (eval_scalar_d(rhs->u.call.args[i], call_site, &v[i], err, errsz) != 0)
            {
                return -1;
            }
        }
        int      is_ls = (strcmp(fn, "linspace") == 0), is_eye = (strcmp(fn, "eye") == 0);
        uint32_t sz[3] = { 1, 1, 1 };
        if (is_ls && (llround(v[2]) <= 0 || llround(v[2]) > 1048576))
        {
            return exec_err(err, errsz, "linspace: n must be in [1, 1048576]");
        }
        for (int i = 0; !is_ls && i < nargs; i++)
        {
            if (llround(v[i]) <= 0)
            {
                return exec_err(err, errsz, "%s: dimensions must be positive", fn);
            }
            sz[i] = (uint32_t) llround(v[i]);
        }
        sz[0] = is_ls ? (uint32_t) llround(v[2]) : sz[0];
        (void) (is_eye && nargs == 1 && (sz[1] = sz[0]));
        uint8_t out_dt = select_ctor_out_dt(slot, _DATATYPE_FLOAT);
        imageID oid = mc_out_prepare(slot, is_ls ? 1 : (is_eye ? 2 : (uint8_t) nargs), sz, out_dt,
                                     in_unit, err, errsz);
        if (oid < 0)
        {
            return -1;
        }
        IMAGE *out = &milk_data.image[oid];
        if (!is_ls)
        {
            memset(out->array.raw, 0,
                   (size_t) out->md[0].nelement * (size_t) ImageStreamIO_typesize(out_dt));
            for (uint32_t i = 0, dlen = (sz[0] < sz[1]) ? sz[0] : sz[1]; is_eye && i < dlen; i++)
            {
                write_im_pix(out, out_dt, (uint64_t) i * sz[0] + i, 1.0);
            }
            mc_out_post(oid);
            return 0;
        }
        double step = (sz[0] > 1) ? (v[1] - v[0]) / (double) (sz[0] - 1) : 0.0;
        for (uint32_t i = 0; i < sz[0]; i++)
        {
            write_im_pix(out, out_dt, i,
                         (sz[0] == 1)       ? v[0]
                         : (i == sz[0] - 1) ? v[1]
                                            : v[0] + step * (double) i);
        }
        mc_out_post(oid);
        return 0;
    }

    imageID src_id = -1, src2_id = -1;
    int     src_tmp = 0, src2_tmp = 0, rc = 0;
    if (resolve_or_eval_im(rhs->u.call.args[0], call_site, avoid_id, in_unit, &src_id, &src_tmp,
                           err, errsz) != 0)
    {
        return -1;
    }
    const IMAGE *src = &milk_data.image[src_id];

    if (rhs->u.call.ext_id >= 0)
    {
        const milk_calc_ext_entry_t *ext = milk_calc_ext_get(rhs->u.call.ext_id);
        uint32_t sz[3] = { src->md[0].size[0], (src->md[0].naxis > 1) ? src->md[0].size[1] : 1,
                           (src->md[0].naxis > 2) ? src->md[0].size[2] : 1 };
        imageID  oid =
            (ext != NULL && ext->blk1_fn != NULL && src->md[0].datatype == _DATATYPE_FLOAT)
                ? mc_out_prepare(slot, src->md[0].naxis, sz, _DATATYPE_FLOAT, in_unit, err, errsz)
                : -1;
        uint8_t pbuf[MILK_CALC_PARAM_BYTES] = { 0 };
        rc = (oid < 0) ? -1
                       : (ext->blk1_fn(milk_data.image[oid].array.F, src->array.F, pbuf,
                                       (int) src->md[0].nelement),
                          mc_out_post(oid), 0);
    }
    else if (strcmp(fn, "vec") == 0 || strcmp(fn, "reshape") == 0)
    {
        uint64_t nel = src->md[0].nelement, prod_pos = 1;
        uint8_t  ndims = 1;
        uint32_t sz[3] = { (uint32_t) nel, 1, 1 };
        if (strcmp(fn, "reshape") == 0)
        {
            ndims        = (uint8_t) (nargs - 1);
            long dims[3] = { 1, 1, 1 };
            int  neg_idx = -1;
            for (int i = 0; i < (int) ndims; i++)
            {
                double dv = 0.0;
                if (eval_scalar_d(rhs->u.call.args[i + 1], call_site, &dv, err, errsz) != 0)
                {
                    rc = -1;
                    goto done_src;
                }
                long d = llround(dv);
                if (d <= 0 && !(d == -1 && neg_idx == -1))
                {
                    rc = exec_err(err, errsz,
                                  (d == -1) ? "reshape: at most one dimension may be -1"
                                            : "reshape: dimensions must be positive");
                    goto done_src;
                }
                neg_idx = (d == -1) ? i : neg_idx;
                dims[i] = (d == -1) ? 1 : d;
                prod_pos *= (uint64_t) dims[i];
            }
            (void) (neg_idx != -1 && prod_pos > 0 && nel % prod_pos == 0 &&
                    ((dims[neg_idx] = (long) (nel / prod_pos)), (prod_pos = nel)));
            if (prod_pos != nel)
            {
                rc = exec_err(err, errsz, "reshape: element count mismatch");
                goto done_src;
            }
            sz[0] = (uint32_t) dims[0], sz[1] = (uint32_t) dims[1], sz[2] = (uint32_t) dims[2];
        }
        uint8_t out_dt = select_ctor_out_dt(slot, src->md[0].datatype);
        imageID oid    = mc_out_prepare(slot, ndims, sz, out_dt, in_unit, err, errsz);
        rc             = (oid < 0 || arith_image_reshape_im(src, &milk_data.image[oid]) != 0)
                             ? -1
                             : (mc_out_post(oid), 0);
    }
    else if (strcmp(fn, "transpose") == 0)
    {
        if (src->md[0].naxis != 1 && src->md[0].naxis != 2)
        {
            rc = exec_err(err, errsz, "transpose: expected 1D or 2D image");
            goto done_src;
        }
        uint32_t sz[3] = { (src->md[0].naxis == 2) ? src->md[0].size[1] : 1, src->md[0].size[0],
                           1 };
        uint8_t  odt   = select_ctor_out_dt(
            slot, (src->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
        imageID oid = mc_out_prepare(slot, 2, sz, odt, in_unit, err, errsz);
        rc          = (oid < 0 || arith_image_transpose_im(src, &milk_data.image[oid]) != 0)
                          ? -1
                          : (mc_out_post(oid), 0);
    }
    else if (strcmp(fn, "outer") == 0)
    {
        uint64_t M = 0, N = 0;
        if (resolve_or_eval_im(rhs->u.call.args[1], call_site, avoid_id, in_unit, &src2_id,
                               &src2_tmp, err, errsz) != 0)
        {
            rc = -1;
            goto done_src;
        }
        const IMAGE *src2 = &milk_data.image[src2_id];
        if (!arith_image_is_1d_vec(src, &M) || !arith_image_is_1d_vec(src2, &N))
        {
            rc = exec_err(err, errsz, "outer: expected 1D vector arguments");
            goto done_src;
        }
        uint32_t sz[3] = { (uint32_t) M, (uint32_t) N, 1 };
        uint8_t  odt   = select_ctor_out_dt(slot, (src->md[0].datatype == _DATATYPE_DOUBLE ||
                                                   src2->md[0].datatype == _DATATYPE_DOUBLE)
                                                      ? _DATATYPE_DOUBLE
                                                      : _DATATYPE_FLOAT);
        imageID  oid   = mc_out_prepare(slot, 2, sz, odt, in_unit, err, errsz);
        rc = (oid < 0 || arith_image_ger_im(src, src2, 1.0, 0.0, &milk_data.image[oid]) != 0)
                 ? -1
                 : (mc_out_post(oid), 0);
    }
    else if (strcmp(fn, "diag") == 0)
    {
        double kd = 0.0;
        if ((nargs == 2 && eval_scalar_d(rhs->u.call.args[1], call_site, &kd, err, errsz) != 0) ||
            (src->md[0].naxis != 1 && src->md[0].naxis != 2))
        {
            rc = (src->md[0].naxis != 1 && src->md[0].naxis != 2)
                     ? exec_err(err, errsz, "diag: expected 1D or 2D image")
                     : -1;
            goto done_src;
        }
        long     k = llround(kd), absk = (k < 0) ? -k : k;
        uint8_t  sdt = src->md[0].datatype;
        uint8_t  odt = select_ctor_out_dt(slot, (sdt == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE
                                                                          : _DATATYPE_FLOAT);
        uint64_t r0 = (k < 0) ? (uint64_t) absk : 0, c0 = (k > 0) ? (uint64_t) absk : 0;
        int      is_1d = (src->md[0].naxis == 1);
        uint64_t M = src->md[0].size[0], N = is_1d ? (M + (uint64_t) absk) : src->md[0].size[1];
        if (!is_1d && (r0 >= M || c0 >= N))
        {
            rc = exec_err(err, errsz, "diag: diagonal offset out of range");
            goto done_src;
        }
        uint64_t L     = is_1d ? M : ((M - r0 < N - c0) ? (M - r0) : (N - c0));
        uint32_t sz[3] = { (uint32_t) (is_1d ? N : L), (uint32_t) (is_1d ? N : 1), 1 };
        imageID  oid   = mc_out_prepare(slot, is_1d ? 2 : 1, sz, odt, in_unit, err, errsz);
        if (oid < 0)
        {
            rc = -1;
            goto done_src;
        }
        IMAGE *out = &milk_data.image[oid];
        (void) (is_1d &&
                memset(out->array.raw, 0,
                       (size_t) out->md[0].nelement * (size_t) ImageStreamIO_typesize(odt)));
        for (uint64_t i = 0; i < L; i++)
        {
            write_im_pix(out, odt, is_1d ? ((c0 + i) * N + (r0 + i)) : i,
                         arith_read_im_pix(src, sdt, is_1d ? i : ((c0 + i) * M + (r0 + i))));
        }
        mc_out_post(oid);
    }

done_src:
    (void) (src_tmp && (mc_tmp_release(src_id), 1));
    (void) (src2_tmp && (mc_tmp_release(src2_id), 1));
    return rc;
}

static int expr_equiv(const mc_expr_t *a, const mc_expr_t *b)
{
    if (a == b)
    {
        return 1;
    }
    if (!a || !b || a->kind != b->kind)
    {
        return 0;
    }
    if (a->kind == MC_EXPR_IDENT)
    {
        return strcmp(a->u.ident.bare, b->u.ident.bare) == 0;
    }
    if (a->kind == MC_EXPR_LONG)
    {
        return a->u.lval == b->u.lval;
    }
    if (a->kind == MC_EXPR_DOUBLE)
    {
        return a->u.dval == b->u.dval;
    }
    if (a->kind == MC_EXPR_UNARY)
    {
        return a->u.unary.op == b->u.unary.op && expr_equiv(a->u.unary.operand, b->u.unary.operand);
    }
    if (a->kind == MC_EXPR_BINARY)
    {
        return a->u.binary.op == b->u.binary.op && expr_equiv(a->u.binary.left, b->u.binary.left) &&
               expr_equiv(a->u.binary.right, b->u.binary.right);
    }
    return 0;
}

typedef struct
{
    mc_kernel_fn  fn;
    mc_operand_t *ops;
} pass_team_ctx_t;

static void pass_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    const pass_team_ctx_t *job = (const pass_team_ctx_t *) ctx;
    (void) (hi > lo && (job->fn(job->ops, lo, hi), 1));
}

int mc_exec_pass(mcalc_prog_t *prog, mc_pass_t *pass, int in_unit, char *err, size_t errsz)
{
    mc_operand_t ops_s[32],
        *ops = (pass->nops <= 32) ? ops_s : calloc((size_t) pass->nops, sizeof(*ops));
    mc_view_t views_s[32],
        *views = (pass->nops <= 32) ? views_s : calloc((size_t) pass->nops, sizeof(*views));
    imageID  expr_ims[64], snap_ims[64], div_snap[32];
    uint64_t cnt0_pre[64];
    int      rc = 0;

    if (!ops || !views || pass->nops > 64 || pass->nouts > 32)
    {
        rc = exec_err(err, errsz, "pass operand limit exceeded");
        goto cleanup;
    }
    memset(ops, 0, (size_t) pass->nops * sizeof(*ops));
    memset(views, 0, (size_t) pass->nops * sizeof(*views));
    memset(expr_ims, 0xff, (size_t) pass->nops * sizeof(imageID));
    memset(snap_ims, 0xff, (size_t) pass->nops * sizeof(imageID));
    memset(div_snap, 0xff, (size_t) pass->nouts * sizeof(imageID));

    int sc_done[64] = { 0 };
    for (int i = 0; i < pass->nops; i++)
    {
        views[i].im_id = views[i].owned_tmp_id = -1;
        const mc_op_desc_t *d                  = &pass->ops[i];
        if (!d->is_scalar)
        {
            continue;
        }
        const mc_expr_t *se = d->scalar_expr;
        if (!sc_done[i] && se != NULL && se->kind == MC_EXPR_CALL && se->u.call.fn != NULL &&
            se->u.call.fn->kind == MC_FN_REDUCE_D && se->u.call.nargs == 1)
        {
            for (int j = i + 1; j < pass->nops; j++)
            {
                const mc_expr_t *sj = pass->ops[j].is_scalar ? pass->ops[j].scalar_expr : NULL;
                if (!sc_done[j] && sj != NULL && sj->kind == MC_EXPR_CALL &&
                    sj->u.call.fn != NULL && sj->u.call.fn->kind == MC_FN_REDUCE_D &&
                    sj->u.call.nargs == 1 && expr_equiv(se->u.call.args[0], sj->u.call.args[0]))
                {
                    if (mc_exec_reduce_pair(se, sj, &ops[i].scalar, &ops[j].scalar, err, errsz) !=
                        0)
                    {
                        rc = -1;
                        goto cleanup;
                    }
                    sc_done[i] = sc_done[j] = 1;
                    break;
                }
            }
        }
        if (!sc_done[i])
        {
            ops[i].scalar =
                (d->slice_expr != NULL && se == NULL) ? (double) arith_next_rng_seed() : 0.0;
            if (se != NULL && eval_scalar_d(se, NULL, &ops[i].scalar, err, errsz) != 0)
            {
                rc = -1;
                goto cleanup;
            }
        }
        if (d->check_div_kind != 0 && ops[i].scalar == 0.0)
        {
            rc = exec_err(err, errsz,
                          (d->check_div_kind == 1)   ? "Division by zero"
                          : (d->check_div_kind == 2) ? "Modulo by zero"
                                                     : "lut: dx must be non-zero");
            goto cleanup;
        }
    }

    uint8_t  vnaxis = 0, rhs_naxis = 0;
    uint32_t vsz[3] = { 1, 1, 1 }, rhs_sz[3] = { 1, 1, 1 };
    for (int g = 0; pass->is_comp && pass->comp_expr != NULL && g < pass->comp_expr->u.comp.ngens;
         g++)
    {
        vnaxis        = (uint8_t) pass->comp_expr->u.comp.ngens;
        double bnd[3] = { 0.0, 0.0, 1.0 };
        for (int i = 0; i < pass->nops; i++)
        {
            int gi = pass->ops[i].comp_gen_idx - g * 3;
            (void) (gi >= 0 && gi < 3 && (bnd[gi] = ops[i].scalar, 1));
        }
        long la = llround(bnd[0]), lb = llround(bnd[1]), lst = llround(bnd[2]);
        if (lst == 0 || (lst > 0 && lb < la) || (lst < 0 && lb > la))
        {
            rc = exec_err(err, errsz,
                          (lst == 0) ? "comprehension step cannot be zero"
                                     : "comprehension range is empty");
            goto cleanup;
        }
        vsz[g] = (uint32_t) ((lst > 0) ? ((lb - la) / lst + 1) : ((la - lb) / (-lst) + 1));
    }

    int has_lhs_anchor = 0;
    for (int j = 0; j < pass->nouts; j++)
    {
        const mc_pass_out_t *out = &pass->outs[j];
        if (out->is_elided || (out->lhs_slice == NULL && out->assign_op == MC_TOK_ASSIGN))
        {
            continue;
        }
        mc_slot_t *slot = &prog->slots[out->slot_idx];
        imageID    id   = (slot->im_id >= 0 && milk_data.image[slot->im_id].used)
                              ? slot->im_id
                              : mc_resolve_or_autoconnect_image(slot->name);
        if (id < 0 ||
            mc_view_resolve(out->lhs_slice, id, 1, &views[out->op_idx], err, errsz) != 0 ||
            mc_view_merge_shape(&vnaxis, vsz, has_lhs_anchor, &views[out->op_idx], err, errsz) != 0)
        {
            rc = (id < 0) ? exec_err(err, errsz, "unknown image '%s'", slot->name) : -1;
            goto cleanup;
        }
        slot->im_id    = id;
        has_lhs_anchor = 1;
    }

    int has_expr_im = 0;
    for (int phase = 0; phase < 2 && (phase == 0 || has_expr_im); phase++)
    {
        int unsqueeze = (has_lhs_anchor || vsz[0] > 1 || vsz[1] > 1 || vsz[2] > 1);
        for (int i = 0; i < pass->nops; i++)
        {
            const mc_op_desc_t *d = &pass->ops[i];
            has_expr_im |= d->is_expr_im;
            if (d->is_scalar || d->is_out || (d->is_expr_im != phase))
            {
                continue;
            }
            if (d->is_expr_im && (expr_ims[i] = eval_expr_to_tmp_im(d->slice_expr, in_unit,
                                                                    unsqueeze, err, errsz)) < 0)
            {
                rc = -1;
                goto cleanup;
            }
            imageID id = d->is_expr_im ? expr_ims[i] : prog->slots[d->slot_idx].im_id;
            if ((id < 0 || !milk_data.image[id].used) &&
                (id = mc_resolve_or_autoconnect_image(prog->slots[d->slot_idx].name)) < 0)
            {
                rc = exec_err(err, errsz, "unknown image '%s'", prog->slots[d->slot_idx].name);
                goto cleanup;
            }
            (void) (!d->is_expr_im && (prog->slots[d->slot_idx].im_id = id, 1));
            for (int j = 0; !d->is_expr_im && j < pass->nouts; j++)
            {
                IMAGE *oim = &milk_data.image[id];
                if (pass->outs[j].slot_idx == d->slot_idx && !pass->outs[j].is_elided &&
                    (pass->nouts > 1 || d->has_slice || d->is_non_pointwise ||
                     pass->outs[j].lhs_slice != NULL ||
                     oim->md[0].datatype != pass->outs[j].out_dt ||
                     (vnaxis > 0 && pass->outs[j].lhs_slice == NULL &&
                      (oim->md[0].naxis != vnaxis || oim->md[0].size[0] != vsz[0] ||
                       (vnaxis > 1 && oim->md[0].size[1] != vsz[1]) ||
                       (vnaxis > 2 && oim->md[0].size[2] != vsz[2])))))
                {
                    id = snap_ims[i] = clone_to_tmp(id);
                    break;
                }
            }
            if (mc_view_resolve(d->has_slice ? d->slice_expr : NULL, id, 0, &views[i], err,
                                errsz) != 0 ||
                (!d->is_table_arg &&
                 (mc_view_merge_shape(&vnaxis, vsz, has_lhs_anchor, &views[i], err, errsz) != 0 ||
                  mc_view_merge_shape(&rhs_naxis, rhs_sz, 0, &views[i], err, errsz) != 0)))
            {
                rc = -1;
                goto cleanup;
            }
            (void) ((d->is_expr_im ||
                     strncmp(prog->slots[d->slot_idx].bare, "_tmpcalcps", 10) == 0) &&
                    !unsqueeze && (vnaxis = views[i].naxis));
        }
    }

    int can_bcast_dst = !pass->is_complex && !pass->no_dst_bcast;
    for (int i = 0; i < pass->nops; i++)
    {
        can_bcast_dst &= !pass->ops[i].is_non_pointwise;
    }
    uint64_t tgen = milk_table_generation();
    for (int j = 0; !has_lhs_anchor && !pass->is_comp && j < pass->nouts; j++)
    {
        mc_slot_t *s  = &prog->slots[pass->outs[j].slot_idx];
        imageID    id = (s->im_id >= 0 && milk_data.image[s->im_id].used &&
                         ((s->checked_gen == tgen && tgen != 0) ||
                          strcmp(milk_data.image[s->im_id].name, s->bare) == 0))
                            ? s->im_id
                            : mc_resolve_or_autoconnect_image(s->name);
        if (pass->outs[j].is_elided || strncmp(s->bare, "_tmpcalc", 8) == 0 || id < 0)
        {
            continue;
        }
        s->im_id                    = id;
        const IMAGE_METADATA *md    = &milk_data.image[id].md[0];
        uint32_t              ds[3] = { md->size[0], (md->naxis > 1) ? md->size[1] : 1,
                                        (md->naxis > 2) ? md->size[2] : 1 };
        if (vnaxis == 0 ||
            (can_bcast_dst && (md->shared == 1 || md->datatype == pass->outs[j].out_dt) &&
             (vsz[0] == 1 || vsz[0] == ds[0]) && (vsz[1] == 1 || vsz[1] == ds[1]) &&
             (vsz[2] == 1 || vsz[2] == ds[2])))
        {
            vnaxis = md->naxis;
            memcpy(vsz, ds, sizeof(vsz));
            break;
        }
    }

    for (int j = 0; j < pass->nouts; j++)
    {
        const mc_pass_out_t *out = &pass->outs[j];
        if (out->is_elided)
        {
            continue;
        }
        mc_slot_t *slot = &prog->slots[out->slot_idx];
        if (out->lhs_slice != NULL || out->assign_op != MC_TOK_ASSIGN)
        {
            (void) (out->assign_op == MC_TOK_DIV_ASSIGN &&
                    (div_snap[j] = clone_to_tmp(slot->im_id), 1));
            for (int k = 0;
                 out->lhs_slice != NULL && out->lhs_slice->u.slice.is_mask_index && k < pass->nops;
                 k++)
            {
                if (!pass->ops[k].is_scalar && !pass->ops[k].is_out && !pass->ops[k].is_table_arg &&
                    (views[k].naxis != views[out->op_idx].naxis ||
                     views[k].vnx != views[out->op_idx].vnx ||
                     views[k].vny != views[out->op_idx].vny ||
                     views[k].vnz != views[out->op_idx].vnz))
                {
                    rc = exec_err(err, errsz, "Mask size does not match target image");
                    goto cleanup;
                }
            }
            milk_data.image[slot->im_id].md[0].write = 1;
        }
        else if (vnaxis == 0 ||
                 (slot->im_id =
                      mc_out_prepare(slot, vnaxis, vsz, out->out_dt, in_unit, err, errsz)) < 0 ||
                 mc_view_resolve(NULL, slot->im_id, 1, &views[out->op_idx], err, errsz) != 0)
        {
            rc = (vnaxis == 0) ? exec_err(err, errsz, "cannot determine output image shape") : -1;
            goto cleanup;
        }
    }

    uint64_t nelem = (uint64_t) vsz[0] * (uint64_t) vsz[1] * (uint64_t) vsz[2];
    for (int i = 0; i < pass->nops; i++)
    {
        int is_elided_out = 0;
        for (int j = 0; j < pass->nouts; j++)
        {
            is_elided_out |= (pass->outs[j].op_idx == i && pass->outs[j].is_elided);
        }
        if (pass->ops[i].is_scalar || is_elided_out || views[i].dt == 0)
        {
            continue;
        }
        if (pass->ops[i].is_non_pointwise && !pass->ops[i].is_table_arg && has_lhs_anchor &&
            memcmp(rhs_sz, vsz, sizeof(vsz)) != 0 &&
            (views[i].vnx != vsz[0] || views[i].vny != vsz[1] || views[i].vnz != vsz[2]))
        {
            rc = exec_err(err, errsz, "Dimension mismatch in image assignment");
            goto cleanup;
        }
        (void) (!pass->ops[i].is_table_arg && (mc_view_apply_broadcast(&views[i], vsz), 1));
        ops[i].base = (uint8_t *) milk_data.image[views[i].im_id].array.raw +
                      views[i].offset_elem * (int64_t) ImageStreamIO_typesize(views[i].dt);
        memcpy(ops[i].st, views[i].st, sizeof(ops[i].st));
        ops[i].vnx = views[i].vnx, ops[i].vny = views[i].vny, ops[i].vnz = views[i].vnz;
    }

    uint64_t nitems = pass->is_1d_contig ? nelem : (uint64_t) vsz[1] * (uint64_t) vsz[2];
    int      nth = milk_omp_team_size(nelem, 262144), is_sh_in[64] = { 0 };
    for (int i = 0; i < pass->nops; i++)
    {
        int is_out_im = 0;
        for (int j = 0; j < pass->nouts; j++)
        {
            is_out_im |= (!pass->outs[j].is_elided &&
                          views[i].im_id == prog->slots[pass->outs[j].slot_idx].im_id);
        }
        is_sh_in[i] = (!pass->ops[i].is_scalar && !pass->ops[i].is_out && !is_out_im &&
                       views[i].dt != 0 && milk_data.image[views[i].im_id].md[0].shared == 1);
    }

    for (int attempt = 0; attempt < 4; attempt++)
    {
        for (int i = 0; i < pass->nops; i++)
        {
            (void) (is_sh_in[i] &&
                    (cnt0_pre[i] = __atomic_load_n(&milk_data.image[views[i].im_id].md[0].cnt0,
                                                   __ATOMIC_ACQUIRE),
                     1));
            (void) (pass->ops[i].is_out && ((ops[i].scalar = 0.0), 1));
        }

        pass_team_ctx_t job = { .fn = pass->fn, .ops = ops };
        (void) ((nth <= 1) ? (pass->fn(ops, 0, nitems), 1)
                           : (arith_team_run(nth, nitems, pass_team_cb, &job), 1));

        for (int j = 0; j < pass->nouts; j++)
        {
            if (pass->outs[j].assign_op == MC_TOK_DIV_ASSIGN &&
                ops[pass->outs[j].op_idx].scalar != 0.0)
            {
                IMAGE *dst = (div_snap[j] >= 0)
                                 ? &milk_data.image[prog->slots[pass->outs[j].slot_idx].im_id]
                                 : NULL;
                (void) (dst != NULL &&
                        (memcpy(dst->array.raw, milk_data.image[div_snap[j]].array.raw,
                                (size_t) dst->md[0].nelement *
                                    (size_t) ImageStreamIO_typesize(dst->md[0].datatype)),
                         (dst->md[0].write = 0), 1));
                rc = exec_err(err, errsz, "Division by zero");
                goto cleanup;
            }
        }

        int torn = 0;
        for (int i = 0; i < pass->nops; i++)
        {
            const IMAGE *vim = is_sh_in[i] ? &milk_data.image[views[i].im_id] : NULL;
            torn |= (vim != NULL &&
                     (__atomic_load_n(&vim->md[0].cnt0, __ATOMIC_ACQUIRE) != cnt0_pre[i] ||
                      __atomic_load_n(&vim->md[0].write, __ATOMIC_ACQUIRE) != 0));
        }
        if (!torn || attempt == 3)
        {
            break;
        }
        __atomic_fetch_add(&g_torn_count, 1, __ATOMIC_RELAXED);
        nanosleep(&(struct timespec) { .tv_sec = 0, .tv_nsec = 100000 }, NULL);
    }

    for (int j = 0; j < pass->nouts; j++)
    {
        (void) (!pass->outs[j].is_elided &&
                (mc_out_post(prog->slots[pass->outs[j].slot_idx].im_id), 1));
    }
    (void) (pass->is_complex && vnaxis > 0 &&
            (mc_ensure_pool_warm(vnaxis, vsz,
                                 pass->work_double ? _DATATYPE_COMPLEX_DOUBLE
                                                   : _DATATYPE_COMPLEX_FLOAT),
             1));

cleanup:
    for (int i = 0; i < pass->nops && i < 64; i++)
    {
        (void) (views[i].owned_tmp_id >= 0 && (mc_tmp_release(views[i].owned_tmp_id), 1));
        (void) (expr_ims[i] >= 0 && (mc_tmp_release(expr_ims[i]), 1));
        (void) (snap_ims[i] >= 0 && (mc_tmp_release(snap_ims[i]), 1));
        (void) (i < pass->nouts && i < 32 && div_snap[i] >= 0 && (mc_tmp_release(div_snap[i]), 1));
    }
    (void) (ops != ops_s && (free(ops), 1)), (void) (views != views_s && (free(views), 1));
    return rc;
}
