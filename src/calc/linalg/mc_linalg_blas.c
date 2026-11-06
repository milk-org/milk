// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// MILK_CMAKE_REQUEST_BLAS

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/image_ID.h"
#include "linalg/mc_linalg.h"
#include "milk_calc_ext.h"

__attribute__((format(printf, 3, 4))) static int blas_err(char       *err,
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

int mc_multi_lhs_is_image(const mc_expr_t *rhs0, int lhs_idx)
{
    if (rhs0 == NULL || rhs0->kind != MC_EXPR_CALL)
    {
        return 0;
    }
    if (rhs0->u.call.ext_id >= 0)
    {
        const milk_calc_ext_entry_t *ext = milk_calc_ext_get(rhs0->u.call.ext_id);
        const milk_calc_op_ports_t  *pts = (ext && ext->get_ports) ? ext->get_ports() : NULL;
        return (pts != NULL && lhs_idx >= 0 && lhs_idx < pts->nout)
                   ? (pts->out_ports[lhs_idx].kind == MILK_PORT_IMAGE)
                   : (lhs_idx == 0);
    }
    return rhs0->u.call.fn != NULL && (strcmp(rhs0->u.call.fn->name, "svd") == 0 ||
                                       strcmp(rhs0->u.call.fn->name, "eigh") == 0);
}

int mc_resolve_linalg_call(mc_expr_t *e,
                           int        nlhs,
                           int        any_im,
                           int        any_cplx,
                           int        any_dbl,
                           char      *err,
                           size_t     errsz)
{
    const char *fn    = e->u.call.fn->name;
    int         nargs = e->u.call.nargs;
    (void) any_im;

    if ((strcmp(fn, "svd") == 0 && nlhs != 1 && nlhs != 3) ||
        (strcmp(fn, "eigh") == 0 && nlhs != 1 && nlhs != 2))
    {
        return blas_err(err, errsz, "%s: wrong number of LHS targets", fn);
    }
    if (strcmp(fn, "einsum") == 0 && (e->u.call.args[0]->res_kind != MC_RES_STRING ||
                                      e->u.call.args[1]->res_kind != MC_RES_IMAGE ||
                                      (nargs == 3 && e->u.call.args[2]->res_kind != MC_RES_IMAGE)))
    {
        return blas_err(err, errsz, "einsum: expected einsum(\"subscripts\", A[, B])");
    }
    if (strcmp(fn, "cat") == 0)
    {
        if (e->u.call.args[0]->res_kind == MC_RES_IMAGE ||
            e->u.call.args[0]->res_kind == MC_RES_STRING)
        {
            return blas_err(err, errsz, "cat: first argument must be an integer axis (0..2)");
        }
        for (int i = 1; i < nargs; i++)
        {
            if (e->u.call.args[i]->res_kind != MC_RES_IMAGE)
            {
                return blas_err(err, errsz, "cat: expected image arguments after axis");
            }
        }
    }
    else if (strcmp(fn, "hstack") == 0 || strcmp(fn, "vstack") == 0 || strcmp(fn, "dstack") == 0)
    {
        for (int i = 0; i < nargs; i++)
        {
            if (e->u.call.args[i]->res_kind != MC_RES_IMAGE)
            {
                return blas_err(err, errsz, "%s: expected image arguments", fn);
            }
        }
    }
    else if (strcmp(fn, "rot90") == 0 &&
             (e->u.call.args[0]->res_kind != MC_RES_IMAGE ||
              (nargs == 2 && (e->u.call.args[1]->res_kind == MC_RES_IMAGE ||
                              e->u.call.args[1]->res_kind == MC_RES_STRING))))
    {
        return blas_err(err, errsz, "rot90: expected rot90(im[, k])");
    }
    else if ((strcmp(fn, "fft") == 0 || strcmp(fn, "ifft") == 0 || strcmp(fn, "fftshift") == 0 ||
              strcmp(fn, "svd") == 0 || strcmp(fn, "eigh") == 0) &&
             e->u.call.args[0]->res_kind != MC_RES_IMAGE)
    {
        return blas_err(err, errsz, "%s: expected image argument", fn);
    }

    e->needs_image = 1, e->res_kind = MC_RES_IMAGE;
    if (strcmp(fn, "fft") == 0 || strcmp(fn, "ifft") == 0)
    {
        e->res_dt = any_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT;
    }
    else if ((strcmp(fn, "fftshift") == 0 || strcmp(fn, "reshape") == 0) &&
             e->u.call.args[0]->res_dt != 0)
    {
        e->res_dt = e->u.call.args[0]->res_dt;
    }
    else if (strcmp(fn, "vec") == 0 || strcmp(fn, "zeros") == 0 || strcmp(fn, "eye") == 0)
    {
        e->res_dt = _DATATYPE_FLOAT;
    }
    else
    {
        e->res_dt = any_cplx ? (any_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT)
                             : (any_dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    }
    return 0;
}

static const mc_expr_t *unwrap_transpose(const mc_expr_t *e,
                                         const mc_expr_t *call_site,
                                         int             *out_trans)
{
    *out_trans = 0;
    while (e != NULL && e->kind == MC_EXPR_IDENT && e->u.ident.def_param_idx >= 0 &&
           call_site != NULL && call_site->kind == MC_EXPR_CALL &&
           e->u.ident.def_param_idx < call_site->u.call.nargs)
    {
        e = call_site->u.call.args[e->u.ident.def_param_idx];
    }
    if (e != NULL && e->kind == MC_EXPR_CALL && e->u.call.fn != NULL &&
        strcmp(e->u.call.fn->name, "transpose") == 0 && e->u.call.nargs == 1)
    {
        *out_trans = 1;
        return e->u.call.args[0];
    }
    return e;
}

static int is_scalar_subexpr(const mc_expr_t *e)
{
    return e != NULL && !e->needs_image && e->res_kind != MC_RES_IMAGE &&
           e->res_kind != MC_RES_STRING;
}

static int is_mm_or_outer_call(const mc_expr_t *e, const mc_expr_t **out_call, int *out_is_outer)
{
    if (e != NULL && e->kind == MC_EXPR_CALL && e->u.call.fn != NULL && e->u.call.nargs == 2 &&
        (strcmp(e->u.call.fn->name, "matmul") == 0 || strcmp(e->u.call.fn->name, "outer") == 0))
    {
        *out_call     = e;
        *out_is_outer = (strcmp(e->u.call.fn->name, "outer") == 0);
        return 1;
    }
    return 0;
}

static int match_scaled_mm(const mc_expr_t  *e,
                           const mc_expr_t **out_call,
                           int              *out_is_outer,
                           const mc_expr_t **out_scale,
                           double           *out_sign)
{
    double sgn = 1.0;
    while (e != NULL && e->kind == MC_EXPR_UNARY &&
           (e->u.unary.op == MC_TOK_MINUS || e->u.unary.op == MC_TOK_PLUS))
    {
        sgn = (e->u.unary.op == MC_TOK_MINUS) ? -sgn : sgn;
        e   = e->u.unary.operand;
    }
    if (is_mm_or_outer_call(e, out_call, out_is_outer))
    {
        *out_scale = NULL, *out_sign = sgn;
        return 1;
    }
    if (e != NULL && e->kind == MC_EXPR_BINARY && e->u.binary.op == MC_TOK_MUL)
    {
        const mc_expr_t *l = e->u.binary.left, *r = e->u.binary.right;
        if (is_scalar_subexpr(l) && is_mm_or_outer_call(r, out_call, out_is_outer))
        {
            *out_scale = l, *out_sign = sgn;
            return 1;
        }
        if (is_scalar_subexpr(r) && is_mm_or_outer_call(l, out_call, out_is_outer))
        {
            *out_scale = r, *out_sign = sgn;
            return 1;
        }
    }
    return 0;
}

static int match_scaled_ident(const mc_expr_t  *e,
                              const char       *target_bare,
                              const mc_expr_t **out_scale,
                              double           *out_sign)
{
    double sgn = 1.0;
    while (e != NULL && e->kind == MC_EXPR_UNARY &&
           (e->u.unary.op == MC_TOK_MINUS || e->u.unary.op == MC_TOK_PLUS))
    {
        sgn = (e->u.unary.op == MC_TOK_MINUS) ? -sgn : sgn;
        e   = e->u.unary.operand;
    }
    if (e != NULL && e->kind == MC_EXPR_IDENT && strcmp(e->u.ident.bare, target_bare) == 0)
    {
        *out_scale = NULL, *out_sign = sgn;
        return 1;
    }
    if (e != NULL && e->kind == MC_EXPR_BINARY && e->u.binary.op == MC_TOK_MUL)
    {
        const mc_expr_t *l = e->u.binary.left, *r = e->u.binary.right;
        if (is_scalar_subexpr(l) && r->kind == MC_EXPR_IDENT &&
            strcmp(r->u.ident.bare, target_bare) == 0)
        {
            *out_scale = l, *out_sign = sgn;
            return 1;
        }
        if (is_scalar_subexpr(r) && l->kind == MC_EXPR_IDENT &&
            strcmp(l->u.ident.bare, target_bare) == 0)
        {
            *out_scale = r, *out_sign = sgn;
            return 1;
        }
    }
    return 0;
}

typedef struct
{
    const mc_expr_t *call;
    int              is_outer;
    const mc_expr_t *alpha_expr;
    double           alpha_sign;
    int              has_beta;
    const mc_expr_t *beta_expr;
    double           beta_sign;
} alg_blas_match_t;

static int match_alg_blas(const mc_expr_t  *rhs,
                          const mc_expr_t  *target,
                          mc_tok_type_t     aop,
                          alg_blas_match_t *m)
{
    memset(m, 0, sizeof(*m));
    if (target == NULL || target->kind != MC_EXPR_IDENT || rhs == NULL)
    {
        return 0;
    }
    if (aop == MC_TOK_ADD_ASSIGN || aop == MC_TOK_SUB_ASSIGN)
    {
        if (!match_scaled_mm(rhs, &m->call, &m->is_outer, &m->alpha_expr, &m->alpha_sign))
        {
            return 0;
        }
        m->alpha_sign = (aop == MC_TOK_SUB_ASSIGN) ? -m->alpha_sign : m->alpha_sign;
        m->has_beta = 1, m->beta_expr = NULL, m->beta_sign = 1.0;
        return 1;
    }
    if (aop != MC_TOK_ASSIGN)
    {
        return 0;
    }
    if (rhs->kind != MC_EXPR_CALL &&
        match_scaled_mm(rhs, &m->call, &m->is_outer, &m->alpha_expr, &m->alpha_sign))
    {
        return 1;
    }
    if (rhs->kind == MC_EXPR_BINARY &&
        (rhs->u.binary.op == MC_TOK_PLUS || rhs->u.binary.op == MC_TOK_MINUS))
    {
        const mc_expr_t *l = rhs->u.binary.left, *r = rhs->u.binary.right;
        double           rop = (rhs->u.binary.op == MC_TOK_MINUS) ? -1.0 : 1.0;
        if (match_scaled_mm(l, &m->call, &m->is_outer, &m->alpha_expr, &m->alpha_sign) &&
            match_scaled_ident(r, target->u.ident.bare, &m->beta_expr, &m->beta_sign))
        {
            m->has_beta = 1, m->beta_sign *= rop;
            return 1;
        }
        if (match_scaled_ident(l, target->u.ident.bare, &m->beta_expr, &m->beta_sign) &&
            match_scaled_mm(r, &m->call, &m->is_outer, &m->alpha_expr, &m->alpha_sign))
        {
            m->has_beta = 1, m->alpha_sign *= rop;
            return 1;
        }
    }
    return 0;
}

int mc_is_blas_or_linalg_ctor(const mc_expr_t *rhs, const mc_expr_t *target, mc_tok_type_t aop)
{
    alg_blas_match_t m;
    return match_alg_blas(rhs, target, aop, &m);
}

static int finish_inplace_blas(mc_slot_t *slot,
                               int        is_anon,
                               imageID    dst_id,
                               int        in_unit,
                               char      *err,
                               size_t     errsz)
{
    IMAGE *dst = &milk_data.image[dst_id];
    mc_out_post(dst_id);
    if (is_anon || strcmp(slot->bare, dst->name) == 0)
    {
        slot->im_id = dst_id;
        return 0;
    }
    uint32_t sz[3] = {
        dst->md[0].size[0],
        (dst->md[0].naxis > 1) ? dst->md[0].size[1] : 1,
        (dst->md[0].naxis > 2) ? dst->md[0].size[2] : 1,
    };
    imageID oid =
        mc_out_prepare(slot, dst->md[0].naxis, sz, dst->md[0].datatype, in_unit, err, errsz);
    if (oid < 0)
    {
        return -1;
    }
    if (oid != dst_id)
    {
        memcpy(milk_data.image[oid].array.raw, dst->array.raw,
               (size_t) dst->md[0].nelement * (size_t) ImageStreamIO_typesize(dst->md[0].datatype));
        mc_out_post(oid);
    }
    return 0;
}

static int resolve_inplace_target(const mc_expr_t *arg,
                                  const mc_expr_t *call_site,
                                  const char      *fn,
                                  imageID         *out_id,
                                  char            *err,
                                  size_t           errsz)
{
    while (arg != NULL && arg->kind == MC_EXPR_IDENT && arg->u.ident.def_param_idx >= 0 &&
           call_site != NULL && call_site->kind == MC_EXPR_CALL &&
           arg->u.ident.def_param_idx < call_site->u.call.nargs)
    {
        arg = call_site->u.call.args[arg->u.ident.def_param_idx];
    }
    if (arg == NULL || arg->kind != MC_EXPR_IDENT)
    {
        return blas_err(err, errsz, "%s: destination argument must be an existing image", fn);
    }
    imageID id = mc_resolve_or_autoconnect_image(arg->u.ident.name);
    if (id < 0)
    {
        return blas_err(err, errsz, "%s: unknown destination image '%s'", fn, arg->u.ident.name);
    }
    *out_id = id;
    return 0;
}

static int exec_matmul_or_alg(mc_slot_t              *slot,
                              int                     is_anon,
                              const alg_blas_match_t *m,
                              imageID                 avoid_id,
                              const mc_expr_t        *call_site,
                              int                     in_unit,
                              char                   *err,
                              size_t                  errsz)
{
    double alpha = 1.0, beta = 1.0;
    if ((m->alpha_expr != NULL &&
         mc_eval_scalar_d(m->alpha_expr, call_site, &alpha, err, errsz) != 0) ||
        (m->has_beta && m->beta_expr != NULL &&
         mc_eval_scalar_d(m->beta_expr, call_site, &beta, err, errsz) != 0))
    {
        return -1;
    }
    alpha *= m->alpha_sign;
    beta = m->has_beta ? (beta * m->beta_sign) : 0.0;

    int              transA = 0, transB = 0;
    const mc_expr_t *eA  = m->is_outer
                               ? m->call->u.call.args[0]
                               : unwrap_transpose(m->call->u.call.args[0], call_site, &transA);
    const mc_expr_t *eB  = m->is_outer
                               ? m->call->u.call.args[1]
                               : unwrap_transpose(m->call->u.call.args[1], call_site, &transB);
    imageID          idA = -1, idB = -1;
    int              tmpA = 0, tmpB = 0, rc = 0;
    if (mc_resolve_or_eval_im(eA, call_site, avoid_id, in_unit, &idA, &tmpA, err, errsz) != 0 ||
        mc_resolve_or_eval_im(eB, call_site, avoid_id, in_unit, &idB, &tmpB, err, errsz) != 0)
    {
        (void) (tmpA && (mc_tmp_release(idA), 1));
        return -1;
    }
    const IMAGE *imA = &milk_data.image[idA], *imB = &milk_data.image[idB];
    uint8_t      dtA = imA->md[0].datatype, dtB = imB->md[0].datatype;
    if ((dtA != _DATATYPE_FLOAT && dtA != _DATATYPE_DOUBLE) || dtA != dtB)
    {
        rc = blas_err(err, errsz, "%s: expected matching float32 or float64 images",
                      m->is_outer ? "outer" : "matmul");
        goto done;
    }

    uint64_t M = 0, N = 0, K = 0;
    long     out_naxis = 2;
    uint32_t out_sz[3] = { 1, 1, 1 };
    if (m->is_outer)
    {
        if (!arith_image_is_1d_vec(imA, &M) || !arith_image_is_1d_vec(imB, &N))
        {
            rc = blas_err(err, errsz, "outer: expected 1D vector arguments");
            goto done;
        }
        out_sz[0] = (uint32_t) M, out_sz[1] = (uint32_t) N;
    }
    else if (arith_image_matmul_dims(imA, transA, imB, transB, &transA, &transB, &M, &K, &N,
                                     &out_naxis, out_sz) != 0)
    {
        rc = blas_err(err, errsz, "matmul: incompatible dimensions");
        goto done;
    }

    imageID oid = -1;
    if (m->has_beta)
    {
        oid = (slot->im_id >= 0 && milk_data.image[slot->im_id].used &&
               strcmp(milk_data.image[slot->im_id].name, slot->bare) == 0)
                  ? slot->im_id
                  : mc_resolve_or_autoconnect_image(slot->name);
        if (oid < 0 || milk_data.image[oid].md[0].datatype != dtA ||
            milk_data.image[oid].md[0].naxis != (uint8_t) out_naxis ||
            milk_data.image[oid].md[0].size[0] != out_sz[0] ||
            (out_naxis == 2 && milk_data.image[oid].md[0].size[1] != out_sz[1]))
        {
            rc = blas_err(err, errsz, "%s: destination '%s' dimension or type mismatch",
                          m->is_outer ? "ger" : "gemm", slot->bare);
            goto done;
        }
        slot->im_id = oid;
    }
    else
    {
        uint8_t out_dt = mc_select_ctor_out_dt(slot, dtA);
        if (out_dt != dtA || (oid = mc_out_prepare(slot, (uint8_t) out_naxis, out_sz, out_dt,
                                                   in_unit, err, errsz)) < 0)
        {
            rc = (out_dt != dtA) ? blas_err(err, errsz, "%s: destination stream datatype mismatch",
                                            m->is_outer ? "outer" : "matmul")
                                 : -1;
            goto done;
        }
    }

    IMAGE *dst       = &milk_data.image[oid];
    dst->md[0].write = 1;
    rc               = m->is_outer ? arith_image_ger_im(imA, imB, alpha, beta, dst)
                       : (idA == idB && transA != transB && out_naxis == 2 && M == N)
                           ? arith_image_syrk_im(imA, transA, alpha, beta, dst)
                           : arith_image_gemm_im(imA, transA, imB, transB, alpha, beta, dst);
    if (rc != 0)
    {
        dst->md[0].write = 0;
        rc = blas_err(err, errsz, "%s: BLAS execution failed", m->is_outer ? "outer" : "matmul");
        goto done;
    }
    mc_out_post(oid);
    (void) is_anon;

done:
    (void) (tmpA && (mc_tmp_release(idA), 1));
    (void) (tmpB && (mc_tmp_release(idB), 1));
    return rc;
}

static int exec_einsum(mc_slot_t       *slot,
                       const mc_expr_t *rhs,
                       imageID          avoid_id,
                       const mc_expr_t *call_site,
                       int              in_unit,
                       char            *err,
                       size_t           errsz)
{
    mc_val_t sv = { 0 };
    if (mc_scalar_eval_expr(rhs->u.call.args[0], &sv, err, errsz) != 0 || sv.type != MC_VAL_STRING)
    {
        return blas_err(err, errsz, "einsum: first argument must be a subscript string");
    }
    int     nin = rhs->u.call.nargs - 1, tmpA = 0, tmpB = 0, rc = 0;
    imageID idA = -1, idB = -1;
    if (mc_resolve_or_eval_im(rhs->u.call.args[1], call_site, avoid_id, in_unit, &idA, &tmpA, err,
                              errsz) != 0 ||
        (nin == 2 && mc_resolve_or_eval_im(rhs->u.call.args[2], call_site, avoid_id, in_unit, &idB,
                                           &tmpB, err, errsz) != 0))
    {
        (void) (tmpA && (mc_tmp_release(idA), 1));
        return -1;
    }
    const IMAGE *imA = &milk_data.image[idA], *imB = (nin == 2) ? &milk_data.image[idB] : NULL;

    char   sub[128] = { 0 };
    size_t slen     = 0;
    for (const char *p = sv.sval; *p != '\0' && slen + 1 < sizeof(sub); p++)
    {
        (void) (!isspace((unsigned char) *p) && (sub[slen++] = *p));
    }
    char *arrow = strstr(sub, "->"), lhs_s[64] = { 0 }, rhs_s[16] = { 0 };
    if (arrow != NULL)
    {
        size_t ll = (size_t) (arrow - sub);
        if (ll >= sizeof(lhs_s) || strlen(arrow + 2) >= sizeof(rhs_s))
        {
            rc = blas_err(err, errsz, "einsum: subscript string too long");
            goto done;
        }
        memcpy(lhs_s, sub, ll), strcpy(rhs_s, arrow + 2);
    }
    else
    {
        snprintf(lhs_s, sizeof(lhs_s), "%s", sub);
    }

    char *comma = strchr(lhs_s, ','), in_sub[2][8] = { { 0 }, { 0 } };
    if ((nin == 2 && (comma == NULL || strchr(comma + 1, ',') != NULL)) ||
        (nin == 1 && comma != NULL))
    {
        rc = blas_err(err, errsz, "einsum: subscript operand count mismatch");
        goto done;
    }
    if (comma != NULL)
    {
        *comma = '\0';
        snprintf(in_sub[0], sizeof(in_sub[0]), "%s", lhs_s);
        snprintf(in_sub[1], sizeof(in_sub[1]), "%s", comma + 1);
    }
    else
    {
        snprintf(in_sub[0], sizeof(in_sub[0]), "%s", lhs_s);
    }

    const IMAGE *ims[2]      = { imA, imB };
    uint32_t     dim_of[256] = { 0 };
    int          cnt_of[256] = { 0 }, nidx = 0;
    char         idx_list[8] = { 0 };

    for (int op = 0; op < nin; op++)
    {
        size_t       nd = strlen(in_sub[op]);
        const IMAGE *im = ims[op];
        if (nd == 0 || nd > 3 || nd != (size_t) im->md[0].naxis)
        {
            rc = blas_err(err, errsz, "einsum: operand %d rank (%u) does not match subscript '%s'",
                          op, (unsigned) im->md[0].naxis, in_sub[op]);
            goto done;
        }
        for (size_t d = 0; d < nd; d++)
        {
            unsigned char c = (unsigned char) in_sub[op][d];
            if (!isalpha(c) || (cnt_of[c] > 0 && dim_of[c] != im->md[0].size[d]))
            {
                rc = blas_err(err, errsz, "einsum: invalid or mismatched index '%c'", (char) c);
                goto done;
            }
            (void) (cnt_of[c] == 0 &&
                    ((dim_of[c] = im->md[0].size[d]), (idx_list[nidx++] = (char) c)));
            cnt_of[c]++;
        }
    }

    if (arrow == NULL)
    {
        int out_n = 0;
        for (int c = 'a'; c <= 'z'; c++)
        {
            (void) (cnt_of[c] == 1 && (rhs_s[out_n++] = (char) c));
        }
        for (int c = 'A'; c <= 'Z'; c++)
        {
            (void) (cnt_of[c] == 1 && (rhs_s[out_n++] = (char) c));
        }
        rhs_s[out_n] = '\0';
    }

    size_t   out_nd      = strlen(rhs_s);
    int      in_out[256] = { 0 }, nsum = 0;
    uint32_t out_sz[3]  = { 1, 1, 1 };
    char     sum_idx[8] = { 0 };
    if (out_nd > 3)
    {
        rc = blas_err(err, errsz, "einsum: output rank > 3 is not supported");
        goto done;
    }
    for (size_t d = 0; d < out_nd; d++)
    {
        unsigned char c = (unsigned char) rhs_s[d];
        if (cnt_of[c] == 0 || in_out[c])
        {
            rc = blas_err(err, errsz, "einsum: invalid or repeated output index '%c'", (char) c);
            goto done;
        }
        in_out[c] = 1, out_sz[d] = dim_of[c];
    }
    for (int i = 0; i < nidx; i++)
    {
        unsigned char c = (unsigned char) idx_list[i];
        (void) (!in_out[c] && (sum_idx[nsum++] = (char) c));
    }

    uint8_t dtA = imA->md[0].datatype, dtB = imB ? imB->md[0].datatype : dtA;
    uint8_t out_dt = mc_select_ctor_out_dt(
        slot,
        (dtA == _DATATYPE_DOUBLE || dtB == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    imageID oid = mc_out_prepare(slot, (out_nd == 0) ? 1 : (uint8_t) out_nd, out_sz, out_dt,
                                 in_unit, err, errsz);
    if (oid < 0)
    {
        rc = -1;
        goto done;
    }
    IMAGE *out = &milk_data.image[oid];

    if (nin == 2 && strlen(in_sub[0]) == 2 && strlen(in_sub[1]) == 2 && out_nd == 2 && nsum == 1 &&
        (dtA == _DATATYPE_FLOAT || dtA == _DATATYPE_DOUBLE) && dtA == dtB && out_dt == dtA)
    {
        char i_c = rhs_s[0], k_c = rhs_s[1], j_c = sum_idx[0];
        int  tA = (in_sub[0][0] == i_c && in_sub[0][1] == j_c)   ? 0
                  : (in_sub[0][0] == j_c && in_sub[0][1] == i_c) ? 1
                                                                 : -1;
        int  tB = (in_sub[1][0] == j_c && in_sub[1][1] == k_c)   ? 0
                  : (in_sub[1][0] == k_c && in_sub[1][1] == j_c) ? 1
                                                                 : -1;
        if (tA >= 0 && tB >= 0 && arith_image_gemm_im(imA, tA, imB, tB, 1.0, 0.0, out) == 0)
        {
            mc_out_post(oid);
            goto done;
        }
    }
    if (nin == 2 && strlen(in_sub[0]) == 3 && (dtA == _DATATYPE_FLOAT || dtA == _DATATYPE_DOUBLE) &&
        dtA == dtB && out_dt == dtA)
    {
        IMAGE          vA = *imA, vB = *imB, vO = *out;
        IMAGE_METADATA mdA = imA->md[0], mdB = imB->md[0], mdO = out->md[0];
        vA.md = &mdA, vB.md = &mdB, vO.md = &mdO;
        uint32_t nxy = imA->md[0].size[0] * imA->md[0].size[1];
        mdA.naxis = 2, mdA.size[0] = nxy, mdA.size[1] = imA->md[0].size[2];
        int ok = 0;
        if (strlen(in_sub[1]) == 2 && out_nd == 1 && in_sub[0][0] == in_sub[1][0] &&
            in_sub[0][1] == in_sub[1][1] && in_sub[0][2] == rhs_s[0])
        {
            mdB.naxis = 1, mdB.size[0] = nxy, mdB.size[1] = 1;
            ok = (arith_image_gemm_im(&vA, 1, &vB, 0, 1.0, 0.0, &vO) == 0);
        }
        else if (strlen(in_sub[1]) == 1 && out_nd == 2 && in_sub[0][0] == rhs_s[0] &&
                 in_sub[0][1] == rhs_s[1] && in_sub[0][2] == in_sub[1][0])
        {
            mdO.naxis = 1, mdO.size[0] = nxy, mdO.size[1] = 1;
            ok = (arith_image_gemm_im(&vA, 0, &vB, 0, 1.0, 0.0, &vO) == 0);
        }
        else if (strlen(in_sub[1]) == 3 && out_nd == 2 && in_sub[0][0] == in_sub[1][0] &&
                 in_sub[0][1] == in_sub[1][1] && in_sub[0][2] == rhs_s[0] &&
                 in_sub[1][2] == rhs_s[1])
        {
            mdB.naxis = 2, mdB.size[0] = nxy, mdB.size[1] = imB->md[0].size[2];
            ok = (arith_image_gemm_im(&vA, 1, &vB, 0, 1.0, 0.0, &vO) == 0);
        }
        if (ok)
        {
            mc_out_post(oid);
            goto done;
        }
    }

    uint64_t stA[3] = {
        1,
        imA->md[0].size[0],
        (uint64_t) imA->md[0].size[0] *
            (uint64_t) ((imA->md[0].naxis > 1) ? imA->md[0].size[1] : 1),
    };
    uint64_t stB[3] = {
        1,
        imB ? imB->md[0].size[0] : 1,
        imB ? (uint64_t) imB->md[0].size[0] *
                  (uint64_t) ((imB->md[0].naxis > 1) ? imB->md[0].size[1] : 1)
            : 1,
    };
    uint64_t sum_dims[3] = {
        (nsum > 0) ? dim_of[(unsigned char) sum_idx[0]] : 1,
        (nsum > 1) ? dim_of[(unsigned char) sum_idx[1]] : 1,
        (nsum > 2) ? dim_of[(unsigned char) sum_idx[2]] : 1,
    };

    uint32_t coord[256] = { 0 };
    size_t   ndA = strlen(in_sub[0]), ndB = (nin == 2) ? strlen(in_sub[1]) : 0;
    for (uint32_t o2 = 0; o2 < out_sz[2]; o2++)
    {
        (void) (out_nd > 2 && (coord[(unsigned char) rhs_s[2]] = o2, 1));
        for (uint32_t o1 = 0; o1 < out_sz[1]; o1++)
        {
            (void) (out_nd > 1 && (coord[(unsigned char) rhs_s[1]] = o1, 1));
            for (uint32_t o0 = 0; o0 < out_sz[0]; o0++)
            {
                (void) (out_nd > 0 && (coord[(unsigned char) rhs_s[0]] = o0, 1));
                double acc = 0.0;
                for (uint64_t s2 = 0; s2 < sum_dims[2]; s2++)
                {
                    (void) (nsum > 2 && (coord[(unsigned char) sum_idx[2]] = (uint32_t) s2, 1));
                    for (uint64_t s1 = 0; s1 < sum_dims[1]; s1++)
                    {
                        (void) (nsum > 1 && (coord[(unsigned char) sum_idx[1]] = (uint32_t) s1, 1));
                        for (uint64_t s0 = 0; s0 < sum_dims[0]; s0++)
                        {
                            (void) (nsum > 0 &&
                                    (coord[(unsigned char) sum_idx[0]] = (uint32_t) s0, 1));
                            uint64_t offA = 0, offB = 0;
                            for (size_t d = 0; d < ndA; d++)
                            {
                                offA += (uint64_t) coord[(unsigned char) in_sub[0][d]] * stA[d];
                            }
                            for (size_t d = 0; nin == 2 && d < ndB; d++)
                            {
                                offB += (uint64_t) coord[(unsigned char) in_sub[1][d]] * stB[d];
                            }
                            double vA = arith_read_im_pix(imA, dtA, offA);
                            acc += (nin == 1) ? vA : (vA * arith_read_im_pix(imB, dtB, offB));
                        }
                    }
                }
                uint64_t out_idx =
                    ((uint64_t) o2 * (uint64_t) out_sz[1] + (uint64_t) o1) * (uint64_t) out_sz[0] +
                    (uint64_t) o0;
                mc_write_im_pix(out, out_dt, out_idx, acc);
            }
        }
    }
    mc_out_post(oid);

done:
    (void) (tmpA && (mc_tmp_release(idA), 1));
    (void) (tmpB && (mc_tmp_release(idB), 1));
    return rc;
}

int mc_exec_blas_call(mcalc_prog_t    *prog,
                      mc_ctor_step_t  *cs,
                      imageID          avoid_id,
                      const mc_expr_t *call_site,
                      int              in_unit,
                      char            *err,
                      size_t           errsz)
{
    mc_slot_t       *slot = &prog->slots[cs->slot_idx];
    const mc_expr_t *rhs  = call_site ? cs->rhs_expr->u.call.udef_body : cs->rhs_expr;
    alg_blas_match_t m;

    if (rhs->kind != MC_EXPR_CALL ||
        (rhs->u.call.fn != NULL && strcmp(rhs->u.call.fn->name, "matmul") == 0))
    {
        if (rhs->kind == MC_EXPR_CALL)
        {
            memset(&m, 0, sizeof(m));
            m.call = rhs, m.alpha_sign = 1.0;
        }
        else if (!match_alg_blas(rhs, cs->target_expr, cs->assign_op, &m))
        {
            return blas_err(err, errsz, "invalid algebraic BLAS expression");
        }
        return exec_matmul_or_alg(slot, cs->is_anon, &m, avoid_id, call_site, in_unit, err, errsz);
    }

    const char *fn = rhs->u.call.fn->name;
    if (strcmp(fn, "einsum") == 0)
    {
        return exec_einsum(slot, rhs, avoid_id, call_site, in_unit, err, errsz);
    }

    double  alpha = 0.0, beta = 0.0;
    imageID idA = -1, idB = -1, idC = -1;
    int     tmpA = 0, tmpB = 0, rc = 0;

    if (mc_eval_scalar_d(rhs->u.call.args[0], call_site, &alpha, err, errsz) != 0)
    {
        return -1;
    }

    if (strcmp(fn, "scal") == 0)
    {
        if (resolve_inplace_target(rhs->u.call.args[1], call_site, fn, &idC, err, errsz) != 0)
        {
            return -1;
        }
        return (arith_image_scal_im(alpha, &milk_data.image[idC]) == 0)
                   ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                   : blas_err(err, errsz, "scal: expected float or double image");
    }

    if (strcmp(fn, "axpy") == 0)
    {
        if (resolve_inplace_target(rhs->u.call.args[2], call_site, fn, &idC, err, errsz) != 0 ||
            mc_resolve_or_eval_im(rhs->u.call.args[1], call_site, -1, in_unit, &idA, &tmpA, err,
                                  errsz) != 0)
        {
            return -1;
        }
        rc = arith_image_axpy_im(alpha, &milk_data.image[idA], &milk_data.image[idC]);
        (void) (tmpA && (mc_tmp_release(idA), 1));
        return (rc == 0) ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                         : blas_err(err, errsz,
                                    (rc == -2) ? "axpy: element count mismatch"
                                               : "axpy: expected matching float or double images");
    }

    if (strcmp(fn, "ger") == 0)
    {
        if (resolve_inplace_target(rhs->u.call.args[3], call_site, fn, &idC, err, errsz) != 0 ||
            mc_resolve_or_eval_im(rhs->u.call.args[1], call_site, idC, in_unit, &idA, &tmpA, err,
                                  errsz) != 0 ||
            mc_resolve_or_eval_im(rhs->u.call.args[2], call_site, idC, in_unit, &idB, &tmpB, err,
                                  errsz) != 0)
        {
            (void) (tmpA && (mc_tmp_release(idA), 1));
            return -1;
        }
        rc = (arith_image_ger_check(&milk_data.image[idA], &milk_data.image[idB],
                                    &milk_data.image[idC]) == 0 &&
              arith_image_ger_im(&milk_data.image[idA], &milk_data.image[idB], alpha, 1.0,
                                 &milk_data.image[idC]) == 0)
                 ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                 : blas_err(err, errsz, "ger: dimension or datatype mismatch");
        (void) (tmpA && (mc_tmp_release(idA), 1));
        (void) (tmpB && (mc_tmp_release(idB), 1));
        return rc;
    }

    if (strcmp(fn, "syrk") == 0)
    {
        int              transA = 0;
        const mc_expr_t *eA     = unwrap_transpose(rhs->u.call.args[1], call_site, &transA);
        if (mc_eval_scalar_d(rhs->u.call.args[2], call_site, &beta, err, errsz) != 0 ||
            resolve_inplace_target(rhs->u.call.args[3], call_site, fn, &idC, err, errsz) != 0 ||
            mc_resolve_or_eval_im(eA, call_site, idC, in_unit, &idA, &tmpA, err, errsz) != 0)
        {
            return -1;
        }
        rc = (arith_image_syrk_check(&milk_data.image[idA], transA, &milk_data.image[idC]) == 0 &&
              arith_image_syrk_im(&milk_data.image[idA], transA, alpha, beta,
                                  &milk_data.image[idC]) == 0)
                 ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                 : blas_err(err, errsz, "syrk: dimension or datatype mismatch");
        (void) (tmpA && (mc_tmp_release(idA), 1));
        return rc;
    }

    if (strcmp(fn, "gemv") == 0 || strcmp(fn, "symv") == 0 || strcmp(fn, "gemm") == 0)
    {
        int              is_mm = (strcmp(fn, "gemm") == 0), is_symv = (strcmp(fn, "symv") == 0);
        int              transA = 0, transB = 0;
        const mc_expr_t *eA = is_symv ? rhs->u.call.args[1]
                                      : unwrap_transpose(rhs->u.call.args[1], call_site, &transA);
        const mc_expr_t *eB =
            is_mm ? unwrap_transpose(rhs->u.call.args[2], call_site, &transB) : rhs->u.call.args[2];
        if (mc_eval_scalar_d(rhs->u.call.args[3], call_site, &beta, err, errsz) != 0 ||
            resolve_inplace_target(rhs->u.call.args[4], call_site, fn, &idC, err, errsz) != 0 ||
            mc_resolve_or_eval_im(eA, call_site, idC, in_unit, &idA, &tmpA, err, errsz) != 0 ||
            mc_resolve_or_eval_im(eB, call_site, idC, in_unit, &idB, &tmpB, err, errsz) != 0)
        {
            (void) (tmpA && (mc_tmp_release(idA), 1));
            return -1;
        }
        const IMAGE *imA = &milk_data.image[idA], *imB = &milk_data.image[idB];
        IMAGE       *imC = &milk_data.image[idC];
        rc = is_symv ? ((arith_image_symv_im(imA, imB, alpha, beta, imC) == 0)
                            ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                            : blas_err(err, errsz, "symv: dimension or datatype mismatch"))
                     : ((arith_image_matmul_check(imA, transA, imB, transB, imC, 0) == 0 &&
                         arith_image_gemm_im(imA, transA, imB, transB, alpha, beta, imC) == 0)
                            ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                            : blas_err(err, errsz, "%s: dimension or datatype mismatch", fn));
        (void) (tmpA && (mc_tmp_release(idA), 1));
        (void) (tmpB && (mc_tmp_release(idB), 1));
        return rc;
    }

    return blas_err(err, errsz, "unknown BLAS function '%s'", fn);
}
