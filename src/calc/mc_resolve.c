// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/variable_ID.h"
#include "linalg/mc_linalg.h"
#include "mc_ast.h"
#include "mcalc_host.h"
#include "milk_calc_ext.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

#define MC_MAX_CREATED_VARS 64
#define MC_UDEF_MAX_DEPTH 4

typedef struct
{
    char               *err;
    size_t              errsz;
    int                 in_unit, in_lazy_branch, slice_ax_size, ncreated, ncomp_vars, def_depth;
    const char         *self_init_bare;
    const IMAGE        *self_init_im;
    char                created_vars[MC_MAX_CREATED_VARS][256], comp_vars[3][64];
    uint8_t             created_dt[MC_MAX_CREATED_VARS]; /* 255 = scalar var, >0 = image dt */
    const mcalc_udef_t *def_u[MC_UDEF_MAX_DEPTH];
    mc_expr_t *const   *def_args[MC_UDEF_MAX_DEPTH];
} mc_resolve_ctx_t;

__attribute__((format(printf, 2, 3))) static int res_err(mc_resolve_ctx_t *ctx,
                                                         const char       *fmt,
                                                         ...)
{
    if (ctx->err != NULL && ctx->errsz > 0 && ctx->err[0] == '\0')
    {
        va_list ap;
        va_start(ap, fmt), vsnprintf(ctx->err, ctx->errsz, fmt, ap), va_end(ap);
    }
    return -1;
}

static inline int is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}
static inline int is_int_dt(uint8_t dt)
{
    return dt == _DATATYPE_UINT8 || dt == _DATATYPE_INT8 || dt == _DATATYPE_UINT16 ||
           dt == _DATATYPE_INT16 || dt == _DATATYPE_UINT32 || dt == _DATATYPE_INT32 ||
           dt == _DATATYPE_UINT64 || dt == _DATATYPE_INT64;
}
static void set_expr_im(mc_expr_t *e, const IMAGE *im, uint8_t dt)
{
    e->res_kind = MC_RES_IMAGE, e->res_dt = dt, e->needs_image = 1;
    (void) (im != NULL && (mc_set_expr_from_im(e, im), 1));
}
static void copy_res_meta(mc_expr_t *dst, const mc_expr_t *src)
{
    dst->res_kind = src->res_kind, dst->res_dt = src->res_dt, dst->res_naxis = src->res_naxis;
    dst->needs_image = src->needs_image,
    memcpy(dst->res_size, src->res_size, sizeof(dst->res_size));
}
static int created_var_op(mc_resolve_ctx_t *ctx, const char *bare, int add_dt)
{
    for (int i = 0; i < ctx->ncreated; i++)
    {
        if (strcmp(ctx->created_vars[i], bare) == 0)
        {
            return (add_dt > 0) ? (ctx->created_dt[i] = (uint8_t) add_dt) : ctx->created_dt[i];
        }
    }
    if (add_dt > 0 && bare != NULL && bare[0] != '\0' && ctx->ncreated < MC_MAX_CREATED_VARS)
    {
        ctx->created_dt[ctx->ncreated] = (uint8_t) add_dt;
        snprintf(ctx->created_vars[ctx->ncreated++], sizeof(ctx->created_vars[0]), "%s", bare);
    }
    return 0;
}

static int lookup_def_param(const mc_resolve_ctx_t *ctx, const char *bare)
{
    for (int j = 0; ctx->def_depth > 0 && j < ctx->def_u[ctx->def_depth - 1]->nargs; j++)
    {
        if (strcmp(ctx->def_u[ctx->def_depth - 1]->arg_names[j], bare) == 0)
        {
            return j;
        }
    }
    return -1;
}

static int resolve_expr(mc_resolve_ctx_t *ctx, mc_expr_t *e, int nlhs);

static int resolve_lazy(mc_resolve_ctx_t *ctx, mc_expr_t *e, int lazy)
{
    int saved = ctx->in_lazy_branch;
    ctx->in_lazy_branch |= lazy;
    int rc              = resolve_expr(ctx, e, 1);
    ctx->in_lazy_branch = saved;
    return rc;
}

static int resolve_ident(mc_resolve_ctx_t *ctx, mc_expr_t *e)
{
    mc_parse_ident_streamname(e->u.ident.name, e->u.ident.bare, &e->u.ident.loc);
    const char *bare  = e->u.ident.bare;
    e->u.ident.var_id = -1, e->u.ident.im_id = -1;
    int pidx                 = (e->u.ident.loc == 0) ? lookup_def_param(ctx, bare) : -1;
    e->u.ident.def_param_idx = pidx;
    for (int g = 0; pidx == -1 && e->u.ident.loc == 0 && g < ctx->ncomp_vars; g++)
    {
        pidx = (strcmp(ctx->comp_vars[g], bare) == 0) ? -2 : -1;
    }
    if (pidx != -1)
    {
        const mc_expr_t *ae = (pidx >= 0) ? ctx->def_args[ctx->def_depth - 1][pidx] : NULL;
        e->u.ident.id_kind  = ae ? MC_ID_DEF_PARAM : MC_ID_VAR;
        e->u.ident.im_id    = (ae && ae->kind == MC_EXPR_IDENT) ? ae->u.ident.im_id : -1;
        return (ae ? copy_res_meta(e, ae) : set_expr_im(e, NULL, _DATATYPE_FLOAT)), 0;
    }
    long vid = variable_ID(bare);
    if (vid != -1 && milk_data.variable[vid].type != 3)
    {
        int vt             = milk_data.variable[vid].type;
        e->u.ident.id_kind = MC_ID_VAR, e->u.ident.var_id = vid, e->needs_image = 0;
        e->res_kind = (vt == 1) ? MC_RES_SCALAR_L : (vt == 2) ? MC_RES_STRING : MC_RES_SCALAR_D;
        return 0;
    }
    imageID imid = mc_resolve_or_autoconnect_image(e->u.ident.name);
    int     cdt  = (imid == -1) ? created_var_op(ctx, bare, 0) : 0;
    if (imid != -1 || (cdt > 0 && cdt != 255))
    {
        e->u.ident.id_kind = MC_ID_IMAGE, e->u.ident.im_id = imid;
        set_expr_im(e, (imid != -1) ? &milk_data.image[imid] : NULL, (uint8_t) cdt);
        return 0;
    }
    const char *cval = mc_host_str_get(bare);
    cval             = (!cval && mc_is_numeric_env_var(bare)) ? getenv(bare) : cval;
    if (cval != NULL || cdt == 255)
    {
        char *ep1 = NULL, *ep2 = NULL;
        (void) (cval && (strtol(cval, &ep1, 0), strtod(cval, &ep2), 1));
        e->u.ident.id_kind = MC_ID_VAR;
        e->res_kind        = !cval                           ? MC_RES_SCALAR_D
                             : (ep1 != cval && *ep1 == '\0') ? MC_RES_SCALAR_L
                             : (ep2 != cval && *ep2 == '\0') ? MC_RES_SCALAR_D
                                                             : MC_RES_STRING;
        return 0;
    }
    const char *dot1 = strchr(e->u.ident.name, '.');
    if (dot1 != NULL && strchr(dot1 + 1, '.') != NULL &&
        milk_entity_ns_kind(e->u.ident.name, NULL) != MILK_NS_NONE)
    {
        double dv = 0.0;
        long   lv = 0;
        int    il = 0, num_ok = (milk_shm_prop_get_num(e->u.ident.name, &dv, &lv, &il) == 0);
        char   sbuf[256];
        int    str_ok = !num_ok && (milk_shm_prop_get(e->u.ident.name, sbuf, sizeof(sbuf)) == 0);
        e->u.ident.id_kind = MC_ID_PROP;
        e->res_kind        = num_ok   ? (il ? MC_RES_SCALAR_L : MC_RES_SCALAR_D)
                             : str_ok ? MC_RES_STRING
                                      : MC_RES_SCALAR_D;
        return (num_ok || str_ok || ctx->in_lazy_branch) ? 0 : res_err(ctx, "property not found");
    }
    int is_c = (e->u.ident.loc == 0 &&
                (strcmp(bare, "pi") == 0 || strcmp(bare, "nan") == 0 || strcmp(bare, "inf") == 0));
    if (is_c || (ctx->self_init_bare != NULL && strcmp(bare, ctx->self_init_bare) == 0))
    {
        uint64_t cbits     = !is_c              ? 0ULL
                             : (bare[0] == 'p') ? 0x400921fb54442d18ULL
                             : (bare[0] == 'n') ? 0x7ff8000000000000ULL
                                                : 0x7ff0000000000000ULL;
        e->u.ident.id_kind = (!is_c && ctx->self_init_im) ? MC_ID_IMAGE : MC_ID_CONST;
        memcpy(&e->u.ident.const_val, &cbits, sizeof(double));
        e->res_kind = MC_RES_SCALAR_D, e->needs_image = 0;
        (void) (!is_c && ctx->self_init_im &&
                (set_expr_im(e, ctx->self_init_im, ctx->self_init_im->md[0].datatype), 1));
        return 0;
    }
    const char *g      = milk_entity_ns_removed(e->u.ident.name);
    e->u.ident.id_kind = MC_ID_UNRESOLVED, e->res_kind = MC_RES_SCALAR_D;
    return ctx->in_lazy_branch ? 0
                               : res_err(ctx, "undefined identifier '%s'%s%s", e->u.ident.name,
                                         g ? ": " : "", g ? g : "");
}

static int resolve_slice(mc_resolve_ctx_t *ctx, mc_expr_t *e, int is_lhs)
{
    mc_parse_ident_streamname(e->u.slice.name, e->u.slice.bare, NULL);
    int              pidx = lookup_def_param(ctx, e->u.slice.bare);
    const mc_expr_t *ae   = (pidx >= 0) ? ctx->def_args[ctx->def_depth - 1][pidx] : NULL;
    imageID          imid = (ae != NULL && ae->kind == MC_EXPR_IDENT && ae->u.ident.im_id != -1)
                                ? ae->u.ident.im_id
                                : mc_resolve_or_autoconnect_image(e->u.slice.bare);
    int              cdt  = (imid == -1) ? created_var_op(ctx, e->u.slice.bare, 0) : 0;
    if (imid == -1 && (cdt <= 0 || cdt == 255) && (ae == NULL || ae->res_kind != MC_RES_IMAGE))
    {
        return res_err(ctx, is_lhs ? "Target image not found for slice assignment"
                                   : "Source image not found for slice");
    }

    const IMAGE *im       = (imid != -1) ? &milk_data.image[imid] : NULL;
    uint8_t      dt       = im             ? im->md[0].datatype
                            : (cdt > 0)    ? (uint8_t) cdt
                            : (ae != NULL) ? (ae->res_dt ? ae->res_dt : _DATATYPE_FLOAT)
                                           : _DATATYPE_FLOAT;
    int          im_naxis = im ? (int) im->md[0].naxis : 3;
    e->u.slice.im_id      = imid, set_expr_im(e, im, dt);
    int saved_slice_ax_sz = ctx->slice_ax_size;
    int single = im && (e->u.slice.naxis == im_naxis) && !is_complex_dt(dt), bad_bnd = 0;
    int nrange = (im && im_naxis > e->u.slice.naxis) ? (im_naxis - e->u.slice.naxis) : 0;

    for (int ax = 0; ax < e->u.slice.naxis; ax++)
    {
        ctx->slice_ax_size       = (im && ax < im_naxis) ? (int) im->md[0].size[ax] : 1;
        mc_slice_axis_t *sa      = &e->u.slice.axes[ax];
        mc_expr_t       *bnds[3] = { sa->start, sa->stop, sa->step };
        for (int b = 0; b < 3; b++)
        {
            if (bnds[b] != NULL && resolve_expr(ctx, bnds[b], 1) != 0)
            {
                ctx->slice_ax_size = saved_slice_ax_sz;
                return -1;
            }
            bad_bnd |= (bnds[b] != NULL &&
                        (bnds[b]->res_kind == MC_RES_IMAGE || bnds[b]->res_kind == MC_RES_STRING));
        }
        single &= !(sa->has_colon || sa->is_full || sa->is_rev_full || sa->start == NULL ||
                    sa->start->needs_image);
        nrange += (sa->has_colon || sa->is_full || sa->is_rev_full);
    }
    ctx->slice_ax_size = saved_slice_ax_sz;
    if (is_lhs && e->u.slice.naxis == 1 && !e->u.slice.axes[0].has_colon &&
        !e->u.slice.axes[0].is_full && !e->u.slice.axes[0].is_rev_full &&
        e->u.slice.axes[0].start != NULL && e->u.slice.axes[0].start->res_kind == MC_RES_IMAGE)
    {
        return (e->u.slice.is_mask_index = 1), 0;
    }
    if (e->u.slice.naxis > im_naxis || bad_bnd)
    {
        return (e->u.slice.naxis > im_naxis)
                   ? res_err(ctx, "slice axis %d > naxis %d", e->u.slice.naxis - 1, im_naxis)
                   : res_err(ctx, "Invalid slice bound expression");
    }
    e->u.slice.is_single_pixel = single;
    (void) ((single && !is_lhs)
                ? ((e->res_kind = is_int_dt(dt) ? MC_RES_SCALAR_L : MC_RES_SCALAR_D),
                   (e->needs_image = 0))
                : (!e->u.slice.is_mask_index && nrange > 0 && (e->res_naxis = (uint8_t) nrange)));
    return 0;
}

static int resolve_udef_call(mc_resolve_ctx_t *ctx, mc_expr_t *e)
{
    const mcalc_udef_t *u = mcalc_udef_get(e->u.call.udef_id);
    if (u == NULL || e->u.call.nargs != u->nargs || ctx->def_depth >= MC_UDEF_MAX_DEPTH)
    {
        return (u == NULL || e->u.call.nargs != u->nargs)
                   ? res_err(ctx, "wrong number of arguments for '%s'", e->u.call.name)
                   : res_err(ctx, "User function call depth exceeded");
    }
    ctx->def_u[ctx->def_depth] = u, ctx->def_args[ctx->def_depth] = e->u.call.args;
    mc_tok_list_t *btl = calloc(1, sizeof(mc_tok_list_t));
    mc_stmt_t     *bst = (btl != NULL && mc_lex(u->body, btl) == 0)
                             ? mc_parse_stmt(btl, ctx->err, ctx->errsz)
                             : NULL;
    if (bst == NULL || bst->kind != MC_STMT_EXPR || bst->u.expr == NULL)
    {
        res_err(ctx, "%s", (btl && btl->errmsg[0]) ? btl->errmsg : "invalid user function body");
        return free(btl), mc_stmt_free(bst), -1;
    }
    free(btl), ctx->def_depth++;
    int brc = resolve_expr(ctx, bst->u.expr, 1);
    ctx->def_depth--;
    (void) (brc == 0 && (copy_res_meta(e, bst->u.expr), mc_expr_free(e->u.call.udef_body),
                         (e->u.call.udef_body = bst->u.expr), (bst->u.expr = NULL), 1));
    mc_stmt_free(bst);
    return brc;
}

static int resolve_call(mc_resolve_ctx_t *ctx, mc_expr_t *e, int nlhs)
{
    e->u.call.udef_id = -1;
    for (int i = 0, n = mcalc_udef_count(); i < n && e->u.call.udef_id < 0; i++)
    {
        const mcalc_udef_t *u = mcalc_udef_get(i);
        e->u.call.udef_id     = (u != NULL && strcmp(u->name, e->u.call.name) == 0) ? i : -1;
    }
    char   base[64] = { 0 };
    size_t blen     = strcspn(e->u.call.name, ":");
    (void) (blen < sizeof(base) && memcpy(base, e->u.call.name, blen));
    e->u.call.ext_id = (e->u.call.udef_id < 0 && blen > 0) ? milk_calc_ext_find(base) : -1;
    e->u.call.fn =
        (e->u.call.udef_id < 0 && e->u.call.ext_id < 0) ? mc_func_find(e->u.call.name) : NULL;
    if (e->u.call.udef_id < 0 && e->u.call.ext_id < 0 && e->u.call.fn == NULL)
    {
        return res_err(ctx, "unknown function '%s'", e->u.call.name);
    }
    if (e->u.call.fn != NULL && strcmp(e->u.call.fn->name, "post") == 0)
    {
        mc_expr_t *a0 = (e->u.call.nargs == 1) ? e->u.call.args[0] : NULL;
        return (!a0 || a0->kind != MC_EXPR_IDENT)
                   ? res_err(ctx, "post: expected post(stream_name)")
                   : (mc_parse_ident_streamname(a0->u.ident.name, a0->u.ident.bare,
                                                &a0->u.ident.loc),
                      (a0->res_kind = e->res_kind = MC_RES_SCALAR_L),
                      (a0->needs_image = e->needs_image = 0));
    }
    for (int i = 0; i < e->u.call.nargs; i++)
    {
        if ((e->u.call.kw_names[i][0] != '\0' && e->u.call.ext_id < 0) ||
            resolve_expr(ctx, e->u.call.args[i], 1) != 0)
        {
            return (e->u.call.kw_names[i][0] != '\0' && e->u.call.ext_id < 0)
                       ? res_err(ctx, "keyword arguments are only supported for user extensions")
                       : -1;
        }
    }
    if (e->u.call.udef_id >= 0 || e->u.call.ext_id >= 0)
    {
        return (e->u.call.udef_id >= 0) ? resolve_udef_call(ctx, e)
                                        : mc_ext_resolve_call(e, nlhs, ctx->err, ctx->errsz);
    }
    const mc_func_t *fn    = e->u.call.fn;
    int              nargs = e->u.call.nargs;
    if (nargs < fn->min_args || (fn->max_args >= 0 && nargs > fn->max_args) || nlhs < fn->min_lhs ||
        nlhs > fn->max_lhs)
    {
        return (nargs < fn->min_args || (fn->max_args >= 0 && nargs > fn->max_args))
                   ? res_err(ctx, "wrong number of arguments for '%s'", fn->name)
                   : res_err(ctx,
                             (fn->min_lhs > 1)
                                 ? "Multi-return reduction must be unpacked into variables"
                                 : "wrong number of LHS targets for '%s'",
                             fn->name);
    }
    int any_im = 0, any_cplx = 0, any_dbl = 0;
    for (int i = 0; i < nargs; i++)
    {
        uint8_t dt = e->u.call.args[i]->res_dt;
        any_im |= (e->u.call.args[i]->needs_image || e->u.call.args[i]->res_kind == MC_RES_IMAGE);
        any_cplx |= (e->u.call.args[i]->res_kind == MC_RES_IMAGE && is_complex_dt(dt));
        any_dbl |= (e->u.call.args[i]->res_kind == MC_RES_IMAGE &&
                    (dt == _DATATYPE_DOUBLE || dt == _DATATYPE_COMPLEX_DOUBLE));
    }
    if (any_cplx && fn->kind != MC_FN_CPLX && strcmp(fn->name, "abs") != 0 &&
        strcmp(fn->name, "fabs") != 0 && strcmp(fn->name, "reshape") != 0 &&
        strcmp(fn->name, "fft") != 0 && strcmp(fn->name, "ifft") != 0 &&
        strcmp(fn->name, "fftshift") != 0)
    {
        return res_err(ctx, "Function of real pixels applied to a complex image; use real(), "
                            "imag() or abs() first");
    }

    switch (fn->kind)
    {
    case MC_FN_SCALAR:
    case MC_FN_WHERE:
    case MC_FN_PROG:
        if ((fn->kind == MC_FN_SCALAR && nargs == 3 &&
             (e->u.call.args[1]->res_kind == MC_RES_IMAGE ||
              e->u.call.args[2]->res_kind == MC_RES_IMAGE)) ||
            (fn->kind == MC_FN_PROG &&
             (strcmp(fn->name, "ema") != 0 || e->u.call.args[0]->res_kind == MC_RES_STRING ||
              e->u.call.args[1]->res_kind == MC_RES_IMAGE ||
              e->u.call.args[1]->res_kind == MC_RES_STRING)))
        {
            return res_err(ctx, (fn->kind == MC_FN_SCALAR)
                                    ? "Only the first argument may be an image"
                                : (strcmp(fn->name, "ema") == 0)
                                    ? "ema: expected ema(expr, alpha) with scalar alpha"
                                    : "function requires hist(stream, N) in a unit program");
        }
        for (int i = 0; any_im && i < nargs && e->res_naxis == 0; i++)
        {
            (void) (e->u.call.args[i]->res_naxis > 0 &&
                    ((e->res_naxis = e->u.call.args[i]->res_naxis),
                     memcpy(e->res_size, e->u.call.args[i]->res_size, sizeof(e->res_size))));
        }
        e->needs_image = any_im;
        e->res_dt   = (fn->kind == MC_FN_PROG && any_im && is_complex_dt(e->u.call.args[0]->res_dt))
                          ? e->u.call.args[0]->res_dt
                      : fn->cast_dt ? fn->cast_dt
                                    : (any_dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
        e->res_kind = any_im                      ? MC_RES_IMAGE
                      : (fn->kind == MC_FN_WHERE) ? e->u.call.args[1]->res_kind
                                                  : MC_RES_SCALAR_D;
        return 0;

    case MC_FN_CPLX:
        if ((strcmp(fn->name, "cplx") == 0 || strcmp(fn->name, "expi") == 0) &&
            (!any_im || any_cplx || e->u.call.args[0]->res_kind == MC_RES_STRING ||
             (nargs == 2 && e->u.call.args[1]->res_kind == MC_RES_STRING)))
        {
            return res_err(ctx, (strcmp(fn->name, "expi") == 0)
                                    ? "expi: expected a real phase image"
                                : !any_im ? "cplx: needs at least one image"
                                          : "cplx: needs two real operands of the same size");
        }
        e->needs_image =
            (strcmp(fn->name, "cplx") == 0 || strcmp(fn->name, "expi") == 0) ? 1 : any_im;
        e->res_dt   = (strcmp(fn->name, "cplx") == 0 || strcmp(fn->name, "expi") == 0)
                          ? (any_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT)
                      : (strcmp(fn->name, "conj") == 0)
                          ? e->u.call.args[0]->res_dt
                          : (any_dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
        e->res_kind = e->needs_image ? MC_RES_IMAGE
                      : (strcmp(fn->name, "real") == 0 || strcmp(fn->name, "conj") == 0)
                          ? e->u.call.args[0]->res_kind
                          : MC_RES_SCALAR_D;
        return 0;

    case MC_FN_PRNG:
    case MC_FN_LUT:
    case MC_FN_COORD:
        if ((fn->kind == MC_FN_PRNG && nargs == 2 &&
             (e->u.call.args[0]->res_kind != MC_RES_IMAGE ||
              e->u.call.args[1]->res_kind == MC_RES_IMAGE)) ||
            (fn->kind != MC_FN_PRNG &&
             (nargs == ((fn->kind == MC_FN_LUT) ? 3 : 2) ||
              e->u.call.args[(fn->kind == MC_FN_LUT) ? 1 : 0]->res_kind != MC_RES_IMAGE ||
              (nargs >= 3 && (e->u.call.args[nargs - 2]->res_kind == MC_RES_IMAGE ||
                              e->u.call.args[nargs - 1]->res_kind == MC_RES_IMAGE)))))
        {
            return res_err(ctx, (fn->kind == MC_FN_PRNG) ? "rand: expected rand([im[, seed]])"
                                : (fn->kind == MC_FN_LUT)
                                    ? "lut: expected lut(x, tbl[, x0, dx])"
                                    : "Expected an image, then optionally the centre: x, y");
        }
        e->needs_image = (fn->kind == MC_FN_PRNG)
                             ? (nargs > 0 && e->u.call.args[0]->res_kind == MC_RES_IMAGE)
                         : (fn->kind == MC_FN_LUT) ? e->u.call.args[0]->needs_image
                                                   : 1;
        e->res_kind    = e->needs_image ? MC_RES_IMAGE : MC_RES_SCALAR_D;
        e->res_dt      = any_dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
        return 0;

    case MC_FN_STR:
        if (any_im ||
            (strcmp(fn->name, "substr") == 0 && e->u.call.args[0]->res_kind != MC_RES_STRING) ||
            (strcmp(fn->name, "replace") == 0 && (e->u.call.args[0]->res_kind != MC_RES_STRING ||
                                                  e->u.call.args[1]->res_kind != MC_RES_STRING ||
                                                  e->u.call.args[2]->res_kind != MC_RES_STRING)))
        {
            return res_err(ctx, any_im ? "string function does not accept an image argument"
                                : (strcmp(fn->name, "substr") == 0)
                                    ? "substr: first arg must be string"
                                    : "replace: all args must be strings");
        }
        e->needs_image = 0,
        e->res_kind    = (strcmp(fn->name, "strlen") == 0) ? MC_RES_SCALAR_L : MC_RES_STRING;
        return 0;

    case MC_FN_REDUCE_D:
    case MC_FN_REDUCE_L:
    case MC_FN_ISTATS:
    case MC_FN_ICENTROID:
    case MC_FN_EAGER_RED:
    case MC_FN_DOT:
    case MC_FN_STENCIL_LIB:
    {
        const mc_expr_t *a0 = e->u.call.args[0], *a1 = (nargs >= 2) ? e->u.call.args[1] : NULL;
        const mc_expr_t *a2 = (nargs >= 3) ? e->u.call.args[2] : NULL;
        if (fn->kind == MC_FN_STENCIL_LIB)
        {
            int is_pu = (strcmp(fn->name, "pixunmap") == 0),
                is_rp = (strcmp(fn->name, "rprof") == 0);
            if (a0->res_kind != MC_RES_IMAGE ||
                (is_pu && (a1->res_kind != MC_RES_IMAGE || (a2 && a2->res_kind == MC_RES_IMAGE))) ||
                (is_rp && (nargs == 2 || (a1 && a1->res_kind == MC_RES_IMAGE) ||
                           (a2 && a2->res_kind == MC_RES_IMAGE))) ||
                (strcmp(fn->name, "conv2") == 0 && a1->res_kind != MC_RES_IMAGE) ||
                (!is_pu && !is_rp && strcmp(fn->name, "conv2") != 0 &&
                 ((a1 && a1->res_kind == MC_RES_IMAGE) || (a2 && a2->res_kind == MC_RES_IMAGE))))
            {
                return res_err(ctx, is_pu   ? "pixunmap: expected pixunmap(im, map[, n])"
                                    : is_rp ? "rprof: expected rprof(im[, cx, cy])"
                                            : "Expected image argument and scalar parameters");
            }
            set_expr_im(e, NULL, any_dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
            (void) (!is_pu && !is_rp && a0->res_naxis > 0 &&
                    ((e->res_naxis = a0->res_naxis),
                     memcpy(e->res_size, a0->res_size, sizeof(e->res_size))));
            return 0;
        }
        if (a0->res_kind != MC_RES_IMAGE ||
            (fn->kind == MC_FN_DOT && a1->res_kind != MC_RES_IMAGE) ||
            (a1 && a1->res_kind == MC_RES_STRING) || (fn->kind == MC_FN_REDUCE_L && a1) ||
            (strcmp(fn->name, "imedian") == 0 && a1 && a1->res_kind != MC_RES_IMAGE) ||
            (strcmp(fn->name, "perc") == 0 &&
             (a1->res_kind == MC_RES_IMAGE || (a2 && a2->res_kind != MC_RES_IMAGE))) ||
            (fn->kind == MC_FN_ICENTROID && a1 && a1->res_kind == MC_RES_IMAGE) ||
            (fn->kind == MC_FN_ISTATS && a1 && a1->res_kind != MC_RES_IMAGE))
        {
            return res_err(
                ctx, (fn->kind == MC_FN_DOT) ? "Expected two image arguments"
                     : (a0->res_kind != MC_RES_IMAGE || (a1 && a1->res_kind == MC_RES_STRING))
                         ? "Expected image argument"
                     : (strcmp(fn->name, "perc") == 0) ? "Expected perc(image, fraction[, mask])"
                     : (fn->kind == MC_FN_ICENTROID)   ? "icentroid: threshold must be a scalar"
                     : (fn->kind == MC_FN_ISTATS)      ? "istats: second argument must be a mask"
                     : (fn->kind == MC_FN_REDUCE_L && a1->res_kind == MC_RES_IMAGE)
                         ? "This function takes no mask argument"
                         : "This function takes no axis argument");
        }
        e->res_dt = any_dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
        e->res_kind =
            (fn->kind == MC_FN_REDUCE_D && a1 && a1->res_kind != MC_RES_IMAGE && a0->res_naxis != 1)
                ? MC_RES_IMAGE
            : (fn->kind == MC_FN_REDUCE_L) ? MC_RES_SCALAR_L
                                           : MC_RES_SCALAR_D;
        if (e->res_kind == MC_RES_IMAGE && a0->res_naxis >= 2 && a1->kind == MC_EXPR_LONG)
        {
            long ax = a1->u.lval;
            if (ax < 0 || ax >= a0->res_naxis)
            {
                return res_err(ctx, "%s: axis out of range", fn->name);
            }
            e->res_naxis = (a0->res_naxis == 2 && ax == 1) ? 1 : 2, e->res_size[2] = 1;
            e->res_size[0] = (a0->res_naxis == 2) ? ((ax == 0) ? 1 : a0->res_size[0])
                                                  : ((ax == 0) ? a0->res_size[1] : a0->res_size[0]);
            e->res_size[1] = (a0->res_naxis == 2) ? ((ax == 0) ? a0->res_size[1] : 1)
                                                  : ((ax == 2) ? a0->res_size[1] : a0->res_size[2]);
        }
        e->needs_image = (e->res_kind == MC_RES_IMAGE);
        return 0;
    }

    default:
        return mc_resolve_linalg_call(e, nlhs, any_im, any_cplx, any_dbl, ctx->err, ctx->errsz);
    }
}

static int resolve_assign(mc_resolve_ctx_t *ctx, mc_expr_t *e)
{
    mc_expr_t    *target = e->u.assign.target, *rhs = e->u.assign.rhs;
    mc_tok_type_t op = e->u.assign.op;
    if (ctx->in_unit && op == MC_TOK_ASSIGN && target->kind == MC_EXPR_IDENT)
    {
        mc_parse_ident_streamname(target->u.ident.name, target->u.ident.bare, &target->u.ident.loc);
        const char  *tb       = target->u.ident.bare;
        const IMAGE *other_im = NULL;
        if (variable_ID(tb) == -1 && mc_resolve_or_autoconnect_image(target->u.ident.name) == -1 &&
            created_var_op(ctx, tb, 0) == 0 && mc_host_str_get(tb) == NULL &&
            mc_expr_mentions_self(rhs, tb, (const void **) &other_im))
        {
            ctx->self_init_bare = tb, ctx->self_init_im = other_im;
            int prc             = resolve_expr(ctx, rhs, 1);
            ctx->self_init_bare = NULL, ctx->self_init_im = NULL;
            (void) (prc == 0 && (mc_self_init_create(
                                     tb, target->u.ident.loc == 'L' || e->u.assign.is_local, rhs),
                                 1));
        }
    }
    if (resolve_expr(ctx, rhs, 1) != 0 ||
        (target->kind == MC_EXPR_SLICE && resolve_slice(ctx, target, 1) != 0))
    {
        return -1;
    }
    if (target->kind == MC_EXPR_SLICE)
    {
        int spix       = target->u.slice.is_single_pixel && !rhs->needs_image &&
                         rhs->res_kind != MC_RES_IMAGE && rhs->res_kind != MC_RES_STRING;
        e->needs_image = !spix, e->res_dt = target->res_dt;
        e->res_kind = !spix                                                       ? MC_RES_IMAGE
                      : (rhs->res_kind == MC_RES_SCALAR_L && op == MC_TOK_ASSIGN) ? MC_RES_SCALAR_L
                                                                                  : MC_RES_SCALAR_D;
        return 0;
    }
    if (target->kind != MC_EXPR_IDENT)
    {
        return res_err(ctx, "invalid assignment target");
    }
    target->u.ident.im_id = target->u.ident.def_param_idx = -1, target->u.ident.var_id = -1;
    mc_parse_ident_streamname(target->u.ident.name, target->u.ident.bare, &target->u.ident.loc);
    const char *bare = target->u.ident.bare, *dot1 = strchr(target->u.ident.name, '.');
    int         is_prop = (dot1 != NULL && strchr(dot1 + 1, '.') != NULL &&
                           milk_entity_ns_kind(target->u.ident.name, NULL) != MILK_NS_NONE);
    long        vid     = is_prop ? -1 : variable_ID(bare);
    if (is_prop || (vid != -1 && milk_data.variable[vid].type != 3))
    {
        target->u.ident.id_kind = is_prop ? MC_ID_PROP : MC_ID_VAR, target->u.ident.var_id = vid;
        e->needs_image = rhs->needs_image, e->res_kind = rhs->res_kind;
        return (rhs->res_kind == MC_RES_IMAGE)
                   ? res_err(ctx,
                             is_prop ? "cannot assign image to property"
                                     : "'%s' is a variable (calculator or bash), not an image",
                             bare)
                   : 0;
    }
    imageID imid = mc_resolve_or_autoconnect_image(target->u.ident.name);
    int     cdt  = (imid == -1) ? created_var_op(ctx, bare, 0) : 0;
    if (imid != -1 || (cdt > 0 && cdt != 255))
    {
        uint8_t tdt = (imid != -1) ? milk_data.image[imid].md[0].datatype : (uint8_t) cdt;
        target->u.ident.id_kind = MC_ID_IMAGE, target->u.ident.im_id = imid;
        set_expr_im(target, (imid != -1) ? &milk_data.image[imid] : NULL, tdt);
        set_expr_im(e, NULL,
                    (op == MC_TOK_ASSIGN && rhs->res_kind == MC_RES_IMAGE &&
                     (imid == -1 || !milk_data.image[imid].md[0].shared))
                        ? (rhs->res_dt ? rhs->res_dt : _DATATYPE_FLOAT)
                        : tdt);
        return 0;
    }
    if (op != MC_TOK_ASSIGN &&
        ((mc_host_str_get(bare) == NULL && !mc_is_numeric_env_var(bare) && cdt == 0) ||
         rhs->res_kind == MC_RES_IMAGE))
    {
        return (rhs->res_kind == MC_RES_IMAGE)
                   ? res_err(ctx, "'%s' is a variable (calculator or bash), not an image", bare)
                   : res_err(ctx, "Variable not found for compound assign");
    }
    target->u.ident.id_kind =
        (op == MC_TOK_ASSIGN && rhs->res_kind == MC_RES_IMAGE) ? MC_ID_NEW : MC_ID_VAR;
    target->u.ident.var_id = -1;
    (void) (op == MC_TOK_ASSIGN &&
            created_var_op(ctx, bare,
                           (rhs->res_kind == MC_RES_IMAGE)
                               ? (rhs->res_dt ? rhs->res_dt : _DATATYPE_FLOAT)
                               : 255));
    e->needs_image = (rhs->res_kind == MC_RES_IMAGE) ? 1 : rhs->needs_image;
    e->res_kind    = (op != MC_TOK_ASSIGN) ? MC_RES_SCALAR_D : rhs->res_kind;
    e->res_dt      = rhs->res_dt ? rhs->res_dt : _DATATYPE_FLOAT;
    return 0;
}

static int resolve_expr(mc_resolve_ctx_t *ctx, mc_expr_t *e, int nlhs)
{
    if (e == NULL)
    {
        return -1;
    }
    switch (e->kind)
    {
    case MC_EXPR_LONG:
    case MC_EXPR_DOUBLE:
    case MC_EXPR_STRING:
        e->res_kind    = (e->kind == MC_EXPR_LONG)     ? MC_RES_SCALAR_L
                         : (e->kind == MC_EXPR_DOUBLE) ? MC_RES_SCALAR_D
                                                       : MC_RES_STRING;
        e->needs_image = 0;
        return 0;

    case MC_EXPR_END:
        if (variable_ID("end") != -1 || mc_host_str_get("end") != NULL ||
            created_var_op(ctx, "end", 0) || lookup_def_param(ctx, "end") >= 0)
        {
            e->kind = MC_EXPR_IDENT, snprintf(e->u.ident.name, sizeof(e->u.ident.name), "end");
            return resolve_ident(ctx, e);
        }
        return (ctx->slice_ax_size <= 0) ? res_err(ctx, "unexpected 'end' outside slice brackets")
                                         : ((e->u.lval = (long) ctx->slice_ax_size - 1),
                                            (e->res_kind = MC_RES_SCALAR_L), (e->needs_image = 0));

    case MC_EXPR_IDENT:
    case MC_EXPR_SLICE:
        return (e->kind == MC_EXPR_IDENT) ? resolve_ident(ctx, e) : resolve_slice(ctx, e, 0);

    case MC_EXPR_CALL:
    case MC_EXPR_ASSIGN:
        return (e->kind == MC_EXPR_CALL) ? resolve_call(ctx, e, nlhs) : resolve_assign(ctx, e);

    case MC_EXPR_UNARY:
        if (resolve_expr(ctx, e->u.unary.operand, 1) != 0 ||
            (e->u.unary.operand->res_kind == MC_RES_STRING && e->u.unary.op == MC_TOK_MINUS) ||
            (e->u.unary.operand->res_kind == MC_RES_IMAGE &&
             is_complex_dt(e->u.unary.operand->res_dt) && e->u.unary.op != MC_TOK_MINUS))
        {
            return (e->u.unary.operand->res_kind == MC_RES_STRING && e->u.unary.op == MC_TOK_MINUS)
                       ? res_err(ctx, "Cannot negate expression")
                   : (e->u.unary.operand->res_kind == MC_RES_IMAGE &&
                      is_complex_dt(e->u.unary.operand->res_dt) && e->u.unary.op != MC_TOK_MINUS)
                       ? res_err(ctx, "Operator not supported for complex images; use real(), "
                                      "imag() or abs() first")
                       : -1;
        }
        copy_res_meta(e, e->u.unary.operand);
        e->res_dt   = (e->res_kind == MC_RES_IMAGE && !is_complex_dt(e->res_dt) &&
                       e->res_dt != _DATATYPE_DOUBLE)
                          ? _DATATYPE_FLOAT
                          : e->res_dt;
        e->res_kind = (e->res_kind != MC_RES_IMAGE &&
                       (e->u.unary.op == MC_TOK_NOT || e->u.unary.op == MC_TOK_BNOT))
                          ? MC_RES_SCALAR_L
                          : e->res_kind;
        return 0;

    case MC_EXPR_BINARY:
    {
        mc_expr_t *le = e->u.binary.left, *re = e->u.binary.right;
        if (resolve_expr(ctx, le, 1) != 0 ||
            resolve_lazy(ctx, re,
                         (e->u.binary.op == MC_TOK_AND || e->u.binary.op == MC_TOK_OR) &&
                             !le->needs_image) != 0)
        {
            return -1;
        }
        mc_tok_type_t op = e->u.binary.op;
        mc_res_kind_t lk = le->res_kind, rk = re->res_kind;
        uint8_t       ldt = le->res_dt, rdt = re->res_dt;
        int           cplx = (lk == MC_RES_IMAGE && is_complex_dt(ldt)) ||
                             (rk == MC_RES_IMAGE && is_complex_dt(rdt));
        if (cplx && op != MC_TOK_PLUS && op != MC_TOK_MINUS && op != MC_TOK_MUL && op != MC_TOK_DIV)
        {
            return res_err(ctx, "Operator not supported for complex images; use real(), imag() or "
                                "abs() first");
        }
        const mc_expr_t *s_im =
            (le->res_naxis > 0 && (re->res_naxis == 0 ||
                                   (le->res_naxis == re->res_naxis &&
                                    memcmp(le->res_size, re->res_size, sizeof(le->res_size)) == 0)))
                ? le
            : (le->res_naxis == 0 && re->res_naxis > 0) ? re
                                                        : NULL;
        (void) (s_im && ((e->res_naxis = s_im->res_naxis),
                         memcpy(e->res_size, s_im->res_size, sizeof(e->res_size))));
        e->needs_image = (le->needs_image || re->needs_image);
        int dbl =
            (lk == MC_RES_IMAGE && (ldt == _DATATYPE_DOUBLE || ldt == _DATATYPE_COMPLEX_DOUBLE)) ||
            (rk == MC_RES_IMAGE && (rdt == _DATATYPE_DOUBLE || rdt == _DATATYPE_COMPLEX_DOUBLE));
        e->res_dt     = cplx  ? (dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT)
                        : dbl ? _DATATYPE_DOUBLE
                              : _DATATYPE_FLOAT;
        int is_int_op = mc_is_rel_tok(op) || op == MC_TOK_EQ || op == MC_TOK_NE ||
                        op == MC_TOK_AND || op == MC_TOK_OR || op == MC_TOK_BAND ||
                        op == MC_TOK_BOR || op == MC_TOK_BXOR || op == MC_TOK_SHL ||
                        op == MC_TOK_SHR;
        e->res_kind   = (lk == MC_RES_IMAGE || rk == MC_RES_IMAGE) ? MC_RES_IMAGE
                        : (op == MC_TOK_PLUS && (lk == MC_RES_STRING || rk == MC_RES_STRING))
                            ? MC_RES_STRING
                        : is_int_op                                        ? MC_RES_SCALAR_L
                        : (lk == MC_RES_SCALAR_D || rk == MC_RES_SCALAR_D) ? MC_RES_SCALAR_D
                                                                           : MC_RES_SCALAR_L;
        return 0;
    }

    case MC_EXPR_CHAIN:
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            if (resolve_lazy(ctx, e->u.chain.operands[i], i > 0 && !e->needs_image) != 0)
            {
                return -1;
            }
            e->needs_image |= e->u.chain.operands[i]->needs_image;
        }
        e->res_kind = e->needs_image ? MC_RES_IMAGE : MC_RES_SCALAR_L;
        return (e->res_dt = _DATATYPE_FLOAT), 0;

    case MC_EXPR_TERNARY:
        if (resolve_expr(ctx, e->u.ternary.cond, 1) != 0 ||
            resolve_lazy(ctx, e->u.ternary.t_expr, !e->u.ternary.cond->needs_image) != 0 ||
            resolve_lazy(ctx, e->u.ternary.f_expr, !e->u.ternary.cond->needs_image) != 0)
        {
            return -1;
        }
        e->needs_image = (e->u.ternary.cond->needs_image || e->u.ternary.t_expr->needs_image ||
                          e->u.ternary.f_expr->needs_image);
        e->res_kind    = e->needs_image ? MC_RES_IMAGE : e->u.ternary.t_expr->res_kind;
        e->res_dt      = (e->u.ternary.t_expr->res_dt == _DATATYPE_DOUBLE ||
                          e->u.ternary.f_expr->res_dt == _DATATYPE_DOUBLE)
                             ? _DATATYPE_DOUBLE
                             : _DATATYPE_FLOAT;
        return 0;

    case MC_EXPR_VEC_LIT:
        for (int i = 0; i < e->u.vec_lit.nelem; i++)
        {
            mc_expr_t *el = e->u.vec_lit.elems[i];
            if (resolve_expr(ctx, el, 1) != 0 || el->res_kind == MC_RES_IMAGE ||
                el->res_kind == MC_RES_STRING)
            {
                return (el->res_kind == MC_RES_IMAGE || el->res_kind == MC_RES_STRING)
                           ? res_err(ctx, "vec: arguments must be numbers")
                           : -1;
            }
        }
        set_expr_im(e, NULL, _DATATYPE_FLOAT), e->res_naxis = 1;
        e->res_size[0] = (uint32_t) e->u.vec_lit.nelem, e->res_size[1] = e->res_size[2] = 1;
        return 0;

    case MC_EXPR_COMP:
        for (int g = 0; g < e->u.comp.ngens; g++)
        {
            if ((e->u.comp.gens[g].start && resolve_expr(ctx, e->u.comp.gens[g].start, 1) != 0) ||
                (e->u.comp.gens[g].stop && resolve_expr(ctx, e->u.comp.gens[g].stop, 1) != 0) ||
                (e->u.comp.gens[g].step && resolve_expr(ctx, e->u.comp.gens[g].step, 1) != 0))
            {
                return -1;
            }
            snprintf(ctx->comp_vars[g], sizeof(ctx->comp_vars[g]), "%s", e->u.comp.gens[g].var);
        }
        ctx->ncomp_vars = e->u.comp.ngens;
        return (resolve_expr(ctx, e->u.comp.body, 1) == 0)
                   ? ((ctx->ncomp_vars = 0), set_expr_im(e, NULL, _DATATYPE_FLOAT), 0)
                   : ((ctx->ncomp_vars = 0), -1);
    }
    return -1;
}

static int resolve_stmt_list(mc_resolve_ctx_t *ctx, mc_stmt_t *s)
{
    static const char *k_im_cond_err =
        "an if or for needs a number, not an image: test imin(...) for all pixels, imax(...) for "
        "any";
    for (; s != NULL; s = s->next)
    {
        s->needs_image = 0;
        if (s->kind <= MC_STMT_EXPR)
        {
            if (s->kind == MC_STMT_EXPR && (resolve_expr(ctx, s->u.expr, 1) != 0 ||
                                            ((s->needs_image = s->u.expr->needs_image), 0)))
            {
                return -1;
            }
            continue;
        }
        if (s->kind == MC_STMT_MULTI_ASSIGN)
        {
            int nrhs = s->u.multi.nrhs, nlhs = s->u.multi.nlhs;
            if (nrhs != nlhs && !(nrhs == 1 && s->u.multi.rhs[0]->kind == MC_EXPR_CALL))
            {
                return res_err(ctx, "Expected ',' in multi-target assignment");
            }
            for (int j = 0; j < nrhs; j++)
            {
                if (resolve_expr(ctx, s->u.multi.rhs[j], (nrhs == 1) ? nlhs : 1) != 0)
                {
                    return -1;
                }
                s->needs_image |= s->u.multi.rhs[j]->needs_image;
            }
            for (int j = 0; j < nlhs; j++)
            {
                mc_expr_t *lhs = s->u.multi.lhs[j];
                if (lhs->kind != MC_EXPR_IDENT)
                {
                    return res_err(ctx, "multi-target LHS must be an identifier");
                }
                mc_parse_ident_streamname(lhs->u.ident.name, lhs->u.ident.bare, &lhs->u.ident.loc);
                imageID imid   = mc_resolve_or_autoconnect_image(lhs->u.ident.name);
                int     rhs_im = (nrhs == nlhs) ? (s->u.multi.rhs[j]->res_kind == MC_RES_IMAGE)
                                                : mc_multi_lhs_is_image(s->u.multi.rhs[0], j);
                lhs->u.ident.id_kind = (imid != -1) ? MC_ID_IMAGE : rhs_im ? MC_ID_NEW : MC_ID_VAR;
                lhs->u.ident.im_id   = imid;
                lhs->u.ident.var_id  = (imid != -1 || rhs_im) ? -1 : variable_ID(lhs->u.ident.bare);
                s->needs_image |= (imid != -1 || rhs_im);
                const mc_expr_t *re  = s->u.multi.rhs[(nrhs == nlhs) ? j : 0];
                uint8_t          rdt = rhs_im ? (re->res_dt ? re->res_dt : _DATATYPE_FLOAT) : 255;
                (void) (imid == -1 && created_var_op(ctx, lhs->u.ident.bare, rdt));
            }
            continue;
        }
        int        nb    = (s->kind == MC_STMT_IF) ? s->u.if_stmt.nbranches : 3;
        mc_expr_t *fe[3] = { s->u.for_stmt.start, s->u.for_stmt.stop, s->u.for_stmt.step };
        for (int b = 0; b < nb; b++)
        {
            mc_expr_t *ce = (s->kind == MC_STMT_IF) ? s->u.if_stmt.branches[b].cond : fe[b];
            if (ce != NULL && (resolve_expr(ctx, ce, 1) != 0 || ce->res_kind == MC_RES_IMAGE))
            {
                return (ce->res_kind == MC_RES_IMAGE) ? res_err(ctx, "%s", k_im_cond_err) : -1;
            }
            s->needs_image |= (ce != NULL && ce->needs_image);
            int saved = ctx->in_lazy_branch;
            if (s->kind == MC_STMT_IF &&
                ((ctx->in_lazy_branch = 1),
                 resolve_stmt_list(ctx, s->u.if_stmt.branches[b].body) != 0))
            {
                return (ctx->in_lazy_branch = saved), -1;
            }
            ctx->in_lazy_branch = saved;
        }
        if (s->kind == MC_STMT_FOR && (created_var_op(ctx, s->u.for_stmt.var, 255),
                                       resolve_stmt_list(ctx, s->u.for_stmt.body) != 0))
        {
            return -1;
        }
    }
    return 0;
}

int mc_resolve_prog_ex(mc_stmt_t *prog, int in_unit, char *err, size_t errsz)
{
    (void) (err && errsz > 0 && (err[0] = '\0'));
    mc_resolve_ctx_t ctx = { .err = err, .errsz = errsz, .in_unit = in_unit };
    return (prog == NULL) ? -1 : resolve_stmt_list(&ctx, prog);
}
