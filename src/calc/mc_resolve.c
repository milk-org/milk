// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_memory/variable_ID.h"
#include "mc_ast.h"
#include "mcalc_host.h"
#include "milk_calc_ext.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

#define MC_MAX_CREATED_VARS 64
#define MC_UDEF_MAX_DEPTH 4

typedef struct
{
    const mcalc_udef_t *u;
    mc_expr_t *const   *args;
} mc_def_res_frame_t;

typedef struct
{
    char              *err;
    size_t             errsz;
    int                in_lazy_branch;
    int                slice_ax_size;
    char               created_vars[MC_MAX_CREATED_VARS][256];
    int                ncreated;
    mc_def_res_frame_t def_stk[MC_UDEF_MAX_DEPTH];
    int                def_depth;
} mc_resolve_ctx_t;

__attribute__((format(printf, 2, 3))) static int res_err(mc_resolve_ctx_t *ctx,
                                                         const char       *fmt,
                                                         ...)
{
    if (ctx->err != NULL && ctx->errsz > 0 && ctx->err[0] == '\0')
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(ctx->err, ctx->errsz, fmt, ap);
        va_end(ap);
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

static int created_var_op(mc_resolve_ctx_t *ctx, const char *bare, int add)
{
    for (int i = 0; i < ctx->ncreated; i++)
    {
        if (strcmp(ctx->created_vars[i], bare) == 0)
        {
            return 1;
        }
    }
    if (add && bare != NULL && bare[0] != '\0' && ctx->ncreated < MC_MAX_CREATED_VARS)
    {
        snprintf(ctx->created_vars[ctx->ncreated++], sizeof(ctx->created_vars[0]), "%s", bare);
    }
    return 0;
}

static int lookup_def_param(const mc_resolve_ctx_t *ctx, const char *bare)
{
    for (int j = 0; ctx->def_depth > 0 && j < ctx->def_stk[ctx->def_depth - 1].u->nargs; j++)
    {
        if (strcmp(ctx->def_stk[ctx->def_depth - 1].u->arg_names[j], bare) == 0)
        {
            return j;
        }
    }
    return -1;
}

static int resolve_expr(mc_resolve_ctx_t *ctx, mc_expr_t *e, int nlhs);

static int resolve_ident(mc_resolve_ctx_t *ctx, mc_expr_t *e)
{
    mc_parse_ident_streamname(e->u.ident.name, e->u.ident.bare, &e->u.ident.loc);
    const char *bare         = e->u.ident.bare;
    e->u.ident.var_id        = -1;
    e->u.ident.im_id         = -1;
    int pidx                 = (e->u.ident.loc == 0) ? lookup_def_param(ctx, bare) : -1;
    e->u.ident.def_param_idx = pidx;
    if (pidx >= 0)
    {
        const mc_expr_t *ae = ctx->def_stk[ctx->def_depth - 1].args[pidx];
        e->u.ident.id_kind  = MC_ID_DEF_PARAM;
        e->u.ident.im_id    = (ae->kind == MC_EXPR_IDENT) ? ae->u.ident.im_id : -1;
        e->res_kind         = ae->res_kind;
        e->res_dt           = ae->res_dt;
        e->res_naxis        = ae->res_naxis;
        memcpy(e->res_size, ae->res_size, sizeof(e->res_size));
        e->needs_image = ae->needs_image;
        return 0;
    }

    long vid = variable_ID(bare);
    if (vid != -1 && milk_data.variable[vid].type != 3)
    {
        int vt             = milk_data.variable[vid].type;
        e->u.ident.id_kind = MC_ID_VAR;
        e->u.ident.var_id  = vid;
        e->res_kind    = (vt == 1) ? MC_RES_SCALAR_L : (vt == 2) ? MC_RES_STRING : MC_RES_SCALAR_D;
        e->needs_image = 0;
        return 0;
    }

    imageID imid = mc_resolve_or_autoconnect_image(e->u.ident.name);
    if (imid != -1)
    {
        e->u.ident.id_kind = MC_ID_IMAGE;
        e->u.ident.im_id   = imid;
        mc_set_expr_from_im(e, &milk_data.image[imid]);
        return 0;
    }

    const char *cval = mc_host_str_get(bare);
    if (cval == NULL && mc_is_numeric_env_var(bare))
    {
        cval = getenv(bare);
    }
    if (cval != NULL || created_var_op(ctx, bare, 0))
    {
        char *ep1 = NULL, *ep2 = NULL;
        if (cval != NULL)
        {
            (void) strtol(cval, &ep1, 0);
            (void) strtod(cval, &ep2);
        }
        e->u.ident.id_kind = MC_ID_VAR;
        e->res_kind        = (cval == NULL)                  ? MC_RES_SCALAR_D
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

    if (e->u.ident.loc == 0 &&
        (strcmp(bare, "pi") == 0 || strcmp(bare, "nan") == 0 || strcmp(bare, "inf") == 0))
    {
        e->u.ident.id_kind   = MC_ID_CONST;
        e->u.ident.const_val = (bare[0] == 'p') ? M_PI : (bare[0] == 'n') ? NAN : INFINITY;
        e->res_kind          = MC_RES_SCALAR_D;
        return 0;
    }

    if (!ctx->in_lazy_branch)
    {
        const char *gone = milk_entity_ns_removed(e->u.ident.name);
        return res_err(ctx, "undefined identifier '%s'%s%s", e->u.ident.name, gone ? ": " : "",
                       gone ? gone : "");
    }
    e->u.ident.id_kind = MC_ID_UNRESOLVED;
    e->res_kind        = MC_RES_SCALAR_D;
    return 0;
}

static int resolve_slice(mc_resolve_ctx_t *ctx, mc_expr_t *e, int is_lhs)
{
    mc_parse_ident_streamname(e->u.slice.name, e->u.slice.bare, NULL);
    int              pidx = lookup_def_param(ctx, e->u.slice.bare);
    const mc_expr_t *ae   = (pidx >= 0) ? ctx->def_stk[ctx->def_depth - 1].args[pidx] : NULL;
    imageID          imid = (ae != NULL && ae->kind == MC_EXPR_IDENT) ? ae->u.ident.im_id : -1;
    if (imid == -1)
    {
        imid = mc_resolve_or_autoconnect_image(e->u.slice.bare);
    }
    if (imid == -1)
    {
        return res_err(ctx, is_lhs ? "Target image not found for slice assignment"
                                   : "Source image not found for slice");
    }

    const IMAGE *im       = &milk_data.image[imid];
    e->u.slice.im_id      = imid;
    int saved_slice_ax_sz = ctx->slice_ax_size;
    int single  = (e->u.slice.naxis == (int) im->md[0].naxis) && !is_complex_dt(im->md[0].datatype);
    int bad_bnd = 0;

    for (int ax = 0; ax < e->u.slice.naxis; ax++)
    {
        ctx->slice_ax_size       = (ax < im->md[0].naxis) ? (int) im->md[0].size[ax] : 1;
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
    }
    ctx->slice_ax_size = saved_slice_ax_sz;

    if (is_lhs && e->u.slice.naxis == 1 && !e->u.slice.axes[0].has_colon &&
        !e->u.slice.axes[0].is_full && !e->u.slice.axes[0].is_rev_full &&
        e->u.slice.axes[0].start != NULL && e->u.slice.axes[0].start->res_kind == MC_RES_IMAGE)
    {
        e->u.slice.is_mask_index = 1;
        mc_set_expr_from_im(e, im);
        return 0;
    }

    if (e->u.slice.naxis > (int) im->md[0].naxis || bad_bnd)
    {
        return (e->u.slice.naxis > (int) im->md[0].naxis)
                   ? res_err(ctx, "slice axis %d > naxis %d", e->u.slice.naxis - 1,
                             (int) im->md[0].naxis)
                   : res_err(ctx, "Invalid slice bound expression");
    }

    e->u.slice.is_single_pixel = single;
    if (single && !is_lhs)
    {
        e->res_kind    = is_int_dt(im->md[0].datatype) ? MC_RES_SCALAR_L : MC_RES_SCALAR_D;
        e->needs_image = 0;
    }
    else
    {
        mc_set_expr_from_im(e, im);
    }
    return 0;
}

static int resolve_call(mc_resolve_ctx_t *ctx, mc_expr_t *e, int nlhs)
{
    e->u.call.udef_id = -1;
    for (int i = 0, n = mcalc_udef_count(); i < n; i++)
    {
        const mcalc_udef_t *u = mcalc_udef_get(i);
        if (u != NULL && strcmp(u->name, e->u.call.name) == 0)
        {
            e->u.call.udef_id = i;
            break;
        }
    }
    e->u.call.ext_id = -1;
    if (e->u.call.udef_id < 0)
    {
        char        base[64];
        const char *colon = strchr(e->u.call.name, ':');
        size_t len = (colon != NULL) ? (size_t) (colon - e->u.call.name) : strlen(e->u.call.name);
        if (len > 0 && len < sizeof(base))
        {
            memcpy(base, e->u.call.name, len);
            base[len]        = '\0';
            e->u.call.ext_id = milk_calc_ext_find(base);
        }
    }
    e->u.call.fn =
        (e->u.call.udef_id < 0 && e->u.call.ext_id < 0) ? mc_func_find(e->u.call.name) : NULL;

    if (e->u.call.udef_id < 0 && e->u.call.ext_id < 0 && e->u.call.fn == NULL)
    {
        return res_err(ctx, "unknown function '%s'", e->u.call.name);
    }

    for (int i = 0; i < e->u.call.nargs; i++)
    {
        if (e->u.call.kw_names[i][0] != '\0' && e->u.call.ext_id < 0)
        {
            return res_err(ctx, "keyword arguments are only supported for user extensions");
        }
        if (resolve_expr(ctx, e->u.call.args[i], 1) != 0)
        {
            return -1;
        }
    }

    if (e->u.call.udef_id >= 0)
    {
        const mcalc_udef_t *u = mcalc_udef_get(e->u.call.udef_id);
        if (u == NULL || e->u.call.nargs != u->nargs || ctx->def_depth >= MC_UDEF_MAX_DEPTH)
        {
            return (u == NULL || e->u.call.nargs != u->nargs)
                       ? res_err(ctx, "wrong number of arguments for '%s'", e->u.call.name)
                       : res_err(ctx, "User function call depth exceeded");
        }
        ctx->def_stk[ctx->def_depth] = (mc_def_res_frame_t) { u, e->u.call.args };
        mc_tok_list_t *btl           = calloc(1, sizeof(mc_tok_list_t));
        if (btl == NULL || mc_lex(u->body, btl) != 0)
        {
            res_err(ctx, "%s", btl ? btl->errmsg : "out of memory");
            free(btl);
            return -1;
        }
        mc_stmt_t *bst = mc_parse_stmt(btl, ctx->err, ctx->errsz);
        free(btl);
        if (bst == NULL || bst->kind != MC_STMT_EXPR || bst->u.expr == NULL)
        {
            mc_stmt_free(bst);
            return res_err(ctx, "invalid user function body in '%s'", u->name);
        }
        ctx->def_depth++;
        int brc = resolve_expr(ctx, bst->u.expr, 1);
        ctx->def_depth--;
        if (brc != 0)
        {
            mc_stmt_free(bst);
            return -1;
        }
        e->res_kind  = bst->u.expr->res_kind;
        e->res_dt    = bst->u.expr->res_dt;
        e->res_naxis = bst->u.expr->res_naxis;
        memcpy(e->res_size, bst->u.expr->res_size, sizeof(e->res_size));
        e->needs_image = bst->u.expr->needs_image;
        mc_expr_free(e->u.call.udef_body);
        e->u.call.udef_body = bst->u.expr;
        bst->u.expr         = NULL;
        mc_stmt_free(bst);
        return 0;
    }

    if (e->u.call.ext_id >= 0)
    {
        e->res_kind    = MC_RES_IMAGE;
        e->needs_image = 1;
        return 0;
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

    int any_im = 0, any_cplx = 0;
    for (int i = 0; i < nargs; i++)
    {
        any_im |= (e->u.call.args[i]->needs_image || e->u.call.args[i]->res_kind == MC_RES_IMAGE);
        any_cplx |= (e->u.call.args[i]->res_kind == MC_RES_IMAGE &&
                     is_complex_dt(e->u.call.args[i]->res_dt));
    }

    if (any_cplx && strcmp(fn->name, "abs") != 0 && strcmp(fn->name, "fabs") != 0 &&
        (fn->kind == MC_FN_SCALAR || fn->kind == MC_FN_WHERE || fn->kind == MC_FN_PRNG ||
         fn->kind == MC_FN_LUT || (fn->kind >= MC_FN_REDUCE_D && fn->kind <= MC_FN_DOT)))
    {
        return res_err(ctx, "Function of real pixels applied to a complex image; use real(), "
                            "imag() or abs() first");
    }

    switch (fn->kind)
    {
    case MC_FN_SCALAR:
    case MC_FN_WHERE:
        if (fn->kind == MC_FN_SCALAR && nargs == 3 &&
            (e->u.call.args[1]->res_kind == MC_RES_IMAGE ||
             e->u.call.args[2]->res_kind == MC_RES_IMAGE))
        {
            return res_err(ctx, "Only the first argument may be an image");
        }
        e->needs_image = any_im;
        e->res_kind    = any_im                      ? MC_RES_IMAGE
                         : (fn->kind == MC_FN_WHERE) ? e->u.call.args[1]->res_kind
                                                     : MC_RES_SCALAR_D;
        return 0;

    case MC_FN_PRNG:
        if (nargs == 0 || (nargs == 1 && e->u.call.args[0]->res_kind != MC_RES_IMAGE))
        {
            e->needs_image = 0;
            e->res_kind    = MC_RES_SCALAR_D;
            return 0;
        }
        if (e->u.call.args[0]->res_kind != MC_RES_IMAGE ||
            (nargs == 2 && e->u.call.args[1]->res_kind == MC_RES_IMAGE))
        {
            return res_err(ctx, "rand: expected rand([im[, seed]])");
        }
        e->needs_image = 1;
        e->res_kind    = MC_RES_IMAGE;
        return 0;

    case MC_FN_LUT:
        if (nargs == 3 || e->u.call.args[1]->res_kind != MC_RES_IMAGE ||
            (nargs == 4 && (e->u.call.args[2]->res_kind == MC_RES_IMAGE ||
                            e->u.call.args[3]->res_kind == MC_RES_IMAGE)))
        {
            return res_err(ctx, "lut: expected lut(x, tbl[, x0, dx])");
        }
        e->needs_image = e->u.call.args[0]->needs_image;
        e->res_kind    = e->needs_image ? MC_RES_IMAGE : MC_RES_SCALAR_D;
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
        e->needs_image = 0;
        e->res_kind    = (strcmp(fn->name, "strlen") == 0) ? MC_RES_SCALAR_L : MC_RES_STRING;
        return 0;

    case MC_FN_REDUCE_D:
    case MC_FN_REDUCE_L:
    case MC_FN_ISTATS:
    case MC_FN_ICENTROID:
    case MC_FN_EAGER_RED:
    case MC_FN_DOT:
        if (e->u.call.args[0]->res_kind != MC_RES_IMAGE ||
            (fn->kind == MC_FN_DOT && e->u.call.args[1]->res_kind != MC_RES_IMAGE))
        {
            return res_err(ctx, (fn->kind == MC_FN_DOT) ? "Expected two image arguments"
                                                        : "Expected image argument");
        }
        e->needs_image = 1;
        e->res_kind    = (fn->kind == MC_FN_REDUCE_L) ? MC_RES_SCALAR_L : MC_RES_SCALAR_D;
        return 0;

    case MC_FN_COORD:
        if (nargs == 2 || e->u.call.args[0]->res_kind != MC_RES_IMAGE ||
            (nargs == 3 && (e->u.call.args[1]->res_kind == MC_RES_IMAGE ||
                            e->u.call.args[2]->res_kind == MC_RES_IMAGE)))
        {
            return res_err(ctx, "Expected an image, then optionally the centre: x, y");
        }
        e->needs_image = 1;
        e->res_kind    = MC_RES_IMAGE;
        return 0;

    default:
        e->needs_image = 1;
        e->res_kind    = MC_RES_IMAGE;
        return 0;
    }
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
            e->kind = MC_EXPR_IDENT;
            snprintf(e->u.ident.name, sizeof(e->u.ident.name), "end");
            return resolve_ident(ctx, e);
        }
        if (ctx->slice_ax_size <= 0)
        {
            return res_err(ctx, "unexpected 'end' outside slice brackets");
        }
        e->kind        = MC_EXPR_LONG;
        e->u.lval      = (long) ctx->slice_ax_size - 1;
        e->res_kind    = MC_RES_SCALAR_L;
        e->needs_image = 0;
        return 0;

    case MC_EXPR_IDENT:
        return resolve_ident(ctx, e);

    case MC_EXPR_SLICE:
        return resolve_slice(ctx, e, 0);

    case MC_EXPR_UNARY:
        if (resolve_expr(ctx, e->u.unary.operand, 1) != 0 ||
            (e->u.unary.operand->res_kind == MC_RES_STRING && e->u.unary.op == MC_TOK_MINUS))
        {
            return (e->u.unary.operand->res_kind == MC_RES_STRING && e->u.unary.op == MC_TOK_MINUS)
                       ? res_err(ctx, "Cannot negate expression")
                       : -1;
        }
        e->needs_image = e->u.unary.operand->needs_image;
        e->res_dt      = e->u.unary.operand->res_dt;
        e->res_naxis   = e->u.unary.operand->res_naxis;
        memcpy(e->res_size, e->u.unary.operand->res_size, sizeof(e->res_size));
        e->res_kind = (e->u.unary.operand->res_kind == MC_RES_IMAGE) ? MC_RES_IMAGE
                      : (e->u.unary.op == MC_TOK_NOT || e->u.unary.op == MC_TOK_BNOT)
                          ? MC_RES_SCALAR_L
                          : e->u.unary.operand->res_kind;
        return 0;

    case MC_EXPR_BINARY:
    {
        if (resolve_expr(ctx, e->u.binary.left, 1) != 0)
        {
            return -1;
        }
        int saved_lazy = ctx->in_lazy_branch;
        ctx->in_lazy_branch |= ((e->u.binary.op == MC_TOK_AND || e->u.binary.op == MC_TOK_OR) &&
                                e->u.binary.left->needs_image == 0);
        int rrc             = resolve_expr(ctx, e->u.binary.right, 1);
        ctx->in_lazy_branch = saved_lazy;
        if (rrc != 0)
        {
            return -1;
        }
        mc_tok_type_t op = e->u.binary.op;
        mc_res_kind_t lk = e->u.binary.left->res_kind, rk = e->u.binary.right->res_kind;
        e->needs_image = (e->u.binary.left->needs_image || e->u.binary.right->needs_image);
        int is_int_op  = mc_is_rel_tok(op) || op == MC_TOK_EQ || op == MC_TOK_NE ||
                         op == MC_TOK_AND || op == MC_TOK_OR || op == MC_TOK_BAND ||
                         op == MC_TOK_BOR || op == MC_TOK_BXOR || op == MC_TOK_SHL ||
                         op == MC_TOK_SHR;
        e->res_kind    = (lk == MC_RES_IMAGE || rk == MC_RES_IMAGE) ? MC_RES_IMAGE
                         : (op == MC_TOK_PLUS && (lk == MC_RES_STRING || rk == MC_RES_STRING))
                             ? MC_RES_STRING
                         : is_int_op                                        ? MC_RES_SCALAR_L
                         : (lk == MC_RES_SCALAR_D || rk == MC_RES_SCALAR_D) ? MC_RES_SCALAR_D
                                                                            : MC_RES_SCALAR_L;
        return 0;
    }

    case MC_EXPR_CHAIN:
    {
        if (resolve_expr(ctx, e->u.chain.operands[0], 1) != 0)
        {
            return -1;
        }
        int saved_lazy = ctx->in_lazy_branch, any_im = e->u.chain.operands[0]->needs_image;
        for (int i = 1; i <= e->u.chain.nops; i++)
        {
            ctx->in_lazy_branch |= !any_im;
            if (resolve_expr(ctx, e->u.chain.operands[i], 1) != 0)
            {
                ctx->in_lazy_branch = saved_lazy;
                return -1;
            }
            any_im |= e->u.chain.operands[i]->needs_image;
        }
        ctx->in_lazy_branch = saved_lazy;
        e->needs_image      = any_im;
        e->res_kind         = any_im ? MC_RES_IMAGE : MC_RES_SCALAR_L;
        return 0;
    }

    case MC_EXPR_TERNARY:
    {
        if (resolve_expr(ctx, e->u.ternary.cond, 1) != 0)
        {
            return -1;
        }
        int saved_lazy = ctx->in_lazy_branch;
        ctx->in_lazy_branch |= (e->u.ternary.cond->needs_image == 0);
        int rc              = (resolve_expr(ctx, e->u.ternary.t_expr, 1) == 0 &&
                               resolve_expr(ctx, e->u.ternary.f_expr, 1) == 0)
                                  ? 0
                                  : -1;
        ctx->in_lazy_branch = saved_lazy;
        if (rc != 0)
        {
            return -1;
        }
        e->needs_image = (e->u.ternary.cond->needs_image || e->u.ternary.t_expr->needs_image ||
                          e->u.ternary.f_expr->needs_image);
        e->res_kind    = e->needs_image ? MC_RES_IMAGE : e->u.ternary.t_expr->res_kind;
        return 0;
    }

    case MC_EXPR_CALL:
        return resolve_call(ctx, e, nlhs);

    case MC_EXPR_ASSIGN:
    {
        mc_expr_t    *target = e->u.assign.target;
        mc_expr_t    *rhs    = e->u.assign.rhs;
        mc_tok_type_t op     = e->u.assign.op;
        if (resolve_expr(ctx, rhs, 1) != 0)
        {
            return -1;
        }
        if (target->kind == MC_EXPR_SLICE)
        {
            if (resolve_slice(ctx, target, 1) != 0)
            {
                return -1;
            }
            int spix       = target->u.slice.is_single_pixel && !rhs->needs_image &&
                             rhs->res_kind != MC_RES_IMAGE && rhs->res_kind != MC_RES_STRING;
            e->needs_image = !spix;
            e->res_kind    = !spix ? MC_RES_IMAGE
                             : (rhs->res_kind == MC_RES_SCALAR_L && op == MC_TOK_ASSIGN)
                                 ? MC_RES_SCALAR_L
                                 : MC_RES_SCALAR_D;
            return 0;
        }
        if (target->kind != MC_EXPR_IDENT)
        {
            return res_err(ctx, "invalid assignment target");
        }
        mc_parse_ident_streamname(target->u.ident.name, target->u.ident.bare, &target->u.ident.loc);
        const char *bare    = target->u.ident.bare;
        const char *dot1    = strchr(target->u.ident.name, '.');
        int         is_prop = (dot1 != NULL && strchr(dot1 + 1, '.') != NULL &&
                               milk_entity_ns_kind(target->u.ident.name, NULL) != MILK_NS_NONE);
        long        vid     = is_prop ? -1 : variable_ID(bare);
        if (is_prop || (vid != -1 && milk_data.variable[vid].type != 3))
        {
            if (rhs->res_kind == MC_RES_IMAGE)
            {
                return res_err(ctx,
                               is_prop ? "cannot assign image to property"
                                       : "'%s' is a variable (calculator or bash), not an image",
                               bare);
            }
            target->u.ident.id_kind = is_prop ? MC_ID_PROP : MC_ID_VAR;
            target->u.ident.var_id  = vid;
            e->needs_image          = rhs->needs_image;
            e->res_kind             = rhs->res_kind;
            return 0;
        }
        imageID imid = mc_resolve_or_autoconnect_image(target->u.ident.name);
        if (imid != -1)
        {
            target->u.ident.id_kind = MC_ID_IMAGE;
            target->u.ident.im_id   = imid;
            e->needs_image          = 1;
            e->res_kind             = MC_RES_IMAGE;
            return 0;
        }
        if (op != MC_TOK_ASSIGN)
        {
            if ((mc_host_str_get(bare) == NULL && !mc_is_numeric_env_var(bare) &&
                 !created_var_op(ctx, bare, 0)) ||
                rhs->res_kind == MC_RES_IMAGE)
            {
                return (rhs->res_kind == MC_RES_IMAGE)
                           ? res_err(ctx, "'%s' is a variable (calculator or bash), not an image",
                                     bare)
                           : res_err(ctx, "Variable not found for compound assign");
            }
            target->u.ident.id_kind = MC_ID_VAR;
            target->u.ident.var_id  = -1;
            e->needs_image          = rhs->needs_image;
            e->res_kind             = MC_RES_SCALAR_D;
            return 0;
        }
        target->u.ident.id_kind = (rhs->res_kind == MC_RES_IMAGE) ? MC_ID_NEW : MC_ID_VAR;
        target->u.ident.var_id  = -1;
        created_var_op(ctx, bare, rhs->res_kind != MC_RES_IMAGE);
        e->needs_image = (rhs->res_kind == MC_RES_IMAGE) ? 1 : rhs->needs_image;
        e->res_kind    = rhs->res_kind;
        return 0;
    }

    case MC_EXPR_VEC_LIT:
        for (int i = 0; i < e->u.vec_lit.nelem; i++)
        {
            if (resolve_expr(ctx, e->u.vec_lit.elems[i], 1) != 0)
            {
                return -1;
            }
            if (e->u.vec_lit.elems[i]->res_kind == MC_RES_IMAGE ||
                e->u.vec_lit.elems[i]->res_kind == MC_RES_STRING)
            {
                return res_err(ctx, "vec: arguments must be numbers");
            }
        }
        e->needs_image = 1;
        e->res_kind    = MC_RES_IMAGE;
        e->res_dt      = _DATATYPE_FLOAT;
        e->res_naxis   = 1;
        e->res_size[0] = (uint32_t) e->u.vec_lit.nelem;
        e->res_size[1] = e->res_size[2] = 1;
        return 0;

    case MC_EXPR_COMP:
        for (int g = 0; g < e->u.comp.ngens; g++)
        {
            if (resolve_expr(ctx, e->u.comp.gens[g].start, 1) != 0 ||
                resolve_expr(ctx, e->u.comp.gens[g].stop, 1) != 0 ||
                (e->u.comp.gens[g].step != NULL &&
                 resolve_expr(ctx, e->u.comp.gens[g].step, 1) != 0))
            {
                return -1;
            }
            created_var_op(ctx, e->u.comp.gens[g].var, 1);
        }
        if (resolve_expr(ctx, e->u.comp.body, 1) != 0)
        {
            return -1;
        }
        e->needs_image = 1;
        e->res_kind    = MC_RES_IMAGE;
        return 0;
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
        switch (s->kind)
        {
        case MC_STMT_EMPTY:
        case MC_STMT_DEF:
        case MC_STMT_SKIP:
            s->needs_image = 0;
            break;

        case MC_STMT_EXPR:
            if (resolve_expr(ctx, s->u.expr, 1) != 0)
            {
                return -1;
            }
            s->needs_image = s->u.expr->needs_image;
            break;

        case MC_STMT_MULTI_ASSIGN:
        {
            int any_im = 0;
            if (s->u.multi.nrhs == 1 && s->u.multi.rhs[0]->kind == MC_EXPR_CALL)
            {
                if (resolve_expr(ctx, s->u.multi.rhs[0], s->u.multi.nlhs) != 0)
                {
                    return -1;
                }
                any_im = s->u.multi.rhs[0]->needs_image;
            }
            else if (s->u.multi.nrhs == s->u.multi.nlhs)
            {
                for (int j = 0; j < s->u.multi.nrhs; j++)
                {
                    if (resolve_expr(ctx, s->u.multi.rhs[j], 1) != 0)
                    {
                        return -1;
                    }
                    any_im |= s->u.multi.rhs[j]->needs_image;
                }
            }
            else
            {
                return res_err(ctx, "Expected ',' in multi-target assignment");
            }

            for (int j = 0; j < s->u.multi.nlhs; j++)
            {
                mc_expr_t *lhs = s->u.multi.lhs[j];
                if (lhs->kind != MC_EXPR_IDENT)
                {
                    return res_err(ctx, "multi-target LHS must be an identifier");
                }
                mc_parse_ident_streamname(lhs->u.ident.name, lhs->u.ident.bare, &lhs->u.ident.loc);
                imageID imid         = mc_resolve_or_autoconnect_image(lhs->u.ident.name);
                lhs->u.ident.id_kind = (imid != -1) ? MC_ID_IMAGE : MC_ID_VAR;
                lhs->u.ident.im_id   = imid;
                lhs->u.ident.var_id  = (imid != -1) ? -1 : variable_ID(lhs->u.ident.bare);
                any_im |= (imid != -1);
                created_var_op(ctx, lhs->u.ident.bare, imid == -1);
            }
            s->needs_image = any_im;
            break;
        }

        case MC_STMT_IF:
        {
            int any_im = 0;
            for (int b = 0; b < s->u.if_stmt.nbranches; b++)
            {
                mc_if_branch_t *br = &s->u.if_stmt.branches[b];
                if (br->cond != NULL)
                {
                    if (resolve_expr(ctx, br->cond, 1) != 0 || br->cond->res_kind == MC_RES_IMAGE)
                    {
                        return (br->cond->res_kind == MC_RES_IMAGE)
                                   ? res_err(ctx, "%s", k_im_cond_err)
                                   : -1;
                    }
                    any_im |= br->cond->needs_image;
                }
                int saved_lazy      = ctx->in_lazy_branch;
                ctx->in_lazy_branch = 1;
                int rc              = resolve_stmt_list(ctx, br->body);
                ctx->in_lazy_branch = saved_lazy;
                if (rc != 0)
                {
                    return -1;
                }
            }
            s->needs_image = any_im;
            break;
        }

        case MC_STMT_FOR:
            if (resolve_expr(ctx, s->u.for_stmt.start, 1) != 0 ||
                resolve_expr(ctx, s->u.for_stmt.stop, 1) != 0 ||
                (s->u.for_stmt.step != NULL && resolve_expr(ctx, s->u.for_stmt.step, 1) != 0))
            {
                return -1;
            }
            if (s->u.for_stmt.start->res_kind == MC_RES_IMAGE ||
                s->u.for_stmt.stop->res_kind == MC_RES_IMAGE ||
                (s->u.for_stmt.step != NULL && s->u.for_stmt.step->res_kind == MC_RES_IMAGE))
            {
                return res_err(ctx, "%s", k_im_cond_err);
            }
            s->needs_image = (s->u.for_stmt.start->needs_image || s->u.for_stmt.stop->needs_image ||
                              (s->u.for_stmt.step != NULL && s->u.for_stmt.step->needs_image));
            created_var_op(ctx, s->u.for_stmt.var, 1);
            if (resolve_stmt_list(ctx, s->u.for_stmt.body) != 0)
            {
                return -1;
            }
            break;
        }
    }
    return 0;
}

int mc_resolve_stmt(mc_stmt_t *s, char *err, size_t errsz)
{
    return (s == NULL) ? -1 : mc_resolve_prog(s, err, errsz);
}

int mc_resolve_prog(mc_stmt_t *prog, char *err, size_t errsz)
{
    if (err != NULL && errsz > 0)
    {
        err[0] = '\0';
    }
    mc_resolve_ctx_t ctx = { .err = err, .errsz = errsz };
    return resolve_stmt_list(&ctx, prog);
}
