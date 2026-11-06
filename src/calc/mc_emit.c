// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "mc_plan.h"
#include "mc_prelude.h"

typedef struct
{
    char  *buf;
    size_t len, cap;
    int    oom;
} emit_buf_t;

__attribute__((format(printf, 2, 3))) static void eb_printf(emit_buf_t *b, const char *fmt, ...)
{
    if (b->oom)
    {
        return;
    }
    va_list ap, ap2;
    va_start(ap, fmt), va_copy(ap2, ap);
    int need = vsnprintf(b->buf ? b->buf + b->len : NULL, b->cap - b->len, fmt, ap);
    va_end(ap);
    if (need >= 0 && b->len + (size_t) need + 1 > b->cap)
    {
        size_t ncap = b->cap ? b->cap * 2 : 8192;
        while (ncap < b->len + (size_t) need + 1)
        {
            ncap *= 2;
        }
        char *nb = realloc(b->buf, ncap);
        (void) ((nb != NULL && (b->buf = nb, b->cap = ncap,
                                vsnprintf(b->buf + b->len, b->cap - b->len, fmt, ap2), 1)) ||
                (need = -1));
    }
    va_end(ap2);
    (void) ((need < 0 && (b->oom = 1)) || (b->len += (size_t) need));
}

static inline int is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}

typedef struct
{
    uint8_t     dt;
    const char *ct, *sfx, *stm;
} dt_info_t;

static const dt_info_t *get_dt_info(uint8_t dt)
{
    static const dt_info_t k_dt[] = {
        { _DATATYPE_DOUBLE, "double", "f64", NULL },
        { _DATATYPE_UINT8, "uint8_t", "u8", "mc_st_u8" },
        { _DATATYPE_INT8, "int8_t", "i8", "mc_st_i8" },
        { _DATATYPE_UINT16, "uint16_t", "u16", "mc_st_u16" },
        { _DATATYPE_INT16, "int16_t", "i16", "mc_st_i16" },
        { _DATATYPE_UINT32, "uint32_t", "u32", "mc_st_u32" },
        { _DATATYPE_INT32, "int32_t", "i32", "mc_st_i32" },
        { _DATATYPE_UINT64, "uint64_t", "u64", "mc_st_u64" },
        { _DATATYPE_INT64, "int64_t", "i64", "mc_st_i64" },
        { _DATATYPE_COMPLEX_FLOAT, "float _Complex", "f32", NULL },
        { _DATATYPE_COMPLEX_DOUBLE, "double _Complex", "f64", NULL },
        { 0, "float", "f32", NULL },
    };
    int i = 0;
    while (k_dt[i].dt != 0 && k_dt[i].dt != dt)
    {
        i++;
    }
    return &k_dt[i];
}

typedef struct
{
    const mcalc_prog_t *prog;
    const mc_pass_t    *pass;
    emit_buf_t         *eb, *pre_eb;
    const char         *wt, *ct;
    const mcalc_udef_t *def_stk[4];
    mc_expr_t *const   *def_args[4];
    const mc_expr_t    *sub_t0;
    int                 def_depth, cur_out_idx, chain_seq, fast_x_interior, fast_div;
    int                 used_complex, used_coord_shift, used_spatial;
} emit_ctx_t;

static int find_earlier_out_in_pass(const emit_ctx_t *c, int sidx)
{
    for (int i = c->cur_out_idx - 1; i >= 0; i--)
    {
        if (c->pass->outs[i].stmt_idx < c->pass->outs[c->cur_out_idx].stmt_idx &&
            c->pass->outs[i].slot_idx == sidx && c->pass->outs[i].assign_op == MC_TOK_ASSIGN &&
            c->pass->outs[i].lhs_slice == NULL)
        {
            return i;
        }
    }
    return -1;
}

static void emit_expr(emit_ctx_t *c, const mc_expr_t *e);

static void emit_scalar_arg(emit_ctx_t *c, const mc_expr_t *e, int is_int)
{
    if (e->kind == MC_EXPR_IDENT && e->u.ident.def_param_idx >= 0 && c->def_depth > 0)
    {
        c->def_depth--;
        emit_scalar_arg(c, c->def_args[c->def_depth][e->u.ident.def_param_idx], is_int);
        c->def_depth++;
        return;
    }
    int k = mc_find_scalar_op(c->pass, e, NULL, -1);
    if (k >= 0)
    {
        eb_printf(c->eb, is_int ? "_ish%d" : "_sc%d", k);
        return;
    }
    (void) (is_int && (eb_printf(c->eb, "llround((double)("), 1));
    emit_expr(c, e);
    (void) (is_int && (eb_printf(c->eb, "))"), 1));
}

static void emit_im_read(emit_ctx_t *c, int op_idx)
{
    const char *cast = is_complex_dt(c->pass->ops[op_idx].dt) ? c->ct : c->wt;
    eb_printf(c->eb,
              c->pass->is_1d_contig ? "((%s)_p%d[_i])"
              : c->fast_x_interior
                  ? (c->pass->ops[op_idx].is_x_bcast ? "((%s)_p%d[_ro%d])"
                                                     : "((%s)_p%d[_ro%d + (int64_t)x])")
                  : "((%s)_p%d[_ro%d + (int64_t)x * _st%d_0])",
              cast, op_idx, op_idx, op_idx);
}

static int resolve_im_arg_op(emit_ctx_t *c, const mc_expr_t *arg, int tbl)
{
    if (arg->kind == MC_EXPR_IDENT && arg->u.ident.def_param_idx >= 0 && c->def_depth > 0)
    {
        c->def_depth--;
        int r = resolve_im_arg_op(c, c->def_args[c->def_depth][arg->u.ident.def_param_idx], tbl);
        c->def_depth++;
        return r;
    }
    int eim = mc_find_im_op(c->pass, -1, 0, arg, tbl, 1);
    if (eim >= 0 || (arg->kind != MC_EXPR_SLICE && arg->kind != MC_EXPR_IDENT))
    {
        return eim;
    }
    int is_sl = (arg->kind == MC_EXPR_SLICE);
    int sidx  = mc_find_slot_by_bare(c->prog, is_sl ? arg->u.slice.name : arg->u.ident.name,
                                     is_sl ? arg->u.slice.bare : arg->u.ident.bare);
    return mc_find_im_op(c->pass, sidx, is_sl, is_sl ? arg : NULL, tbl, 0);
}

static const char *rel_op_str(mc_tok_type_t op)
{
    return (op == MC_TOK_LT)   ? "<"
           : (op == MC_TOK_LE) ? "<="
           : (op == MC_TOK_GT) ? ">"
           : (op == MC_TOK_GE) ? ">="
           : (op == MC_TOK_NE) ? "!="
                               : "==";
}

static void emit_ternary(emit_ctx_t      *c,
                         const mc_expr_t *cond,
                         const mc_expr_t *t_e,
                         const mc_expr_t *f_e)
{
    eb_printf(c->eb, "((("), emit_expr(c, cond), eb_printf(c->eb, ") != 0) ? (");
    emit_expr(c, t_e), eb_printf(c->eb, ") : ("), emit_expr(c, f_e), eb_printf(c->eb, "))");
}

static int is_const_num(const mc_expr_t *e, double *out_v)
{
    if (e->kind == MC_EXPR_LONG || e->kind == MC_EXPR_DOUBLE)
    {
        *out_v = (e->kind == MC_EXPR_LONG) ? (double) e->u.lval : e->u.dval;
        return 1;
    }
    return (e->kind == MC_EXPR_UNARY && e->u.unary.op == MC_TOK_MINUS &&
            is_const_num(e->u.unary.operand, out_v))
               ? (*out_v = -(*out_v), 1)
               : 0;
}

static void emit_call(emit_ctx_t *c, const mc_expr_t *e)
{
    if (e->u.call.udef_id >= 0)
    {
        c->def_stk[c->def_depth]    = mcalc_udef_get(e->u.call.udef_id);
        c->def_args[c->def_depth++] = e->u.call.args;
        eb_printf(c->eb, "("), emit_expr(c, e->u.call.udef_body), eb_printf(c->eb, ")");
        c->def_depth--;
        return;
    }

    const mc_func_t *fn    = e->u.call.fn;
    int              nargs = e->u.call.nargs, wd = c->pass->work_double;
    const char      *cfn = wd ? fn->c_f64 : fn->c_f32;

    switch (fn->kind)
    {
    case MC_FN_WHERE:
        emit_ternary(c, e->u.call.args[0], e->u.call.args[1], e->u.call.args[2]);
        return;

    case MC_FN_CPLX:
    {
        int is_c0 = (nargs >= 1 && is_complex_dt(e->u.call.args[0]->res_dt));
        if (!is_c0 && (strcmp(fn->name, "imag") == 0 || strcmp(fn->name, "real") == 0 ||
                       strcmp(fn->name, "conj") == 0))
        {
            (void) ((strcmp(fn->name, "imag") == 0 && (eb_printf(c->eb, "((%s)0)", c->wt), 1)) ||
                    (emit_expr(c, e->u.call.args[0]), 1));
            return;
        }
        c->used_complex |=
            (is_c0 || strcmp(fn->name, "cplx") == 0 || strcmp(fn->name, "expi") == 0);
        cfn = (strcmp(fn->name, "real") == 0)   ? (wd ? "creal" : "crealf")
              : (strcmp(fn->name, "imag") == 0) ? (wd ? "cimag" : "cimagf")
              : (strcmp(fn->name, "conj") == 0) ? (wd ? "conj" : "conjf")
              : (strcmp(fn->name, "cplx") == 0) ? (wd ? "mc_cplxd" : "mc_cplxf")
              : (strcmp(fn->name, "expi") == 0) ? (wd ? "mc_expid" : "mc_expif")
              : is_c0                           ? (wd ? "mc_cargd" : "mc_cargf")
                                                : (wd ? "mc_rargd" : "mc_rargf");
        break;
    }

    case MC_FN_COORD:
    {
        c->used_coord_shift = 1;
        int op_idx          = resolve_im_arg_op(c, e->u.call.args[0], 0);
        int ref_op          = (op_idx >= 0) ? op_idx : c->pass->outs[c->cur_out_idx].op_idx;
        int mode            = (strcmp(fn->name, "ix") == 0)   ? 0
                              : (strcmp(fn->name, "iy") == 0) ? 1
                              : (strcmp(fn->name, "ir") == 0) ? 2
                                                              : 3;
        eb_printf(c->eb, "%s(_vnx%d, _vny%d, (int32_t)x, (int32_t)y, ",
                  wd ? "mc_coordd" : "mc_coordf", ref_op, ref_op);
        (void) ((nargs == 3 && (emit_scalar_arg(c, e->u.call.args[1], 0), eb_printf(c->eb, ", "),
                                emit_scalar_arg(c, e->u.call.args[2], 0), 1)) ||
                (eb_printf(c->eb, "0, 0"), 1));
        eb_printf(c->eb, ", %d, %d)", (nargs == 3), mode);
        return;
    }

    case MC_FN_PRNG:
        eb_printf(c->eb, "%s(_i, (double)_sc%d)",
                  (strcmp(fn->name, "randn") == 0 || strcmp(fn->name, "grand") == 0)
                      ? (wd ? "mc_grandd" : "mc_grandf")
                      : (wd ? "mc_urandd" : "mc_urandf"),
                  mc_find_scalar_op(c->pass, NULL, e, -1));
        return;

    case MC_FN_SHIFT:
    {
        c->used_coord_shift = 1;
        int         op_idx  = resolve_im_arg_op(c, e->u.call.args[0], 0);
        int         is_roll = (strcmp(fn->name, "roll") == 0), sh_id = ++c->chain_seq;
        emit_buf_t *saved_eb = c->eb;
        if (c->fast_x_interior && c->pre_eb != NULL)
        {
            c->eb = c->pre_eb;
            eb_printf(c->eb, "        const int64_t _sh%d = mc_shift_row(&ops[%d], y, z", sh_id,
                      op_idx);
        }
        else
        {
            eb_printf(c->eb, "((%s)_p%d[mc_shift_idx(&ops[%d], x, y, z", c->wt, op_idx, op_idx);
        }
        for (int a = 1; a <= 3; a++)
        {
            eb_printf(c->eb, ", ");
            if (nargs > a)
            {
                emit_scalar_arg(c, e->u.call.args[a], 1);
            }
            else
            {
                eb_printf(c->eb, "0LL");
            }
        }
        eb_printf(c->eb, c->eb == c->pre_eb ? ", %d);\n" : ", %d)])", is_roll);
        if (c->eb == c->pre_eb)
        {
            c->eb = saved_eb;
            eb_printf(c->eb, "((%s)_p%d[_sh%d + (int64_t)x])", c->wt, op_idx, sh_id);
        }
        return;
    }

    case MC_FN_LUT:
    case MC_FN_GATHER:
    {
        c->used_spatial    = 1;
        int         is_lut = (fn->kind == MC_FN_LUT), is_i2 = (strcmp(fn->name, "interp2") == 0);
        int         tbl_op = resolve_im_arg_op(c, e->u.call.args[is_lut ? 1 : 0], 1);
        const char *sfx =
            get_dt_info((tbl_op >= 0) ? c->pass->ops[tbl_op].dt : _DATATYPE_FLOAT)->sfx;
        eb_printf(c->eb, "((%s)mc_%s_%s(&ops[%d], (double)(", c->wt,
                  is_lut  ? "lut"
                  : is_i2 ? "interp2"
                          : "gather",
                  sfx, tbl_op);
        emit_expr(c, e->u.call.args[is_lut ? 0 : 1]);
        if ((is_lut && nargs != 4) || (!is_lut && !is_i2 && nargs < 3))
        {
            eb_printf(c->eb, is_lut ? "), 0.0, 1.0))" : "), 0.0))");
            return;
        }
        eb_printf(c->eb, "), (double)(");
        if (is_i2)
        {
            emit_expr(c, e->u.call.args[2]);
        }
        else
        {
            emit_scalar_arg(c, e->u.call.args[2], 0);
        }
        if (is_lut)
        {
            eb_printf(c->eb, "), (double)(");
            emit_scalar_arg(c, e->u.call.args[3], 0);
        }
        eb_printf(c->eb, ")))");
        return;
    }

    case MC_FN_STENCIL_LIB:
    {
        c->used_spatial    = 1;
        int         op_idx = resolve_im_arg_op(c, e->u.call.args[0], 0);
        const char *sfx =
            get_dt_info((op_idx >= 0) ? c->pass->ops[op_idx].dt : _DATATYPE_FLOAT)->sfx;
        int is_mf = (strncmp(fn->name, "medfilt", 7) == 0);
        int r     = (strcmp(fn->name, "medfilt3") == 0)   ? 1
                    : (strcmp(fn->name, "medfilt5") == 0) ? 2
                                                          : 3;
        eb_printf(c->eb, "((%s)mc_%s_%s(&ops[%d], x, y, z, %d))", c->wt,
                  is_mf ? "medfilt" : "minmax3", sfx, op_idx,
                  is_mf ? r : (strcmp(fn->name, "imax3") == 0));
        return;
    }

    default:
    {
        double exp_v = 0.0;
        if ((strcmp(fn->name, "abs") == 0 || strcmp(fn->name, "fabs") == 0) &&
            is_complex_dt(e->u.call.args[0]->res_dt))
        {
            c->used_complex = 1;
            cfn             = wd ? "mc_cabsd" : "mc_cabsf";
        }
        else if (strcmp(fn->name, "pow") == 0 && is_const_num(e->u.call.args[1], &exp_v) &&
                 (exp_v == 0.0 || exp_v == 1.0 || exp_v == 2.0 || exp_v == 3.0 || exp_v == -1.0 ||
                  exp_v == 0.5 || exp_v == -0.5))
        {
            if (exp_v == 0.0 || exp_v == 1.0)
            {
                (void) ((exp_v == 0.0 && (eb_printf(c->eb, "((%s)1)", c->wt), 1)) ||
                        (emit_expr(c, e->u.call.args[0]), 1));
                return;
            }
            nargs = 1;
            cfn   = (exp_v == 2.0)    ? (wd ? "mc_sqd" : "mc_sqf")
                    : (exp_v == 3.0)  ? (wd ? "mc_cbd" : "mc_cbf")
                    : (exp_v == -1.0) ? (wd ? "mc_rcpd" : "mc_rcpf")
                    : (exp_v == 0.5)  ? (wd ? "sqrt" : "sqrtf")
                                      : (wd ? "mc_rsqrtd" : "mc_rsqrtf");
        }
        break;
    }
    }
    eb_printf(c->eb, "%s(", cfn ? cfn : fn->name);
    for (int i = 0; i < nargs; i++)
    {
        (void) (i > 0 && (eb_printf(c->eb, ", "), 1));
        emit_expr(c, e->u.call.args[i]);
    }
    eb_printf(c->eb, ")");
}

static void emit_expr(emit_ctx_t *c, const mc_expr_t *e)
{
    if (e == NULL)
    {
        eb_printf(c->eb, "((%s)0)", c->wt);
        return;
    }
    if (e->kind == MC_EXPR_IDENT && e->u.ident.def_param_idx >= 0 && c->def_depth > 0)
    {
        const mc_expr_t *sub = c->def_args[c->def_depth - 1][e->u.ident.def_param_idx];
        c->def_depth--;
        emit_expr(c, sub);
        c->def_depth++;
        return;
    }
    for (int g = 0; c->pass->is_comp && e->kind == MC_EXPR_IDENT && c->pass->comp_expr != NULL &&
                    g < c->pass->comp_expr->u.comp.ngens;
         g++)
    {
        const char *cn[3] = { "x", "y", "z" };
        if (strcmp(e->u.ident.bare, c->pass->comp_expr->u.comp.gens[g].var) == 0)
        {
            eb_printf(c->eb, "(_g_start%d + (%s)%s * _g_step%d)", g, c->wt, cn[g], g);
            return;
        }
    }
    int eim_op = (e->res_kind == MC_RES_IMAGE) ? mc_find_im_op(c->pass, -1, 0, e, 0, 1) : -1;
    if (eim_op >= 0)
    {
        emit_im_read(c, eim_op);
        return;
    }
    int sop = (e->res_kind != MC_RES_IMAGE) ? mc_find_scalar_op(c->pass, e, NULL, -1) : -1;
    if (sop >= 0)
    {
        eb_printf(c->eb, "_sc%d", sop);
        return;
    }

    switch (e->kind)
    {
    case MC_EXPR_LONG:
        eb_printf(c->eb, "((%s)%ldLL)", c->wt, e->u.lval);
        return;
    case MC_EXPR_DOUBLE:
        eb_printf(c->eb, "((%s)%.17g)", c->wt, e->u.dval);
        return;
    case MC_EXPR_IDENT:
    case MC_EXPR_SLICE:
    {
        if (e->kind == MC_EXPR_IDENT && e->u.ident.id_kind == MC_ID_CONST)
        {
            eb_printf(c->eb, "((%s)%s)", c->wt,
                      (strcmp(e->u.ident.bare, "pi") == 0)    ? "3.14159265358979323846"
                      : (strcmp(e->u.ident.bare, "nan") == 0) ? "NAN"
                                                              : "INFINITY");
            return;
        }
        int is_sl    = (e->kind == MC_EXPR_SLICE);
        int sidx     = mc_find_slot_by_bare(c->prog, is_sl ? e->u.slice.name : e->u.ident.name,
                                            is_sl ? e->u.slice.bare : e->u.ident.bare);
        int prev_out = (!is_sl && sidx >= 0) ? find_earlier_out_in_pass(c, sidx) : -1;
        int op_idx =
            (prev_out < 0) ? mc_find_im_op(c->pass, sidx, is_sl, is_sl ? e : NULL, 0, 0) : -1;
        (void) ((prev_out >= 0 && (eb_printf(c->eb, "_v%d", prev_out), 1)) ||
                (op_idx >= 0 && (emit_im_read(c, op_idx), 1)) ||
                (eb_printf(c->eb, "((%s)0)", c->wt), 1));
        return;
    }

    case MC_EXPR_UNARY:
        if (e->u.unary.op == MC_TOK_PLUS)
        {
            emit_expr(c, e->u.unary.operand);
            return;
        }
        eb_printf(c->eb,
                  (e->u.unary.op == MC_TOK_MINUS) ? "(-("
                  : (e->u.unary.op == MC_TOK_NOT) ? "((%s)(("
                                                  : "((%s)mc_bnot((double)(",
                  c->wt);
        emit_expr(c, e->u.unary.operand);
        eb_printf(c->eb, (e->u.unary.op == MC_TOK_MINUS) ? "))"
                         : (e->u.unary.op == MC_TOK_NOT) ? ") == 0))"
                                                         : ")))");
        return;

    case MC_EXPR_BINARY:
    {
        mc_tok_type_t op   = e->u.binary.op;
        int           cplx = is_complex_dt(e->res_dt), wd = c->pass->work_double;
        c->used_complex |= cplx;
        char        pre[64], mid[64];
        const char *post = "))";
        if (cplx && (op == MC_TOK_MUL || op == MC_TOK_DIV))
        {
            snprintf(pre, sizeof(pre), "%s((%s)(",
                     (op == MC_TOK_MUL) ? (wd ? "mc_cmuld" : "mc_cmulf")
                                        : (wd ? "mc_cdivd" : "mc_cdivf"),
                     c->ct);
            snprintf(mid, sizeof(mid), "), (%s)(", c->ct);
        }
        else if (op == MC_TOK_PLUS || op == MC_TOK_MINUS || op == MC_TOK_MUL || op == MC_TOK_DIV)
        {
            const char *os = (op == MC_TOK_PLUS)    ? "+"
                             : (op == MC_TOK_MINUS) ? "-"
                             : (op == MC_TOK_MUL)   ? "*"
                                                    : "/";
            snprintf(pre, sizeof(pre), "((%s)(", cplx ? c->ct : c->wt);
            snprintf(mid, sizeof(mid), ") %s (%s)(", os, cplx ? c->ct : c->wt);
        }
        else if (op == MC_TOK_MOD)
        {
            snprintf(pre, sizeof(pre), "%s(", wd ? "fmod" : "fmodf");
            snprintf(mid, sizeof(mid), ", ");
            post = ")";
        }
        else if (mc_is_rel_tok(op) || op == MC_TOK_EQ || op == MC_TOK_NE || op == MC_TOK_AND ||
                 op == MC_TOK_OR)
        {
            int is_log = (op == MC_TOK_AND || op == MC_TOK_OR);
            snprintf(pre, sizeof(pre), is_log ? "((%s)(((" : "((%s)((", c->wt);
            snprintf(mid, sizeof(mid), is_log ? ") != 0) %s ((" : ") %s (",
                     is_log ? ((op == MC_TOK_AND) ? "&" : "|") : rel_op_str(op));
            post = is_log ? ") != 0)))" : ")))";
        }
        else
        {
            const char *bfn = (op == MC_TOK_BAND)   ? "mc_band"
                              : (op == MC_TOK_BOR)  ? "mc_bor"
                              : (op == MC_TOK_BXOR) ? "mc_bxor"
                              : (op == MC_TOK_SHL)  ? "mc_shl"
                                                    : "mc_shr";
            snprintf(pre, sizeof(pre), "((%s)%s((double)(", c->wt, bfn);
            snprintf(mid, sizeof(mid), "), (double)(");
            post = ")))";
        }
        eb_printf(c->eb, "%s", pre), emit_expr(c, e->u.binary.left);
        eb_printf(c->eb, "%s", mid), emit_expr(c, e->u.binary.right);
        eb_printf(c->eb, "%s", post);
        return;
    }

    case MC_EXPR_CHAIN:
    {
        int id = ++c->chain_seq;
        eb_printf(c->eb, "({ ");
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            eb_printf(c->eb, "%s _ch%d_%d = (", c->wt, id, i);
            emit_expr(c, e->u.chain.operands[i]);
            eb_printf(c->eb, "); ");
        }
        eb_printf(c->eb, "(%s)(", c->wt);
        for (int i = 0; i < e->u.chain.nops; i++)
        {
            (void) (i > 0 && (eb_printf(c->eb, " && "), 1));
            eb_printf(c->eb, "(_ch%d_%d %s _ch%d_%d)", id, i, rel_op_str(e->u.chain.ops[i]), id,
                      i + 1);
        }
        eb_printf(c->eb, "); })");
        return;
    }

    case MC_EXPR_TERNARY:
        emit_ternary(c, e->u.ternary.cond, e->u.ternary.t_expr, e->u.ternary.f_expr);
        return;

    case MC_EXPR_COMP:
        emit_expr(c, e->u.comp.body);
        return;

    case MC_EXPR_CALL:
        emit_call(c, e);
        return;

    default:
        eb_printf(c->eb, "((%s)0)", c->wt);
        return;
    }
}

static void format_dst_lval(const emit_ctx_t *c, const mc_pass_out_t *out, char *dst, size_t dsz)
{
    int fast =
        c->fast_x_interior || out->lhs_slice == NULL || out->lhs_slice->u.slice.is_mask_index;
    snprintf(dst, dsz,
             c->pass->is_1d_contig ? "_p%d[_i]"
             : fast                ? "_p%d[_ro%d + (int64_t)x]"
                                   : "_p%d[_ro%d + (int64_t)x * _st%d_0]",
             out->op_idx, out->op_idx, out->op_idx);
}

static void emit_out_write(emit_ctx_t *c, int out_idx, int phase)
{
    const mc_pass_out_t *out   = &c->pass->outs[out_idx];
    uint8_t              dt    = out->out_dt;
    const dt_info_t     *dti   = get_dt_info(dt);
    const char          *ctype = dti->ct, *st_m = dti->stm;
    const char          *vtype = is_complex_dt(dt) ? c->ct : c->wt;

    if (out->is_elided)
    {
        if (phase != 2)
        {
            eb_printf(c->eb, "        const %s _v%d = (", vtype, out_idx);
            emit_expr(c, out->rhs_expr);
            eb_printf(c->eb, ");\n");
        }
        return;
    }

    int is_masked = (out->lhs_slice != NULL && out->lhs_slice->u.slice.is_mask_index);
    if (phase != 2)
    {
        if (is_masked)
        {
            eb_printf(c->eb, "        if ((");
            emit_expr(c, out->lhs_slice->u.slice.axes[0].start);
            eb_printf(c->eb, ") != 0) {\n");
        }
        eb_printf(c->eb, "        %s _r%d = (", vtype, out_idx);
        emit_expr(c, out->rhs_expr);
        eb_printf(c->eb, ");\n");
        if (out->assign_op == MC_TOK_ASSIGN && out->lhs_slice == NULL)
        {
            eb_printf(c->eb,
                      st_m ? "        const %s _v%d = (%s)%s(_r%d); (void)_v%d;\n"
                           : "        const %s _v%d = (%s)(%s)_r%d; (void)_v%d;\n",
                      vtype, out_idx, vtype, st_m ? st_m : ctype, out_idx, out_idx);
        }
        if (phase == 1)
        {
            return;
        }
    }

    char dst_lval[256];
    format_dst_lval(c, out, dst_lval, sizeof(dst_lval));
    (void) (!c->fast_div && out->assign_op == MC_TOK_DIV_ASSIGN &&
            (eb_printf(c->eb, "        if (_r%d == 0) { ops[%d].scalar = 1.0; } else {\n", out_idx,
                       out->op_idx),
             1));

    if (out->assign_op == MC_TOK_ASSIGN)
    {
        eb_printf(c->eb, st_m ? "        %s = %s(_r%d);\n" : "        %s = (%s)_r%d;\n", dst_lval,
                  st_m ? st_m : ctype, out_idx);
    }
    else
    {
        if (is_complex_dt(dt) &&
            (out->assign_op == MC_TOK_MUL_ASSIGN || out->assign_op == MC_TOK_DIV_ASSIGN))
        {
            const char *cfn = (out->assign_op == MC_TOK_MUL_ASSIGN)
                                  ? (c->pass->work_double ? "mc_cmuld" : "mc_cmulf")
                                  : (c->pass->work_double ? "mc_cdivd" : "mc_cdivf");
            eb_printf(c->eb, "        %s _nv%d = %s((%s)%s, _r%d);\n", vtype, out_idx, cfn, vtype,
                      dst_lval, out_idx);
        }
        else
        {
            const char *aop = (out->assign_op == MC_TOK_ADD_ASSIGN)   ? "+"
                              : (out->assign_op == MC_TOK_SUB_ASSIGN) ? "-"
                              : (out->assign_op == MC_TOK_MUL_ASSIGN) ? "*"
                                                                      : "/";
            eb_printf(c->eb, "        %s _nv%d = (%s)%s %s _r%d;\n", vtype, out_idx, vtype,
                      dst_lval, aop, out_idx);
        }
        eb_printf(c->eb, st_m ? "        %s = %s(_nv%d);\n" : "        %s = (%s)_nv%d;\n", dst_lval,
                  st_m ? st_m : ctype, out_idx);
    }

    (void) (!c->fast_div && out->assign_op == MC_TOK_DIV_ASSIGN &&
            (eb_printf(c->eb, "        }\n"), 1));
    (void) (is_masked && (eb_printf(c->eb, "        }\n"), 1));
}

static void emit_all_outs(emit_ctx_t *c)
{
    int split = (c->pass->nouts > 1);
    for (int i = 0; split && i < c->pass->nouts; i++)
    {
        const mc_pass_out_t *o = &c->pass->outs[i];
        split &= (o->assign_op == MC_TOK_ASSIGN &&
                  !(o->lhs_slice != NULL && o->lhs_slice->u.slice.is_mask_index));
    }
    for (int ph = split ? 1 : 0; ph <= (split ? 2 : 0); ph++)
    {
        for (int i = 0; i < c->pass->nouts; i++)
        {
            c->cur_out_idx = i;
            emit_out_write(c, i, ph);
        }
    }
}

int mc_emit_pass(const mcalc_prog_t *prog, mc_pass_t *pass, char *err, size_t errsz)
{
    emit_buf_t eb = { 0 }, k_eb = { 0 };
    emit_ctx_t c = { 0 };
    c.prog = prog, c.pass = pass, c.eb = &k_eb, c.used_complex = pass->is_complex;
    c.wt = pass->work_double ? "double" : "float";
    c.ct = pass->work_double ? "double _Complex" : "float _Complex";

    eb_printf(&k_eb, "void mc_k(mc_operand_t * restrict ops, uint64_t y0, uint64_t y1) {\n");

    for (int i = 0; i < pass->nops; i++)
    {
        if (pass->ops[i].is_scalar)
        {
            eb_printf(&k_eb,
                      "    const %s _sc%d = (%s)ops[%d].scalar; "
                      "const int64_t _ish%d = (int64_t)llround(ops[%d].scalar); "
                      "(void)_sc%d; (void)_ish%d;\n",
                      c.wt, i, c.wt, i, i, i, i, i);
        }
        else
        {
            const char *ct = get_dt_info(pass->ops[i].dt)->ct,
                       *q  = pass->ops[i].is_out ? "" : "const ";
            eb_printf(&k_eb, "    %s%s * restrict _p%d = (%s%s *)ops[%d].base; (void)_p%d;\n", q,
                      ct, i, q, ct, i, i);
            (void) (!pass->is_1d_contig &&
                    (eb_printf(
                         &k_eb,
                         "    const int64_t _st%d_0 = ops[%d].st[0], _st%d_1 = ops[%d].st[1], "
                         "_st%d_2 = ops[%d].st[2];\n"
                         "    const uint32_t _vnx%d = ops[%d].vnx, _vny%d = ops[%d].vny, "
                         "_vnz%d = ops[%d].vnz;\n"
                         "    (void)_st%d_0; (void)_st%d_1; (void)_st%d_2; "
                         "(void)_vnx%d; (void)_vny%d; (void)_vnz%d;\n",
                         i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i),
                     1));
        }
    }

    for (int g = 0; pass->is_comp && pass->comp_expr != NULL && g < pass->comp_expr->u.comp.ngens;
         g++)
    {
        int ks = mc_find_scalar_op(pass, pass->comp_expr->u.comp.gens[g].start, NULL, g * 3 + 0);
        int kt =
            (pass->comp_expr->u.comp.gens[g].step != NULL)
                ? mc_find_scalar_op(pass, pass->comp_expr->u.comp.gens[g].step, NULL, g * 3 + 2)
                : -1;
        eb_printf(&k_eb,
                  (kt >= 0) ? "    const %s _g_start%d = _sc%d, _g_step%d = _sc%d;\n"
                            : "    const %s _g_start%d = _sc%d, _g_step%d = 1;\n",
                  c.wt, g, ks, g, kt);
    }

    int fast_div =
        (pass->is_1d_contig && !pass->is_complex && pass->nouts == 1 &&
         pass->outs[0].assign_op == MC_TOK_DIV_ASSIGN && pass->outs[0].lhs_slice == NULL);
    c.fast_div   = fast_div;
    int can_simd = !pass->is_complex;
    for (int i = 0; !fast_div && i < pass->nouts; i++)
    {
        can_simd &= (pass->outs[i].assign_op != MC_TOK_DIV_ASSIGN);
    }

    if (pass->is_1d_contig)
    {
        int mt = (can_simd && pass->nouts > 1);
        for (int i = 0; mt && i < pass->nouts; i++)
        {
            mt &= (!pass->outs[i].is_elided && pass->outs[i].stmt_idx == pass->outs[0].stmt_idx);
        }
        if (mt)
        {
            eb_printf(&k_eb, "    for (uint64_t _b = y0; _b < y1; _b += 1024) {\n"
                             "        const uint64_t _be = (_b + 1024 < y1) ? (_b + 1024) : y1;\n");
            for (int i = 0; i < pass->nouts; i++)
            {
                c.cur_out_idx = i;
                eb_printf(&k_eb, "        #pragma omp simd\n"
                                 "        for (uint64_t _i = _b; _i < _be; _i++) {\n");
                emit_out_write(&c, i, 0);
                eb_printf(&k_eb, "        }\n");
            }
            eb_printf(&k_eb, "    }\n}\n");
        }
        else
        {
            if (fast_div)
            {
                eb_printf(&k_eb, "    int _zdiv = 0;\n"
                                 "    #pragma omp simd reduction(|:_zdiv)\n"
                                 "    for (uint64_t _i = y0; _i < y1; _i++) {\n"
                                 "        _zdiv |= ((");
                emit_expr(&c, pass->outs[0].rhs_expr);
                eb_printf(&k_eb,
                          ") == 0);\n    }\n"
                          "    if (_zdiv) { ops[%d].scalar = 1.0; return; }\n",
                          pass->outs[0].op_idx);
            }
            (void) (can_simd && (eb_printf(&k_eb, "    #pragma omp simd\n"), 1));
            eb_printf(&k_eb, "    for (uint64_t _i = y0; _i < y1; _i++) {\n");
            emit_all_outs(&c);
            eb_printf(&k_eb, "    }\n}\n");
        }
    }
    else
    {
        int ref_op = pass->outs[pass->nouts - 1].op_idx;
        eb_printf(&k_eb,
                  "    const uint32_t _nx = _vnx%d, _ny = _vny%d;\n"
                  "    int _fast_x = %d; int64_t _x0 = 0, _x1 = (int64_t)_nx;\n",
                  ref_op, ref_op, can_simd);
        for (int i = 0; i < pass->nops; i++)
        {
            if (!pass->ops[i].is_scalar && !pass->ops[i].is_table_arg)
            {
                eb_printf(&k_eb,
                          pass->ops[i].is_x_bcast
                              ? "    _fast_x &= (_st%d_0 == 0);\n"
                              : "    _fast_x &= (_st%d_0 == 1 && _vnx%d == _nx);\n",
                          i, i);
            }
            else if (pass->ops[i].is_scalar && pass->ops[i].is_non_pointwise)
            {
                eb_printf(&k_eb,
                          "    { int64_t _a = _ish%d < 0 ? -_ish%d : _ish%d; "
                          "if (_a > _x0) _x0 = _a; "
                          "if ((int64_t)_nx - _a < _x1) _x1 = (int64_t)_nx - _a; }\n",
                          i, i, i);
            }
        }
        eb_printf(&k_eb,
                  "    if (!_fast_x || _x0 >= _x1) { _x0 = (int64_t)_nx; _x1 = (int64_t)_nx; }\n"
                  "    uint64_t z = _ny ? (y0 / _ny) : 0, y = _ny ? (y0 %% _ny) : 0;\n"
                  "    (void)z; (void)y;\n"
                  "    for (uint64_t _yz = y0; _yz < y1; _yz++) {\n"
                  "        const uint64_t _row_base = _yz * _nx;\n");
        for (int i = 0; i < pass->nops; i++)
        {
            (void) (!pass->ops[i].is_scalar && !pass->ops[i].is_table_arg &&
                    (eb_printf(&k_eb,
                               "        const int64_t _ro%d = (int64_t)y * _st%d_1 + "
                               "(int64_t)z * _st%d_2; (void)_ro%d;\n",
                               i, i, i, i),
                     1));
        }
        eb_printf(&k_eb, "        for (int _seg = 0; _seg < 2; _seg++) {\n"
                         "            const uint64_t _xb = _seg ? (uint64_t)_x1 : 0, "
                         "_xe = _seg ? _nx : (uint64_t)_x0;\n"
                         "            for (uint64_t x = _xb; x < _xe; x++) {\n"
                         "                const uint64_t _i = _row_base + x; (void)_i;\n");
        c.fast_x_interior = 0;
        emit_all_outs(&c);
        eb_printf(&k_eb, "            }\n        }\n");
        if (can_simd)
        {
            emit_buf_t pre_eb = { 0 }, body_eb = { 0 };
            c.pre_eb = &pre_eb, c.eb = &body_eb, c.fast_x_interior = 1;
            emit_all_outs(&c);
            c.pre_eb = NULL, c.eb = &k_eb;
            (void) (pre_eb.buf != NULL && (eb_printf(&k_eb, "%s", pre_eb.buf), 1));
            eb_printf(&k_eb,
                      "        #pragma omp simd\n"
                      "        for (int32_t x = (int32_t)_x0; x < (int32_t)_x1; x++) {\n"
                      "            const uint64_t _i = _row_base + (uint64_t)x; (void)_i;\n%s"
                      "        }\n",
                      body_eb.buf ? body_eb.buf : "");
            k_eb.oom |= pre_eb.oom | body_eb.oom;
            free(pre_eb.buf), free(body_eb.buf);
        }
        eb_printf(&k_eb, "        if (++y == _ny) { y = 0; z++; }\n    }\n}\n");
    }
    eb_printf(&eb, "%s%s%s%s", MC_PRELUDE_CORE_SRC, c.used_complex ? MC_PRELUDE_COMPLEX_SRC : "",
              (c.used_coord_shift || c.used_spatial) ? MC_PRELUDE_COORD_SHIFT_SRC : "",
              c.used_spatial ? MC_PRELUDE_SPATIAL_MACRO_SRC : "");
    for (int i = 0, seen = 0; c.used_spatial && i <= pass->nops; i++)
    {
        uint8_t dt = (i < pass->nops && !pass->ops[i].is_scalar && !is_complex_dt(pass->ops[i].dt))
                         ? pass->ops[i].dt
                         : _DATATYPE_FLOAT;
        if ((seen & (1 << (dt & 31))) == 0)
        {
            seen |= (1 << (dt & 31));
            const dt_info_t *d = get_dt_info(dt);
            eb_printf(&eb, "MC_DEF_SPATIAL(%s, %s)\n", d->sfx, d->ct);
        }
    }
    eb_printf(&eb, "\n%s", k_eb.buf ? k_eb.buf : "");
    eb.oom |= k_eb.oom;
    free(k_eb.buf);
    if (eb.oom)
    {
        free(eb.buf);
        (void) (err && errsz > 0 && snprintf(err, errsz, "out of memory"));
        return -1;
    }
    pass->c_src = eb.buf;
    return 0;
}
