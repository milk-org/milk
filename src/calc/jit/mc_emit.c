// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_emit.c
 * @brief C kernel source code generator for elementwise passes in libmilkcalc.
 *
 * Purpose:     run a fused statement, or several fused statements, as one compiled loop over
 *              the pixels that the C compiler vectorizes, with no temporary image.
 * Does:        mc_emit_pass() writes
 *              `void mc_k(mc_operand_t * restrict ops, uint64_t y0, uint64_t y1)` into
 *              pass->c_src, after only the prelude sections the kernel uses (less for the
 *              compiler to parse). Scalars, slice bounds and shift offsets come from ops[] at
 *              run time, so new values reuse the compiled kernel. A contiguous pass is one
 *              `#pragma omp simd` loop over [y0, y1) (several outputs of one statement are
 *              strip-mined by 1024, one vectorized loop per output); a sliced, broadcast or
 *              stencil pass loops over (y, z) rows, with scalar edge segments around a SIMD
 *              interior and hoisted row offsets. pow(x, c) for c in
 *              `{0, 1, 2, 3, -1, 0.5, -0.5}` becomes a product, root or reciprocal; real,
 *              imag and conj of a real input fold; with several `=` outputs, all values are
 *              computed before any is stored, so no output overwrites a pixel another one
 *              still reads.
 * Used by:     plan/mc_plan.c (mc_plan_build(), once per pass). Declared in plan/mc_plan.h.
 * Uses:        jit/mc_prelude.h (the prelude sections), plan/mc_views.c (slot and operand
 *              lookup).
 * See also:    jit/mc_cc.c (compiles the source); docs/guide/calculator.md,
 *              "Fusion & Cost Model".
 * Constraints: jit/mc_cc.c reads this text: it finds the entry point by `void mc_k(` (renamed
 *              per kernel in a batch file), hashes the core prelude once when the source
 *              starts with it, and finds the typed helpers a batch needs by the
 *              `MC_DEF_SPATIAL(sfx, type)` lines.
 */

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "plan/mc_plan.h"
#include "jit/mc_prelude.h"

/* ========================================================================== */
/* Dynamic String Buffer & Datatype Metadata Table                            */
/* ========================================================================== */

/**
 * struct emit_buf_t - Growable heap string buffer for C code generation
 * @buf: NUL-terminated character buffer (malloc'd)
 * @len: Current string length in bytes (excluding NUL)
 * @cap: Allocated capacity of @buf in bytes
 * @oom: Sticky out-of-memory flag (1 if any allocation failed)
 */
typedef struct
{
    char  *buf;
    size_t len;
    size_t cap;
    int    oom;
} emit_buf_t;

/**
 * eb_printf - Append formatted text to a growable emit buffer
 * @b:   Target emit buffer
 * @fmt: printf-style format string
 *
 * Doubles buffer capacity on demand (starting at 8 KiB). Sets `b->oom = 1` on
 * allocation or formatting failure so callers only need to check `oom` once at
 * the end of code generation.
 *
 * Return: void
 */
__attribute__((format(printf, 2, 3))) static void eb_printf(emit_buf_t *b, const char *fmt, ...)
{
    if (b->oom)
    {
        return;
    }
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
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
        if (nb != NULL)
        {
            b->buf = nb;
            b->cap = ncap;
            vsnprintf(b->buf + b->len, b->cap - b->len, fmt, ap2);
        }
        else
        {
            need = -1;
        }
    }
    va_end(ap2);
    if (need < 0)
    {
        b->oom = 1;
    }
    else
    {
        b->len += (size_t) need;
    }
}

/**
 * is_complex_dt - Check if an ImageStreamIO datatype is complex
 * @dt: Datatype code (`_DATATYPE_*`)
 *
 * Return: 1 if complex float or complex double, 0 otherwise
 */
static inline int is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}

/**
 * struct dt_info_t - C type and helper naming metadata for an ImageStreamIO datatype
 * @dt:  ImageStreamIO datatype code (`_DATATYPE_*`, or 0 for default float)
 * @ct:  C type specifier string (e.g. `"float"`, `"uint16_t"`, `"float _Complex"`)
 * @sfx: Short suffix for typed spatial helpers (e.g. `"f32"`, `"u16"`)
 * @stm: Saturating/rounding store helper name for integer types (e.g. `"mc_st_u16"`),
 *       or NULL for floating-point/complex types that use a direct cast
 */
typedef struct
{
    uint8_t     dt;
    const char *ct;
    const char *sfx;
    const char *stm;
} dt_info_t;

/**
 * get_dt_info - Look up C type and store helper metadata for a datatype
 * @dt: ImageStreamIO datatype code (`_DATATYPE_*`)
 *
 * Return: Pointer to a static `dt_info_t` entry (falls back to `float` if unknown)
 */
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

/* ========================================================================== */
/* Expression & Operand Code Emission                                         */
/* ========================================================================== */

/**
 * struct emit_ctx_t - Code generation state for a single elementwise pass
 * @prog:             Enclosing planned program
 * @pass:             Pass being emitted
 * @eb:               Active emit buffer receiving expression/statement code
 * @pre_eb:           Optional row-preamble buffer for hoisting `_sh` shift bases
 *                    before the inner SIMD `x` loop
 * @wt:               Real working C type (`"float"` or `"double"`)
 * @ct:               Complex working C type (`"float _Complex"` or `"double _Complex"`)
 * @sub_t0:           Reserved sub-expression pointer
 * @cur_out_idx:      Index in `pass->outs[]` currently being emitted
 * @chain_seq:        Monotonic counter for unique local variable names (`_ch`, `_sh`)
 * @fast_x_interior:  1 when emitting the unit-stride interior `x` SIMD loop
 * @fast_div:         1 when `/=` zero-divisor check was hoisted to a pre-loop reduction
 * @used_complex:     1 if the kernel requires `MC_PRELUDE_COMPLEX_SRC`
 * @used_coord_shift: 1 if the kernel requires `MC_PRELUDE_COORD_SHIFT_SRC`
 * @used_spatial:     1 if the kernel requires `MC_PRELUDE_SPATIAL_MACRO_SRC`
 */
typedef struct
{
    const mcalc_prog_t *prog;
    const mc_pass_t    *pass;
    emit_buf_t         *eb;
    emit_buf_t         *pre_eb;
    const char         *wt;
    const char         *ct;
    const mc_expr_t    *sub_t0;
    int                 cur_out_idx;
    int                 chain_seq;
    int                 fast_x_interior;
    int                 fast_div;
    int                 used_complex;
    int                 used_coord_shift;
    int                 used_spatial;
} emit_ctx_t;

/**
 * find_earlier_out_in_pass - Find a prior output in the same pass writing slot @sidx
 * @c:    Active emit context
 * @sidx: Target slot index in `prog->slots[]`
 *
 * When two fused statements in the same pass write to and then read from the
 * same slot (e.g. `local t = a + b; out = t * 2`), the later expression reads
 * the register variable `_v<out_idx>` computed earlier in the loop body rather
 * than reloading from memory.
 *
 * Return: Earlier output index in `c->pass->outs[]`, or -1 if none
 */
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

/**
 * emit_scalar_arg - Emit a scalar argument expression (using hoisted `_sc` / `_ish` if available)
 * @c:      Active emit context
 * @e:      Scalar argument expression
 * @is_int: 1 to emit an integer shift offset (`_ish<k>` or `llround(...)`), 0 for float/double
 *
 * Return: void
 */
static void emit_scalar_arg(emit_ctx_t *c, const mc_expr_t *e, int is_int)
{
    int k = mc_find_scalar_op(c->pass, e, NULL, -1);
    if (k >= 0)
    {
        eb_printf(c->eb, is_int ? "_ish%d" : "_sc%d", k);
        return;
    }
    if (is_int)
    {
        eb_printf(c->eb, "mc_ish((double)(");
    }
    emit_expr(c, e);
    if (is_int)
    {
        eb_printf(c->eb, "))");
    }
}

/**
 * emit_im_read - Emit an indexed pixel load for image operand @op_idx
 * @c:      Active emit context
 * @op_idx: Operand index in `c->pass->ops[]`
 *
 * Selects the optimal addressing mode for the active loop nest:
 * - `_p<k>[_i]` in 1D contiguous passes
 * - `_p<k>[_ro<k>]` for axis-0 broadcast operands in the fast interior `x` loop
 * - `_p<k>[_ro<k> + (int64_t)x]` for unit-stride operands in the fast interior `x` loop
 * - `_p<k>[_ro<k> + (int64_t)x * _st<k>_0]` in general strided/boundary loops
 *
 * Return: void
 */
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

/**
 * resolve_im_arg_op - Find the pass operand index corresponding to an image argument
 * @c:   Active emit context
 * @arg: Image argument AST node
 * @tbl: 1 if looking up a table/source argument (`is_table_arg`), 0 otherwise
 *
 * Return: Operand index in `c->pass->ops[]`, or -1 if not found
 */
static int resolve_im_arg_op(emit_ctx_t *c, const mc_expr_t *arg, int tbl)
{
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

/**
 * rel_op_str - Convert a relational operator token to its C operator string
 * @op: Token type (`MC_TOK_LT`, `MC_TOK_LE`, `MC_TOK_GT`, `MC_TOK_GE`, `MC_TOK_NE`, `MC_TOK_EQ`)
 *
 * Return: Static C operator string literal
 */
static const char *rel_op_str(mc_tok_type_t op)
{
    return (op == MC_TOK_LT)   ? "<"
           : (op == MC_TOK_LE) ? "<="
           : (op == MC_TOK_GT) ? ">"
           : (op == MC_TOK_GE) ? ">="
           : (op == MC_TOK_NE) ? "!="
                               : "==";
}

/**
 * emit_ternary - Emit a C conditional expression `(((cond) != 0) ? (t_e) : (f_e))`
 * @c:    Active emit context
 * @cond: Condition expression
 * @t_e:  True-branch expression
 * @f_e:  False-branch expression
 *
 * Return: void
 */
static void emit_ternary(emit_ctx_t      *c,
                         const mc_expr_t *cond,
                         const mc_expr_t *t_e,
                         const mc_expr_t *f_e)
{
    eb_printf(c->eb, "(((");
    emit_expr(c, cond);
    eb_printf(c->eb, ") != 0) ? (");
    emit_expr(c, t_e);
    eb_printf(c->eb, ") : (");
    emit_expr(c, f_e);
    eb_printf(c->eb, "))");
}

/**
 * is_const_num - Extract a compile-time numeric literal value (including unary minus)
 * @e:     Expression AST node to inspect
 * @out_v: Output double receiving the constant value when returning 1
 *
 * Used to recognize small constant exponents in `pow(x, c)` so they can be
 * replaced by inline multiplication, `sqrt`, or reciprocal helpers.
 *
 * Return: 1 if @e is a constant number, 0 otherwise
 */
static int is_const_num(const mc_expr_t *e, double *out_v)
{
    if (e->kind == MC_EXPR_LONG || e->kind == MC_EXPR_DOUBLE)
    {
        *out_v = (e->kind == MC_EXPR_LONG) ? (double) e->u.lval : e->u.dval;
        return 1;
    }
    if (e->kind == MC_EXPR_UNARY && e->u.unary.op == MC_TOK_MINUS &&
        is_const_num(e->u.unary.operand, out_v))
    {
        *out_v = -(*out_v);
        return 1;
    }
    return 0;
}

/**
 * emit_call - Emit C code for a function or macro call expression
 * @c: Active emit context
 * @e: `MC_EXPR_CALL` AST node
 *
 * Handles `where()`, complex
 * constructors/projections (`real`, `imag`, `conj`, `cplx`, `expi`, `arg`),
 * pixel coordinate generators (`ix`, `iy`, `ir`, `ia`), PRNG calls (`rand`,
 * `randn`), `shift`/`roll` (hoisting `_sh` row base offsets in the interior
 * SIMD loop), 1D/2D `lut`/`pixremap`/`interp2` table lookups, and algebraic
 * `pow(x, c)` specializations.
 *
 * Return: void
 */
static void emit_call(emit_ctx_t *c, const mc_expr_t *e)
{
    const mc_func_t *fn    = e->u.call.fn;
    int              nargs = e->u.call.nargs;
    int              wd    = c->pass->work_double;
    const char      *cfn   = wd ? fn->c_f64 : fn->c_f32;

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
            if (strcmp(fn->name, "imag") == 0)
            {
                eb_printf(c->eb, "((%s)0)", c->wt);
            }
            else
            {
                emit_expr(c, e->u.call.args[0]);
            }
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
        if (nargs == 3)
        {
            emit_scalar_arg(c, e->u.call.args[1], 0);
            eb_printf(c->eb, ", ");
            emit_scalar_arg(c, e->u.call.args[2], 0);
        }
        else
        {
            eb_printf(c->eb, "0, 0");
        }
        eb_printf(c->eb, ", %d, %d)", (nargs == 3), mode);
        return;
    }

    case MC_FN_PRNG:
        eb_printf(c->eb, "%s(_i, (double)_sc%d)",
                  (strcmp(fn->name, "randn") == 0) ? (wd ? "mc_grandd" : "mc_grandf")
                                                   : (wd ? "mc_urandd" : "mc_urandf"),
                  mc_find_scalar_op(c->pass, NULL, e, -1));
        return;

    case MC_FN_SHIFT:
    {
        c->used_coord_shift = 1;
        int         op_idx  = resolve_im_arg_op(c, e->u.call.args[0], 0);
        const char *scast = (op_idx >= 0 && is_complex_dt(c->pass->ops[op_idx].dt)) ? c->ct : c->wt;
        int         is_roll  = (strcmp(fn->name, "roll") == 0);
        int         sh_id    = ++c->chain_seq;
        emit_buf_t *saved_eb = c->eb;
        if (c->fast_x_interior && c->pre_eb != NULL)
        {
            c->eb = c->pre_eb;
            eb_printf(c->eb, "        const int64_t _sh%d = mc_shift_row(&ops[%d], y, z", sh_id,
                      op_idx);
        }
        else
        {
            eb_printf(c->eb, "((%s)_p%d[mc_shift_idx(&ops[%d], x, y, z", scast, op_idx, op_idx);
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
            eb_printf(c->eb, "((%s)_p%d[_sh%d + (int64_t)x])", scast, op_idx, sh_id);
        }
        return;
    }

    case MC_FN_LUT:
    case MC_FN_GATHER:
    {
        c->used_spatial    = 1;
        int         is_lut = (fn->kind == MC_FN_LUT);
        int         is_i2  = (strcmp(fn->name, "interp2") == 0);
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

    default:
    {
        double exp_v = 0.0;
        if (strcmp(fn->name, "abs") == 0 && is_complex_dt(e->u.call.args[0]->res_dt))
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
                if (exp_v == 0.0)
                {
                    eb_printf(c->eb, "((%s)1)", c->wt);
                }
                else
                {
                    emit_expr(c, e->u.call.args[0]);
                }
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
        if (i > 0)
        {
            eb_printf(c->eb, ", ");
        }
        emit_expr(c, e->u.call.args[i]);
    }
    eb_printf(c->eb, ")");
}

/* The literal c (0 <= c < 2^31) of `x & c` or `c & x`, or NULL */
static const mc_expr_t *band_mask(const mc_expr_t *e)
{
    const mc_expr_t *k = (e->u.binary.right->kind == MC_EXPR_LONG)  ? e->u.binary.right
                         : (e->u.binary.left->kind == MC_EXPR_LONG) ? e->u.binary.left
                                                                    : NULL;
    return (k != NULL && k->u.lval >= 0 && k->u.lval <= INT32_MAX) ? k : NULL;
}

/**
 * emit_expr - Recursively emit a C expression for an AST node inside a pass kernel
 * @c: Active emit context
 * @e: Expression AST node to emit
 *
 * Handles comprehension generator
 * coordinates (`_g_start + coord * _g_step`), pre-evaluated image and scalar
 * operands, constants (`pi`, `nan`, `inf`), earlier-in-pass register forwarding
 * (`_v<k>`), unary/binary/chained relational operators (using GCC statement
 * expressions `({ ... })` to evaluate middle operands of `a < b < c` once),
 * and function calls.
 *
 * Return: void
 */
static void emit_expr(emit_ctx_t *c, const mc_expr_t *e)
{
    if (e == NULL)
    {
        eb_printf(c->eb, "((%s)0)", c->wt);
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
        if (prev_out >= 0)
        {
            eb_printf(c->eb, "_v%d", prev_out);
        }
        else if (op_idx >= 0)
        {
            emit_im_read(c, op_idx);
        }
        else
        {
            eb_printf(c->eb, "((%s)0)", c->wt);
        }
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
        int           cplx = is_complex_dt(e->res_dt);
        int           wd   = c->pass->work_double;
        c->used_complex |= cplx;
        char        pre[64];
        char        mid[64];
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
            snprintf(pre, sizeof(pre), "%s(", wd ? "mc_fmodd" : "mc_fmodf");
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
        else if (op == MC_TOK_BAND && band_mask(e) != NULL)
        {
            /* x & c with 0 <= c < 2^31: 32-bit integers, which vectorize, give the same bits */
            const mc_expr_t *k = band_mask(e);
            eb_printf(c->eb, "((%s)mc_band32((", c->wt);
            emit_expr(c, (k == e->u.binary.right) ? e->u.binary.left : e->u.binary.right);
            eb_printf(c->eb, "), %ld))", k->u.lval);
            return;
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
        eb_printf(c->eb, "%s", pre);
        emit_expr(c, e->u.binary.left);
        eb_printf(c->eb, "%s", mid);
        emit_expr(c, e->u.binary.right);
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
            if (i > 0)
            {
                eb_printf(c->eb, " & ");
            }
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

/* ========================================================================== */
/* Pass Output Store & Loop Nest Generation                                   */
/* ========================================================================== */

/**
 * format_dst_lval - Format the C lvalue expression for an output pixel store
 * @c:   Active emit context
 * @out: Output descriptor in `c->pass->outs[]`
 * @dst: Destination buffer receiving the formatted lvalue
 * @dsz: Size of @dst in bytes
 *
 * Return: void
 */
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

/**
 * emit_out_write - Emit evaluation and/or pixel store statements for one pass output
 * @c:       Active emit context
 * @out_idx: Index in `c->pass->outs[]`
 * @phase:   0 = combined compute+store, 1 = compute into `_r<k>`/`_v<k>` only,
 *           2 = store previously computed `_r<k>` to destination memory
 *
 * Splitting multi-output passes into phase 1 (compute) and phase 2 (store)
 * ensures that all RHS reads from input arrays complete before any destination
 * array is overwritten at pixel `x`, avoiding false aliasing hazards.
 *
 * Return: void
 */
static void emit_out_write(emit_ctx_t *c, int out_idx, int phase)
{
    const mc_pass_out_t *out   = &c->pass->outs[out_idx];
    uint8_t              dt    = out->out_dt;
    const dt_info_t     *dti   = get_dt_info(dt);
    const char          *ctype = dti->ct;
    const char          *st_m  = dti->stm;
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
    if (!c->fast_div && out->assign_op == MC_TOK_DIV_ASSIGN)
    {
        eb_printf(c->eb, "        if (_r%d == 0) { ops[%d].scalar = 1.0; } else {\n", out_idx,
                  out->op_idx);
    }

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

    if (!c->fast_div && out->assign_op == MC_TOK_DIV_ASSIGN)
    {
        eb_printf(c->eb, "        }\n");
    }
    if (is_masked)
    {
        eb_printf(c->eb, "        }\n");
    }
}

/**
 * emit_all_outs - Emit all output computations and stores for the inner pixel loop body
 * @c: Active emit context
 *
 * Uses two-phase (compute all `_r<k>`, then store all `_r<k>`) emission when
 * multiple unmasked `=` outputs share a loop body.
 *
 * Return: void
 */
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

/**
 * mc_emit_pass - Generate the complete C kernel source string for an elementwise pass
 * @prog:  Enclosing planned program
 * @pass:  Pass whose `c_src` field will be populated with a heap-allocated C string
 * @err:   Error message buffer
 * @errsz: Size of @err in bytes
 *
 * Emits operand pointer/stride unpacking, comprehension generator steps, the
 * 1D contiguous or 2D/3D strided/interior-SIMD loop nest, and prepends only the
 * prelude sections actually referenced during expression emission.
 *
 * Return: 0 on success, -1 on out-of-memory
 */
int mc_emit_pass(const mcalc_prog_t *prog, mc_pass_t *pass, char *err, size_t errsz)
{
    emit_buf_t eb   = { 0 };
    emit_buf_t k_eb = { 0 };
    emit_ctx_t c    = { 0 };
    c.prog          = prog;
    c.pass          = pass;
    c.eb            = &k_eb;
    c.used_complex  = pass->is_complex;
    c.wt            = pass->work_double ? "double" : "float";
    c.ct            = pass->work_double ? "double _Complex" : "float _Complex";

    eb_printf(&k_eb, "void mc_k(mc_operand_t * restrict ops, uint64_t y0, uint64_t y1) {\n");

    /* Unpack scalar parameters and `restrict`-qualified typed data pointers */
    for (int i = 0; i < pass->nops; i++)
    {
        if (pass->ops[i].is_scalar)
        {
            eb_printf(&k_eb,
                      "    const %s _sc%d = (%s)ops[%d].scalar; "
                      "const int64_t _ish%d = mc_ish(ops[%d].scalar); "
                      "(void)_sc%d; (void)_ish%d;\n",
                      c.wt, i, c.wt, i, i, i, i, i);
        }
        else
        {
            const char *ct = get_dt_info(pass->ops[i].dt)->ct;
            const char *q  = pass->ops[i].is_out ? "" : "const ";
            eb_printf(&k_eb, "    %s%s * restrict _p%d = (%s%s *)ops[%d].base; (void)_p%d;\n", q,
                      ct, i, q, ct, i, i);
            if (!pass->is_1d_contig)
            {
                eb_printf(&k_eb,
                          "    const int64_t _st%d_0 = ops[%d].st[0], _st%d_1 = ops[%d].st[1], "
                          "_st%d_2 = ops[%d].st[2];\n"
                          "    const uint32_t _vnx%d = ops[%d].vnx, _vny%d = ops[%d].vny, "
                          "_vnz%d = ops[%d].vnz;\n"
                          "    (void)_st%d_0; (void)_st%d_1; (void)_st%d_2; "
                          "(void)_vnx%d; (void)_vny%d; (void)_vnz%d;\n",
                          i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i);
            }
        }
    }

    /* Unpack comprehension generator start and step values */
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
        /* Fast 1D contiguous loop (with 1024-element L1 strip-mining for multi-output) */
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
            if (can_simd)
            {
                eb_printf(&k_eb, "    #pragma omp simd\n");
            }
            eb_printf(&k_eb, "    for (uint64_t _i = y0; _i < y1; _i++) {\n");
            emit_all_outs(&c);
            eb_printf(&k_eb, "    }\n}\n");
        }
    }
    else
    {
        /* 2D/3D strided loop with boundary scalar segments and interior SIMD `x` loop */
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
            if (!pass->ops[i].is_scalar && !pass->ops[i].is_table_arg)
            {
                eb_printf(&k_eb,
                          "        const int64_t _ro%d = (int64_t)y * _st%d_1 + "
                          "(int64_t)z * _st%d_2; (void)_ro%d;\n",
                          i, i, i, i);
            }
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
            emit_buf_t pre_eb  = { 0 };
            emit_buf_t body_eb = { 0 };
            c.pre_eb           = &pre_eb;
            c.eb               = &body_eb;
            c.fast_x_interior  = 1;
            emit_all_outs(&c);
            c.pre_eb = NULL;
            c.eb     = &k_eb;
            if (pre_eb.buf != NULL)
            {
                eb_printf(&k_eb, "%s", pre_eb.buf);
            }
            eb_printf(&k_eb,
                      "        #pragma omp simd\n"
                      "        for (int32_t x = (int32_t)_x0; x < (int32_t)_x1; x++) {\n"
                      "            const uint64_t _i = _row_base + (uint64_t)x; (void)_i;\n%s"
                      "        }\n",
                      body_eb.buf ? body_eb.buf : "");
            k_eb.oom |= pre_eb.oom | body_eb.oom;
            free(pre_eb.buf);
            free(body_eb.buf);
        }
        eb_printf(&k_eb, "        if (++y == _ny) { y = 0; z++; }\n    }\n}\n");
    }

    /* Assemble final translation unit: required prelude sections + kernel function */
    eb_printf(&eb, "%s%s%s%s", MC_PRELUDE_CORE_SRC, c.used_complex ? MC_PRELUDE_COMPLEX_SRC : "",
              (c.used_coord_shift || c.used_spatial) ? MC_PRELUDE_COORD_SHIFT_SRC : "",
              c.used_spatial ? MC_PRELUDE_SPATIAL_MACRO_SRC : "");
    for (int i = 0, seen = 0; c.used_spatial && i <= pass->nops; i++)
    {
        uint8_t dt = (i < pass->nops && !pass->ops[i].is_scalar && pass->ops[i].dt != 0 &&
                      !is_complex_dt(pass->ops[i].dt))
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
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "out of memory");
        }
        return -1;
    }
    pass->c_src = eb.buf;
    return 0;
}
