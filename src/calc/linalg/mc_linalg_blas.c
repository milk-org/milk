// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_linalg_blas.c
 * @brief BLAS-level linear algebra, algebraic GEMM/GER lowering, and Einstein summation for
 *        `libmilkcalc` (see mc_linalg.h).
 *
 * The type check of the linear algebra, FFT and stacking calls promotes their output to float,
 * double or complex as the arguments require. An assignment of a scaled matrix or outer product,
 * optionally accumulated into its target (`C = alpha * (A @ B) + beta * C`, `C -= outer(x, y)`),
 * is matched on its AST and runs as one in-place GEMM, SYRK (A times its own transpose) or GER
 * call through the COREMOD_arith BLAS wrappers, without a temporary image; a top-level
 * transpose() or `.T` becomes the BLAS transpose flag. The explicit BLAS calls (scal, axpy, ger,
 * syrk, gemv, symv, gemm) update their destination in place. einsum() reads explicit
 * (`"ij,jk->ik"`) and implicit (`"ij,jk"`) subscripts on 1-D to 3-D tensors, sends 2-D products
 * and cube contractions over the first two axes to arith_image_gemm_im(), and runs any other
 * pattern as a strided loop.
 */

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

/* ============================================================================
 * Error Reporting and Call Type Resolution
 * ============================================================================ */

/**
 * blas_err - Format a diagnostic message into @err (if still empty) and return -1
 * @err:   Destination error buffer (may be NULL)
 * @errsz: Capacity of @err in bytes
 * @fmt:   printf-style format string
 *
 * Preserves the first error recorded during compilation or execution by only
 * writing into @err when `err[0] == '\0'`.
 *
 * Return: Always -1 to allow concise `return blas_err(...);` error paths
 */
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

/**
 * mc_multi_lhs_is_image - Check whether a specific LHS target of a multi-output call is an image
 * @rhs0:    Right-hand-side call expression of the multi-assignment statement
 * @lhs_idx: Zero-based index of the left-hand-side target being queried
 *
 * Queries the port descriptor table for registered `MILK_EXT_OP` extensions, or
 * checks built-in multi-output decompositions (`svd` and `eigh`, all of whose
 * outputs are images/vectors).
 *
 * Return: 1 if LHS target @lhs_idx receives an image result, 0 if scalar or invalid
 */
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
    return rhs0->u.call.fn != NULL &&
           (rhs0->u.call.fn->id == MC_ID_svd || rhs0->u.call.fn->id == MC_ID_eigh);
}

/**
 * mc_resolve_linalg_call - Validate arguments and infer output type for a `MC_FN_LINALG` call
 * @e:        Call expression node to validate and annotate in place
 * @nlhs:     Number of left-hand-side assignment targets receiving this call's output
 * @any_im:   Non-zero if any argument to the call is an image expression (unused)
 * @any_cplx: Non-zero if any argument has a complex datatype
 * @any_dbl:  Non-zero if any argument has double precision (`float64` or `complex_double`)
 * @err:      Error message buffer populated on validation failure
 * @errsz:    Size of @err in bytes
 *
 * Enforces the LHS target counts of `svd` (1 or 3) and `eigh` (1 or 2); the argument kinds are
 * checked by the resolver from the catalog (mc_func_t.args). Annotates @e with `MC_RES_IMAGE`
 * and the promoted ImageStreamIO datatype.
 *
 * Return: 0 on success, -1 on validation error (with @err populated)
 */
int mc_resolve_linalg_call(mc_expr_t *e,
                           int        nlhs,
                           int        any_im,
                           int        any_cplx,
                           int        any_dbl,
                           char      *err,
                           size_t     errsz)
{
    const mc_fn_id_t fn    = e->u.call.fn->id;
    const char      *fname = e->u.call.fn->name;
    int              nargs = e->u.call.nargs;
    (void) any_im;

    /* Validate multi-LHS arity for matrix decompositions */
    if ((fn == MC_ID_svd && nlhs != 1 && nlhs != 3) || (fn == MC_ID_eigh && nlhs != 1 && nlhs != 2))
    {
        return blas_err(err, errsz, "%s: wrong number of LHS targets", fname);
    }

    /* Mark result as an image and infer output element datatype and static shape */
    e->needs_image = 1;
    e->res_kind    = MC_RES_IMAGE;
    if (fn == MC_ID_zeros)
    {
        int all_long = 1;
        for (int i = 0; i < nargs; i++)
        {
            all_long &= (e->u.call.args[i]->kind == MC_EXPR_LONG && e->u.call.args[i]->u.lval > 0);
        }
        if (all_long && nargs >= 1 && nargs <= 3)
        {
            e->res_naxis   = (uint8_t) nargs;
            e->res_size[0] = (uint32_t) e->u.call.args[0]->u.lval;
            e->res_size[1] = (nargs >= 2) ? (uint32_t) e->u.call.args[1]->u.lval : 1;
            e->res_size[2] = (nargs >= 3) ? (uint32_t) e->u.call.args[2]->u.lval : 1;
        }
    }
    else if (fn == MC_ID_eye && e->u.call.args[0]->kind == MC_EXPR_LONG &&
             e->u.call.args[0]->u.lval > 0 &&
             (nargs == 1 ||
              (e->u.call.args[1]->kind == MC_EXPR_LONG && e->u.call.args[1]->u.lval > 0)))
    {
        e->res_naxis   = 2;
        e->res_size[0] = (uint32_t) e->u.call.args[0]->u.lval;
        e->res_size[1] = (nargs == 2) ? (uint32_t) e->u.call.args[1]->u.lval : e->res_size[0];
        e->res_size[2] = 1;
    }
    else if (fn == MC_ID_vec)
    {
        e->res_naxis   = 1;
        e->res_size[0] = (uint32_t) nargs;
        e->res_size[1] = 1;
        e->res_size[2] = 1;
    }
    else if (fn == MC_ID_reshape && nargs >= 2 && nargs <= 4)
    {
        int all_long = 1;
        for (int i = 1; i < nargs; i++)
        {
            all_long &= (e->u.call.args[i]->kind == MC_EXPR_LONG && e->u.call.args[i]->u.lval > 0);
        }
        if (all_long)
        {
            e->res_naxis   = (uint8_t) (nargs - 1);
            e->res_size[0] = (uint32_t) e->u.call.args[1]->u.lval;
            e->res_size[1] = (nargs >= 3) ? (uint32_t) e->u.call.args[2]->u.lval : 1;
            e->res_size[2] = (nargs >= 4) ? (uint32_t) e->u.call.args[3]->u.lval : 1;
        }
    }
    else if (fn == MC_ID_transpose && e->u.call.args[0]->res_naxis >= 1)
    {
        e->res_naxis   = 2;
        e->res_size[0] = (e->u.call.args[0]->res_naxis >= 2) ? e->u.call.args[0]->res_size[1] : 1;
        e->res_size[1] = e->u.call.args[0]->res_size[0];
        e->res_size[2] = 1;
    }
    else if ((fn == MC_ID_fft || fn == MC_ID_ifft || fn == MC_ID_fftshift) &&
             e->u.call.args[0]->res_naxis > 0)
    {
        e->res_naxis = e->u.call.args[0]->res_naxis;
        memcpy(e->res_size, e->u.call.args[0]->res_size, sizeof(e->res_size));
    }
    if (fn == MC_ID_fft || fn == MC_ID_ifft)
    {
        e->res_dt = any_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT;
    }
    else if ((fn == MC_ID_fftshift || fn == MC_ID_reshape) && e->u.call.args[0]->res_dt != 0)
    {
        e->res_dt = e->u.call.args[0]->res_dt;
    }
    else if (fn == MC_ID_vec || fn == MC_ID_zeros || fn == MC_ID_eye)
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

/* ============================================================================
 * Algebraic BLAS Pattern Matching (GEMM / SYRK / GER Lowering)
 * ============================================================================ */

/**
 * unwrap_transpose - Follow user-function parameter bindings and strip an outer `transpose(X)`
 * @e:         Operand expression to inspect
 * @out_trans: Output flag set to 1 if an outer `transpose(...)` wrapper was stripped, else 0
 *
 * Allows BLAS routines that accept a transpose flag (`CblasTrans`) to consume
 * `transpose(A)` directly without materializing a transposed temporary image.
 *
 * Return: Underlying operand expression inside `transpose(...)`, or @e (after param resolution)
 */
static const mc_expr_t *unwrap_transpose(const mc_expr_t *e, int *out_trans)
{
    *out_trans = 0;
    if (e != NULL && e->kind == MC_EXPR_CALL && e->u.call.fn != NULL &&
        e->u.call.fn->id == MC_ID_transpose && e->u.call.nargs == 1)
    {
        *out_trans = 1;
        return e->u.call.args[0];
    }
    return e;
}

/**
 * is_scalar_subexpr - Test whether expression @e evaluates to a numeric scalar
 * @e: Expression node to test
 *
 * Return: 1 if @e is a non-NULL scalar expression, 0 if image, string, or NULL
 */
static int is_scalar_subexpr(const mc_expr_t *e)
{
    return e != NULL && !e->needs_image && e->res_kind != MC_RES_IMAGE &&
           e->res_kind != MC_RES_STRING;
}

/**
 * is_mm_or_outer_call - Test whether @e is a 2-argument `matmul(A, B)` or `outer(x, y)` call
 * @e:            Expression node to inspect
 * @out_call:     Output pointer receiving @e when matched
 * @out_is_outer: Output flag set to 1 for `outer`, 0 for `matmul`
 *
 * Return: 1 if @e is a `matmul` or `outer` call, 0 otherwise
 */
static int is_mm_or_outer_call(const mc_expr_t *e, const mc_expr_t **out_call, int *out_is_outer)
{
    if (e != NULL && e->kind == MC_EXPR_CALL && e->u.call.fn != NULL && e->u.call.nargs == 2 &&
        (e->u.call.fn->id == MC_ID_matmul || e->u.call.fn->id == MC_ID_outer))
    {
        *out_call     = e;
        *out_is_outer = (e->u.call.fn->id == MC_ID_outer);
        return 1;
    }
    return 0;
}

/**
 * match_scaled_mm - Match `[+|-] [alpha *] (matmul(A, B) | outer(x, y))`
 * @e:            Sub-expression to match
 * @out_call:     Output pointer to the matched `matmul` or `outer` call node
 * @out_is_outer: Output flag set to 1 if `outer`, 0 if `matmul`
 * @out_scale:    Output pointer to the scalar multiplier sub-expression (NULL if unscaled)
 * @out_sign:     Output sign multiplier (+1.0 or -1.0) accumulated from unary operators
 *
 * Recognizes both `alpha * matmul(A, B)` and `matmul(A, B) * alpha` with any
 * number of leading unary `+` or `-` signs.
 *
 * Return: 1 if @e matches a (possibly scaled/signed) `matmul` or `outer` term, 0 otherwise
 */
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
        *out_scale = NULL;
        *out_sign  = sgn;
        return 1;
    }
    if (e != NULL && e->kind == MC_EXPR_BINARY && e->u.binary.op == MC_TOK_MUL)
    {
        const mc_expr_t *l = e->u.binary.left;
        const mc_expr_t *r = e->u.binary.right;
        if (is_scalar_subexpr(l) && is_mm_or_outer_call(r, out_call, out_is_outer))
        {
            *out_scale = l;
            *out_sign  = sgn;
            return 1;
        }
        if (is_scalar_subexpr(r) && is_mm_or_outer_call(l, out_call, out_is_outer))
        {
            *out_scale = r;
            *out_sign  = sgn;
            return 1;
        }
    }
    return 0;
}

/**
 * match_scaled_ident - Match `[+|-] [beta *] <target_bare>` for in-place beta accumulation
 * @e:           Sub-expression to inspect
 * @target_bare: Bare identifier name of the LHS destination image `C`
 * @out_scale:   Output pointer to the scalar multiplier sub-expression `beta` (NULL if 1.0)
 * @out_sign:    Output sign multiplier (+1.0 or -1.0) from unary operators
 *
 * Return: 1 if @e is @target_bare optionally multiplied by a scalar expression, 0 otherwise
 */
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
        *out_scale = NULL;
        *out_sign  = sgn;
        return 1;
    }
    if (e != NULL && e->kind == MC_EXPR_BINARY && e->u.binary.op == MC_TOK_MUL)
    {
        const mc_expr_t *l = e->u.binary.left;
        const mc_expr_t *r = e->u.binary.right;
        if (is_scalar_subexpr(l) && r->kind == MC_EXPR_IDENT &&
            strcmp(r->u.ident.bare, target_bare) == 0)
        {
            *out_scale = l;
            *out_sign  = sgn;
            return 1;
        }
        if (is_scalar_subexpr(r) && l->kind == MC_EXPR_IDENT &&
            strcmp(l->u.ident.bare, target_bare) == 0)
        {
            *out_scale = r;
            *out_sign  = sgn;
            return 1;
        }
    }
    return 0;
}

/**
 * struct alg_blas_match_t - Decomposed parameters of an algebraic BLAS assignment
 * @call:       Matched `matmul(A, B)` or `outer(x, y)` call node
 * @is_outer:   1 if @call is `outer(x, y)` (GER), 0 if `matmul(A, B)` (GEMM/GEMV/SYRK)
 * @alpha_expr: Optional scalar expression multiplying the product term (NULL implies 1.0)
 * @alpha_sign: Sign factor (+1.0 or -1.0) applied to the evaluated @alpha_expr
 * @has_beta:   1 if the destination image `C` is accumulated into (`beta * C`), 0 otherwise
 * @beta_expr:  Optional scalar expression multiplying `C` (NULL implies 1.0 when @has_beta is 1)
 * @beta_sign:  Sign factor (+1.0 or -1.0) applied to the evaluated @beta_expr
 */
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

/**
 * match_alg_blas - Attempt to decompose an assignment into a single BLAS GEMM/SYRK/GER operation
 * @rhs:    Right-hand-side expression of the assignment
 * @target: Left-hand-side target expression (must be an identifier)
 * @aop:    Assignment operator token (`MC_TOK_ASSIGN`, `MC_TOK_ADD_ASSIGN`, `MC_TOK_SUB_ASSIGN`)
 * @m:      Output match descriptor populated on success
 *
 * Matches the following algebraic forms:
 * - `C += [alpha *] matmul(A, B)` or `C -= [alpha *] matmul(A, B)` (and `outer`)
 * - `C = alpha * matmul(A, B)` (non-trivial scaling or negation of `matmul` / `outer`)
 * - `C = [alpha *] matmul(A, B) +/- [beta *] C`
 * - `C = [beta *] C +/- [alpha *] matmul(A, B)`
 *
 * Return: 1 if the assignment matches an algebraic BLAS pattern, 0 otherwise
 */
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
        m->has_beta   = 1;
        m->beta_expr  = NULL;
        m->beta_sign  = 1.0;
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
        const mc_expr_t *l   = rhs->u.binary.left;
        const mc_expr_t *r   = rhs->u.binary.right;
        double           rop = (rhs->u.binary.op == MC_TOK_MINUS) ? -1.0 : 1.0;
        if (match_scaled_mm(l, &m->call, &m->is_outer, &m->alpha_expr, &m->alpha_sign) &&
            match_scaled_ident(r, target->u.ident.bare, &m->beta_expr, &m->beta_sign))
        {
            m->has_beta = 1;
            m->beta_sign *= rop;
            return 1;
        }
        if (match_scaled_ident(l, target->u.ident.bare, &m->beta_expr, &m->beta_sign) &&
            match_scaled_mm(r, &m->call, &m->is_outer, &m->alpha_expr, &m->alpha_sign))
        {
            m->has_beta = 1;
            m->alpha_sign *= rop;
            return 1;
        }
    }
    return 0;
}

/**
 * mc_is_blas_or_linalg_ctor - Test whether an assignment lowers to an algebraic BLAS step
 * @rhs:    Right-hand-side expression
 * @target: Left-hand-side target identifier expression
 * @aop:    Assignment token (`=`, `+=`, `-=`)
 *
 * Called by the planner (`plan/mc_plan.c`) to decide whether an assignment should be
 * scheduled as a `MC_STEP_CTOR` BLAS step rather than an element-wise fused kernel.
 *
 * Return: 1 if (@target, @aop, @rhs) matches an algebraic BLAS pattern, 0 otherwise
 */
int mc_is_blas_or_linalg_ctor(const mc_expr_t *rhs, const mc_expr_t *target, mc_tok_type_t aop)
{
    alg_blas_match_t m;
    return match_alg_blas(rhs, target, aop, &m);
}

/* ============================================================================
 * BLAS Execution Helpers
 * ============================================================================ */

/**
 * finish_inplace_blas - Post semaphore updates and bind or copy an in-place BLAS destination
 * @slot:    Destination program slot for the constructor step
 * @is_anon: Non-zero if the statement was called without an explicit LHS assignment
 * @dst_id:  Image ID of the in-place destination modified by the BLAS routine
 * @in_unit: Non-zero when executing inside a persistent FPS compute unit
 * @err:     Error buffer populated on failure
 * @errsz:   Capacity of @err in bytes
 *
 * Posts metadata/semaphores on @dst_id. If the statement also assigned the result
 * to a distinct LHS slot name (`D = gemm(alpha, A, B, beta, C)`), prepares `@slot`
 * and copies the updated payload of @dst_id into it.
 *
 * Return: 0 on success, -1 on output preparation failure
 */
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

/**
 * resolve_inplace_target - Resolve the destination image argument of an explicit in-place BLAS call
 * @arg:       AST argument node naming the destination image
 * @fname:     BLAS function name for error reporting
 * @out_id:    Output image ID of the resolved destination image
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Follows any user-function parameter bindings on @arg, verifies that it is an
 * image identifier, and resolves or autoconnects it in `milk_data.image` / `/dev/shm`.
 *
 * Return: 0 on success, -1 if @arg is not an existing image identifier
 */
static int resolve_inplace_target(const mc_expr_t *arg,
                                  const char      *fname,
                                  imageID         *out_id,
                                  char            *err,
                                  size_t           errsz)
{
    if (arg == NULL || arg->kind != MC_EXPR_IDENT)
    {
        return blas_err(err, errsz, "%s: destination argument must be an existing image", fname);
    }
    imageID id = mc_resolve_or_autoconnect_image(arg->u.ident.name);
    if (id < 0)
    {
        return blas_err(err, errsz, "%s: unknown destination image '%s'", fname, arg->u.ident.name);
    }
    *out_id = id;
    return 0;
}

/**
 * exec_matmul_or_alg - Execute a `matmul(A, B)` call or lowered algebraic GEMM/SYRK/GER step
 * @slot:      Destination slot to write or update
 * @is_anon:   Non-zero if the step target is anonymous (unused; slot is always updated)
 * @m:         Matched algebraic BLAS descriptor (`call`, `alpha`, `beta`, `is_outer`)
 * @avoid_id:  Image ID to avoid clobbering when evaluating sub-expression temporaries
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Evaluates scalar multipliers `alpha` and `beta`, unwraps `transpose(...)` on
 * operands, resolves or evaluates `A` and `B`, validates dimensions, and dispatches:
 * - `arith_image_ger_im()` for `outer(x, y)`
 * - `arith_image_syrk_im()` when `A` and `B` are the same image with opposite transposes
 * - `arith_image_gemm_im()` for general matrix-matrix and matrix-vector products
 *
 * Return: 0 on success, -1 on dimension/type mismatch or BLAS error
 */
static int exec_matmul_or_alg(mc_slot_t              *slot,
                              int                     is_anon,
                              const alg_blas_match_t *m,
                              imageID                 avoid_id,
                              int                     in_unit,
                              char                   *err,
                              size_t                  errsz)
{
    double alpha = 1.0, beta = 1.0;
    if ((m->alpha_expr != NULL && mc_eval_scalar_d(m->alpha_expr, &alpha, err, errsz) != 0) ||
        (m->has_beta && m->beta_expr != NULL &&
         mc_eval_scalar_d(m->beta_expr, &beta, err, errsz) != 0))
    {
        return -1;
    }
    alpha *= m->alpha_sign;
    beta = m->has_beta ? (beta * m->beta_sign) : 0.0;

    int              transA = 0, transB = 0;
    const mc_expr_t *eA =
        m->is_outer ? m->call->u.call.args[0] : unwrap_transpose(m->call->u.call.args[0], &transA);
    const mc_expr_t *eB =
        m->is_outer ? m->call->u.call.args[1] : unwrap_transpose(m->call->u.call.args[1], &transB);
    imageID idA = -1, idB = -1;
    int     tmpA = 0, tmpB = 0, rc = 0;
    if (mc_resolve_or_eval_im(eA, avoid_id, in_unit, &idA, &tmpA, err, errsz) != 0 ||
        mc_resolve_or_eval_im(eB, avoid_id, in_unit, &idB, &tmpB, err, errsz) != 0)
    {
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        return -1;
    }
    const IMAGE *imA = &milk_data.image[idA], *imB = &milk_data.image[idB];
    uint8_t      dtA = imA->md[0].datatype, dtB = imB->md[0].datatype;
    if (dtA == _DATATYPE_COMPLEX_FLOAT || dtA == _DATATYPE_COMPLEX_DOUBLE ||
        dtB == _DATATYPE_COMPLEX_FLOAT || dtB == _DATATYPE_COMPLEX_DOUBLE)
    {
        rc = blas_err(err, errsz, "%s: complex images are not supported",
                      m->is_outer ? "outer" : "matmul");
        goto done;
    }
    uint8_t pref_dt =
        (dtA == _DATATYPE_DOUBLE || dtB == _DATATYPE_DOUBLE || dtA == _DATATYPE_INT64 ||
         dtA == _DATATYPE_UINT64 || dtB == _DATATYPE_INT64 || dtB == _DATATYPE_UINT64)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;

    /* Compute output shape and validate inner contraction dimensions */
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
        out_sz[0] = (uint32_t) M;
        out_sz[1] = (uint32_t) N;
    }
    else if (arith_image_matmul_dims(imA, transA, imB, transB, &transA, &transB, &M, &K, &N,
                                     &out_naxis, out_sz) != 0)
    {
        rc = blas_err(err, errsz, "matmul: incompatible dimensions");
        goto done;
    }

    /* Resolve existing destination when accumulating (has_beta), else prepare output slot */
    imageID oid = -1;
    if (m->has_beta)
    {
        oid = (slot->im_id >= 0 && milk_data.image[slot->im_id].used &&
               strcmp(milk_data.image[slot->im_id].name, slot->bare) == 0)
                  ? slot->im_id
                  : mc_resolve_or_autoconnect_image(slot->name);
        if (oid < 0 ||
            (milk_data.image[oid].md[0].datatype != _DATATYPE_FLOAT &&
             milk_data.image[oid].md[0].datatype != _DATATYPE_DOUBLE) ||
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
        uint8_t out_dt = mc_select_ctor_out_dt(slot, pref_dt);
        if ((out_dt != _DATATYPE_FLOAT && out_dt != _DATATYPE_DOUBLE) ||
            (oid = mc_out_prepare(slot, (uint8_t) out_naxis, out_sz, out_dt, in_unit, err, errsz)) <
                0)
        {
            rc = (out_dt != _DATATYPE_FLOAT && out_dt != _DATATYPE_DOUBLE)
                     ? blas_err(err, errsz, "%s: destination stream datatype mismatch",
                                m->is_outer ? "outer" : "matmul")
                     : -1;
            goto done;
        }
    }

    /* Dispatch to GER, SYRK (when A*A^T or A^T*A), or general GEMM */
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
    if (tmpA)
    {
        mc_tmp_release(idA);
    }
    if (tmpB)
    {
        mc_tmp_release(idB);
    }
    return rc;
}

/* ============================================================================
 * Einstein Summation (einsum)
 * ============================================================================ */

/**
 * exec_einsum - Execute a 1- or 2-operand `einsum("subscripts", A[, B])` tensor contraction
 * @slot:      Destination slot to receive the contracted output tensor
 * @rhs:       `einsum` call expression node
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Parses the Einstein summation specification (`"ij,jk->ik"` or implicit `"ij,jk"`),
 * validates axis ranks (1..3) and index dimensions, and checks for BLAS GEMM fast paths:
 * - 2D x 2D -> 2D matrix multiplication (with any transpose combination on A and B)
 * - 3D cube contracted over its first two spatial axes with a 2D slice (`"ijk,ij->k"`),
 *   a 1D mode vector (`"ijk,k->ij"`), or another 3D cube (`"ijk,ijl->kl"`) by reshaping
 *   the `(nx * ny)` spatial plane into a flat column-major matrix dimension.
 *
 * When no GEMM fast path applies (e.g. trace `"ii->"`, diagonal extraction `"ii->i"`,
 * axis permutations, or general contractions), executes a stride-driven nested loop.
 *
 * Return: 0 on success, -1 on syntax/dimension error
 */
static int exec_einsum(mc_slot_t       *slot,
                       const mc_expr_t *rhs,
                       imageID          avoid_id,
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
    if (mc_resolve_or_eval_im(rhs->u.call.args[1], avoid_id, in_unit, &idA, &tmpA, err, errsz) !=
            0 ||
        (nin == 2 && mc_resolve_or_eval_im(rhs->u.call.args[2], avoid_id, in_unit, &idB, &tmpB, err,
                                           errsz) != 0))
    {
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        return -1;
    }
    const IMAGE *imA = &milk_data.image[idA];
    const IMAGE *imB = (nin == 2) ? &milk_data.image[idB] : NULL;

    /* Strip whitespace from the subscript string and split around "->" if present */
    char   sub[128] = { 0 };
    size_t slen     = 0;
    for (const char *p = sv.sval; *p != '\0' && slen + 1 < sizeof(sub); p++)
    {
        if (!isspace((unsigned char) *p))
        {
            sub[slen++] = *p;
        }
    }
    char *arrow     = strstr(sub, "->");
    char  lhs_s[64] = { 0 };
    char  rhs_s[16] = { 0 };
    if (arrow != NULL)
    {
        size_t ll = (size_t) (arrow - sub);
        if (ll >= sizeof(lhs_s) || strlen(arrow + 2) >= sizeof(rhs_s))
        {
            rc = blas_err(err, errsz, "einsum: subscript string too long");
            goto done;
        }
        memcpy(lhs_s, sub, ll);
        strcpy(rhs_s, arrow + 2);
    }
    else
    {
        snprintf(lhs_s, sizeof(lhs_s), "%s", sub);
    }

    /* Split LHS subscript specification by ',' into per-operand index strings */
    char *comma        = strchr(lhs_s, ',');
    char  in_sub[2][8] = { { 0 }, { 0 } };
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

    /* Validate operand ranks and record dimension size and occurrence count per index label */
    const IMAGE *ims[2]      = { imA, imB };
    uint32_t     dim_of[256] = { 0 };
    int          cnt_of[256] = { 0 };
    int          nidx        = 0;
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
            if (cnt_of[c] == 0)
            {
                dim_of[c]        = im->md[0].size[d];
                idx_list[nidx++] = (char) c;
            }
            cnt_of[c]++;
        }
    }

    /* In implicit mode (no "->"), output indices are non-repeated labels in alphabetical order */
    if (arrow == NULL)
    {
        int out_n = 0;
        for (int c = 'a'; c <= 'z'; c++)
        {
            if (cnt_of[c] == 1)
            {
                rhs_s[out_n++] = (char) c;
            }
        }
        for (int c = 'A'; c <= 'Z'; c++)
        {
            if (cnt_of[c] == 1)
            {
                rhs_s[out_n++] = (char) c;
            }
        }
        rhs_s[out_n] = '\0';
    }

    /* Partition indices into output axes (out_sz) and contracted summation axes (sum_idx) */
    size_t   out_nd      = strlen(rhs_s);
    int      in_out[256] = { 0 };
    int      nsum        = 0;
    uint32_t out_sz[3]   = { 1, 1, 1 };
    char     sum_idx[8]  = { 0 };
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
        in_out[c] = 1;
        out_sz[d] = dim_of[c];
    }
    for (int i = 0; i < nidx; i++)
    {
        unsigned char c = (unsigned char) idx_list[i];
        if (!in_out[c])
        {
            sum_idx[nsum++] = (char) c;
        }
    }

    uint8_t dtA    = imA->md[0].datatype;
    uint8_t dtB    = imB ? imB->md[0].datatype : dtA;
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

    /* Fast path 1: 2D matrix multiplication ("ij,jk->ik" and transposed variants) */
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

    /* Fast path 2: 3D cube contractions over flattened spatial planes (nx * ny) via GEMM */
    if (nin == 2 && strlen(in_sub[0]) == 3 && in_sub[0][0] != in_sub[0][1] &&
        in_sub[0][0] != in_sub[0][2] && in_sub[0][1] != in_sub[0][2] &&
        (dtA == _DATATYPE_FLOAT || dtA == _DATATYPE_DOUBLE) && dtA == dtB && out_dt == dtA)
    {
        IMAGE          vA = *imA, vB = *imB, vO = *out;
        IMAGE_METADATA mdA = imA->md[0], mdB = imB->md[0], mdO = out->md[0];
        vA.md        = &mdA;
        vB.md        = &mdB;
        vO.md        = &mdO;
        uint32_t nxy = imA->md[0].size[0] * imA->md[0].size[1];
        mdA.naxis    = 2;
        mdA.size[0]  = nxy;
        mdA.size[1]  = imA->md[0].size[2];
        int ok       = 0;
        if (strlen(in_sub[1]) == 2 && out_nd == 1 && in_sub[0][0] == in_sub[1][0] &&
            in_sub[0][1] == in_sub[1][1] && in_sub[0][2] == rhs_s[0])
        {
            mdB.naxis   = 1;
            mdB.size[0] = nxy;
            mdB.size[1] = 1;
            ok          = (arith_image_gemm_im(&vA, 1, &vB, 0, 1.0, 0.0, &vO) == 0);
        }
        else if (strlen(in_sub[1]) == 1 && out_nd == 2 && in_sub[0][0] == rhs_s[0] &&
                 in_sub[0][1] == rhs_s[1] && in_sub[0][2] == in_sub[1][0])
        {
            mdO.naxis   = 1;
            mdO.size[0] = nxy;
            mdO.size[1] = 1;
            ok          = (arith_image_gemm_im(&vA, 0, &vB, 0, 1.0, 0.0, &vO) == 0);
        }
        else if (strlen(in_sub[1]) == 3 && out_nd == 2 && in_sub[0][0] == in_sub[1][0] &&
                 in_sub[0][1] == in_sub[1][1] && in_sub[0][2] == rhs_s[0] &&
                 in_sub[1][2] == rhs_s[1])
        {
            mdB.naxis   = 2;
            mdB.size[0] = nxy;
            mdB.size[1] = imB->md[0].size[2];
            ok          = (arith_image_gemm_im(&vA, 1, &vB, 0, 1.0, 0.0, &vO) == 0);
        }
        if (ok)
        {
            mc_out_post(oid);
            goto done;
        }
    }

    /* General fallback: stride-driven contraction loop over output and summed coordinates */
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
    size_t   ndA        = strlen(in_sub[0]);
    size_t   ndB        = (nin == 2) ? strlen(in_sub[1]) : 0;
    for (uint32_t o2 = 0; o2 < out_sz[2]; o2++)
    {
        if (out_nd > 2)
        {
            coord[(unsigned char) rhs_s[2]] = o2;
        }
        for (uint32_t o1 = 0; o1 < out_sz[1]; o1++)
        {
            if (out_nd > 1)
            {
                coord[(unsigned char) rhs_s[1]] = o1;
            }
            for (uint32_t o0 = 0; o0 < out_sz[0]; o0++)
            {
                if (out_nd > 0)
                {
                    coord[(unsigned char) rhs_s[0]] = o0;
                }
                double acc = 0.0;
                for (uint64_t s2 = 0; s2 < sum_dims[2]; s2++)
                {
                    if (nsum > 2)
                    {
                        coord[(unsigned char) sum_idx[2]] = (uint32_t) s2;
                    }
                    for (uint64_t s1 = 0; s1 < sum_dims[1]; s1++)
                    {
                        if (nsum > 1)
                        {
                            coord[(unsigned char) sum_idx[1]] = (uint32_t) s1;
                        }
                        for (uint64_t s0 = 0; s0 < sum_dims[0]; s0++)
                        {
                            if (nsum > 0)
                            {
                                coord[(unsigned char) sum_idx[0]] = (uint32_t) s0;
                            }
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
    if (tmpA)
    {
        mc_tmp_release(idA);
    }
    if (tmpB)
    {
        mc_tmp_release(idB);
    }
    return rc;
}

/* ============================================================================
 * Top-Level BLAS Constructor Dispatcher
 * ============================================================================ */

/**
 * mc_exec_blas_call - Execute a BLAS or `einsum` constructor step in a compiled `milk.calc` plan
 * @prog:      Compiled calculator program holding the slot table
 * @cs:        Constructor step descriptor to execute
 * @avoid_id:  Image ID to avoid clobbering when evaluating temporary sub-expressions
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Dispatches:
 * - Algebraic BLAS expressions and `matmul(A, B)` to `exec_matmul_or_alg()`
 * - `einsum(...)` to `exec_einsum()`
 * - Explicit in-place Level-1/2/3 BLAS primitives (`scal`, `axpy`, `ger`, `syrk`,
 *   `gemv`, `symv`, `gemm`) to the corresponding `COREMOD_arith` functions.
 *
 * Return: 0 on success, -1 on evaluation or BLAS execution failure
 */
int mc_exec_blas_call(mcalc_prog_t   *prog,
                      mc_ctor_step_t *cs,
                      imageID         avoid_id,
                      int             in_unit,
                      char           *err,
                      size_t          errsz)
{
    mc_slot_t       *slot = &prog->slots[cs->slot_idx];
    const mc_expr_t *rhs  = cs->rhs_expr;
    alg_blas_match_t m;

    if (rhs->kind != MC_EXPR_CALL || (rhs->u.call.fn != NULL && rhs->u.call.fn->id == MC_ID_matmul))
    {
        /* `x += matmul(A, B)` / `x -= ...` accumulate into x: match_alg_blas() sets beta */
        if (rhs->kind == MC_EXPR_CALL && cs->assign_op != MC_TOK_ADD_ASSIGN &&
            cs->assign_op != MC_TOK_SUB_ASSIGN)
        {
            memset(&m, 0, sizeof(m));
            m.call       = rhs;
            m.alpha_sign = 1.0;
        }
        else if (!match_alg_blas(rhs, cs->target_expr, cs->assign_op, &m))
        {
            return blas_err(err, errsz, "invalid algebraic BLAS expression");
        }
        return exec_matmul_or_alg(slot, cs->is_anon, &m, avoid_id, in_unit, err, errsz);
    }

    const mc_fn_id_t fn    = rhs->u.call.fn->id;
    const char      *fname = rhs->u.call.fn->name;
    if (fn == MC_ID_einsum)
    {
        return exec_einsum(slot, rhs, avoid_id, in_unit, err, errsz);
    }

    double  alpha = 0.0, beta = 0.0;
    imageID idA = -1, idB = -1, idC = -1;
    int     tmpA = 0, tmpB = 0, rc = 0;

    if (mc_eval_scalar_d(rhs->u.call.args[0], &alpha, err, errsz) != 0)
    {
        return -1;
    }

    if (fn == MC_ID_scal)
    {
        if (resolve_inplace_target(rhs->u.call.args[1], fname, &idC, err, errsz) != 0)
        {
            return -1;
        }
        milk_data.image[idC].md[0].write = 1;
        if (arith_image_scal_im(alpha, &milk_data.image[idC]) == 0)
        {
            return finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz);
        }
        milk_data.image[idC].md[0].write = 0;
        return blas_err(err, errsz, "scal: expected float or double image");
    }

    if (fn == MC_ID_axpy)
    {
        if (resolve_inplace_target(rhs->u.call.args[2], fname, &idC, err, errsz) != 0 ||
            mc_resolve_or_eval_im(rhs->u.call.args[1], idC, in_unit, &idA, &tmpA, err, errsz) != 0)
        {
            return -1;
        }
        milk_data.image[idC].md[0].write = 1;
        rc = arith_image_axpy_im(alpha, &milk_data.image[idA], &milk_data.image[idC]);
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        if (rc != 0)
        {
            milk_data.image[idC].md[0].write = 0;
        }
        return (rc == 0) ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                         : blas_err(err, errsz,
                                    (rc == -2) ? "axpy: element count mismatch"
                                               : "axpy: expected matching float or double images");
    }

    if (fn == MC_ID_ger)
    {
        if (resolve_inplace_target(rhs->u.call.args[3], fname, &idC, err, errsz) != 0 ||
            mc_resolve_or_eval_im(rhs->u.call.args[1], idC, in_unit, &idA, &tmpA, err, errsz) !=
                0 ||
            mc_resolve_or_eval_im(rhs->u.call.args[2], idC, in_unit, &idB, &tmpB, err, errsz) != 0)
        {
            if (tmpA)
            {
                mc_tmp_release(idA);
            }
            return -1;
        }
        milk_data.image[idC].md[0].write = 1;
        int ok = (arith_image_ger_check(&milk_data.image[idA], &milk_data.image[idB],
                                        &milk_data.image[idC]) == 0 &&
                  arith_image_ger_im(&milk_data.image[idA], &milk_data.image[idB], alpha, 1.0,
                                     &milk_data.image[idC]) == 0);
        if (!ok)
        {
            milk_data.image[idC].md[0].write = 0;
        }
        rc = ok ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                : blas_err(err, errsz, "ger: dimension or datatype mismatch");
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        if (tmpB)
        {
            mc_tmp_release(idB);
        }
        return rc;
    }

    if (fn == MC_ID_syrk)
    {
        int              transA = 0;
        const mc_expr_t *eA     = unwrap_transpose(rhs->u.call.args[1], &transA);
        if (mc_eval_scalar_d(rhs->u.call.args[2], &beta, err, errsz) != 0 ||
            resolve_inplace_target(rhs->u.call.args[3], fname, &idC, err, errsz) != 0 ||
            mc_resolve_or_eval_im(eA, idC, in_unit, &idA, &tmpA, err, errsz) != 0)
        {
            return -1;
        }
        milk_data.image[idC].md[0].write = 1;
        int ok =
            (arith_image_syrk_check(&milk_data.image[idA], transA, &milk_data.image[idC]) == 0 &&
             arith_image_syrk_im(&milk_data.image[idA], transA, alpha, beta,
                                 &milk_data.image[idC]) == 0);
        if (!ok)
        {
            milk_data.image[idC].md[0].write = 0;
        }
        rc = ok ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                : blas_err(err, errsz, "syrk: dimension or datatype mismatch");
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        return rc;
    }

    if (fn == MC_ID_gemv || fn == MC_ID_symv || fn == MC_ID_gemm)
    {
        int              is_mm   = (fn == MC_ID_gemm);
        int              is_symv = (fn == MC_ID_symv);
        int              transA = 0, transB = 0;
        const mc_expr_t *eA =
            is_symv ? rhs->u.call.args[1] : unwrap_transpose(rhs->u.call.args[1], &transA);
        const mc_expr_t *eB =
            is_mm ? unwrap_transpose(rhs->u.call.args[2], &transB) : rhs->u.call.args[2];
        if (mc_eval_scalar_d(rhs->u.call.args[3], &beta, err, errsz) != 0 ||
            resolve_inplace_target(rhs->u.call.args[4], fname, &idC, err, errsz) != 0 ||
            mc_resolve_or_eval_im(eA, idC, in_unit, &idA, &tmpA, err, errsz) != 0 ||
            mc_resolve_or_eval_im(eB, idC, in_unit, &idB, &tmpB, err, errsz) != 0)
        {
            if (tmpA)
            {
                mc_tmp_release(idA);
            }
            return -1;
        }
        const IMAGE *imA = &milk_data.image[idA], *imB = &milk_data.image[idB];
        IMAGE       *imC = &milk_data.image[idC];
        imC->md[0].write = 1;
        int ok = is_symv ? (arith_image_symv_im(imA, imB, alpha, beta, imC) == 0)
                         : (arith_image_matmul_check(imA, transA, imB, transB, imC, 0) == 0 &&
                            arith_image_gemm_im(imA, transA, imB, transB, alpha, beta, imC) == 0);
        if (!ok)
        {
            imC->md[0].write = 0;
        }
        rc = ok ? finish_inplace_blas(slot, cs->is_anon, idC, in_unit, err, errsz)
                : blas_err(err, errsz, "%s: dimension or datatype mismatch", fname);
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        if (tmpB)
        {
            mc_tmp_release(idB);
        }
        return rc;
    }

    return blas_err(err, errsz, "unknown BLAS function '%s'", fname);
}
