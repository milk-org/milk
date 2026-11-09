// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_plan.c
 * @brief Execution planner and elementwise pass optimizer for libmilkcalc.
 *
 * Translates the resolved abstract syntax tree (`p->ast_head`) produced by the
 * parser and semantic analyzer into a linear sequence of execution steps
 * (`MC_STEP_SCALAR`, `MC_STEP_CTOR`, and `MC_STEP_PASS`):
 *
 * 1. **Slot Discovery & Type Inference**: Registers every referenced or created
 *    image in `p->slots[]`, tracking whether it existed at plan time, its
 *    storage location (`@S:` shared vs `@L:` local), and its planned datatype
 *    and dimensions.
 * 2. **Pass Construction & Pre-staging**: Walks image-producing expressions to
 *    collect input image views and hoisted scalar parameters (`mc_op_desc_t`).
 *    Non-elementwise constructors (`linspace`, BLAS/LAPACK calls, separable
 *    convolutions, axis reductions, `ema`) embedded inside larger expressions
 *    are automatically hoisted into preceding `MC_STEP_CTOR` pre-stage steps
 *    writing into `_tmpcalcpsN` slots.
 * 3. **Pass Fusion & Local Elision**: Consecutive elementwise passes with
 *    compatible loop geometry and no cyclic hazards are fused into a single
 *    multi-output pass (`merge_passes_and_elide_locals`), and intermediate
 *    `local` images that are never read outside their defining pass are elided
 *    into register variables (`mc_elide_unused_locals`).
 * 4. **Code Generation & Compilation**: Invokes `mc_emit_pass()` for each
 *    finalized pass and `mc_cc_compile_prog()` to compile all C kernels.
 */

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libfps/fps_types.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "linalg/mc_linalg.h"
#include "plan/mc_plan.h"

/* ========================================================================== */
/* Error Reporting & Datatype Classification Helpers                          */
/* ========================================================================== */

/**
 * plan_err - Format the first planning error message and return -1
 * @err:   Destination error buffer (may be NULL)
 * @errsz: Size of @err in bytes
 * @fmt:   printf-style format string
 *
 * Preserves the earliest error message if @err is already non-empty so that
 * cascading failures do not overwrite the root cause.
 *
 * Return: Always -1
 */
__attribute__((format(printf, 3, 4))) static int plan_err(char       *err,
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
 * is_float_exact_dt - Check if a datatype is exactly representable in float32
 * @dt: ImageStreamIO datatype code (`_DATATYPE_*`)
 *
 * 8-bit and 16-bit integer types as well as float32 fit without precision loss
 * in IEEE-754 single precision (24-bit mantissa), allowing passes over these
 * types to run at `float` working precision instead of `double`.
 *
 * Return: 1 if representable in `float`, 0 otherwise
 */
static inline int is_float_exact_dt(uint8_t dt)
{
    return dt == _DATATYPE_FLOAT || dt == _DATATYPE_UINT8 || dt == _DATATYPE_INT8 ||
           dt == _DATATYPE_UINT16 || dt == _DATATYPE_INT16;
}

/**
 * is_complex_dt - Check if a datatype is complex float or complex double
 * @dt: ImageStreamIO datatype code (`_DATATYPE_*`)
 *
 * Return: 1 if complex, 0 otherwise
 */
static inline int is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}

/**
 * mark_assigned_var - Record that a bound scalar variable is modified by the program
 * @p:    Program being planned
 * @bare: Bare identifier name of the variable
 *
 * Marks `p->assigned[i] = 1` so `mc_prog_store_vars()` knows to write the
 * updated value back to the caller's bound variable pointer after execution.
 *
 * Return: void
 */
static void mark_assigned_var(mcalc_prog_t *p, const char *bare)
{
    for (int i = 0; p != NULL && bare != NULL && i < p->nvars; i++)
    {
        if (strcmp(p->vars[i].name, bare) == 0)
        {
            p->assigned[i] = 1;
        }
    }
}

/* ========================================================================== */
/* Expression & Built-In Function Classification                              */
/* ========================================================================== */

/**
 * is_ctor_fn - Check if a built-in function is a standalone image constructor
 * @fn: Built-in function descriptor (may be NULL)
 *
 * Return: 1 if @fn is a `MC_FN_LINSPACE` or `MC_FN_LINALG` constructor, 0 otherwise
 */
static int is_ctor_fn(const mc_func_t *fn)
{
    return fn != NULL && (fn->kind == MC_FN_LINSPACE || fn->kind == MC_FN_LINALG);
}

/**
 * is_prestage_stencil_fn - Check if a stencil function requires a pre-stage step
 * @fn: Built-in function descriptor (may be NULL)
 *
 * Library stencils (`medfilt*`, `imin3`, `imax3`, `boxmean`, `conv2`, `gauss`, `cumsum`,
 * `pixunmap`, `rprof`) run as whole-image pre-stages (mc_exec_prestage()), mostly COREMOD_arith
 * kernels; `shift` and `roll` stay in the fused pass.
 *
 * Return: 1 if @fn is a library stencil, 0 otherwise
 */
static int is_prestage_stencil_fn(const mc_func_t *fn)
{
    return fn != NULL && fn->kind == MC_FN_STENCIL_LIB;
}

/**
 * is_ctor_call - Determine whether an expression must execute as a constructor step
 * @e: Expression AST node to inspect
 *
 * Identifies vector literals, `linspace`/`zeros`/`eye`, linear algebra and shape
 * operators, pre-staged stencil filters, registered C extensions, image-returning
 * axis reductions (`isum(im, axis)`, etc.), and stateful `ema()` calls.
 *
 * Return: 1 if @e requires a `MC_STEP_CTOR` execution step, 0 otherwise
 */
static int is_ctor_call(const mc_expr_t *e)
{
    const mc_func_t *fn = (e != NULL && e->kind == MC_EXPR_CALL) ? e->u.call.fn : NULL;
    return (e != NULL && e->kind == MC_EXPR_VEC_LIT) || is_ctor_fn(fn) ||
           is_prestage_stencil_fn(fn) ||
           (e != NULL && e->kind == MC_EXPR_CALL && e->u.call.ext_id >= 0) ||
           (fn != NULL && e->res_kind == MC_RES_IMAGE &&
            (fn->kind == MC_FN_REDUCE_D ||
             (fn->kind == MC_FN_PROG && strcmp(fn->name, "ema") == 0)));
}

/**
 * check_expr_supported - Verify that all calls in an expression tree are supported
 * @e:     Expression AST node to validate
 * @err:   Error message buffer
 * @errsz: Size of @err in bytes
 *
 * Recursively traverses @e and rejects unresolved function calls that belong
 * neither to the built-in table (`fn != NULL`) nor to the C extension registry
 * (`ext_id >= 0`).
 *
 * Return: 0 if supported, -1 on error (with @err populated)
 */
static int check_expr_supported(const mc_expr_t *e, char *err, size_t errsz)
{
    int rc =
        (e == NULL || e->kind <= MC_EXPR_END) ? 0
        : (e->kind == MC_EXPR_UNARY)          ? check_expr_supported(e->u.unary.operand, err, errsz)
        : (e->kind == MC_EXPR_BINARY)  ? (check_expr_supported(e->u.binary.left, err, errsz) ||
                                          check_expr_supported(e->u.binary.right, err, errsz))
        : (e->kind == MC_EXPR_TERNARY) ? (check_expr_supported(e->u.ternary.cond, err, errsz) ||
                                          check_expr_supported(e->u.ternary.t_expr, err, errsz) ||
                                          check_expr_supported(e->u.ternary.f_expr, err, errsz))
        : (e->kind == MC_EXPR_ASSIGN)  ? (check_expr_supported(e->u.assign.target, err, errsz) ||
                                          check_expr_supported(e->u.assign.rhs, err, errsz))
        : (e->kind == MC_EXPR_COMP)    ? check_expr_supported(e->u.comp.body, err, errsz)
                                       : 0;
    for (int i = 0; e && e->kind == MC_EXPR_CHAIN && i <= e->u.chain.nops && !rc; i++)
    {
        rc = check_expr_supported(e->u.chain.operands[i], err, errsz);
    }
    for (int a = 0; e && e->kind == MC_EXPR_SLICE && a < e->u.slice.naxis && !rc; a++)
    {
        rc = check_expr_supported(e->u.slice.axes[a].start, err, errsz) ||
             check_expr_supported(e->u.slice.axes[a].stop, err, errsz) ||
             check_expr_supported(e->u.slice.axes[a].step, err, errsz);
    }
    for (int i = 0; e && e->kind == MC_EXPR_VEC_LIT && i < e->u.vec_lit.nelem && !rc; i++)
    {
        rc = check_expr_supported(e->u.vec_lit.elems[i], err, errsz);
    }
    for (int g = 0; e && e->kind == MC_EXPR_COMP && g < e->u.comp.ngens && !rc; g++)
    {
        rc = check_expr_supported(e->u.comp.gens[g].start, err, errsz) ||
             check_expr_supported(e->u.comp.gens[g].stop, err, errsz) ||
             check_expr_supported(e->u.comp.gens[g].step, err, errsz);
    }
    for (int i = 0; e && e->kind == MC_EXPR_CALL && i < e->u.call.nargs && !rc; i++)
    {
        rc = check_expr_supported(e->u.call.args[i], err, errsz);
    }
    if (!e || e->kind != MC_EXPR_CALL || rc != 0)
    {
        return rc;
    }
    const mc_fn_kind_t fk =
        (e->u.call.ext_id < 0 && e->u.call.fn) ? e->u.call.fn->kind : (mc_fn_kind_t) -1;
    return (e->u.call.ext_id >= 0 || fk >= 0)
               ? 0
               : plan_err(err, errsz, "'%s' cannot run in a compiled program", e->u.call.name);
}

/* ========================================================================== */
/* Program Slot & Pass Operand Table Management                               */
/* ========================================================================== */

/**
 * find_or_add_slot - Look up or register a named image slot in a planned program
 * @prog:     Program whose slot table is queried/extended
 * @name:     Qualified image name (e.g. `@S:im1` or `im1`)
 * @bare:     Bare image name without location prefix
 * @loc:      Location qualifier character (`'S'`, `'L'`, or `0`)
 * @is_local: 1 if declared with the `local` keyword
 * @im_id:    Known `imageID` at plan time, or -1 to probe/autoconnect
 * @dt:       Planned datatype (`_DATATYPE_*`), or 0 if unknown
 * @naxis:    Planned number of axes, or 0 if unknown
 * @sz:       Planned axis sizes `[3]`, or NULL if unknown
 *
 * Deduplicates slots by @bare name. If the slot already exists, updates its
 * planned datatype/shape when newly inferred by an assignment step.
 *
 * Return: Non-negative index into `prog->slots[]`
 */
static int find_or_add_slot(mcalc_prog_t  *prog,
                            const char    *name,
                            const char    *bare,
                            int            loc,
                            int            is_local,
                            imageID        im_id,
                            uint8_t        dt,
                            uint8_t        naxis,
                            const uint32_t sz[3])
{
    const char *key = (bare != NULL && bare[0] != '\0') ? bare : name;
    for (int i = 0; i < prog->nslots; i++)
    {
        if (strcmp(prog->slots[i].bare, key) == 0)
        {
            if (is_local && !prog->slots[i].existed_at_plan)
            {
                prog->slots[i].is_local = 1;
            }
            if (dt != 0)
            {
                prog->slots[i].plan_dt = dt;
            }
            if (naxis > 0 && sz != NULL && sz[0] > 0)
            {
                prog->slots[i].plan_naxis = naxis;
                memcpy(prog->slots[i].plan_size, sz, 3 * sizeof(uint32_t));
            }
            return i;
        }
    }
    if (prog->nslots >= prog->cap_slots)
    {
        prog->cap_slots = (prog->cap_slots == 0) ? 16 : prog->cap_slots * 2;
        prog->slots     = realloc(prog->slots, (size_t) prog->cap_slots * sizeof(*prog->slots));
    }
    mc_slot_t *s = &prog->slots[prog->nslots];
    memset(s, 0, sizeof(*s));
    snprintf(s->name, sizeof(s->name), "%s", name != NULL ? name : key);
    snprintf(s->bare, sizeof(s->bare), "%s", key);
    s->loc      = (loc == 'S') ? 1 : (loc == 'L') ? 2 : 0;
    s->is_local = is_local;
    if (im_id < 0 && key[0] != '\0' && strcmp(key, "_tmpcalc0") != 0)
    {
        im_id = mc_resolve_or_autoconnect_image(s->name);
    }
    s->im_id           = im_id;
    s->existed_at_plan = (im_id >= 0);
    const IMAGE *im    = (im_id >= 0) ? &milk_data.image[im_id] : NULL;
    s->orig_dt         = im ? im->md[0].datatype : (dt ? dt : _DATATYPE_FLOAT);
    s->plan_dt         = s->orig_dt;
    s->plan_naxis      = im ? im->md[0].naxis : naxis;
    s->plan_size[0]    = im ? im->md[0].size[0] : (sz ? sz[0] : 0);
    s->plan_size[1]    = im ? ((s->plan_naxis > 1) ? im->md[0].size[1] : 1) : (sz ? sz[1] : 0);
    s->plan_size[2]    = im ? ((s->plan_naxis > 2) ? im->md[0].size[2] : 1) : (sz ? sz[2] : 0);
    return prog->nslots++;
}

/**
 * add_pass_op - Add or deduplicate an operand descriptor in an elementwise pass
 * @pass: Target elementwise pass
 * @desc: Operand descriptor to insert
 *
 * Read-only image and scalar operands with identical slot/expression identity
 * share a single entry in `pass->ops[]`; if any reference is non-pointwise or
 * requires a division-by-zero check, those flags are merged onto the existing
 * operand descriptor. Output operands (`is_out == 1`) always receive a fresh
 * entry.
 *
 * Return: Index of the operand in `pass->ops[]`
 */
static int add_pass_op(mc_pass_t *pass, const mc_op_desc_t *desc)
{
    for (int i = 0; i < pass->nops; i++)
    {
        const mc_op_desc_t *o = &pass->ops[i];
        if (!desc->is_out && !o->is_out && desc->is_scalar == o->is_scalar &&
            desc->slot_idx == o->slot_idx && desc->has_slice == o->has_slice &&
            desc->slice_expr == o->slice_expr && desc->scalar_expr == o->scalar_expr &&
            desc->is_table_arg == o->is_table_arg && desc->is_expr_im == o->is_expr_im &&
            desc->comp_gen_idx == o->comp_gen_idx)
        {
            pass->ops[i].is_non_pointwise |= desc->is_non_pointwise;
            if (desc->check_div_kind && pass->ops[i].check_div_kind == 0)
            {
                pass->ops[i].check_div_kind = desc->check_div_kind;
            }
            return i;
        }
    }
    if (pass->nops >= pass->cap_ops)
    {
        pass->cap_ops = (pass->cap_ops == 0) ? 16 : pass->cap_ops * 2;
        pass->ops     = realloc(pass->ops, (size_t) pass->cap_ops * sizeof(*pass->ops));
    }
    pass->ops[pass->nops] = *desc;
    return pass->nops++;
}

/**
 * add_scalar_op - Register a hoisted scalar sub-expression on an elementwise pass
 * @pass:   Target elementwise pass
 * @se:     Scalar AST sub-expression evaluated once prior to kernel launch
 * @div_c:  Non-zero check kind (1 = division, 2 = modulo, 3 = lut dx, 0 = none)
 * @non_pt: 1 if this scalar is an integer shift offset for `shift`/`roll`
 * @cg:     Comprehension generator bound index (`g * 3 + {0,1,2}`), or -1
 *
 * Return: 0 on success, -1 on failure
 */
static int add_scalar_op(mc_pass_t *pass, const mc_expr_t *se, int div_c, int non_pt, int cg)
{
    mc_op_desc_t d     = { 0 };
    d.slot_idx         = -1;
    d.is_scalar        = 1;
    d.is_non_pointwise = (uint8_t) non_pt;
    d.scalar_expr      = se;
    d.check_div_kind   = div_c;
    d.comp_gen_idx     = cg;
    return (add_pass_op(pass, &d) < 0) ? -1 : 0;
}

/* ========================================================================== */
/* Pass Expression Tree Walker                                                */
/* ========================================================================== */

/**
 * struct plan_walk_ctx_t - State carried while walking an expression tree for a pass
 * @prog:       Enclosing program being planned
 * @pass:       Pointer to the active pass (updated if a pre-stage step reallocs `steps`)
 * @err:        Error message buffer
 * @errsz:      Size of @err in bytes
 * @comp_gens:  Active array comprehension generators (or NULL)
 * @ncomp_gens: Number of active comprehension generators
 * @in_guarded: 1 when inside a short-circuit or ternary branch where zero divisors
 *              may be dynamically guarded and must not fail unconditionally before
 *              the kernel runs
 */
typedef struct
{
    mcalc_prog_t        *prog;
    mc_pass_t           *pass;
    char                *err;
    size_t               errsz;
    const mc_comp_gen_t *comp_gens;
    int                  ncomp_gens;
    int                  in_guarded;
} plan_walk_ctx_t;

/**
 * walk_pass_expr - Traverse an expression tree and populate pass operands
 * @w:         Walker context (program, pass, `def` stack, comprehension state)
 * @e:         Expression AST node to walk
 * @div_check: Non-zero check kind if @e is the divisor of `/`, `%=`, or `lut`
 * @non_pt:    1 if @e is accessed non-pointwise (e.g. `shift`, `lut` table, stencil)
 * @tbl:       1 if @e is a lookup table / source image excluded from pass shape
 *
 * Identifies scalar subtrees that do not depend on pixel coordinates and hoists
 * them as `is_scalar` operands evaluated once before the kernel runs. Hoists
 * embedded constructor calls (`is_ctor_call`) into preceding `MC_STEP_CTOR`
 * steps writing to `_tmpcalcpsN` slots. Marks non-pointwise compound image
 * arguments (`is_expr_im`) for runtime sub-program evaluation.
 *
 * Return: 0 on success, -1 on error
 */
static int walk_pass_expr(plan_walk_ctx_t *w,
                          const mc_expr_t *e,
                          int              div_check,
                          int              non_pt,
                          int              tbl)
{
    if (e == NULL)
    {
        return 0;
    }

    /* Comprehension generator variables (e.g. `i`, `j`) are synthesized in-kernel */
    for (int g = 0; e->kind == MC_EXPR_IDENT && g < w->ncomp_gens; g++)
    {
        if (strcmp(e->u.ident.bare, w->comp_gens[g].var) == 0)
        {
            return 0;
        }
    }

    /* Hoist pure scalar subtrees as pre-evaluated scalar pass parameters */
    if (e->res_kind != MC_RES_IMAGE)
    {
        return add_scalar_op(w->pass, e, w->in_guarded ? 0 : div_check, non_pt, -1);
    }

    /* Hoist embedded constructor/prestage calls into a preceding MC_STEP_CTOR step */
    if (is_ctor_call(e))
    {
        mcalc_prog_t *p = w->prog;
        char          ps_nm[32];
        snprintf(ps_nm, sizeof(ps_nm), "_tmpcalcps%d", p->nslots);
        int ps_sidx =
            find_or_add_slot(p, ps_nm, ps_nm, 'L', 1, -1, e->res_dt, e->res_naxis, e->res_size);
        if (p->nsteps >= p->cap_steps)
        {
            p->cap_steps = (p->cap_steps == 0) ? 8 : p->cap_steps * 2;
            p->steps     = realloc(p->steps, (size_t) p->cap_steps * sizeof(*p->steps));
        }
        int stmt_idx              = p->steps[p->nsteps - 1].first_stmt_idx;
        p->steps[p->nsteps]       = p->steps[p->nsteps - 1];
        p->stmt_to_step[stmt_idx] = p->nsteps;
        mc_step_t *cs             = &p->steps[p->nsteps - 1];
        memset(cs, 0, sizeof(*cs));
        cs->kind           = MC_STEP_CTOR;
        cs->first_stmt_idx = stmt_idx;
        cs->last_stmt_idx  = stmt_idx;
        cs->ctor.slot_idx  = ps_sidx;
        cs->ctor.is_anon   = 1;
        cs->ctor.assign_op = MC_TOK_ASSIGN;
        cs->ctor.rhs_expr  = e;
        p->nsteps++;
        w->pass = &p->steps[p->nsteps - 1].pass;
        w->pass->prestage_cnt++;
        mc_op_desc_t d = { .slot_idx         = ps_sidx,
                           .dt               = p->slots[ps_sidx].plan_dt,
                           .is_non_pointwise = non_pt,
                           .is_table_arg     = tbl,
                           .slice_expr       = e,
                           .comp_gen_idx     = -1 };
        return (add_pass_op(w->pass, &d) < 0) ? -1 : 0;
    }

    /* Compound image expression passed to a non-pointwise operator (e.g. shift(im + 1, 2)) */
    if (is_ctor_call(e) || (non_pt && e->kind != MC_EXPR_IDENT && e->kind != MC_EXPR_SLICE))
    {
        mc_op_desc_t d = { .slot_idx         = -1,
                           .dt               = e->res_dt ? e->res_dt : _DATATYPE_FLOAT,
                           .is_non_pointwise = non_pt,
                           .is_table_arg     = tbl,
                           .is_expr_im       = 1,
                           .slice_expr       = e,
                           .comp_gen_idx     = -1 };
        return (add_pass_op(w->pass, &d) < 0) ? -1 : 0;
    }

    int rc    = 0;
    int saved = w->in_guarded;
    switch (e->kind)
    {
    case MC_EXPR_IDENT:
    case MC_EXPR_SLICE:
    {
        int is_sl = (e->kind == MC_EXPR_SLICE);
        int sidx =
            is_sl ? find_or_add_slot(w->prog, e->u.slice.name, e->u.slice.bare, 0, 0,
                                     e->u.slice.im_id, e->res_dt, 0, NULL)
                  : find_or_add_slot(w->prog, e->u.ident.name, e->u.ident.bare, e->u.ident.loc, 0,
                                     e->u.ident.im_id, e->res_dt, e->res_naxis, e->res_size);
        mc_op_desc_t d = { .slot_idx         = sidx,
                           .dt               = (sidx >= 0) ? w->prog->slots[sidx].plan_dt : 0,
                           .is_non_pointwise = non_pt,
                           .is_table_arg     = tbl,
                           .has_slice        = is_sl,
                           .slice_expr       = is_sl ? e : NULL,
                           .comp_gen_idx     = -1 };
        return (sidx < 0 || add_pass_op(w->pass, &d) < 0) ? -1 : 0;
    }
    case MC_EXPR_UNARY:
        return walk_pass_expr(w, e->u.unary.operand, 0, 0, 0);
    case MC_EXPR_BINARY:
        rc = walk_pass_expr(w, e->u.binary.left, 0, 0, 0);
        w->in_guarded |= (e->u.binary.op == MC_TOK_AND || e->u.binary.op == MC_TOK_OR);
        rc            = rc ? rc
                           : walk_pass_expr(w, e->u.binary.right,
                                            (e->u.binary.op == MC_TOK_DIV)   ? 1
                                            : (e->u.binary.op == MC_TOK_MOD) ? 2
                                                                             : 0,
                                            0, 0);
        w->in_guarded = saved;
        return rc;
    case MC_EXPR_CHAIN:
        for (int i = 0; i <= e->u.chain.nops && rc == 0; i++)
        {
            rc = walk_pass_expr(w, e->u.chain.operands[i], 0, 0, 0);
        }
        return rc;
    case MC_EXPR_TERNARY:
        rc            = walk_pass_expr(w, e->u.ternary.cond, 0, 0, 0);
        w->in_guarded = 1;
        rc            = rc ? rc
                           : (walk_pass_expr(w, e->u.ternary.t_expr, 0, 0, 0) ||
                              walk_pass_expr(w, e->u.ternary.f_expr, 0, 0, 0));
        w->in_guarded = saved;
        return rc;
    case MC_EXPR_COMP:
        w->pass->is_comp                = 1;
        w->pass->uses_coords_or_stencil = 1;
        w->pass->comp_expr              = e;
        w->comp_gens                    = e->u.comp.gens;
        w->ncomp_gens                   = e->u.comp.ngens;
        for (int g = 0; g < e->u.comp.ngens && rc == 0; g++)
        {
            rc = (e->u.comp.gens[g].start &&
                  add_scalar_op(w->pass, e->u.comp.gens[g].start, 0, 0, g * 3 + 0)) ||
                 (e->u.comp.gens[g].stop &&
                  add_scalar_op(w->pass, e->u.comp.gens[g].stop, 0, 0, g * 3 + 1)) ||
                 (e->u.comp.gens[g].step &&
                  add_scalar_op(w->pass, e->u.comp.gens[g].step, 0, 0, g * 3 + 2));
        }
        return rc ? -1 : walk_pass_expr(w, e->u.comp.body, 0, 0, 0);
    case MC_EXPR_CALL:
        if (e->u.call.fn == NULL)
        {
            return -1;
        }
        switch (e->u.call.fn->kind)
        {
        case MC_FN_WHERE:
            rc            = walk_pass_expr(w, e->u.call.args[0], 0, 0, 0);
            w->in_guarded = 1;
            rc            = rc ? rc
                               : (walk_pass_expr(w, e->u.call.args[1], 0, 0, 0) ||
                                  walk_pass_expr(w, e->u.call.args[2], 0, 0, 0));
            w->in_guarded = saved;
            return rc;
        case MC_FN_PRNG:
        {
            w->pass->uses_coords_or_stencil = 1;
            mc_op_desc_t d = { .slot_idx     = -1,
                               .is_scalar    = 1,
                               .slice_expr   = e,
                               .comp_gen_idx = -1,
                               .scalar_expr  = (e->u.call.nargs >= 2) ? e->u.call.args[1] : NULL };
            return (walk_pass_expr(w, e->u.call.args[0], 0, 0, 0) || add_pass_op(w->pass, &d) < 0)
                       ? -1
                       : 0;
        }
        case MC_FN_SHIFT:
            if (e->u.call.args[0]->res_kind != MC_RES_IMAGE)
            {
                return plan_err(w->err, w->errsz, "%s: expected %s(im, dx[, dy])",
                                e->u.call.fn->name, e->u.call.fn->name);
            }
            w->pass->uses_coords_or_stencil = 1;
            rc                              = walk_pass_expr(w, e->u.call.args[0], 0, 1, 0);
            for (int i = 1; i < e->u.call.nargs && rc == 0; i++)
            {
                rc = (e->u.call.args[i]->res_kind == MC_RES_IMAGE)
                         ? plan_err(w->err, w->errsz, "%s: expected %s(im, dx[, dy])",
                                    e->u.call.fn->name, e->u.call.fn->name)
                         : walk_pass_expr(w, e->u.call.args[i], 0, (i == 1), 0);
            }
            return rc;
        case MC_FN_LUT:
            rc = walk_pass_expr(w, e->u.call.args[0], 0, 0, 0) ||
                 walk_pass_expr(w, e->u.call.args[1], 0, 1, 1);
            for (int i = 2; i < e->u.call.nargs && rc == 0; i++)
            {
                rc = walk_pass_expr(w, e->u.call.args[i], (i == 3) ? 3 : 0, 0, 0);
            }
            return rc;
        case MC_FN_GATHER:
            if (e->u.call.args[0]->res_kind != MC_RES_IMAGE ||
                e->u.call.args[1]->res_kind != MC_RES_IMAGE ||
                (e->u.call.nargs == 3 && (strcmp(e->u.call.fn->name, "interp2") == 0) !=
                                             (e->u.call.args[2]->res_kind == MC_RES_IMAGE)))
            {
                return plan_err(w->err, w->errsz,
                                (strcmp(e->u.call.fn->name, "interp2") == 0)
                                    ? "interp2: expected interp2(im, xmap, ymap)"
                                    : "pixremap: expected pixremap(im, map[, fill])");
            }
            rc = walk_pass_expr(w, e->u.call.args[0], 0, 1, 1) ||
                 walk_pass_expr(w, e->u.call.args[1], 0, 0, 0);
            return (rc == 0 && e->u.call.nargs == 3) ? walk_pass_expr(w, e->u.call.args[2], 0, 0, 0)
                                                     : rc;
        default:
            w->pass->uses_coords_or_stencil |= (e->u.call.fn->kind == MC_FN_COORD);
            for (int i = 0; i < e->u.call.nargs && rc == 0; i++)
            {
                rc = walk_pass_expr(w, e->u.call.args[i], 0, 0, 0);
            }
            return rc;
        }
    default:
        return -1;
    }
}

/* ========================================================================== */
/* Pass Output Registration & Metadata Finalization                           */
/* ========================================================================== */

/**
 * add_pass_out - Register an output assignment on an elementwise pass
 * @prog:      Program being planned
 * @pass:      Elementwise pass receiving the output
 * @stmt_idx:  Statement index in `prog`
 * @is_local:  1 if the target is declared `local`
 * @is_multi:  1 if part of a fused multi-assignment statement
 * @assign_op: Assignment operator token (`=`, `+=`, `-=`, `*=`, `/=`)
 * @target:    LHS target expression (`MC_EXPR_IDENT`, `MC_EXPR_SLICE`, or NULL)
 * @rhs:       RHS expression tree
 * @err:       Error message buffer
 * @errsz:     Size of @err in bytes
 *
 * Resolves the destination image slot, determines the output datatype
 * (preserving existing datatype for shared streams, slice/compound assignments,
 * or scalar-to-image fills; promoting to `double` when a `double` input is
 * present unless an explicit type-cast function wraps the RHS), walks @rhs
 * (and any mask expression on the LHS), and appends the `mc_pass_out_t` entry.
 *
 * Return: 0 on success, -1 on error
 */
static int add_pass_out(mcalc_prog_t    *prog,
                        mc_pass_t       *pass,
                        int              stmt_idx,
                        int              is_local,
                        int              is_multi,
                        mc_tok_type_t    assign_op,
                        const mc_expr_t *target,
                        const mc_expr_t *rhs,
                        char            *err,
                        size_t           errsz)
{
    const mc_expr_t *lhs_sl = (target != NULL && target->kind == MC_EXPR_SLICE) ? target : NULL;
    int              tid    = (target != NULL && target->kind == MC_EXPR_IDENT);
    const char *name = lhs_sl ? lhs_sl->u.slice.name : tid ? target->u.ident.name : "_tmpcalc0";
    const char *bare = lhs_sl ? lhs_sl->u.slice.bare : tid ? target->u.ident.bare : "_tmpcalc0";
    if (tid)
    {
        mark_assigned_var(prog, bare);
    }

    int sidx = find_or_add_slot(prog, name, bare, tid ? target->u.ident.loc : 'L', is_local,
                                lhs_sl ? lhs_sl->u.slice.im_id : (tid ? target->u.ident.im_id : -1),
                                0, 0, NULL);
    mc_slot_t *slot = (sidx >= 0) ? &prog->slots[sidx] : NULL;
    int        keep_t =
        slot && (lhs_sl != NULL || assign_op != MC_TOK_ASSIGN ||
                 (slot->existed_at_plan && milk_data.image[slot->im_id].md[0].shared == 1) ||
                 (rhs->res_kind != MC_RES_IMAGE && slot->existed_at_plan) ||
                 (is_multi && slot->existed_at_plan && slot->orig_dt == _DATATYPE_FLOAT &&
                  rhs->kind == MC_EXPR_IDENT && !is_complex_dt(rhs->res_dt) &&
                  rhs->res_dt != _DATATYPE_DOUBLE));
    if (sidx < 0 || (keep_t && !is_complex_dt(slot->orig_dt) && is_complex_dt(rhs->res_dt)))
    {
        return plan_err(err, errsz,
                        (sidx < 0) ? "failed to allocate slot"
                                   : "Cannot store a complex image in a real one; use real(), "
                                     "imag() or abs() first");
    }
    uint8_t         out_dt = keep_t ? slot->orig_dt : (rhs->res_dt ? rhs->res_dt : _DATATYPE_FLOAT);
    plan_walk_ctx_t w      = { .prog = prog, .pass = pass, .err = err, .errsz = errsz };
    if (walk_pass_expr(&w, rhs, (assign_op == MC_TOK_DIV_ASSIGN && rhs->res_kind != MC_RES_IMAGE),
                       0, 0) != 0 ||
        (lhs_sl != NULL && lhs_sl->u.slice.is_mask_index &&
         walk_pass_expr(&w, lhs_sl->u.slice.axes[0].start, 0, 0, 0) != 0))
    {
        return plan_err(err, errsz, "failed to plan pass expression");
    }
    pass = w.pass;
    for (int j = 0; !keep_t && out_dt == _DATATYPE_FLOAT &&
                    !(rhs->kind == MC_EXPR_CALL && rhs->u.call.fn && rhs->u.call.fn->cast_dt) &&
                    j < pass->nops;
         j++)
    {
        if (!pass->ops[j].is_scalar && !pass->ops[j].is_out && !pass->ops[j].is_table_arg &&
            pass->ops[j].dt == _DATATYPE_DOUBLE)
        {
            out_dt = _DATATYPE_DOUBLE;
        }
    }
    prog->slots[sidx].plan_dt = out_dt;

    mc_op_desc_t od  = { 0 };
    od.slot_idx      = sidx;
    od.dt            = out_dt;
    od.is_out        = 1;
    od.comp_gen_idx  = -1;
    od.is_inplace_rw = (assign_op != MC_TOK_ASSIGN || lhs_sl != NULL);
    od.has_slice     = (lhs_sl != NULL && !lhs_sl->u.slice.is_mask_index);
    od.slice_expr    = lhs_sl;
    int op_idx       = add_pass_op(pass, &od);
    if (pass->nouts >= pass->cap_outs)
    {
        pass->cap_outs = (pass->cap_outs == 0) ? 4 : pass->cap_outs * 2;
        pass->outs     = realloc(pass->outs, (size_t) pass->cap_outs * sizeof(*pass->outs));
    }
    mc_pass_out_t *po = &pass->outs[pass->nouts++];
    memset(po, 0, sizeof(*po));
    po->op_idx      = op_idx;
    po->mask_op_idx = -1;
    po->slot_idx    = sidx;
    po->stmt_idx    = stmt_idx;
    po->assign_op   = assign_op;
    po->rhs_expr    = rhs;
    po->lhs_slice   = lhs_sl;
    po->out_dt      = out_dt;
    return (op_idx < 0 || pass->outs == NULL) ? -1 : 0;
}

/**
 * unwrap_ax_red - Check if a pass operand is a pre-staged axis reduction call
 * @o: Operand descriptor in a pass
 *
 * When an axis reduction such as `imean(im, 0)` is combined with a 2D/3D image
 * in the same pass (e.g. `im - imean(im, 0)`), the reduction output must retain
 * a singleton dimension along the reduced axis (`1 x ny` instead of squeezed
 * `ny`) so standard axis broadcasting aligns the axes.
 *
 * Return: Pointer to the underlying 2-argument `MC_FN_REDUCE_D` call node, or NULL
 */
static const mc_expr_t *unwrap_ax_red(const mc_op_desc_t *o)
{
    const mc_expr_t *se =
        (!o->is_scalar && !o->is_out && !o->has_slice && !o->is_expr_im) ? o->slice_expr : NULL;
    return (se && se->kind == MC_EXPR_CALL && se->u.call.fn &&
            se->u.call.fn->kind == MC_FN_REDUCE_D && se->u.call.nargs == 2)
               ? se
               : NULL;
}

/**
 * finalize_pass_meta - Compute working precision, broadcast geometry, and loop mode
 * @prog: Enclosing program
 * @pass: Elementwise pass to finalize
 *
 * Inspects all inputs and outputs of @pass to determine:
 * - `is_complex` and `work_double` (whether the kernel computes in `float` or
 *   `double`, upgrading to `double` for >16-bit ints, `double` images, or large
 *   scalar constants exceeding 2^24).
 * - Unsqueezed broadcasting for pre-staged axis reductions (`is_anon = 2` and
 *   `is_x_bcast`).
 * - Planned output dimensions (`vnx`, `vny`, `vnz`, `naxis`) and whether the
 *   pass qualifies for the fast 1D contiguous loop (`is_1d_contig`).
 *
 * Return: void
 */
static void finalize_pass_meta(mcalc_prog_t *prog, mc_pass_t *pass)
{
    int is_cplx        = 0;
    int need_d         = 0;
    int has_non_red_im = 0;

    for (int i = 0; i < pass->nouts; i++)
    {
        uint8_t dt  = pass->outs[i].out_dt;
        uint8_t rdt = pass->outs[i].rhs_expr->res_dt;
        is_cplx |= (is_complex_dt(dt) || is_complex_dt(rdt));
        need_d |= (!is_float_exact_dt(dt) && dt != _DATATYPE_COMPLEX_FLOAT) ||
                  rdt == _DATATYPE_DOUBLE || rdt == _DATATYPE_COMPLEX_DOUBLE;
        has_non_red_im |= (pass->outs[i].lhs_slice != NULL);
    }
    for (int i = 0; i < pass->nops; i++)
    {
        const mc_op_desc_t *o = &pass->ops[i];
        is_cplx |= (!o->is_scalar && is_complex_dt(o->dt));
        need_d |=
            (!o->is_scalar && ((!is_float_exact_dt(o->dt) && o->dt != _DATATYPE_COMPLEX_FLOAT) ||
                               (o->is_non_pointwise && o->dt != _DATATYPE_FLOAT &&
                                o->dt != _DATATYPE_COMPLEX_FLOAT))) ||
            (o->is_scalar && o->scalar_expr != NULL &&
             ((o->scalar_expr->kind == MC_EXPR_LONG && labs(o->scalar_expr->u.lval) > 16777216L) ||
              (o->scalar_expr->kind == MC_EXPR_DOUBLE &&
               fabs(o->scalar_expr->u.dval) > 16777216.0)));
        const mc_slot_t *os = (o->slot_idx >= 0) ? &prog->slots[o->slot_idx] : NULL;
        has_non_red_im |=
            (!o->is_scalar && !o->is_out && !o->is_table_arg && !unwrap_ax_red(o) &&
             (!os || os->plan_size[0] != 1 || os->plan_size[1] != 1 || os->plan_size[2] != 1));
    }

    /* When an axis reduction is combined with a full image, keep reduced axis as size 1 */
    for (int i = 0; has_non_red_im && i < pass->nops; i++)
    {
        mc_op_desc_t    *o  = &pass->ops[i];
        const mc_expr_t *ub = unwrap_ax_red(o);
        for (int s = 0; ub != NULL && s < prog->nsteps; s++)
        {
            if (prog->steps[s].kind == MC_STEP_CTOR && prog->steps[s].ctor.slot_idx == o->slot_idx)
            {
                prog->steps[s].ctor.is_anon = 2;
            }
        }
        const mc_expr_t *a0 = ub ? ub->u.call.args[0] : NULL;
        long             ax =
            (ub && ub->u.call.args[1]->kind == MC_EXPR_LONG) ? ub->u.call.args[1]->u.lval : -1;
        if (ub != NULL)
        {
            o->is_x_bcast = (ax == 0);
        }
        mc_slot_t *sl = &prog->slots[o->slot_idx];
        if (a0 && a0->res_naxis > 0 && a0->res_size[0] > 0 && ax >= 0 && ax < a0->res_naxis)
        {
            sl->plan_naxis   = a0->res_naxis;
            sl->plan_size[0] = (ax == 0) ? 1 : a0->res_size[0];
            sl->plan_size[1] = (ax == 1) ? 1 : ((a0->res_naxis > 1) ? a0->res_size[1] : 1);
            sl->plan_size[2] = (ax == 2) ? 1 : ((a0->res_naxis > 2) ? a0->res_size[2] : 1);
        }
    }
    pass->is_complex  = is_cplx;
    pass->work_double = need_d;

    /* Determine whether all operands share a static, non-broadcast 1D-contiguous layout */
    int      contig = (!pass->uses_coords_or_stencil && !pass->is_comp);
    int      dyn_sz = pass->is_comp;
    uint8_t  nax    = 0;
    uint32_t sz[3]  = { 0, 0, 0 };
    for (int i = 0; i < pass->nouts; i++)
    {
        if (pass->outs[i].lhs_slice != NULL)
        {
            contig = 0;
            dyn_sz = 1;
        }
    }
    for (int i = 0; i < pass->nops; i++)
    {
        const mc_op_desc_t *o = &pass->ops[i];
        contig &= (o->is_scalar || !o->is_non_pointwise);
        if (!o->is_scalar && (o->has_slice || o->is_expr_im))
        {
            contig = 0;
            dyn_sz = 1;
        }
        if (o->is_scalar || o->is_table_arg || o->is_out || o->slot_idx < 0 || o->has_slice ||
            o->is_expr_im)
        {
            continue;
        }
        const mc_slot_t *s = &prog->slots[o->slot_idx];
        if (s->plan_naxis > 0 && s->plan_size[0] > 0 && sz[0] == 0)
        {
            nax = s->plan_naxis;
            memcpy(sz, s->plan_size, sizeof(sz));
        }
        else if (s->plan_naxis == 0 || s->plan_size[0] == 0 || s->plan_naxis != nax ||
                 (s->plan_size[0] != sz[0] && s->plan_size[0] != 1 && sz[0] != 1) ||
                 (s->plan_size[1] != sz[1] && s->plan_size[1] != 1 && sz[1] != 1) ||
                 (s->plan_size[2] != sz[2] && s->plan_size[2] != 1 && sz[2] != 1))
        {
            contig = 0;
            dyn_sz = 1;
        }
        else if (memcmp(sz, s->plan_size, sizeof(sz)) != 0)
        {
            contig = 0;
            for (int a = 0; a < 3; a++)
            {
                sz[a] = (sz[a] > s->plan_size[a]) ? sz[a] : s->plan_size[a];
            }
        }
    }
    const mc_slot_t *s0 =
        (!dyn_sz && pass->nouts > 0) ? &prog->slots[pass->outs[0].slot_idx] : NULL;
    if (s0 && sz[0] == 0 && s0->plan_naxis > 0 && s0->plan_size[0] > 0)
    {
        nax = s0->plan_naxis;
        memcpy(sz, s0->plan_size, sizeof(sz));
    }
    else if (s0 && sz[0] > 0 && pass->outs[0].lhs_slice == NULL && s0->existed_at_plan &&
             s0->plan_naxis > 0 && s0->plan_size[0] > 0 && !pass->is_complex &&
             !pass->no_dst_bcast &&
             (milk_data.image[s0->im_id].md[0].shared == 1 ||
              s0->orig_dt == pass->outs[0].out_dt) &&
             (s0->plan_naxis != nax || memcmp(sz, s0->plan_size, sizeof(sz)) != 0) &&
             (sz[0] == 1 || sz[0] == s0->plan_size[0]) &&
             (sz[1] == 1 || sz[1] == s0->plan_size[1]) && (sz[2] == 1 || sz[2] == s0->plan_size[2]))
    {
        contig = 0;
        nax    = s0->plan_naxis;
        memcpy(sz, s0->plan_size, sizeof(sz));
    }
    if (dyn_sz)
    {
        nax = 0;
        memset(sz, 0, sizeof(sz));
    }
    for (int i = 0; i < pass->nouts; i++)
    {
        mc_slot_t *s = &prog->slots[pass->outs[i].slot_idx];
        if (pass->outs[i].lhs_slice == NULL &&
            !(s->existed_at_plan && milk_data.image[s->im_id].md[0].shared == 1))
        {
            s->plan_naxis = nax;
            memcpy(s->plan_size, sz, sizeof(sz));
        }
    }
    pass->naxis        = nax;
    pass->vnx          = sz[0];
    pass->vny          = sz[1];
    pass->vnz          = sz[2];
    pass->is_1d_contig = (contig && sz[0] > 0);

    /* An image one pixel wide is broadcast along x: the SIMD row loop reads its one value per row
     * (the kernel checks at run time that its x stride is 0) */
    for (int i = 0; sz[0] > 1 && i < pass->nops; i++)
    {
        mc_op_desc_t *o = &pass->ops[i];
        if (!o->is_scalar && !o->is_out && !o->is_table_arg && !o->has_slice && !o->is_expr_im &&
            !o->is_non_pointwise && o->slot_idx >= 0 && prog->slots[o->slot_idx].plan_size[0] == 1)
        {
            o->is_x_bcast = 1;
        }
    }
}

/* ========================================================================== */
/* Pass Fusion & Top-Level Plan Builder                                       */
/* ========================================================================== */

/**
 * merge_passes_and_elide_locals - Fuse consecutive compatible passes and elide dead locals
 * @prog: Planned program whose step array is compacted in place
 *
 * Scans `prog->steps[]` and merges adjacent `MC_STEP_PASS` steps whenever
 * `mc_can_merge_passes()` approves. When pass B reads a slot written by pass A,
 * the redundant read operand is omitted from the merged operand list so the
 * kernel can read the already-computed local value `_v<out_idx>` directly from
 * a register. Finally calls `mc_elide_unused_locals()` to eliminate memory
 * writes for `local` intermediates that have no consumers outside their pass.
 *
 * Return: void
 */
static void merge_passes_and_elide_locals(mcalc_prog_t *prog)
{
    int w = 0;
    for (int r = 0; r < prog->nsteps; r++)
    {
        if (w > 0 && prog->steps[w - 1].kind == MC_STEP_PASS &&
            prog->steps[r].kind == MC_STEP_PASS &&
            prog->steps[w - 1].is_hist_step == prog->steps[r].is_hist_step &&
            mc_can_merge_passes(prog, &prog->steps[w - 1].pass, &prog->steps[r].pass))
        {
            mc_pass_t *pa = &prog->steps[w - 1].pass;
            mc_pass_t *pb = &prog->steps[r].pass;
            for (int i = 0; i < pb->nops; i++)
            {
                int skip = pb->ops[i].is_out;
                for (int j = 0; !skip && !pb->ops[i].is_scalar && j < pa->nouts; j++)
                {
                    skip |= (pa->outs[j].slot_idx == pb->ops[i].slot_idx);
                }
                if (!skip)
                {
                    (void) add_pass_op(pa, &pb->ops[i]);
                }
            }
            for (int i = 0; i < pb->nouts; i++)
            {
                mc_pass_out_t out = pb->outs[i];
                out.op_idx        = add_pass_op(pa, &pb->ops[out.op_idx]);
                if (pa->nouts >= pa->cap_outs)
                {
                    pa->cap_outs *= 2;
                    pa->outs = realloc(pa->outs, (size_t) pa->cap_outs * sizeof(*pa->outs));
                }
                pa->outs[pa->nouts++] = out;
            }
            pa->uses_coords_or_stencil |= pb->uses_coords_or_stencil;
            finalize_pass_meta(prog, pa);
            prog->steps[w - 1].last_stmt_idx = prog->steps[r].last_stmt_idx;
            for (int s = prog->steps[r].first_stmt_idx; s <= prog->steps[r].last_stmt_idx; s++)
            {
                if (prog->stmt_to_step[s] >= 0)
                {
                    prog->stmt_to_step[s] = w - 1;
                }
            }
            free(pb->ops);
            free(pb->outs);
            continue;
        }
        if (w != r)
        {
            prog->steps[w] = prog->steps[r];
            for (int s = prog->steps[w].first_stmt_idx; s <= prog->steps[w].last_stmt_idx; s++)
            {
                if (prog->stmt_to_step[s] >= 0)
                {
                    prog->stmt_to_step[s] = w;
                }
            }
        }
        w++;
    }
    prog->nsteps = w;
    mc_elide_unused_locals(prog);
}

/**
 * alloc_step - Append and zero-initialize a new execution step in `p->steps[]`
 * @p:        Program being planned
 * @kind:     Step kind (`MC_STEP_SCALAR`, `MC_STEP_PASS`, or `MC_STEP_CTOR`)
 * @stmt_idx: AST statement index associated with this step
 *
 * Return: Pointer to the newly allocated `mc_step_t`
 */
static mc_step_t *alloc_step(mcalc_prog_t *p, mc_step_kind_t kind, int stmt_idx)
{
    if (p->nsteps >= p->cap_steps)
    {
        p->cap_steps = (p->cap_steps == 0) ? 8 : p->cap_steps * 2;
        p->steps     = realloc(p->steps, (size_t) p->cap_steps * sizeof(*p->steps));
    }
    mc_step_t *st = &p->steps[p->nsteps];
    memset(st, 0, sizeof(*st));
    st->kind                  = kind;
    st->first_stmt_idx        = stmt_idx;
    st->last_stmt_idx         = stmt_idx;
    p->stmt_to_step[stmt_idx] = p->nsteps++;
    return st;
}

/**
 * mc_plan_build - Build execution steps, emit C kernels, and compile a program
 * @p:        Program with resolved AST (`p->ast_head`)
 * @has_vars: Unused legacy flag (retained for internal signature compatibility)
 * @err:      Error message buffer
 * @errsz:    Size of @err in bytes
 *
 * Iterates over all top-level statements in `p->ast_head`, classifying each
 * into `MC_STEP_SCALAR`, `MC_STEP_CTOR`, or `MC_STEP_PASS` (including fused
 * multi-LHS assignments such as `a, b = im1 + im2, im1 - im2`). After building
 * the initial step list, fuses consecutive elementwise passes, elides unused
 * local intermediates, emits C source for each pass, and compiles all kernels
 * via `mc_cc_compile_prog()`.
 *
 * Return: 0 on success, -1 on error (with @err populated)
 */
int mc_plan_build(mcalc_prog_t *p, int has_vars, char *err, size_t errsz)
{
    (void) has_vars;
    int idx = 0;
    for (mc_stmt_t *s = p->ast_head; s != NULL; s = s->next, idx++)
    {
        if (s->kind == MC_STMT_EMPTY || s->kind == MC_STMT_DEF)
        {
            continue;
        }
        int bad_s = (s->kind == MC_STMT_MULTI_ASSIGN && s->u.multi.nrhs != s->u.multi.nlhs &&
                     !(s->u.multi.nrhs == 1 && s->u.multi.nlhs >= 2));
        if (bad_s || (s->kind == MC_STMT_EXPR && check_expr_supported(s->u.expr, err, errsz) != 0))
        {
            return bad_s ? plan_err(err, errsz, "give as many values as names (a, b = x, y)") : -1;
        }
        for (int j = 0; s->kind == MC_STMT_MULTI_ASSIGN && j < s->u.multi.nrhs; j++)
        {
            if (check_expr_supported(s->u.multi.rhs[j], err, errsz) != 0)
            {
                return -1;
            }
        }

        /* Single-call multi-LHS image decomposition (e.g. `u, s, vt = svd(A)`) */
        if (s->kind == MC_STMT_MULTI_ASSIGN && s->u.multi.nrhs == 1 && s->u.multi.nlhs >= 2 &&
            mc_multi_lhs_is_image(s->u.multi.rhs[0], 0))
        {
            mc_step_t *st       = alloc_step(p, MC_STEP_CTOR, idx);
            st->ctor.multi_stmt = s;
            st->ctor.rhs_expr   = s->u.multi.rhs[0];
            for (int j = 0; j < s->u.multi.nlhs; j++)
            {
                const mc_expr_t *lhs = s->u.multi.lhs[j];
                mark_assigned_var(p, lhs->u.ident.bare);
                if (mc_multi_lhs_is_image(s->u.multi.rhs[0], j))
                {
                    int sidx = find_or_add_slot(p, lhs->u.ident.name, lhs->u.ident.bare,
                                                lhs->u.ident.loc, s->is_local, lhs->u.ident.im_id,
                                                s->u.multi.rhs[0]->res_dt, 0, NULL);
                    if (j == 0)
                    {
                        st->ctor.slot_idx = sidx;
                    }
                }
            }
            continue;
        }

        /* Pure scalar statements, control flow (`if`, `for`, `skip`), or scalar reductions */
        if (!s->needs_image || s->kind == MC_STMT_IF || s->kind == MC_STMT_FOR ||
            s->kind == MC_STMT_SKIP ||
            (s->kind == MC_STMT_MULTI_ASSIGN && s->u.multi.nrhs == 1 && s->u.multi.nlhs >= 2) ||
            (s->kind == MC_STMT_EXPR && s->u.expr->res_kind != MC_RES_IMAGE &&
             !(s->u.expr->kind == MC_EXPR_ASSIGN &&
               s->u.expr->u.assign.target->res_kind == MC_RES_IMAGE)))
        {
            alloc_step(p, MC_STEP_SCALAR, idx)->scalar_stmt = s;
            continue;
        }

        /* Single-expression image statement (constructor or elementwise pass) */
        if (s->kind == MC_STMT_EXPR)
        {
            const mc_expr_t *e      = s->u.expr;
            const mc_expr_t *target = (e->kind == MC_EXPR_ASSIGN) ? e->u.assign.target : NULL;
            const mc_expr_t *rhs    = (e->kind == MC_EXPR_ASSIGN) ? e->u.assign.rhs : e;
            mc_tok_type_t    aop    = (e->kind == MC_EXPR_ASSIGN) ? e->u.assign.op : MC_TOK_ASSIGN;
            int loc_fl = s->is_local || (e->kind == MC_EXPR_ASSIGN && e->u.assign.is_local);

            if ((is_ctor_call(rhs) && aop == MC_TOK_ASSIGN &&
                 (target == NULL || target->kind == MC_EXPR_IDENT)) ||
                mc_is_blas_or_linalg_ctor(rhs, target, aop))
            {
                mc_step_t  *st     = alloc_step(p, MC_STEP_CTOR, idx);
                int         tid    = (target != NULL);
                const char *br     = tid ? target->u.ident.bare : "_tmpcalc0";
                st->is_hist_step   = s->is_hist_stmt;
                st->ctor.slot_idx  = find_or_add_slot(p, tid ? target->u.ident.name : "_tmpcalc0",
                                                      br, tid ? target->u.ident.loc : 'L', loc_fl,
                                                      tid ? target->u.ident.im_id : -1, rhs->res_dt,
                                                      rhs->res_naxis, rhs->res_size);
                st->ctor.is_anon   = !tid;
                st->ctor.assign_op = aop;
                st->ctor.target_expr = target;
                st->ctor.rhs_expr    = rhs;
                continue;
            }
            alloc_step(p, MC_STEP_PASS, idx)->is_hist_step = s->is_hist_stmt;
            if (add_pass_out(p, &p->steps[p->nsteps - 1].pass, idx, loc_fl, 0, aop, target, rhs,
                             err, errsz) != 0)
            {
                return -1;
            }
            finalize_pass_meta(p, &p->steps[p->nsteps - 1].pass);
            continue;
        }

        /* Tuple multi-assignment (`a, b = expr1, expr2`) */
        int any_im = 0;
        int any_sc = 0;
        for (int j = 0; j < s->u.multi.nlhs; j++)
        {
            const mc_expr_t *lhs = s->u.multi.lhs[j];
            const mc_expr_t *rhs = s->u.multi.rhs[j];
            int is_im = (lhs->u.ident.id_kind == MC_ID_IMAGE || lhs->u.ident.id_kind == MC_ID_NEW ||
                         rhs->res_kind == MC_RES_IMAGE);
            any_im |= is_im;
            any_sc |= !is_im;
        }
        int      multi_fused = (any_im && !any_sc && s->u.multi.nlhs <= 4);
        uint32_t dsz[3]      = { 0, 0, 0 };
        for (int j = 0; multi_fused && j < s->u.multi.nlhs; j++)
        {
            const mc_expr_t *lhs = s->u.multi.lhs[j];
            const mc_expr_t *rhs = s->u.multi.rhs[j];
            imageID          id  = lhs->u.ident.im_id;
            uint8_t          fdt =
                (rhs->res_kind == MC_RES_IMAGE && rhs->kind == MC_EXPR_CALL &&
                 rhs->u.call.fn != NULL && rhs->u.call.fn->cast_dt != 0)
                    ? rhs->u.call.fn->cast_dt
                    : (rhs->res_dt == _DATATYPE_DOUBLE ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
            for (int k = 0; k < j; k++)
            {
                multi_fused &= (strcmp(s->u.multi.lhs[k]->u.ident.bare, lhs->u.ident.bare) != 0);
            }
            const IMAGE_METADATA *md    = (id >= 0) ? &milk_data.image[id].md[0] : NULL;
            uint32_t              s3[3] = { md ? md->size[0] : rhs->res_size[0],
                                            md ? ((md->naxis > 1) ? md->size[1] : 1)
                                               : ((rhs->res_naxis > 1) ? rhs->res_size[1] : 1),
                                            md ? ((md->naxis > 2) ? md->size[2] : 1)
                                               : ((rhs->res_naxis > 2) ? rhs->res_size[2] : 1) };
            multi_fused &= !((id < 0) != (s->u.multi.lhs[0]->u.ident.im_id < 0) ||
                             is_complex_dt(rhs->res_dt) ||
                             (id < 0 && (rhs->res_kind != MC_RES_IMAGE || rhs->res_size[0] == 0)) ||
                             (id >= 0 && md->shared != 1 && md->datatype != fdt) ||
                             (j > 0 && memcmp(dsz, s3, sizeof(dsz)) != 0));
            if (j == 0)
            {
                memcpy(dsz, s3, sizeof(dsz));
            }
            multi_fused &= !mc_has_bcast_stencil(rhs, dsz);
        }
        if (multi_fused)
        {
            (void) alloc_step(p, MC_STEP_PASS, idx);
        }
        for (int j = 0; any_im && j < s->u.multi.nlhs; j++)
        {
            const mc_expr_t *lhs = s->u.multi.lhs[j];
            const mc_expr_t *rhs = s->u.multi.rhs[j];
            if (lhs->u.ident.id_kind != MC_ID_IMAGE && lhs->u.ident.id_kind != MC_ID_NEW &&
                rhs->res_kind != MC_RES_IMAGE)
            {
                continue;
            }
            if (!multi_fused)
            {
                alloc_step(p, MC_STEP_PASS, idx)->pass.no_dst_bcast = 1;
            }
            if (add_pass_out(p, &p->steps[p->nsteps - 1].pass, idx, s->is_local, multi_fused,
                             MC_TOK_ASSIGN, lhs, rhs, err, errsz) != 0)
            {
                return -1;
            }
            if (!multi_fused)
            {
                finalize_pass_meta(p, &p->steps[p->nsteps - 1].pass);
            }
        }
        if (multi_fused)
        {
            finalize_pass_meta(p, &p->steps[p->nsteps - 1].pass);
        }
        if (any_sc)
        {
            alloc_step(p, MC_STEP_SCALAR, idx)->scalar_stmt = s;
        }
    }

    merge_passes_and_elide_locals(p);

    for (int st = 0; st < p->nsteps; st++)
    {
        if (p->steps[st].kind == MC_STEP_PASS &&
            mc_emit_pass(p, &p->steps[st].pass, err, errsz) != 0)
        {
            return -1;
        }
    }
    if (mc_cc_compile_prog(p, err, errsz) != 0)
    {
        return -1;
    }
    p->plan_gen = milk_table_generation();
    return 0;
}
