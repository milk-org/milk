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

#include "libfps/fps_types.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "linalg/mc_linalg.h"
#include "mc_plan.h"

__attribute__((format(printf, 3, 4))) static int plan_err(char       *err,
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
static inline int is_float_exact_dt(uint8_t dt)
{
    return dt == _DATATYPE_FLOAT || dt == _DATATYPE_UINT8 || dt == _DATATYPE_INT8 ||
           dt == _DATATYPE_UINT16 || dt == _DATATYPE_INT16;
}
static inline int is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}
static void mark_assigned_var(mcalc_prog_t *p, const char *bare)
{
    for (int i = 0; p != NULL && bare != NULL && i < p->nvars; i++)
    {
        (void) (strcmp(p->vars[i].name, bare) == 0 && (p->assigned[i] = 1));
    }
}
static int is_ctor_fn(const mc_func_t *fn)
{
    return fn != NULL && (fn->kind == MC_FN_LINSPACE || fn->kind == MC_FN_LINALG);
}
static int is_inline_stencil_fn(const mc_func_t *fn)
{
    return fn != NULL && fn->kind == MC_FN_STENCIL_LIB &&
           (strncmp(fn->name, "medfilt", 7) == 0 || strcmp(fn->name, "imin3") == 0 ||
            strcmp(fn->name, "imax3") == 0);
}
static int is_prestage_stencil_fn(const mc_func_t *fn)
{
    return fn != NULL && fn->kind == MC_FN_STENCIL_LIB && !is_inline_stencil_fn(fn);
}

static int is_ctor_call(const mc_expr_t *e)
{
    const mc_func_t *fn = (e != NULL && e->kind == MC_EXPR_CALL) ? e->u.call.fn : NULL;
    return (e != NULL && e->kind == MC_EXPR_CALL && e->u.call.udef_id >= 0 &&
            e->u.call.udef_body != NULL)
               ? is_ctor_call(e->u.call.udef_body)
               : ((e != NULL && e->kind == MC_EXPR_VEC_LIT) || is_ctor_fn(fn) ||
                  is_prestage_stencil_fn(fn) ||
                  (e != NULL && e->kind == MC_EXPR_CALL && e->u.call.ext_id >= 0) ||
                  (fn != NULL && fn->kind == MC_FN_REDUCE_D && e->res_kind == MC_RES_IMAGE));
}

static int expr_uses_def_param(const mc_expr_t *e)
{
    int r = (e != NULL) &&
            ((e->kind == MC_EXPR_IDENT && e->u.ident.def_param_idx >= 0) ||
             (e->kind == MC_EXPR_UNARY && expr_uses_def_param(e->u.unary.operand)) ||
             (e->kind == MC_EXPR_BINARY &&
              (expr_uses_def_param(e->u.binary.left) || expr_uses_def_param(e->u.binary.right))) ||
             (e->kind == MC_EXPR_TERNARY &&
              (expr_uses_def_param(e->u.ternary.cond) || expr_uses_def_param(e->u.ternary.t_expr) ||
               expr_uses_def_param(e->u.ternary.f_expr))));
    for (int i = 0; e != NULL && e->kind == MC_EXPR_CHAIN && i <= e->u.chain.nops; i++)
    {
        r |= expr_uses_def_param(e->u.chain.operands[i]);
    }
    for (int i = 0; e != NULL && e->kind == MC_EXPR_CALL && i < e->u.call.nargs; i++)
    {
        r |= expr_uses_def_param(e->u.call.args[i]);
    }
    return r;
}

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
    if (!e || e->kind != MC_EXPR_CALL || rc != 0 || e->u.call.udef_id >= 0)
    {
        return (rc || !e || e->kind != MC_EXPR_CALL)
                   ? rc
                   : check_expr_supported(e->u.call.udef_body, err, errsz);
    }
    const mc_fn_kind_t fk =
        (e->u.call.ext_id < 0 && e->u.call.fn) ? e->u.call.fn->kind : (mc_fn_kind_t) -1;
    return (e->u.call.ext_id >= 0 || (fk >= 0 && fk != MC_FN_PROG))
               ? 0
               : plan_err(err, errsz, "not implemented in the new engine");
}

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
            (void) (is_local && !prog->slots[i].existed_at_plan && (prog->slots[i].is_local = 1));
            (void) (dt != 0 && (prog->slots[i].plan_dt = dt));
            (void) (naxis > 0 && sz != NULL && sz[0] > 0 && (prog->slots[i].plan_naxis = naxis) &&
                    memcpy(prog->slots[i].plan_size, sz, 3 * sizeof(uint32_t)));
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
    s->loc             = (loc == 'S') ? 1 : (loc == 'L') ? 2 : 0;
    s->is_local        = is_local;
    im_id              = (im_id < 0 && key[0] != '\0' && strcmp(key, "_tmpcalc0") != 0)
                             ? mc_resolve_or_autoconnect_image(s->name)
                             : im_id;
    s->im_id           = im_id;
    s->existed_at_plan = (im_id >= 0);
    const IMAGE *im    = (im_id >= 0) ? &milk_data.image[im_id] : NULL;
    s->orig_dt = s->plan_dt = im ? im->md[0].datatype : (dt ? dt : _DATATYPE_FLOAT);
    s->plan_naxis           = im ? im->md[0].naxis : naxis;
    s->plan_size[0]         = im ? im->md[0].size[0] : (sz ? sz[0] : 0);
    s->plan_size[1]         = im ? ((s->plan_naxis > 1) ? im->md[0].size[1] : 1) : (sz ? sz[1] : 0);
    s->plan_size[2]         = im ? ((s->plan_naxis > 2) ? im->md[0].size[2] : 1) : (sz ? sz[2] : 0);
    return prog->nslots++;
}

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
            (void) (desc->check_div_kind && pass->ops[i].check_div_kind == 0 &&
                    (pass->ops[i].check_div_kind = desc->check_div_kind));
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

static int add_scalar_op(mc_pass_t *pass, const mc_expr_t *se, int div_c, int non_pt, int cg)
{
    mc_op_desc_t d = { 0 };
    d.slot_idx = -1, d.is_scalar = 1, d.is_non_pointwise = (uint8_t) non_pt;
    d.scalar_expr = se, d.check_div_kind = div_c, d.comp_gen_idx = cg;
    return (add_pass_op(pass, &d) < 0) ? -1 : 0;
}

typedef struct
{
    mcalc_prog_t        *prog;
    mc_pass_t           *pass;
    char                *err;
    size_t               errsz;
    const mc_comp_gen_t *comp_gens;
    int                  ncomp_gens;
    const mcalc_udef_t  *def_stk[4];
    mc_expr_t *const    *def_args[4];
    int                  def_depth;
    int                  in_guarded;
} plan_walk_ctx_t;

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
    if (e->kind == MC_EXPR_IDENT && e->u.ident.def_param_idx >= 0 && w->def_depth > 0)
    {
        w->def_depth--;
        int rc = walk_pass_expr(w, w->def_args[w->def_depth][e->u.ident.def_param_idx], div_check,
                                non_pt, tbl);
        w->def_depth++;
        return rc;
    }
    for (int g = 0; e->kind == MC_EXPR_IDENT && g < w->ncomp_gens; g++)
    {
        if (strcmp(e->u.ident.bare, w->comp_gens[g].var) == 0)
        {
            return 0;
        }
    }
    if (e->res_kind != MC_RES_IMAGE && !(w->def_depth > 0 && expr_uses_def_param(e)))
    {
        return add_scalar_op(w->pass, e, w->in_guarded ? 0 : div_check, non_pt, -1);
    }
    if (is_ctor_call(e) && !(w->def_depth > 0 && expr_uses_def_param(e)))
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
        cs->kind = MC_STEP_CTOR, cs->first_stmt_idx = cs->last_stmt_idx = stmt_idx;
        cs->ctor.slot_idx = ps_sidx, cs->ctor.is_anon = 1, cs->ctor.assign_op = MC_TOK_ASSIGN;
        cs->ctor.rhs_expr = e, p->nsteps++;
        w->pass           = &p->steps[p->nsteps - 1].pass, w->pass->prestage_cnt++;
        mc_op_desc_t d    = { .slot_idx         = ps_sidx,
                              .dt               = p->slots[ps_sidx].plan_dt,
                              .is_non_pointwise = non_pt,
                              .is_table_arg     = tbl,
                              .slice_expr       = e,
                              .comp_gen_idx     = -1 };
        return (add_pass_op(w->pass, &d) < 0) ? -1 : 0;
    }
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

    int rc = 0, saved = w->in_guarded;
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
        w->pass->is_comp = w->pass->uses_coords_or_stencil = 1;
        w->pass->comp_expr = e, w->comp_gens = e->u.comp.gens, w->ncomp_gens = e->u.comp.ngens;
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
        if (e->u.call.udef_id >= 0)
        {
            const mcalc_udef_t *u = mcalc_udef_get(e->u.call.udef_id);
            if (u == NULL || w->def_depth >= 4)
            {
                return -1;
            }
            w->def_stk[w->def_depth] = u, w->def_args[w->def_depth++] = e->u.call.args;
            rc = walk_pass_expr(w, e->u.call.udef_body, 0, 0, 0), w->def_depth--;
            return rc;
        }
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
        {
            int is_st = is_inline_stencil_fn(e->u.call.fn);
            w->pass->uses_coords_or_stencil |= (e->u.call.fn->kind == MC_FN_COORD || is_st);
            for (int i = 0; i < e->u.call.nargs && rc == 0; i++)
            {
                rc = walk_pass_expr(w, e->u.call.args[i], 0, (is_st && i == 0), 0);
            }
            return rc;
        }
        }
    default:
        return -1;
    }
}

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
    (void) (tid && (mark_assigned_var(prog, bare), 1));

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
                        (sidx < 0)
                            ? "failed to allocate slot"
                            : "Cannot store a complex image in a real one; use real(), imag() "
                              "or abs() first");
    }
    uint8_t         out_dt = keep_t ? slot->orig_dt : (rhs->res_dt ? rhs->res_dt : _DATATYPE_FLOAT);
    plan_walk_ctx_t w      = { .prog = prog, .pass = pass, .err = err, .errsz = errsz };
    if (walk_pass_expr(&w, rhs,
                       (assign_op == MC_TOK_DIV_ASSIGN && rhs->res_kind != MC_RES_IMAGE) ? 1 : 0, 0,
                       0) != 0 ||
        (lhs_sl != NULL && lhs_sl->u.slice.is_mask_index &&
         walk_pass_expr(&w, lhs_sl->u.slice.axes[0].start, 0, 0, 0) != 0))
    {
        return plan_err(err, errsz, "failed to plan pass expression");
    }
    pass                      = w.pass;
    prog->slots[sidx].plan_dt = out_dt;

    mc_op_desc_t od = { 0 };
    od.slot_idx = sidx, od.dt = out_dt, od.is_out = 1, od.comp_gen_idx = -1;
    od.is_inplace_rw = (assign_op != MC_TOK_ASSIGN || lhs_sl != NULL);
    od.has_slice = (lhs_sl != NULL && !lhs_sl->u.slice.is_mask_index), od.slice_expr = lhs_sl;
    int op_idx = add_pass_op(pass, &od);
    if (pass->nouts >= pass->cap_outs)
    {
        pass->cap_outs = (pass->cap_outs == 0) ? 4 : pass->cap_outs * 2;
        pass->outs     = realloc(pass->outs, (size_t) pass->cap_outs * sizeof(*pass->outs));
    }
    mc_pass_out_t *po = &pass->outs[pass->nouts++];
    memset(po, 0, sizeof(*po));
    po->op_idx = op_idx, po->mask_op_idx = -1, po->slot_idx = sidx, po->stmt_idx = stmt_idx;
    po->assign_op = assign_op, po->rhs_expr = rhs, po->lhs_slice = lhs_sl, po->out_dt = out_dt;
    return (op_idx < 0 || pass->outs == NULL) ? -1 : 0;
}

static const mc_expr_t *unwrap_ax_red(const mc_op_desc_t *o)
{
    const mc_expr_t *se =
        (!o->is_scalar && !o->is_out && !o->has_slice && !o->is_expr_im) ? o->slice_expr : NULL;
    const mc_expr_t *ub =
        (se && se->kind == MC_EXPR_CALL && se->u.call.udef_id >= 0 && se->u.call.udef_body)
            ? se->u.call.udef_body
            : se;
    return (ub && ub->kind == MC_EXPR_CALL && ub->u.call.fn &&
            ub->u.call.fn->kind == MC_FN_REDUCE_D && ub->u.call.nargs == 2)
               ? ub
               : NULL;
}

static void finalize_pass_meta(mcalc_prog_t *prog, mc_pass_t *pass)
{
    int is_cplx = 0, need_d = 0, has_non_red_im = 0;
    for (int i = 0; i < pass->nouts; i++)
    {
        uint8_t dt = pass->outs[i].out_dt, rdt = pass->outs[i].rhs_expr->res_dt;
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
    for (int i = 0; has_non_red_im && i < pass->nops; i++)
    {
        mc_op_desc_t    *o  = &pass->ops[i];
        const mc_expr_t *ub = unwrap_ax_red(o);
        for (int s = 0; ub != NULL && s < prog->nsteps; s++)
        {
            (void) (prog->steps[s].kind == MC_STEP_CTOR &&
                    prog->steps[s].ctor.slot_idx == o->slot_idx &&
                    (prog->steps[s].ctor.is_anon = 2));
        }
        const mc_expr_t *a0 = ub ? ub->u.call.args[0] : NULL;
        long             ax =
            (ub && ub->u.call.args[1]->kind == MC_EXPR_LONG) ? ub->u.call.args[1]->u.lval : -1;
        (void) (ub && (o->is_x_bcast = (ax == 0), 1));
        mc_slot_t *sl = &prog->slots[o->slot_idx];
        if (a0 && a0->res_naxis > 0 && a0->res_size[0] > 0 && ax >= 0 && ax < a0->res_naxis)
        {
            sl->plan_naxis   = a0->res_naxis;
            sl->plan_size[0] = (ax == 0) ? 1 : a0->res_size[0];
            sl->plan_size[1] = (ax == 1) ? 1 : ((a0->res_naxis > 1) ? a0->res_size[1] : 1);
            sl->plan_size[2] = (ax == 2) ? 1 : ((a0->res_naxis > 2) ? a0->res_size[2] : 1);
        }
    }
    pass->is_complex = is_cplx, pass->work_double = need_d;

    int      contig = (!pass->uses_coords_or_stencil && !pass->is_comp), dyn_sz = pass->is_comp;
    uint8_t  nax   = 0;
    uint32_t sz[3] = { 0, 0, 0 };
    for (int i = 0; i < pass->nouts; i++)
    {
        (void) (pass->outs[i].lhs_slice != NULL && ((contig = 0), (dyn_sz = 1)));
    }
    for (int i = 0; i < pass->nops; i++)
    {
        const mc_op_desc_t *o = &pass->ops[i];
        contig &= (o->is_scalar || !o->is_non_pointwise);
        (void) (!o->is_scalar && (o->has_slice || o->is_expr_im) && ((contig = 0), (dyn_sz = 1)));
        if (o->is_scalar || o->is_table_arg || o->is_out || o->slot_idx < 0 || o->has_slice ||
            o->is_expr_im)
        {
            continue;
        }
        const mc_slot_t *s = &prog->slots[o->slot_idx];
        if (s->plan_naxis > 0 && s->plan_size[0] > 0 && sz[0] == 0)
        {
            nax = s->plan_naxis, memcpy(sz, s->plan_size, sizeof(sz));
        }
        else if (s->plan_naxis == 0 || s->plan_size[0] == 0 || s->plan_naxis != nax ||
                 (s->plan_size[0] != sz[0] && s->plan_size[0] != 1 && sz[0] != 1) ||
                 (s->plan_size[1] != sz[1] && s->plan_size[1] != 1 && sz[1] != 1) ||
                 (s->plan_size[2] != sz[2] && s->plan_size[2] != 1 && sz[2] != 1))
        {
            contig = 0, dyn_sz = 1;
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
        nax = s0->plan_naxis, memcpy(sz, s0->plan_size, sizeof(sz));
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
        contig = 0, nax = s0->plan_naxis, memcpy(sz, s0->plan_size, sizeof(sz));
    }
    (void) (dyn_sz && ((nax = 0), memset(sz, 0, sizeof(sz))));
    for (int i = 0; i < pass->nouts; i++)
    {
        mc_slot_t *s = &prog->slots[pass->outs[i].slot_idx];
        (void) (pass->outs[i].lhs_slice == NULL &&
                !(s->existed_at_plan && milk_data.image[s->im_id].md[0].shared == 1) &&
                ((s->plan_naxis = nax), memcpy(s->plan_size, sz, sizeof(sz))));
    }
    pass->naxis = nax, pass->vnx = sz[0], pass->vny = sz[1], pass->vnz = sz[2];
    pass->is_1d_contig = (contig && sz[0] > 0);
}

static void merge_passes_and_elide_locals(mcalc_prog_t *prog)
{
    int w = 0;
    for (int r = 0; r < prog->nsteps; r++)
    {
        if (w > 0 && prog->steps[w - 1].kind == MC_STEP_PASS &&
            prog->steps[r].kind == MC_STEP_PASS &&
            mc_can_merge_passes(&prog->steps[w - 1].pass, &prog->steps[r].pass))
        {
            mc_pass_t *pa = &prog->steps[w - 1].pass, *pb = &prog->steps[r].pass;
            for (int i = 0; i < pb->nops; i++)
            {
                int skip = pb->ops[i].is_out;
                for (int j = 0; !skip && !pb->ops[i].is_scalar && j < pa->nouts; j++)
                {
                    skip |= (pa->outs[j].slot_idx == pb->ops[i].slot_idx);
                }
                (void) (!skip && (add_pass_op(pa, &pb->ops[i]), 1));
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
                (void) (prog->stmt_to_step[s] >= 0 && (prog->stmt_to_step[s] = w - 1, 1));
            }
            free(pb->ops), free(pb->outs);
            continue;
        }
        if (w != r)
        {
            prog->steps[w] = prog->steps[r];
            for (int s = prog->steps[w].first_stmt_idx; s <= prog->steps[w].last_stmt_idx; s++)
            {
                (void) (prog->stmt_to_step[s] >= 0 && (prog->stmt_to_step[s] = w, 1));
            }
        }
        w++;
    }
    prog->nsteps = w;
    mc_elide_unused_locals(prog);
}

static mc_step_t *alloc_step(mcalc_prog_t *p, mc_step_kind_t kind, int stmt_idx)
{
    if (p->nsteps >= p->cap_steps)
    {
        p->cap_steps = (p->cap_steps == 0) ? 8 : p->cap_steps * 2;
        p->steps     = realloc(p->steps, (size_t) p->cap_steps * sizeof(*p->steps));
    }
    mc_step_t *st = &p->steps[p->nsteps];
    memset(st, 0, sizeof(*st));
    st->kind = kind, st->first_stmt_idx = st->last_stmt_idx = stmt_idx;
    p->stmt_to_step[stmt_idx] = p->nsteps++;
    return st;
}

int mc_plan_build(mcalc_prog_t *p, int has_vars, char *err, size_t errsz)
{
    int idx = 0;
    for (mc_stmt_t *s = p->ast_head; s != NULL; s = s->next, idx++)
    {
        if (s->kind == MC_STMT_EMPTY || s->kind == MC_STMT_DEF)
        {
            continue;
        }
        int bad_s = (s->kind == MC_STMT_SKIP ||
                     ((s->kind == MC_STMT_IF || s->kind == MC_STMT_FOR) && has_vars) ||
                     (s->kind == MC_STMT_MULTI_ASSIGN && s->u.multi.nrhs != s->u.multi.nlhs &&
                      !(s->u.multi.nrhs == 1 && s->u.multi.nlhs >= 2)));
        if (bad_s || (s->kind == MC_STMT_EXPR && check_expr_supported(s->u.expr, err, errsz) != 0))
        {
            return bad_s ? plan_err(err, errsz, "not implemented in the new engine") : -1;
        }
        for (int j = 0; s->kind == MC_STMT_MULTI_ASSIGN && j < s->u.multi.nrhs; j++)
        {
            if (check_expr_supported(s->u.multi.rhs[j], err, errsz) != 0)
            {
                return -1;
            }
        }

        if (s->kind == MC_STMT_MULTI_ASSIGN && s->u.multi.nrhs == 1 && s->u.multi.nlhs >= 2 &&
            mc_multi_lhs_is_image(s->u.multi.rhs[0], 0))
        {
            mc_step_t *st       = alloc_step(p, MC_STEP_CTOR, idx);
            st->ctor.multi_stmt = s, st->ctor.rhs_expr = s->u.multi.rhs[0];
            for (int j = 0; j < s->u.multi.nlhs; j++)
            {
                const mc_expr_t *lhs = s->u.multi.lhs[j];
                mark_assigned_var(p, lhs->u.ident.bare);
                if (mc_multi_lhs_is_image(s->u.multi.rhs[0], j))
                {
                    int sidx = find_or_add_slot(p, lhs->u.ident.name, lhs->u.ident.bare,
                                                lhs->u.ident.loc, s->is_local, lhs->u.ident.im_id,
                                                s->u.multi.rhs[0]->res_dt, 0, NULL);
                    (void) (j == 0 && (st->ctor.slot_idx = sidx, 1));
                }
            }
            continue;
        }

        if (!s->needs_image || s->kind == MC_STMT_IF || s->kind == MC_STMT_FOR ||
            (s->kind == MC_STMT_MULTI_ASSIGN && s->u.multi.nrhs == 1 && s->u.multi.nlhs >= 2) ||
            (s->kind == MC_STMT_EXPR && s->u.expr->res_kind != MC_RES_IMAGE &&
             !(s->u.expr->kind == MC_EXPR_ASSIGN &&
               s->u.expr->u.assign.target->res_kind == MC_RES_IMAGE)))
        {
            (void) (s->kind == MC_STMT_EXPR && s->u.expr->kind == MC_EXPR_ASSIGN &&
                    s->u.expr->u.assign.target->kind == MC_EXPR_IDENT &&
                    (mark_assigned_var(p, s->u.expr->u.assign.target->u.ident.bare), 1));
            for (int j = 0; s->kind == MC_STMT_MULTI_ASSIGN && j < s->u.multi.nlhs; j++)
            {
                mark_assigned_var(p, s->u.multi.lhs[j]->u.ident.bare);
            }
            alloc_step(p, MC_STEP_SCALAR, idx)->scalar_stmt = s;
            continue;
        }

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
                mc_step_t  *st  = alloc_step(p, MC_STEP_CTOR, idx);
                int         tid = (target != NULL);
                const char *br  = tid ? target->u.ident.bare : "_tmpcalc0";
                (void) (tid && (mark_assigned_var(p, br), 1));
                st->ctor.slot_idx = find_or_add_slot(p, tid ? target->u.ident.name : "_tmpcalc0",
                                                     br, tid ? target->u.ident.loc : 'L', loc_fl,
                                                     tid ? target->u.ident.im_id : -1, rhs->res_dt,
                                                     rhs->res_naxis, rhs->res_size);
                st->ctor.is_anon = !tid, st->ctor.assign_op = aop, st->ctor.target_expr = target;
                st->ctor.rhs_expr = rhs;
                continue;
            }
            alloc_step(p, MC_STEP_PASS, idx);
            if (add_pass_out(p, &p->steps[p->nsteps - 1].pass, idx, loc_fl, 0, aop, target, rhs,
                             err, errsz) != 0)
            {
                return -1;
            }
            finalize_pass_meta(p, &p->steps[p->nsteps - 1].pass);
            continue;
        }
        int any_im = 0, any_sc = 0;
        for (int j = 0; j < s->u.multi.nlhs; j++)
        {
            const mc_expr_t *lhs = s->u.multi.lhs[j], *rhs = s->u.multi.rhs[j];
            int is_im = (lhs->u.ident.id_kind == MC_ID_IMAGE || lhs->u.ident.id_kind == MC_ID_NEW ||
                         rhs->res_kind == MC_RES_IMAGE);
            any_im |= is_im, any_sc |= !is_im;
        }
        int      multi_fused = (any_im && !any_sc && s->u.multi.nlhs <= 4);
        uint32_t dsz[3]      = { 0, 0, 0 };
        for (int j = 0; multi_fused && j < s->u.multi.nlhs; j++)
        {
            const mc_expr_t *lhs = s->u.multi.lhs[j], *rhs = s->u.multi.rhs[j];
            imageID          id = lhs->u.ident.im_id;
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
            (void) (j == 0 && memcpy(dsz, s3, sizeof(dsz)));
            multi_fused &= !mc_has_bcast_stencil(rhs, dsz);
        }
        (void) (multi_fused && (alloc_step(p, MC_STEP_PASS, idx), 1));
        for (int j = 0; any_im && j < s->u.multi.nlhs; j++)
        {
            const mc_expr_t *lhs = s->u.multi.lhs[j], *rhs = s->u.multi.rhs[j];
            if (lhs->u.ident.id_kind != MC_ID_IMAGE && lhs->u.ident.id_kind != MC_ID_NEW &&
                rhs->res_kind != MC_RES_IMAGE)
            {
                continue;
            }
            (void) (!multi_fused && (alloc_step(p, MC_STEP_PASS, idx)->pass.no_dst_bcast = 1));
            if (add_pass_out(p, &p->steps[p->nsteps - 1].pass, idx, s->is_local, multi_fused,
                             MC_TOK_ASSIGN, lhs, rhs, err, errsz) != 0)
            {
                return -1;
            }
            (void) (!multi_fused && (finalize_pass_meta(p, &p->steps[p->nsteps - 1].pass), 1));
        }
        (void) (multi_fused && (finalize_pass_meta(p, &p->steps[p->nsteps - 1].pass), 1));
        (void) (any_sc && (alloc_step(p, MC_STEP_SCALAR, idx)->scalar_stmt = s, 1));
    }

    merge_passes_and_elide_locals(p);

    for (int st = 0; st < p->nsteps; st++)
    {
        mc_pass_t *pa = &p->steps[st].pass;
        if (p->steps[st].kind == MC_STEP_PASS &&
            (mc_emit_pass(p, pa, err, errsz) != 0 ||
             (pa->fn = mc_cc_compile(pa->c_src, pa->so_path, sizeof(pa->so_path), err, errsz)) ==
                 NULL))
        {
            return -1;
        }
    }
    return 0;
}
