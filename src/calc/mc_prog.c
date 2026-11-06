// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libfps/fps_types.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_memory/create_variable.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/read_shmim.h"
#include "COREMOD_memory/variable_ID.h"
#include "mc_plan.h"

static _Thread_local int g_prog_skip       = 0;
static _Thread_local int g_prog_unit_depth = 0;

void mc_prog_set_skip(int on)
{
    g_prog_skip = on;
}

int mc_prog_skip_active(void)
{
    return g_prog_skip;
}

int mc_prog_in_unit(void)
{
    return g_prog_unit_depth > 0;
}

void mcalc_prog_attach(void)
{
}

void mc_prog_load_vars(mcalc_prog_t *p)
{
    if (p == NULL || p->vars == NULL || p->var_ids == NULL)
    {
        return;
    }
    for (int i = 0; i < p->nvars; i++)
    {
        if (p->var_ids[i] < 0 || p->vars[i].ptr == NULL)
        {
            continue;
        }
        VARIABLE *v = &milk_data.variable[p->var_ids[i]];
        void     *x = p->vars[i].ptr;
        switch (p->vars[i].type)
        {
        case FPTYPE_INT64:
            v->type    = 1;
            v->value.l = (long) *(int64_t *) x;
            break;
        case FPTYPE_INT32:
            v->type    = 1;
            v->value.l = (long) *(int32_t *) x;
            break;
        case FPTYPE_UINT64:
            v->type    = 1;
            v->value.l = (long) *(uint64_t *) x;
            break;
        case FPTYPE_UINT32:
            v->type    = 1;
            v->value.l = (long) *(uint32_t *) x;
            break;
        case FPTYPE_ONOFF:
            v->type    = 1;
            v->value.l = (*(int32_t *) x != 0);
            break;
        case FPTYPE_FLOAT64:
            v->type    = 0;
            v->value.f = *(double *) x;
            break;
        case FPTYPE_FLOAT32:
            v->type    = 0;
            v->value.f = (double) *(float *) x;
            break;
        case FPTYPE_STREAMNAME:
            if (v->type == 3 && strcmp(v->value.s, (const char *) x) == 0)
            {
                break;
            }
            v->type = 3;
            snprintf(v->value.s, sizeof(v->value.s), "%s", (const char *) x);
            if (v->value.s[0] != '\0' &&
                image_ID(v->value.s, milk_data.image, milk_data.NB_MAX_IMAGE) == -1)
            {
                read_sharedmem_image(v->value.s, milk_data.image, milk_data.NB_MAX_IMAGE);
            }
            break;
        default:
            v->type = 2;
            snprintf(v->value.s, sizeof(v->value.s), "%s", (const char *) x);
            break;
        }
    }
}

void mc_prog_store_vars(mcalc_prog_t *p)
{
    if (p == NULL || p->vars == NULL || p->var_ids == NULL || p->assigned == NULL)
    {
        return;
    }
    for (int i = 0; i < p->nvars; i++)
    {
        if (!p->assigned[i] || p->var_ids[i] < 0 || p->vars[i].ptr == NULL ||
            p->vars[i].type == FPTYPE_STREAMNAME)
        {
            continue;
        }
        const VARIABLE *v = &milk_data.variable[p->var_ids[i]];
        void           *x = p->vars[i].ptr;
        double          d = (v->type == 1) ? (double) v->value.l : v->value.f;
        int is_str = (p->vars[i].type == FPTYPE_STRING || p->vars[i].type == FPTYPE_FITSFILENAME ||
                      p->vars[i].type == FPTYPE_FILENAME);
        if ((v->type == 2) != is_str)
        {
            continue;
        }
        switch (p->vars[i].type)
        {
        case FPTYPE_INT64:
            *(int64_t *) x = (v->type == 1) ? v->value.l : llround(d);
            break;
        case FPTYPE_INT32:
            *(int32_t *) x = (int32_t) ((v->type == 1) ? v->value.l : llround(d));
            break;
        case FPTYPE_UINT64:
            *(uint64_t *) x = (d < 0) ? 0 : (uint64_t) ((v->type == 1) ? v->value.l : llround(d));
            break;
        case FPTYPE_UINT32:
            *(uint32_t *) x = (d < 0) ? 0 : (uint32_t) ((v->type == 1) ? v->value.l : llround(d));
            break;
        case FPTYPE_ONOFF:
            *(int32_t *) x = (d != 0.0);
            break;
        case FPTYPE_FLOAT64:
            *(double *) x = d;
            break;
        case FPTYPE_FLOAT32:
            *(float *) x = (float) d;
            break;
        default:
            snprintf((char *) x, FUNCTION_PARAMETER_STRMAXLEN, "%s", v->value.s);
            break;
        }
    }
}

static int is_readonly_unit_var(const char *name)
{
    return (strcmp(name, "_n") == 0 || strcmp(name, "_t") == 0 || strcmp(name, "_dt") == 0 ||
            strcmp(name, "_cnt0") == 0 || strcmp(name, "_ts") == 0 ||
            strcmp(name, "_missed") == 0 || strcmp(name, "_torn") == 0);
}

static int mark_assigned_name(mcalc_prog_t *p, const char *name, char *err, size_t errsz)
{
    if (name == NULL || name[0] == '\0')
    {
        return 0;
    }
    char nbuf[256], bare[256];
    snprintf(nbuf, sizeof(nbuf), "%s", name);
    mc_parse_ident_streamname(nbuf, bare, NULL);
    const char *key = (bare[0] != '\0') ? bare : name;
    if (p->vars != NULL && is_readonly_unit_var(key))
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "cannot assign to read-only variable '%s'", key);
        }
        return -1;
    }
    for (int i = 0; i < p->nvars; i++)
    {
        if (strcmp(p->vars[i].name, key) == 0 && p->vars[i].type != FPTYPE_STREAMNAME)
        {
            p->assigned[i] = 1;
        }
    }
    return 0;
}

static int scan_assigned_stmts(mcalc_prog_t *p, const mc_stmt_t *s, char *err, size_t errsz)
{
    for (; s != NULL; s = s->next)
    {
        if (s->kind == MC_STMT_EXPR && s->u.expr != NULL && s->u.expr->kind == MC_EXPR_ASSIGN)
        {
            const mc_expr_t *t = s->u.expr->u.assign.target;
            const char *nm = (t->kind == MC_EXPR_IDENT)
                                 ? (t->u.ident.bare[0] != '\0' ? t->u.ident.bare : t->u.ident.name)
                             : (t->kind == MC_EXPR_SLICE)
                                 ? (t->u.slice.bare[0] != '\0' ? t->u.slice.bare : t->u.slice.name)
                                 : NULL;
            if (mark_assigned_name(p, nm, err, errsz) != 0)
            {
                return -1;
            }
        }
        else if (s->kind == MC_STMT_MULTI_ASSIGN)
        {
            for (int j = 0; j < s->u.multi.nlhs; j++)
            {
                const mc_expr_t *t = s->u.multi.lhs[j];
                const char      *nm =
                    (t != NULL && t->kind == MC_EXPR_IDENT)
                        ? (t->u.ident.bare[0] != '\0' ? t->u.ident.bare : t->u.ident.name)
                        : NULL;
                if (mark_assigned_name(p, nm, err, errsz) != 0)
                {
                    return -1;
                }
            }
        }
        else if (s->kind == MC_STMT_FOR)
        {
            if (mark_assigned_name(p, s->u.for_stmt.var, err, errsz) != 0 ||
                scan_assigned_stmts(p, s->u.for_stmt.body, err, errsz) != 0)
            {
                return -1;
            }
        }
        else if (s->kind == MC_STMT_IF)
        {
            for (int b = 0; b < s->u.if_stmt.nbranches; b++)
            {
                if (scan_assigned_stmts(p, s->u.if_stmt.branches[b].body, err, errsz) != 0)
                {
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int bind_prog_vars(mcalc_prog_t        *p,
                          const mcalc_binding *vars,
                          int                  nvars,
                          char                *err,
                          size_t               errsz)
{
    p->nvars = nvars;
    if (vars == NULL || nvars <= 0)
    {
        return 0;
    }
    p->vars     = calloc((size_t) nvars, sizeof(*p->vars));
    p->var_ids  = calloc((size_t) nvars, sizeof(*p->var_ids));
    p->assigned = calloc((size_t) nvars, sizeof(*p->assigned));
    if (p->vars == NULL || p->var_ids == NULL || p->assigned == NULL)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "out of memory");
        }
        return -1;
    }
    for (int i = 0; i < nvars; i++)
    {
        int is_str    = (vars[i].type == FPTYPE_STRING || vars[i].type == FPTYPE_STREAMNAME ||
                         vars[i].type == FPTYPE_FITSFILENAME || vars[i].type == FPTYPE_FILENAME);
        int is_flt    = (vars[i].type == FPTYPE_FLOAT64 || vars[i].type == FPTYPE_FLOAT32);
        p->vars[i]    = vars[i];
        p->var_ids[i] = is_str   ? create_variable_string_ID(vars[i].name, "")
                        : is_flt ? create_variable_ID(vars[i].name, 0.0)
                                 : create_variable_long_ID(vars[i].name, 0);
        if (p->var_ids[i] < 0)
        {
            if (err != NULL && errsz > 0)
            {
                snprintf(err, errsz, "cannot create variable '%s'", vars[i].name);
            }
            return -1;
        }
    }
    mc_prog_load_vars(p);
    return 0;
}

static void split_stmt_texts(mcalc_prog_t *p, const char *text)
{
    if (p->nstmts <= 0 || text == NULL)
    {
        return;
    }
    p->stmt_texts = calloc((size_t) p->nstmts, sizeof(char *));
    if (p->stmt_texts == NULL)
    {
        return;
    }
    int         idx   = 0;
    int         depth = 0;
    char        quote = 0;
    const char *start = text;
    for (const char *c = text;; c++)
    {
        if (quote != 0)
        {
            if (*c == '\\' && c[1] != '\0')
            {
                c++;
                continue;
            }
            if (*c == quote)
            {
                quote = 0;
            }
        }
        else if (*c == '"' || *c == '\'')
        {
            quote = *c;
        }
        else if (*c == '(' || *c == '[' || *c == '{')
        {
            depth++;
        }
        else if ((*c == ')' || *c == ']' || *c == '}') && depth > 0)
        {
            depth--;
        }
        else if (depth == 0 && (*c == ';' || *c == '\n' || *c == '#' || *c == '\0'))
        {
            const char *s0 = start;
            const char *s1 = c;
            while (s0 < s1 && isspace((unsigned char) *s0))
            {
                s0++;
            }
            while (s1 > s0 && isspace((unsigned char) s1[-1]))
            {
                s1--;
            }
            if (s1 > s0 && idx < p->nstmts)
            {
                size_t len         = (size_t) (s1 - s0);
                p->stmt_texts[idx] = malloc(len + 1);
                if (p->stmt_texts[idx] != NULL)
                {
                    memcpy(p->stmt_texts[idx], s0, len);
                    p->stmt_texts[idx][len] = '\0';
                }
                idx++;
            }
            if (*c == '#')
            {
                while (*c != '\0' && *c != '\n')
                {
                    c++;
                }
            }
            if (*c == '\0')
            {
                break;
            }
            start = c + 1;
        }
    }
}

mcalc_prog *mcalc_prog_new(const char          *text,
                           const mcalc_binding *vars,
                           int                  nvars,
                           char                *err,
                           size_t               errsz)
{
    if (err != NULL && errsz > 0)
    {
        err[0] = '\0';
    }
    if (text == NULL)
    {
        return NULL;
    }
    if (vars != NULL)
    {
        mcalc_udef_clear();
    }
    mcalc_prog_t *p = calloc(1, sizeof(*p));
    if (p == NULL)
    {
        return NULL;
    }
    p->last_im_id = -1;
    p->src_text   = strdup(text);
    if (bind_prog_vars(p, vars, nvars, err, errsz) != 0)
    {
        mcalc_prog_free(p);
        return NULL;
    }

    int   nhist = 0;
    char *rt    = mc_rewrite_history(text, &nhist, err, errsz);
    if (rt == NULL)
    {
        mcalc_prog_free(p);
        return NULL;
    }
    mc_tok_list_t *tl = calloc(1, sizeof(*tl));
    if (tl == NULL || mc_lex(rt, tl) != 0)
    {
        if (tl != NULL && err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "%s", tl->errmsg[0] != '\0' ? tl->errmsg : "syntax error");
        }
        free(tl);
        free(rt);
        mcalc_prog_free(p);
        return NULL;
    }
    p->ast_head = mc_parse_prog(tl, err, errsz);
    free(tl);
    if (p->ast_head == NULL && err != NULL && err[0] != '\0')
    {
        free(rt);
        mcalc_prog_free(p);
        return NULL;
    }
    for (mc_stmt_t *s = p->ast_head; s != NULL; s = s->next)
    {
        p->nstmts++;
    }
    if (p->nstmts - nhist > 128)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "more than 128 statements");
        }
        free(rt);
        mcalc_prog_free(p);
        return NULL;
    }
    if (p->nstmts > 0)
    {
        p->stmt_to_step = malloc((size_t) p->nstmts * sizeof(int));
        if (p->stmt_to_step == NULL)
        {
            free(rt);
            mcalc_prog_free(p);
            return NULL;
        }
        for (int i = 0; i < p->nstmts; i++)
        {
            p->stmt_to_step[i] = -1;
        }
        split_stmt_texts(p, rt);
        int idx = 0;
        for (mc_stmt_t *s = p->ast_head; s != NULL; s = s->next, idx++)
        {
            s->is_hist_stmt = (idx >= p->nstmts - nhist);
            s->needs_image  = -1;
            if (s->kind == MC_STMT_DEF && (p->stmt_texts == NULL || p->stmt_texts[idx] == NULL ||
                                           mcalc_udef_try_parse(p->stmt_texts[idx]) <= 0))
            {
                (void) (err != NULL && errsz > 0 && snprintf(err, errsz, "invalid def syntax"));
                free(rt);
                mcalc_prog_free(p);
                return NULL;
            }
        }
    }
    free(rt);
    if (scan_assigned_stmts(p, p->ast_head, err, errsz) != 0)
    {
        mcalc_prog_free(p);
        return NULL;
    }
    if (mc_resolve_prog_ex(p->ast_head, vars != NULL, err, errsz) != 0)
    {
        if (vars != NULL && err != NULL && err[0] != '\0' && strstr(err, " in '") == NULL &&
            p->stmt_texts != NULL)
        {
            int fail_idx = -1, idx = 0;
            for (mc_stmt_t *s = p->ast_head; s != NULL; s = s->next, idx++)
            {
                if (s->needs_image != -1)
                {
                    fail_idx = idx;
                }
            }
            if (fail_idx >= 0 && fail_idx < p->nstmts && p->stmt_texts[fail_idx] != NULL)
            {
                size_t el = strlen(err);
                snprintf(err + el, errsz - el, " in '%s'", p->stmt_texts[fail_idx]);
            }
        }
        mcalc_prog_free(p);
        return NULL;
    }
    if (mc_plan_build(p, vars != NULL, err, errsz) != 0)
    {
        mcalc_prog_free(p);
        return NULL;
    }
    return p;
}

void mcalc_sub_prog_free(struct mcalc_prog *p)
{
    if (p != NULL)
    {
        if (p->ast_head != NULL && p->ast_head->kind == MC_STMT_EXPR && p->ast_head->next == NULL &&
            p->ast_head->u.expr != NULL && p->ast_head->u.expr->sub_prog == p)
        {
            p->ast_head->u.expr = NULL;
        }
        else
        {
            p->ast_head = NULL;
        }
        mcalc_prog_free(p);
    }
}

static void clear_prog_plan(mcalc_prog_t *p)
{
    for (int i = 0; i < p->nslots; i++)
    {
        imageID id = p->slots[i].im_id;
        if (strncmp(p->slots[i].bare, "_tmpcalc", 8) == 0 &&
            strncmp(p->slots[i].bare, "_tmpcalc0", 9) != 0 && id >= 0 && milk_data.image != NULL &&
            id < milk_data.NB_MAX_IMAGE && milk_data.image[id].used &&
            strcmp(milk_data.image[id].name, p->slots[i].bare) == 0)
        {
            mc_tmp_release_force(id);
        }
    }
    free(p->slots);
    p->slots  = NULL;
    p->nslots = p->cap_slots = 0;
    for (int i = 0; i < p->nsteps; i++)
    {
        if (p->steps[i].kind == MC_STEP_CTOR)
        {
            mc_ctor_step_free(&p->steps[i].ctor);
        }
        else if (p->steps[i].kind == MC_STEP_PASS)
        {
            free(p->steps[i].pass.ops);
            free(p->steps[i].pass.outs);
            free(p->steps[i].pass.c_src);
        }
    }
    free(p->steps);
    p->steps  = NULL;
    p->nsteps = p->cap_steps = 0;
}

int mc_prog_slots_valid(mcalc_prog_t *p, int same_gen)
{
    for (int i = 0; p != NULL && i < p->nslots; i++)
    {
        mc_slot_t *s = &p->slots[i];
        if (strncmp(s->bare, "_tmpcalc", 8) == 0)
        {
            continue;
        }
        imageID id =
            (s->im_id >= 0 && s->im_id < milk_data.NB_MAX_IMAGE && milk_data.image[s->im_id].used &&
             (same_gen || strcmp(milk_data.image[s->im_id].name, s->bare) == 0))
                ? s->im_id
                : mc_resolve_or_autoconnect_image(s->name);
        long vid = (same_gen || id >= 0) ? -1 : variable_ID(s->bare);
        if ((vid >= 0 && milk_data.variable[vid].type != 3) ||
            (id >= 0) != (s->existed_at_plan != 0))
        {
            return 0;
        }
        if (s->existed_at_plan)
        {
            const IMAGE_METADATA *md = (id >= 0) ? &milk_data.image[id].md[0] : NULL;
            uint32_t sz[3] = { md ? md->size[0] : 0, (md && md->naxis >= 2) ? md->size[1] : 1,
                               (md && md->naxis >= 3) ? md->size[2] : 1 };
            if (!md || md->datatype != s->orig_dt ||
                (s->plan_naxis > 0 &&
                 (md->naxis != s->plan_naxis || memcmp(sz, s->plan_size, sizeof(sz)) != 0)))
            {
                return 0;
            }
        }
        s->im_id = id;
    }
    return 1;
}

static int replan_prog_in_place(mcalc_prog_t *p, int in_unit, char *err, size_t errsz)
{
    clear_prog_plan(p);
    for (int i = 0; p->stmt_to_step != NULL && i < p->nstmts; i++)
    {
        p->stmt_to_step[i] = -1;
    }
    return (mc_resolve_prog_ex(p->ast_head, in_unit, err, errsz) == 0 &&
            mc_plan_build(p, in_unit, err, errsz) == 0)
               ? 0
               : -1;
}

void mcalc_prog_free(mcalc_prog *p)
{
    if (p == NULL)
    {
        return;
    }
    free(p->src_text);
    mc_stmt_free(p->ast_head);
    clear_prog_plan(p);
    free(p->stmt_to_step);
    if (p->stmt_texts != NULL)
    {
        for (int i = 0; i < p->nstmts; i++)
        {
            free(p->stmt_texts[i]);
        }
        free(p->stmt_texts);
    }
    free(p->vars);
    free(p->var_ids);
    free(p->assigned);
    free(p);
}

int mcalc_prog_assigns(const mcalc_prog *p, int i)
{
    return (p != NULL && p->assigned != NULL && i >= 0 && i < p->nvars) ? p->assigned[i] : 0;
}

void mcalc_prog_set_processinfo(mcalc_prog *p, void *processinfo)
{
    if (p != NULL)
    {
        p->processinfo = processinfo;
    }
}

static int run_prog_steps(mcalc_prog_t *p, int in_unit, char *err, size_t errsz)
{
    int rc = 0;
    for (int i = 0; i < p->nsteps && rc == 0; i++)
    {
        mc_step_t *st = &p->steps[i];
        if (g_prog_skip && !st->is_hist_step)
        {
            continue;
        }
        if (st->kind == MC_STEP_SCALAR)
        {
            rc             = mc_scalar_exec_stmt(st->scalar_stmt, &p->last_scalar, err, errsz);
            p->last_was_im = 0;
        }
        else if (st->kind == MC_STEP_CTOR)
        {
            rc             = mc_exec_ctor(p, &st->ctor, in_unit, err, errsz);
            p->last_was_im = 1;
            p->last_im_id  = (rc == 0) ? p->slots[st->ctor.slot_idx].im_id : -1;
        }
        else if (st->kind == MC_STEP_PASS)
        {
            rc             = mc_exec_pass(p, &st->pass, in_unit, err, errsz);
            p->last_was_im = 1;
            for (int j = st->pass.nouts - 1; rc == 0 && j >= 0; j--)
            {
                if (!st->pass.outs[j].is_elided)
                {
                    p->last_im_id = p->slots[st->pass.outs[j].slot_idx].im_id;
                    break;
                }
            }
        }
        if (rc != 0 && in_unit && err != NULL && err[0] != '\0' && strstr(err, " in '") == NULL &&
            p->stmt_texts != NULL && st->first_stmt_idx >= 0 && st->first_stmt_idx < p->nstmts &&
            p->stmt_texts[st->first_stmt_idx] != NULL)
        {
            size_t el = strlen(err);
            snprintf(err + el, errsz - el, " in '%s'", p->stmt_texts[st->first_stmt_idx]);
        }
    }
    return rc;
}

int mc_exec_sub_prog(struct mcalc_prog **io_prog, mc_stmt_t *head, char *err, size_t errsz)
{
    if (head == NULL)
    {
        return 0;
    }
    int           in_unit = mc_prog_in_unit();
    mcalc_prog_t *sub     = (io_prog != NULL) ? *io_prog : NULL;
    if (sub == NULL)
    {
        int nstmts = 0;
        for (mc_stmt_t *s = head; s != NULL; s = s->next)
        {
            nstmts++;
        }
        sub = calloc(1, sizeof(*sub));
        if (sub == NULL)
        {
            return -1;
        }
        sub->ast_head     = head;
        sub->nstmts       = nstmts;
        sub->last_im_id   = -1;
        sub->stmt_to_step = calloc((size_t) nstmts, sizeof(int));
        if (sub->stmt_to_step == NULL)
        {
            mcalc_sub_prog_free(sub);
            return -1;
        }
        for (int i = 0; i < nstmts; i++)
        {
            sub->stmt_to_step[i] = -1;
        }
        if (mc_resolve_prog_ex(head, in_unit, err, errsz) != 0 ||
            mc_plan_build(sub, in_unit, err, errsz) != 0)
        {
            mcalc_sub_prog_free(sub);
            return -1;
        }
        if (io_prog != NULL)
        {
            *io_prog = sub;
        }
    }
    else if (!mc_prog_slots_valid(sub,
                                  sub->plan_gen == milk_table_generation() && sub->plan_gen != 0) &&
             replan_prog_in_place(sub, in_unit, err, errsz) != 0)
    {
        return -1;
    }
    int rc = run_prog_steps(sub, in_unit, err, errsz);
    (void) (rc == 0 && (sub->plan_gen = milk_table_generation()));
    if (io_prog == NULL)
    {
        mcalc_sub_prog_free(sub);
    }
    return rc;
}

int mcalc_prog_run(mcalc_prog *p, char *err, size_t errsz)
{
    if (p == NULL)
    {
        return -1;
    }
    int in_unit    = (p->vars != NULL || p->processinfo != NULL);
    p->last_was_im = 0;
    p->last_im_id  = -1;
    mc_prog_load_vars(p);
    mc_post_defer(1);
    int saved_skip = g_prog_skip;
    g_prog_skip    = 0;
    if (in_unit)
    {
        g_prog_unit_depth++;
    }
    int same_gen = (p->plan_gen == milk_table_generation() && p->plan_gen != 0);
    int rc =
        (!mc_prog_slots_valid(p, same_gen) && replan_prog_in_place(p, in_unit, err, errsz) != 0)
            ? -1
            : run_prog_steps(p, in_unit, err, errsz);
    (void) (rc == 0 && (p->plan_gen = milk_table_generation()));
    if (in_unit)
    {
        g_prog_unit_depth--;
    }
    g_prog_skip = saved_skip;
    mc_post_flush(p->processinfo);
    if (rc == 0)
    {
        mc_prog_store_vars(p);
    }
    return rc;
}

int mc_exec_stmt_list(mc_stmt_t *head, char *err, size_t errsz)
{
    return mc_exec_sub_prog(NULL, head, err, errsz);
}

int mcalc_prog_exec_once(const char          *text,
                         const mcalc_binding *vars,
                         int                  nvars,
                         unsigned char       *assigned_out,
                         char                *err,
                         size_t               errsz)
{
    mcalc_prog *p = mcalc_prog_new(text, vars, nvars, err, errsz);
    if (p == NULL)
    {
        return -1;
    }
    int rc = mcalc_prog_run(p, err, errsz);
    if (rc == 0 && assigned_out != NULL && nvars > 0 && p->assigned != NULL)
    {
        memcpy(assigned_out, p->assigned, (size_t) nvars);
    }
    mcalc_prog_free(p);
    return rc;
}
