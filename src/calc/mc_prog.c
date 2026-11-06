// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
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
    if (p->vars != NULL && is_readonly_unit_var(name))
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "cannot assign to read-only variable '%s'", name);
        }
        return -1;
    }
    for (int i = 0; i < p->nvars; i++)
    {
        if (strcmp(p->vars[i].name, name) == 0 && p->vars[i].type != FPTYPE_STREAMNAME)
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

    mc_tok_list_t *tl = calloc(1, sizeof(*tl));
    if (tl == NULL || mc_lex(text, tl) != 0)
    {
        if (tl != NULL && err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "%s", tl->errmsg[0] != '\0' ? tl->errmsg : "syntax error");
        }
        free(tl);
        mcalc_prog_free(p);
        return NULL;
    }
    p->ast_head = mc_parse_prog(tl, err, errsz);
    free(tl);
    if (p->ast_head == NULL && err != NULL && err[0] != '\0')
    {
        mcalc_prog_free(p);
        return NULL;
    }
    for (mc_stmt_t *s = p->ast_head; s != NULL; s = s->next)
    {
        p->nstmts++;
    }
    if (p->nstmts > 0)
    {
        p->stmt_to_step = malloc((size_t) p->nstmts * sizeof(int));
        if (p->stmt_to_step == NULL)
        {
            mcalc_prog_free(p);
            return NULL;
        }
        for (int i = 0; i < p->nstmts; i++)
        {
            p->stmt_to_step[i] = -1;
        }
        split_stmt_texts(p, text);
        int idx = 0;
        for (mc_stmt_t *s = p->ast_head; s != NULL; s = s->next, idx++)
        {
            if (s->kind == MC_STMT_DEF && (p->stmt_texts == NULL || p->stmt_texts[idx] == NULL ||
                                           mcalc_udef_try_parse(p->stmt_texts[idx]) <= 0))
            {
                (void) (err != NULL && errsz > 0 && snprintf(err, errsz, "invalid def syntax"));
                mcalc_prog_free(p);
                return NULL;
            }
        }
    }
    if (mc_resolve_prog(p->ast_head, err, errsz) != 0 ||
        scan_assigned_stmts(p, p->ast_head, err, errsz) != 0 ||
        mc_plan_build(p, vars != NULL, err, errsz) != 0)
    {
        mcalc_prog_free(p);
        return NULL;
    }
    return p;
}

void mcalc_prog_free(mcalc_prog *p)
{
    if (p == NULL)
    {
        return;
    }
    free(p->src_text);
    mc_stmt_free(p->ast_head);
    free(p->slots);
    for (int i = 0; i < p->nsteps; i++)
    {
        if (p->steps[i].kind == MC_STEP_PASS)
        {
            free(p->steps[i].pass.ops);
            free(p->steps[i].pass.outs);
            free(p->steps[i].pass.c_src);
        }
    }
    free(p->steps);
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

int mcalc_prog_run(mcalc_prog *p, char *err, size_t errsz)
{
    if (p == NULL)
    {
        return -1;
    }
    int in_unit    = (p->vars != NULL || p->processinfo != NULL);
    int rc         = 0;
    p->last_was_im = 0;
    p->last_im_id  = -1;
    mc_prog_load_vars(p);
    mc_post_defer(1);
    for (int i = 0; i < p->nsteps && rc == 0; i++)
    {
        mc_step_t *st = &p->steps[i];
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
    }
    mc_post_flush(p->processinfo);
    if (rc == 0)
    {
        mc_prog_store_vars(p);
    }
    return rc;
}

int mc_exec_stmt_list(mc_stmt_t *head, char *err, size_t errsz)
{
    if (head == NULL)
    {
        return 0;
    }
    int nstmts = 0;
    for (mc_stmt_t *s = head; s != NULL; s = s->next)
    {
        nstmts++;
    }
    mcalc_prog_t p = { .ast_head = head, .nstmts = nstmts, .last_im_id = -1 };
    p.stmt_to_step = calloc((size_t) nstmts, sizeof(int));
    if (p.stmt_to_step == NULL)
    {
        return -1;
    }
    for (int i = 0; i < nstmts; i++)
    {
        p.stmt_to_step[i] = -1;
    }
    int rc = (mc_resolve_prog(head, err, errsz) == 0 && mc_plan_build(&p, 0, err, errsz) == 0 &&
              mcalc_prog_run(&p, err, errsz) == 0)
                 ? 0
                 : -1;
    for (int i = 0; i < p.nsteps; i++)
    {
        if (p.steps[i].kind == MC_STEP_PASS)
        {
            free(p.steps[i].pass.ops);
            free(p.steps[i].pass.outs);
            free(p.steps[i].pass.c_src);
        }
    }
    free(p.steps);
    free(p.slots);
    free(p.stmt_to_step);
    return rc;
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

int mcalc_prog_explain(const char *text, FILE *out)
{
    char        err[512] = "";
    mcalc_prog *p        = mcalc_prog_new(text, NULL, 0, err, sizeof(err));
    if (p == NULL || mcalc_prog_run(p, err, sizeof(err)) != 0)
    {
        if (out != NULL)
        {
            fprintf(out, "milk.calc: %s\n", err[0] != '\0' ? err : "compilation error");
        }
        mcalc_prog_free(p);
        return 1;
    }
    int nkept = 0;
    for (int i = 0; i < p->nstmts; i++)
    {
        const char *st_txt =
            (p->stmt_texts != NULL && p->stmt_texts[i] != NULL) ? p->stmt_texts[i] : "";
        int  sidx = (p->stmt_to_step != NULL) ? p->stmt_to_step[i] : -1;
        char how[512];
        if (sidx < 0)
        {
            snprintf(how, sizeof(how), "def, inlined where it is called");
        }
        else
        {
            const mc_step_t *st = &p->steps[sidx];
            nkept++;
            if (st->kind == MC_STEP_PASS)
            {
                const char *wtype =
                    st->pass.is_complex
                        ? (st->pass.work_double ? "complex double" : "complex float32")
                        : (st->pass.work_double ? "double" : "float32");
                if (i > st->first_stmt_idx)
                {
                    snprintf(how, sizeof(how),
                             "kept, merged into the fused pass of statement %d (%s)",
                             st->first_stmt_idx + 1, wtype);
                }
                else
                {
                    snprintf(how, sizeof(how),
                             "kept: compiled C pass (%s, %s, %d operand%s, %d output%s)", wtype,
                             st->pass.is_1d_contig ? "1D" : "ND", st->pass.nops,
                             (st->pass.nops == 1) ? "" : "s", st->pass.nouts,
                             (st->pass.nouts == 1) ? "" : "s");
                }
            }
            else if (st->kind == MC_STEP_CTOR)
            {
                snprintf(how, sizeof(how), "kept: constructor/shape step");
            }
            else
            {
                snprintf(how, sizeof(how), "kept: scalar step");
            }
        }
        if (out != NULL)
        {
            fprintf(out, "%3d  %s\n       %s\n", i + 1, st_txt, how);
        }
    }
    if (out != NULL)
    {
        fprintf(out, "%d statement%s kept, 0 evaluated as text\n", nkept, (nkept == 1) ? "" : "s");
    }
    mcalc_prog_free(p);
    return 0;
}
