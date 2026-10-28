// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_prog.c
 * @brief Calculator programs for milk.calc units: variable binding, compilation, and per-frame
 *        execution.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "cli_calc_prog_internal.h"

/**
 * cli_calc_prog_attach - Bind the CLI core image and variable tables to `milk_data`
 *
 * Points `data.core` at `milk_data` so standalone `milk.calc` compute units
 * share the same in-memory image and variable arrays as the calculator engine.
 *
 * Return: void
 */
void cli_calc_prog_attach(void)
{
    data.core = milk_data;
}

/**
 * eval_text - Evaluate a single calculator statement from source text
 */
static int eval_text(const char *stmt, char *err, size_t errsz)
{
    char out[256];
    int  rc = 2;

    if (cli_calc_eval_str(stmt, 0, &rc, out, sizeof(out)) && rc == 0)
    {
        return 0;
    }
    snprintf(err, errsz, "%s in '%s'", (cli_calc_lasterr[0] != '\0') ? cli_calc_lasterr : "error",
             stmt);
    return -1;
}

int cli_calc_prog_quiet = 0;

/** note_why() - keep the reason the last keep attempt of @st failed, "" once it is kept. */
static void note_why(prog_stmt *st)
{
    if (st->kept != NULL)
    {
        st->why[0] = '\0';
        return;
    }
    snprintf(st->why, sizeof(st->why), "%s", cli_calc_fuse_reject_reason());
}

/** is_def_stmt() - 1 if @text is a `def f(...) = ...` line (registered once, nothing to keep). */
static int is_def_stmt(const char *text)
{
    while (*text == ' ' || *text == '\t')
    {
        text++;
    }
    return strncmp(text, "def", 3) == 0 && (text[3] == ' ' || text[3] == '\t');
}

/** print_not_kept_notes() - say once, at compile, which statements will be evaluated as text. */
static void print_not_kept_notes(const cli_calc_prog *p)
{
    for (int i = 0; i < p->n; i++)
    {
        const prog_stmt *st = &p->st[i];
        if (st->kind != STMT_NORMAL || st->kept != NULL || st->why[0] == '\0' ||
            is_def_stmt(st->text))
        {
            continue;
        }
        fprintf(stderr,
                "milk.calc: statement %d is not kept, it is evaluated as text each tick (%s): %s\n",
                i + 1, st->why, st->text);
    }
}

/** share() - make local image @name a shared stream of the same name and content. */
static int share(const char *name, char *err, size_t errsz)
{
    char tmp[STRINGMAXLEN_IMAGE_NAME];

    snprintf(tmp, sizeof(tmp), "_calcprog_%.60s", name);
    if (chname_image_ID(name, tmp) == -1 || copy_image_ID(tmp, name, 1) == -1)
    {
        snprintf(err, errsz, "cannot create stream '%s'", name);
        return -1;
    }
    delete_image_ID(tmp, DELETE_IMAGE_ERRMODE_IGNORE);
    return 0;
}

/**
 * load_vars - Copy bound C parameter values into calculator variables before execution
 */
static void load_vars(cli_calc_prog *p)
{
    for (int i = 0; i < p->nvars; i++)
    {
        VARIABLE *v = &data.core.variable[p->vid[i]];
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
                image_ID(v->value.s, data.core.image, data.core.NB_MAX_IMAGE) == -1)
            {
                read_sharedmem_image(v->value.s, data.core.image, data.core.NB_MAX_IMAGE);
            }
            break;
        default:
            v->type = 2;
            snprintf(v->value.s, sizeof(v->value.s), "%s", (const char *) x);
        }
    }
}

/**
 * store_vars - Write assigned calculator variable values back to bound C parameters
 */
static void store_vars(cli_calc_prog *p)
{
    for (int i = 0; i < p->nvars; i++)
    {
        const VARIABLE *v = &data.core.variable[p->vid[i]];
        void           *x = p->vars[i].ptr;
        double          d = (v->type == 1) ? (double) v->value.l : v->value.f;
        int is_str = (p->vars[i].type == FPTYPE_STRING || p->vars[i].type == FPTYPE_FITSFILENAME ||
                      p->vars[i].type == FPTYPE_FILENAME);
        if (!p->assigned[i] || p->vars[i].type == FPTYPE_STREAMNAME || (v->type == 2) != is_str)
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
        }
    }
}

/** bind_vars() - create the bound variables with their current values. */
static int bind_vars(cli_calc_prog *p, const cli_calc_binding *vars, char *err, size_t errsz)
{
    p->vars     = calloc((size_t) p->nvars + 1, sizeof(*p->vars));
    p->vid      = calloc((size_t) p->nvars + 1, sizeof(*p->vid));
    p->assigned = calloc((size_t) p->nvars + 1, 1);
    if (p->vars == NULL || p->vid == NULL || p->assigned == NULL)
    {
        return -1;
    }
    for (int i = 0; i < p->nvars; i++)
    {
        int is_str = (vars[i].type == FPTYPE_STRING || vars[i].type == FPTYPE_STREAMNAME ||
                      vars[i].type == FPTYPE_FITSFILENAME || vars[i].type == FPTYPE_FILENAME);
        int is_flt = (vars[i].type == FPTYPE_FLOAT64 || vars[i].type == FPTYPE_FLOAT32);
        p->vars[i] = vars[i];
        p->vid[i]  = is_str   ? create_variable_string_ID(vars[i].name, "")
                     : is_flt ? create_variable_ID(vars[i].name, 0.0)
                              : create_variable_long_ID(vars[i].name, 0);
        if (p->vid[i] < 0)
        {
            snprintf(err, errsz, "cannot create variable '%s'", vars[i].name);
            return -1;
        }
    }
    load_vars(p);
    return 0;
}

static int guard_on(prog_stmt *st, char *err, size_t errsz, int *on)
{
    *on = 1;
    if (st->guard == NULL)
    {
        return 0;
    }
    if (st->guard_kept != NULL)
    {
        if (cli_calc_fused_run(st->guard_kept) != 0)
        {
            snprintf(err, errsz, "%s in '%s'", cli_calc_lasterr, st->guard);
            return -1;
        }
    }
    else if (eval_text(st->guard, err, errsz) != 0)
    {
        return -1;
    }
    const VARIABLE *v = &data.core.variable[st->guard_vid];
    *on               = (v->type == 1) ? (v->value.l != 0) : (v->value.f != 0.0);
    return 0;
}

static int eval_for_begin(prog_stmt *st, char *err, size_t errsz, int *cont)
{
    if (st->guard_kept != NULL)
    {
        if (cli_calc_fused_run(st->guard_kept) != 0)
        {
            snprintf(err, errsz, "%s in '%s'", cli_calc_lasterr, st->guard);
            return -1;
        }
    }
    else if (eval_text(st->guard, err, errsz) != 0)
    {
        return -1;
    }
    if (st->for_step_kept != NULL)
    {
        if (cli_calc_fused_run(st->for_step_kept) != 0)
        {
            snprintf(err, errsz, "%s in '%s'", cli_calc_lasterr, st->for_step);
            return -1;
        }
    }
    else if (eval_text(st->for_step, err, errsz) != 0)
    {
        return -1;
    }
    const VARIABLE *vv   = &data.core.variable[st->for_var_vid];
    const VARIABLE *stop = &data.core.variable[st->guard_vid];
    const VARIABLE *step = &data.core.variable[st->for_step_vid];
    double          cur  = (vv->type == 1) ? (double) vv->value.l : vv->value.f;
    double          endv = (stop->type == 1) ? (double) stop->value.l : stop->value.f;
    double          stpv = (step->type == 1) ? (double) step->value.l : step->value.f;
    if (stpv == 0.0)
    {
        snprintf(err, errsz, "for loop step cannot be 0");
        return -1;
    }
    if (st->for_iters >= FOR_MAX_ITER)
    {
        snprintf(err, errsz, "for loop exceeded %d iterations", FOR_MAX_ITER);
        return -1;
    }
    *cont = (stpv > 0.0) ? (cur <= endv + 1e-12) : (cur >= endv - 1e-12);
    if (*cont)
    {
        st->for_iters++;
    }
    else
    {
        st->for_iters = 0;
    }
    return 0;
}

static void eval_for_end(cli_calc_prog *p, const prog_stmt *st)
{
    const prog_stmt *fst  = &p->st[st->jump_idx];
    VARIABLE        *vv   = &data.core.variable[fst->for_var_vid];
    const VARIABLE  *step = &data.core.variable[fst->for_step_vid];
    if (vv->type == 1 && step->type == 1)
    {
        vv->value.l += step->value.l;
    }
    else
    {
        double cur  = (vv->type == 1) ? (double) vv->value.l : vv->value.f;
        double stpv = (step->type == 1) ? (double) step->value.l : step->value.f;
        if (vv->type == 1 && stpv == (double) (long) stpv)
        {
            vv->value.l += (long) stpv;
        }
        else
        {
            vv->type    = 0;
            vv->value.f = cur + stpv;
        }
    }
    if (p->nvars == 0)
    {
        char numv[64];
        if (vv->type == 1)
        {
            snprintf(numv, sizeof(numv), "%ld", vv->value.l);
        }
        else
        {
            snprintf(numv, sizeof(numv), "%.17g", vv->value.f);
        }
        cli_var_set(fst->post, numv);
    }
}

static int post_image(const char *name, char *err, size_t errsz)
{
    imageID id = cli_calc_prog_connect_stream(name);
    if (id == -1)
    {
        snprintf(err, errsz, "post: image '%s' not found", name);
        return -1;
    }
    cli_calc_post_output(&data.core.image[id]);
    return 0;
}

/**
 * compile_multi_target_stmt - Warm-run and keep a multi-target unpacking statement `a, b = ...`
 */
static int compile_multi_target_stmt(cli_calc_prog *p, int i, char *err, size_t errsz)
{
    char mnames[8][STRINGMAXLEN_IMAGE_NAME];
    int  mloc[8] = { 0 };
    int  nm      = cli_calc_prog_parse_multi_targets(p->st[i].text, mnames, mloc, 8);
    if (nm < 2)
    {
        return 0;
    }
    if (eval_text(p->st[i].text, err, errsz) != 0)
    {
        return -1;
    }
    for (int j = 0; j < nm; j++)
    {
        imageID mid = image_ID(mnames[j], data.core.image, data.core.NB_MAX_IMAGE);
        if (mid != -1 && data.core.image[mid].md[0].shared != 1 && !mloc[j])
        {
            if (share(mnames[j], err, errsz) != 0)
            {
                return -1;
            }
        }
    }
    cli_calc_prog_default_shared = !p->st[i].is_local;
    p->st[i].kept                = cli_calc_keep_statement(p->st[i].text);
    cli_calc_prog_default_shared = 0;
    return 1;
}

static int compile_prog_stmt(cli_calc_prog          *p,
                             int                     i,
                             const cli_calc_binding *vars,
                             int                     nvars,
                             char                   *err,
                             size_t                  errsz)
{
    char name[STRINGMAXLEN_IMAGE_NAME];
    int  op = 0;
    int  b  = 0;
    if (cli_calc_prog_parse_prefix(p, i, err, errsz) != 0)
    {
        return -1;
    }
    if (p->st[i].guard != NULL)
    {
        p->st[i].guard_kept = cli_calc_keep_statement(p->st[i].guard);
    }
    if (p->st[i].for_step != NULL)
    {
        p->st[i].for_step_kept = cli_calc_keep_statement(p->st[i].for_step);
    }
    if (p->st[i].kind == STMT_FOR_BEGIN)
    {
        p->st[i].for_var_vid = variable_ID(p->st[i].post);
        if (p->st[i].for_var_vid < 0)
        {
            p->st[i].for_var_vid = create_variable_long_ID(p->st[i].post, 0);
        }
        for (int vb = 0; vb < nvars; vb++)
        {
            if (strcmp(vars[vb].name, p->st[i].post) == 0 && vars[vb].type != FPTYPE_STREAMNAME)
            {
                p->assigned[vb] = 1;
            }
        }
        return 0;
    }
    if (p->st[i].kind == STMT_SKIP || p->st[i].kind == STMT_POST || p->st[i].kind == STMT_NOP ||
        p->st[i].kind == STMT_IF_BEGIN || p->st[i].kind == STMT_ELSE_JUMP ||
        p->st[i].kind == STMT_FOR_END)
    {
        return 0;
    }
    if (cli_calc_prog_classify_target(p, i, p->st[i].text, vars, name, sizeof(name), &op, &b, err,
                                      errsz) != 0)
    {
        return -1;
    }
    cli_calc_prog_default_shared = !p->st[i].is_local;
    p->st[i].kept                = cli_calc_keep_statement(p->st[i].text);
    cli_calc_prog_default_shared = 0;
    note_why(&p->st[i]);
    if (p->st[i].kept == NULL && op == 0)
    {
        int mrc = compile_multi_target_stmt(p, i, err, errsz);
        if (mrc != 0)
        {
            return (mrc < 0) ? -1 : 0;
        }
    }
    if (p->st[i].kept != NULL || op != '=' || (b < nvars && vars[b].type != FPTYPE_STREAMNAME) ||
        image_ID(name, data.core.image, data.core.NB_MAX_IMAGE) != -1 ||
        (b == nvars && variable_ID(name) != -1))
    {
        return 0;
    }
    if (eval_text(p->st[i].text, err, errsz) != 0)
    {
        return -1;
    }
    imageID id = image_ID(name, data.core.image, data.core.NB_MAX_IMAGE);
    if (id != -1 && data.core.image[id].md[0].shared != 1 && !p->st[i].is_local)
    {
        if (share(name, err, errsz) != 0)
        {
            return -1;
        }
    }
    cli_calc_prog_default_shared = !p->st[i].is_local;
    p->st[i].kept                = cli_calc_keep_statement(p->st[i].text);
    cli_calc_prog_default_shared = 0;
    note_why(&p->st[i]);
    return 0;
}

/**
 * cli_calc_prog_new - Parse, bind, and pre-compile a multi-statement calculator program
 */
cli_calc_prog *cli_calc_prog_new(const char             *text,
                                 const cli_calc_binding *vars,
                                 int                     nvars,
                                 char                   *err,
                                 size_t                  errsz)
{
    cli_calc_prog *p = calloc(1, sizeof(*p));

    if (p == NULL)
    {
        return NULL;
    }
    if (vars != NULL)
    {
        cli_calc_udef_clear();
    }
    p->nvars = nvars;
    snprintf(err, errsz, "out of memory");
    if (bind_vars(p, vars, err, errsz) != 0)
    {
        cli_calc_prog_free(p);
        return NULL;
    }
    load_vars(p);
    char *rt = cli_calc_prog_rewrite_history(text, err, errsz);
    if (rt == NULL || cli_calc_prog_split(p, rt, err, errsz) != 0)
    {
        free(rt);
        cli_calc_prog_free(p);
        return NULL;
    }
    free(rt);
    for (int i = 0; i < p->n; i++)
    {
        if (compile_prog_stmt(p, i, vars, nvars, err, errsz) != 0)
        {
            cli_calc_prog_free(p);
            return NULL;
        }
    }
    cli_calc_prog_merge_basic_blocks(p);
    if (!cli_calc_prog_quiet)
    {
        print_not_kept_notes(p);
    }
    err[0] = '\0';
    return p;
}

static const char *stmt_kind_name(int kind)
{
    switch (kind)
    {
    case STMT_IF_BEGIN:
        return "if";
    case STMT_ELSE_JUMP:
        return "else";
    case STMT_FOR_BEGIN:
        return "for";
    case STMT_FOR_END:
        return "end of for";
    case STMT_SKIP:
        return "skip";
    case STMT_POST:
        return "post";
    default:
        return "control";
    }
}

/**
 * cli_calc_prog_explain - Compile @text as a unit would and print how each statement will run
 * @text: Program text (statements separated by newlines or `;`)
 * @out:  Stream for the report
 *
 * New targets are created as at a unit's start (their statements run once). The program is freed.
 *
 * Return: 0 when the program compiled, 1 otherwise (the error is printed).
 */
int cli_calc_prog_explain(const char *text, FILE *out)
{
    char err[512] = "";
    int  saved    = cli_calc_prog_quiet;

    cli_calc_prog_quiet = 1;
    cli_calc_prog *p    = cli_calc_prog_new(text, NULL, 0, err, sizeof(err));
    cli_calc_prog_quiet = saved;
    if (p == NULL)
    {
        fprintf(out, "milk.calc: %s\n", err);
        return 1;
    }
    int nkept = 0;
    int ntext = 0;
    for (int i = 0; i < p->n; i++)
    {
        const prog_stmt *st = &p->st[i];
        char             how[400];
        if (is_def_stmt(st->text))
        {
            snprintf(how, sizeof(how), "def, inlined where it is called");
        }
        else if (st->kind == STMT_NOP)
        {
            snprintf(how, sizeof(how), "kept, merged into the fused pass of statement %d",
                     st->merged_into);
            nkept++;
        }
        else if (st->kind != STMT_NORMAL)
        {
            snprintf(how, sizeof(how), "%s%s", stmt_kind_name(st->kind),
                     (st->guard != NULL)
                         ? ((st->guard_kept != NULL) ? ", guard kept as a scalar tape"
                                                     : ", guard as text")
                         : "");
        }
        else if (st->kept != NULL)
        {
            char d[300];
            cli_calc_kept_describe(st->kept, d, sizeof(d));
            snprintf(how, sizeof(how), "kept: %s", d);
            nkept++;
        }
        else
        {
            snprintf(how, sizeof(how), "NOT kept, evaluated as text each tick: %s", st->why);
            ntext++;
        }
        fprintf(out, "%3d  %s\n       %s\n", i + 1, st->text, how);
    }
    fprintf(out, "%d statement%s kept, %d evaluated as text\n", nkept, (nkept == 1) ? "" : "s",
            ntext);
    cli_calc_prog_free(p);
    return 0;
}

/**
 * step_control_stmt - Handle control-flow, guard, skip, and post statements
 *
 * Return: 1 if handled (caller should `continue`), 0 if normal statement to execute, -1 on error.
 */
static int step_control_stmt(cli_calc_prog *p,
                             prog_stmt     *st,
                             int           *i,
                             int           *skipped,
                             char          *err,
                             size_t         errsz)
{
    int on = 1;
    if (st->kind == STMT_NOP || (*skipped && st->kind != STMT_HIST))
    {
        return 1;
    }
    if (st->kind == STMT_IF_BEGIN)
    {
        if (guard_on(st, err, errsz, &on) != 0)
        {
            return -1;
        }
        if (!on)
        {
            *i = st->jump_idx - 1;
        }
        return 1;
    }
    if (st->kind == STMT_ELSE_JUMP)
    {
        *i = st->jump_idx - 1;
        return 1;
    }
    if (st->kind == STMT_FOR_BEGIN)
    {
        if (eval_for_begin(st, err, errsz, &on) != 0)
        {
            return -1;
        }
        if (!on)
        {
            *i = st->jump_idx - 1;
        }
        return 1;
    }
    if (st->kind == STMT_FOR_END)
    {
        eval_for_end(p, st);
        *i = st->jump_idx - 1;
        return 1;
    }
    if (guard_on(st, err, errsz, &on) != 0)
    {
        return -1;
    }
    if (!on)
    {
        return 1;
    }
    if (st->kind == STMT_SKIP)
    {
        *skipped = 1;
        return 1;
    }
    if (st->kind == STMT_POST)
    {
        return (post_image(st->post, err, errsz) != 0) ? -1 : 1;
    }
    return 0;
}

/**
 * cli_calc_prog_run - Execute all statements of a compiled calculator program for one frame
 */
int cli_calc_prog_run(cli_calc_prog *p, char *err, size_t errsz)
{
    int rc      = 0;
    int skipped = 0;

    for (int k = 0; k < p->n; k++)
    {
        p->st[k].for_iters = 0;
    }
    load_vars(p);
    cli_calc_post_defer(1);
    for (int i = 0; i < p->n; i++)
    {
        prog_stmt *st  = &p->st[i];
        int        ctl = step_control_stmt(p, st, &i, &skipped, err, errsz);
        if (ctl < 0)
        {
            rc = -1;
            break;
        }
        if (ctl > 0)
        {
            continue;
        }
        if (st->kept != NULL)
        {
            if (cli_calc_fused_run(st->kept) != 0)
            {
                snprintf(err, errsz, "%s in '%s'", cli_calc_lasterr, st->text);
                rc = -1;
                break;
            }
        }
        else if (eval_text(st->text, err, errsz) != 0)
        {
            rc = -1;
            break;
        }
    }
    cli_calc_post_flush(p->processinfo);
    if (rc == 0)
    {
        store_vars(p);
    }
    return rc;
}

/**
 * cli_calc_prog_exec_once - Parse and execute a calculator program once without keeping it
 */
int cli_calc_prog_exec_once(const char             *text,
                            const cli_calc_binding *vars,
                            int                     nvars,
                            unsigned char          *assigned_out,
                            char                   *err,
                            size_t                  errsz)
{
    cli_calc_prog *p = calloc(1, sizeof(*p));

    if (p == NULL)
    {
        snprintf(err, errsz, "out of memory");
        return -1;
    }
    if (vars != NULL)
    {
        cli_calc_udef_clear();
    }
    p->nvars = nvars;
    snprintf(err, errsz, "out of memory");
    if (bind_vars(p, vars, err, errsz) != 0 || cli_calc_prog_split(p, text, err, errsz) != 0)
    {
        cli_calc_prog_free(p);
        return -1;
    }
    for (int i = 0; i < p->n; i++)
    {
        if (cli_calc_prog_parse_prefix(p, i, err, errsz) != 0)
        {
            cli_calc_prog_free(p);
            return -1;
        }
        if (p->st[i].kind == STMT_FOR_BEGIN)
        {
            p->st[i].for_var_vid = variable_ID(p->st[i].post);
            if (p->st[i].for_var_vid < 0)
            {
                p->st[i].for_var_vid = create_variable_long_ID(p->st[i].post, 0);
            }
            for (int vb = 0; vb < nvars; vb++)
            {
                if (strcmp(vars[vb].name, p->st[i].post) == 0 && vars[vb].type != FPTYPE_STREAMNAME)
                {
                    p->assigned[vb] = 1;
                }
            }
        }
    }
    int skipped = 0;
    for (int i = 0; i < p->n && !skipped; i++)
    {
        char name[STRINGMAXLEN_IMAGE_NAME];
        int  op  = 0;
        int  b   = 0;
        int  ctl = step_control_stmt(p, &p->st[i], &i, &skipped, err, errsz);
        if (ctl < 0)
        {
            cli_calc_prog_free(p);
            return -1;
        }
        if (ctl > 0)
        {
            continue;
        }
        if (cli_calc_prog_classify_target(p, i, p->st[i].text, vars, name, sizeof(name), &op, &b,
                                          err, errsz) != 0 ||
            eval_text(p->st[i].text, err, errsz) != 0)
        {
            cli_calc_prog_free(p);
            return -1;
        }
        if (vars != NULL && op == '=' && (b == nvars || vars[b].type == FPTYPE_STREAMNAME) &&
            (b < nvars || variable_ID(name) == -1))
        {
            imageID id = image_ID(name, data.core.image, data.core.NB_MAX_IMAGE);
            if (id != -1 && data.core.image[id].md[0].shared != 1 && !p->st[i].is_local &&
                share(name, err, errsz) != 0)
            {
                cli_calc_prog_free(p);
                return -1;
            }
        }
    }
    store_vars(p);
    if (assigned_out != NULL && nvars > 0)
    {
        memcpy(assigned_out, p->assigned, (size_t) nvars);
    }
    cli_calc_prog_free(p);
    err[0] = '\0';
    return 0;
}
