// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "COREMOD_memory/delete_image.h"
#include "libmilkdata/milkdata.h"
#include "mc_ast.h"
#include "mcalc.h"
#include "mcalc_host.h"

int  mcalc_out_mode     = MCALC_OUT_VALUES;
int  mcalc_test_image   = 0;
int  mcalc_float_digits = 6;
char mcalc_lasterr[256] = "";

static mcalc_host_ops g_host     = { NULL, NULL };
static char          *g_calc_out = NULL;
static size_t         g_calc_sz  = 0;

/* Fallback string variable table when no host callbacks are registered */
#define MC_FB_VARS_MAX 128
typedef struct
{
    char name[128];
    char val[MC_TOK_STR_MAX];
    int  used;
} mc_fb_var_t;

static mc_fb_var_t g_fb_vars[MC_FB_VARS_MAX];

void mcalc_set_host(const mcalc_host_ops *ops)
{
    if (ops != NULL)
    {
        g_host = *ops;
    }
    else
    {
        g_host.str_get = NULL;
        g_host.str_set = NULL;
    }
}

const char *mc_host_str_get(const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return NULL;
    }
    if (g_host.str_get != NULL)
    {
        return g_host.str_get(name);
    }
    for (int i = 0; i < MC_FB_VARS_MAX; i++)
    {
        if (g_fb_vars[i].used && strcmp(g_fb_vars[i].name, name) == 0)
        {
            return g_fb_vars[i].val;
        }
    }
    return NULL;
}

void mc_host_str_set(const char *name, const char *val)
{
    if (name == NULL || name[0] == '\0' || val == NULL)
    {
        return;
    }
    if (g_host.str_set != NULL)
    {
        g_host.str_set(name, val);
        return;
    }
    int free_slot = -1;
    for (int i = 0; i < MC_FB_VARS_MAX; i++)
    {
        if (g_fb_vars[i].used && strcmp(g_fb_vars[i].name, name) == 0)
        {
            snprintf(g_fb_vars[i].val, sizeof(g_fb_vars[i].val), "%s", val);
            return;
        }
        if (!g_fb_vars[i].used && free_slot < 0)
        {
            free_slot = i;
        }
    }
    if (free_slot >= 0)
    {
        g_fb_vars[free_slot].used = 1;
        snprintf(g_fb_vars[free_slot].name, sizeof(g_fb_vars[free_slot].name), "%s", name);
        snprintf(g_fb_vars[free_slot].val, sizeof(g_fb_vars[free_slot].val), "%s", val);
    }
}

void mc_host_str_clear(void)
{
    memset(g_fb_vars, 0, sizeof(g_fb_vars));
}

/* User-defined inline function table (def f(x, ...) = body) */
static mcalc_udef_t g_udefs[MCALC_UDEF_MAX];
static int          g_nudefs = 0;

void mcalc_udef_clear(void)
{
    g_nudefs = 0;
}

int mcalc_udef_count(void)
{
    return g_nudefs;
}

const mcalc_udef_t *mcalc_udef_get(int idx)
{
    return (idx >= 0 && idx < g_nudefs) ? &g_udefs[idx] : NULL;
}

int mcalc_udef_add(const char *name, int nargs, const char arg_names[][64], const char *body)
{
    if (name == NULL || body == NULL || nargs < 1 || nargs > MCALC_UDEF_MAXARGS)
    {
        return -1;
    }
    int idx = -1;
    for (int i = 0; i < g_nudefs; i++)
    {
        if (strcmp(g_udefs[i].name, name) == 0)
        {
            idx = i;
            break;
        }
    }
    if (idx < 0)
    {
        if (g_nudefs >= MCALC_UDEF_MAX)
        {
            return -1;
        }
        idx = g_nudefs++;
    }
    snprintf(g_udefs[idx].name, sizeof(g_udefs[idx].name), "%s", name);
    g_udefs[idx].nargs = nargs;
    for (int i = 0; i < nargs; i++)
    {
        snprintf(g_udefs[idx].arg_names[i], sizeof(g_udefs[idx].arg_names[i]), "%s", arg_names[i]);
    }
    snprintf(g_udefs[idx].body, sizeof(g_udefs[idx].body), "%s", body);
    return idx;
}

int mcalc_udef_try_parse(const char *stmt)
{
    if (stmt == NULL)
    {
        return 0;
    }
    while (isspace((unsigned char) *stmt))
    {
        stmt++;
    }
    if (strncmp(stmt, "def", 3) != 0 || !isspace((unsigned char) stmt[3]))
    {
        return 0;
    }
    const char *p = stmt + 4;
    while (isspace((unsigned char) *p))
    {
        p++;
    }
    const char *nstart = p;
    while (isalnum((unsigned char) *p) || *p == '_')
    {
        p++;
    }
    size_t nlen = (size_t) (p - nstart);
    while (isspace((unsigned char) *p))
    {
        p++;
    }
    if (nlen == 0 || nlen >= 64 || *p != '(')
    {
        return -1;
    }
    char fname[64];
    memcpy(fname, nstart, nlen);
    fname[nlen] = '\0';
    p++;
    char args[MCALC_UDEF_MAXARGS][64];
    int  nargs = 0;
    while (*p != '\0' && *p != ')')
    {
        while (isspace((unsigned char) *p))
        {
            p++;
        }
        const char *astart = p;
        while (isalnum((unsigned char) *p) || *p == '_')
        {
            p++;
        }
        size_t alen = (size_t) (p - astart);
        while (isspace((unsigned char) *p))
        {
            p++;
        }
        if (alen == 0 || alen >= 64 || nargs >= MCALC_UDEF_MAXARGS || (*p != ',' && *p != ')'))
        {
            return -1;
        }
        memcpy(args[nargs], astart, alen);
        args[nargs][alen] = '\0';
        nargs++;
        if (*p == ',')
        {
            p++;
        }
    }
    if (*p != ')' || nargs == 0)
    {
        return -1;
    }
    p++;
    while (isspace((unsigned char) *p))
    {
        p++;
    }
    if (*p != '=' || p[1] == '=')
    {
        return -1;
    }
    p++;
    while (isspace((unsigned char) *p))
    {
        p++;
    }
    if (*p == '\0')
    {
        return -1;
    }
    return (mcalc_udef_add(fname, nargs, (const char (*)[64]) args, p) >= 0) ? 1 : -1;
}

static void report_err(const char *input, const char *msg)
{
    const char *m = (msg != NULL && msg[0] != '\0') ? msg : "syntax error";
    snprintf(mcalc_lasterr, sizeof(mcalc_lasterr), "%s", m);
    if (input != NULL && input[0] != '\0')
    {
        fprintf(stderr, "milk: calc error: %s in '%s'\n", m, input);
    }
    else
    {
        fprintf(stderr, "milk: calc error: %s\n", m);
    }
}

static void strip_arg_quotes(const char *lp, char *out, size_t outsz)
{
    while (*lp == ' ' || *lp == '\t')
    {
        lp++;
    }
    size_t len = strlen(lp);
    while (len > 0 && isspace((unsigned char) lp[len - 1]))
    {
        len--;
    }
    if (len >= 2 &&
        ((lp[0] == '"' && lp[len - 1] == '"') || (lp[0] == '\'' && lp[len - 1] == '\'')))
    {
        lp++;
        len -= 2;
    }
    if (len >= outsz)
    {
        len = outsz - 1;
    }
    memcpy(out, lp, len);
    out[len] = '\0';
}

static int eval_special_directives(const char *input, int is_test, int *out_rc)
{
    const char *lp = input;
    while (*lp == ' ' || *lp == '\t')
    {
        lp++;
    }
    if (strncmp(lp, "loadlib", 7) == 0 && isspace((unsigned char) lp[7]))
    {
        char lpath[1024];
        char lerr[512] = "";
        strip_arg_quotes(lp + 7, lpath, sizeof(lpath));
        if (mcalc_ext_load_list(lpath, lerr, sizeof(lerr)) != 0)
        {
            report_err(input, lerr[0] != '\0' ? lerr : "failed to load extension library");
            return 0;
        }
        if (out_rc != NULL)
        {
            *out_rc = 0;
        }
        return 1;
    }
    if ((strncmp(lp, "__list_engines", 14) == 0 &&
         (lp[14] == '\0' || isspace((unsigned char) lp[14]))) ||
        (strncmp(lp, "listengines", 11) == 0 &&
         (lp[11] == '\0' || isspace((unsigned char) lp[11]))))
    {
        lp += (lp[0] == '_') ? 14 : 11;
        while (*lp == ' ' || *lp == '\t')
        {
            lp++;
        }
        int jmode = 0;
        if (strncmp(lp, "--json", 6) == 0 && (lp[6] == '\0' || isspace((unsigned char) lp[6])))
        {
            jmode = 1;
            lp += 6;
        }
        char pat[256];
        strip_arg_quotes(lp, pat, sizeof(pat));
        int lrc = mcalc_print_catalog(pat, jmode);
        if (out_rc != NULL)
        {
            *out_rc = lrc;
        }
        return 1;
    }
    int udef_rc = mcalc_udef_try_parse(input);
    if (udef_rc != 0)
    {
        if (udef_rc < 0)
        {
            report_err(input, "invalid def syntax (expected: def f(x, ...) = expr)");
            return 0;
        }
        if (out_rc != NULL)
        {
            *out_rc = 0;
        }
        return 1;
    }
    if (!is_test &&
        ((strncmp(lp, "if", 2) == 0 && (isspace((unsigned char) lp[2]) || lp[2] == '(')) ||
         (strncmp(lp, "for", 3) == 0 && isspace((unsigned char) lp[3]))))
    {
        char perr[256] = "";
        if (mcalc_prog_exec_once(lp, NULL, 0, NULL, perr, sizeof(perr)) != 0)
        {
            report_err(input, perr[0] != '\0' ? perr : "block execution error");
            return 0;
        }
        if (mcalc_out_mode == MCALC_OUT_BIND)
        {
            printf("\n");
            fflush(stdout);
        }
        if (out_rc != NULL)
        {
            *out_rc = 0;
        }
        return 1;
    }
    return -1;
}

static void emit_scalar_result(const mc_val_t *result, int is_assign_op, const char *lhs_name)
{
    char val_buf[MC_TOK_STR_MAX + 32] = "";
    if (result->type == MC_VAL_LONG)
    {
        snprintf(val_buf, sizeof(val_buf), "%ld", result->lval);
    }
    else if (result->type == MC_VAL_DOUBLE)
    {
        snprintf(val_buf, sizeof(val_buf), "%.*g", mcalc_float_digits, result->dval);
    }
    else if (result->type == MC_VAL_STRING)
    {
        const char *cv = mc_host_str_get(result->sval);
        snprintf(val_buf, sizeof(val_buf), "%s", (cv != NULL) ? cv : result->sval);
    }

    if (mcalc_out_mode == MCALC_OUT_BIND || mcalc_out_mode == MCALC_OUT_PIXELS)
    {
        printf("%s\n", val_buf);
        fflush(stdout);
    }
    else if (mcalc_out_mode == MCALC_OUT_REPL)
    {
        if (is_assign_op && lhs_name != NULL && lhs_name[0] != '\0')
        {
            printf("%s %s\n", lhs_name, val_buf);
            fflush(stdout);
        }
        else if (result->type == MC_VAL_LONG || result->type == MC_VAL_DOUBLE ||
                 result->type == MC_VAL_STRING)
        {
            printf("%s\n", val_buf);
            fflush(stdout);
        }
    }
    else if (!is_assign_op)
    {
        if (g_calc_out != NULL)
        {
            snprintf(g_calc_out, g_calc_sz, "%s", val_buf);
        }
        else if (result->type == MC_VAL_LONG || result->type == MC_VAL_DOUBLE ||
                 result->type == MC_VAL_STRING)
        {
            printf("%s\n", val_buf);
            fflush(stdout);
        }
    }
}

int mcalc_eval_bash(const char *input, int is_test, int *out_rc)
{
    mcalc_test_image = 0;
    if (out_rc != NULL)
    {
        *out_rc = 2;
    }
    if (input == NULL)
    {
        return 0;
    }

    const char *env_libs = getenv("MILK_CALC_LIBS");
    if (env_libs != NULL && env_libs[0] != '\0')
    {
        char lerr[256] = "";
        if (mcalc_ext_load_list(env_libs, lerr, sizeof(lerr)) != 0)
        {
            report_err(input, lerr);
            return 0;
        }
    }

    int drc = eval_special_directives(input, is_test, out_rc);
    if (drc >= 0)
    {
        return drc;
    }

    mcalc_lasterr[0]  = '\0';
    mc_tok_list_t *tl = calloc(1, sizeof(mc_tok_list_t));
    if (tl == NULL)
    {
        report_err(input, "out of memory");
        return 0;
    }
    if (mc_lex(input, tl) != 0)
    {
        report_err(input, tl->errmsg[0] != '\0' ? tl->errmsg : "syntax error");
        free(tl);
        return 0;
    }

    int p0 = 0;
    while (p0 < tl->ntok &&
           (tl->toks[p0].type == MC_TOK_NEWLINE || tl->toks[p0].type == MC_TOK_SEMI))
    {
        p0++;
    }
    if (p0 >= tl->ntok || tl->toks[p0].type == MC_TOK_EOF)
    {
        free(tl);
        if (out_rc != NULL)
        {
            *out_rc = is_test ? 1 : 0;
        }
        return 1;
    }

    char       err[256] = "";
    mc_stmt_t *s        = mc_parse_stmt(tl, err, sizeof(err));
    free(tl);
    if (s == NULL)
    {
        report_err(input, err[0] != '\0' ? err : "syntax error");
        return 0;
    }

    if (mc_resolve_stmt(s, err, sizeof(err)) != 0)
    {
        mc_stmt_free(s);
        report_err(input, err[0] != '\0' ? err : "evaluation error");
        return 0;
    }

    if (is_test && s->kind == MC_STMT_EXPR && s->u.expr != NULL &&
        s->u.expr->res_kind == MC_RES_IMAGE)
    {
        mc_stmt_free(s);
        fprintf(
            stderr,
            "milk: calc error: the condition is an image in '%s': test imin(...) for all pixels, "
            "imax(...) for any\n",
            input);
        snprintf(mcalc_lasterr, sizeof(mcalc_lasterr), "the condition is an image");
        mcalc_test_image = 1;
        return 0;
    }

    mc_val_t result;
    memset(&result, 0, sizeof(result));
    if (mc_scalar_exec_stmt(s, &result, err, sizeof(err)) != 0)
    {
        mc_stmt_free(s);
        report_err(input, err[0] != '\0' ? err : "evaluation error");
        return 0;
    }

    if (is_test)
    {
        int truthy = 0;
        if (result.type == MC_VAL_LONG)
        {
            truthy = (result.lval != 0);
        }
        else if (result.type == MC_VAL_DOUBLE)
        {
            truthy = (result.dval != 0.0 && !isnan(result.dval));
        }
        else if (result.type == MC_VAL_STRING)
        {
            truthy = (result.sval[0] != '\0' && strcmp(result.sval, "0") != 0);
        }
        mc_stmt_free(s);
        if (out_rc != NULL)
        {
            *out_rc = truthy ? 0 : 1;
        }
        return 1;
    }

    int  is_assign     = (s->kind == MC_STMT_MULTI_ASSIGN);
    char lhs_name[256] = "";
    if (s->kind == MC_STMT_EXPR && s->u.expr != NULL && s->u.expr->kind == MC_EXPR_ASSIGN)
    {
        is_assign             = 1;
        const mc_expr_t *tget = s->u.expr->u.assign.target;
        if (tget->kind == MC_EXPR_IDENT)
        {
            snprintf(lhs_name, sizeof(lhs_name), "%s",
                     tget->u.ident.bare[0] != '\0' ? tget->u.ident.bare : tget->u.ident.name);
        }
        else if (tget->kind == MC_EXPR_SLICE)
        {
            snprintf(lhs_name, sizeof(lhs_name), "%s", tget->u.slice.name);
        }
    }

    if (s->kind == MC_STMT_MULTI_ASSIGN)
    {
        if (mcalc_out_mode == MCALC_OUT_BIND)
        {
            printf("\n");
            fflush(stdout);
        }
    }
    else
    {
        emit_scalar_result(&result, is_assign, lhs_name);
    }

    mc_stmt_free(s);
    if (out_rc != NULL)
    {
        *out_rc = 0;
    }
    return 1;
}

int mcalc_eval_str(const char *input, int is_test, int *out_rc, char *out, size_t outsz)
{
    if (out != NULL && outsz > 0)
    {
        out[0] = '\0';
    }
    g_calc_out = out;
    g_calc_sz  = outsz;
    int r      = mcalc_eval_bash(input, is_test, out_rc);
    g_calc_out = NULL;
    g_calc_sz  = 0;
    return r;
}

int mcalc_eval_math_to_val(const char *input, int *out_type, long *out_lval, double *out_dval)
{
    if (input == NULL)
    {
        return 0;
    }
    mc_tok_list_t *tl = calloc(1, sizeof(mc_tok_list_t));
    if (tl == NULL)
    {
        return 0;
    }
    if (mc_lex(input, tl) != 0)
    {
        free(tl);
        return 0;
    }
    char       err[256] = "";
    mc_stmt_t *s        = mc_parse_stmt(tl, err, sizeof(err));
    free(tl);
    if (s == NULL || s->kind != MC_STMT_EXPR || s->u.expr == NULL)
    {
        mc_stmt_free(s);
        return 0;
    }
    if (mc_resolve_stmt(s, err, sizeof(err)) != 0 || s->needs_image)
    {
        mc_stmt_free(s);
        return 0;
    }
    mc_val_t v;
    if (mc_scalar_eval_expr(s->u.expr, &v, err, sizeof(err)) != 0 ||
        (v.type != MC_VAL_LONG && v.type != MC_VAL_DOUBLE))
    {
        mc_stmt_free(s);
        return 0;
    }
    mc_stmt_free(s);
    if (v.type == MC_VAL_LONG)
    {
        if (out_type != NULL)
        {
            *out_type = 1;
        }
        if (out_lval != NULL)
        {
            *out_lval = v.lval;
        }
        if (out_dval != NULL)
        {
            *out_dval = (double) v.lval;
        }
    }
    else
    {
        if (out_type != NULL)
        {
            *out_type = 2;
        }
        if (out_lval != NULL)
        {
            *out_lval = (long) v.dval;
        }
        if (out_dval != NULL)
        {
            *out_dval = v.dval;
        }
    }
    return 1;
}

void mcalc_tmp_cleanup(void)
{
    if (milk_data.image == NULL || milk_data.NB_MAX_IMAGE <= 0)
    {
        return;
    }
    for (long i = 0; i < milk_data.NB_MAX_IMAGE; i++)
    {
        if (milk_data.image[i].used && strncmp(milk_data.image[i].name, "_tmpcalc", 8) == 0)
        {
            delete_image_ID(milk_data.image[i].name, DELETE_IMAGE_ERRMODE_WARNING);
        }
    }
}

/* Phase 4 stubs for C extensions and compiled programs (implemented in Phases 6-7) */

int mcalc_ext_load(const char *path, char *err, size_t errsz)
{
    (void) path;
    if (err != NULL && errsz > 0)
    {
        snprintf(err, errsz, "not implemented in the new engine");
    }
    return -1;
}

int mcalc_ext_load_list(const char *libs_str, char *err, size_t errsz)
{
    if (libs_str == NULL || libs_str[0] == '\0')
    {
        return 0;
    }
    if (err != NULL && errsz > 0)
    {
        snprintf(err, errsz, "not implemented in the new engine");
    }
    return -1;
}

int mcalc_ext_count(void)
{
    return milk_calc_ext_count();
}

const milk_calc_ext_entry_t *mcalc_ext_get(int idx)
{
    return milk_calc_ext_get(idx);
}

void mcalc_ext_scan_source(const char               *src,
                           mcalc_ext_param_cb_t      param_cb,
                           mcalc_ext_scalar_out_cb_t scalar_out_cb,
                           void                     *ud)
{
    (void) src;
    (void) param_cb;
    (void) scalar_out_cb;
    (void) ud;
}

void mcalc_prog_attach(void)
{
}

mcalc_prog *mcalc_prog_new(const char          *text,
                           const mcalc_binding *vars,
                           int                  nvars,
                           char                *err,
                           size_t               errsz)
{
    (void) text;
    (void) vars;
    (void) nvars;
    if (err != NULL && errsz > 0)
    {
        snprintf(err, errsz, "not implemented in the new engine");
    }
    return NULL;
}

int mcalc_prog_assigns(const mcalc_prog *p, int i)
{
    (void) p;
    (void) i;
    return 0;
}

int mcalc_prog_run(mcalc_prog *p, char *err, size_t errsz)
{
    (void) p;
    if (err != NULL && errsz > 0)
    {
        snprintf(err, errsz, "not implemented in the new engine");
    }
    return -1;
}

uint64_t mcalc_torn_count(void)
{
    return 0;
}

void mcalc_prog_set_processinfo(mcalc_prog *p, void *processinfo)
{
    (void) p;
    (void) processinfo;
}

int mcalc_prog_exec_once(const char          *text,
                         const mcalc_binding *vars,
                         int                  nvars,
                         unsigned char       *assigned_out,
                         char                *err,
                         size_t               errsz)
{
    (void) assigned_out;
    if (vars != NULL || nvars != 0 || text == NULL)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "not implemented in the new engine");
        }
        return -1;
    }
    mc_tok_list_t *tl = calloc(1, sizeof(mc_tok_list_t));
    if (tl == NULL)
    {
        return -1;
    }
    if (mc_lex(text, tl) != 0)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "%s", tl->errmsg[0] != '\0' ? tl->errmsg : "syntax error");
        }
        free(tl);
        return -1;
    }
    mc_stmt_t *prog = mc_parse_prog(tl, err, errsz);
    free(tl);
    if (prog == NULL)
    {
        return -1;
    }
    if (mc_resolve_prog(prog, err, errsz) != 0)
    {
        mc_stmt_free(prog);
        return -1;
    }
    int rc = mc_scalar_exec_prog(prog, err, errsz);
    mc_stmt_free(prog);
    return rc;
}

void mcalc_prog_free(mcalc_prog *p)
{
    (void) p;
}

int mcalc_prog_explain(const char *text, FILE *out)
{
    (void) text;
    if (out != NULL)
    {
        fprintf(out, "milk.calc: not implemented in the new engine\n");
    }
    return 1;
}
