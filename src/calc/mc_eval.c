// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libmilkdata/milkdata.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/variable_ID.h"
#include "mc_ast.h"
#include "mc_plan.h"
#include "mcalc.h"
#include "mcalc_host.h"

int  mcalc_out_mode     = MCALC_OUT_VALUES;
int  mcalc_test_image   = 0;
int  mcalc_float_digits = 15;
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

/* 64-entry LRU statement cache for single-statement image programs in `@calc` */
#define MC_STMT_CACHE_MAX 64
typedef struct
{
    char          text[1024];
    uint64_t      hash;
    size_t        len;
    int           udef_cnt;
    uint64_t      table_gen;
    uint64_t      lru_tick;
    int           is_im_stmt;
    int           is_assign;
    int           is_multi;
    char          lhs_name[256];
    mcalc_prog_t *prog;
} mc_stmt_cache_t;

static mc_stmt_cache_t g_stmt_cache[MC_STMT_CACHE_MAX];
static uint64_t        g_stmt_lru_clock = 0;

static void stmt_cache_clear(void)
{
    for (int i = 0; i < MC_STMT_CACHE_MAX; i++)
    {
        if (g_stmt_cache[i].prog != NULL)
        {
            mcalc_prog_free(g_stmt_cache[i].prog);
            g_stmt_cache[i].prog = NULL;
        }
        g_stmt_cache[i].len = 0;
    }
}

void mcalc_set_host(const mcalc_host_ops *ops)
{
    g_host = (ops != NULL) ? *ops : (mcalc_host_ops) { NULL, NULL };
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
        (void) (!g_fb_vars[i].used && free_slot < 0 && (free_slot = i));
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
    stmt_cache_clear();
}

int mc_host_has_str_sink(void)
{
    return g_host.str_set != NULL;
}

/* User-defined inline function table (def f(x, ...) = body) */
static mcalc_udef_t g_udefs[MCALC_UDEF_MAX];
static int          g_nudefs = 0;

void mcalc_udef_clear(void)
{
    g_nudefs = 0;
    stmt_cache_clear();
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
    for (int i = 0; i < g_nudefs && idx < 0; i++)
    {
        (void) (strcmp(g_udefs[i].name, name) == 0 && (idx = i));
    }
    if (idx < 0 && (idx = (g_nudefs < MCALC_UDEF_MAX) ? g_nudefs++ : -1) < 0)
    {
        return -1;
    }
    snprintf(g_udefs[idx].name, sizeof(g_udefs[idx].name), "%s", name);
    g_udefs[idx].nargs = nargs;
    for (int i = 0; i < nargs; i++)
    {
        snprintf(g_udefs[idx].arg_names[i], sizeof(g_udefs[idx].arg_names[i]), "%s", arg_names[i]);
    }
    snprintf(g_udefs[idx].body, sizeof(g_udefs[idx].body), "%s", body);
    stmt_cache_clear();
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
    char fname[64], args[MCALC_UDEF_MAXARGS][64];
    int  nargs = 0;
    memcpy(fname, nstart, nlen);
    fname[nlen] = '\0';
    p++;
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
        args[nargs++][alen] = '\0';
        p += (*p == ',');
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
    return (*p == '\0')
               ? -1
               : ((mcalc_udef_add(fname, nargs, (const char (*)[64]) args, p) >= 0) ? 1 : -1);
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

static int emit_image_result(imageID im_id, int is_assign, int is_multi, const char *lhs_name)
{
    if (is_multi || mcalc_out_mode == MCALC_OUT_BIND)
    {
        (void) (mcalc_out_mode == MCALC_OUT_BIND && (printf("\n"), fflush(stdout), 1));
        return 0;
    }
    if (mcalc_out_mode == MCALC_OUT_PIXELS)
    {
        char *pixels = mc_format_pixel_line(im_id);
        if (pixels == NULL)
        {
            snprintf(mcalc_lasterr, sizeof(mcalc_lasterr), "pixel format error");
            return -1;
        }
        printf("%s\n", pixels), fflush(stdout), free(pixels);
        return 0;
    }
    if (mcalc_out_mode == MCALC_OUT_REPL)
    {
        char summary[256] = "";
        mc_format_image_summary(im_id, is_assign ? lhs_name : NULL, summary, sizeof(summary));
        printf("%s\n", summary), fflush(stdout);
        return 0;
    }
    if (!is_assign && im_id >= 0 && milk_data.image != NULL && milk_data.image[im_id].used)
    {
        const char *nm = milk_data.image[im_id].name;
        (void) ((g_calc_out != NULL) ? snprintf(g_calc_out, g_calc_sz, "%s", nm)
                                     : (printf("%s\n", nm), fflush(stdout), 0));
    }
    return 0;
}

static int expr_has_shadowed_var(const mc_expr_t *e)
{
    int r = (e != NULL) &&
            ((e->kind == MC_EXPR_IDENT && e->u.ident.id_kind == MC_ID_VAR &&
              image_ID(e->u.ident.bare, milk_data.image, milk_data.NB_MAX_IMAGE) >= 0) ||
             (e->kind == MC_EXPR_UNARY && expr_has_shadowed_var(e->u.unary.operand)) ||
             (e->kind == MC_EXPR_BINARY && (expr_has_shadowed_var(e->u.binary.left) ||
                                            expr_has_shadowed_var(e->u.binary.right))) ||
             (e->kind == MC_EXPR_TERNARY && (expr_has_shadowed_var(e->u.ternary.cond) ||
                                             expr_has_shadowed_var(e->u.ternary.t_expr) ||
                                             expr_has_shadowed_var(e->u.ternary.f_expr))) ||
             (e->kind == MC_EXPR_ASSIGN && expr_has_shadowed_var(e->u.assign.rhs)));
    for (int i = 0; e && e->kind == MC_EXPR_CALL && i < e->u.call.nargs && !r; i++)
    {
        r = expr_has_shadowed_var(e->u.call.args[i]);
    }
    return r || (e && e->kind == MC_EXPR_CALL && expr_has_shadowed_var(e->u.call.udef_body));
}

static int validate_cached_prog(mcalc_prog_t *p, int same_gen)
{
    for (const mc_stmt_t *st = same_gen ? NULL : p->ast_head; st != NULL; st = st->next)
    {
        if (st->kind == MC_STMT_EXPR && expr_has_shadowed_var(st->u.expr))
        {
            return 0;
        }
    }
    for (int i = 0; i < p->nslots; i++)
    {
        mc_slot_t *s = &p->slots[i];
        imageID    id =
            (s->im_id >= 0 && s->im_id < milk_data.NB_MAX_IMAGE && milk_data.image[s->im_id].used &&
             (same_gen || strcmp(milk_data.image[s->im_id].name, s->bare) == 0))
                ? s->im_id
                : mc_resolve_or_autoconnect_image(s->name);
        long vid = (same_gen || id >= 0) ? -1 : variable_ID(s->bare);
        if (vid >= 0 && milk_data.variable[vid].type != 3)
        {
            return 0;
        }
        if ((id >= 0) != (s->existed_at_plan != 0))
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

static int try_stmt_cache_hit(const char *input, uint64_t in_hash, size_t in_len, int *out_rc)
{
    int ucnt = mcalc_udef_count();
    for (int i = 0; i < MC_STMT_CACHE_MAX; i++)
    {
        mc_stmt_cache_t *e = &g_stmt_cache[i];
        if (e->prog == NULL || e->hash != in_hash || e->len != in_len || e->udef_cnt != ucnt ||
            strcmp(e->text, input) != 0)
        {
            continue;
        }
        int same_gen = (e->table_gen == milk_table_generation() && e->table_gen != 0);
        if ((!e->is_im_stmt && !same_gen) || !validate_cached_prog(e->prog, same_gen))
        {
            mcalc_prog_free(e->prog);
            e->prog = NULL, e->len = 0;
            return 0;
        }
        char err[256] = "";
        e->lru_tick   = ++g_stmt_lru_clock;
        if (mcalc_prog_run(e->prog, err, sizeof(err)) != 0)
        {
            report_err(input, err[0] != '\0' ? err : "evaluation error");
            return -1;
        }
        e->table_gen = milk_table_generation();
        if (e->is_im_stmt)
        {
            if (emit_image_result(e->prog->last_im_id, e->is_assign, e->is_multi, e->lhs_name) != 0)
            {
                return -1;
            }
        }
        else if (e->is_multi)
        {
            if (mcalc_out_mode == MCALC_OUT_BIND)
            {
                printf("\n"), fflush(stdout);
            }
        }
        else
        {
            emit_scalar_result(&e->prog->last_scalar, e->is_assign, e->lhs_name);
        }
        (void) (out_rc != NULL && (*out_rc = 0, 1));
        return 1;
    }
    return 0;
}

static void store_stmt_cache(const char   *input,
                             uint64_t      in_hash,
                             size_t        in_len,
                             int           is_im_stmt,
                             int           is_assign,
                             int           is_multi,
                             const char   *lhs_name,
                             mcalc_prog_t *prog)
{
    if (in_len == 0 || in_len >= sizeof(g_stmt_cache[0].text))
    {
        mcalc_prog_free(prog);
        return;
    }
    int victim = 0;
    for (int i = 0; i < MC_STMT_CACHE_MAX; i++)
    {
        if (g_stmt_cache[i].prog == NULL)
        {
            victim = i;
            break;
        }
        if (g_stmt_cache[i].lru_tick < g_stmt_cache[victim].lru_tick)
        {
            victim = i;
        }
    }
    mc_stmt_cache_t *e = &g_stmt_cache[victim];
    mcalc_prog_free(e->prog);
    memcpy(e->text, input, in_len + 1);
    e->hash = in_hash, e->len = in_len, e->udef_cnt = mcalc_udef_count();
    e->table_gen = milk_table_generation(), e->lru_tick = ++g_stmt_lru_clock;
    e->is_im_stmt = is_im_stmt, e->is_assign = is_assign, e->is_multi = is_multi;
    snprintf(e->lhs_name, sizeof(e->lhs_name), "%s", lhs_name != NULL ? lhs_name : "");
    e->prog = prog;
}

static int eval_bash_inner(const char *input, int is_test, int *out_rc)
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

    mcalc_lasterr[0] = '\0';
    uint64_t in_hash = 14695981039346656037ULL;
    size_t   in_len  = 0;
    for (const unsigned char *p = (const unsigned char *) input; *p != '\0'; p++)
    {
        in_hash = (in_hash ^ (uint64_t) *p) * 1099511628211ULL;
        in_len++;
    }
    if (!is_test && in_len > 0 && in_len < sizeof(g_stmt_cache[0].text))
    {
        int hit = try_stmt_cache_hit(input, in_hash, in_len, out_rc);
        if (hit != 0)
        {
            return (hit > 0) ? 1 : 0;
        }
    }

    static _Thread_local mc_tok_list_t s_tl;
    mc_tok_list_t                     *tl = &s_tl;
    if (mc_lex(input, tl) != 0)
    {
        report_err(input, tl->errmsg[0] != '\0' ? tl->errmsg : "syntax error");
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
        if (out_rc != NULL)
        {
            *out_rc = is_test ? 1 : 0;
        }
        return 1;
    }

    char       err[256] = "";
    mc_stmt_t *s        = mc_parse_stmt(tl, err, sizeof(err));
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

    int  is_assign = (s->kind == MC_STMT_MULTI_ASSIGN), is_multi = is_assign;
    char lhs_name[256] = "";
    if (s->kind == MC_STMT_EXPR && s->u.expr != NULL && s->u.expr->kind == MC_EXPR_ASSIGN)
    {
        is_assign             = 1;
        const mc_expr_t *tget = s->u.expr->u.assign.target;
        snprintf(lhs_name, sizeof(lhs_name), "%s",
                 (tget->kind == MC_EXPR_IDENT)
                     ? (tget->u.ident.bare[0] != '\0' ? tget->u.ident.bare : tget->u.ident.name)
                     : ((tget->kind == MC_EXPR_SLICE) ? tget->u.slice.name : ""));
    }
    else if (s->kind == MC_STMT_EXPR && s->u.expr != NULL && s->u.expr->kind == MC_EXPR_CALL &&
             s->u.expr->u.call.fn != NULL && s->u.expr->u.call.nargs >= 2)
    {
        const char      *fnm  = s->u.expr->u.call.fn->name;
        const mc_expr_t *last = s->u.expr->u.call.args[s->u.expr->u.call.nargs - 1];
        if (strcmp(fnm, "gemm") == 0 || strcmp(fnm, "gemv") == 0 || strcmp(fnm, "symv") == 0 ||
            strcmp(fnm, "syrk") == 0 || strcmp(fnm, "ger") == 0 || strcmp(fnm, "axpy") == 0 ||
            strcmp(fnm, "scal") == 0)
        {
            is_assign = 1;
            (void) (last->kind == MC_EXPR_IDENT &&
                    snprintf(lhs_name, sizeof(lhs_name), "%s", last->u.ident.bare));
        }
    }

    int is_im_stmt = (s->kind == MC_STMT_EXPR && s->u.expr != NULL &&
                      (s->u.expr->res_kind == MC_RES_IMAGE ||
                       (s->u.expr->kind == MC_EXPR_ASSIGN &&
                        s->u.expr->u.assign.target->res_kind == MC_RES_IMAGE)));
    for (int j = 0; s->kind == MC_STMT_MULTI_ASSIGN && j < s->u.multi.nlhs; j++)
    {
        const mc_expr_t *lhs = s->u.multi.lhs[j];
        is_im_stmt |= (lhs->u.ident.id_kind == MC_ID_IMAGE || lhs->u.ident.id_kind == MC_ID_NEW ||
                       (j < s->u.multi.nrhs && s->u.multi.rhs[j]->res_kind == MC_RES_IMAGE));
    }

    if (is_test && is_im_stmt)
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

    if (is_im_stmt || (s->needs_image && !is_test))
    {
        if (is_im_stmt && s->kind == MC_STMT_EXPR && s->u.expr->kind == MC_EXPR_IDENT)
        {
            imageID id = s->u.expr->u.ident.im_id;
            mc_stmt_free(s);
            if (emit_image_result(id, 0, 0, "") != 0)
            {
                return 0;
            }
            (void) (out_rc != NULL && (*out_rc = 0, 1));
            return 1;
        }
        mc_stmt_free(s);
        mcalc_prog_t *prog = mcalc_prog_new(input, NULL, 0, err, sizeof(err));
        if (prog == NULL || mcalc_prog_run(prog, err, sizeof(err)) != 0)
        {
            mcalc_prog_free(prog);
            report_err(input, err[0] != '\0' ? err : "evaluation error");
            return 0;
        }
        imageID  res_im  = prog->last_im_id;
        mc_val_t res_val = prog->last_scalar;
        store_stmt_cache(input, in_hash, in_len, is_im_stmt, is_assign, is_multi, lhs_name, prog);
        if (is_im_stmt)
        {
            if (emit_image_result(res_im, is_assign, is_multi, lhs_name) != 0)
            {
                return 0;
            }
        }
        else if (is_multi)
        {
            if (mcalc_out_mode == MCALC_OUT_BIND)
            {
                printf("\n"), fflush(stdout);
            }
        }
        else
        {
            emit_scalar_result(&res_val, is_assign, lhs_name);
        }
        (void) (out_rc != NULL && (*out_rc = 0, 1));
        return 1;
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
        int truthy = (result.type == MC_VAL_LONG) ? (result.lval != 0)
                     : (result.type == MC_VAL_DOUBLE)
                         ? (result.dval != 0.0 && !mc_isnan(result.dval))
                         : (result.type == MC_VAL_STRING && result.sval[0] != '\0' &&
                            strcmp(result.sval, "0") != 0);
        mc_stmt_free(s);
        (void) (out_rc != NULL && (*out_rc = truthy ? 0 : 1, 1));
        return 1;
    }

    if (s->kind == MC_STMT_MULTI_ASSIGN)
    {
        (void) (mcalc_out_mode == MCALC_OUT_BIND && (printf("\n"), fflush(stdout), 1));
    }
    else
    {
        emit_scalar_result(&result, is_assign, lhs_name);
    }

    mc_stmt_free(s);
    (void) (out_rc != NULL && (*out_rc = 0, 1));
    return 1;
}

int mcalc_eval_bash(const char *input, int is_test, int *out_rc)
{
    mc_scalar_sync_host(1);
    int r = eval_bash_inner(input, is_test, out_rc);
    mc_scalar_sync_host(-1);
    return r;
}

int mcalc_eval_str(const char *input, int is_test, int *out_rc, char *out, size_t outsz)
{
    (void) (out != NULL && outsz > 0 && (out[0] = '\0', 1));
    int saved_mode = mcalc_out_mode;
    mcalc_out_mode = MCALC_OUT_VALUES;
    g_calc_out = out, g_calc_sz = outsz;
    int r      = mcalc_eval_bash(input, is_test, out_rc);
    g_calc_out = NULL, g_calc_sz = 0;
    mcalc_out_mode = saved_mode;
    return r;
}

int mcalc_eval_math_to_val(const char *input, int *out_type, long *out_lval, double *out_dval)
{
    if (input == NULL)
    {
        return 0;
    }
    static _Thread_local mc_tok_list_t s_tl;
    if (mc_lex(input, &s_tl) != 0)
    {
        return 0;
    }
    char       err[256] = "";
    mc_stmt_t *s        = mc_parse_stmt(&s_tl, err, sizeof(err));
    mc_val_t   v;
    if (s == NULL || s->kind != MC_STMT_EXPR || s->u.expr == NULL ||
        s->u.expr->kind == MC_EXPR_ASSIGN || mc_resolve_stmt(s, err, sizeof(err)) != 0 ||
        s->needs_image || mc_scalar_eval_expr(s->u.expr, &v, err, sizeof(err)) != 0 ||
        (v.type != MC_VAL_LONG && v.type != MC_VAL_DOUBLE))
    {
        mc_stmt_free(s);
        return 0;
    }
    mc_stmt_free(s);
    int    is_l = (v.type == MC_VAL_LONG);
    double dv   = is_l ? (double) v.lval : v.dval;
    long   lv   = is_l ? v.lval : (long) v.dval;
    (void) (out_type != NULL && (*out_type = is_l ? 1 : 2));
    (void) (out_lval != NULL && (*out_lval = lv, 1));
    (void) (out_dval != NULL && (*out_dval = dv, 1));
    return 1;
}
