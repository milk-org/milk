// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_memory/create_image.h"
#include "COREMOD_memory/create_variable.h"
#include "COREMOD_memory/image_ID.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "mc_ast.h"
#include "mc_plan.h"
#include "mcalc.h"

#define MC_HIST_MAX 16

typedef struct
{
    char name[STRINGMAXLEN_IMAGE_NAME];
    int  N;
    int  naxis;
} mc_hist_decl_t;

int mc_expr_mentions_self(const mc_expr_t *e, const char *bare, const void **out_im)
{
    if (e == NULL)
    {
        return 0;
    }
    int hit = 0;
    switch (e->kind)
    {
    case MC_EXPR_IDENT:
    case MC_EXPR_SLICE:
    {
        char nbuf[256], bbuf[256];
        snprintf(nbuf, sizeof(nbuf), "%s",
                 (e->kind == MC_EXPR_IDENT) ? e->u.ident.name : e->u.slice.name);
        mc_parse_ident_streamname(nbuf, bbuf, NULL);
        if (strcmp(bbuf, bare) == 0)
        {
            hit = 1;
        }
        else if (out_im != NULL && *out_im == NULL)
        {
            imageID id = mc_resolve_or_autoconnect_image(nbuf);
            if (id != -1)
            {
                *out_im = &milk_data.image[id];
            }
        }
        for (int ax = 0; e->kind == MC_EXPR_SLICE && ax < e->u.slice.naxis; ax++)
        {
            hit |= mc_expr_mentions_self(e->u.slice.axes[ax].start, bare, NULL);
            hit |= mc_expr_mentions_self(e->u.slice.axes[ax].stop, bare, NULL);
            hit |= mc_expr_mentions_self(e->u.slice.axes[ax].step, bare, NULL);
        }
        return hit;
    }
    case MC_EXPR_UNARY:
        return mc_expr_mentions_self(e->u.unary.operand, bare, out_im);
    case MC_EXPR_BINARY:
        hit |= mc_expr_mentions_self(e->u.binary.left, bare, out_im);
        hit |= mc_expr_mentions_self(e->u.binary.right, bare, out_im);
        return hit;
    case MC_EXPR_TERNARY:
        hit |= mc_expr_mentions_self(e->u.ternary.cond, bare, out_im);
        hit |= mc_expr_mentions_self(e->u.ternary.t_expr, bare, out_im);
        hit |= mc_expr_mentions_self(e->u.ternary.f_expr, bare, out_im);
        return hit;
    case MC_EXPR_CHAIN:
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            hit |= mc_expr_mentions_self(e->u.chain.operands[i], bare, out_im);
        }
        return hit;
    case MC_EXPR_CALL:
    {
        const mc_func_t *fn     = mc_func_find(e->u.call.name);
        int              is_red = (fn != NULL &&
                                   (fn->kind == MC_FN_REDUCE_D || fn->kind == MC_FN_REDUCE_L ||
                                    fn->kind == MC_FN_ISTATS || fn->kind == MC_FN_ICENTROID ||
                                    fn->kind == MC_FN_EAGER_RED || fn->kind == MC_FN_DOT) &&
                                   e->u.call.nargs < 2);
        for (int i = 0; i < e->u.call.nargs; i++)
        {
            hit |= mc_expr_mentions_self(e->u.call.args[i], bare, is_red ? NULL : out_im);
        }
        return hit;
    }
    case MC_EXPR_ASSIGN:
        return mc_expr_mentions_self(e->u.assign.rhs, bare, out_im);
    default:
        return 0;
    }
}

void mc_self_init_create(const char *tb, int is_local, const mc_expr_t *rhs)
{
    if (tb == NULL || tb[0] == '\0' || rhs == NULL)
    {
        return;
    }
    if (rhs->res_kind == MC_RES_IMAGE && rhs->res_naxis > 0 && rhs->res_size[0] > 0)
    {
        uint32_t sz[3] = {
            rhs->res_size[0],
            (rhs->res_naxis >= 2 && rhs->res_size[1] > 0) ? rhs->res_size[1] : 1,
            (rhs->res_naxis >= 3 && rhs->res_size[2] > 0) ? rhs->res_size[2] : 1,
        };
        uint8_t dt  = rhs->res_dt ? rhs->res_dt : _DATATYPE_FLOAT;
        imageID nid = -1;
        create_image_ID(tb, rhs->res_naxis, sz, dt, is_local ? 0 : 1, 10, 0, &nid);
    }
    else if (rhs->res_kind == MC_RES_SCALAR_L)
    {
        create_variable_long_ID(tb, 0);
    }
    else if (rhs->res_kind == MC_RES_SCALAR_D)
    {
        create_variable_ID(tb, 0.0);
    }
}

static char *trim_hist(char *s)
{
    while (isspace((unsigned char) *s))
    {
        s++;
    }
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char) s[n - 1]))
    {
        s[--n] = '\0';
    }
    return s;
}

static int hist_ring(mc_hist_decl_t *d, char *err, size_t errsz)
{
    char    ring[STRINGMAXLEN_IMAGE_NAME];
    imageID id = mc_resolve_or_autoconnect_image(d->name);
    snprintf(ring, sizeof(ring), "%.*s_hist", (int) sizeof(ring) - 6, d->name);
    if (id == -1)
    {
        snprintf(err, errsz, "hist: stream '%s' not found", d->name);
        return -1;
    }
    const IMAGE *im = &milk_data.image[id];
    d->naxis        = im->md[0].naxis;
    if (d->naxis < 1 || d->naxis > 2 || d->N < 2)
    {
        snprintf(err, errsz, "hist: '%s' must be 1-D or 2-D and the depth at least 2", d->name);
        return -1;
    }
    uint32_t sz[3] = {
        im->md[0].size[0],
        (d->naxis == 2) ? im->md[0].size[1] : (uint32_t) d->N,
        (d->naxis == 2) ? (uint32_t) d->N : 1,
    };
    imageID rid = mc_resolve_or_autoconnect_image(ring);
    if (rid == -1)
    {
        create_image_ID(ring, d->naxis + 1, sz, im->md[0].datatype, 1, 10, 0, &rid);
        if (rid == -1)
        {
            snprintf(err, errsz, "hist: cannot create stream '%s'", ring);
            return -1;
        }
        return 0;
    }
    const IMAGE *r = &milk_data.image[rid];
    if (r->md[0].naxis != d->naxis + 1 || r->md[0].size[0] != sz[0] || r->md[0].size[1] != sz[1] ||
        (d->naxis == 2 && r->md[0].size[2] != sz[2]))
    {
        snprintf(err, errsz, "hist: stream '%s' exists with another shape", ring);
        return -1;
    }
    return 0;
}

static int hist_find(const mc_hist_decl_t *decl, int nd, const char *name)
{
    for (int i = 0; i < nd; i++)
    {
        if (strcmp(decl[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

static const char *call_args(const char *c, char *a1, size_t a1sz, long *a2, long def)
{
    const char *o     = c;
    int         depth = 0;
    while (*c != '\0' && !(*c == ',' && depth == 0) && !(*c == ')' && depth == 0))
    {
        depth += (*c == '(' || *c == '[') - (*c == ')' || *c == ']');
        c++;
    }
    if (*c == '\0' || (size_t) (c - o) >= a1sz)
    {
        return NULL;
    }
    char tmp[STRINGMAXLEN_IMAGE_NAME];
    snprintf(tmp, sizeof(tmp), "%.*s", (int) (c - o), o);
    snprintf(a1, a1sz, "%s", trim_hist(tmp));
    *a2 = def;
    if (*c == ',')
    {
        *a2 = strtol(c + 1, (char **) &c, 10);
        while (isspace((unsigned char) *c))
        {
            c++;
        }
    }
    return (*c == ')') ? c + 1 : NULL;
}

static int at_ident_start(const char *text, const char *c)
{
    return c == text || !(isalnum((unsigned char) c[-1]) || c[-1] == '_');
}

static int ensure_out_cap(char **out, size_t *cap, size_t len, size_t extra)
{
    if (len + extra < *cap)
    {
        return 0;
    }
    size_t new_cap = (*cap * 2 > len + extra + 1024) ? (*cap * 2) : (len + extra + 1024);
    char  *nb      = realloc(*out, new_cap);
    if (nb == NULL)
    {
        return -1;
    }
    *out = nb, *cap = new_cap;
    return 0;
}

static int scan_hist_decls(const char *text, mc_hist_decl_t *decl, int *nd, char *err, size_t errsz)
{
    char name[STRINGMAXLEN_IMAGE_NAME];
    long num = 0;
    char q   = 0;

    for (const char *h = text; *h != '\0'; h++)
    {
        if (q != 0)
        {
            if (*h == '\\' && h[1] != '\0')
            {
                h++;
            }
            else if (*h == q)
            {
                q = 0;
            }
            continue;
        }
        if (*h == '"' || *h == '\'')
        {
            q = *h;
            continue;
        }
        if (*h == '#')
        {
            while (h[1] != '\0' && h[1] != '\n')
            {
                h++;
            }
            continue;
        }
        const char *e = NULL;
        if (at_ident_start(text, h) && strncmp(h, "hist(", 5) == 0 &&
            (e = call_args(h + 5, name, sizeof(name), &num, 0)) != NULL)
        {
            int k = hist_find(decl, *nd, name);
            if (k >= 0 && decl[k].N != (int) num)
            {
                snprintf(err, errsz, "hist(%s) declared twice with different depths", name);
                return -1;
            }
            if (k < 0)
            {
                if (*nd == MC_HIST_MAX)
                {
                    snprintf(err, errsz, "more than %d hist() declarations", MC_HIST_MAX);
                    return -1;
                }
                snprintf(decl[*nd].name, sizeof(decl[*nd].name), "%s", name);
                decl[(*nd)++].N = (int) num;
            }
            h = e - 1;
        }
    }
    return 0;
}

static int expand_prev_call(mc_hist_decl_t *decl,
                            int            *nd,
                            const char     *name,
                            long            num,
                            char          **out,
                            size_t         *cap,
                            size_t         *len,
                            char           *err,
                            size_t          errsz)
{
    int k = hist_find(decl, *nd, name);
    if (k < 0)
    {
        if (*nd == MC_HIST_MAX)
        {
            snprintf(err, errsz, "more than %d streams with a history", MC_HIST_MAX);
            return -1;
        }
        snprintf(decl[*nd].name, sizeof(decl[*nd].name), "%s", name);
        decl[*nd].N = 2;
        k           = (*nd)++;
    }
    if (num < 1 || num >= decl[k].N)
    {
        snprintf(err, errsz, "prev(%s, %ld): the ring holds %d frames", name, num, decl[k].N);
        return -1;
    }
    if (decl[k].naxis == 0 && hist_ring(&decl[k], err, errsz) != 0)
    {
        return -1;
    }
    *len +=
        (size_t) snprintf(*out + *len, *cap - *len, "%s_hist[%s(_n + %d) %% %d]", name,
                          (decl[k].naxis == 2) ? ":,:," : ":,", decl[k].N - (int) num, decl[k].N);
    return 0;
}

char *mc_rewrite_history(const char *text, int *out_nhist, char *err, size_t errsz)
{
    if (out_nhist != NULL)
    {
        *out_nhist = 0;
    }
    mc_hist_decl_t decl[MC_HIST_MAX];
    int            nd  = 0;
    size_t         cap = strlen(text) * 4 + 1024;
    char          *out = malloc(cap);
    size_t         len = 0;
    const char    *c   = text;
    char           name[STRINGMAXLEN_IMAGE_NAME];
    long           num = 0;
    char           q   = 0;

    memset(decl, 0, sizeof(decl));
    if (out == NULL || scan_hist_decls(text, decl, &nd, err, errsz) != 0)
    {
        free(out);
        return NULL;
    }
    out[0] = '\0';
    while (*c != '\0')
    {
        if (ensure_out_cap(&out, &cap, len, 512) != 0)
        {
            free(out);
            return NULL;
        }
        if (q != 0)
        {
            if (*c == '\\' && c[1] != '\0')
            {
                out[len++] = *c++;
            }
            else if (*c == q)
            {
                q = 0;
            }
            out[len++] = *c++;
            out[len]   = '\0';
            continue;
        }
        if (*c == '"' || *c == '\'')
        {
            q          = *c;
            out[len++] = *c++;
            out[len]   = '\0';
            continue;
        }
        if (*c == '#')
        {
            while (*c != '\0' && *c != '\n')
            {
                if (ensure_out_cap(&out, &cap, len, 8) != 0)
                {
                    free(out);
                    return NULL;
                }
                out[len++] = *c++;
            }
            out[len] = '\0';
            continue;
        }
        const char *e = NULL;
        if (at_ident_start(text, c) && strncmp(c, "hist(", 5) == 0 &&
            (e = call_args(c + 5, name, sizeof(name), &num, 0)) != NULL)
        {
            c = e;
            continue;
        }
        if (at_ident_start(text, c) && strncmp(c, "prev(", 5) == 0 &&
            (e = call_args(c + 5, name, sizeof(name), &num, 1)) != NULL)
        {
            if (expand_prev_call(decl, &nd, name, num, &out, &cap, &len, err, errsz) != 0)
            {
                free(out);
                return NULL;
            }
            c = e;
            continue;
        }
        out[len++] = *c++;
        out[len]   = '\0';
    }
    for (int k = 0; k < nd; k++)
    {
        if (ensure_out_cap(&out, &cap, len, 512) != 0 ||
            (decl[k].naxis == 0 && hist_ring(&decl[k], err, errsz) != 0))
        {
            free(out);
            return NULL;
        }
        len += (size_t) snprintf(out + len, cap - len, "\n%s_hist[%s_n %% %d] = %s", decl[k].name,
                                 (decl[k].naxis == 2) ? ":,:," : ":,", decl[k].N, decl[k].name);
    }
    if (out_nhist != NULL)
    {
        *out_nhist = nd;
    }
    return out;
}

static void count_scalar_ops_reds(const mc_expr_t *e, int *ops, int *reds)
{
    if (e == NULL)
    {
        return;
    }
    if (e->kind == MC_EXPR_ASSIGN)
    {
        count_scalar_ops_reds(e->u.assign.rhs, ops, reds);
        return;
    }
    if (e->kind == MC_EXPR_CALL)
    {
        const mc_func_t *fn = e->u.call.fn;
        if (fn != NULL && (fn->kind == MC_FN_REDUCE_D || fn->kind == MC_FN_REDUCE_L ||
                           fn->kind == MC_FN_DOT || fn->kind == MC_FN_EAGER_RED ||
                           fn->kind == MC_FN_ISTATS || fn->kind == MC_FN_ICENTROID))
        {
            (*reds)++, (*ops)++;
            return;
        }
        (*ops)++;
        for (int i = 0; i < e->u.call.nargs; i++)
        {
            count_scalar_ops_reds(e->u.call.args[i], ops, reds);
        }
        return;
    }
    if (e->kind == MC_EXPR_UNARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.unary.operand, ops, reds);
    }
    else if (e->kind == MC_EXPR_BINARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.binary.left, ops, reds);
        count_scalar_ops_reds(e->u.binary.right, ops, reds);
    }
    else if (e->kind == MC_EXPR_TERNARY)
    {
        (*ops)++;
        count_scalar_ops_reds(e->u.ternary.cond, ops, reds);
        count_scalar_ops_reds(e->u.ternary.t_expr, ops, reds);
        count_scalar_ops_reds(e->u.ternary.f_expr, ops, reds);
    }
    else if (e->kind == MC_EXPR_CHAIN)
    {
        (*ops) += e->u.chain.nops;
        for (int i = 0; i <= e->u.chain.nops; i++)
        {
            count_scalar_ops_reds(e->u.chain.operands[i], ops, reds);
        }
    }
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
    int              nkept = 0;
    const mc_stmt_t *cur_s = p->ast_head;
    for (int i = 0; i < p->nstmts; i++, cur_s = cur_s ? cur_s->next : NULL)
    {
        if (cur_s != NULL && cur_s->is_hist_stmt)
        {
            continue;
        }
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
                    snprintf(how, sizeof(how), "kept, merged into the fused pass of statement %d",
                             st->first_stmt_idx + 1);
                }
                else
                {
                    char   ps_buf[64]  = "";
                    char   wr_buf[256] = "";
                    size_t wlen        = 0;
                    if (st->pass.prestage_cnt > 0)
                    {
                        snprintf(ps_buf, sizeof(ps_buf), ", %d pre-stage%s", st->pass.prestage_cnt,
                                 (st->pass.prestage_cnt == 1) ? "" : "s");
                    }
                    int nin_ops = 0;
                    for (int k = 0; k < st->pass.nops; k++)
                    {
                        nin_ops += (!st->pass.ops[k].is_out && !st->pass.ops[k].is_scalar);
                    }
                    for (int k = 0; k < st->pass.nouts; k++)
                    {
                        if (!st->pass.outs[k].is_elided && wlen + 32 < sizeof(wr_buf))
                        {
                            wlen += (size_t) snprintf(wr_buf + wlen, sizeof(wr_buf) - wlen, " %s",
                                                      p->slots[st->pass.outs[k].slot_idx].bare);
                        }
                    }
                    const char *lim_note = (cur_s != NULL && cur_s->kind == MC_STMT_MULTI_ASSIGN &&
                                            cur_s->u.multi.nlhs > 4)
                                               ? " (more than 4 targets (FUSE_MAXDST))"
                                               : "";
                    snprintf(how, sizeof(how),
                             "kept: fused pass over %ux%ux%u (%s, %s, %d operand%s%s)%s, writes%s",
                             st->pass.vnx ? st->pass.vnx : 1, st->pass.vny ? st->pass.vny : 1,
                             st->pass.vnz ? st->pass.vnz : 1, wtype,
                             st->pass.is_1d_contig ? "1D" : "ND", nin_ops,
                             (nin_ops == 1) ? "" : "s", ps_buf, lim_note, wr_buf);
                }
            }
            else if (st->kind == MC_STEP_CTOR)
            {
                const char *lim_note = (cur_s != NULL && cur_s->kind == MC_STMT_MULTI_ASSIGN &&
                                        cur_s->u.multi.nlhs > 4)
                                           ? " (more than 4 targets (FUSE_MAXDST))"
                                           : "";
                snprintf(how, sizeof(how), "kept: constructor/library step%s", lim_note);
            }
            else
            {
                int nops = 0, nreds = 0;
                if (st->scalar_stmt != NULL && st->scalar_stmt->kind == MC_STMT_EXPR)
                {
                    count_scalar_ops_reds(st->scalar_stmt->u.expr, &nops, &nreds);
                }
                else if (st->scalar_stmt != NULL && st->scalar_stmt->kind == MC_STMT_MULTI_ASSIGN)
                {
                    for (int j = 0; j < st->scalar_stmt->u.multi.nrhs; j++)
                    {
                        count_scalar_ops_reds(st->scalar_stmt->u.multi.rhs[j], &nops, &nreds);
                    }
                }
                if (nops == 0)
                {
                    nops = 1;
                }
                snprintf(how, sizeof(how),
                         "kept: scalar tape: %d op%s, %d reduction%s, 0 pre-stages", nops,
                         (nops == 1) ? "" : "s", nreds, (nreds == 1) ? "" : "s");
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
