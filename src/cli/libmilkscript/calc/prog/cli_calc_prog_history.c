// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_prog_history.c
 * @brief Temporal `hist(s, N)` and `prev(s[, k])` ring-buffer rewriting for calculator programs.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "cli_calc_internal.h"
#include "cli_calc_prog_internal.h"

typedef struct
{
    char name[STRINGMAXLEN_IMAGE_NAME];
    int  N;
    int  naxis; /* of the stream (1 or 2); the ring has one axis more */
} hist_decl;

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

static int hist_ring(hist_decl *d, char *err, size_t errsz)
{
    char    ring[STRINGMAXLEN_IMAGE_NAME];
    imageID id = cli_calc_prog_connect_stream(d->name);

    snprintf(ring, sizeof(ring), "%.*s_hist", (int) sizeof(ring) - 6, d->name);
    if (id == -1)
    {
        snprintf(err, errsz, "hist: stream '%s' not found", d->name);
        return -1;
    }
    const IMAGE *im = &data.core.image[id];
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
    imageID rid = cli_calc_prog_connect_stream(ring);
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
    const IMAGE *r = &data.core.image[rid];
    if (r->md[0].naxis != d->naxis + 1 || r->md[0].size[0] != sz[0] || r->md[0].size[1] != sz[1] ||
        (d->naxis == 2 && r->md[0].size[2] != sz[2]))
    {
        snprintf(err, errsz, "hist: stream '%s' exists with another shape", ring);
        return -1;
    }
    return 0;
}

static int hist_find(const hist_decl *decl, int nd, const char *name)
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
    *out = nb;
    *cap = new_cap;
    return 0;
}

static int scan_hist_decls(const char *text, hist_decl *decl, int *nd, char *err, size_t errsz)
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
                if (*nd == HIST_MAX)
                {
                    snprintf(err, errsz, "more than %d hist() declarations", HIST_MAX);
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

static int expand_prev_call(hist_decl  *decl,
                            int        *nd,
                            const char *name,
                            long        num,
                            char      **out,
                            size_t     *cap,
                            size_t     *len,
                            char       *err,
                            size_t      errsz)
{
    int k = hist_find(decl, *nd, name);
    if (k < 0)
    {
        if (*nd == HIST_MAX)
        {
            snprintf(err, errsz, "more than %d streams with a history", HIST_MAX);
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

/**
 * cli_calc_prog_rewrite_history - Replace `hist(s, N)` and `prev(s[, k])` with ring-buffer slices
 */
char *cli_calc_prog_rewrite_history(const char *text, char *err, size_t errsz)
{
    hist_decl   decl[HIST_MAX];
    int         nd  = 0;
    size_t      cap = strlen(text) * 4 + 1024;
    char       *out = malloc(cap);
    size_t      len = 0;
    const char *c   = text;
    char        name[STRINGMAXLEN_IMAGE_NAME];
    long        num = 0;
    char        q   = 0;

    memset(decl, 0, sizeof(decl));
    if (out == NULL)
    {
        return NULL;
    }
    if (scan_hist_decls(text, decl, &nd, err, errsz) != 0)
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
        const char *e;
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
        len += (size_t) snprintf(out + len, cap - len, "\n%c%s_hist[%s_n %% %d] = %s", HIST_MARK,
                                 decl[k].name, (decl[k].naxis == 2) ? ":,:," : ":,", decl[k].N,
                                 decl[k].name);
    }
    return out;
}

/** target() - the name statement @s assigns as a whole (`name =` or `name op=`) into @name; '=', the op, or 0. */
static int target(const char *s, char *name, size_t sz, int *out_is_local)
{
    if (out_is_local != NULL)
    {
        *out_is_local = 0;
    }
    while (isspace((unsigned char) *s))
    {
        s++;
    }
    if (strncmp(s, "local", 5) == 0 && isspace((unsigned char) s[5]))
    {
        s += 5;
        while (isspace((unsigned char) *s))
        {
            s++;
        }
        if (out_is_local != NULL)
        {
            *out_is_local = 1;
        }
    }
    if (*s == '@')
    {
        s++;
    }
    if ((s[0] == 'L' || s[0] == 'S') && s[1] == ':')
    {
        if (s[0] == 'L' && out_is_local != NULL)
        {
            *out_is_local = 1;
        }
        s += 2;
    }
    size_t n = 0;
    if (!(isalpha((unsigned char) s[0]) || s[0] == '_'))
    {
        return 0;
    }
    while (isalnum((unsigned char) s[n]) || s[n] == '_')
    {
        n++;
    }
    const char *o  = s + n + strspn(s + n, " \t");
    int         op = (o[0] == '=' && o[1] != '=')                            ? '='
                     : (o[0] != '\0' && strchr("+-*/", o[0]) && o[1] == '=') ? o[0]
                                                                             : 0;
    if (op == 0 || n >= sz)
    {
        return 0;
    }
    snprintf(name, sz, "%.*s", (int) n, s);
    return op;
}

static int is_readonly_var_name(const char *name)
{
    return (strcmp(name, "_n") == 0 || strcmp(name, "_t") == 0 || strcmp(name, "_dt") == 0 ||
            strcmp(name, "_cnt0") == 0 || strcmp(name, "_ts") == 0 ||
            strcmp(name, "_missed") == 0 || strcmp(name, "_torn") == 0);
}

/**
 * cli_calc_prog_parse_multi_targets - Parse comma-separated LHS identifiers before `=`
 */
int cli_calc_prog_parse_multi_targets(const char *s,
                                      char        names[][STRINGMAXLEN_IMAGE_NAME],
                                      int         is_local[],
                                      int         max_n)
{
    int         n = 0;
    const char *p = s;
    while (*p != '\0' && n < max_n)
    {
        while (isspace((unsigned char) *p))
        {
            p++;
        }
        int loc = 0;
        if (n == 0 && strncmp(p, "local", 5) == 0 && isspace((unsigned char) p[5]))
        {
            p += 5;
            while (isspace((unsigned char) *p))
            {
                p++;
            }
            loc = 1;
        }
        if (*p == '@')
        {
            p++;
        }
        if ((p[0] == 'L' || p[0] == 'S') && p[1] == ':')
        {
            if (p[0] == 'L')
            {
                loc = 1;
            }
            p += 2;
        }
        const char *ns = p;
        while (isalnum((unsigned char) *p) || *p == '_')
        {
            p++;
        }
        size_t len = (size_t) (p - ns);
        if (len == 0 || len >= STRINGMAXLEN_IMAGE_NAME)
        {
            return 0;
        }
        memcpy(names[n], ns, len);
        names[n][len] = '\0';
        is_local[n]   = loc;
        n++;
        while (isspace((unsigned char) *p))
        {
            p++;
        }
        if (*p == ',')
        {
            p++;
        }
        else
        {
            break;
        }
    }
    if (n < 2 || *p != '=' || p[1] == '=')
    {
        return 0;
    }
    return n;
}

/**
 * cli_calc_prog_classify_target - Inspect a statement's LHS target and resolve bound variables
 */
int cli_calc_prog_classify_target(cli_calc_prog          *p,
                                  int                     stmt_idx,
                                  const char             *stmt,
                                  const cli_calc_binding *vars,
                                  char                   *name,
                                  size_t                  namesz,
                                  int                    *out_op,
                                  int                    *out_bind,
                                  char                   *err,
                                  size_t                  errsz)
{
    int is_local             = 0;
    int op                   = target(stmt, name, namesz, &is_local);
    p->st[stmt_idx].is_local = is_local;
    if (op == 0)
    {
        char mnames[8][STRINGMAXLEN_IMAGE_NAME];
        int  mloc[8] = { 0 };
        int  nm      = cli_calc_prog_parse_multi_targets(stmt, mnames, mloc, 8);
        for (int j = 0; j < nm; j++)
        {
            if (is_readonly_var_name(mnames[j]))
            {
                snprintf(err, errsz, "cannot assign to read-only variable '%s'", mnames[j]);
                return -1;
            }
            for (int b = 0; b < p->nvars; b++)
            {
                if (strcmp(vars[b].name, mnames[j]) == 0 && vars[b].type != FPTYPE_STREAMNAME)
                {
                    p->assigned[b] = 1;
                }
            }
        }
    }
    if (op != 0 && is_readonly_var_name(name))
    {
        snprintf(err, errsz, "cannot assign to read-only variable '%s'", name);
        return -1;
    }
    int b = 0;
    while (op != 0 && b < p->nvars && strcmp(vars[b].name, name) != 0)
    {
        b++;
    }
    if (op != 0 && b < p->nvars && vars[b].type != FPTYPE_STREAMNAME)
    {
        p->assigned[b] = 1;
    }
    else if (op != 0 && b < p->nvars && vars[b].type == FPTYPE_STREAMNAME)
    {
        snprintf(name, namesz, "%s", (const char *) vars[b].ptr);
    }
    if (op != 0)
    {
        snprintf(p->st[stmt_idx].dst_name, sizeof(p->st[stmt_idx].dst_name), "%s", name);
    }
    *out_op   = op;
    *out_bind = b;
    return 0;
}

/**
 * cli_calc_prog_parse_prefix - Split statement @i of @p into its guard, kind, and body
 */
int cli_calc_prog_parse_prefix(cli_calc_prog *p, int i, char *err, size_t errsz)
{
    prog_stmt *st = &p->st[i];
    char      *s  = st->text;

    if (st->kind != STMT_NORMAL)
    {
        return 0;
    }
    if (s[0] == HIST_MARK)
    {
        memmove(s, s + 1, strlen(s));
        st->kind = STMT_HIST;
        return 0;
    }
    int urc = cli_calc_try_parse_udef(s);
    if (urc != 0)
    {
        if (urc < 0)
        {
            snprintf(err, errsz, "invalid def syntax in '%s'", s);
            return -1;
        }
        st->kind = STMT_NOP;
        return 0;
    }
    if (strncmp(s, "if", 2) == 0 && isspace((unsigned char) s[2]))
    {
        int   depth = 0;
        char *c     = s + 3;
        for (; *c != '\0'; c++)
        {
            if (*c == '(' || *c == '[')
            {
                depth++;
            }
            else if (*c == ')' || *c == ']')
            {
                depth--;
            }
            else if (*c == ':' && depth == 0)
            {
                break;
            }
        }
        if (*c != ':')
        {
            snprintf(err, errsz, "if without ':' in '%s'", s);
            return -1;
        }
        *c         = '\0';
        char *cond = trim_hist(s + 3);
        char *body = trim_hist(c + 1);
        if (cond[0] == '\0' || body[0] == '\0')
        {
            snprintf(err, errsz, "empty condition or statement in 'if %s:'", cond);
            return -1;
        }
        char   gname[32];
        size_t glen = strlen(cond) + sizeof(gname) + 8;
        snprintf(gname, sizeof(gname), "__g%d", i);
        st->guard = malloc(glen);
        if (st->guard == NULL)
        {
            return -1;
        }
        snprintf(st->guard, glen, "%s = (%s)", gname, cond);
        st->guard_vid = create_variable_ID(gname, 0.0);
        char *nb      = strdup(body);
        if (nb == NULL || st->guard_vid < 0)
        {
            free(nb);
            snprintf(err, errsz, "cannot keep the condition of 'if %s:'", cond);
            return -1;
        }
        free(st->text);
        st->text = nb;
        s        = nb;
    }
    if (strcmp(s, "skip") == 0)
    {
        st->kind = STMT_SKIP;
    }
    else if (strncmp(s, "post(", 5) == 0 && s[strlen(s) - 1] == ')')
    {
        char tmp[STRINGMAXLEN_IMAGE_NAME];
        snprintf(tmp, sizeof(tmp), "%.*s", (int) (strlen(s) - 6), s + 5);
        snprintf(st->post, sizeof(st->post), "%s", trim_hist(tmp));
        st->kind = STMT_POST;
    }
    return 0;
}

/**
 * cli_calc_prog_text_mentions_ident - Return 1 if @text references whole-word identifier @ident
 */
int cli_calc_prog_text_mentions_ident(const char *text, const char *ident)
{
    if (text == NULL || ident == NULL || ident[0] == '\0')
    {
        return 0;
    }
    size_t      len = strlen(ident);
    const char *p   = text;
    while ((p = strstr(p, ident)) != NULL)
    {
        int prev_ok = (p == text) || !(isalnum((unsigned char) p[-1]) || p[-1] == '_');
        int next_ok = !(isalnum((unsigned char) p[len]) || p[len] == '_');
        if (prev_ok && next_ok)
        {
            return 1;
        }
        p += len;
    }
    return 0;
}

/**
 * cli_calc_prog_connect_stream - Lookup image @name, mapping its SHM stream on demand if present
 */
imageID cli_calc_prog_connect_stream(const char *name)
{
    imageID id = image_ID(name, data.core.image, data.core.NB_MAX_IMAGE);
    char    path[1024];
    if (id == -1 && ImageStreamIO_filename(path, sizeof(path), name) == IMAGESTREAMIO_SUCCESS &&
        access(path, F_OK) == 0)
    {
        read_sharedmem_image(name, data.core.image, data.core.NB_MAX_IMAGE);
        id = image_ID(name, data.core.image, data.core.NB_MAX_IMAGE);
    }
    return id;
}
