// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_prog.c
 * @brief Calculator programs for milk.calc units: parse once, run on each frame.
 *
 * Each statement is kept either as a fused one-pass program (cli_calc_keep_statement(): an element-wise
 * image assignment, variables read when it runs) or as text evaluated on each run. Bound variables carry
 * the unit's parameters in and its scalar results out.
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "cli_calc_prog.h"

#define PROG_MAXSTMT 64

typedef struct
{
    char *text;
    void *kept; /* fused program, or NULL: evaluated as text */
} prog_stmt;

struct cli_calc_prog
{
    prog_stmt         st[PROG_MAXSTMT];
    int               n;
    cli_calc_binding *vars;
    long             *vid; /* calculator variable of each binding */
    unsigned char    *assigned;
    int               nvars;
};

void cli_calc_prog_attach(void)
{
    data.core = milk_data; /* the same arrays: what the runtime loads, the calculator sees */
}

static int add_stmt(cli_calc_prog *p, const char *s, size_t len, char *err, size_t errsz)
{
    while (len > 0 && isspace((unsigned char) *s))
    {
        s++;
        len--;
    }
    while (len > 0 && isspace((unsigned char) s[len - 1]))
    {
        len--;
    }
    if (len == 0)
    {
        return 0;
    }
    if (p->n == PROG_MAXSTMT)
    {
        snprintf(err, errsz, "more than %d statements", PROG_MAXSTMT);
        return -1;
    }
    p->st[p->n].text = strndup(s, len);
    return (p->st[p->n++].text == NULL) ? -1 : 0;
}

/** split() - statements end at a `;` or newline outside quotes and brackets; `#` starts a comment. */
static int split(cli_calc_prog *p, const char *text, char *err, size_t errsz)
{
    const char *st    = text;
    int         depth = 0;
    char        q     = 0;

    for (const char *c = text; *c != '\0'; c++)
    {
        if (q != 0)
        {
            q = (*c == q) ? 0 : q;
        }
        else if (*c == '"' || *c == '\'')
        {
            q = *c;
        }
        else if (*c == '(' || *c == '[')
        {
            depth++;
        }
        else if ((*c == ')' || *c == ']') && depth > 0)
        {
            depth--;
        }
        else if (*c == '#' || (depth == 0 && (*c == ';' || *c == '\n')))
        {
            if (add_stmt(p, st, (size_t) (c - st), err, errsz) != 0)
            {
                return -1;
            }
            if (*c == '#')
            {
                c += strcspn(c, "\n");
            }
            if (*c == '\0')
            {
                return 0;
            }
            st = c + 1;
        }
    }
    return add_stmt(p, st, strlen(st), err, errsz);
}

/** target() - the name statement @s assigns as a whole (`name =` or `name op=`) into @name; '=', the op, or 0. */
static int target(const char *s, char *name, size_t sz)
{
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
        case FPTYPE_UINT64:
            v->type    = 1;
            v->value.l = (long) *(uint64_t *) x;
            break;
        case FPTYPE_ONOFF:
            v->type    = 1;
            v->value.l = (*(int32_t *) x != 0);
            break;
        case FPTYPE_FLOAT64:
            v->type    = 0;
            v->value.f = *(double *) x;
            break;
        case FPTYPE_STREAMNAME:
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

static void store_vars(cli_calc_prog *p)
{
    for (int i = 0; i < p->nvars; i++)
    {
        const VARIABLE *v = &data.core.variable[p->vid[i]];
        void           *x = p->vars[i].ptr;
        double          d = (v->type == 1) ? (double) v->value.l : v->value.f;
        int is_str = (p->vars[i].type == FPTYPE_STRING || p->vars[i].type == FPTYPE_FITSFILENAME);
        if (!p->assigned[i] || p->vars[i].type == FPTYPE_STREAMNAME || (v->type == 2) != is_str)
        {
            continue;
        }
        switch (p->vars[i].type)
        {
        case FPTYPE_INT64:
            *(int64_t *) x = (v->type == 1) ? v->value.l : llround(d);
            break;
        case FPTYPE_UINT64:
            *(uint64_t *) x = (d < 0) ? 0 : (uint64_t) ((v->type == 1) ? v->value.l : llround(d));
            break;
        case FPTYPE_ONOFF:
            *(int32_t *) x = (d != 0.0);
            break;
        case FPTYPE_FLOAT64:
            *(double *) x = d;
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
                      vars[i].type == FPTYPE_FITSFILENAME);
        p->vars[i] = vars[i];
        p->vid[i]  = is_str ? create_variable_string_ID(vars[i].name, "")
                     : (vars[i].type == FPTYPE_FLOAT64) ? create_variable_ID(vars[i].name, 0.0)
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
    p->nvars = nvars;
    snprintf(err, errsz, "out of memory");
    if (bind_vars(p, vars, err, errsz) != 0 || split(p, text, err, errsz) != 0)
    {
        cli_calc_prog_free(p);
        return NULL;
    }
    for (int i = 0; i < p->n; i++)
    {
        char name[STRINGMAXLEN_IMAGE_NAME];
        int  op = target(p->st[i].text, name, sizeof(name));
        if (op != 0 &&
            (strcmp(name, "_n") == 0 || strcmp(name, "_t") == 0 || strcmp(name, "_dt") == 0))
        {
            snprintf(err, errsz, "cannot assign to read-only variable '%s'", name);
            cli_calc_prog_free(p);
            return NULL;
        }
        int b = 0;
        while (op != 0 && b < nvars && strcmp(vars[b].name, name) != 0)
        {
            b++;
        }
        if (op != 0 && b < nvars && vars[b].type != FPTYPE_STREAMNAME)
        {
            p->assigned[b] = 1; /* a result: evaluated as text */
            continue;
        }
        if (op != 0 && b < nvars && vars[b].type == FPTYPE_STREAMNAME)
        {
            snprintf(name, sizeof(name), "%s", (const char *) vars[b].ptr);
        }
        p->st[i].kept = cli_calc_keep_statement(p->st[i].text);
        if (p->st[i].kept != NULL || op != '=' ||
            image_ID(name, data.core.image, data.core.NB_MAX_IMAGE) != -1 ||
            (b == nvars && variable_ID(name) != -1))
        {
            continue;
        }
        if (eval_text(p->st[i].text, err, errsz) != 0) /* a new name: make it now */
        {
            cli_calc_prog_free(p);
            return NULL;
        }
        imageID id = image_ID(name, data.core.image, data.core.NB_MAX_IMAGE);
        if (id != -1 && data.core.image[id].md[0].shared != 1)
        {
            if (share(name, err, errsz) != 0)
            {
                cli_calc_prog_free(p);
                return NULL;
            }
            p->st[i].kept = cli_calc_keep_statement(p->st[i].text);
        }
    }
    err[0] = '\0';
    return p;
}

int cli_calc_prog_assigns(const cli_calc_prog *p, int i)
{
    return (i >= 0 && i < p->nvars) ? p->assigned[i] : 0;
}

int cli_calc_prog_run(cli_calc_prog *p, char *err, size_t errsz)
{
    load_vars(p);
    for (int i = 0; i < p->n; i++)
    {
        if (p->st[i].kept != NULL)
        {
            if (cli_calc_fused_run(p->st[i].kept) != 0)
            {
                snprintf(err, errsz, "%s in '%s'", cli_calc_lasterr, p->st[i].text);
                return -1;
            }
        }
        else if (eval_text(p->st[i].text, err, errsz) != 0)
        {
            return -1;
        }
    }
    store_vars(p);
    return 0;
}

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
    p->nvars = nvars;
    snprintf(err, errsz, "out of memory");
    if (bind_vars(p, vars, err, errsz) != 0 || split(p, text, err, errsz) != 0)
    {
        cli_calc_prog_free(p);
        return -1;
    }
    for (int i = 0; i < p->n; i++)
    {
        char name[STRINGMAXLEN_IMAGE_NAME];
        int  op = target(p->st[i].text, name, sizeof(name));
        if (op != 0 &&
            (strcmp(name, "_n") == 0 || strcmp(name, "_t") == 0 || strcmp(name, "_dt") == 0))
        {
            snprintf(err, errsz, "cannot assign to read-only variable '%s'", name);
            cli_calc_prog_free(p);
            return -1;
        }
        int b = 0;
        while (op != 0 && b < nvars && strcmp(vars[b].name, name) != 0)
        {
            b++;
        }
        if (op != 0 && b < nvars && vars[b].type != FPTYPE_STREAMNAME)
        {
            p->assigned[b] = 1;
        }
        if (op != 0 && b < nvars && vars[b].type == FPTYPE_STREAMNAME)
        {
            snprintf(name, sizeof(name), "%s", (const char *) vars[b].ptr);
        }
        if (eval_text(p->st[i].text, err, errsz) != 0)
        {
            cli_calc_prog_free(p);
            return -1;
        }
        if (op == '=' && (b == nvars || vars[b].type == FPTYPE_STREAMNAME) &&
            (b < nvars || variable_ID(name) == -1))
        {
            imageID id = image_ID(name, data.core.image, data.core.NB_MAX_IMAGE);
            if (id != -1 && data.core.image[id].md[0].shared != 1 && share(name, err, errsz) != 0)
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

void cli_calc_prog_free(cli_calc_prog *p)
{
    if (p == NULL)
    {
        return;
    }
    for (int i = 0; i < p->n; i++)
    {
        free(p->st[i].text);
        free(p->st[i].kept);
    }
    free(p->vars);
    free(p->vid);
    free(p->assigned);
    free(p);
}
