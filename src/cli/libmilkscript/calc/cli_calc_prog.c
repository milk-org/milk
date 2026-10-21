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
    data.core = milk_data; /* the same arrays: what the runtime loads, the calculator sees */
}

/**
 * add_stmt - Trim whitespace and append a statement string to a program
 * @p:     Calculator program being constructed
 * @s:     Start of statement substring
 * @len:   Length of statement substring in bytes
 * @err:   Buffer for error message if `PROG_MAXSTMT` is exceeded
 * @errsz: Size of @err in bytes
 *
 * Strips leading and trailing whitespace, ignores empty statements, and
 * duplicates non-empty statements into `p->st` up to `PROG_MAXSTMT`.
 *
 * Return: 0 on success (including empty statements), -1 on limit or allocation failure
 */
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

/**
 * eval_text - Evaluate a single calculator statement from source text
 * @stmt:  Null-terminated calculator statement string
 * @err:   Buffer to receive a formatted diagnostic message on error
 * @errsz: Size of @err in bytes
 *
 * Runs `cli_calc_eval_str()` on @stmt and formats `cli_calc_lasterr` with the
 * offending statement into @err when evaluation fails.
 *
 * Return: 0 on success, -1 on parse or evaluation error
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
 * @p: Compiled calculator program with variable bindings
 *
 * Synchronizes each bound FPS parameter (`INT64`, `UINT64`, `ONOFF`, `FLOAT64`,
 * `STREAMNAME`, or string) into its corresponding `data.core.variable` slot,
 * mapping shared-memory streams on demand when `FPTYPE_STREAMNAME` is bound.
 *
 * Return: void
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

/**
 * store_vars - Write assigned calculator variable values back to bound C parameters
 * @p: Compiled calculator program with variable bindings
 *
 * Updates only the bound C pointers whose calculator variables were assigned
 * by statements in @p (`p->assigned[i] != 0`), converting numeric and string
 * types appropriately.
 *
 * Return: void
 */
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

/**
 * cli_calc_prog_new - Parse, bind, and pre-compile a multi-statement calculator program
 * @text:  Source text containing `;`- or newline-separated calculator statements
 * @vars:  Array of variable bindings connecting FPS parameters to calculator names
 * @nvars: Number of entries in @vars
 * @err:   Buffer to receive an error description on failure
 * @errsz: Size of @err in bytes
 *
 * Splits @text into statements, creates bound variables, evaluates initial
 * assignments that create new output images (promoting them to shared streams),
 * and compiles eligible image assignments into fused one-pass programs.
 *
 * Return: Newly allocated `cli_calc_prog` pointer, or NULL on error
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

/**
 * cli_calc_prog_assigns - Query whether a bound variable is assigned by the program
 * @p: Compiled calculator program
 * @i: 0-based index into the variable bindings array
 *
 * Allows callers to inspect which bound parameters act as outputs written by
 * the program during execution.
 *
 * Return: 1 if binding @i is assigned by @p, 0 otherwise
 */
int cli_calc_prog_assigns(const cli_calc_prog *p, int i)
{
    return (i >= 0 && i < p->nvars) ? p->assigned[i] : 0;
}

/**
 * cli_calc_prog_run - Execute all statements of a compiled calculator program for one frame
 * @p:     Compiled calculator program
 * @err:   Buffer to receive an error message if a statement fails
 * @errsz: Size of @err in bytes
 *
 * Loads current bound variable values, runs each statement via its pre-compiled
 * fused program (`cli_calc_fused_run()`) or text fallback (`eval_text()`), and
 * stores assigned scalar outputs back into their bound C variables.
 *
 * Return: 0 on success, -1 on runtime evaluation error
 */
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

/**
 * cli_calc_prog_exec_once - Parse and execute a calculator program once without keeping it
 * @text:         Source text of calculator statements (`begin` or `end` hook)
 * @vars:         Array of variable bindings
 * @nvars:        Number of entries in @vars
 * @assigned_out: Optional array of length @nvars set to 1 for each assigned binding
 * @err:          Buffer to receive an error message on failure
 * @errsz:        Size of @err in bytes
 *
 * Evaluates a one-shot statement block (such as an initialization or cleanup
 * script), promotes newly created images to shared memory streams, and updates
 * bound output variables before freeing temporary program state.
 *
 * Return: 0 on success, -1 on error
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

/**
 * cli_calc_prog_free - Release all memory associated with a compiled calculator program
 * @p: Calculator program to free (may be NULL)
 *
 * Frees statement strings, compiled fused programs, binding arrays, and the
 * `cli_calc_prog` structure itself.
 *
 * Return: void
 */
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
