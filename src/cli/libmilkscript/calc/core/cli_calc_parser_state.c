// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_parser_state.c
 * @brief Token stream cursors, temporary image pool, user-function registry, and CLI parser state.
 *
 * Role & Goal:
 *   Maintains the active token streams (`parse_tokens`, `eval_tokens`), temporary
 *   calculator image lifecycle (`_tmpcalcN`), user-defined inline function table
 *   (`def f(x) = ...`), value constructors (`val_t`), and CLI command-argument
 *   expression evaluation (`cli_parse`, `cli_calc_eval_math_to_val`).
 */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "cli_calc_tokenizer.h"

/** token stream and current index for cli_parse */
static cli_token parse_tokens[CLI_CALC_MAX_TOKENS];
static int       parse_pos;
static int       parse_ntok;
int              parse_error;

/** token stream and current index for cli_calc_eval_line */
cli_token         eval_tokens[CLI_CALC_MAX_TOKENS];
static cli_token *active_eval_tokens = eval_tokens;
int               eval_pos;
int               eval_ntok;
int               eval_error;

static cli_calc_udef_t udef_table[CALC_UDEF_MAX];
static int             udef_count = 0;

/**
 * cli_calc_udef_clear - Clear all registered user-defined calculator functions
 */
void cli_calc_udef_clear(void)
{
    udef_count = 0;
}

/**
 * cli_calc_udef_find - Look up a user-defined function by name and length
 * @name: Function name character slice
 * @len:  Length of @name in bytes
 *
 * Return: Table index (`0 .. udef_count - 1`), or `-1` if not found.
 */
int cli_calc_udef_find(const char *name, size_t len)
{
    for (int i = 0; i < udef_count; i++)
    {
        if (strlen(udef_table[i].name) == len && strncmp(udef_table[i].name, name, len) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * cli_calc_udef_get - Retrieve a user-defined function entry by index
 * @idx: Entry index in `udef_table`
 *
 * Return: Pointer to the `cli_calc_udef_t` descriptor, or NULL if out of bounds
 */
const cli_calc_udef_t *cli_calc_udef_get(int idx)
{
    return (idx >= 0 && idx < udef_count) ? &udef_table[idx] : NULL;
}

/**
 * cli_calc_udef_add - Register or overwrite a user-defined calculator function
 * @name:      Function name
 * @nargs:     Number of parameters (`1 .. CALC_UDEF_MAXARGS`)
 * @arg_names: Parameter name array
 * @body:      Expression body string
 *
 * Return: Table index on success, or -1 on error
 */
int cli_calc_udef_add(const char *name, int nargs, const char arg_names[][64], const char *body)
{
    if (name == NULL || body == NULL || nargs < 1 || nargs > CALC_UDEF_MAXARGS)
    {
        return -1;
    }
    int idx = cli_calc_udef_find(name, strlen(name));
    if (idx < 0)
    {
        if (udef_count >= CALC_UDEF_MAX)
        {
            return -1;
        }
        idx = udef_count++;
    }
    snprintf(udef_table[idx].name, sizeof(udef_table[idx].name), "%s", name);
    udef_table[idx].nargs = nargs;
    for (int i = 0; i < nargs; i++)
    {
        snprintf(udef_table[idx].arg_names[i], sizeof(udef_table[idx].arg_names[i]), "%s",
                 arg_names[i]);
    }
    snprintf(udef_table[idx].body, sizeof(udef_table[idx].body), "%s", body);
    return idx;
}

/**
 * cli_calc_try_parse_udef - Parse and register a `def f(x, ...) = expr` statement
 * @stmt: Statement string to inspect
 *
 * Return: 1 if registered, 0 if @stmt does not start with `def `, -1 on syntax error.
 */
int cli_calc_try_parse_udef(const char *stmt)
{
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
    char args[CALC_UDEF_MAXARGS][64];
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
        if (alen == 0 || alen >= 64 || nargs >= CALC_UDEF_MAXARGS || (*p != ',' && *p != ')'))
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
    return (cli_calc_udef_add(fname, nargs, (const char (*)[64]) args, p) >= 0) ? 1 : -1;
}

/*
 * Temporary images (_tmpcalcN) created while evaluating. Each evaluation deletes only the
 * temps it created; the outermost one also reaps any left by command-argument parsing.
 */
#define CALC_TMPNAME_RING 16
static char   tmpname_ring[CALC_TMPNAME_RING][32];
static int    tmpname_next = 0;
static long  *tmp_ids      = NULL;
static size_t tmp_n        = 0;
static size_t tmp_cap      = 0;
static int    eval_depth   = 0;

/*
 * Reaped temporaries kept for reuse under hidden names (listings skip `_tmpcalc`): a statement run
 * again, in a loop or a unit's frames, gets its temporaries back instead of allocating, zeroing and
 * freeing them each time.
 */
#define CALC_POOL_N 4
#define CALC_POOL_BYTES ((size_t) 256 << 20)
static imageID pool_ids[CALC_POOL_N] = { -1, -1, -1, -1 };

/** im_bytes - size of the data of local image @im. */
static size_t im_bytes(const IMAGE *im)
{
    return (size_t) im->md->nelement * (size_t) ImageStreamIO_typesize(im->md->datatype);
}

/** pool_im - the image kept in pool slot @k, or NULL if the slot is free or the image was deleted. */
static IMAGE *pool_im(int k)
{
    char    name[32];
    imageID id = pool_ids[k];

    snprintf(name, sizeof(name), "_tmpcalcpool%d", k);
    if (id < 0 || id >= data.core.NB_MAX_IMAGE || !data.core.image[id].used ||
        strcmp(data.core.image[id].name, name) != 0)
    {
        pool_ids[k] = -1;
        return NULL;
    }
    return &data.core.image[id];
}

/** pool_put - keep reaped temporary @id for reuse. Return: 1 kept, 0 to delete (shared, or no room). */
static int pool_put(imageID id)
{
    IMAGE *im    = &data.core.image[id];
    size_t total = 0;
    int    slot  = -1;
    char   name[32];

    if (im->md == NULL || im->md->shared != 0 || im->array.raw == NULL)
    {
        return 0;
    }
    for (int k = 0; k < CALC_POOL_N; k++)
    {
        IMAGE *pk = pool_im(k);
        if (pk != NULL)
        {
            total += im_bytes(pk);
        }
        else if (slot < 0)
        {
            slot = k;
        }
    }
    if (slot < 0 || total + im_bytes(im) > CALC_POOL_BYTES)
    {
        return 0;
    }
    snprintf(name, sizeof(name), "_tmpcalcpool%d", slot);
    chname_image_ID(im->name, name);
    if (strcmp(im->name, name) != 0)
    {
        return 0;
    }
    pool_ids[slot] = id;
    return 1;
}

/**
 * cli_calc_tmp_image - create local temporary image @name of @naxis axes @size and datatype @dt, reusing
 * a kept one of the same shape and type, with counters and keywords reset: zeroed if @zero, else as the
 * last statement left it (for a caller that writes every element). Return: create_image_ID()'s status.
 */
errno_t cli_calc_tmp_image(const char *name,
                           long        naxis,
                           uint32_t   *size,
                           uint8_t     dt,
                           int         zero,
                           imageID    *id)
{
    for (int k = 0; k < CALC_POOL_N; k++)
    {
        IMAGE *im   = pool_im(k);
        int    same = (im != NULL && im->md->datatype == dt && im->md->naxis == naxis);
        for (long a = 0; same && a < naxis; a++)
        {
            same = (im->md->size[a] == size[a]);
        }
        if (!same)
        {
            continue;
        }
        chname_image_ID(im->name, name);
        if (strcmp(im->name, name) != 0)
        {
            continue;
        }
        *id           = pool_ids[k];
        pool_ids[k]   = -1;
        im->md->cnt0  = 0;
        im->md->cnt1  = 0;
        im->md->write = 0;
        for (int i = 0; im->kw != NULL && i < im->md->NBkw; i++)
        {
            im->kw[i].type = 'N';
        }
        if (zero)
        {
            memset(im->array.raw, 0, im_bytes(im));
        }
        return RETURN_SUCCESS;
    }
    return create_image_ID(name, naxis, size, dt, 0, 10, 0, id);
}

/**
 * cli_calc_tmp_begin - Enter a temporary-image tracking scope for expression evaluation
 *
 * Return: Saved mark to pass to `cli_calc_tmp_end()`
 */
size_t cli_calc_tmp_begin(void)
{
    eval_depth++;
    return tmp_n;
}

/**
 * tmp_delete_to - Reap temporary calculator images down to a saved mark: kept for reuse, or deleted
 * @mark: Index in `tmp_ids` down to which temporaries are reaped
 */
static void tmp_delete_to(size_t mark)
{
    while (tmp_n > mark)
    {
        char name[32];
        snprintf(name, sizeof(name), "_tmpcalc%ld", tmp_ids[--tmp_n]);
        imageID tid = image_ID(name, data.core.image, data.core.NB_MAX_IMAGE);
        if (tid != -1 && !pool_put(tid))
        {
            delete_image_ID(name, DELETE_IMAGE_ERRMODE_WARNING);
        }
    }
    if (tmp_n == 0)
    {
        data.calctmp_imindex = 0;
    }
}

/**
 * cli_calc_tmp_end - Exit a temporary-image tracking scope and reap its temporaries
 * @mark: Mark returned by `cli_calc_tmp_begin()`
 */
void cli_calc_tmp_end(size_t mark)
{
    eval_depth--;
    tmp_delete_to(eval_depth == 0 ? 0 : mark);
}

/**
 * cli_calc_tmp_cleanup - Delete all tracked temporary calculator images
 */
void cli_calc_tmp_cleanup(void)
{
    tmp_delete_to(0);
}

int           parse_mode   = 0;
const char   *cli_calc_src = NULL;
char          cli_calc_lasterr[256];
int           cli_calc_out_mode     = CLI_CALC_OUT_VALUES;
int           cli_calc_mode         = 0;
int           cli_calc_test_image   = 0;
unsigned long cli_calc_errs_printed = 0;

/**
 * cur_parse - Return the current token in the `cli_parse` token stream
 *
 * Return: Pointer to the current `cli_token`
 */
cli_token *cur_parse(void)
{
    return &parse_tokens[parse_pos];
}

/**
 * advance_parse - Consume and return the current token in the `cli_parse` stream
 *
 * Return: Pointer to the consumed `cli_token`
 */
cli_token *advance_parse(void)
{
    cli_token *t = &parse_tokens[parse_pos];
    if (parse_pos < parse_ntok)
    {
        parse_pos++;
    }
    return t;
}

/**
 * cur_eval - Return the current token in the `cli_calc_eval_line` token stream
 *
 * Return: Pointer to the current `cli_token`
 */
cli_token *cur_eval(void)
{
    return &active_eval_tokens[eval_pos];
}

/**
 * cli_calc_tok_push - Push a nested token buffer onto the evaluation cursor stack
 * @new_tokens: Token array to activate
 * @new_ntok:   Number of tokens in @new_tokens
 * @saved:      Output state slot for restoring via `cli_calc_tok_pop()`
 */
void cli_calc_tok_push(cli_token *new_tokens, int new_ntok, cli_calc_tok_state *saved)
{
    saved->tokens      = active_eval_tokens;
    saved->pos         = eval_pos;
    saved->ntok        = eval_ntok;
    saved->mode        = parse_mode;
    active_eval_tokens = new_tokens;
    eval_pos           = 0;
    eval_ntok          = new_ntok;
    if (parse_mode == 0)
    {
        parse_mode = 1;
    }
}

/**
 * cli_calc_tok_pop - Restore a previously saved token buffer and cursor
 * @saved: State saved by `cli_calc_tok_push()`
 */
void cli_calc_tok_pop(const cli_calc_tok_state *saved)
{
    active_eval_tokens = saved->tokens;
    eval_pos           = saved->pos;
    eval_ntok          = saved->ntok;
    parse_mode         = saved->mode;
}

/**
 * cli_calc_tok_range_eq - Compare two token ranges for structural equality (CSE)
 * @b1: Start index of first token range
 * @e1: End index (exclusive) of first token range
 * @b2: Start index of second token range
 * @e2: End index (exclusive) of second token range
 *
 * Return: 1 if both ranges are identical and pure, 0 otherwise
 */
int cli_calc_tok_range_eq(int b1, int e1, int b2, int e2)
{
    const cli_token *toks = (parse_mode == 0) ? parse_tokens : active_eval_tokens;
    if (e1 - b1 != e2 - b2 || e1 <= b1 || b1 < 0 || b2 < 0)
    {
        return 0;
    }
    for (int i = 0; i < e1 - b1; i++)
    {
        const cli_token *a = &toks[b1 + i];
        const cli_token *b = &toks[b2 + i];
        if (a->type != b->type || a->val_l != b->val_l || a->val_d != b->val_d ||
            a->fnctptr != b->fnctptr || strcmp(a->sval, b->sval) != 0)
        {
            return 0;
        }
        if (a->type == TOK_FUNC_COORD &&
            (strncmp(a->sval, "rand", 4) == 0 || strncmp(a->sval, "urand", 5) == 0 ||
             strncmp(a->sval, "grand", 5) == 0))
        {
            return 0;
        }
    }
    return 1;
}

/**
 * cli_calc_tok_pos - Return the current cursor index in the active token stream
 *
 * Return: Current token index
 */
int cli_calc_tok_pos(void)
{
    return (parse_mode == 0) ? parse_pos : eval_pos;
}

/**
 * cli_calc_tok_seek - Restore the token cursor in the active token stream
 * @pos: Token index to seek to
 */
void cli_calc_tok_seek(int pos)
{
    if (parse_mode == 0)
    {
        parse_pos = pos;
    }
    else
    {
        eval_pos = pos;
    }
}

/**
 * cli_calc_tokens_have_image - Check if any remaining token references an image or udef
 *
 * Return: 1 if an image or user-defined function token appears, 0 otherwise
 */
int cli_calc_tokens_have_image(void)
{
    for (int i = eval_pos; i < eval_ntok; i++)
    {
        if (active_eval_tokens[i].type == TOK_IMAGE || active_eval_tokens[i].type == TOK_FUNC_UDEF)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * advance_eval - Consume and return the current token in the `eval_tokens` stream
 *
 * Return: Pointer to the consumed `cli_token`
 */
cli_token *advance_eval(void)
{
    cli_token *t = &active_eval_tokens[eval_pos];
    if (eval_pos < eval_ntok)
    {
        eval_pos++;
    }
    return t;
}

/**
 * parse_errmsg - Report a calculator parse/evaluation error and set error flags
 * @msg: Error message string
 */
void parse_errmsg(const char *msg)
{
    snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s", msg);
    if ((parse_mode == 1 || data.core.Debug > 0) && parse_mode != 2)
    {
        cli_calc_errs_printed++;
        if (cli_calc_src != NULL)
        {
            fprintf(stderr, "milk: calc error: %s in '%s'\n", msg, cli_calc_src);
        }
        else
        {
            fprintf(stderr, "milk: calc error: %s\n", msg);
        }
    }
    if (parse_mode != 2)
    {
        data.parseerror = 1;
    }
    parse_error = 1;
    eval_error  = 1;
}

/**
 * to_double - Promote a `val_t` to `double`
 * @v: Value to convert
 *
 * Return: Double-precision numeric representation of @v
 */
double to_double(val_t v)
{
    if (v.type == VAL_LONG)
    {
        return (double) v.lval;
    }
    return v.dval;
}

/**
 * mk_long - Construct a `VAL_LONG` value
 * @v: Integer value
 *
 * Return: Initialized `val_t`
 */
val_t mk_long(long v)
{
    val_t r;
    r.type    = VAL_LONG;
    r.lval    = v;
    r.dval    = 0.0;
    r.sval[0] = '\0';
    return r;
}

/**
 * mk_double - Construct a `VAL_DOUBLE` value
 * @v: Floating-point value
 *
 * Return: Initialized `val_t`
 */
val_t mk_double(double v)
{
    val_t r;
    r.type    = VAL_DOUBLE;
    r.lval    = 0;
    r.dval    = v;
    r.sval[0] = '\0';
    return r;
}

/**
 * mk_string - Construct a `VAL_STRING` value
 * @s: Null-terminated string value
 *
 * Return: Initialized `val_t`
 */
val_t mk_string(const char *s)
{
    val_t r;
    r.type = VAL_STRING;
    r.lval = 0;
    r.dval = 0.0;
    snprintf(r.sval, CLI_CALC_TOKEN_MAXLEN, "%s", s);
    return r;
}

/**
 * check_image - Verify that an image exists in `data.core.image`
 * @name: Image name to look up
 *
 * Return: 1 if found, 0 (after reporting error) if missing
 */
int check_image(const char *name)
{
    if (image_ID(name, data.core.image, data.core.NB_MAX_IMAGE) == -1)
    {
        char msg[200];
        snprintf(msg, 200, "Image '%s' not found", name);
        parse_errmsg(msg);
        return 0;
    }
    return 1;
}

/**
 * alloc_tmpname - Allocate a fresh `_tmpcalc<id>` temporary image name
 *
 * Return: Pointer to a temporary image name in the ring buffer
 */
const char *alloc_tmpname(void)
{
    char *name = tmpname_ring[tmpname_next++ % CALC_TMPNAME_RING];
    long  id;
    do
    {
        id = data.calctmp_imindex++;
        snprintf(name, 32, "_tmpcalc%ld", id);
    } while (image_ID(name, data.core.image, data.core.NB_MAX_IMAGE) != -1);
    if (tmp_n == tmp_cap)
    {
        size_t ncap = tmp_cap ? 2 * tmp_cap : 32;
        long  *nids = (long *) realloc(tmp_ids, ncap * sizeof(long));
        if (nids == NULL)
        {
            return name;
        }
        tmp_ids = nids;
        tmp_cap = ncap;
    }
    tmp_ids[tmp_n++] = id;
    return name;
}

/**
 * cli_parse - Parse a CLI input token and populate `data.cmdargtoken[data.cmdNBarg]`
 * @input: Input token string
 */
void cli_parse(const char *input)
{
    parse_mode  = 0;
    parse_error = 0;
    eval_error  = 0;

    parse_ntok = cli_tokenize(input, parse_tokens, CLI_CALC_MAX_TOKENS);
    if (parse_ntok <= 0)
    {
        return;
    }
    parse_pos = 0;
    if (parse_tokens[0].type == TOK_NEWLINE || parse_tokens[0].type == TOK_EOF)
    {
        return;
    }

    val_t result = parse_expr(0);
    if (parse_error)
    {
        data.parseerror                      = 0;
        parse_error                          = 0;
        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_RAWSTRING;
        snprintf(data.cmdargtoken[data.cmdNBarg].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s",
                 input);
        size_t len = strlen(data.cmdargtoken[data.cmdNBarg].val.string);
        if (len > 0 && data.cmdargtoken[data.cmdNBarg].val.string[len - 1] == '\n')
        {
            data.cmdargtoken[data.cmdNBarg].val.string[len - 1] = '\0';
        }
        return;
    }

    if (cur_parse()->type == TOK_NEWLINE)
    {
        advance_parse();
    }

    if (result.type == VAL_DOUBLE)
    {
        if (data.core.Debug > 0)
        {
            printf("\t double: %.10g\n", result.dval);
        }
        data.cmdargtoken[data.cmdNBarg].type     = CMDARGTOKEN_TYPE_FLOAT;
        data.cmdargtoken[data.cmdNBarg].val.numf = result.dval;
    }
    else if (result.type == VAL_LONG)
    {
        if (data.core.Debug > 0)
        {
            printf("\t long:   %ld\n", result.lval);
        }
        data.cmdargtoken[data.cmdNBarg].type     = CMDARGTOKEN_TYPE_LONG;
        data.cmdargtoken[data.cmdNBarg].val.numl = result.lval;
    }
    else if (result.type == VAL_STRING)
    {
        if (data.core.Debug > 0)
        {
            printf("\t string: %s\n", result.sval);
        }
        snprintf(data.cmdargtoken[data.cmdNBarg].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s",
                 result.sval);
    }
}

/**
 * cli_calc_eval_math_to_val - Evaluate a string as a pure numeric math expression silently
 * @input:    Null-terminated expression string
 * @out_type: Output type tag (1 for long, 2 for double)
 * @out_lval: Output integer value when `@out_type == 1`
 * @out_dval: Output floating-point value when `@out_type == 2`
 *
 * Return: 1 on success, 0 if @input is not a pure numeric expression
 */
int cli_calc_eval_math_to_val(const char *input, int *out_type, long *out_lval, double *out_dval)
{
    int       saved_mode  = parse_mode;
    int       saved_pos   = eval_pos;
    int       saved_ntok  = eval_ntok;
    int       saved_perr  = parse_error;
    int       saved_eerr  = eval_error;
    int       saved_dperr = data.parseerror;
    cli_token saved_tokens[CLI_CALC_MAX_TOKENS];
    memcpy(saved_tokens, eval_tokens, sizeof(eval_tokens));

    parse_mode = 2;
    char tbuf[8192];
    snprintf(tbuf, 8192, "%s\n", input);

    eval_ntok       = cli_tokenize(tbuf, eval_tokens, CLI_CALC_MAX_TOKENS);
    data.parseerror = 0;
    parse_error     = 0;
    eval_error      = 0;
    eval_pos        = 0;

    int success = 0;

    if (eval_ntok > 0 && cur_eval()->type != TOK_NEWLINE && cur_eval()->type != TOK_EOF)
    {
        size_t tmark  = cli_calc_tmp_begin();
        val_t  result = parse_expr(0);
        cli_calc_tmp_end(tmark);

        if (!parse_error && !eval_error &&
            (cur_eval()->type == TOK_EOF || cur_eval()->type == TOK_NEWLINE) &&
            !(result.type == VAL_STRING && eval_ntok <= 2))
        {
            if (result.type == VAL_LONG)
            {
                if (out_type)
                {
                    *out_type = 1;
                }
                if (out_lval)
                {
                    *out_lval = result.lval;
                }
                success = 1;
            }
            else if (result.type == VAL_DOUBLE)
            {
                if (out_type)
                {
                    *out_type = 1 + 1;
                }
                if (out_dval)
                {
                    *out_dval = result.dval;
                }
                success = 1;
            }
        }
    }

    memcpy(eval_tokens, saved_tokens, sizeof(eval_tokens));
    parse_mode      = saved_mode;
    eval_pos        = saved_pos;
    eval_ntok       = saved_ntok;
    parse_error     = saved_perr;
    eval_error      = saved_eerr;
    data.parseerror = saved_dperr;

    return success;
}
