// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_parser.c
 * @brief Hand-written Pratt parser for CLI expressions
 *
 * Architecture Overview:
 * The Pratt (precedence-climbing) parser replaces legacy bison-generated parsers
 * for expression evaluation. It parses operator precedence natively supporting
 * +, -, *, /, ^, % arithmetic, logical/relational operators, and dynamic variable
 * assignments. Supports `long`, `double`, and `string` (image name) evaluation.
 * Evaluates expressions immediately, supporting math on images as well
 * as per-pixel masking. Populates the global execution context `data.cmdargtoken`.
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/stream_slice.h"

#include "cli_calc_tokenizer.h"
#include "cli_calc_parser.h"
#include "libmilkdata/milk_type_dispatch.h"
#include "CLIcore_script.h"
#include "cli_calc_internal.h"
/* --------------------------------------------------------
 * Parser state
 * -------------------------------------------------------- */

/** token stream and current index for cli_parse */
static cli_token parse_tokens[CLI_CALC_MAX_TOKENS];
static int       parse_pos;
static int       parse_ntok;
int              parse_error;

/** token stream and current index for cli_calc_eval_line */
static cli_token eval_tokens[CLI_CALC_MAX_TOKENS];
static int       eval_pos;
static int       eval_ntok;
int              eval_error;


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

/**
 * tmp_begin - Enter a temporary-image tracking scope for expression evaluation
 *
 * Increments `eval_depth` and records the current count of `_tmpcalc` images so
 * nested evaluations only delete temporaries created within their own scope.
 *
 * Return: Mark index into `tmp_ids` at the start of this evaluation scope
 */
static size_t tmp_begin(void)
{
    eval_depth++;
    return tmp_n;
}

/**
 * tmp_delete_to - Delete temporary calculator images down to a saved mark
 * @mark: Target `tmp_n` watermark to restore
 *
 * Deletes `_tmpcalc<id>` images created since @mark from `data.core.image` and
 * resets `data.calctmp_imindex` when all tracked temporaries have been reaped.
 *
 * Return: void
 */
static void tmp_delete_to(size_t mark)
{
    while (tmp_n > mark)
    {
        char name[32];
        snprintf(name, sizeof(name), "_tmpcalc%ld", tmp_ids[--tmp_n]);
        if (image_ID(name, data.core.image, data.core.NB_MAX_IMAGE) != -1)
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
 * tmp_end - Exit a temporary-image tracking scope and reap its temporaries
 * @mark: Watermark returned by the matching `tmp_begin()` call
 *
 * Decrements `eval_depth` and deletes temporaries back to @mark, or down to 0
 * when exiting the outermost evaluation scope.
 *
 * Return: void
 */
static void tmp_end(size_t mark)
{
    eval_depth--;
    tmp_delete_to(eval_depth == 0 ? 0 : mark);
}

/**
 * cli_calc_tmp_cleanup - Delete all tracked temporary calculator images
 *
 * Reaps every `_tmpcalc` image recorded in `tmp_ids` after command or
 * expression execution completes.
 *
 * Return: void
 */
void cli_calc_tmp_cleanup(void)
{
    tmp_delete_to(0);
}

/* --------------------------------------------------------
 * Token stream helpers
 * -------------------------------------------------------- */

int parse_mode = 0; // 0 for cli_parse, 1 for cli_calc_eval_line

static const char *calc_src =
    NULL; /* expression being evaluated by cli_calc_eval_bash, for messages */
char          cli_calc_lasterr[256];    /* the last error message */
static char  *calc_out          = NULL; /* when set, the result goes here instead of stdout */
static size_t calc_out_sz       = 0;
int           cli_calc_out_mode = CLI_CALC_OUT_VALUES;
int           cli_calc_mode     = 0;

#define CLI_CALC_MAX_PIXELS 1048576 /* `@calc -a`: a bash array, not a data path */

/**
 * cur_parse - Return the current token in the `cli_parse` token stream
 *
 * Provides lookahead on `parse_tokens` without advancing `parse_pos`.
 *
 * Return: Pointer to the current `cli_token` in `parse_tokens`
 */
cli_token *cur_parse(void)
{
    return &parse_tokens[parse_pos];
}

/**
 * advance_parse - Consume and return the current token in the `cli_parse` stream
 *
 * Advances `parse_pos` if tokens remain and returns the token that was at the
 * cursor prior to advancing.
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
 * Provides lookahead on `eval_tokens` without advancing `eval_pos`.
 *
 * Return: Pointer to the current `cli_token` in `eval_tokens`
 */
cli_token *cur_eval(void)
{
    return &eval_tokens[eval_pos];
}

/** cli_calc_tok_pos() / cli_calc_tok_seek() - position in the active token stream, to rewind a lookahead. */
int cli_calc_tok_pos(void)
{
    return (parse_mode == 0) ? parse_pos : eval_pos;
}

/**
 * cli_calc_tok_seek - Restore the token cursor in the active token stream
 * @pos: Token index previously obtained from `cli_calc_tok_pos()`
 *
 * Rewinds `parse_pos` or `eval_pos` (depending on `parse_mode`) when a
 * speculative pass such as fused-expression compilation bails out.
 *
 * Return: void
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

/** cli_calc_tokens_have_image() - 1 if an image name follows in the expression being evaluated. */
int cli_calc_tokens_have_image(void)
{
    for (int i = eval_pos; i < eval_ntok; i++)
    {
        if (eval_tokens[i].type == TOK_IMAGE)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * advance_eval - Consume and return the current token in the `eval_tokens` stream
 *
 * Advances `eval_pos` if tokens remain and returns the token that was at the
 * cursor prior to advancing.
 *
 * Return: Pointer to the consumed `cli_token`
 */
cli_token *advance_eval(void)
{
    cli_token *t = &eval_tokens[eval_pos];
    if (eval_pos < eval_ntok)
    {
        eval_pos++;
    }
    return t;
}


/**
 * @brief Report a parse error and set error flag
 */
void parse_errmsg(const char *msg)
{
    snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s", msg);
    if ((parse_mode == 1 || data.core.Debug > 0) &&
        parse_mode != 2) /* quiet hides banners, not errors */
    {
        if (calc_src != NULL)
        {
            fprintf(stderr, "milk: calc error: %s in '%s'\n", msg, calc_src);
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


/* --------------------------------------------------------
 * Type promotion helpers
 *
 * When mixing long and double in an operation, promote
 * the long to double.  When an image (string) is
 * involved, dispatch to the COREMOD_arith functions.
 * -------------------------------------------------------- */

/**
 * @brief Promote a val_t to double if it is long
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
 * @brief Make a long result
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
 * @brief Make a double result
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
 * @brief Make a string/image result
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
 * @brief Check if an image name is valid
 *
 * If the image is not found, reports a parse error
 * and returns 0.
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
 * @brief Allocate a temporary image name
 */
const char *alloc_tmpname(void)
{
    char *name = tmpname_ring[tmpname_next++ % CALC_TMPNAME_RING];
    long  id;
    do // never hand out the name of a live image (e.g. a user image called _tmpcalc0)
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
            return name; // untracked: reaped by the next full cleanup at worst
        }
        tmp_ids = nids;
        tmp_cap = ncap;
    }
    tmp_ids[tmp_n++] = id;
    return name;
}

/* --------------------------------------------------------
 * Public entry point
 * -------------------------------------------------------- */

/**
 * @brief Parse a CLI input token and populate
 *        data.cmdargtoken[data.cmdNBarg]
 *
 * This replaces the old yy_scan_string + yyparse +
 * yylex_destroy sequence.  The input should be a
 * single token/expression terminated by '\\n'.
 *
 * After parsing:
 *  - long result -> type = CMDARGTOKEN_TYPE_LONG
 *  - double result -> type = CMDARGTOKEN_TYPE_FLOAT
 *  - string result -> type set during parse
 *    (COMMAND, EXISTINGIMAGE, STRING, etc.)
 *    and val.string is filled
 */
void cli_parse(const char *input)
{
    parse_mode  = 0;
    parse_error = 0; // Reset parse_error for cli_parse
    eval_error  = 0; // Do not trip error abortion logic


    parse_ntok = cli_tokenize(input, parse_tokens, CLI_CALC_MAX_TOKENS);

    if (parse_ntok <= 0)
    {
        return;
    }

    parse_pos = 0;

    /* skip if only a newline */
    if (parse_tokens[0].type == TOK_NEWLINE || parse_tokens[0].type == TOK_EOF)
    {
        return;
    }

    val_t result = parse_expr(0);
    if (parse_error)
    {
        /* Fallback: if it's not a valid expression, treat it as a raw string */
        data.parseerror = 0;
        parse_error     = 0;

        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_RAWSTRING;
        snprintf(data.cmdargtoken[data.cmdNBarg].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s",
                 input);

        /* Remove trailing newline if `input` had one (which it does via snprintf earlier) */
        size_t len = strlen(data.cmdargtoken[data.cmdNBarg].val.string);
        if (len > 0 && data.cmdargtoken[data.cmdNBarg].val.string[len - 1] == '\n')
        {
            data.cmdargtoken[data.cmdNBarg].val.string[len - 1] = '\0';
        }
        return;
    }

    /* consume the trailing newline if present */
    if (cur_parse()->type == TOK_NEWLINE)
    {
        advance_parse();
    }

    /* store result in data.cmdargtoken */
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
        // The type for STRING was already set by AST logic
    }
}

/**
 * @brief Evaluate a string as a pure math expression, returning the result value silently.
 *
 * @param input     Expression string
 * @param out_type  Pointer to receive the parsed type (1=long, 2=double)
 * @param out_lval  Pointer to receive long value
 * @param out_dval  Pointer to receive double value
 * @return 1 on success (pure math), 0 on failure/string
 */
int cli_calc_eval_math_to_val(const char *input, int *out_type, long *out_lval, double *out_dval)
{
    /* Save parser state in case of re-entrant evaluation */
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
        size_t tmark  = tmp_begin();
        val_t  result = parse_expr(0);
        tmp_end(tmark);

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
                    *out_type = 2;
                }
                if (out_dval)
                {
                    *out_dval = result.dval;
                }
                success = 1;
            }
        }
    }

    /* Restore parser state */
    memcpy(eval_tokens, saved_tokens, sizeof(eval_tokens));
    parse_mode      = saved_mode;
    eval_pos        = saved_pos;
    eval_ntok       = saved_ntok;
    parse_error     = saved_perr;
    eval_error      = saved_eerr;
    data.parseerror = saved_dperr;

    return success;
}

/** pixel_line() - the pixels of image @name space-separated, numbers as @calc prints them (malloc'd);
 * NULL after a message if the image is complex or larger than CLI_CALC_MAX_PIXELS. */
static char *pixel_line(const char *name)
{
    const IMAGE *im = &data.core.image[image_ID(name, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t     n  = im->md[0].nelement;
    uint8_t      dt = im->md[0].datatype;
    char        *out;
    size_t       pos = 0;

    if (n > CLI_CALC_MAX_PIXELS || (out = (char *) malloc(n * 26 + 1)) == NULL)
    {
        fprintf(stderr, "milk: calc error: %s has %lu pixels (-a takes %d at most)\n", name,
                (unsigned long) n, CLI_CALC_MAX_PIXELS);
        return NULL;
    }
    out[0] = '\0';
#define PIX_INT(MBR)                                                                              \
    for (uint64_t i = 0; i < n; i++)                                                              \
    {                                                                                             \
        pos += (size_t) sprintf(out + pos, "%s%lld", i ? " " : "", (long long) im->array.MBR[i]); \
    }
#define PIX_REAL(MBR)                                                                \
    for (uint64_t i = 0; i < n; i++)                                                 \
    {                                                                                \
        pos += (size_t) sprintf(out + pos, "%s%.*g", i ? " " : "", cli_float_digits, \
                                (double) im->array.MBR[i]);                          \
    }
    if (dt < _DATATYPE_UINT8 || dt > _DATATYPE_DOUBLE) /* complex, half: not dispatched below */
    {
        fprintf(stderr, "milk: calc error: %s: -a takes real images\n", name);
        free(out);
        return NULL;
    }
    if (dt == _DATATYPE_FLOAT)
    {
        PIX_REAL(F)
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        PIX_REAL(D)
    }
    else
    {
        MILK_FOR_EACH_DATATYPE(dt, PIX_INT, PIX_INT(D))
    }
#undef PIX_INT
#undef PIX_REAL
    return out;
}

/** format_image_summary() - format [<name> ][<dims>] <type> for CLI_CALC_OUT_REPL. */
static void format_image_summary(const char *name, int is_assign, char *out, size_t outsz)
{
    imageID id = image_ID(name, data.core.image, data.core.NB_MAX_IMAGE);
    if (id == -1)
    {
        snprintf(out, outsz, "%s", name);
        return;
    }
    const IMAGE_METADATA *md   = &data.core.image[id].md[0];
    size_t                dpos = 0;
    char                  dims[128];
    dims[0] = '\0';
    for (uint8_t a = 0; a < md->naxis && dpos + 24 < sizeof(dims); a++)
    {
        dpos += (size_t) snprintf(dims + dpos, sizeof(dims) - dpos, "%s%lu", (a > 0) ? " x " : "",
                                  (unsigned long) md->size[a]);
    }
    const char *tstr = "?";
    switch (md->datatype)
    {
    case _DATATYPE_FLOAT:
        tstr = "f32";
        break;
    case _DATATYPE_DOUBLE:
        tstr = "f64";
        break;
    case _DATATYPE_UINT8:
        tstr = "u8";
        break;
    case _DATATYPE_INT8:
        tstr = "i8";
        break;
    case _DATATYPE_UINT16:
        tstr = "u16";
        break;
    case _DATATYPE_INT16:
        tstr = "i16";
        break;
    case _DATATYPE_UINT32:
        tstr = "u32";
        break;
    case _DATATYPE_INT32:
        tstr = "i32";
        break;
    case _DATATYPE_UINT64:
        tstr = "u64";
        break;
    case _DATATYPE_INT64:
        tstr = "i64";
        break;
    case _DATATYPE_COMPLEX_FLOAT:
        tstr = "cf32";
        break;
    case _DATATYPE_COMPLEX_DOUBLE:
        tstr = "cf64";
        break;
    case _DATATYPE_HALF:
        tstr = "f16";
        break;
    default:
        break;
    }
    if (is_assign || strncmp(name, "_tmpcalc", 8) != 0)
    {
        snprintf(out, outsz, "%s [%s] %s", name, dims, tstr);
    }
    else
    {
        snprintf(out, outsz, "[%s] %s", dims, tstr);
    }
}

/**
 * @brief Evaluate an expression for @calc or @test in bash mode.
 */
int cli_calc_eval_bash(const char *input, int is_test, int *out_rc)
{
    if (out_rc != NULL)
    {
        *out_rc = 2;
    }
    if (input == NULL)
    {
        return 0;
    }

    parse_mode = 1;
    char tbuf[8192];
    snprintf(tbuf, sizeof(tbuf), "%s\n", input);

    cli_calc_lasterr[0] = '\0';
    eval_ntok           = cli_tokenize(tbuf, eval_tokens, CLI_CALC_MAX_TOKENS);
    data.parseerror     = 0;
    parse_error         = 0;
    eval_error          = 0;
    eval_pos            = 0;

    if (eval_ntok < 0)
    {
        return 0; // tokenizer already reported the error; *out_rc stays 2
    }
    if (eval_ntok == 0 || cur_eval()->type == TOK_NEWLINE || cur_eval()->type == TOK_EOF)
    {
        if (out_rc != NULL)
        {
            *out_rc = is_test ? 1 : 0;
        }
        return 1;
    }

    int  is_assign_op = 0;
    char lhs_name[CLI_CALC_TOKEN_MAXLEN];
    lhs_name[0] = '\0';
    if (eval_ntok >= 2)
    {
        cli_token_type atype = eval_tokens[1].type;
        if (atype == TOK_EQUAL || atype == TOK_OP_PLUS_EQ || atype == TOK_OP_MINUS_EQ ||
            atype == TOK_OP_STAR_EQ || atype == TOK_OP_SLASH_EQ)
        {
            is_assign_op = 1;
            snprintf(lhs_name, sizeof(lhs_name), "%s", eval_tokens[0].sval);
        }
        else if (eval_tokens[0].type == TOK_FUNC_IMIM_IM &&
                 (strcmp(eval_tokens[0].sval, "gemm(") == 0 ||
                  strcmp(eval_tokens[0].sval, "gemv(") == 0 ||
                  strcmp(eval_tokens[0].sval, "symv(") == 0 ||
                  strcmp(eval_tokens[0].sval, "syrk(") == 0 ||
                  strcmp(eval_tokens[0].sval, "ger(") == 0 ||
                  strcmp(eval_tokens[0].sval, "axpy(") == 0 ||
                  strcmp(eval_tokens[0].sval, "scal(") == 0))
        {
            int depth = 1;
            int idx   = 1;
            while (idx < eval_ntok && depth > 0 && eval_tokens[idx].type != TOK_NEWLINE &&
                   eval_tokens[idx].type != TOK_EOF)
            {
                cli_token_type t = eval_tokens[idx].type;
                if (t == TOK_LPAREN || (t >= TOK_FUNC_D_D && t <= TOK_FUNC_COORD &&
                                        strchr(eval_tokens[idx].sval, '(') != NULL))
                {
                    depth++;
                }
                else if (t == TOK_RPAREN)
                {
                    depth--;
                }
                idx++;
            }
            if (depth == 0 && (idx >= eval_ntok || eval_tokens[idx].type == TOK_NEWLINE ||
                               eval_tokens[idx].type == TOK_EOF))
            {
                is_assign_op = 1;
                if (idx >= 2 && eval_tokens[idx - 2].type == TOK_IMAGE)
                {
                    snprintf(lhs_name, sizeof(lhs_name), "%s", eval_tokens[idx - 2].sval);
                }
            }
        }
    }

    size_t tmark           = tmp_begin();
    calc_src               = input;
    val_t result           = parse_expr(0);
    calc_src               = NULL;
    int   is_img           = (!parse_error && !eval_error && result.type == VAL_STRING &&
                              image_ID(result.sval, data.core.image, data.core.NB_MAX_IMAGE) != -1);
    char *pixels           = NULL; /* before tmp_end() deletes a temporary result image */
    int   pix_bad          = 0;
    char  img_summary[256] = "";
    if (is_img && cli_calc_out_mode == CLI_CALC_OUT_PIXELS && !is_test)
    {
        pix_bad = ((pixels = pixel_line(result.sval)) == NULL);
    }
    else if (is_img && cli_calc_out_mode == CLI_CALC_OUT_REPL && !is_test)
    {
        format_image_summary(result.sval, is_assign_op, img_summary, sizeof(img_summary));
    }
    tmp_end(tmark);

    if (parse_error || eval_error || pix_bad ||
        (cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE))
    {
        free(pixels);
        if (!eval_error)
        {
            fprintf(stderr, "milk: calc error: syntax error in '%s'\n", input);
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "syntax error");
        }
        if (out_rc != NULL)
        {
            *out_rc = 2;
        }
        return 0;
    }

    if (is_test)
    {
        int truthy = 0;
        if (result.type == VAL_LONG)
        {
            truthy = (result.lval != 0);
        }
        else if (result.type == VAL_DOUBLE)
        {
            truthy = (result.dval != 0.0 && cli_calc_isnan(result.dval) == 0.0);
        }
        else if (result.type == VAL_STRING)
        {
            truthy = (result.sval[0] != '\0' && strcmp(result.sval, "0") != 0);
        }
        if (out_rc != NULL)
        {
            *out_rc = truthy ? 0 : 1;
        }
        return 1;
    }

    if (cli_calc_out_mode == CLI_CALC_OUT_BIND ||
        cli_calc_out_mode == CLI_CALC_OUT_PIXELS) /* -v/-a: one line per statement */
    {
        const char *sv = "";
        char        num[64];
        if (pixels != NULL)
        {
            sv = pixels;
        }
        else if (result.type == VAL_LONG || result.type == VAL_DOUBLE)
        {
            if (result.type == VAL_LONG)
            {
                snprintf(num, sizeof(num), "%ld", result.lval);
            }
            else
            {
                snprintf(num, sizeof(num), "%.*g", cli_float_digits, result.dval);
            }
            sv = num;
        }
        else if (result.type == VAL_STRING && !is_img) /* an assignment gives the variable's name */
        {
            const char *v = is_assign_op ? cli_var_get(result.sval) : result.sval;
            sv            = (v != NULL) ? v : "";
        }
        printf("%s\n", sv);
        fflush(stdout);
        free(pixels);
    }
    else if (cli_calc_out_mode == CLI_CALC_OUT_REPL)
    {
        if (is_img)
        {
            printf("%s\n", img_summary);
            fflush(stdout);
        }
        else
        {
            char val_buf[CLI_CALC_TOKEN_MAXLEN + 32] = "";
            if (result.type == VAL_LONG)
            {
                snprintf(val_buf, sizeof(val_buf), "%ld", result.lval);
            }
            else if (result.type == VAL_DOUBLE)
            {
                snprintf(val_buf, sizeof(val_buf), "%.*g", cli_float_digits, result.dval);
            }
            else if (result.type == VAL_STRING)
            {
                const char *v = is_assign_op ? cli_var_get(result.sval) : result.sval;
                snprintf(val_buf, sizeof(val_buf), "%s", (v != NULL) ? v : "");
            }
            if (is_assign_op && lhs_name[0] != '\0')
            {
                printf("%s %s\n", lhs_name, val_buf);
                fflush(stdout);
            }
            else if (result.type == VAL_LONG || result.type == VAL_DOUBLE ||
                     result.type == VAL_STRING)
            {
                printf("%s\n", val_buf);
                fflush(stdout);
            }
        }
    }
    else if (!is_assign_op)
    {
        char buf[CLI_CALC_TOKEN_MAXLEN + 32] = "";
        if (result.type == VAL_LONG)
        {
            snprintf(buf, sizeof(buf), "%ld", result.lval);
        }
        else if (result.type == VAL_DOUBLE)
        {
            snprintf(buf, sizeof(buf), "%.*g", cli_float_digits, result.dval);
        }
        else if (result.type == VAL_STRING)
        {
            snprintf(buf, sizeof(buf), "%s", result.sval);
        }
        if (calc_out != NULL)
        {
            snprintf(calc_out, calc_out_sz, "%s", buf);
        }
        else if (result.type == VAL_LONG || result.type == VAL_DOUBLE || result.type == VAL_STRING)
        {
            printf("%s\n", buf);
            fflush(stdout);
        }
    }

    if (out_rc != NULL)
    {
        *out_rc = 0;
    }
    return 1;
}

/**
 * cli_calc_keep_statement - Pre-compile an image assignment statement into a reusable fused program
 * @stmt: Calculator statement string to tokenize and compile
 *
 * Tokenizes @stmt and attempts to compile a whole-image assignment (`im = ...`,
 * `im += ...`, `im -= ...`, `im *= ...`) or a standalone in-place BLAS routine
 * (`gemm`, `gemv`, `syrk`, `ger`) via `cli_calc_fuse_keep()` so `milk.calc`
 * compute units can execute it on every frame without re-parsing.
 *
 * Return: Opaque compiled fused program pointer, or NULL if @stmt cannot be fused
 */
void *cli_calc_keep_statement(const char *stmt)
{
    char tbuf[8192];

    snprintf(tbuf, sizeof(tbuf), "%s\n", stmt);
    parse_mode      = 1;
    eval_ntok       = cli_tokenize(tbuf, eval_tokens, CLI_CALC_MAX_TOKENS);
    data.parseerror = 0;
    parse_error     = 0;
    eval_error      = 0;
    if (eval_ntok >= 4 && eval_tokens[0].type == TOK_FUNC_IMIM_IM &&
        (strcmp(eval_tokens[0].sval, "gemm(") == 0 || strcmp(eval_tokens[0].sval, "gemv(") == 0 ||
         strcmp(eval_tokens[0].sval, "syrk(") == 0 || strcmp(eval_tokens[0].sval, "ger(") == 0))
    {
        int end = eval_ntok;
        while (end > 0 &&
               (eval_tokens[end - 1].type == TOK_NEWLINE || eval_tokens[end - 1].type == TOK_EOF))
        {
            end--;
        }
        if (end >= 3 && eval_tokens[end - 1].type == TOK_RPAREN &&
            eval_tokens[end - 2].type == TOK_IMAGE &&
            strchr(eval_tokens[end - 2].sval, '[') == NULL)
        {
            imageID id =
                image_ID(eval_tokens[end - 2].sval, data.core.image, data.core.NB_MAX_IMAGE);
            eval_pos = 0;
            void *k  = (id == -1) ? NULL : cli_calc_fuse_keep(&data.core.image[id], TOK_EQUAL);
            if (k != NULL && cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
            {
                free(k);
                k = NULL;
            }
            return k;
        }
    }
    eval_pos = 2;
    if (eval_ntok < 3 || eval_tokens[0].type != TOK_IMAGE ||
        strchr(eval_tokens[0].sval, '[') != NULL ||
        !(eval_tokens[1].type == TOK_EQUAL || eval_tokens[1].type == TOK_OP_PLUS_EQ ||
          eval_tokens[1].type == TOK_OP_MINUS_EQ || eval_tokens[1].type == TOK_OP_STAR_EQ))
    {
        return NULL;
    }
    imageID id = image_ID(eval_tokens[0].sval, data.core.image, data.core.NB_MAX_IMAGE);
    void   *k  = (id == -1) ? NULL : cli_calc_fuse_keep(&data.core.image[id], eval_tokens[1].type);
    if (k != NULL && cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
    {
        free(k);
        k = NULL;
    }
    return k;
}

/**
 * cli_calc_eval_str() - cli_calc_eval_bash() with the result written to @out (empty for assignments)
 * instead of stdout.
 */
int cli_calc_eval_str(const char *input, int is_test, int *out_rc, char *out, size_t outsz)
{
    if (out != NULL && outsz > 0)
    {
        out[0] = '\0';
    }
    calc_out    = out;
    calc_out_sz = outsz;
    int r       = cli_calc_eval_bash(input, is_test, out_rc);
    calc_out    = NULL;
    calc_out_sz = 0;
    return r;
}
