// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_parser.c
 * @brief `@calc` / `@test` statement evaluation, LRU statement cache, and output formatting.
 *
 * Role & Goal:
 *   Evaluates top-level `@calc` and `@test` expressions, manages the LRU kept-statement
 *   cache (`stmt_cache`), and formats output for values, REPL summaries, and bash pixel arrays.
 */

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_slice.h"
#include "cli_calc_engines.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "cli_calc_prog.h"
#include "cli_calc_tokenizer.h"
#include "libmilkdata/milk_type_dispatch.h"

static char  *calc_out    = NULL;
static size_t calc_out_sz = 0;

#define CLI_CALC_MAX_PIXELS 1048576

/** pixel_line - Format all pixels of real image @name as a space-separated string. */
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
    if (dt < _DATATYPE_UINT8 || dt > _DATATYPE_DOUBLE)
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

/** format_image_summary - Format `[<name> ][<dims>] <type>` for `CLI_CALC_OUT_REPL`. */
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
    static const struct
    {
        uint8_t     dt;
        const char *name;
    } dt_names[] = {
        { _DATATYPE_FLOAT, "f32" },          { _DATATYPE_DOUBLE, "f64" },
        { _DATATYPE_UINT8, "u8" },           { _DATATYPE_INT8, "i8" },
        { _DATATYPE_UINT16, "u16" },         { _DATATYPE_INT16, "i16" },
        { _DATATYPE_UINT32, "u32" },         { _DATATYPE_INT32, "i32" },
        { _DATATYPE_UINT64, "u64" },         { _DATATYPE_INT64, "i64" },
        { _DATATYPE_COMPLEX_FLOAT, "cf32" }, { _DATATYPE_COMPLEX_DOUBLE, "cf64" },
        { _DATATYPE_HALF, "f16" },
    };
    for (size_t i = 0; i < sizeof(dt_names) / sizeof(dt_names[0]); i++)
    {
        if (dt_names[i].dt == md->datatype)
        {
            tstr = dt_names[i].name;
            break;
        }
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

#define CALC_STMT_CACHE_SZ 64
typedef struct
{
    uint64_t hash;
    size_t   len;
    char     stmt[256];
    char     lhs[CLI_CALC_TOKEN_MAXLEN];
    void    *kept;
    uint32_t lru;
    int      is_img;
} calc_stmt_cache_entry;

static calc_stmt_cache_entry stmt_cache[CALC_STMT_CACHE_SZ];
static uint32_t              stmt_cache_clock = 0;

/** try_stmt_cache_hit - Check the LRU statement cache for @input and execute on hit. */
static int try_stmt_cache_hit(const char *input, uint64_t in_hash, size_t in_len, int *out_rc)
{
    for (int ci = 0; ci < CALC_STMT_CACHE_SZ; ci++)
    {
        calc_stmt_cache_entry *ce = &stmt_cache[ci];
        if (ce->kept != NULL && ce->hash == in_hash && ce->len == in_len &&
            memcmp(ce->stmt, input, in_len) == 0)
        {
            if (!ce->is_img && (cli_calc_out_mode != CLI_CALC_OUT_VALUES || calc_out != NULL))
            {
                return 0;
            }
            int saved_mode = parse_mode;
            parse_mode     = 2;
            int hit_ok     = (cli_calc_fused_run(ce->kept) == 0);
            parse_mode     = saved_mode;
            if (hit_ok)
            {
                cli_calc_lasterr[0] = '\0';
                ce->lru             = ++stmt_cache_clock;
                if (ce->is_img && cli_calc_out_mode == CLI_CALC_OUT_BIND)
                {
                    printf("\n");
                    fflush(stdout);
                }
                else if (ce->is_img && cli_calc_out_mode == CLI_CALC_OUT_REPL)
                {
                    char img_summary[256];
                    format_image_summary(ce->lhs, 1, img_summary, sizeof(img_summary));
                    printf("%s\n", img_summary);
                    fflush(stdout);
                }
                if (out_rc != NULL)
                {
                    *out_rc = 0;
                }
                return 1;
            }
            cli_calc_kept_free(ce->kept);
            ce->kept            = NULL;
            ce->hash            = 0;
            ce->len             = 0;
            cli_calc_lasterr[0] = '\0';
            break;
        }
    }
    return 0;
}

/** store_stmt_cache - Compile and insert a kept assignment statement into `stmt_cache`. */
static void store_stmt_cache(const char *input,
                             uint64_t    in_hash,
                             size_t      in_len,
                             const char *res_sval,
                             int         is_img)
{
    if (eval_tokens[1].type != TOK_COMMA && eval_tokens[0].type == TOK_NVAR)
    {
        eval_tokens[0].type = is_img ? TOK_IMAGE : TOK_VAR;
    }
    int   save_pos  = eval_pos;
    void *kept      = cli_calc_keep_from_eval_tokens();
    eval_pos        = save_pos;
    parse_error     = 0;
    eval_error      = 0;
    data.parseerror = 0;
    if (kept != NULL)
    {
        int victim = 0;
        for (int ci = 0; ci < CALC_STMT_CACHE_SZ; ci++)
        {
            if (stmt_cache[ci].kept == NULL)
            {
                victim = ci;
                break;
            }
            if (stmt_cache[ci].lru < stmt_cache[victim].lru)
            {
                victim = ci;
            }
        }
        cli_calc_kept_free(stmt_cache[victim].kept);
        stmt_cache[victim].hash = in_hash;
        stmt_cache[victim].len  = in_len;
        memcpy(stmt_cache[victim].stmt, input, in_len + 1);
        snprintf(stmt_cache[victim].lhs, sizeof(stmt_cache[victim].lhs), "%s", res_sval);
        stmt_cache[victim].kept   = kept;
        stmt_cache[victim].lru    = ++stmt_cache_clock;
        stmt_cache[victim].is_img = is_img;
    }
}

/** detect_assign_op - Inspect `eval_tokens` to detect top-level assignment or in-place BLAS op. */
static int detect_assign_op(char *lhs_name, size_t lhs_sz)
{
    lhs_name[0] = '\0';
    if (eval_ntok < 2)
    {
        return 0;
    }
    cli_token_type atype = eval_tokens[1].type;
    if (atype == TOK_EQUAL || atype == TOK_OP_PLUS_EQ || atype == TOK_OP_MINUS_EQ ||
        atype == TOK_OP_STAR_EQ || atype == TOK_OP_SLASH_EQ)
    {
        snprintf(lhs_name, lhs_sz, "%s", eval_tokens[0].sval);
        return 1;
    }
    if (eval_tokens[0].type == TOK_FUNC_IMIM_IM &&
        (strcmp(eval_tokens[0].sval, "gemm(") == 0 || strcmp(eval_tokens[0].sval, "gemv(") == 0 ||
         strcmp(eval_tokens[0].sval, "symv(") == 0 || strcmp(eval_tokens[0].sval, "syrk(") == 0 ||
         strcmp(eval_tokens[0].sval, "ger(") == 0 || strcmp(eval_tokens[0].sval, "axpy(") == 0 ||
         strcmp(eval_tokens[0].sval, "scal(") == 0))
    {
        int depth = 1;
        int idx   = 1;
        while (idx < eval_ntok && depth > 0 && eval_tokens[idx].type != TOK_NEWLINE &&
               eval_tokens[idx].type != TOK_EOF)
        {
            cli_token_type t = eval_tokens[idx].type;
            if (t == TOK_LPAREN ||
                (t >= TOK_FUNC_D_D && t <= TOK_FUNC_USER && strchr(eval_tokens[idx].sval, '(')))
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
            if (idx >= 2 && eval_tokens[idx - 2].type == TOK_IMAGE)
            {
                snprintf(lhs_name, lhs_sz, "%s", eval_tokens[idx - 2].sval);
            }
            return 1;
        }
    }
    return 0;
}

/** emit_calc_result - Format and print the result of `cli_calc_eval_bash()`. */
static void emit_calc_result(val_t       result,
                             int         is_img,
                             int         is_assign_op,
                             const char *lhs_name,
                             char       *pixels,
                             const char *img_summary)
{
    if (cli_calc_out_mode == CLI_CALC_OUT_BIND || cli_calc_out_mode == CLI_CALC_OUT_PIXELS)
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
        else if (result.type == VAL_STRING && !is_img)
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
}

/** strip_arg_quotes - Trim whitespace and optional outer quotes from @lp into @out. */
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

/** eval_special_directives - Handle `loadlib`, `listengines`, `def`, and `if`/`for` blocks. */
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
        if (cli_calc_ext_load_list(lpath, lerr, sizeof(lerr)) != 0)
        {
            parse_errmsg(lerr[0] != '\0' ? lerr : "failed to load extension library");
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
        int lrc = cli_calc_print_engines(pat, jmode, cli_calc_ext_get, cli_calc_ext_count());
        if (out_rc != NULL)
        {
            *out_rc = lrc;
        }
        return 1;
    }
    int udef_rc = cli_calc_try_parse_udef(input);
    if (udef_rc != 0)
    {
        if (udef_rc < 0)
        {
            parse_errmsg("invalid def syntax (expected: def f(x, ...) = expr)");
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
        char          perr[256] = "";
        unsigned long nprinted  = cli_calc_errs_printed;
        if (cli_calc_prog_exec_once(lp, NULL, 0, NULL, perr, sizeof(perr)) != 0)
        {
            if (cli_calc_errs_printed == nprinted) /* else the failed statement printed why */
            {
                parse_errmsg(perr[0] != '\0' ? perr : "block execution error");
            }
            return 0;
        }
        if (cli_calc_out_mode == CLI_CALC_OUT_BIND)
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

/** eval_bash_expr_body - Evaluate tokenized `eval_tokens` and format or test the result. */
static int eval_bash_expr_body(const char *input,
                               int         is_test,
                               uint64_t    in_hash,
                               size_t      in_len,
                               int        *out_rc)
{
    char lhs_name[CLI_CALC_TOKEN_MAXLEN];
    int  is_assign_op = detect_assign_op(lhs_name, sizeof(lhs_name));

    size_t tmark           = cli_calc_tmp_begin();
    cli_calc_src           = input;
    val_t result           = parse_expr(0);
    cli_calc_src           = NULL;
    int   is_img           = (!parse_error && !eval_error && result.type == VAL_STRING &&
                              image_ID(result.sval, data.core.image, data.core.NB_MAX_IMAGE) != -1);
    char *pixels           = NULL;
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
    cli_calc_tmp_end(tmark);

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

    if (is_test && is_img)
    {
        fprintf(
            stderr,
            "milk: calc error: the condition is an image in '%s': test imin(...) for all pixels, "
            "imax(...) for any\n",
            input);
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "the condition is an image");
        cli_calc_test_image = 1;
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

    if (is_assign_op && lhs_name[0] != '\0' && in_len > 0 && in_len < sizeof(stmt_cache[0].stmt) &&
        (is_img || result.type == VAL_LONG || result.type == VAL_DOUBLE))
    {
        store_stmt_cache(input, in_hash, in_len, is_img ? result.sval : lhs_name, is_img);
    }

    emit_calc_result(result, is_img, is_assign_op, lhs_name, pixels, img_summary);

    if (out_rc != NULL)
    {
        *out_rc = 0;
    }
    return 1;
}

/**
 * cli_calc_eval_bash - Evaluate an expression for `@calc` or `@test` in bash mode
 * @input:   Expression string to evaluate
 * @is_test: 1 if invoked as `@test`, 0 for `@calc`
 * @out_rc:  Output status code (0 on success/truthy, 1 on falsy test, 2 on error)
 *
 * Return: 1 on success, 0 on error
 */
int cli_calc_eval_bash(const char *input, int is_test, int *out_rc)
{
    cli_calc_test_image = 0;
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
        if (cli_calc_ext_load_list(env_libs, lerr, sizeof(lerr)) != 0)
        {
            parse_errmsg(lerr);
            return 0;
        }
    }

    int drc = eval_special_directives(input, is_test, out_rc);
    if (drc >= 0)
    {
        return drc;
    }

    uint64_t in_hash = 14695981039346656037ULL;
    size_t   in_len  = 0;
    for (const unsigned char *p = (const unsigned char *) input; *p != '\0'; p++)
    {
        in_hash = (in_hash ^ (uint64_t) *p) * 1099511628211ULL;
        in_len++;
    }
    if (!is_test && cli_calc_out_mode != CLI_CALC_OUT_PIXELS && in_len > 0 &&
        in_len < sizeof(stmt_cache[0].stmt) && try_stmt_cache_hit(input, in_hash, in_len, out_rc))
    {
        return 1;
    }

    parse_mode = 1;
    char tbuf[8192];
    cli_calc_rewrite_local_prefix(input, tbuf, sizeof(tbuf));

    cli_calc_lasterr[0] = '\0';
    eval_ntok           = cli_tokenize(tbuf, eval_tokens, CLI_CALC_MAX_TOKENS);
    data.parseerror     = 0;
    parse_error         = 0;
    eval_error          = 0;
    eval_pos            = 0;

    if (eval_ntok < 0)
    {
        return 0;
    }
    if (eval_ntok == 0 || cur_eval()->type == TOK_NEWLINE || cur_eval()->type == TOK_EOF)
    {
        if (out_rc != NULL)
        {
            *out_rc = is_test ? 1 : 0;
        }
        return 1;
    }
    if (!is_test && eval_ntok >= 3 &&
        (eval_tokens[1].type == TOK_COMMA ||
         (eval_tokens[1].type == TOK_EQUAL && eval_tokens[2].type == TOK_FUNC_USER)))
    {
        int mrc = cli_calc_eval_multi_assign_bash(input, out_rc);
        if (mrc >= 0)
        {
            if (mrc == 1 && in_len > 0 && in_len < sizeof(stmt_cache[0].stmt))
            {
                store_stmt_cache(input, in_hash, in_len, "", 0);
            }
            return mrc;
        }
    }

    return eval_bash_expr_body(input, is_test, in_hash, in_len, out_rc);
}

/**
 * cli_calc_eval_str - Evaluate an expression with output written to @out instead of stdout
 * @input:   Expression string to evaluate
 * @is_test: 1 if invoked as `@test`
 * @out_rc:  Output status code
 * @out:     Destination string buffer
 * @outsz:   Size of @out in bytes
 *
 * Return: 1 on success, 0 on error
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
