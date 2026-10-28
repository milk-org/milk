// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_parser_assign.c
 * @brief Kept statement compilation, multi-target assignment unpacking, and variable/image storage.
 *
 * Role & Goal:
 *   Compiles single- and multi-target statements into reusable `fkept` programs
 *   (`cli_calc_keep_from_eval_tokens`, `cli_calc_keep_statement`), evaluates multi-target
 *   unpacking (`a, b = ...`, `m, s = istats(im)`, `cx, cy = icentroid(im)`), and stores
 *   assigned values into CLI variables or images (`cli_calc_store_var_or_im`).
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "cli_calc_prog.h"
#include "cli_calc_tokenizer.h"

int cli_calc_prog_default_shared = 0;

/**
 * cli_calc_rewrite_local_prefix - Rewrite a leading `local x = ...` to `@L:x = ...`
 * @input: Input statement string
 * @out:   Destination buffer (with trailing newline)
 * @outsz: Size of @out in bytes
 */
void cli_calc_rewrite_local_prefix(const char *input, char *out, size_t outsz)
{
    const char *p = input;
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }
    if (strncmp(p, "local", 5) == 0 && (p[5] == ' ' || p[5] == '\t'))
    {
        p += 5;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (isalpha((unsigned char) *p) || *p == '_')
        {
            snprintf(out, outsz, "@L:%s\n", p);
            return;
        }
    }
    snprintf(out, outsz, "%s\n", input);
}

/**
 * keep_multi_from_eval_tokens - Compile a multi-LHS statement in `eval_tokens` into a kept program
 *
 * Return: Heap-allocated `fkept` pointer, or NULL if not fusible
 */
static void *keep_multi_from_eval_tokens(void)
{
    IMAGE *dst_im[4]  = { NULL, NULL, NULL, NULL };
    int    dst_vid[4] = { -1, -1, -1, -1 };
    int    ndst       = 0;
    int    pos        = 0;
    while (ndst < 4 && pos < eval_ntok)
    {
        const cli_token *t = &eval_tokens[pos];
        if (t->type != TOK_VAR && t->type != TOK_NVAR && t->type != TOK_IMAGE)
        {
            cli_calc_fuse_reject("the target list is not a list of names");
            return NULL;
        }
        ndst++;
        pos++;
        if (pos < eval_ntok && eval_tokens[pos].type == TOK_COMMA)
        {
            pos++;
        }
        else
        {
            break;
        }
    }
    if (ndst < 2 || pos >= eval_ntok || eval_tokens[pos].type != TOK_EQUAL)
    {
        cli_calc_fuse_reject((ndst >= 4) ? "more than 4 targets (FUSE_MAXDST)"
                                         : "not an assignment to a list of names");
        return NULL;
    }
    if (pos + 1 < eval_ntok && eval_tokens[pos + 1].type == TOK_FUNC_USER)
    {
        const milk_calc_ext_entry_t *e = cli_calc_ext_get((int) eval_tokens[pos + 1].val_l);
        if (e != NULL && e->kind == MILK_EXT_OP)
        {
            char lhs_names[MILK_CALC_MAX_PORTS][STRINGMAXLEN_IMAGE_NAME];
            int  is_loc[MILK_CALC_MAX_PORTS] = { 0 };
            int  stmt_loc                    = (strncmp(eval_tokens[0].sval, "@L:", 3) == 0);
            for (int j = 0; j < ndst; j++)
            {
                const cli_token      *t  = &eval_tokens[j * 2];
                FPS_STREAMNAME_PARSED sp = fps_streamname_parse(t->sval);
                snprintf(lhs_names[j], sizeof(lhs_names[j]), "%s", t->sval);
                is_loc[j] = (sp.loc == 'L') ? 1 : (sp.loc == 'S') ? 0 : stmt_loc;
            }
            eval_pos = pos + 1;
            return cli_calc_fuse_keep_user_op(ndst,
                                              (const char (*)[STRINGMAXLEN_IMAGE_NAME]) lhs_names,
                                              is_loc, cli_calc_prog_default_shared);
        }
    }
    int is_istats = (pos + 1 < eval_ntok && eval_tokens[pos + 1].type == TOK_FUNC_IM_D &&
                     (strcmp(eval_tokens[pos + 1].sval, "istats(") == 0 ||
                      strcmp(eval_tokens[pos + 1].sval, "icentroid(") == 0));
    for (int j = 0; j < ndst; j++)
    {
        const cli_token      *t  = &eval_tokens[j * 2];
        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(t->sval);
        const char *bare = (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : t->sval;
        imageID     id   = is_istats ? -1 : image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE);
        long        vid  = variable_ID(bare);
        if (id != -1 && strchr(t->sval, '[') == NULL)
        {
            dst_im[j] = &data.core.image[id];
        }
        else if (vid >= 0)
        {
            dst_vid[j] = (int) vid;
        }
        else
        {
            return NULL;
        }
    }
    eval_pos = pos + 1;
    void *k  = cli_calc_fuse_keep_multi(ndst, dst_im, dst_vid);
    if (k != NULL && cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
    {
        cli_calc_kept_free(k);
        k = NULL;
    }
    return k;
}

/**
 * cli_calc_keep_from_eval_tokens - Compile the current `eval_tokens` stream into a reusable `fkept`
 *
 * Return: Heap-allocated `fkept` pointer, or NULL if not fusible
 */
void *cli_calc_keep_from_eval_tokens(void)
{
    if (eval_ntok >= 5 && eval_tokens[1].type == TOK_COMMA)
    {
        return keep_multi_from_eval_tokens();
    }
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
                cli_calc_kept_free(k);
                k = NULL;
            }
            return k;
        }
    }
    eval_pos = 2;
    if (eval_ntok < 3 ||
        !(eval_tokens[1].type == TOK_EQUAL || eval_tokens[1].type == TOK_OP_PLUS_EQ ||
          eval_tokens[1].type == TOK_OP_MINUS_EQ || eval_tokens[1].type == TOK_OP_STAR_EQ ||
          eval_tokens[1].type == TOK_OP_SLASH_EQ))
    {
        cli_calc_fuse_reject("not an assignment to one image or variable");
        return NULL;
    }
    if (eval_tokens[1].type == TOK_EQUAL && eval_tokens[2].type == TOK_FUNC_USER)
    {
        const milk_calc_ext_entry_t *e = cli_calc_ext_get((int) eval_tokens[2].val_l);
        if (e != NULL && e->kind == MILK_EXT_OP)
        {
            char                  lhs_names[1][STRINGMAXLEN_IMAGE_NAME];
            FPS_STREAMNAME_PARSED sp = fps_streamname_parse(eval_tokens[0].sval);
            snprintf(lhs_names[0], sizeof(lhs_names[0]), "%s", eval_tokens[0].sval);
            int is_loc[1] = { (sp.loc == 'L') ? 1 : 0 };
            eval_pos      = 2;
            return cli_calc_fuse_keep_user_op(1,
                                              (const char (*)[STRINGMAXLEN_IMAGE_NAME]) lhs_names,
                                              is_loc, cli_calc_prog_default_shared);
        }
    }
    void *k = NULL;
    if (eval_tokens[0].type == TOK_IMAGE && strchr(eval_tokens[0].sval, '[') != NULL)
    {
        k = cli_calc_fuse_keep_tok(NULL, eval_tokens[0].sval, eval_tokens[1].type);
    }
    else if (eval_tokens[0].type == TOK_IMAGE)
    {
        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(eval_tokens[0].sval);
        const char           *bare =
            (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : eval_tokens[0].sval;
        imageID id = image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE);
        k = (id == -1) ? NULL : cli_calc_fuse_keep(&data.core.image[id], eval_tokens[1].type);
    }
    else if (eval_tokens[0].type == TOK_VAR)
    {
        long vid = variable_ID(eval_tokens[0].sval);
        k        = (vid < 0) ? NULL : cli_calc_fuse_keep_var((int) vid, eval_tokens[1].type);
    }
    else
    {
        char why[160];
        snprintf(why, sizeof(why), "destination %.60s is not an existing image or numeric variable",
                 eval_tokens[0].sval);
        cli_calc_fuse_reject(why);
    }
    if (k != NULL && cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
    {
        cli_calc_fuse_reject("text after the expression (several values or statements)");
        cli_calc_kept_free(k);
        k = NULL;
    }
    return k;
}

/**
 * cli_calc_keep_statement - Pre-compile an image assignment statement into a reusable fused program
 * @stmt: Statement text to compile
 *
 * Return: Heap-allocated `fkept` pointer, or NULL if not fusible
 */
void *cli_calc_keep_statement(const char *stmt)
{
    char tbuf[8192];

    cli_calc_rewrite_local_prefix(stmt, tbuf, sizeof(tbuf));
    cli_calc_fuse_reject_reset();
    parse_mode      = 1;
    eval_ntok       = cli_tokenize(tbuf, eval_tokens, CLI_CALC_MAX_TOKENS);
    data.parseerror = 0;
    parse_error     = 0;
    eval_error      = 0;
    return cli_calc_keep_from_eval_tokens();
}

/**
 * eval_multi_istats_or_centroid - Evaluate `m, s = istats(im)` or `cx, cy = icentroid(im)`
 * @lhs:    LHS target names
 * @ndst:   Number of LHS targets
 * @eq_pos: Index of `=` token in `eval_tokens`
 *
 * Return: 1 on success, 0 on error
 */
static int eval_multi_istats_or_centroid(char lhs[][CLI_CALC_TOKEN_MAXLEN], int ndst, int eq_pos)
{
    long vids[4];
    for (int j = 0; j < ndst; j++)
    {
        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(lhs[j]);
        const char *bare = (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : lhs[j];
        vids[j]          = variable_ID(bare);
        if (vids[j] < 0)
        {
            vids[j] = create_variable_ID(bare, 0.0);
        }
    }
    void *k = cli_calc_keep_from_eval_tokens();
    if (k == NULL)
    {
        parse_errmsg((strcmp(eval_tokens[eq_pos + 1].sval, "icentroid(") == 0)
                         ? "invalid icentroid() unpacking"
                         : "invalid istats() unpacking");
        return 0;
    }
    int run_rc = cli_calc_fused_run(k);
    cli_calc_kept_free(k);
    if (run_rc != 0)
    {
        return 0;
    }
    for (int j = 0; j < ndst; j++)
    {
        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(lhs[j]);
        const char *bare = (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : lhs[j];
        double      dv   = (data.core.variable[vids[j]].type == 1)
                               ? (double) data.core.variable[vids[j]].value.l
                               : data.core.variable[vids[j]].value.f;
        char        numv[64];
        snprintf(numv, sizeof(numv), "%.*g", cli_float_digits, dv);
        cli_var_set(bare, numv);
    }
    return 1;
}

/** store_multi_target - Store @v into target @name of `a, b = ...`: an existing image as by `name = v`. */
static void store_multi_target(const char *name, val_t v)
{
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(name);
    const char *bare = (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : name;

    if (image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE) != -1)
    {
        cli_calc_eval_image_whole_assign(name, TOK_EQUAL, v);
        return;
    }
    cli_calc_store_var_or_im(name, v, 1);
}

/**
 * eval_multi_rhs_list - Fallback comma-separated RHS evaluation for `a, b = expr1, expr2`
 * @input:  Original statement string
 * @lhs:    LHS target names
 * @ndst:   Number of LHS targets
 * @eq_pos: Index of `=` token in `eval_tokens`
 *
 * Return: 1 on success, 0 on error
 */
static int eval_multi_rhs_list(const char *input,
                               char        lhs[][CLI_CALC_TOKEN_MAXLEN],
                               int         ndst,
                               int         eq_pos)
{
    parse_error     = 0;
    eval_error      = 0;
    data.parseerror = 0;
    eval_pos        = eq_pos + 1;
    size_t tmark    = cli_calc_tmp_begin();
    cli_calc_src    = input;
    val_t rhs[8];
    for (int j = 0; j < ndst; j++)
    {
        if (j > 0)
        {
            if (cur_eval()->type != TOK_COMMA)
            {
                parse_errmsg("Expected ',' in multi-target assignment");
                break;
            }
            advance_eval();
        }
        rhs[j] = parse_expr(0);
        if (parse_error || eval_error)
        {
            break;
        }
    }
    cli_calc_src = NULL;
    if (!parse_error && !eval_error &&
        (cur_eval()->type == TOK_EOF || cur_eval()->type == TOK_NEWLINE))
    {
        for (int j = 0; j < ndst && !parse_error && !eval_error; j++)
        {
            store_multi_target(lhs[j], rhs[j]);
        }
    }
    else if (!eval_error)
    {
        parse_errmsg("syntax error in multi-target assignment");
    }
    cli_calc_tmp_end(tmark);
    return (parse_error || eval_error) ? 0 : 1;
}

/** sync_kept_multi_vars_to_cli - Mirror scalar variables updated by a kept op into `cli_var_set()`. */
static void sync_kept_multi_vars_to_cli(char lhs[][CLI_CALC_TOKEN_MAXLEN], int ndst)
{
    for (int j = 0; j < ndst; j++)
    {
        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(lhs[j]);
        const char *bare = (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : lhs[j];
        long        vid  = variable_ID(bare);
        if (vid >= 0)
        {
            double dv = (data.core.variable[vid].type == 1)
                            ? (double) data.core.variable[vid].value.l
                            : data.core.variable[vid].value.f;
            char   numv[64];
            snprintf(numv, sizeof(numv), "%.*g", cli_float_digits, dv);
            cli_var_set(bare, numv);
        }
    }
}

/**
 * cli_calc_eval_multi_assign_bash - Evaluate multi-target unpacking `a, b = ...` in bash mode
 * @input:  Statement string
 * @out_rc: Output status code
 *
 * Return: 1 on success, 0 on error, -1 if not a multi-target assignment
 */
int cli_calc_eval_multi_assign_bash(const char *input, int *out_rc)
{
    char lhs[8][CLI_CALC_TOKEN_MAXLEN];
    int  ndst = 0;
    int  pos  = 0;
    while (ndst < 8 && pos < eval_ntok)
    {
        const cli_token *t = &eval_tokens[pos];
        if (t->type != TOK_VAR && t->type != TOK_NVAR && t->type != TOK_IMAGE)
        {
            return -1;
        }
        snprintf(lhs[ndst], sizeof(lhs[ndst]), "%s", t->sval);
        ndst++;
        pos++;
        if (pos < eval_ntok && eval_tokens[pos].type == TOK_COMMA)
        {
            pos++;
        }
        else
        {
            break;
        }
    }
    int eq_pos = pos;
    int is_uop = 0;
    if (eq_pos < eval_ntok && eval_tokens[eq_pos].type == TOK_EQUAL && eq_pos + 1 < eval_ntok &&
        eval_tokens[eq_pos + 1].type == TOK_FUNC_USER)
    {
        const milk_calc_ext_entry_t *e = cli_calc_ext_get((int) eval_tokens[eq_pos + 1].val_l);
        is_uop                         = (e != NULL && e->kind == MILK_EXT_OP);
    }
    if ((ndst < 2 && !is_uop) || pos >= eval_ntok || eval_tokens[pos].type != TOK_EQUAL)
    {
        return -1;
    }
    if (eq_pos + 1 < eval_ntok && eval_tokens[eq_pos + 1].type == TOK_FUNC_IM_D &&
        (strcmp(eval_tokens[eq_pos + 1].sval, "istats(") == 0 ||
         strcmp(eval_tokens[eq_pos + 1].sval, "icentroid(") == 0))
    {
        if (!eval_multi_istats_or_centroid(lhs, ndst, eq_pos))
        {
            return 0;
        }
    }
    else
    {
        void *k = cli_calc_keep_from_eval_tokens();
        if (k != NULL)
        {
            int run_rc = cli_calc_fused_run(k);
            cli_calc_kept_free(k);
            if (run_rc == 0)
            {
                sync_kept_multi_vars_to_cli(lhs, ndst);
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
            if (is_uop)
            {
                return 0;
            }
        }
        else if (is_uop)
        {
            if (!parse_error && !eval_error)
            {
                parse_errmsg("invalid operator call");
            }
            return 0;
        }
        if (!eval_multi_rhs_list(input, lhs, ndst, eq_pos))
        {
            return 0;
        }
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

/**
 * cli_calc_store_var_or_im - Store a resolved value @v into variable or image name @varname
 * @varname:             Target variable or image name
 * @v:                   Value to store
 * @sync_cli_var_always: 1 to always mirror numeric values into `cli_var_set()`
 *
 * Return: Stored `val_t`
 */
val_t cli_calc_store_var_or_im(const char *varname, val_t v, int sync_cli_var_always)
{
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(varname);
    const char *dst_bare = (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : varname;

    if (v.type == VAL_STRING)
    {
        if (image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE) != -1)
        {
            int want_shm = (sp.loc == 'S') ? 1 : 0;

            if (variable_ID(dst_bare) >= 0) /* the image would need the variable's name */
            {
                char msg[CLI_CALC_TOKEN_MAXLEN + 64];
                if (strncmp(dst_bare, "__g", 3) == 0) /* the condition of an if, or a for's bound */
                {
                    snprintf(msg, sizeof(msg),
                             "an if or for needs a number, not an image: test "
                             "imin(...) for all pixels, imax(...) for any");
                }
                else
                {
                    snprintf(msg, sizeof(msg),
                             "'%s' is a variable (calculator or bash), not an image", dst_bare);
                }
                parse_errmsg(msg);
                return mk_double(0);
            }
            if (strncmp(v.sval, "_tmpcalc", 8) == 0 && !want_shm)
            {
                chname_image_ID(v.sval, dst_bare);
            }
            else
            {
                copy_image_ID(v.sval, dst_bare, want_shm);
            }
            if (data.core.Debug > 0)
            {
                printf("changing name\n");
            }
            return mk_string(dst_bare);
        }
        const char *strval = cli_var_get(v.sval);
        if (strval == NULL)
        {
            strval = v.sval;
        }
        cli_var_set(dst_bare, strval);
        return mk_string(dst_bare);
    }

    if (v.type == VAL_LONG)
    {
        if (data.core.Debug > 0 && !sync_cli_var_always)
        {
            printf("creating long\n");
        }
        create_variable_long_ID(dst_bare, v.lval);
        char numv[64];
        snprintf(numv, sizeof(numv), "%ld", v.lval);
        if (sync_cli_var_always || parse_mode == 1)
        {
            cli_var_set(dst_bare, numv);
        }
        return mk_long(v.lval);
    }

    if (data.core.Debug > 0 && !sync_cli_var_always)
    {
        printf("creating double\n");
    }
    double dv = to_double(v);
    create_variable_ID(dst_bare, dv);
    char numv[64];
    snprintf(numv, sizeof(numv), "%.*g", cli_float_digits, dv);
    if (sync_cli_var_always || parse_mode == 1)
    {
        cli_var_set(dst_bare, numv);
    }
    return mk_double(dv);
}

/**
 * cli_calc_eval_existing_var_assign - Evaluate `=` or compound `op=` to an existing variable
 * @varname: Target variable name
 * @aop:     Assignment operator token
 * @v:       RHS value
 *
 * Return: Assigned `val_t`
 */
val_t cli_calc_eval_existing_var_assign(const char *varname, cli_token_type aop, val_t v)
{
    if (aop != TOK_EQUAL)
    {
        long   vid = variable_ID(varname);
        double old = 0.0;
        if (vid == -1)
        {
            const char *cval = cli_var_get(varname);
            if (cval == NULL)
            {
                cval = getenv(varname);
            }
            if (cval != NULL)
            {
                old = strtod(cval, NULL);
            }
            else
            {
                parse_errmsg("Variable not found for compound assign");
                return mk_double(0);
            }
        }
        else if (data.core.variable[vid].type == 1)
        {
            old = (double) data.core.variable[vid].value.l;
        }
        else
        {
            old = data.core.variable[vid].value.f;
        }

        double nv = to_double(v);
        switch (aop)
        {
        case TOK_OP_PLUS_EQ:
            nv = old + nv;
            break;
        case TOK_OP_MINUS_EQ:
            nv = old - nv;
            break;
        case TOK_OP_STAR_EQ:
            nv = old * nv;
            break;
        case TOK_OP_SLASH_EQ:
            if (nv == 0.0)
            {
                parse_errmsg("Division by zero");
                return mk_double(0);
            }
            nv = old / nv;
            break;
        default:
            break;
        }
        if (v.type == VAL_LONG && ((vid != -1 && data.core.variable[vid].type == 1) || vid == -1) &&
            aop != TOK_OP_SLASH_EQ)
        {
            v = mk_long((long) nv);
        }
        else
        {
            v = mk_double(nv);
        }
    }

    return cli_calc_store_var_or_im(varname, v, 1);
}
