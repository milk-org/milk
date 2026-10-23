// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_eval.c
 * @brief Pratt (precedence-climbing) expression parser and primary atom evaluator.
 *
 * Role & Goal:
 *   Evaluates CLI arithmetic and assignment expressions using a precedence-climbing
 *   parser (`parse_expr()`) and a primary atom handler (`parse_primary()`).
 *   Delegates N-D image slice bounds and in-place assignments to `cli_calc_slice.c`,
 *   built-in function calls to `cli_calc_functions.c`, and binary operator
 *   dispatch to `cli_calc_binops.c`.
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_slice.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"

/* Operator precedence and associativity: cli_calc_scalar.c (shared with the in-shell evaluator) */

/**
 * is_assign_op - Test whether a token type is an assignment operator
 * @t: Token type to inspect
 *
 * Identifies simple (`=`) and compound (`+=`, `-=`, `*=`, `/=`) assignment
 * tokens so primary expression handlers can branch into assignment evaluation.
 *
 * Return: 1 if @t is an assignment operator, 0 otherwise
 */
static inline int is_assign_op(cli_token_type t)
{
    return (t == TOK_EQUAL) || (t == TOK_OP_PLUS_EQ) || (t == TOK_OP_MINUS_EQ) ||
           (t == TOK_OP_STAR_EQ) || (t == TOK_OP_SLASH_EQ);
}

/**
 * is_func_token - Test whether a token type represents a built-in function call
 * @t: Token type to inspect
 *
 * Checks if @t belongs to any of the calculator's built-in scalar, image,
 * string, complex, or coordinate function categories (whose tokens include the
 * opening parenthesis).
 *
 * Return: 1 if @t is a function token, 0 otherwise
 */
static inline int is_func_token(cli_token_type t)
{
    return (t == TOK_FUNC_D_D) || (t == TOK_FUNC_DD_D) || (t == TOK_FUNC_DDD_D) ||
           (t == TOK_FUNC_IM_D) || (t == TOK_FUNC_IMD_D) || (t == TOK_FUNC_WHERE) ||
           (t == TOK_FUNC_IMIM_D) || (t == TOK_FUNC_IMIM_IM) || (t == TOK_FUNC_S_D) ||
           (t == TOK_FUNC_S_S) || (t == TOK_FUNC_SDD_S) || (t == TOK_FUNC_SSS_S) ||
           (t == TOK_FUNC_D_S) || (t == TOK_FUNC_IM_L) || (t == TOK_FUNC_CPLX) ||
           (t == TOK_FUNC_COORD);
}

static int skip_expr(int min_prec);

/** skip_primary() - consume one atom like parse_primary(), evaluating nothing; 0 or -1 on syntax error. */
static int skip_primary(void)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    cli_token_type t                 = cur_func()->type;

    advance_func();
    switch (t)
    {
    case TOK_LONG:
    case TOK_DOUBLE:
    case TOK_COMMAND:
        return 0;
    case TOK_OP_MINUS:
    case TOK_OP_BNOT:
    case TOK_OP_NOT:
        return skip_primary();
    case TOK_VAR:
    case TOK_NVAR:
    case TOK_IMAGE:
        if (is_assign_op(cur_func()->type))
        {
            advance_func();
            return skip_expr(0);
        }
        return 0;
    case TOK_LPAREN:
        if (skip_expr(0) != 0)
        {
            return -1;
        }
        break;
    default:
        if (!is_func_token(t))
        {
            parse_errmsg("Unexpected token");
            return -1;
        }
        do // function tokens include their '('
        {
            if (skip_expr(0) != 0)
            {
                return -1;
            }
        } while (cur_func()->type == TOK_COMMA && advance_func());
        break;
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return -1;
    }
    advance_func();
    return 0;
}

/** skip_expr() - consume an expression like parse_expr(@min_prec) without side effects (lazy operands). */
static int skip_expr(int min_prec)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();

    if (skip_primary() != 0)
    {
        return -1;
    }
    for (;;)
    {
        cli_token_type op   = cur_func()->type;
        int            prec = cli_calc_op_prec(op);
        if (prec < min_prec)
        {
            return 0;
        }
        advance_func();
        if (op == TOK_OP_QUESTION)
        {
            if (skip_expr(0) != 0)
            {
                return -1;
            }
            if (cur_func()->type != TOK_OP_COLON)
            {
                parse_errmsg("Expected ':' in ternary");
                return -1;
            }
            advance_func();
        }
        if (skip_expr((op == TOK_OP_QUESTION)       ? 0
                      : cli_calc_op_right_assoc(op) ? prec
                                                    : prec + 1) != 0)
        {
            return -1;
        }
    }
}

/**
 * @brief Evaluate unary minus `-expr` on scalar or image operands.
 */
static val_t eval_unary_minus(void)
{
    val_t v = parse_primary();
    if (parse_error || eval_error)
    {
        return mk_long(0);
    }
    if (v.type == VAL_LONG)
    {
        if (data.core.Debug > 0)
        {
            printf("-long\n");
        }
        return mk_long(-v.lval);
    }
    if (v.type == VAL_DOUBLE)
    {
        if (data.core.Debug > 0)
        {
            printf("-double\n");
        }
        return mk_double(-v.dval);
    }
    if (v.type == VAL_STRING)
    {
        if (data.core.Debug > 0)
        {
            printf("-image\n");
        }
        if (!check_image(v.sval))
        {
            return mk_double(0);
        }
        const IMAGE *z = cli_calc_complex_image(&v);
        if (z != NULL)
        {
            return cli_calc_complex_neg(z);
        }
        const char *tmpn = alloc_tmpname();
        arith_image_cstsubm(v.sval, 0.0, tmpn);
        return mk_string(tmpn);
    }
    parse_errmsg("Cannot negate expression");
    return mk_double(0);
}

/**
 * @brief Store a resolved value @v into variable or image name @varname.
 */
static val_t store_variable_or_image_value(const char *varname, val_t v, int sync_cli_var_always)
{
    if (v.type == VAL_STRING)
    {
        if (image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE) != -1)
        {
            FPS_STREAMNAME_PARSED sp = fps_streamname_parse(varname);
            const char           *dst_bare =
                (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : varname;
            int want_shm = (sp.loc == 'S') ? 1 : 0;

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
        cli_var_set(varname, strval);
        return mk_string(varname);
    }

    if (v.type == VAL_LONG)
    {
        if (data.core.Debug > 0 && !sync_cli_var_always)
        {
            printf("creating long\n");
        }
        create_variable_long_ID(varname, v.lval);
        char numv[64];
        snprintf(numv, sizeof(numv), "%ld", v.lval);
        if (sync_cli_var_always || parse_mode == 1)
        {
            cli_var_set(varname, numv);
        }
        return mk_long(v.lval);
    }

    if (data.core.Debug > 0 && !sync_cli_var_always)
    {
        printf("creating double\n");
    }
    double dv = to_double(v);
    create_variable_ID(varname, dv);
    char numv[64];
    snprintf(numv, sizeof(numv), "%.*g", cli_float_digits, dv);
    if (sync_cli_var_always || parse_mode == 1)
    {
        cli_var_set(varname, numv);
    }
    return mk_double(dv);
}

/**
 * @brief Evaluate assignment or compound assignment (`=`, `+=`, `-=`, `*=`, `/=`) to an existing variable.
 */
static val_t eval_existing_var_assign(const char *varname, cli_token_type aop, val_t v)
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

    return store_variable_or_image_value(varname, v, 1);
}

/**
 * @brief Resolve an existing variable read reference `TOK_VAR`.
 */
val_t eval_existing_var_ref(const char *varname)
{
    long vID = variable_ID(varname);
    if (vID == -1)
    {
        const char *cval = cli_var_get(varname);
        if (cval == NULL)
        {
            cval = getenv(varname);
        }
        if (cval != NULL)
        {
            char *endp = NULL;
            long  lval = strtol(cval, &endp, 0);
            if (*endp == '\0' && endp != cval)
            {
                return mk_long(lval);
            }
            double dval = strtod(cval, &endp);
            if (*endp == '\0' && endp != cval)
            {
                return mk_double(dval);
            }
            return mk_string(cval);
        }
        char msg[2048];
        snprintf(msg, sizeof(msg), "Variable '%s' not found", varname);
        parse_errmsg(msg);
        return mk_double(0);
    }

    if (data.core.variable[vID].type == 1)
    {
        return mk_long(data.core.variable[vID].value.l);
    }
    if (data.core.variable[vID].type == 2)
    {
        return mk_string(data.core.variable[vID].value.s);
    }
    return mk_double(data.core.variable[vID].value.f);
}

/**
 * @brief Handle a `TOK_NVAR` token (new variable assignment or standalone string/image argument).
 */
static val_t eval_nvar_token(const char *varname,
                             int         literal,
                             cli_token *(*cur_func)(void),
                             cli_token *(*advance_func)(void) )
{
    if (cur_func()->type == TOK_EQUAL)
    {
        advance_func();
        val_t v = parse_expr(0);
        if (parse_error || eval_error)
        {
            return v;
        }
        return store_variable_or_image_value(varname, v, 0);
    }

    if (parse_mode == 0)
    {
        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_STRING;
        if (data.core.Debug > 0)
        {
            printf("this is a string (new variable/image)\n");
        }
    }
    else if (strchr(varname, '[') != NULL)
    {
        char bare[200];
        char btext[200];
        int  has_brk =
            imgid_slice_split_name(varname, bare, (int) sizeof(bare), btext, (int) sizeof(btext));
        parse_errmsg(has_brk ? "Source image not found for slice"
                             : "Malformed slice: missing closing ']'");
        return mk_double(0);
    }
    else if (!literal)
    {
        char msg[CLI_CALC_TOKEN_MAXLEN + 32];
        snprintf(msg, sizeof(msg), "undefined identifier '%s'", varname);
        parse_errmsg(msg);
        return mk_double(0);
    }
    return mk_string(varname);
}

/**
 * @brief Handle a `TOK_IMAGE` token (whole-image/slice assignment or image/slice read).
 */
static val_t eval_image_token(const char *imname,
                              cli_token *(*cur_func)(void),
                              cli_token *(*advance_func)(void) )
{
    cli_token_type aop = cur_func()->type;
    if (is_assign_op(aop))
    {
        advance_func();
        imageID id = (strchr(imname, '[') == NULL)
                         ? image_ID(imname, data.core.image, data.core.NB_MAX_IMAGE)
                         : -1;
        int     fr = (id != -1) ? cli_calc_fuse_assign(&data.core.image[id], aop) : 0;
        if (fr != 0)
        {
            return (fr > 0) ? mk_string(imname) : mk_double(0);
        }
        val_t v = parse_expr(0);
        if (parse_error || eval_error)
        {
            return v;
        }
        if (strchr(imname, '[') != NULL)
        {
            return cli_calc_eval_image_slice_assign(imname, aop, v);
        }
        return cli_calc_eval_image_whole_assign(imname, aop, v);
    }

    if (!eval_error)
    {
        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_EXISTINGIMAGE;
        if (data.core.Debug > 0)
        {
            printf("this is a string (existing image)\n");
        }
    }
    if (strchr(imname, '[') != NULL)
    {
        return cli_calc_eval_image_slice_read(imname);
    }
    return mk_string(imname);
}

/**
 * @brief Parse a primary (non-operator) expression atom.
 */
val_t parse_primary(void)
{
    if (parse_error || eval_error)
    {
        return mk_long(0);
    }

    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    cli_token *t                     = cur_func();

    if (t->type == TOK_LONG)
    {
        advance_func();
        if (data.core.Debug > 0)
        {
            printf("this is a long\n");
        }
        return mk_long(t->val_l);
    }

    if (t->type == TOK_DOUBLE)
    {
        advance_func();
        if (data.core.Debug > 0)
        {
            printf("this is a double\n");
        }
        return mk_double(t->val_d);
    }

    if (t->type == TOK_OP_MINUS)
    {
        advance_func();
        return eval_unary_minus();
    }

    if (t->type == TOK_OP_BNOT)
    {
        advance_func();
        val_t v = parse_primary();
        if (parse_error || eval_error)
        {
            return mk_long(0);
        }
        long iv = (v.type == VAL_LONG) ? v.lval : (long) v.dval;
        return mk_long(~iv);
    }

    if (t->type == TOK_OP_NOT)
    {
        advance_func();
        val_t v = parse_primary();
        if (parse_error || eval_error)
        {
            return mk_long(0);
        }
        if (v.type != VAL_STRING)
        {
            return mk_long(to_double(v) == 0.0);
        }
        if (!check_image(v.sval))
        {
            return mk_long(0);
        }
        const char *tmpn = alloc_tmpname();
        arith_image_cstteste(v.sval, 0.0, tmpn);
        return mk_string(tmpn);
    }

    if (t->type == TOK_LPAREN)
    {
        advance_func();
        val_t v = parse_expr(0);
        if (parse_error || eval_error)
        {
            return v;
        }
        if (cur_func()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_long(0);
        }
        advance_func();
        return v;
    }

    if (is_func_token(t->type))
    {
        advance_func();
        return parse_funccall(t);
    }

    if (t->type == TOK_VAR)
    {
        advance_func();
        cli_token_type aop = cur_func()->type;
        if (is_assign_op(aop))
        {
            advance_func();
            val_t v = parse_expr(0);
            if (parse_error || eval_error)
            {
                return v;
            }
            return eval_existing_var_assign(t->sval, aop, v);
        }
        return eval_existing_var_ref(t->sval);
    }

    if (t->type == TOK_NVAR)
    {
        advance_func();
        return eval_nvar_token(t->sval, t->literal, cur_func, advance_func);
    }

    if (t->type == TOK_IMAGE)
    {
        advance_func();
        return eval_image_token(t->sval, cur_func, advance_func);
    }

    if (t->type == TOK_COMMAND)
    {
        advance_func();
        if (parse_mode == 0)
        {
            data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_COMMAND;
            if (data.core.Debug > 0)
            {
                printf("this is a string (command)\n");
            }
        }
        return mk_string(t->sval);
    }

    parse_errmsg("Unexpected token");
    return mk_long(0);
}

/**
 * @brief Evaluate the ternary conditional `cond ? true_val : false_val` for scalar or image @left.
 */
static val_t eval_ternary_op(val_t left,
                             cli_token *(*cur_func)(void),
                             cli_token *(*advance_func)(void) )
{
    if (left.type != VAL_STRING) // scalar condition: evaluate only the chosen branch
    {
        int   c = (to_double(left) != 0.0);
        val_t r = c ? parse_expr(0) : mk_long(0);
        if (parse_error || eval_error || (!c && skip_expr(0) != 0))
        {
            return left;
        }
        if (cur_func()->type != TOK_OP_COLON)
        {
            parse_errmsg("Expected ':' in ternary");
            return mk_double(0);
        }
        advance_func();
        if (c)
        {
            return (skip_expr(0) != 0) ? left : r;
        }
        return parse_expr(0);
    }

    val_t true_val = parse_expr(0);
    if (parse_error || eval_error)
    {
        return left;
    }
    if (cur_func()->type != TOK_OP_COLON)
    {
        parse_errmsg("Expected ':' in ternary");
        return mk_double(0);
    }
    advance_func();
    val_t false_val = parse_expr(0);
    if (parse_error || eval_error)
    {
        return left;
    }

    return cli_calc_where(left, true_val, false_val);
}

/**
 * @brief Parse an expression with minimum precedence @min_prec.
 */
val_t parse_expr(int min_prec)
{
    if (parse_error || eval_error)
    {
        return mk_long(0);
    }

    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();

    val_t fused;
    int   fr = cli_calc_fuse_expr(min_prec, &fused);
    if (fr != 0)
    {
        return (fr > 0) ? fused : mk_long(0);
    }

    val_t left = parse_primary();
    if (parse_error || eval_error)
    {
        return left;
    }

    for (;;)
    {
        cli_token_type op   = cur_func()->type;
        int            prec = cli_calc_op_prec(op);
        if (prec < min_prec)
        {
            break;
        }
        advance_func();

        if (op == TOK_OP_QUESTION)
        {
            left = eval_ternary_op(left, cur_func, advance_func);
            if (parse_error || eval_error)
            {
                return left;
            }
            continue;
        }

        int next_min = cli_calc_op_right_assoc(op) ? prec : prec + 1;
        if ((op == TOK_OP_AND || op == TOK_OP_OR) && left.type != VAL_STRING &&
            (to_double(left) != 0.0) == (op == TOK_OP_OR)) // result already known: skip the RHS
        {
            if (skip_expr(next_min) != 0)
            {
                return left;
            }
            left = mk_long(op == TOK_OP_OR);
            continue;
        }

        val_t right = parse_expr(next_min);
        if (parse_error || eval_error)
        {
            return left;
        }
        left = eval_binop(op, left, right);
        if (parse_error || eval_error)
        {
            return left;
        }
    }

    return left;
}
