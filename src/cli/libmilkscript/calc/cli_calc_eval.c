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
 *   dispatch to `cli_calc_binop.c`.
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

/* --------------------------------------------------------
 * Operator precedence table
 *
 * Higher number = tighter binding.
 * Right-associative operators use prec for the
 * right-hand parse.
 * -------------------------------------------------------- */

static inline int get_prec(cli_token_type t)
{
    switch (t)
    {
    case TOK_OP_QUESTION:
        return 0; // below ||, reachable from a top-level parse_expr(0)
    case TOK_OP_OR:
        return 1;
    case TOK_OP_AND:
        return 2;
    case TOK_OP_BOR:
        return 3;
    case TOK_OP_BXOR:
        return 4;
    case TOK_OP_BAND:
        return 5;
    case TOK_OP_EQ:
    case TOK_OP_NEQ:
        return 6;
    case TOK_OP_LT:
    case TOK_OP_LE:
    case TOK_OP_GT:
    case TOK_OP_GE:
        return 7;
    case TOK_OP_LSHIFT:
    case TOK_OP_RSHIFT:
        return 8;
    case TOK_OP_PLUS:
    case TOK_OP_MINUS:
        return 9;
    case TOK_OP_STAR:
    case TOK_OP_SLASH:
    case TOK_OP_MOD:
        return 10;
    case TOK_OP_CARET:
        return 11;
    case TOK_EQUAL:
    case TOK_OP_PLUS_EQ:
    case TOK_OP_MINUS_EQ:
    case TOK_OP_STAR_EQ:
    case TOK_OP_SLASH_EQ:
        return 0;
    default:
        return -1;
    }
}

/**
 * @brief Check if an operator is right-associative.
 */
static inline int is_right_assoc(cli_token_type t)
{
    return (t == TOK_OP_CARET) || (t == TOK_EQUAL) || (t == TOK_OP_PLUS_EQ) ||
           (t == TOK_OP_MINUS_EQ) || (t == TOK_OP_STAR_EQ) || (t == TOK_OP_SLASH_EQ) ||
           (t == TOK_OP_QUESTION);
}

static inline int is_assign_op(cli_token_type t)
{
    return (t == TOK_EQUAL) || (t == TOK_OP_PLUS_EQ) || (t == TOK_OP_MINUS_EQ) ||
           (t == TOK_OP_STAR_EQ) || (t == TOK_OP_SLASH_EQ);
}

static inline int is_func_token(cli_token_type t)
{
    return (t == TOK_FUNC_D_D) || (t == TOK_FUNC_DD_D) || (t == TOK_FUNC_DDD_D) ||
           (t == TOK_FUNC_IM_D) || (t == TOK_FUNC_IMD_D) || (t == TOK_FUNC_WHERE) ||
           (t == TOK_FUNC_IMIM_D) || (t == TOK_FUNC_IMIM_IM) || (t == TOK_FUNC_S_D) ||
           (t == TOK_FUNC_S_S) || (t == TOK_FUNC_SDD_S) || (t == TOK_FUNC_SSS_S) ||
           (t == TOK_FUNC_D_S);
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
static val_t eval_existing_var_ref(const char *varname)
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
            return mk_string(varname);
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
        return eval_nvar_token(t->sval, cur_func, advance_func);
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

    if (left.type != VAL_STRING)
    {
        double c = to_double(left);
        return (c != 0.0) ? true_val : false_val;
    }

    if (!check_image(left.sval))
    {
        return mk_string("");
    }
    const char *tmask  = alloc_tmpname();
    const char *timask = alloc_tmpname();
    const char *tpart  = alloc_tmpname();
    const char *fpart  = alloc_tmpname();
    const char *tmpn   = alloc_tmpname();
    arith_image_csttestne(left.sval, 0.0, tmask);
    arith_image_cstteste(left.sval, 0.0, timask);
    if (true_val.type == VAL_STRING)
    {
        if (!check_image(true_val.sval))
        {
            return mk_string("");
        }
        arith_image_mult(true_val.sval, tmask, tpart);
    }
    else
    {
        arith_image_cstmult(tmask, to_double(true_val), tpart);
    }
    if (false_val.type == VAL_STRING)
    {
        if (!check_image(false_val.sval))
        {
            return mk_string("");
        }
        arith_image_mult(false_val.sval, timask, fpart);
    }
    else
    {
        arith_image_cstmult(timask, to_double(false_val), fpart);
    }
    arith_image_add(tpart, fpart, tmpn);
    return mk_string(tmpn);
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

    val_t left = parse_primary();
    if (parse_error || eval_error)
    {
        return left;
    }

    for (;;)
    {
        cli_token_type op   = cur_func()->type;
        int            prec = get_prec(op);
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

        int   next_min = is_right_assoc(op) ? prec : prec + 1;
        val_t right    = parse_expr(next_min);
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
