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
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

/* Operator precedence and associativity: cli_calc_scalar.c (shared with the in-shell evaluator) */

/** is_assign_op() - 1 if @t is an assignment operator (`=`, `+=`, `-=`, `*=`, `/=`). */
static inline int is_assign_op(cli_token_type t)
{
    return (t == TOK_EQUAL) || (t == TOK_OP_PLUS_EQ) || (t == TOK_OP_MINUS_EQ) ||
           (t == TOK_OP_STAR_EQ) || (t == TOK_OP_SLASH_EQ);
}

/** is_func_token() - 1 if @t represents a built-in function call token. */
static inline int is_func_token(cli_token_type t)
{
    return (t == TOK_FUNC_D_D) || (t == TOK_FUNC_DD_D) || (t == TOK_FUNC_DDD_D) ||
           (t == TOK_FUNC_IM_D) || (t == TOK_FUNC_IMD_D) || (t == TOK_FUNC_WHERE) ||
           (t == TOK_FUNC_IMIM_D) || (t == TOK_FUNC_IMIM_IM) || (t == TOK_FUNC_S_D) ||
           (t == TOK_FUNC_S_S) || (t == TOK_FUNC_SDD_S) || (t == TOK_FUNC_SSS_S) ||
           (t == TOK_FUNC_D_S) || (t == TOK_FUNC_IM_L) || (t == TOK_FUNC_CPLX) ||
           (t == TOK_FUNC_COORD) || (t == TOK_FUNC_SHIFT) || (t == TOK_FUNC_LUT) ||
           (t == TOK_FUNC_MEDFILT) || (t == TOK_FUNC_PIXREMAP) || (t == TOK_FUNC_PIXUNMAP) ||
           (t == TOK_FUNC_GAUSS) || (t == TOK_FUNC_CUMSUM) || (t == TOK_FUNC_INTERP2) ||
           (t == TOK_FUNC_UDEF) || (t == TOK_FUNC_USER);
}

static int skip_expr(int min_prec);

/** skip_primary() - consume one atom like parse_primary(), evaluating nothing; 0 or -1 on error. */
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
        do /* function tokens include their '(' */
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

/** skip_expr() - consume an expression like parse_expr(@min_prec) without side effects. */
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

/** eval_unary_minus() - evaluate unary minus `-expr` on scalar or image operands. */
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

/** eval_unary_bnot_not - Evaluate bitwise `~expr` or logical `!expr` on scalar or image. */
static val_t eval_unary_bnot_not(cli_token_type utype)
{
    val_t v = parse_primary();
    if (parse_error || eval_error)
    {
        return mk_long(0);
    }
    if (utype == TOK_OP_BNOT)
    {
        if (v.type == VAL_STRING)
        {
            if (!check_image(v.sval))
            {
                return mk_long(0);
            }
            const IMAGE *im =
                &data.core.image[image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE)];
            const char *res = cli_calc_eval_binop_im(TOK_OP_BXOR, im, 0.0, NULL, -1.0);
            if (res == NULL)
            {
                parse_errmsg("Unsupported image ~ op");
                return mk_long(0);
            }
            return mk_string(res);
        }
        long iv = (v.type == VAL_LONG) ? v.lval : (long) v.dval;
        return mk_long(~iv);
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

/**
 * eval_existing_var_ref - Resolve an existing variable read reference `TOK_VAR`
 * @varname: Variable name to look up
 *
 * Return: Variable value as `val_t`
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

/** eval_shm_prop_token - Evaluate a SHM property read or assignment (`s.cam.cnt0`, `fps.x.y = v`). */
static int eval_shm_prop_token(const char *varname,
                               cli_token *(*cur_func)(void),
                               cli_token *(*advance_func)(void),
                               val_t *out)
{
    cli_token_type aop = cur_func()->type;
    if (is_assign_op(aop))
    {
        advance_func();
        val_t v = parse_expr(0);
        if (parse_error || eval_error)
        {
            *out = v;
            return 1;
        }
        if (aop != TOK_EQUAL)
        {
            double old    = 0.0;
            long   old_l  = 0;
            int    islong = 0;
            if (milk_shm_prop_get_num(varname, &old, &old_l, &islong) != 0)
            {
                parse_errmsg("Property not found or non-numeric for compound assign");
                *out = mk_double(0);
                return 1;
            }
            if (islong)
            {
                old = (double) old_l;
            }
            double nv = to_double(v);
            if (aop == TOK_OP_PLUS_EQ)
            {
                nv = old + nv;
            }
            else if (aop == TOK_OP_MINUS_EQ)
            {
                nv = old - nv;
            }
            else if (aop == TOK_OP_STAR_EQ)
            {
                nv = old * nv;
            }
            else if (aop == TOK_OP_SLASH_EQ)
            {
                if (nv == 0.0)
                {
                    parse_errmsg("Division by zero");
                    *out = mk_double(0);
                    return 1;
                }
                nv = old / nv;
            }
            v = (islong && v.type == VAL_LONG && aop != TOK_OP_SLASH_EQ) ? mk_long((long) nv)
                                                                         : mk_double(nv);
        }
        char vstr[256];
        if (v.type == VAL_LONG)
        {
            snprintf(vstr, sizeof(vstr), "%ld", v.lval);
        }
        else if (v.type == VAL_DOUBLE)
        {
            snprintf(vstr, sizeof(vstr), "%.17g", v.dval);
        }
        else
        {
            snprintf(vstr, sizeof(vstr), "%s", v.sval);
        }
        if (milk_shm_prop_set(varname, vstr, NULL) != 0)
        {
            parse_errmsg("Failed to set property");
            *out = mk_double(0);
            return 1;
        }
        *out = v;
        return 1;
    }
    double dval   = 0.0;
    long   lval   = 0;
    int    islong = 0;
    if (milk_shm_prop_get_num(varname, &dval, &lval, &islong) == 0)
    {
        *out = islong ? mk_long(lval) : mk_double(dval);
        return 1;
    }
    char sbuf[256];
    if (milk_shm_prop_get(varname, sbuf, sizeof(sbuf)) == 0)
    {
        *out = mk_string(sbuf);
        return 1;
    }
    if (parse_mode != 0)
    {
        parse_errmsg("property not found");
        *out = mk_double(0);
        return 1;
    }
    return 0;
}

/** eval_nvar_token - Handle a `TOK_NVAR` token (new variable assignment, SHM property, or literal). */
static val_t eval_nvar_token(const char *varname,
                             int         literal,
                             cli_token *(*cur_func)(void),
                             cli_token *(*advance_func)(void) )
{
    const char *dot1 = strchr(varname, '.');
    if (!literal && dot1 != NULL && strchr(dot1 + 1, '.') != NULL &&
        milk_entity_ns_kind(varname, NULL) != MILK_NS_NONE)
    {
        val_t prop_val;
        if (eval_shm_prop_token(varname, cur_func, advance_func, &prop_val))
        {
            return prop_val;
        }
    }

    if (cur_func()->type == TOK_EQUAL)
    {
        advance_func();
        val_t v = parse_expr(0);
        if (parse_error || eval_error)
        {
            return v;
        }
        return cli_calc_store_var_or_im(varname, v, 0);
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
        char        msg[CLI_CALC_TOKEN_MAXLEN + 256];
        const char *gone = milk_entity_ns_removed(varname); /* q.x.status: say why */
        snprintf(msg, sizeof(msg), "undefined identifier '%s'%s%s", varname, gone ? ": " : "",
                 gone ? gone : "");
        parse_errmsg(msg);
        return mk_double(0);
    }
    return mk_string(varname);
}

/** eval_image_token - Handle a `TOK_IMAGE` token (whole-image/slice assignment or image/slice read). */
static val_t eval_image_token(const char *imname,
                              cli_token *(*cur_func)(void),
                              cli_token *(*advance_func)(void) )
{
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(imname);
    const char    *bare = (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : imname;
    cli_token_type aop  = cur_func()->type;
    if (is_assign_op(aop))
    {
        advance_func();
        const char *bk = strchr(bare, '[');
        imageID id = (bk == NULL) ? image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE) : -1;
        int     fr = (id != -1)     ? cli_calc_fuse_assign(&data.core.image[id], aop)
                     : (bk != NULL) ? cli_calc_fuse_assign_tok(NULL, imname, aop)
                                    : 0;
        if (fr != 0)
        {
            if (fr <= 0)
            {
                return mk_double(0);
            }
            if (bk != NULL)
            {
                char   sbare[CLI_CALC_TOKEN_MAXLEN];
                size_t blen = (size_t) (bk - bare);
                if (blen >= sizeof(sbare))
                {
                    blen = sizeof(sbare) - 1;
                }
                memcpy(sbare, bare, blen);
                sbare[blen] = '\0';
                return mk_string(sbare);
            }
            return mk_string(bare);
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
    return mk_string(parse_mode == 0 ? imname : bare);
}

/**
 * parse_primary - Parse a primary (non-operator) expression atom
 *
 * Return: Evaluated `val_t`
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

    if (t->type == TOK_LONG || t->type == TOK_DOUBLE)
    {
        advance_func();
        return (t->type == TOK_LONG) ? mk_long(t->val_l) : mk_double(t->val_d);
    }
    if (t->type == TOK_OP_MINUS)
    {
        advance_func();
        return eval_unary_minus();
    }
    if (t->type == TOK_OP_BNOT || t->type == TOK_OP_NOT)
    {
        advance_func();
        return eval_unary_bnot_not(t->type);
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
            return (parse_error || eval_error) ? v
                                               : cli_calc_eval_existing_var_assign(t->sval, aop, v);
        }
        val_t uval;
        return cli_calc_udef_eval_lookup(t->sval, &uval) ? uval : eval_existing_var_ref(t->sval);
    }
    if (t->type == TOK_NVAR || t->type == TOK_IMAGE)
    {
        advance_func();
        val_t uval;
        if (!t->literal && !is_assign_op(cur_func()->type) &&
            cli_calc_udef_eval_lookup(t->sval, &uval))
        {
            return uval;
        }
        return (t->type == TOK_NVAR) ? eval_nvar_token(t->sval, t->literal, cur_func, advance_func)
                                     : eval_image_token(t->sval, cur_func, advance_func);
    }
    if (t->type == TOK_COMMAND)
    {
        advance_func();
        if (parse_mode == 0)
        {
            data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_COMMAND;
        }
        return mk_string(t->sval);
    }

    parse_errmsg("Unexpected token");
    return mk_long(0);
}

/** eval_ternary_op - Evaluate the ternary conditional `cond ? true_val : false_val`. */
static val_t eval_ternary_op(val_t left,
                             cli_token *(*cur_func)(void),
                             cli_token *(*advance_func)(void) )
{
    if (left.type != VAL_STRING) /* scalar condition: evaluate only the chosen branch */
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
 * parse_expr - Parse an expression with minimum precedence @min_prec
 * @min_prec: Minimum operator binding power
 *
 * Return: Evaluated `val_t`
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
            (to_double(left) != 0.0) == (op == TOK_OP_OR)) /* result already known: skip the RHS */
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
